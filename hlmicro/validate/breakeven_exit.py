"""Breakeven exit for the imbalance z-score signal, against the timeout exit.

The position stays open until the first of: gross return at or below the
stop (``stop``), net return at or above the breakeven level (``breakeven``),
or the safeguard cap on minutes held (``timeout``). Positions still open
when the data ends are reported as ``open_at_end`` and left out of the
statistics. Reference rows use the timeout exit at ``reference_hold_min``.

The report flags the failure modes this kind of exit is prone to:

* CENSORING      >= 10% of positions unresolved at the end of the data
* WIN-RATE TRAP  win rate >= 75% with mean net PnL below 0.05%: winners pinned
                 at breakeven, losers at the full stop
* NEG SKEW       skewness of net PnL below -1
* LOCKUP         median hold at least four times ``reference_hold_min``;
                 per-trade edge must be weighed against trade frequency

Usage::

    python -m hlmicro breakeven-exit --symbol BTC
    python -m hlmicro breakeven-exit --symbol BTC --breakeven-levels 0,0.3,0.5 \\
        --max-hold-levels 240,480,720,1440
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, default_capture_dir
from ..parse import discover, minute_grid
from ..schemas import BOOK_REST_PREFIX
from .common import rule
from .obi_signal import (ExitSpec, SignalSpec, fold_mean_cells, folds_by_entry, render_fold_row,
                         simulate)

log = logging.getLogger(__name__)


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    threshold: float = 2.5
    single_window: bool = False
    cooldown_min: int = 60
    stop_levels: list[float] = field(default_factory=lambda: [-2.5, -1.0])
    breakeven_levels: list[float] = field(default_factory=lambda: [0.0])
    max_hold_levels: list[int] = field(default_factory=lambda: [240, 720, 1440, 2880, 4320])
    reference_hold_min: int = 240
    next_bar_entry: bool = False
    n_folds: int = 4
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None

    @property
    def signal(self) -> SignalSpec:
        return SignalSpec(self.threshold, 60, None if self.single_window else 30)

    def exits(self, stop: float, hold: int, breakeven: float | None) -> ExitSpec:
        return ExitSpec(hold_min=hold, cooldown_min=self.cooldown_min, stop_pct=stop,
                        target_pct=breakeven, target_on_net=True, target_label="breakeven",
                        next_bar_entry=self.next_bar_entry, close_at_end=True)


def summarise(trades: list[dict], days: float) -> dict:
    resolved = [t for t in trades if t["exit_reason"] != "open_at_end"]
    n_open = len(trades) - len(resolved)
    if not resolved:
        return {"n": 0, "n_open": n_open}
    pnl = np.array([t["pnl_pct_net"] for t in resolved])
    holds = np.array([t["minutes_held"] for t in resolved])
    n, m = len(pnl), float(pnl.mean())
    s = float(pnl.std(ddof=1)) if n > 1 else 0.0
    reasons = pd.Series([t["exit_reason"] for t in resolved]).value_counts()
    return {
        "n": n, "n_open": n_open, "winr": 100 * float((pnl > 0).mean()), "mean": m, "stdev": s,
        "sum": float(pnl.sum()), "sharpe": m / s if s > 0 else 0.0,
        "tstat": m / s * np.sqrt(n) if s > 0 else 0.0,
        "skew": float(((pnl - m) ** 3).mean() / s ** 3) if n > 2 and s > 0 else np.nan,
        "p10": float(np.percentile(pnl, 10)), "p90": float(np.percentile(pnl, 90)),
        "med_hold": int(np.median(holds)), "mean_hold": int(holds.mean()), "max_hold": int(holds.max()),
        "n_stop": int(reasons.get("stop", 0)), "n_breakeven": int(reasons.get("breakeven", 0)),
        "n_timeout": int(reasons.get("timeout", 0)),
        "trades_per_day": n / days if days > 0 else np.nan,
        "market_pct": 100 * holds.sum() / (days * 1440) if days > 0 else np.nan,
    }


def failure_flags(s: dict, reference_hold_min: int) -> list[str]:
    if s.get("n", 0) == 0:
        return []
    flags = []
    if s["n_open"] / (s["n"] + s["n_open"]) >= 0.10:
        flags.append("CENSORING>=10%")
    if s["winr"] >= 75 and s["mean"] < 0.05:
        flags.append("WIN-RATE TRAP")
    if not np.isnan(s["skew"]) and s["skew"] < -1.0:
        flags.append(f"NEG SKEW ({s['skew']:+.2f})")
    if s["med_hold"] >= 4 * reference_hold_min:
        flags.append(f"LOCKUP (median hold {s['med_hold']}m)")
    return flags


@dataclass
class Row:
    mode: str              # "timeout" (reference) or "breakeven"
    stop: float
    breakeven: float | None
    max_hold: int
    trades: list[dict]
    stats: dict

    @property
    def label(self) -> str:
        be = "-" if self.breakeven is None else f"{self.breakeven:+.1f}%"
        return f"{self.mode:<9} stop={self.stop:+.1f}% BE={be:<6} cap={self.max_hold}m"


@dataclass
class Result:
    config: Config
    files: list[Path]
    grid_minutes: int
    span_days: float
    first_ms: int
    last_ms: int
    references: list[Row]
    sweep: list[Row]


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    grid = minute_grid(files, cfg.symbol)
    if grid.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    first, last = int(grid["ts_ms"].iloc[0]), int(grid["ts_ms"].iloc[-1])
    days = (last - first) / 86_400_000

    def row(mode: str, stop: float, be: float | None, hold: int) -> Row:
        trades = simulate(grid, cfg.signal, cfg.exits(stop, hold, be), cfg.fee_bps)
        return Row(mode, stop, be, hold, trades, summarise(trades, days))

    references = [row("timeout", stop, None, cfg.reference_hold_min) for stop in cfg.stop_levels]
    sweep = [row("breakeven", stop, be, hold) for stop in cfg.stop_levels
             for be in cfg.breakeven_levels for hold in cfg.max_hold_levels]
    return Result(cfg, files, len(grid), days, first, last, references, sweep)


def _stats_line(r: Row) -> str:
    s = r.stats
    if s.get("n", 0) == 0:
        return f"  {r.label:<44}  0 resolved trades (open {s.get('n_open', 0)})"
    exits = f"{s['n_stop']}/{s['n_breakeven']}/{s['n_timeout']}"
    skew = "na" if np.isnan(s["skew"]) else f"{s['skew']:+.2f}"
    return (f"  {r.label:<44}{s['n']:>5}{s['n_open']:>5}{s['winr']:>7.1f}%{s['mean']:>+8.3f}"
            f"{s['p10']:>+8.3f}{s['p90']:>+8.3f}{skew:>7}{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}"
            f"{s['med_hold']:>7}m{s['trades_per_day']:>6.1f}{s['market_pct']:>6.1f}%{exits:>12}")


def render(result: Result) -> str:
    cfg = result.config
    header = (f"  {'configuration':<44}{'n':>5}{'open':>5}{'win%':>8}{'mean%':>8}{'p10':>8}{'p90':>8}"
              f"{'skew':>7}{'sharpe':>8}{'t':>7}{'med hold':>8}{'/day':>6}{'mkt%':>7}{'S/BE/T':>12}")
    out = [f"Breakeven exit - {cfg.symbol}",
           f"  files: {len(result.files)} ({result.files[0].name} .. {result.files[-1].name})",
           f"  signal: {cfg.signal.describe()}  cooldown={cfg.cooldown_min}m  "
           f"cost={cfg.fee_bps:.2f}bps round trip" + ("  entry=next bar" if cfg.next_bar_entry else ""),
           f"  grid: {result.grid_minutes:,} minutes over {result.span_days:.1f} days",
           "  win rate is not a useful measure for this exit; judge by mean, skew and trades per day",
           "", rule(150), "REFERENCE: timeout exit", rule(150), header]
    out += [_stats_line(r) for r in result.references]
    out += ["", rule(150), "BREAKEVEN EXIT: first of stop, net >= BE, or the hold cap", rule(150), header]
    out += [_stats_line(r) for r in result.sweep]

    out += ["", rule(150), "FAILURE-MODE FLAGS", rule(150)]
    flagged = [(r, failure_flags(r.stats, cfg.reference_hold_min)) for r in result.sweep]
    flagged = [(r, f) for r, f in flagged if f]
    out += [f"  {r.label:<44}  {', '.join(f)}" for r, f in flagged] or ["  none"]

    out += ["", rule(150), f"MEAN NET PnL% PER FOLD ({cfg.n_folds} folds, trades assigned by entry time)",
            rule(150)]
    for r in result.references + result.sweep:
        resolved = [t for t in r.trades if t["exit_reason"] != "open_at_end"]
        cells, positive, mom = fold_mean_cells(
            folds_by_entry(resolved, result.first_ms, result.last_ms, cfg.n_folds))
        out.append(render_fold_row(r.label, cells, positive, mom))
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> Path | None:
    rows = [{**t, "mode": r.mode, "stop": r.stop, "breakeven": r.breakeven, "max_hold": r.max_hold}
            for r in result.references + result.sweep for t in r.trades]
    if not rows:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"breakeven_exit_{result.config.symbol.lower()}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro breakeven-exit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=d.threshold)
    parser.add_argument("--single-window", action="store_true", help="60-minute z-score only")
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--stop-levels", type=csv_list(float), default=d.stop_levels,
                        help="stops in %% (negative; write --stop-levels=-2.5,-1.0)")
    parser.add_argument("--breakeven-levels", type=csv_list(float), default=d.breakeven_levels,
                        help="net-PnL exit levels in %%; 0 is literal breakeven")
    parser.add_argument("--max-hold-levels", type=csv_list(int), default=d.max_hold_levels,
                        help="safeguard caps on minutes held")
    parser.add_argument("--reference-hold-min", type=int, default=d.reference_hold_min)
    parser.add_argument("--next-bar-entry", action="store_true",
                        help="enter at the minute after the trigger instead of the trigger minute")
    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--outdir", type=Path, default=Path("results/breakeven_exit"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, threshold=args.threshold,
                 single_window=args.single_window, cooldown_min=args.cooldown_min,
                 stop_levels=args.stop_levels, breakeven_levels=args.breakeven_levels,
                 max_hold_levels=args.max_hold_levels, reference_hold_min=args.reference_hold_min,
                 next_bar_entry=args.next_bar_entry, n_folds=args.n_folds, fee_bps=args.fee_bps,
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    path = write_outputs(result, args.outdir)
    if path:
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
