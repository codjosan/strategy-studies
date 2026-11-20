"""Signal-conditional exits for the imbalance z-score signal.

Two modifications of the timeout exit, motivated by the primary (60-minute)
z-score sometimes turning against a trade and renewing around the timeout:

reversal
    Exit when the primary z-score against the trade reaches ``reversal_z``.
    The rule only applies once ``reversal_min_hold`` minutes have passed
    (default: the hold time): the z-score naturally decays from its entry
    extreme in the first hour, so an ungated rule would exit on that decay
    rather than on signal failure.
extension
    At the hold time, keep the position while the primary z-score with the
    trade is at least ``extension_z``, up to ``max_hold_min``.

Exit priority each minute: stop, reversal, take-profit (on net return),
timeout or extension. ``0`` in a level list disables that rule.

For trades closed by the reversal rule the report compares the actual net PnL
with the net PnL the trade would have had at the hold time; without that
counterfactual an early exit can look good in aggregate while mostly cutting
winners short.

Usage::

    python -m hlmicro signal-exit --symbol BTC
    python -m hlmicro signal-exit --symbol SOL --tp-pct 1.0 --reversal-levels 0,1,1.5,2
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
                         simulate, trade_stats)

log = logging.getLogger(__name__)

EXIT_REASONS = ("stop", "reversal", "tp", "timeout", "max_hold")


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    threshold: float = 2.5
    single_window: bool = False
    hold_min: int = 240
    max_hold_min: int = 480
    cooldown_min: int = 60
    stop_pct: float = -1.0
    reference_stop_pct: float = -2.5
    tp_pct: float = 0.0
    reversal_levels: list[float] = field(default_factory=lambda: [0.0, 1.0, 1.5, 2.0])
    reversal_min_hold: int | None = None
    extension_levels: list[float] = field(default_factory=lambda: [0.0, 1.0])
    counterfactual_margin_pct: float = 0.02
    next_bar_entry: bool = False
    n_folds: int = 4
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None

    @property
    def signal(self) -> SignalSpec:
        return SignalSpec(self.threshold, 60, None if self.single_window else 30)

    def exits(self, stop: float, tp: float, reversal_z: float = 0.0, extension_z: float = 0.0) -> ExitSpec:
        return ExitSpec(hold_min=self.hold_min, cooldown_min=self.cooldown_min, stop_pct=stop,
                        target_pct=tp if tp > 0 else None, target_on_net=True,
                        reversal_z=reversal_z, reversal_min_hold=self.reversal_min_hold,
                        extension_z=extension_z,
                        max_hold_min=self.max_hold_min if extension_z > 0 else self.hold_min,
                        next_bar_entry=self.next_bar_entry)


@dataclass
class Variant:
    label: str
    reversal_z: float
    extension_z: float
    trades: list[dict]

    @property
    def stats(self) -> dict:
        s = trade_stats(self.trades)
        s["exits"] = {r: sum(t["exit_reason"] == r for t in self.trades) for r in EXIT_REASONS}
        s["extended"] = sum(t["extended"] for t in self.trades)
        return s


def reversal_counterfactual(trades: list[dict]) -> dict | None:
    """Reversal exits: actual net PnL against net PnL at the hold time."""
    rev = [t for t in trades if t["exit_reason"] == "reversal" and not np.isnan(t["pnl_at_hold_net"])]
    if not rev:
        return None
    actual = np.array([t["pnl_pct_net"] for t in rev])
    cf = np.array([t["pnl_at_hold_net"] for t in rev])
    return {"n": len(rev), "actual_mean": float(actual.mean()), "cf_mean": float(cf.mean()),
            "delta": float(actual.mean() - cf.mean()), "beats_pct": 100 * float((actual > cf).mean()),
            "cf_win_pct": 100 * float((cf > 0).mean())}


@dataclass
class Result:
    config: Config
    files: list[Path]
    grid_minutes: int
    span_days: float
    first_ms: int
    last_ms: int
    references: list[Variant]
    sweep: list[Variant]


def _label(stop: float, tp: float, rev: float = 0.0, ext: float = 0.0) -> str:
    parts = [f"stop={stop:+.1f}%", f"TP=+{tp:.1f}%" if tp > 0 else "no TP"]
    if rev > 0:
        parts.append(f"rev>={rev:g}")
    if ext > 0:
        parts.append(f"ext>={ext:g}")
    return " ".join(parts)


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    grid = minute_grid(files, cfg.symbol)
    if grid.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    first, last = int(grid["ts_ms"].iloc[0]), int(grid["ts_ms"].iloc[-1])

    def variant(stop: float, tp: float, rev: float = 0.0, ext: float = 0.0) -> Variant:
        return Variant(_label(stop, tp, rev, ext), rev, ext,
                       simulate(grid, cfg.signal, cfg.exits(stop, tp, rev, ext), cfg.fee_bps))

    refs = [(cfg.reference_stop_pct, 0.0), (cfg.stop_pct, 0.0)]
    if cfg.tp_pct > 0:
        refs.append((cfg.stop_pct, cfg.tp_pct))
    references = [variant(stop, tp) for stop, tp in dict.fromkeys(refs)]
    sweep = [variant(cfg.stop_pct, cfg.tp_pct, rev, ext)
             for rev in cfg.reversal_levels for ext in cfg.extension_levels if rev > 0 or ext > 0]
    return Result(cfg, files, len(grid), (last - first) / 86_400_000, first, last, references, sweep)


def _row(v: Variant) -> str:
    s = v.stats
    if s["n"] == 0:
        return f"  {v.label:<44}  no trades"
    exits = "/".join(str(s["exits"][r]) for r in EXIT_REASONS)
    return (f"  {v.label:<44}{s['n']:>5}{s['winr']:>7.1f}%{s['mean']:>+9.3f}{s['sum']:>+9.2f}"
            f"{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}  {exits:<18}{s['extended']:>5}")


def render(result: Result) -> str:
    cfg = result.config
    rmh = cfg.hold_min if cfg.reversal_min_hold is None else cfg.reversal_min_hold
    header = (f"  {'configuration':<44}{'n':>5}{'win%':>8}{'mean%':>9}{'sum%':>9}{'sharpe':>8}{'t':>7}"
              f"  {'stop/rev/tp/to/max':<18}{'ext':>5}")
    out = [f"Signal-conditional exits - {cfg.symbol}",
           f"  files: {len(result.files)} ({result.files[0].name} .. {result.files[-1].name})",
           f"  signal: {cfg.signal.describe()}  hold={cfg.hold_min}m  max hold when extending="
           f"{cfg.max_hold_min}m  cooldown={cfg.cooldown_min}m  cost={cfg.fee_bps:.2f}bps round trip"
           + ("  entry=next bar" if cfg.next_bar_entry else ""),
           f"  reversal rule applies after {rmh}m held; take-profit is on net return",
           f"  grid: {result.grid_minutes:,} minutes over {result.span_days:.1f} days",
           "", rule(130), "REFERENCE (no conditional exits)", rule(130), header]
    out += [_row(v) for v in result.references]
    out += ["", rule(130), "CONDITIONAL-EXIT SWEEP", rule(130), header]
    out += [_row(v) for v in result.sweep]

    out += ["", rule(130), f"REVERSAL COUNTERFACTUAL: actual net PnL vs net PnL at {cfg.hold_min}m "
            f"(delta > {cfg.counterfactual_margin_pct}pp: helped; < -{cfg.counterfactual_margin_pct}pp: "
            f"cut winners short)", rule(130),
            f"  {'configuration':<44}{'n':>5}{'actual':>10}{'at hold':>10}{'delta':>9}{'beat%':>8}"
            f"{'cf win%':>9}  verdict"]
    for v in result.sweep:
        if v.reversal_z <= 0:
            continue
        cf = reversal_counterfactual(v.trades)
        if cf is None:
            out.append(f"  {v.label:<44}    0  (no reversal exits)")
            continue
        m = cfg.counterfactual_margin_pct
        verdict = "helped" if cf["delta"] > m else "hurt" if cf["delta"] < -m else "neutral"
        out.append(f"  {v.label:<44}{cf['n']:>5}{cf['actual_mean']:>+9.3f}%{cf['cf_mean']:>+9.3f}%"
                   f"{cf['delta']:>+8.3f}%{cf['beats_pct']:>7.0f}%{cf['cf_win_pct']:>8.0f}%  {verdict}")

    out += ["", rule(130), f"MEAN NET PnL% PER FOLD ({cfg.n_folds} folds, trades assigned by entry time)",
            rule(130)]
    for v in result.references + result.sweep:
        cells, positive, mom = fold_mean_cells(
            folds_by_entry(v.trades, result.first_ms, result.last_ms, cfg.n_folds))
        out.append(render_fold_row(v.label, cells, positive, mom))
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> Path | None:
    rows = [{**t, "reversal_z": v.reversal_z, "extension_z": v.extension_z}
            for v in result.sweep for t in v.trades]
    if not rows:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"signal_exit_{result.config.symbol.lower()}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro signal-exit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=d.threshold)
    parser.add_argument("--single-window", action="store_true", help="60-minute z-score only")
    parser.add_argument("--hold-min", type=int, default=d.hold_min)
    parser.add_argument("--max-hold-min", type=int, default=d.max_hold_min,
                        help="cap on minutes held when the extension rule is on")
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--stop-pct", type=float, default=d.stop_pct,
                        help="stop used in the sweep (negative; write --stop-pct=-1.0)")
    parser.add_argument("--reference-stop-pct", type=float, default=d.reference_stop_pct,
                        help="stop of the first reference row")
    parser.add_argument("--tp-pct", type=float, default=d.tp_pct, help="take-profit on net return; 0 disables")
    parser.add_argument("--reversal-levels", type=csv_list(float), default=d.reversal_levels)
    parser.add_argument("--reversal-min-hold", type=int, default=None,
                        help="minutes held before the reversal rule applies (default: --hold-min)")
    parser.add_argument("--extension-levels", type=csv_list(float), default=d.extension_levels)
    parser.add_argument("--next-bar-entry", action="store_true",
                        help="enter at the minute after the trigger instead of the trigger minute")
    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--outdir", type=Path, default=Path("results/signal_exit"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, threshold=args.threshold,
                 single_window=args.single_window, hold_min=args.hold_min, max_hold_min=args.max_hold_min,
                 cooldown_min=args.cooldown_min, stop_pct=args.stop_pct,
                 reference_stop_pct=args.reference_stop_pct, tp_pct=args.tp_pct,
                 reversal_levels=args.reversal_levels, reversal_min_hold=args.reversal_min_hold,
                 extension_levels=args.extension_levels, next_bar_entry=args.next_bar_entry,
                 n_folds=args.n_folds, fee_bps=args.fee_bps, start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    path = write_outputs(result, args.outdir)
    if path:
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
