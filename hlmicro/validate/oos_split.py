"""Out-of-sample check of a fixed imbalance z-score configuration.

The configuration is not retuned: the same parameters run on a training
window and on the data after it, split at ``--split-date`` (or after the
first ``--train-frac`` of the time span), and the two are compared. Each
window is simulated separately, so z-score warm-up restarts at the split.

Entries are at the minute after the trigger by default, so the entry price is
strictly later than the information used (``--same-bar-entry`` reverts to the
``tp-multifold`` convention). Exits are by timeout only.

Verdict on the test window:

* PASS      mean has the training sign and is positive, win rate >= 50%
* MARGINAL  sign kept but only one of (positive mean, win rate >= 50%)
* FAIL      otherwise

``--threshold-sweep`` repeats the test-window backtest over a range of
thresholds; a robust signal degrades gradually rather than at a cliff.

Usage::

    python -m hlmicro oos-split --symbol BTC --split-date 20260430
    python -m hlmicro oos-split --symbol SOL --threshold-sweep
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
from .obi_signal import ExitSpec, SignalSpec, simulate, trade_stats

log = logging.getLogger(__name__)


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    threshold: float = 2.5
    single_window: bool = False
    hold_min: int = 240
    cooldown_min: int = 60
    split_date: str | None = None
    train_frac: float = 0.35
    next_bar_entry: bool = True
    threshold_sweep: bool = False
    sweep_thresholds: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5])
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None

    def signal(self, threshold: float | None = None) -> SignalSpec:
        return SignalSpec(self.threshold if threshold is None else threshold, 60,
                          None if self.single_window else 30)

    @property
    def exits(self) -> ExitSpec:
        return ExitSpec(hold_min=self.hold_min, cooldown_min=self.cooldown_min,
                        next_bar_entry=self.next_bar_entry)


def summarise(trades: list[dict], days: float) -> dict:
    s = trade_stats(trades)
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    win = lambda ts: 100 * sum(t["pnl_pct_net"] > 0 for t in ts) / len(ts) if ts else np.nan
    return {**s, "days": days, "trades_per_day": s["n"] / days if days > 0 else np.nan,
            "n_long": len(longs), "n_short": len(shorts),
            "winr_long": win(longs), "winr_short": win(shorts)}


def verdict(train: dict, test: dict) -> str:
    if test["n"] == 0:
        return "no test trades"
    if train["n"] == 0:
        return "no training trades"
    same_sign = np.sign(train["mean"]) == np.sign(test["mean"])
    if same_sign and test["mean"] > 0 and test["winr"] >= 50:
        return "PASS"
    if same_sign and (test["mean"] > 0 or test["winr"] >= 50):
        return "MARGINAL"
    return "FAIL"


@dataclass
class Result:
    config: Config
    files: list[Path]
    boundary_ms: int
    windows: dict        # "full" / "train" / "test" -> (first_ms, last_ms, minutes)
    stats: dict          # "full" / "train" / "test" -> summary
    trades: list[dict]   # full-window trades, with "period"
    verdict: str
    sweep: list[dict]


def _days(grid: pd.DataFrame) -> float:
    return (grid["ts_ms"].iloc[-1] - grid["ts_ms"].iloc[0]) / 86_400_000


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    grid = minute_grid(files, cfg.symbol)
    if grid.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    first, last = int(grid["ts_ms"].iloc[0]), int(grid["ts_ms"].iloc[-1])
    if cfg.split_date:
        boundary = int(pd.Timestamp(cfg.split_date, tz="UTC").timestamp() * 1000)
    else:
        boundary = int(first + cfg.train_frac * (last - first))
    parts = {"full": grid,
             "train": grid[grid["ts_ms"] < boundary].reset_index(drop=True),
             "test": grid[grid["ts_ms"] >= boundary].reset_index(drop=True)}
    for name in ("train", "test"):
        if len(parts[name]) < 2:
            raise ValueError(f"the {name} window is empty; adjust --split-date or --train-frac")

    stats, windows, trades = {}, {}, []
    for name, part in parts.items():
        tr = simulate(part, cfg.signal(), cfg.exits, cfg.fee_bps)
        stats[name] = summarise(tr, _days(part))
        windows[name] = (int(part["ts_ms"].iloc[0]), int(part["ts_ms"].iloc[-1]), len(part))
        if name == "full":
            trades = [{**t, "period": "train" if t["entry_ts_ms"] < boundary else "test"} for t in tr]

    sweep = []
    if cfg.threshold_sweep:
        for thr in cfg.sweep_thresholds:
            tr = simulate(parts["test"], cfg.signal(thr), cfg.exits, cfg.fee_bps)
            sweep.append({"threshold": thr, **summarise(tr, stats["test"]["days"])})
    return Result(cfg, files, boundary, windows, stats, trades,
                  verdict(stats["train"], stats["test"]), sweep)


def _fmt_ts(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def render(result: Result) -> str:
    cfg = result.config
    out = [f"Out-of-sample split - {cfg.symbol}",
           f"  files: {len(result.files)} ({result.files[0].name} .. {result.files[-1].name})",
           f"  signal: {cfg.signal().describe()}  hold={cfg.hold_min}m  cooldown={cfg.cooldown_min}m  "
           f"cost={cfg.fee_bps:.2f}bps round trip  entry={'next bar' if cfg.next_bar_entry else 'trigger bar'}"]
    for name in ("train", "test"):
        a, b, m = result.windows[name]
        out.append(f"  {name:<5}: {_fmt_ts(a)} .. {_fmt_ts(b)}  ({result.stats[name]['days']:.1f} days, "
                   f"{m:,} minutes)")
    out += ["", rule(110), f"  {'window':<8}{'n':>5}{'win%':>8}{'mean%':>9}{'sum%':>9}{'sharpe':>8}{'t':>7}"
            f"   {'long (win%)':>12}{'short (win%)':>14}{'per day':>9}", rule(110)]
    for name in ("full", "train", "test"):
        s = result.stats[name]
        if s["n"] == 0:
            out.append(f"  {name:<8}    0 trades")
            continue
        out.append(f"  {name:<8}{s['n']:>5}{s['winr']:>7.1f}%{s['mean']:>+9.3f}{s['sum']:>+9.2f}"
                   f"{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}   {s['n_long']:>4} ({s['winr_long']:>4.0f}%)"
                   f"{s['n_short']:>6} ({s['winr_short']:>4.0f}%){s['trades_per_day']:>9.2f}")

    tr, te = result.stats["train"], result.stats["test"]
    out += ["", rule(110), "OUT-OF-SAMPLE COMPARISON", rule(110)]
    if tr["n"] and te["n"]:
        same = np.sign(tr["mean"]) == np.sign(te["mean"])
        out.append(f"  sign:       train {tr['mean']:+.3f}%  test {te['mean']:+.3f}%  -> "
                   f"{'same sign' if same else 'flipped'}")
        out.append(f"  win rate:   train {tr['winr']:.1f}%  test {te['winr']:.1f}%  "
                   f"({te['winr'] - tr['winr']:+.1f}pp)")
        if tr["mean"] != 0:
            ratio = te["mean"] / tr["mean"]
            band = "preserved" if 0.5 < ratio < 2.0 else "degraded" if ratio < 0.5 else "amplified"
            out.append(f"  magnitude:  test/train {ratio:+.2f}x ({band})")
        sig = ("significant at 95%" if abs(te["tstat"]) > 2 else
               "marginal" if abs(te["tstat"]) > 1 else "not significant")
        out.append(f"  test t:     {te['tstat']:+.2f} ({sig}); test sharpe {te['sharpe']:+.3f} "
                   f"(train {tr['sharpe']:+.3f})")
    out.append(f"\n  VERDICT: {result.verdict}")

    if result.sweep:
        out += ["", rule(110), "THRESHOLD SENSITIVITY ON THE TEST WINDOW", rule(110),
                f"  {'threshold':<11}{'n':>5}{'win%':>8}{'mean%':>9}{'sum%':>9}{'sharpe':>8}{'t':>7}{'per day':>9}"]
        for s in result.sweep:
            mark = "  (configured)" if abs(s["threshold"] - cfg.threshold) < 1e-9 else ""
            if s["n"] == 0:
                out.append(f"  +/-{s['threshold']:<8.2f}    0 trades{mark}")
                continue
            out.append(f"  +/-{s['threshold']:<8.2f}{s['n']:>5}{s['winr']:>7.1f}%{s['mean']:>+9.3f}"
                       f"{s['sum']:>+9.2f}{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}{s['trades_per_day']:>9.2f}{mark}")
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> Path | None:
    if not result.trades:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"oos_trades_{result.config.symbol.lower()}.csv"
    pd.DataFrame(result.trades).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro oos-split", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=d.threshold)
    parser.add_argument("--single-window", action="store_true", help="60-minute z-score only")
    parser.add_argument("--hold-min", type=int, default=d.hold_min)
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--split-date", default=None, help="first test date, YYYYMMDD")
    parser.add_argument("--train-frac", type=float, default=d.train_frac,
                        help="training share of the time span when --split-date is not given")
    parser.add_argument("--same-bar-entry", action="store_true",
                        help="enter at the trigger minute instead of the next one")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--sweep-thresholds", type=csv_list(float), default=d.sweep_thresholds)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--outdir", type=Path, default=Path("results/oos_split"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, threshold=args.threshold,
                 single_window=args.single_window, hold_min=args.hold_min, cooldown_min=args.cooldown_min,
                 split_date=args.split_date, train_frac=args.train_frac,
                 next_bar_entry=not args.same_bar_entry, threshold_sweep=args.threshold_sweep,
                 sweep_thresholds=args.sweep_thresholds, fee_bps=args.fee_bps,
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    path = write_outputs(result, args.outdir)
    if path:
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
