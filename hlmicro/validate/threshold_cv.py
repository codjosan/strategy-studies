"""Threshold selection for the imbalance z-score signal across chronological folds.

Runs the signal of ``tp-multifold`` (dual 60/30-minute z-scores, or the
60-minute z-score alone with ``--single-window``) for each threshold on each
of ``n_folds`` equal-duration folds, optionally with a stop-loss, so the
threshold and the stop are validated together.

Reports per-fold mean net PnL, pooled statistics, and a composite rank per
threshold: the sum of its ranks by positive folds, mean of fold means, spread
of fold means (lower is better) and pooled t-statistic. Lower is better.

Usage::

    python -m hlmicro threshold-cv --symbol BTC
    python -m hlmicro threshold-cv --symbol BTC --thresholds 2.5,2.75,3.0 --stop-pct=-1.0
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
from .obi_signal import ExitSpec, SignalSpec, simulate, split_folds, trade_stats

log = logging.getLogger(__name__)


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "SOL"
    thresholds: list[float] = field(default_factory=lambda: [2.0, 2.25, 2.5, 2.75, 3.0])
    reference_threshold: float = 2.5
    single_window: bool = False
    hold_min: int = 240
    cooldown_min: int = 60
    stop_pct: float | None = None
    next_bar_entry: bool = False
    n_folds: int = 4
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None

    def signal(self, threshold: float) -> SignalSpec:
        return SignalSpec(threshold, 60, None if self.single_window else 30)

    @property
    def exits(self) -> ExitSpec:
        return ExitSpec(hold_min=self.hold_min, cooldown_min=self.cooldown_min,
                        stop_pct=self.stop_pct, next_bar_entry=self.next_bar_entry)


def _stats(trades: list[dict]) -> dict:
    return {**trade_stats(trades), "stop_exits": sum(t["exit_reason"] == "stop" for t in trades)}


@dataclass
class Result:
    config: Config
    files: list[Path]
    grid_minutes: int
    span_days: float
    folds: list[dict]
    cells: dict          # (threshold, fold) -> stats
    pooled: dict         # threshold -> stats (+ trades_per_day)
    summary: list[dict]  # per threshold: folds_positive, folds_total, mean_of_means, stdev_means
    ranking: list[dict]  # sorted by composite rank


def _rank(rows: list[dict], key: str, descending: bool) -> dict[float, int]:
    ordered = sorted(rows, key=lambda r: r[key], reverse=descending)
    return {r["threshold"]: i + 1 for i, r in enumerate(ordered)}


def run(cfg: Config) -> Result:
    if cfg.stop_pct is not None and cfg.stop_pct >= 0:
        raise ValueError(f"stop_pct must be negative, got {cfg.stop_pct}")
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    grid = minute_grid(files, cfg.symbol)
    if grid.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    span = (grid["ts_ms"].iloc[-1] - grid["ts_ms"].iloc[0]) / 86_400_000
    folds = split_folds(grid, cfg.n_folds)
    if len(folds) < 2:
        raise ValueError(f"only {len(folds)} fold(s) with data; need at least 2")

    thresholds = sorted(set(cfg.thresholds))
    cells, pooled, summary = {}, {}, []
    for thr in thresholds:
        all_trades, means = [], []
        for fold in folds:
            trades = simulate(fold["grid"], cfg.signal(thr), cfg.exits, cfg.fee_bps)
            cells[(thr, fold["idx"])] = _stats(trades)
            all_trades += trades
            if trades:
                means.append(cells[(thr, fold["idx"])]["mean"])
        s = _stats(all_trades)
        if s["n"]:
            pooled[thr] = {**s, "trades_per_day": s["n"] / span if span > 0 else np.nan}
        if means:
            summary.append({"threshold": thr, "folds_positive": sum(m > 0 for m in means),
                            "folds_total": len(means), "mean_of_means": float(np.mean(means)),
                            "stdev_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0})

    ranking = []
    rankable = [r for r in summary if r["threshold"] in pooled]
    if rankable:
        by_pos = _rank(rankable, "folds_positive", True)
        by_mean = _rank(rankable, "mean_of_means", True)
        by_sd = _rank(rankable, "stdev_means", False)
        by_t = _rank([{"threshold": t, "tstat": p["tstat"]} for t, p in pooled.items()], "tstat", True)
        for r in rankable:
            t = r["threshold"]
            ranking.append({**r, "tstat": pooled[t]["tstat"],
                            "composite": by_pos[t] + by_mean[t] + by_sd[t] + by_t[t]})
        ranking.sort(key=lambda r: r["composite"])
    return Result(cfg, files, len(grid), span, folds, cells, pooled, summary, ranking)


def render(result: Result) -> str:
    cfg = result.config
    stop = f"{cfg.stop_pct:+.2f}%" if cfg.stop_pct is not None else "none (timeout only)"
    ref = lambda t: "  (reference)" if abs(t - cfg.reference_threshold) < 1e-9 else ""
    out = [f"Threshold cross-validation - {cfg.symbol}",
           f"  files: {len(result.files)} ({result.files[0].name} .. {result.files[-1].name})",
           f"  signal: {'z60 only' if cfg.single_window else 'z60 and z30'}  hold={cfg.hold_min}m  "
           f"cooldown={cfg.cooldown_min}m  stop={stop}  cost={cfg.fee_bps:.2f}bps round trip"
           + ("  entry=next bar" if cfg.next_bar_entry else ""),
           f"  grid: {result.grid_minutes:,} minutes over {result.span_days:.1f} days", "", "Folds:"]
    for f in result.folds:
        a = pd.Timestamp(f["start_ms"], unit="ms", tz="UTC").strftime("%Y-%m-%d")
        b = pd.Timestamp(f["end_ms"], unit="ms", tz="UTC").strftime("%Y-%m-%d")
        out.append(f"  fold {f['idx']}: {a} .. {b} ({f['days']:.1f}d)")

    out += ["", rule(110), "MEAN NET PnL% PER FOLD", rule(110),
            f"  {'threshold':<11}" + "".join(f"{'fold' + str(f['idx']):>17}" for f in result.folds)
            + f"{'folds+':>9}{'mean':>9}{'stdev':>8}"]
    summary = {r["threshold"]: r for r in result.summary}
    for thr in sorted(set(cfg.thresholds)):
        row = f"  +/-{thr:<8.2f}"
        for f in result.folds:
            s = result.cells[(thr, f["idx"])]
            row += f"{'0 trades':>17}" if s["n"] == 0 else f"{s['mean']:>+9.3f} (n={s['n']:>3})"
        if thr in summary:
            r = summary[thr]
            row += (f"{r['folds_positive']:>6}/{r['folds_total']}{r['mean_of_means']:>+9.3f}"
                    f"{r['stdev_means']:>8.3f}")
        out.append(row)

    out += ["", rule(110), "POOLED ACROSS FOLDS", rule(110),
            f"  {'threshold':<11}{'n':>5}{'win%':>8}{'mean%':>9}{'sum%':>9}{'sharpe':>8}{'t':>7}"
            f"{'per day':>9}" + (f"{'stops':>10}" if cfg.stop_pct is not None else "")]
    for thr in sorted(set(cfg.thresholds)):
        s = result.pooled.get(thr)
        if s is None:
            out.append(f"  +/-{thr:<8.2f}    0 trades")
            continue
        line = (f"  +/-{thr:<8.2f}{s['n']:>5}{s['winr']:>7.1f}%{s['mean']:>+9.3f}{s['sum']:>+9.2f}"
                f"{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}{s['trades_per_day']:>9.2f}")
        if cfg.stop_pct is not None:
            line += f"{s['stop_exits']:>6}/{s['n']}"
        out.append(line + ref(thr))

    out += ["", rule(110), "RANKING (composite = sum of ranks; lower is better)", rule(110)]
    if not result.ranking:
        out.append("  no threshold produced trades")
        return "\n".join(out)
    out.append(f"  {'threshold':<11}{'folds+':>8}{'mean':>9}{'stdev':>8}{'t':>7}{'composite':>11}")
    for r in result.ranking:
        out.append(f"  +/-{r['threshold']:<8.2f}{r['folds_positive']:>6}/{r['folds_total']}"
                   f"{r['mean_of_means']:>+9.3f}{r['stdev_means']:>8.3f}{r['tstat']:>+7.2f}"
                   f"{r['composite']:>11}{ref(r['threshold'])}")
    best = result.ranking[0]
    p = result.pooled[best["threshold"]]
    out += ["", f"  best: +/-{best['threshold']:.2f} (composite {best['composite']})  pooled n={p['n']}  "
                f"win={p['winr']:.1f}%  mean={p['mean']:+.3f}%  sharpe={p['sharpe']:+.3f}  t={p['tstat']:+.2f}  "
                f"folds positive {best['folds_positive']}/{best['folds_total']}"]
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    suffix = f"stop{abs(cfg.stop_pct):.1f}" if cfg.stop_pct is not None else "nostop"
    path = outdir / f"threshold_cv_{cfg.symbol.lower()}_{suffix}.csv"
    folds = {f["idx"]: f for f in result.folds}
    pd.DataFrame([{"threshold": thr, "fold_idx": k, "fold_start_ms": folds[k]["start_ms"],
                   "fold_end_ms": folds[k]["end_ms"], "fold_days": folds[k]["days"],
                   "stop_pct": cfg.stop_pct, **s}
                  for (thr, k), s in result.cells.items()]).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro threshold-cv", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--thresholds", type=csv_list(float), default=d.thresholds)
    parser.add_argument("--reference-threshold", type=float, default=d.reference_threshold,
                        help="threshold flagged as the reference in the tables")
    parser.add_argument("--single-window", action="store_true", help="60-minute z-score only")
    parser.add_argument("--hold-min", type=int, default=d.hold_min)
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--stop-pct", type=float, default=None,
                        help="stop-loss in %% (negative; write --stop-pct=-1.0)")
    parser.add_argument("--next-bar-entry", action="store_true",
                        help="enter at the minute after the trigger instead of the trigger minute")
    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--outdir", type=Path, default=Path("results/threshold_cv"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, thresholds=args.thresholds,
                 reference_threshold=args.reference_threshold, single_window=args.single_window,
                 hold_min=args.hold_min, cooldown_min=args.cooldown_min, stop_pct=args.stop_pct,
                 next_bar_entry=args.next_bar_entry, n_folds=args.n_folds, fee_bps=args.fee_bps,
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    print(f"\nwrote {write_outputs(result, args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
