"""Stop-loss / take-profit grid for the dual-window imbalance z-score signal,
evaluated across chronological folds.

Signal: enter long when both the 60- and 30-minute z-scores of cross-venue
mean order-book imbalance exceed +threshold, short when both are below
-threshold. Exits: stop, take-profit, or timeout after ``hold_min`` minutes,
followed by a ``cooldown_min`` lockout. Entries and exits are at the minute
mid; ``fee_bps`` is charged per round trip.

Outputs:

* MAE/MFE distribution of the reference configuration (``reference_stop``, no TP)
* pooled and per-fold statistics for every (stop, TP) cell
* ``tp_sweep_<symbol>.csv`` (per-cell, per-fold stats) and
  ``tp_trades_<symbol>.csv`` (per-trade rows, consumed by ``hf-features``)

Usage::

    python -m hlmicro tp-multifold --symbol SOL
    python -m hlmicro tp-multifold --symbol SOL --tp-levels 0.3,0.5,0.7 --stop-levels -2.5,-1.0
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, default_capture_dir
from ..parse import discover, minute_grid
from ..schemas import BOOK_REST_PREFIX
from .common import describe, rolling_zscore, rule

log = logging.getLogger(__name__)

TRADE_COLUMNS = ["stop_pct", "tp_pct", "fold_idx", "side", "entry_ts_ms", "exit_ts_ms",
                 "entry_px", "exit_px", "minutes_held", "exit_reason", "ret_gross_pct",
                 "pnl_pct_net", "mae_pct", "mfe_pct"]


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "SOL"
    threshold: float = 2.5
    primary_window_min: int = 60
    confirm_window_min: int = 30
    hold_min: int = 240
    cooldown_min: int = 60
    tp_levels: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0, 1.5])
    stop_levels: list[float] = field(default_factory=lambda: [-2.5, -1.0])
    reference_stop: float = -2.5
    n_folds: int = 4
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    verdict_margin_pct: float = 0.05
    start: str | None = None
    end: str | None = None


def simulate(grid: pd.DataFrame, cfg: Config, stop_pct: float | None,
             tp_pct: float | None) -> list[dict]:
    """Single-position backtest on a minute grid. Returns one dict per trade.

    ``stop_pct`` is negative (``-1.0`` = exit at -1%); ``tp_pct`` positive.
    ``None`` (or 0 for TP) disables that exit.
    """
    obi = grid["obi_ntl"].to_numpy()
    mid = grid["mid"].to_numpy()
    ts_ms = grid["ts_ms"].to_numpy()
    z_primary = rolling_zscore(obi, cfg.primary_window_min)
    z_confirm = rolling_zscore(obi, cfg.confirm_window_min)
    fee_pct = cfg.fee_bps / 100.0

    trades: list[dict] = []
    side = None
    entry_idx = deadline = 0
    entry_px = 0.0
    last_exit = -10**9
    mae = mfe = 0.0
    for i in range(max(cfg.primary_window_min, cfg.confirm_window_min), len(grid)):
        px = mid[i]
        if not px > 0:
            continue
        if side is not None:
            ret = (px - entry_px) / entry_px * 100
            if side == "SHORT":
                ret = -ret
            mae, mfe = min(mae, ret), max(mfe, ret)
            reason = None
            if stop_pct is not None and ret <= stop_pct:
                reason = "stop"
            elif tp_pct and ret >= tp_pct:
                reason = "tp"
            elif i >= deadline:
                reason = "timeout"
            if reason:
                trades.append({
                    "side": side, "entry_ts_ms": int(ts_ms[entry_idx]), "exit_ts_ms": int(ts_ms[i]),
                    "entry_px": float(entry_px), "exit_px": float(px),
                    "minutes_held": i - entry_idx, "exit_reason": reason,
                    "ret_gross_pct": ret, "pnl_pct_net": ret - fee_pct,
                    "mae_pct": mae, "mfe_pct": mfe,
                })
                side = None
                last_exit = i
        if side is None and i - last_exit >= cfg.cooldown_min:
            zp, zc = z_primary[i], z_confirm[i]
            if np.isnan(zp) or np.isnan(zc):
                continue
            if zp > cfg.threshold and zc > cfg.threshold:
                side = "LONG"
            elif zp < -cfg.threshold and zc < -cfg.threshold:
                side = "SHORT"
            if side:
                entry_idx, entry_px, deadline = i, px, i + cfg.hold_min
                mae = mfe = 0.0
    return trades


def split_folds(grid: pd.DataFrame, n_folds: int) -> list[dict]:
    """Equal-duration chronological folds over the grid's time span."""
    first, last = int(grid["ts_ms"].iloc[0]), int(grid["ts_ms"].iloc[-1])
    edges = np.linspace(first, last, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        start, end = int(edges[k]), int(edges[k + 1])
        part = grid[(grid["ts_ms"] >= start) & (grid["ts_ms"] < end)].reset_index(drop=True)
        if len(part):
            folds.append({"idx": k, "start_ms": start, "end_ms": end,
                          "days": (end - start) / 86_400_000, "grid": part})
    return folds


def cell_stats(trades: list[dict]) -> dict:
    s = describe([t["pnl_pct_net"] for t in trades], ddof=1)
    return {"n": s["n"], "winr": s["win_rate"] * 100 if s["n"] else np.nan,
            "mean": s["mean"], "sharpe": s["sharpe"], "tstat": s["t"], "sum": s["total"]}


@dataclass
class Result:
    config: Config
    files: list[Path]
    grid_minutes: int
    span_days: float
    reference_trades: list[dict]
    folds: list[dict] = field(default_factory=list)
    cells: dict = field(default_factory=dict)    # (stop, tp, fold) -> stats
    pooled: dict = field(default_factory=dict)   # (stop, tp) -> (stats, trades)


def run(cfg: Config, mae_mfe_only: bool = False) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    log.info("loading %d files for %s", len(files), cfg.symbol)
    grid = minute_grid(files, cfg.symbol)
    if grid.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    span = (grid["ts_ms"].iloc[-1] - grid["ts_ms"].iloc[0]) / 86_400_000
    result = Result(config=cfg, files=files, grid_minutes=len(grid), span_days=span,
                    reference_trades=simulate(grid, cfg, cfg.reference_stop, None))
    if mae_mfe_only:
        return result

    result.folds = split_folds(grid, cfg.n_folds)
    if len(result.folds) < 2:
        raise ValueError(f"only {len(result.folds)} fold(s) with data; need at least 2")
    for stop in sorted(set(cfg.stop_levels)):
        for tp in sorted(set(cfg.tp_levels)):
            pooled_trades = []
            for fold in result.folds:
                trades = simulate(fold["grid"], cfg, stop if stop < 0 else None, tp if tp > 0 else None)
                for t in trades:
                    t["fold_idx"] = fold["idx"]
                result.cells[(stop, tp, fold["idx"])] = cell_stats(trades)
                pooled_trades.extend(trades)
            result.pooled[(stop, tp)] = (cell_stats(pooled_trades), pooled_trades)
    return result


def _distribution(values: list[float], label: str) -> str:
    if not values:
        return f"    {label}: n=0"
    a = sorted(values)
    n = len(a)
    return (f"    {label}: n={n}  min={a[0]:+.3f}  p10={a[n // 10]:+.3f}  p25={a[n // 4]:+.3f}  "
            f"med={median(a):+.3f}  p75={a[3 * n // 4]:+.3f}  p90={a[9 * n // 10]:+.3f}  "
            f"max={a[-1]:+.3f}  mean={float(np.mean(a)):+.3f}")


def _tp_label(tp: float) -> str:
    return "no_TP" if tp <= 0 else f"+{tp:.1f}%"


def render(result: Result) -> str:
    cfg = result.config
    out = [
        f"Stop/TP multi-fold sweep - {cfg.symbol}",
        f"  files: {len(result.files)} ({result.files[0].name} .. {result.files[-1].name})",
        f"  signal: z{cfg.primary_window_min} and z{cfg.confirm_window_min} beyond +/-{cfg.threshold}  "
        f"hold={cfg.hold_min}m  cooldown={cfg.cooldown_min}m  cost={cfg.fee_bps:.2f}bps round trip",
        f"  grid: {result.grid_minutes:,} minutes over {result.span_days:.1f} days",
        "",
        rule(100),
        f"MAE/MFE - reference configuration (stop={cfg.reference_stop:+.2f}%, no TP)",
        rule(100),
    ]
    trades = result.reference_trades
    if not trades:
        out.append("  no trades")
    else:
        winners = [t for t in trades if t["pnl_pct_net"] > 0]
        losers = [t for t in trades if t["pnl_pct_net"] <= 0]
        out.append("  WINNERS")
        out += [_distribution([t[k] for t in winners], label) for k, label in
                (("mfe_pct", "MFE"), ("mae_pct", "MAE"), ("pnl_pct_net", "net PnL"),
                 ("minutes_held", "minutes held"))]
        out += ["", "  LOSERS"]
        out += [_distribution([t[k] for t in losers], label) for k, label in
                (("mfe_pct", "MFE"), ("mae_pct", "MAE"), ("pnl_pct_net", "net PnL"))]
        if winners:
            retrace = [t["mfe_pct"] - t["pnl_pct_net"] for t in winners]
            med = median(retrace)
            band = "high (>0.30pp)" if med > 0.3 else "low (<0.10pp)" if med < 0.1 else "mid"
            out += ["", "  WINNERS: MFE minus final net PnL", _distribution(retrace, "retracement"),
                    f"  median retracement {med:.3f}pp - band: {band}"]
    if not result.folds:
        return "\n".join(out)

    out += ["", "Folds:"]
    for f in result.folds:
        a = pd.Timestamp(f["start_ms"], unit="ms", tz="UTC").strftime("%Y-%m-%d")
        b = pd.Timestamp(f["end_ms"], unit="ms", tz="UTC").strftime("%Y-%m-%d")
        out.append(f"  fold {f['idx']}: {a} .. {b} ({f['days']:.1f}d)")

    stops = sorted(set(cfg.stop_levels))
    tps = sorted(set(cfg.tp_levels))
    out += ["", rule(110), "POOLED STATISTICS BY (STOP, TP)", rule(110),
            f"  {'stop':<8}{'tp':<8}{'n':>4}{'win%':>8}{'mean%':>9}{'sum%':>9}"
            f"{'sharpe':>8}{'t':>7}   exits stop/tp/timeout"]
    for stop in stops:
        for tp in tps:
            s, trades = result.pooled[(stop, tp)]
            if s["n"] == 0:
                continue
            exits = [sum(t["exit_reason"] == r for t in trades) for r in ("stop", "tp", "timeout")]
            ref = "  (reference)" if abs(stop - cfg.reference_stop) < 0.01 and tp <= 0 else ""
            out.append(f"  {stop:<+7.2f} {_tp_label(tp):<8}{s['n']:>4}{s['winr']:>7.1f}%"
                       f"{s['mean']:>+9.3f}{s['sum']:>+9.2f}{s['sharpe']:>+8.3f}{s['tstat']:>+7.2f}"
                       f"   {exits[0]}/{exits[1]}/{exits[2]}{ref}")

    out += ["", rule(110), "MEAN NET PnL% PER FOLD", rule(110)]
    header = f"  {'stop':<8}{'tp':<8}" + "".join(
        f"{'fold' + str(f['idx']) + ' (' + format(f['days'], '.1f') + 'd)':>20}" for f in result.folds)
    out.append(header + f"{'folds+':>9}{'mean':>9}")
    for stop in stops:
        for tp in tps:
            row = f"  {stop:<+7.2f} {_tp_label(tp):<8}"
            means = []
            for f in result.folds:
                s = result.cells[(stop, tp, f["idx"])]
                if s["n"] == 0:
                    row += f"{'0 trades':>20}"
                else:
                    row += f"{s['mean']:>+11.3f} (n={s['n']:>3})"
                    means.append(s["mean"])
            if means:
                row += f"{sum(m > 0 for m in means):>6}/{len(means)}{float(np.mean(means)):>+9.3f}"
            out.append(row)

    base_stop = min((s for s in stops if abs(s - cfg.reference_stop) < 0.5),
                    key=lambda s: abs(s - cfg.reference_stop), default=None)
    base = result.pooled.get((base_stop, 0.0)) if base_stop is not None else None
    out += ["", rule(110), "TAKE-PROFIT LEVELS VS NO TP", rule(110)]
    if not base:
        out.append("  no-TP baseline not in grid (add 0 to --tp-levels)")
    else:
        b, _ = base
        out.append(f"  baseline stop={base_stop:+.2f}% no TP: n={b['n']}  mean={b['mean']:+.3f}%  "
                   f"sharpe={b['sharpe']:+.3f}  t={b['tstat']:+.2f}")
        out.append(f"  {'TP':<8}{'mean%':>9}{'delta pp':>10}{'sharpe':>8}{'t':>7}{'win%':>8}  "
                   f"vs baseline (margin {cfg.verdict_margin_pct:.2f}pp)")
        for tp in tps:
            if tp <= 0:
                continue
            s, _ = result.pooled[(base_stop, tp)]
            if s["n"] == 0:
                continue
            delta = s["mean"] - b["mean"]
            verdict = ("better" if delta > cfg.verdict_margin_pct else
                       "worse" if delta < -cfg.verdict_margin_pct else "no difference")
            out.append(f"  +{tp:<7.1f}{s['mean']:>+9.3f}{delta:>+10.3f}{s['sharpe']:>+8.3f}"
                       f"{s['tstat']:>+7.2f}{s['winr']:>7.1f}%  {verdict}")
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    sym = result.config.symbol.lower()
    sweep = outdir / f"tp_sweep_{sym}.csv"
    pd.DataFrame([{"stop_pct": stop, "tp_pct": tp, "fold_idx": k, **s}
                  for (stop, tp, k), s in result.cells.items()]).to_csv(sweep, index=False)
    trades = outdir / f"tp_trades_{sym}.csv"
    rows = [{"stop_pct": stop, "tp_pct": tp, **{c: t[c] for c in TRADE_COLUMNS[2:]}}
            for (stop, tp), (_, ts) in result.pooled.items() for t in ts]
    pd.DataFrame(rows, columns=TRADE_COLUMNS).to_csv(trades, index=False)
    return sweep, trades


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro tp-multifold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=d.threshold)
    parser.add_argument("--hold-min", type=int, default=d.hold_min)
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--tp-levels", type=csv_list(float), default=d.tp_levels,
                        help="take-profit levels in %%; 0 means no TP")
    parser.add_argument("--stop-levels", type=csv_list(float), default=d.stop_levels,
                        help="stop levels in %% (negative); 0 means no stop")
    parser.add_argument("--reference-stop", type=float, default=d.reference_stop,
                        help="stop used for the MAE/MFE diagnostic and the TP comparison")
    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--mae-mfe-only", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("results/tp_multifold"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, threshold=args.threshold,
                 hold_min=args.hold_min, cooldown_min=args.cooldown_min,
                 tp_levels=args.tp_levels, stop_levels=args.stop_levels,
                 reference_stop=args.reference_stop, n_folds=args.n_folds,
                 fee_bps=args.fee_bps, start=args.start, end=args.end)
    result = run(cfg, mae_mfe_only=args.mae_mfe_only)
    print(render(result))
    if not args.mae_mfe_only:
        sweep, trades = write_outputs(result, args.outdir)
        print(f"\nwrote {sweep}\nwrote {trades}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
