"""Tick-resolution scalp signal explorer with a passive-fill realism check.

Builds a continuous tick grid (``tick_s`` seconds; websocket books for ticks
of 2 s or less, otherwise 5-second REST snapshots) of per-venue and
cross-venue order-book imbalance, per-venue taker-flow imbalance (TFI) from
the Hyperliquid, OKX and Bybit tapes, and Hyperliquid liquidation flow. Then
it sweeps single-feature threshold rules:

* z-scores (``*_z<W>s``) of cross-venue imbalance, its dispersion, venue
  agreement, Hyperliquid-minus-average imbalance, mean TFI and each venue's
  imbalance, plus ``liq_spike_z`` (5-minute mean liquidation notional against
  its 60-minute baseline), at ``z_thresholds``
* rolling TFI per venue (``tfi_<venue>_<W>s``) and ``liq_imbalance`` at
  ``imb_thresholds``

A rule fires on the tick where the feature first crosses the threshold
(``above``: up through +thr, ``below``: down through -thr), is traded with
(``follow``) or against (``fade``) it, enters at the next tick's mid (no
look-ahead), exits after a fixed hold, and never overlaps the previous
position. Every rule is scored under taker, maker and zero fees, with a
50/50 split by trade order.

Maker-fee rules with positive Sharpe overall and in both halves are ranked,
and the best ``realism_top_n`` are re-run with passive limits ``k_bps``
inside the mid that fill only when the Hyperliquid tape trades through them.

Usage::

    python -m hlmicro scalp-explore --symbol ETH --tick-sec 5
    python -m hlmicro scalp-explore --symbol ETH --tick-sec 1 --no-realism
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, FeeModel, default_capture_dir
from ..parse import (book_files_for_resolution, bucket_venue_means, discover, load_liquidations,
                     tape_per_bucket, tape_prefix)
from ..schemas import HL_LIQUIDATIONS_PREFIX
from .common import describe, rolling_zscore, rule
from .passive_fills import simulate_passive_fills

log = logging.getLogger(__name__)

LIQ_SPIKE_WINDOW_S = 300
LIQ_BASELINE_WINDOW_S = 3600


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "ETH"
    tick_s: int = 5
    flow_venues: list[str] = field(default_factory=lambda: ["hl", "okx", "bybit"])
    holds_s: list[int] = field(default_factory=lambda: [60, 120, 300, 600, 900, 1800])
    z_windows_s: list[int] = field(default_factory=lambda: [300, 900, 3600])
    z_thresholds: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    imb_windows_s: list[int] = field(default_factory=lambda: [60, 300, 900])
    imb_thresholds: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6])
    min_trades: int = 30
    top_n: int = 30
    realism: bool = True
    realism_top_n: int = 15
    k_bps: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    entry_patience_s: int = 5
    exit_patience_s: int = 5
    fees: FeeModel = DEFAULT_FEES
    start: str | None = None
    end: str | None = None

    def fee_regimes(self) -> dict[str, float]:
        return {"taker": self.fees.taker_round_trip_bps / 1e4,
                "maker": self.fees.maker_round_trip_bps / 1e4, "zero": 0.0}

    def ticks(self, seconds: int) -> int:
        return max(1, seconds // self.tick_s)


def build_features(cfg: Config) -> pd.DataFrame:
    """Continuous tick grid of book, flow and liquidation features."""
    tick_ms = cfg.tick_s * 1000
    book = bucket_venue_means(book_files_for_resolution(cfg.capture_dir, cfg.tick_s, cfg.start, cfg.end),
                              cfg.symbol, tick_ms)
    if book.empty:
        return pd.DataFrame()
    obi = book.pivot(index="bucket", columns="venue", values="obi_ntl").astype("float32")
    mid = book.pivot(index="bucket", columns="venue", values="mid").astype("float32")
    del book
    venues = list(obi.columns)
    grid = pd.DataFrame(index=obi.index)
    for v in venues:
        grid[f"obi_{v}"] = obi[v]
    grid["obi_mean"] = obi.mean(axis=1).astype("float32")
    grid["obi_std"] = obi.std(axis=1).astype("float32")
    grid["n_venues"] = obi.count(axis=1).astype("float32")
    signs = np.sign(obi.to_numpy())
    grid["venue_agreement"] = ((signs == np.sign(grid["obi_mean"].to_numpy())[:, None]) & (signs != 0)
                               ).sum(axis=1).astype("float32")
    if HL_VENUE in obi:
        grid["hl_minus_avg_obi"] = (obi[HL_VENUE] - grid["obi_mean"]).astype("float32")
        cex = [v for v in venues if v != HL_VENUE]
        if cex:
            grid["hl_minus_cex_obi"] = (obi[HL_VENUE] - obi[cex].mean(axis=1)).astype("float32")
    grid["mid"] = (mid[HL_VENUE] if HL_VENUE in mid else mid.mean(axis=1)).astype("float32")
    del obi, mid, signs

    grid = grid.reset_index().rename(columns={"bucket": "tick_ts"})
    full = pd.DataFrame({"tick_ts": np.arange(grid["tick_ts"].min(), grid["tick_ts"].max() + 1, dtype=np.int64)})
    grid = full.merge(grid, on="tick_ts", how="left")

    tfi_cols = []
    for v in cfg.flow_venues:
        tape = tape_per_bucket(discover(cfg.capture_dir, tape_prefix(v), cfg.start, cfg.end),
                               cfg.symbol, tick_ms, v)
        if tape.empty:
            continue
        grid = grid.merge(tape.rename(columns={"bucket": "tick_ts"}), on="tick_ts", how="left")
        buy, sell = grid[f"{v}_buy_ntl"], grid[f"{v}_sell_ntl"]
        grid[f"tfi_{v}"] = ((buy - sell) / (buy + sell).replace(0, np.nan)).clip(-1, 1).astype("float32")
        tfi_cols.append(f"tfi_{v}")
        buy, sell = buy.fillna(0.0), sell.fillna(0.0)
        for w_s in cfg.imb_windows_s:
            w = cfg.ticks(w_s)
            b = buy.rolling(w, min_periods=w).sum()
            s = sell.rolling(w, min_periods=w).sum()
            grid[f"tfi_{v}_{w_s}s"] = ((b - s) / (b + s).replace(0, np.nan)).clip(-1, 1).astype("float32")
        gc.collect()
    if tfi_cols:
        grid["tfi_mean"] = grid[tfi_cols].mean(axis=1).astype("float32")

    liq = load_liquidations(discover(cfg.capture_dir, HL_LIQUIDATIONS_PREFIX, cfg.start, cfg.end), cfg.symbol)
    if not liq.empty:
        per_tick = pd.DataFrame({
            "tick_ts": liq["ts_ms"].to_numpy() // tick_ms,
            "long_liq_ntl": np.where(liq["sign"] < 0, liq["notional"], 0.0),
            "short_liq_ntl": np.where(liq["sign"] > 0, liq["notional"], 0.0),
        }).groupby("tick_ts", as_index=False).agg(long_liq_ntl=("long_liq_ntl", "sum"),
                                                  short_liq_ntl=("short_liq_ntl", "sum"),
                                                  liq_count=("long_liq_ntl", "size"))
        grid = grid.merge(per_tick, on="tick_ts", how="left")
        for c in ("long_liq_ntl", "short_liq_ntl", "liq_count"):
            grid[c] = grid[c].fillna(0.0)
        total = grid["long_liq_ntl"] + grid["short_liq_ntl"]
        grid["liq_imbalance"] = ((grid["long_liq_ntl"] - grid["short_liq_ntl"])
                                 / total.replace(0, np.nan)).clip(-1, 1).astype("float32")
        grid["liq_total"] = total.astype("float32")
        base = cfg.ticks(LIQ_BASELINE_WINDOW_S)
        spike = total.rolling(cfg.ticks(LIQ_SPIKE_WINDOW_S)).mean()
        sd = total.rolling(base, min_periods=base).std().replace(0, np.nan)
        grid["liq_spike_z"] = ((spike - total.rolling(base, min_periods=base).mean()) / sd).astype("float32")

    targets = ["obi_mean", "obi_std", "venue_agreement", "hl_minus_avg_obi", "hl_minus_cex_obi", "tfi_mean"]
    targets += [f"obi_{v}" for v in venues]
    for col in (c for c in targets if c in grid):
        for w_s in cfg.z_windows_s:
            grid[f"{col}_z{w_s}s"] = rolling_zscore(grid[col].to_numpy(), cfg.ticks(w_s)).astype("float32")
        gc.collect()
    return grid


def feature_rules(grid: pd.DataFrame, cfg: Config):
    """(feature, threshold, direction) for every rule in the sweep."""
    z_cols = [c for c in grid if "_z" in c and c.rsplit("_z", 1)[-1][:-1].isdigit() and c.endswith("s")]
    if "liq_spike_z" in grid:
        z_cols.append("liq_spike_z")
    imb_cols = [c for c in grid if c.startswith("tfi_") and c.endswith("s") and c.split("_")[-1][:-1].isdigit()]
    if "liq_imbalance" in grid:
        imb_cols.append("liq_imbalance")
    for cols, thresholds in ((z_cols, cfg.z_thresholds), (imb_cols, cfg.imb_thresholds)):
        for col in cols:
            for thr in thresholds:
                yield col, thr, "above"
                yield col, thr, "below"


def threshold_cross(values: np.ndarray, thr: float, direction: str) -> np.ndarray:
    """Ticks where the value crosses up through +thr (``above``) or down through -thr (``below``)."""
    if direction == "above":
        hit = (values[1:] >= thr) & (values[:-1] < thr)
    else:
        hit = (values[1:] <= -thr) & (values[:-1] > -thr)
    return np.flatnonzero(hit) + 1


def no_overlap(idx: np.ndarray, hold_ticks: int) -> np.ndarray:
    """Triggers that arrive after the previous position (entry at i+1, exit
    at i+1+hold) has closed."""
    kept, j = [], 0
    while j < len(idx):
        kept.append(idx[j])
        j = np.searchsorted(idx, idx[j] + 1 + hold_ticks, side="right")
    return np.asarray(kept, dtype=np.int64)


def mid_returns(mid: np.ndarray, idx: np.ndarray, hold_ticks: int, side: int) -> np.ndarray:
    """Gross return from mid[i+1] to mid[i+1+hold] for each trigger."""
    entry, exit_ = idx + 1, idx + 1 + hold_ticks
    ok = exit_ < len(mid)
    e, x = mid[entry[ok]], mid[exit_[ok]]
    valid = np.isfinite(e) & np.isfinite(x) & (e > 0)
    return side * (x[valid] - e[valid]) / e[valid]


def _score(net: np.ndarray, min_trades: int) -> dict | None:
    if len(net) < min_trades:
        return None
    s = describe(net, ddof=0)
    return {"n": s["n"], "win_rate": s["win_rate"], "mean": s["mean"], "sharpe": s["sharpe"], "total": s["total"]}


def explore(grid: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mid = grid["mid"].to_numpy(np.float64)
    regimes = cfg.fee_regimes()
    rows = []
    for col, thr, direction in feature_rules(grid, cfg):
        triggers = threshold_cross(grid[col].to_numpy(np.float64), thr, direction)
        if len(triggers) < cfg.min_trades:
            continue
        signal_side = 1 if direction == "above" else -1
        for hold_s in cfg.holds_s:
            hold = cfg.ticks(hold_s)
            kept = no_overlap(triggers, hold)
            if len(kept) < cfg.min_trades:
                continue
            for behavior, side in (("follow", signal_side), ("fade", -signal_side)):
                gross = mid_returns(mid, kept, hold, side)
                for regime, fee in regimes.items():
                    net = gross - fee
                    full = _score(net, cfg.min_trades)
                    if full is None:
                        continue
                    first = second = None
                    if len(net) >= 2 * cfg.min_trades:
                        cut = len(net) // 2
                        first, second = _score(net[:cut], cfg.min_trades), _score(net[cut:], cfg.min_trades)
                    rows.append({
                        "feature": col, "threshold": thr, "direction": direction, "behavior": behavior,
                        "hold_s": hold_s, "hold_ticks": hold, "fees": regime, "n": full["n"],
                        "win_rate": round(full["win_rate"], 4), "mean_bps": round(full["mean"] * 1e4, 3),
                        "sharpe": round(full["sharpe"], 3), "total_bps": round(full["total"] * 1e4, 1),
                        "sharpe_first_half": round(first["sharpe"], 3) if first else None,
                        "sharpe_second_half": round(second["sharpe"], 3) if second else None,
                        "mean_bps_first_half": round(first["mean"] * 1e4, 3) if first else None,
                        "mean_bps_second_half": round(second["mean"] * 1e4, 3) if second else None,
                    })
    return pd.DataFrame(rows)


def survivors(sweep: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Rules under ``regime`` fees with positive Sharpe overall and in both halves, best first."""
    if sweep.empty:
        return sweep
    rows = sweep[(sweep["fees"] == regime) & (sweep["sharpe"] > 0)
                 & (sweep["sharpe_first_half"].fillna(-1) > 0) & (sweep["sharpe_second_half"].fillna(-1) > 0)]
    return rows.sort_values("sharpe", ascending=False).reset_index(drop=True)


def realism(grid: pd.DataFrame, top: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mid-price maker baseline and passive fills at each ``k_bps`` for each rule."""
    mid = grid["mid"].to_numpy(np.float64)
    entry_p, exit_p = cfg.ticks(cfg.entry_patience_s), cfg.ticks(cfg.exit_patience_s)
    rows = []
    for _, r in top.iterrows():
        signal_side = 1 if r["direction"] == "above" else -1
        side = signal_side if r["behavior"] == "follow" else -signal_side
        hold = int(r["hold_ticks"])
        signal = f"{r['feature']}@{r['threshold']:.2f}_{r['direction']}_{r['behavior']}_h{int(r['hold_s'])}s"
        triggers = threshold_cross(grid[r["feature"]].to_numpy(np.float64), float(r["threshold"]), r["direction"])
        net = mid_returns(mid, no_overlap(triggers, hold), hold, side) - cfg.fees.maker_round_trip_bps / 1e4
        if len(net) >= cfg.min_trades:
            s = describe(net, ddof=0)
            rows.append({"signal": signal, "hold_s": int(r["hold_s"]), "fill": "mid (maker fees)",
                         "k_bps": np.nan, "n": s["n"], "n_triggers": len(triggers), "entry_fill_rate": 1.0,
                         "exit_maker_rate": 1.0, "mean_net_bps": round(s["mean"] * 1e4, 3),
                         "sharpe": round(s["sharpe"], 3), "total_bps": round(s["total"] * 1e4, 1)})
        for k in cfg.k_bps:
            sim = simulate_passive_fills(grid, triggers, hold, side, float(k), entry_p, exit_p, cfg.fees)
            a = sim.attrs
            eligible = max(1, a["n_triggers"] - a["n_in_position"])
            row = {"signal": signal, "hold_s": int(r["hold_s"]), "fill": f"passive k={k:g}bps", "k_bps": float(k),
                   "n": len(sim), "n_triggers": a["n_triggers"],
                   "entry_fill_rate": round(a["n_filled"] / eligible, 3)}
            if len(sim):
                net = sim["net_ret"]
                row.update({"exit_maker_rate": round(float(sim["exit_was_maker"].mean()), 3),
                            "mean_net_bps": round(float(net.mean()) * 1e4, 3),
                            "sharpe": round(float(net.mean() / net.std()), 3) if net.std() > 0 else 0.0,
                            "total_bps": round(float(net.sum()) * 1e4, 1)})
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Result:
    config: Config
    ticks: int
    span_days: float
    columns: int
    hl_tape_coverage: float
    sweep: pd.DataFrame
    top: pd.DataFrame
    realism: pd.DataFrame


def run(cfg: Config) -> Result:
    started = time.time()
    grid = build_features(cfg)
    if grid.empty:
        raise ValueError(f"no book data for {cfg.symbol}")
    log.info("feature grid %s built in %.1fs", grid.shape, time.time() - started)
    sweep = explore(grid, cfg)
    top = survivors(sweep, "maker").head(cfg.top_n)
    real = pd.DataFrame()
    if cfg.realism and not top.empty:
        if "hl_px_min" in grid:
            real = realism(grid, top.head(cfg.realism_top_n), cfg)
        else:
            log.warning("no Hyperliquid trade tape for %s: passive-fill stage skipped", cfg.symbol)
    coverage = float(grid["hl_px_min"].notna().mean()) if "hl_px_min" in grid else 0.0
    span = (grid["tick_ts"].max() - grid["tick_ts"].min()) * cfg.tick_s / 86_400
    return Result(cfg, len(grid), span, grid.shape[1], coverage, sweep, top, real)


def render(result: Result) -> str:
    cfg = result.config
    cols = ["feature", "threshold", "direction", "behavior", "hold_s", "n", "win_rate", "mean_bps", "sharpe",
            "sharpe_first_half", "sharpe_second_half"]
    out = [f"Scalp explorer - {cfg.symbol}  tick={cfg.tick_s}s",
           f"  grid: {result.ticks:,} ticks over {result.span_days:.1f} days, {result.columns} columns; "
           f"Hyperliquid prints in {result.hl_tape_coverage:.1%} of ticks",
           f"  fees per side: taker {cfg.fees.taker_bps:.2f} bps, maker {cfg.fees.maker_bps:.2f} bps",
           f"  holds {cfg.holds_s}s  z-windows {cfg.z_windows_s}s  rules scored: {len(result.sweep):,}",
           "", rule(110), f"TOP {cfg.top_n} MAKER-FEE RULES (positive Sharpe overall and in both halves)", rule(110)]
    if result.top.empty:
        out.append("  no rule survives at maker fees")
        zero = survivors(result.sweep, "zero").head(10)
        if zero.empty:
            out.append("  none at zero fees either: no edge at this resolution")
        else:
            out += ["", "  zero-fee survivors (edge before fees):", zero[cols].to_string(index=False)]
    else:
        out.append(result.top[cols].to_string(index=False))
    if not result.realism.empty:
        out += ["", rule(110), f"PASSIVE-FILL REALISM on the top {cfg.realism_top_n} rules", rule(110)]
        view = ["fill", "n", "entry_fill_rate", "exit_maker_rate", "mean_net_bps", "sharpe", "total_bps"]
        for signal, block in result.realism.groupby("signal", sort=False):
            out += [f"  {signal}", block.reindex(columns=view).to_string(index=False), ""]
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"scalp_{result.config.symbol.lower()}_t{result.config.tick_s}s"
    paths = []
    for name, frame in (("full", result.sweep), ("top_maker", result.top), ("realism", result.realism)):
        if not frame.empty:
            paths.append(outdir / f"{stem}_{name}.csv")
            frame.to_csv(paths[-1], index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro scalp-explore", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--tick-sec", type=int, default=d.tick_s,
                        help="grid resolution; 2 s or less uses websocket books")
    parser.add_argument("--holds-s", type=csv_list(int), default=d.holds_s)
    parser.add_argument("--min-trades", type=int, default=d.min_trades)
    parser.add_argument("--top-n", type=int, default=d.top_n)
    parser.add_argument("--no-realism", action="store_true", help="skip the passive-fill stage")
    parser.add_argument("--realism-top-n", type=int, default=d.realism_top_n)
    parser.add_argument("--k-bps", type=float, nargs="+", default=d.k_bps)
    parser.add_argument("--entry-patience-sec", type=int, default=d.entry_patience_s)
    parser.add_argument("--exit-patience-sec", type=int, default=d.exit_patience_s)
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--maker-bps", type=float, default=d.fees.maker_bps)
    parser.add_argument("--outdir", type=Path, default=Path("results/scalp_explore"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, tick_s=args.tick_sec, holds_s=args.holds_s,
                 min_trades=args.min_trades, top_n=args.top_n, realism=not args.no_realism,
                 realism_top_n=args.realism_top_n, k_bps=args.k_bps, entry_patience_s=args.entry_patience_sec,
                 exit_patience_s=args.exit_patience_sec,
                 fees=FeeModel(taker_bps=args.taker_bps, maker_bps=args.maker_bps),
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    for path in write_outputs(result, args.outdir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
