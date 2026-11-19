"""Short-horizon AND-gate entry rules on tick-resolution data.

Two rule families, each with buy and sell variants, built from Hyperliquid
order-book imbalance (``obi_hyperliquid``) and the Hyperliquid trade tape:

``tick_speed``
    trade-count z-score (15 min) >= 2, trade-flow imbalance (60 s) beyond
    +/-0.2 and imbalance on the same side. Fixed holds.
``vwap_dev``
    mid at least 20 bps beyond the 5-minute VWAP, flow and imbalance pointing
    back towards it. Exit when the mid returns to VWAP (with a time failsafe),
    plus fixed holds for comparison.

A rule fires on the tick where all its conditions first become true. Entry
is at the next tick (no look-ahead) and positions never overlap. Results are
reported under taker, maker and zero fees, with first-half / second-half
Sharpe. An optional fill simulation posts passive limits ``k_bps`` inside the
mid, checks fills against the tape's traded range, and falls back to a taker
exit when the exit limit does not fill.

Usage::

    python -m hlmicro multi-gate --symbol ETH --tick-sec 5
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

from ..cli import add_common_args, add_date_args, setup_logging
from ..config import DEFAULT_FEES, FeeModel, default_capture_dir
from ..parse import book_files_for_resolution, bucket_venue_means, discover, tape_per_bucket
from ..schemas import HL_TRADES_PREFIX
from .common import describe

log = logging.getLogger(__name__)

TFI_WINDOW_S = 60
COUNT_Z_WINDOW_S = 900
VWAP_WINDOW_S = 300
TFI_COL = f"tfi_hl_{TFI_WINDOW_S}s"
COUNT_Z_COL = f"hl_count_z{COUNT_Z_WINDOW_S}s"
VWAP_COL = f"vwap_{VWAP_WINDOW_S}s"
VWAP_DEV_COL = f"price_vs_vwap_{VWAP_WINDOW_S}s_bps"


@dataclass
class Rule:
    name: str
    side: int
    conditions: list[tuple[str, float, str]]   # (column, threshold, "above" | "below")
    holds_s: list[int]
    vwap_exit: bool = False


RULES = [
    Rule("tick_speed_buy", +1, [(COUNT_Z_COL, 2.0, "above"), (TFI_COL, 0.2, "above"),
                                ("obi_hyperliquid", 0.0, "above")], [30, 60, 120, 300]),
    Rule("tick_speed_sell", -1, [(COUNT_Z_COL, 2.0, "above"), (TFI_COL, -0.2, "below"),
                                 ("obi_hyperliquid", 0.0, "below")], [30, 60, 120, 300]),
    Rule("vwap_dev_buy", +1, [(VWAP_DEV_COL, -20.0, "below"), (TFI_COL, 0.0, "above"),
                              ("obi_hyperliquid", 0.0, "above")], [120, 300, 600, 1800], vwap_exit=True),
    Rule("vwap_dev_sell", -1, [(VWAP_DEV_COL, 20.0, "above"), (TFI_COL, 0.0, "below"),
                               ("obi_hyperliquid", 0.0, "below")], [120, 300, 600, 1800], vwap_exit=True),
]


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "ETH"
    tick_s: int = 5
    vwap_failsafe_s: int = 1800
    fills: bool = True
    k_bps: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    entry_patience_s: int = 5
    exit_patience_s: int = 5
    min_trades: int = 20
    fees: FeeModel = DEFAULT_FEES
    start: str | None = None
    end: str | None = None

    def fee_regimes(self) -> dict[str, float]:
        return {"taker": self.fees.taker_round_trip_bps / 1e4,
                "maker": self.fees.maker_round_trip_bps / 1e4, "zero": 0.0}


def build_features(cfg: Config) -> pd.DataFrame:
    """Continuous tick grid with per-venue imbalance, HL mid, and HL flow features."""
    bucket_ms = cfg.tick_s * 1000
    book_files = book_files_for_resolution(cfg.capture_dir, cfg.tick_s, cfg.start, cfg.end)
    tape_files = discover(cfg.capture_dir, HL_TRADES_PREFIX, cfg.start, cfg.end)
    log.info("book files: %d  HL tape files: %d", len(book_files), len(tape_files))
    book = bucket_venue_means(book_files, cfg.symbol, bucket_ms)
    if book.empty:
        return pd.DataFrame()
    tape = tape_per_bucket(tape_files, cfg.symbol, bucket_ms, "hl")

    obi = book.pivot(index="bucket", columns="venue", values="obi_ntl").astype("float32")
    mid = book.pivot(index="bucket", columns="venue", values="mid").astype("float32")
    grid = pd.DataFrame(index=obi.index)
    for venue in obi.columns:
        grid[f"obi_{venue}"] = obi[venue]
    grid["obi_mean"] = obi.mean(axis=1).astype("float32")
    grid["mid"] = (mid["hyperliquid"] if "hyperliquid" in mid else mid.mean(axis=1)).astype("float32")
    del obi, mid, book
    grid = grid.reset_index().rename(columns={"bucket": "tick_ts"})
    full = pd.DataFrame({"tick_ts": np.arange(grid["tick_ts"].min(), grid["tick_ts"].max() + 1, dtype=np.int64)})
    grid = full.merge(grid, on="tick_ts", how="left")
    if not tape.empty:
        grid = grid.merge(tape.rename(columns={"bucket": "tick_ts"}), on="tick_ts", how="left")
    grid = grid.sort_values("tick_ts").reset_index(drop=True)
    gc.collect()

    for col in ("hl_buy_ntl", "hl_sell_ntl", "hl_count", "hl_sum_ntl", "hl_sum_sz"):
        if col not in grid:
            grid[col] = np.nan

    w = max(1, TFI_WINDOW_S // cfg.tick_s)
    buys = grid["hl_buy_ntl"].fillna(0).rolling(w, min_periods=w).sum()
    sells = grid["hl_sell_ntl"].fillna(0).rolling(w, min_periods=w).sum()
    grid[TFI_COL] = ((buys - sells) / (buys + sells).replace(0, np.nan)).clip(-1, 1).astype("float32")

    w = max(1, COUNT_Z_WINDOW_S // cfg.tick_s)
    count = grid["hl_count"].fillna(0)
    grid[COUNT_Z_COL] = ((count - count.rolling(w, min_periods=w).mean())
                         / count.rolling(w, min_periods=w).std().replace(0, np.nan)).astype("float32")

    w = max(1, VWAP_WINDOW_S // cfg.tick_s)
    notional = grid["hl_sum_ntl"].fillna(0).rolling(w, min_periods=w).sum()
    size = grid["hl_sum_sz"].fillna(0).rolling(w, min_periods=w).sum().replace(0, np.nan)
    grid[VWAP_COL] = (notional / size).astype("float32")
    grid[VWAP_DEV_COL] = ((grid["mid"] - grid[VWAP_COL]) / grid[VWAP_COL] * 1e4).astype("float32")
    return grid


def triggers(grid: pd.DataFrame, conditions: list[tuple[str, float, str]]) -> np.ndarray:
    """Indices where all conditions become true together (NaN never satisfies)."""
    met = np.ones(len(grid), dtype=bool)
    for column, threshold, direction in conditions:
        if column not in grid:
            return np.empty(0, dtype=np.int64)
        values = grid[column].to_numpy()
        met &= (values >= threshold) if direction == "above" else (values <= threshold)
    fire = np.zeros(len(grid), dtype=bool)
    fire[1:] = met[1:] & ~met[:-1]
    return np.flatnonzero(fire)


def simulate_fixed_hold(grid: pd.DataFrame, idx: np.ndarray, hold_ticks: int, side: int) -> pd.DataFrame:
    """Mid-to-mid return from tick i+1 to i+1+hold."""
    mid, ts = grid["mid"].to_numpy(), grid["tick_ts"].to_numpy()
    rows, busy_until = [], -1
    for i in idx:
        if i <= busy_until:
            continue
        entry, exit_ = i + 1, i + 1 + hold_ticks
        if exit_ >= len(grid):
            continue
        e, x = mid[entry], mid[exit_]
        if not (np.isfinite(e) and np.isfinite(x) and e > 0):
            continue
        rows.append({"entry_tick": int(ts[entry]), "exit_tick": int(ts[exit_]), "entry_px": float(e),
                     "exit_px": float(x), "gross_ret": float(side * (x - e) / e),
                     "ticks_held": int(ts[exit_] - ts[entry])})
        busy_until = exit_
    return pd.DataFrame(rows)


def simulate_vwap_exit(grid: pd.DataFrame, idx: np.ndarray, side: int, failsafe_ticks: int) -> pd.DataFrame:
    """Hold from tick i+1 until the mid crosses back to VWAP, or the failsafe."""
    mid, vwap, ts = grid["mid"].to_numpy(), grid[VWAP_COL].to_numpy(), grid["tick_ts"].to_numpy()
    n = len(grid)
    rows, busy_until = [], -1
    for i in idx:
        if i <= busy_until or i + 1 >= n:
            continue
        entry = i + 1
        e = mid[entry]
        if not (np.isfinite(e) and e > 0):
            continue
        end = min(n, entry + 1 + failsafe_ticks)
        exit_, hit = None, False
        for j in range(entry + 1, end):
            m, v = mid[j], vwap[j]
            if np.isfinite(m) and np.isfinite(v) and m > 0 and (m >= v if side > 0 else m <= v):
                exit_, hit = j, True
                break
        if exit_ is None:
            exit_ = min(end - 1, n - 1)
            if not (np.isfinite(mid[exit_]) and mid[exit_] > 0):
                continue
        x = mid[exit_]
        rows.append({"entry_tick": int(ts[entry]), "exit_tick": int(ts[exit_]), "entry_px": float(e),
                     "exit_px": float(x), "gross_ret": float(side * (x - e) / e),
                     "ticks_held": int(ts[exit_] - ts[entry]), "target_hit": hit})
        busy_until = exit_
    return pd.DataFrame(rows)


def simulate_passive_fills(grid: pd.DataFrame, idx: np.ndarray, hold_ticks: int, side: int,
                           k_bps: float, entry_patience: int, exit_patience: int,
                           fees: FeeModel) -> pd.DataFrame:
    """Passive limit ``k_bps`` inside the mid at entry and exit; a limit fills
    when the tape trades through it within the patience window. Unfilled exits
    are taken at the mid with taker fees."""
    if "hl_px_min" not in grid or "hl_px_max" not in grid:
        raise ValueError("fill simulation needs the Hyperliquid trade tape")
    mid, lo, hi, ts = (grid[c].to_numpy() for c in ("mid", "hl_px_min", "hl_px_max", "tick_ts"))
    n = len(grid)
    maker, taker = fees.maker_bps / 1e4, fees.taker_bps / 1e4
    rows, busy_until = [], -1
    counts = {"n_triggers": 0, "n_filled": 0, "n_exit_maker": 0, "n_exit_taker": 0}
    for i in idx:
        counts["n_triggers"] += 1
        if i <= busy_until or not (np.isfinite(mid[i]) and mid[i] > 0):
            continue
        offset = k_bps * 1e-4 * mid[i]
        post_entry = mid[i] - offset if side > 0 else mid[i] + offset
        entry = next((k for k in range(i + 1, min(i + 1 + entry_patience, n))
                      if (side > 0 and np.isfinite(lo[k]) and lo[k] <= post_entry)
                      or (side < 0 and np.isfinite(hi[k]) and hi[k] >= post_entry)), None)
        if entry is None:
            continue
        counts["n_filled"] += 1
        ref = entry + hold_ticks
        if ref >= n or not (np.isfinite(mid[ref]) and mid[ref] > 0):
            continue
        offset = k_bps * 1e-4 * mid[ref]
        post_exit = mid[ref] + offset if side > 0 else mid[ref] - offset
        exit_ = next((k for k in range(ref + 1, min(ref + 1 + exit_patience, n))
                      if (side > 0 and np.isfinite(hi[k]) and hi[k] >= post_exit)
                      or (side < 0 and np.isfinite(lo[k]) and lo[k] <= post_exit)), None)
        exit_px, exit_maker = post_exit, True
        if exit_ is None:
            exit_ = min(ref + exit_patience, n - 1)
            if not (np.isfinite(mid[exit_]) and mid[exit_] > 0):
                continue
            exit_px, exit_maker = mid[exit_], False
            counts["n_exit_taker"] += 1
        else:
            counts["n_exit_maker"] += 1
        gross = side * (exit_px - post_entry) / post_entry
        rows.append({"entry_tick": int(ts[entry]), "exit_tick": int(ts[exit_]),
                     "entry_px": float(post_entry), "exit_px": float(exit_px), "gross_ret": float(gross),
                     "net_ret": float(gross - (maker + (maker if exit_maker else taker))),
                     "exit_was_maker": exit_maker})
        busy_until = exit_
    frame = pd.DataFrame(rows)
    frame.attrs.update(counts)
    return frame


def _score(net: np.ndarray, min_trades: int) -> dict | None:
    if len(net) < min_trades:
        return None
    s = describe(net, ddof=0)
    return {"n": s["n"], "win_rate": s["win_rate"], "mean": s["mean"], "sharpe": s["sharpe"], "total": s["total"]}


def _halves(net: np.ndarray, min_trades: int) -> tuple[dict | None, dict | None]:
    if len(net) < 2 * min_trades:
        return None, None
    cut = int(len(net) * 0.5)
    return _score(net[:cut], min_trades), _score(net[cut:], min_trades)


def _rows(rule: Rule, exit_kind: str, hold_s: int, gross: np.ndarray, cfg: Config, extra: dict) -> list[dict]:
    rows = []
    for regime, fee in cfg.fee_regimes().items():
        net = gross - fee
        full = _score(net, cfg.min_trades)
        if full is None:
            continue
        first, second = _halves(net, cfg.min_trades)
        rows.append({"rule": rule.name, "exit": exit_kind, "hold_s": hold_s, "fees": regime,
                     "n": full["n"], "win_rate": round(full["win_rate"], 4),
                     "mean_bps": round(full["mean"] * 1e4, 3), "sharpe": round(full["sharpe"], 3),
                     "total_bps": round(full["total"] * 1e4, 1),
                     "sharpe_first_half": round(first["sharpe"], 3) if first else None,
                     "sharpe_second_half": round(second["sharpe"], 3) if second else None, **extra})
    return rows


@dataclass
class Result:
    config: Config
    ticks: int
    span_days: float
    trigger_counts: dict
    sweep: pd.DataFrame
    fills: pd.DataFrame


def run(cfg: Config) -> Result:
    grid = build_features(cfg)
    if grid.empty:
        raise ValueError(f"no book data for {cfg.symbol}")
    span = (grid["tick_ts"].max() - grid["tick_ts"].min()) * cfg.tick_s / 86_400
    counts, sweep_rows, fill_rows = {}, [], []
    for rule in RULES:
        idx = triggers(grid, rule.conditions)
        counts[rule.name] = len(idx)
        if len(idx) < cfg.min_trades:
            continue
        for hold in rule.holds_s:
            trades = simulate_fixed_hold(grid, idx, max(1, hold // cfg.tick_s), rule.side)
            if len(trades) >= cfg.min_trades:
                sweep_rows += _rows(rule, "fixed", hold, trades["gross_ret"].to_numpy(), cfg, {})
        if rule.vwap_exit:
            trades = simulate_vwap_exit(grid, idx, rule.side, max(1, cfg.vwap_failsafe_s // cfg.tick_s))
            if len(trades) >= cfg.min_trades:
                extra = {"target_hit_rate": round(float(trades["target_hit"].mean()), 3),
                         "mean_ticks_held": round(float(trades["ticks_held"].mean()), 1)}
                sweep_rows += _rows(rule, "vwap", -1, trades["gross_ret"].to_numpy(), cfg, extra)
        if cfg.fills:
            entry_p = max(1, cfg.entry_patience_s // cfg.tick_s)
            exit_p = max(1, cfg.exit_patience_s // cfg.tick_s)
            for hold in rule.holds_s:
                for k in cfg.k_bps:
                    sim = simulate_passive_fills(grid, idx, max(1, hold // cfg.tick_s), rule.side,
                                                 float(k), entry_p, exit_p, cfg.fees)
                    if len(sim) < cfg.min_trades:
                        continue
                    s = _score(sim["net_ret"].to_numpy(), cfg.min_trades)
                    a = sim.attrs
                    fill_rows.append({
                        "rule": rule.name, "hold_s": hold, "k_bps": float(k), "n": s["n"],
                        "n_triggers": a["n_triggers"],
                        "fill_rate": round(a["n_filled"] / max(1, a["n_triggers"]), 3),
                        "exit_maker_rate": round(a["n_exit_maker"] / max(1, s["n"]), 3),
                        "mean_net_bps": round(s["mean"] * 1e4, 3), "sharpe": round(s["sharpe"], 3),
                        "total_bps": round(s["total"] * 1e4, 1)})
    return Result(cfg, len(grid), span, counts, pd.DataFrame(sweep_rows), pd.DataFrame(fill_rows))


def render(result: Result) -> str:
    cfg = result.config
    out = [f"Multi-condition rules - {cfg.symbol}  tick={cfg.tick_s}s",
           f"  grid: {result.ticks:,} ticks over {result.span_days:.1f} days",
           f"  fees per side: taker {cfg.fees.taker_bps:.2f} bps, maker {cfg.fees.maker_bps:.2f} bps", "",
           "  triggers per rule:"]
    for rule in RULES:
        n = result.trigger_counts.get(rule.name, 0)
        note = "" if n >= cfg.min_trades else f"   (below {cfg.min_trades}, not simulated)"
        out.append(f"    {rule.name:<16}{n:>7}{note}")
    if not result.sweep.empty:
        view = result.sweep[result.sweep["fees"].isin(["maker", "zero"])].sort_values(
            ["rule", "exit", "fees", "hold_s"])
        cols = ["rule", "exit", "hold_s", "fees", "n", "win_rate", "mean_bps", "sharpe",
                "sharpe_first_half", "sharpe_second_half"]
        out += ["", "=" * 110, "Mid-price fills (entry at next tick, no overlap); maker and zero-fee rows",
                "=" * 110, view[cols].to_string(index=False)]
    if not result.fills.empty:
        cols = ["rule", "hold_s", "k_bps", "n", "fill_rate", "exit_maker_rate", "mean_net_bps", "sharpe", "total_bps"]
        out += ["", "=" * 110, "Passive-limit fills (taker fallback on exit)", "=" * 110,
                result.fills.sort_values(["rule", "hold_s", "k_bps"])[cols].to_string(index=False)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro multi-gate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--tick-sec", type=int, default=d.tick_s)
    parser.add_argument("--no-fills", action="store_true", help="skip the passive-fill simulation")
    parser.add_argument("--k-bps", type=float, nargs="+", default=d.k_bps)
    parser.add_argument("--entry-patience-sec", type=int, default=d.entry_patience_s)
    parser.add_argument("--exit-patience-sec", type=int, default=d.exit_patience_s)
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--maker-bps", type=float, default=d.fees.maker_bps)
    parser.add_argument("--outdir", type=Path, default=Path("results/multi_gate"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, tick_s=args.tick_sec,
                 fills=not args.no_fills, k_bps=args.k_bps, entry_patience_s=args.entry_patience_sec,
                 exit_patience_s=args.exit_patience_sec,
                 fees=FeeModel(taker_bps=args.taker_bps, maker_bps=args.maker_bps),
                 start=args.start, end=args.end)
    started = time.time()
    result = run(cfg)
    print(render(result))
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol.lower()}_t{cfg.tick_s}s"
    for name, frame in (("multi_gate", result.sweep), ("multi_gate_fills", result.fills)):
        if not frame.empty:
            path = args.outdir / f"{name}_{stem}.csv"
            frame.to_csv(path, index=False)
            print(f"\nwrote {path}")
    log.info("finished in %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
