#!/usr/bin/env python3
"""Parameter sweep of the multi-gate ``vwap_dev_buy`` rule with out-of-sample testing.

Entry rule (one tick, all true; entry at the next tick, positions never overlap):
    mid <= VWAP(window) - dev_bps, trade-flow imbalance (60 s) >= tfi_min,
    Hyperliquid book imbalance >= obi_min
Exit: after hold_s. Execution:
    taker      mid-to-mid, taker round trip (8.64 bps)
    passive k  limit k bps inside the mid at entry and exit, 1 tick patience each;
               unfilled entries are skipped, unfilled exits are taken at the mid
               with taker fee (same model as multi-gate's fill simulation)

Parameters are chosen on the TRAIN period only and judged on the TEST period,
in both directions (train on the first part / test on the second, and the
reverse). A configuration that only looks good in-sample is selection noise.

Usage:
    PYTHONPATH=. python scripts/vwap_dev_sweep.py --symbols ETH,BTC --split 20260914
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hlmicro.config import DEFAULT_FEES
from hlmicro.validate import multi_gate as mg

TICK_S = 5
GRID = {
    "vwap_window_s": [120, 300, 600, 900],
    "dev_bps": [10, 15, 20, 30, 40],
    "tfi_min": [0.0, 0.2, 0.4],
    "obi_min": [0.0, 0.2, 0.4],
    "hold_s": [300, 600, 1200, 1800, 3600],
    "exec": ["taker", 0.0, 0.5, 1.0, 2.0],
}


def features(capture_dir: Path, symbol: str, start: str, end: str) -> pd.DataFrame:
    g = mg.build_features(mg.Config(capture_dir=capture_dir, symbol=symbol, start=start, end=end))
    for w in GRID["vwap_window_s"]:
        n = w // TICK_S
        ntl = g["hl_sum_ntl"].fillna(0).rolling(n, min_periods=n).sum()
        sz = g["hl_sum_sz"].fillna(0).rolling(n, min_periods=n).sum().replace(0, np.nan)
        vwap = ntl / sz
        g[f"dev_{w}"] = ((g["mid"] - vwap) / vwap * 1e4)
    return g


def signal_ticks(g: pd.DataFrame, window: int, dev: float, tfi: float, obi: float) -> np.ndarray:
    """Ticks where all conditions first become true (as multi_gate.triggers)."""
    with np.errstate(invalid="ignore"):
        met = ((g[f"dev_{window}"].to_numpy() <= -dev) & (g[mg.TFI_COL].to_numpy() >= tfi)
               & (g["obi_hyperliquid"].to_numpy() >= obi))
    fire = np.zeros(len(met), bool)
    fire[1:] = met[1:] & ~met[:-1]
    return np.flatnonzero(fire)


def simulate(g_arr: dict, idx: np.ndarray, hold: int, execution, through_ticks: int = 0,
             tick_px: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (entry_tick_index, net_bps) per trade. ``through_ticks`` > 0 requires the
    tape to trade that many ticks through a passive limit before it counts as filled."""
    mid, lo, hi = g_arr["mid"], g_arr["lo"], g_arr["hi"]
    n = len(mid)
    taker_rt = DEFAULT_FEES.taker_round_trip_bps
    maker, taker = DEFAULT_FEES.maker_bps, DEFAULT_FEES.taker_bps
    entries, nets, busy = [], [], -1
    for i in idx:
        if i <= busy:
            continue
        if execution == "taker":
            e, x = i + 1, i + 1 + hold
            if x >= n or not (mid[e] > 0 and mid[x] > 0):
                continue
            entries.append(e)
            nets.append((mid[x] / mid[e] - 1) * 1e4 - taker_rt)
            busy = x
            continue
        k = float(execution)
        if not mid[i] > 0 or i + 1 >= n:
            continue
        post = mid[i] * (1 - k * 1e-4)
        if not (np.isfinite(lo[i + 1]) and lo[i + 1] <= post - through_ticks * tick_px):
            continue
        e = i + 1
        ref = e + hold
        if ref + 1 >= n or not mid[ref] > 0:
            continue
        post_x = mid[ref] * (1 + k * 1e-4)
        if np.isfinite(hi[ref + 1]) and hi[ref + 1] >= post_x + through_ticks * tick_px:
            gross, fee, x = (post_x / post - 1) * 1e4, 2 * maker, ref + 1
        else:
            x = ref + 1
            if not mid[x] > 0:
                continue
            gross, fee = (mid[x] / post - 1) * 1e4, maker + taker
        entries.append(e)
        nets.append(gross - fee)
        busy = x
    return np.asarray(entries, dtype=np.int64), np.asarray(nets)


def stats(x: np.ndarray) -> dict:
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {"n": n, "mean": x.mean(), "median": np.median(x), "win": (x > 0).mean(),
            "t": x.mean() / sd * np.sqrt(n) if sd and sd > 0 else np.nan,
            "mean_ex_top3": np.sort(x)[:-3].mean() if n > 3 else np.nan}


def sweep(g: pd.DataFrame, split_tick: int, through_ticks=0, tick_px=0.0) -> pd.DataFrame:
    arr = {"mid": g["mid"].to_numpy(np.float64), "lo": g["hl_px_min"].to_numpy(np.float64),
           "hi": g["hl_px_max"].to_numpy(np.float64)}
    ticks = g["tick_ts"].to_numpy()
    rows = []
    for w, dev, tfi, obi in itertools.product(GRID["vwap_window_s"], GRID["dev_bps"], GRID["tfi_min"], GRID["obi_min"]):
        idx = signal_ticks(g, w, dev, tfi, obi)
        for hold_s, ex in itertools.product(GRID["hold_s"], GRID["exec"]):
            e, net = simulate(arr, idx, hold_s // TICK_S, ex, through_ticks, tick_px)
            first = ticks[e] < split_tick if len(e) else np.zeros(0, bool)
            row = {"vwap_window_s": w, "dev_bps": dev, "tfi_min": tfi, "obi_min": obi, "hold_s": hold_s,
                   "exec": ex if ex == "taker" else f"passive {ex:g}"}
            for part, mask in (("A", first), ("B", ~first), ("all", np.ones(len(net), bool))):
                row.update({f"{part}_{k}": v for k, v in stats(net[mask]).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--symbols", default="ETH,BTC")
    ap.add_argument("--start", default="20260825")
    ap.add_argument("--end", default="20260927")
    ap.add_argument("--split", default="20260914", help="first day of period B (A = before, B = from this day)")
    ap.add_argument("--out", type=Path, default=Path("results/vwap_dev_sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split_tick = int(pd.Timestamp(args.split, tz="UTC").timestamp()) // TICK_S
    for sym in args.symbols.split(","):
        t0 = time.time()
        g = features(args.capture_dir, sym, args.start, args.end)
        res = sweep(g, split_tick)
        res.insert(0, "symbol", sym)
        res.to_csv(args.out / f"sweep_{sym}.csv", index=False)
        print(f"{sym}: {len(res)} configurations in {time.time() - t0:.0f}s -> {args.out}/sweep_{sym}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
