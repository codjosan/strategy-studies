from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .file_utils import DAY_S, day_start_ms

def load_coin_grid(days_all: list[str], cache: Path, coin: str):
    n = len(days_all) * DAY_S
    px = np.full(n, np.nan, dtype="float32")
    ls = np.zeros(n, dtype="float32")
    lb = np.zeros(n, dtype="float32")
    for i, d in enumerate(days_all):
        f = cache / f"sec_{d}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            if f"px_{coin}" not in z:
                continue
            sl = slice(i * DAY_S, (i + 1) * DAY_S)
            px[sl] = z[f"px_{coin}"]
            ls[sl] = z[f"ls_{coin}"]
            lb[sl] = z[f"lb_{coin}"]
    px = pd.Series(px).ffill(limit=300).to_numpy(dtype="float32")
    return px, ls, lb

def window_sum(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(x, dtype="float64")
    out = np.empty(len(x), dtype="float32")
    out[:w] = cs[:w]
    B = 1_000_000
    for i in range(w, len(x), B):
        j = min(i + B, len(x))
        out[i:j] = cs[i:j] - cs[i - w:j - w]
    del cs
    return out

def find_events(coin: str, days_all: list[str], px: np.ndarray, ls: np.ndarray,
                lb: np.ndarray, window_s: int, qs: list[float], lookback_days: int,
                warmup_days: int, latency_s: int, horizons_s: list[int]) -> list[dict]:
    n = len(px)
    ws = window_sum(ls, window_s)
    wb = window_sum(lb, window_s)
    tot = ws + wb
    D = len(days_all)
    day_nonzero = []
    for i in range(D):
        v = tot[i * DAY_S:(i + 1) * DAY_S]
        day_nonzero.append(v[v > 0])
    max_h = max(horizons_s)
    events = []
    for q in qs:
        thr = np.full(D, np.inf)
        for i in range(D):
            prev = [p for p in day_nonzero[max(0, i - lookback_days):i] if len(p)]
            if not prev or len(prev) < warmup_days:
                continue
            pool = np.concatenate(prev)
            if len(pool) >= 100:
                thr[i] = float(np.quantile(pool, q))
        trig = np.concatenate([
            np.flatnonzero(tot[i * DAY_S:(i + 1) * DAY_S] >= thr[i]) + i * DAY_S
            for i in range(D) if np.isfinite(thr[i])] or [np.array([], dtype="int64")])
        next_ok = -1
        for t in trig:
            if t < next_ok:
                continue
            e = t + latency_s
            if e + max_h >= n or t - window_s < 0:
                continue
            p0, pw, pt = float(px[e]), float(px[t - window_s]), float(px[t])
            if not (np.isfinite(p0) and np.isfinite(pw) and p0 > 0):
                continue
            sell_dom = ws[t] >= wb[t]
            fade_dir = 1 if sell_dom else -1
            pre = (pt / pw - 1) * 1e4 if np.isfinite(pt) else np.nan
            di = t // DAY_S
            ev = {
                "coin": coin, "day": days_all[di],
                "ts": pd.Timestamp(day_start_ms(days_all[di]) // 1000 + t % DAY_S,
                                   unit="s", tz="UTC"),
                "window_s": window_s, "q": q,
                "cascade": "longs_liquidated" if sell_dom else "shorts_liquidated",
                "liq_ntl": float(tot[t]), "thr": float(thr[di]),
                "dominance": float(max(ws[t], wb[t]) / tot[t]),
                "pre_move_bps": float(-fade_dir * pre) if np.isfinite(pre) else np.nan,
            }
            for h in horizons_s:
                p1 = float(px[e + h])
                ev[f"fade_{h // 60}m"] = float(fade_dir * (p1 / p0 - 1) * 1e4) \
                    if np.isfinite(p1) else np.nan
            events.append(ev)
            next_ok = t + max_h
    del ws, wb, tot, day_nonzero
    return events
