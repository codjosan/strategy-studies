from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .file_utils import DAY_S, day_start_ms, _chunks

def extract_day(day: str, tf: Path | None, lf: Path | None, symbols: list[str],
                out_npz: Path) -> str:
    d0 = day_start_ms(day)
    px = {c: np.full(DAY_S, np.nan, dtype="float64") for c in symbols}
    lsell = {c: np.zeros(DAY_S, dtype="float64") for c in symbols}
    lbuy = {c: np.zeros(DAY_S, dtype="float64") for c in symbols}
    n_tr, n_lq = 0, 0
    sym = set(symbols)

    if tf is not None:
        for ch in _chunks(tf, ["ts_ms", "coin", "px"], ["ts_ms", "px"], sym):
            ch = ch[ch["px"] > 0]
            sec = ((ch["ts_ms"].to_numpy(dtype="int64") - d0) // 1000)
            ok = (sec >= 0) & (sec < DAY_S)
            ch = ch[ok].assign(sec=sec[ok])
            n_tr += len(ch)
            for c, g in ch.groupby("coin", sort=False):
                g = g.sort_values("ts_ms", kind="stable")
                last = g.drop_duplicates("sec", keep="last")
                px[c][last["sec"].to_numpy()] = last["px"].to_numpy()
    if lf is not None:
        for ch in _chunks(lf, ["ts_ms", "coin", "px", "sz", "side"], ["ts_ms", "px", "sz"], sym):
            sec = ((ch["ts_ms"].to_numpy(dtype="int64") - d0) // 1000)
            ok = (sec >= 0) & (sec < DAY_S)
            ch = ch[ok].assign(sec=sec[ok], ntl=(ch["px"] * ch["sz"])[ok])
            n_lq += len(ch)
            for c, g in ch.groupby("coin", sort=False):
                s = g["sec"].to_numpy()
                w = g["ntl"].to_numpy()
                is_sell = (g["side"].to_numpy() == "A")
                lsell[c] += np.bincount(s[is_sell], weights=w[is_sell], minlength=DAY_S)
                lbuy[c] += np.bincount(s[~is_sell], weights=w[~is_sell], minlength=DAY_S)

    arrays = {}
    for c in symbols:
        arrays[f"px_{c}"] = px[c]
        arrays[f"ls_{c}"] = lsell[c].astype("float32")
        arrays[f"lb_{c}"] = lbuy[c].astype("float32")
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_npz)
    cov = np.mean([np.isfinite(px[c]).mean() for c in symbols]) * 100
    return (f"  {day}: trades={n_tr:>10,}  liqs={n_lq:>7,}  "
            f"seconds with a trade={cov:5.1f}%")
