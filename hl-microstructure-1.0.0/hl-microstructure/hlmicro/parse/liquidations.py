"""Hyperliquid liquidation prints (``hl_liquidations``).

Side convention: ``A`` is a liquidated long being force-sold, ``B`` a
liquidated short being force-bought.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .reader import ReadStats, iter_chunks, select_symbol
from .trades import aggressor_sign

COLUMNS = ["ts_ms", "coin", "px", "sz", "side", "sign", "notional", "tid", "liq_user"]


def load_liquidations(files: Sequence[Path], symbol: Optional[str] = None,
                      stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """All liquidation prints, optionally for one symbol, sorted by time.

    ``sign`` is -1 for forced sells (``A``) and +1 for forced buys (``B``);
    ``notional`` is ``px * sz`` in USD. ``liq_user`` is an empty string when
    the feed did not record the liquidated account.
    """
    parts = []
    for path in files:
        for chunk in iter_chunks(path, ["ts_ms", "coin", "px", "sz", "side", "tid", "liq_user"],
                                 numeric=["ts_ms", "px", "sz"],
                                 required=["ts_ms", "coin", "px", "sz", "side"],
                                 stats=stats):
            rows = select_symbol(chunk, symbol) if symbol else chunk
            if not rows.empty:
                parts.append(rows)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["ts_ms"] = out["ts_ms"].astype("int64")
    out["coin"] = out["coin"].astype(str).str.upper()
    out["sign"] = aggressor_sign(out["side"])
    out = out[out["sign"] != 0].copy()
    out["side"] = np.where(out["sign"] > 0, "B", "A")
    out["notional"] = out["px"].astype("float64") * out["sz"].astype("float64")
    if "liq_user" not in out:
        out["liq_user"] = ""
    out["liq_user"] = out["liq_user"].fillna("").astype(str)
    if "tid" not in out:
        out["tid"] = pd.NA
    return out[COLUMNS].sort_values("ts_ms", kind="stable").reset_index(drop=True)


def minute_liquidation_flow(liquidations: pd.DataFrame) -> pd.DataFrame:
    """Per-minute forced flow: ``long_liq_ntl`` (side A), ``short_liq_ntl``
    (side B) and ``liq_count``, indexed by ``minute_ts`` (``ts_ms // 60000``)."""
    if liquidations.empty:
        return pd.DataFrame(columns=["minute_ts", "long_liq_ntl", "short_liq_ntl", "liq_count"])
    frame = pd.DataFrame({
        "minute_ts": liquidations["ts_ms"].to_numpy() // 60_000,
        "long_liq_ntl": np.where(liquidations["sign"] < 0, liquidations["notional"], 0.0),
        "short_liq_ntl": np.where(liquidations["sign"] > 0, liquidations["notional"], 0.0),
    })
    return frame.groupby("minute_ts", as_index=False).agg(
        long_liq_ntl=("long_liq_ntl", "sum"),
        short_liq_ntl=("short_liq_ntl", "sum"),
        liq_count=("long_liq_ntl", "size"),
    )
