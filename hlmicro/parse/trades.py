"""Trade tapes (``hl_trades``, ``okx_trades``, ``bybit_trades``).

Venues label the aggressor side differently: Hyperliquid uses ``B``/``A``,
other feeds may use ``B``/``S`` or ``buy``/``sell``. Everything here goes
through :func:`aggressor_sign`, so flow measures are venue-independent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .filecache import map_files
from .reader import ReadStats, iter_chunks, select_symbol

log = logging.getLogger(__name__)

BUY_LABELS = frozenset({"B", "BUY", "BID"})
SELL_LABELS = frozenset({"A", "S", "SELL", "ASK"})


def aggressor_sign(side: pd.Series) -> np.ndarray:
    """+1 for buyer-initiated, -1 for seller-initiated, 0 for unrecognised labels."""
    labels = side.astype(str).str.strip().str.upper()
    return np.where(labels.isin(BUY_LABELS), 1,
                    np.where(labels.isin(SELL_LABELS), -1, 0)).astype(np.int8)


def _tape_part(path: Path, symbol: str, chunksize: int = 200_000):
    stats = ReadStats()
    parts = []
    for chunk in iter_chunks(path, ["ts_ms", "recv_ts_ms", "coin", "px", "sz", "side"],
                             numeric=["ts_ms", "recv_ts_ms", "px", "sz"],
                             required=["ts_ms", "px", "sz", "side"],
                             chunksize=chunksize, stats=stats):
        rows = select_symbol(chunk, symbol)
        if rows.empty:
            continue
        sign = aggressor_sign(rows["side"])
        keep = sign != 0
        if not keep.all():
            log.debug("%s: %d trades with unrecognised side", path.name, int((~keep).sum()))
        part = pd.DataFrame({
            "ts_ms": rows["ts_ms"].astype("int64").to_numpy()[keep],
            "px": rows["px"].astype("float64").to_numpy()[keep],
            "sz": rows["sz"].astype("float64").to_numpy()[keep],
            "sign": sign[keep],
        })
        if "recv_ts_ms" in rows:
            part["recv_ts_ms"] = rows["recv_ts_ms"].to_numpy()[keep]
        parts.append(part)
    return (pd.concat(parts, ignore_index=True) if parts else None), stats


def load_tape(files: Sequence[Path], symbol: str, chunksize: int = 200_000,
              stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Individual trades for one symbol, sorted by exchange time.

    Columns: ``ts_ms``, ``px``, ``sz``, ``sign`` (+1/-1), ``side`` (canonical
    ``"B"``/``"A"``), and ``recv_ts_ms`` when the tape records it. Trades with
    an unrecognised side label are dropped.
    """
    parts = [p for p in map_files(_tape_part, files, {"symbol": symbol}, {"chunksize": chunksize}, stats)
             if p is not None]
    if not parts:
        return pd.DataFrame(columns=["ts_ms", "px", "sz", "sign", "side"])
    out = pd.concat(parts, ignore_index=True)
    out["side"] = np.where(out["sign"] > 0, "B", "A")
    return out.sort_values("ts_ms", kind="stable").reset_index(drop=True)


_BUCKET_NAMES = ["buy_ntl", "sell_ntl", "count", "sum_sz", "sum_ntl", "px_min", "px_max"]
_BUCKET_AGG = dict(buy_ntl=("buy_ntl", "sum"), sell_ntl=("sell_ntl", "sum"), count=("count", "sum"),
                   sum_sz=("sum_sz", "sum"), sum_ntl=("sum_ntl", "sum"),
                   px_min=("px_min", "min"), px_max=("px_max", "max"))


def _tape_bucket_part(path: Path, symbol: str, bucket_ms: int, chunksize: int = 250_000):
    stats = ReadStats()
    parts = []
    for chunk in iter_chunks(path, ["ts_ms", "coin", "px", "sz", "side"],
                             numeric=["ts_ms", "px", "sz"], required=["ts_ms", "px", "sz"],
                             chunksize=chunksize, stats=stats):
        rows = select_symbol(chunk, symbol)
        if rows.empty:
            continue
        px = rows["px"].astype("float64")
        sz = rows["sz"].astype("float64")
        notional = px * sz
        sign = aggressor_sign(rows["side"]) if "side" in rows else np.zeros(len(rows), np.int8)
        frame = pd.DataFrame({
            "bucket": rows["ts_ms"].astype("int64") // bucket_ms,
            "buy_ntl": np.where(sign > 0, notional, 0.0),
            "sell_ntl": np.where(sign < 0, notional, 0.0),
            "sz": sz, "notional": notional, "px": px,
        })
        parts.append(frame.groupby("bucket", as_index=False).agg(
            buy_ntl=("buy_ntl", "sum"), sell_ntl=("sell_ntl", "sum"),
            count=("sz", "size"), sum_sz=("sz", "sum"), sum_ntl=("notional", "sum"),
            px_min=("px", "min"), px_max=("px", "max"),
        ))
    if not parts:
        return None, stats
    return pd.concat(parts, ignore_index=True).groupby("bucket", as_index=False).agg(**_BUCKET_AGG), stats


def tape_per_bucket(files: Sequence[Path], symbol: str, bucket_ms: int, label: str,
                    chunksize: int = 250_000,
                    stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Per-bucket trade-flow aggregates for one symbol.

    Columns (each prefixed with ``<label>_``): ``buy_ntl``, ``sell_ntl``,
    ``count``, ``sum_sz``, ``sum_ntl``, ``px_min``, ``px_max``; plus ``bucket``
    (``ts_ms // bucket_ms``). All aggregates are mergeable, so results do not
    depend on chunk boundaries.
    """
    parts = [p for p in map_files(_tape_bucket_part, files, {"symbol": symbol, "bucket_ms": bucket_ms},
                                  {"chunksize": chunksize}, stats) if p is not None]
    if not parts:
        return pd.DataFrame(columns=["bucket", *[f"{label}_{n}" for n in _BUCKET_NAMES]])
    out = pd.concat(parts, ignore_index=True).groupby("bucket", as_index=False).agg(**_BUCKET_AGG)
    return out.rename(columns={n: f"{label}_{n}" for n in _BUCKET_NAMES})
