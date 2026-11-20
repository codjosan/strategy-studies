"""Trade tapes (``hl_trades``, ``okx_trades``, ``bybit_trades``).

Venues label the aggressor side differently: Hyperliquid uses ``B``/``A``,
other feeds may use ``B``/``S`` or ``buy``/``sell``. Everything here goes
through :func:`aggressor_sign`, so flow measures are venue-independent.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import HL_VENUE
from ..schemas import CEX_TRADE_PREFIXES, HL_TRADES_PREFIX
from .files import discover
from .reader import ReadStats, iter_chunks, select_symbol

log = logging.getLogger(__name__)

BUY_LABELS = frozenset({"B", "BUY", "BID"})
SELL_LABELS = frozenset({"A", "S", "SELL", "ASK"})


def aggressor_sign(side: pd.Series) -> np.ndarray:
    """+1 for buyer-initiated, -1 for seller-initiated, 0 for unrecognised labels."""
    labels = side.astype(str).str.strip().str.upper()
    return np.where(labels.isin(BUY_LABELS), 1,
                    np.where(labels.isin(SELL_LABELS), -1, 0)).astype(np.int8)


def load_tape(files: Sequence[Path], symbol: str, chunksize: int = 200_000,
              stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Individual trades for one symbol, sorted by exchange time.

    Columns: ``ts_ms``, ``px``, ``sz``, ``sign`` (+1/-1), ``side`` (canonical
    ``"B"``/``"A"``), and ``recv_ts_ms`` when the tape records it. Trades with
    an unrecognised side label are dropped.
    """
    parts = []
    for path in files:
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
    if not parts:
        return pd.DataFrame(columns=["ts_ms", "px", "sz", "sign", "side"])
    out = pd.concat(parts, ignore_index=True)
    out["side"] = np.where(out["sign"] > 0, "B", "A")
    return out.sort_values("ts_ms", kind="stable").reset_index(drop=True)


def tape_per_bucket(files: Sequence[Path], symbol: str, bucket_ms: int, label: str,
                    chunksize: int = 250_000,
                    stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Per-bucket trade-flow aggregates for one symbol.

    Columns (each prefixed with ``<label>_``): ``buy_ntl``, ``sell_ntl``,
    ``count``, ``sum_sz``, ``sum_ntl``, ``px_min``, ``px_max``; plus ``bucket``
    (``ts_ms // bucket_ms``). All aggregates are mergeable, so results do not
    depend on chunk boundaries.
    """
    parts = []
    for path in files:
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
        gc.collect()
    names = ["buy_ntl", "sell_ntl", "count", "sum_sz", "sum_ntl", "px_min", "px_max"]
    if not parts:
        return pd.DataFrame(columns=["bucket", *[f"{label}_{n}" for n in names]])
    out = pd.concat(parts, ignore_index=True).groupby("bucket", as_index=False).agg(
        buy_ntl=("buy_ntl", "sum"), sell_ntl=("sell_ntl", "sum"), count=("count", "sum"),
        sum_sz=("sum_sz", "sum"), sum_ntl=("sum_ntl", "sum"),
        px_min=("px_min", "min"), px_max=("px_max", "max"),
    )
    return out.rename(columns={n: f"{label}_{n}" for n in names})


def tape_prefix(venue: str) -> str:
    """File prefix of a venue's trade tape: ``hl`` / ``hyperliquid``, ``okx``, ``bybit``."""
    venue = venue.lower()
    if venue in ("hl", HL_VENUE):
        return HL_TRADES_PREFIX
    if venue in CEX_TRADE_PREFIXES:
        return CEX_TRADE_PREFIXES[venue]
    raise ValueError(f"no trade tape for venue {venue!r}")


def cross_venue_flow(directory: Path, symbol: str, bucket_ms: int,
                     venues: Sequence[str] = ("hl", "okx", "bybit"),
                     start: Optional[str] = None, end: Optional[str] = None,
                     stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Taker-flow imbalance per bucket on each venue and across venues.

    Per venue, ``tfi_<venue> = (buy - sell) / (buy + sell)`` by notional (NaN
    in buckets without recognised trades) and ``ntl_<venue>`` the traded
    notional. ``xv_tfi`` is the mean over venues with a value, ``n_venues``
    their number and ``total_ntl`` the summed notional. Only buckets with at
    least one trade on some venue are returned, sorted by ``bucket``.
    """
    frames = []
    for venue in venues:
        files = discover(directory, tape_prefix(venue), start, end)
        tape = tape_per_bucket(files, symbol, bucket_ms, venue, stats=stats)
        if tape.empty:
            continue
        buy, sell = tape[f"{venue}_buy_ntl"], tape[f"{venue}_sell_ntl"]
        total = buy + sell
        frames.append(pd.DataFrame({
            "bucket": tape["bucket"].astype("int64"),
            f"tfi_{venue}": ((buy - sell) / total.where(total > 0)).to_numpy(),
            f"ntl_{venue}": total.to_numpy(),
        }).set_index("bucket"))
    if not frames:
        return pd.DataFrame(columns=["bucket", "xv_tfi", "n_venues", "total_ntl"])
    merged = pd.concat(frames, axis=1).sort_index()
    tfi = merged[[c for c in merged if c.startswith("tfi_")]]
    merged["xv_tfi"] = tfi.mean(axis=1)
    merged["n_venues"] = tfi.notna().sum(axis=1)
    merged["total_ntl"] = merged[[c for c in merged if c.startswith("ntl_")]].sum(axis=1)
    return merged.reset_index().rename(columns={"index": "bucket"})
