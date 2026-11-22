"""Order-book snapshot files (``obi_5s``, ``obi_ws``) to analysis-ready series."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import HL_VENUE
from ..schemas import BOOK_REST_PREFIX, BOOK_WS_PREFIX
from .filecache import map_files
from .files import discover, file_date
from .reader import DEFAULT_CHUNKSIZE, ReadStats, iter_chunks, select_symbol

log = logging.getLogger(__name__)

MINUTE_MS = 60_000


def _bucket_part(path: Path, symbol: str, bucket_ms: int, columns: list[str],
                 chunksize: int = DEFAULT_CHUNKSIZE):
    """(sum, count) of ``columns`` per (bucket, venue) in one file."""
    stats = ReadStats()
    parts = []
    for chunk in iter_chunks(path, ["ts_ms", "coin", "venue", *columns],
                             numeric=["ts_ms", *columns], required=["ts_ms", *columns],
                             chunksize=chunksize, stats=stats):
        rows = select_symbol(chunk, symbol)
        if "mid" in columns:
            rows = rows[rows["mid"] > 0]
        if rows.empty:
            continue
        keyed = pd.DataFrame({
            "bucket": rows["ts_ms"].astype("int64") // bucket_ms,
            "venue": rows["venue"].astype(str).str.lower(),
        })
        for col in columns:
            keyed[col] = rows[col].astype("float64")
        parts.append(keyed.groupby(["bucket", "venue"], sort=False).agg(
            **{f"{c}__sum": (c, "sum") for c in columns},
            **{f"{c}__n": (c, "count") for c in columns},
        ))
    if not parts:
        return None, stats
    return pd.concat(parts).groupby(level=["bucket", "venue"]).sum(), stats


def bucket_venue_means(files: Sequence[Path], symbol: str, bucket_ms: int,
                       columns: Sequence[str] = ("mid", "obi_ntl"),
                       chunksize: int = DEFAULT_CHUNKSIZE,
                       stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Mean of ``columns`` per (time bucket, venue) for one symbol.

    Means are accumulated as (sum, count) across chunks and files, so they are
    exact even when a bucket straddles a chunk boundary. Rows with a
    non-positive mid are discarded when ``mid`` is requested.

    Returns columns ``bucket`` (``ts_ms // bucket_ms``), ``venue``, *columns*,
    sorted by bucket then venue.
    """
    columns = list(columns)
    parts = [p for p in map_files(_bucket_part, files, {"symbol": symbol, "bucket_ms": bucket_ms,
                                                        "columns": columns},
                                  {"chunksize": chunksize}, stats) if p is not None]
    if not parts:
        return pd.DataFrame(columns=["bucket", "venue", *columns])
    # Combine the per-file (sum, count) tables on one integer key per (bucket, venue):
    # much faster than a MultiIndex groupby on millions of rows at tick resolution.
    total = pd.concat(parts).reset_index()
    venue = total["venue"].astype("category")            # categories sorted -> same order as before
    n_venues = len(venue.cat.categories)
    key = total["bucket"].to_numpy(np.int64) * n_venues + venue.cat.codes.to_numpy(np.int64)
    inv, uniq = pd.factorize(key, sort=True)          # hash-based; sorts only the distinct keys
    out = pd.DataFrame({"bucket": uniq // n_venues,
                        "venue": venue.cat.categories.to_numpy()[uniq % n_venues]})
    for col in columns:
        sums = np.bincount(inv, weights=total[f"{col}__sum"].to_numpy(), minlength=len(uniq))
        counts = np.bincount(inv, weights=total[f"{col}__n"].to_numpy(), minlength=len(uniq))
        out[col] = sums / counts
    return out

def minute_grid(files: Sequence[Path], symbol: str,
                chunksize: int = DEFAULT_CHUNKSIZE,
                stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Continuous one-minute grid of cross-venue mean mid and imbalance.

    Per minute: the mean across venues of each venue's mean ``mid`` and
    ``obi_ntl``, and the number of venues present. Minutes without data are
    included; ``mid`` is forward-filled and ``obi_ntl`` set to 0 there.

    Columns: ``minute_ts``, ``mid``, ``obi_ntl``, ``n_venues``, ``ts_ms``, ``dt_utc``.
    """
    per_venue = bucket_venue_means(files, symbol, MINUTE_MS, ("mid", "obi_ntl"),
                                   chunksize=chunksize, stats=stats)
    if per_venue.empty:
        return pd.DataFrame()
    per_minute = per_venue.groupby("bucket", as_index=False).agg(
        mid=("mid", "mean"), obi_ntl=("obi_ntl", "mean"), n_venues=("venue", "nunique"),
    ).rename(columns={"bucket": "minute_ts"})
    full = pd.DataFrame({"minute_ts": np.arange(per_minute["minute_ts"].min(),
                                                per_minute["minute_ts"].max() + 1,
                                                dtype=np.int64)})
    grid = full.merge(per_minute, on="minute_ts", how="left")
    grid["mid"] = grid["mid"].ffill()
    grid["obi_ntl"] = grid["obi_ntl"].fillna(0.0)
    grid["ts_ms"] = grid["minute_ts"] * MINUTE_MS
    grid["dt_utc"] = pd.to_datetime(grid["ts_ms"], unit="ms", utc=True)
    return grid


def cross_venue_minutes(files: Sequence[Path], symbol: str,
                        chunksize: int = DEFAULT_CHUNKSIZE,
                        stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Per-minute cross-venue mean imbalance and the Hyperliquid mid.

    Only minutes with data are returned (no gap filling). ``mid_hl`` falls back
    to the cross-venue mean mid when the files contain no Hyperliquid rows.

    Columns: ``ts_min`` (minute start, ms), ``avg_obi``, ``mid_hl``.
    """
    per_venue = bucket_venue_means(files, symbol, MINUTE_MS, ("mid", "obi_ntl"),
                                   chunksize=chunksize, stats=stats)
    if per_venue.empty:
        return pd.DataFrame(columns=["ts_min", "avg_obi", "mid_hl"])
    obi = per_venue.pivot(index="bucket", columns="venue", values="obi_ntl")
    mid = per_venue.pivot(index="bucket", columns="venue", values="mid")
    avg_obi = obi.mean(axis=1)
    mid_hl = mid[HL_VENUE] if HL_VENUE in mid.columns else mid.mean(axis=1)
    return pd.DataFrame({
        "ts_min": avg_obi.index.to_numpy(dtype=np.int64) * MINUTE_MS,
        "avg_obi": avg_obi.to_numpy(),
        "mid_hl": mid_hl.reindex(avg_obi.index).to_numpy(),
    })


def _intra_part(path: Path, symbol: str, chunksize: int = 1_000_000):
    stats = ReadStats()
    rows = [select_symbol(c, symbol)
            for c in iter_chunks(path, ["ts_ms", "coin", "venue", "obi_ntl"],
                                 numeric=["ts_ms", "obi_ntl"], required=["ts_ms", "obi_ntl"],
                                 chunksize=chunksize, stats=stats)]
    rows = [r for r in rows if not r.empty]
    if not rows:
        return None, stats
    day = pd.concat(rows, ignore_index=True)
    day = pd.DataFrame({
        "ts_ms": day["ts_ms"].astype("int64"),
        "venue": day["venue"].astype(str).str.lower(),
        "obi": day["obi_ntl"].astype("float64"),
    }).sort_values("ts_ms", kind="stable")
    day["ts_min"] = (day["ts_ms"] // MINUTE_MS) * MINUTE_MS
    per_venue = day.groupby(["ts_min", "venue"], sort=False)["obi"].agg(
        ["count", "std", "min", "max", "first", "last"]).reset_index()
    per_venue["rng"] = per_venue["max"] - per_venue["min"]
    per_venue["drift"] = per_venue["last"] - per_venue["first"]
    part = per_venue.groupby("ts_min").agg(
        n_updates=("count", "sum"),
        obi_intra_std=("std", "mean"),
        obi_range=("rng", "mean"),
        obi_drift=("drift", "mean"),
        ws_obi_mean=("last", "mean"),
    ).reset_index()
    log.debug("%s: %d minutes", path.name, len(part))
    return part, stats


def intra_minute_book_stats(files: Sequence[Path], symbol: str,
                            chunksize: int = 1_000_000,
                            stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Within-minute behaviour of the imbalance from high-frequency book updates.

    Per (minute, venue), over updates ordered by exchange time: update count,
    standard deviation, range and drift (last minus first) of ``obi_ntl``, and
    the last value. These are then combined across venues per minute.

    Each file is reduced to one symbol before aggregation, so memory is bounded
    by one day of one symbol's updates.

    Columns: ``ts_min``, ``n_updates`` (sum over venues), ``obi_intra_std``,
    ``obi_range``, ``obi_drift``, ``ws_obi_mean`` (cross-venue means; the last
    is the mean of each venue's final value in the minute).
    """
    parts = [p for p in map_files(_intra_part, files, {"symbol": symbol}, {"chunksize": chunksize}, stats)
             if p is not None]
    if not parts:
        return pd.DataFrame(columns=["ts_min", "n_updates", "obi_intra_std", "obi_range",
                                     "obi_drift", "ws_obi_mean"])
    # A minute can appear in two files when an update crosses the day boundary.
    return (pd.concat(parts, ignore_index=True)
            .groupby("ts_min", as_index=False)
            .agg(n_updates=("n_updates", "sum"), obi_intra_std=("obi_intra_std", "mean"),
                 obi_range=("obi_range", "mean"), obi_drift=("obi_drift", "mean"),
                 ws_obi_mean=("ws_obi_mean", "mean"))
            .sort_values("ts_min").reset_index(drop=True))

def _mid_part(path: Path, symbol: str, chunksize: int = 200_000):
    """``(ts_ms, venue, mid)`` of one symbol on every venue, in file order. One
    pass (and one cache entry) per file serves any venue of the symbol."""
    stats = ReadStats()
    parts = []
    for chunk in iter_chunks(path, ["ts_ms", "coin", "venue", "mid"],
                             numeric=["ts_ms", "mid"], required=["ts_ms", "mid"],
                             chunksize=chunksize, stats=stats):
        rows = select_symbol(chunk, symbol)
        if not rows.empty:
            parts.append(pd.DataFrame({"ts_ms": rows["ts_ms"].astype("int64"),
                                       "venue": rows["venue"].astype(str).str.lower().astype("category"),
                                       "mid": rows["mid"].astype("float64")}))
    if not parts:
        return None, stats
    out = pd.concat(parts, ignore_index=True)
    out["venue"] = out["venue"].astype("category")
    return out, stats


def venue_mid_series(files: Sequence[Path], symbol: str, venue: str,
                     chunksize: int = 200_000,
                     stats: Optional[ReadStats] = None) -> pd.DataFrame:
    """Raw ``(ts_ms, mid)`` observations for one symbol on one venue.

    Sorted by ``ts_ms``; ties keep file order, so the later write wins in
    as-of lookups. Non-positive mids are dropped.
    """
    venue = venue.lower()
    parts = []
    for part in map_files(_mid_part, files, {"symbol": symbol}, {"chunksize": chunksize}, stats):
        if part is not None:
            rows = part[(part["venue"] == venue).to_numpy()]
            if len(rows):
                parts.append(rows[["ts_ms", "mid"]])
    if not parts:
        return pd.DataFrame({"ts_ms": np.array([], dtype=np.int64),
                             "mid": np.array([], dtype=np.float64)})
    out = pd.concat(parts, ignore_index=True)
    out = out[out["mid"] > 0]
    return out.sort_values("ts_ms", kind="stable").reset_index(drop=True)


def book_files_for_resolution(directory: Path, tick_sec: int,
                              start: Optional[str] = None,
                              end: Optional[str] = None) -> list[Path]:
    """One book file per date: websocket files for ticks of 2 s or less, else
    REST snapshots, falling back to the other source for dates it lacks."""
    primary, fallback = ((BOOK_WS_PREFIX, BOOK_REST_PREFIX) if tick_sec <= 2
                         else (BOOK_REST_PREFIX, BOOK_WS_PREFIX))
    by_date = {file_date(p): p for p in discover(directory, primary, start, end)}
    for path in discover(directory, fallback, start, end):
        by_date.setdefault(file_date(path), path)
    return [by_date[d] for d in sorted(by_date)]
