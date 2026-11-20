"""Loaders for capture files: order books, trade tapes, liquidations, funding.

All loaders stream files in chunks, filter to one symbol as early as possible,
and aggregate before concatenating, so memory scales with the output size
rather than the raw file size.
"""
from .books import (
    book_files_for_resolution,
    bucket_venue_means,
    cross_venue_minutes,
    intra_minute_book_stats,
    minute_grid,
    venue_mid_series,
)
from .files import discover, discover_glob, file_date
from .funding import fetch_funding_history, funding_by_hour, load_funding_csv
from .liquidations import load_liquidations, minute_liquidation_flow
from .reader import iter_chunks
from .trades import aggressor_sign, cross_venue_flow, load_tape, tape_per_bucket, tape_prefix

__all__ = [
    "aggressor_sign", "book_files_for_resolution", "bucket_venue_means",
    "cross_venue_flow", "cross_venue_minutes", "discover", "discover_glob",
    "fetch_funding_history", "file_date", "funding_by_hour", "intra_minute_book_stats",
    "iter_chunks", "load_funding_csv", "load_liquidations", "load_tape", "minute_grid",
    "minute_liquidation_flow", "tape_per_bucket", "tape_prefix", "venue_mid_series",
]
