"""Chunked, fault-tolerant CSV reading shared by every loader."""
from __future__ import annotations

import gzip
import io
import logging
import re
import warnings
import zlib
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_CHUNKSIZE = 500_000
TEXT_COLUMNS = ("coin", "venue", "side")
_BLOCK = 1 << 16


class ReadStats:
    """Row accounting for one or more :func:`iter_chunks` calls.

    ``rows_read`` counts rows the CSV parser accepted; lines it rejects
    outright (surplus fields) are not counted.
    """

    def __init__(self) -> None:
        self.rows_read = 0
        self.rows_dropped = 0
        self.files_truncated: list[str] = []

    def __repr__(self) -> str:
        return (f"ReadStats(read={self.rows_read}, dropped={self.rows_dropped}, "
                f"truncated={self.files_truncated})")


class _CompleteLinesReader(io.RawIOBase):
    """Byte stream of a capture file that only ever releases complete lines.

    A gzip stream that ends abruptly (an interrupted write, a full disk) is
    treated as end of file instead of an error, and the incomplete final line
    of any file is withheld, so a damaged file yields every complete row
    before the damage.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._src = gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")
        self._ready = b""
        self._tail = b""
        self._done = False
        self.truncated = False
        self.partial_line_dropped = False

    def readable(self) -> bool:
        return True

    def _fill(self) -> None:
        while not self._ready and not self._done:
            try:
                block = self._src.read(_BLOCK)
            except (EOFError, OSError, zlib.error):
                self.truncated = True
                self._finish(keep_tail=False)
                return
            if not block:
                # Normal end of stream. A final line without a newline is a
                # write in progress or a cut-off file: withhold it.
                self._finish(keep_tail=False)
                return
            data = self._tail + block
            cut = data.rfind(b"\n") + 1
            self._ready, self._tail = data[:cut], data[cut:]

    def _finish(self, keep_tail: bool) -> None:
        if self._tail.strip():
            self.partial_line_dropped = not keep_tail
            if keep_tail:
                self._ready += self._tail
        self._tail = b""
        self._done = True

    def readinto(self, buffer) -> int:
        self._fill()
        n = min(len(buffer), len(self._ready))
        buffer[:n] = self._ready[:n]
        self._ready = self._ready[n:]
        return n

    def close(self) -> None:
        self._src.close()
        super().close()


def read_header(path: Path) -> list[str]:
    with _CompleteLinesReader(Path(path)) as raw:
        return list(pd.read_csv(io.BufferedReader(raw), nrows=0).columns)


def _plausible_ts_range(path: Path) -> Optional[tuple[int, int]]:
    """Timestamps a daily file can legitimately contain: its UTC date, one day either side."""
    match = re.search(r"_(\d{8})\.csv(?:\.gz)?$", path.name)
    if not match:
        return None
    day = pd.Timestamp(match.group(1), tz="UTC")
    return (int((day - pd.Timedelta(days=1)).timestamp() * 1000),
            int((day + pd.Timedelta(days=2)).timestamp() * 1000))


def iter_chunks(path: str | Path, columns: Sequence[str], *,
                numeric: Sequence[str] = (), required: Sequence[str] = (),
                chunksize: int = DEFAULT_CHUNKSIZE,
                stats: Optional[ReadStats] = None) -> Iterator[pd.DataFrame]:
    """Yield cleaned chunks of ``columns`` from one CSV (plain or gzip).

    * Columns missing from the header are left out; callers decide whether a
      missing column matters.
    * Rows with more fields than the header are dropped. These typically come
      from a collector restarting and appending to a half-written line.
    * An incomplete final line is ignored, and a truncated gzip stream ends
      the file after its last complete row (logged, and recorded in ``stats``).
    * ``numeric`` columns are coerced; unparseable values become NaN.
    * Rows with NaN in any ``required`` column are dropped, as are ``ts_ms``
      values outside the file's date (+/- one day).
    """
    path = Path(path)
    try:
        header = read_header(path)
    except Exception as exc:
        log.warning("skipping %s: unreadable header (%s)", path.name, exc)
        return
    selected = [c for c in columns if c in header]
    if not selected:
        return
    dtype = {c: "object" for c in selected if c in TEXT_COLUMNS}
    ts_range = _plausible_ts_range(path) if "ts_ms" in selected else None
    present_numeric = [c for c in dict.fromkeys([*numeric, *(["ts_ms"] if ts_range else [])])
                       if c in selected]
    present_required = [c for c in required if c in selected]

    skip = [1] if _first_row_has_surplus_fields(path, len(header)) else None
    raw = _CompleteLinesReader(path)
    try:
        # Whole rows are parsed (no usecols) because pandas only rejects rows
        # with surplus fields when it parses every column. The first data row is
        # the exception: pandas would take its extra field as an index column and
        # shift every row, so it is checked separately and skipped.
        reader = pd.read_csv(io.BufferedReader(raw, buffer_size=_BLOCK), dtype=dtype,
                             chunksize=chunksize, on_bad_lines="skip", skiprows=skip)
        chunks = iter(reader)
        while True:
            try:
                with warnings.catch_warnings():
                    # Mixed-type columns from damaged rows are coerced below.
                    warnings.simplefilter("ignore", pd.errors.DtypeWarning)
                    chunk = next(chunks)
            except StopIteration:
                break
            except Exception as exc:
                log.warning("%s: read stopped early (%s)", path.name, exc)
                if stats is not None:
                    stats.files_truncated.append(path.name)
                break
            yield _clean(chunk[selected], present_numeric, present_required, ts_range, stats)
    finally:
        if raw.truncated:
            log.warning("%s: compressed stream is truncated; rows up to the break were kept", path.name)
            if stats is not None and path.name not in stats.files_truncated:
                stats.files_truncated.append(path.name)
        raw.close()


def _first_row_has_surplus_fields(path: Path, n_fields: int) -> bool:
    with _CompleteLinesReader(path) as raw:
        stream = io.BufferedReader(raw)
        stream.readline()
        first = stream.readline()
    return first.count(b",") > n_fields - 1


def _clean(chunk: pd.DataFrame, numeric: list[str], required: list[str],
           ts_range: Optional[tuple[int, int]], stats: Optional[ReadStats]) -> pd.DataFrame:
    n_raw = len(chunk)
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(chunk[col]):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    if required:
        chunk = chunk.dropna(subset=required)
    if ts_range is not None:
        ts = chunk["ts_ms"]
        chunk = chunk[(ts >= ts_range[0]) & (ts < ts_range[1])]
    if stats is not None:
        stats.rows_read += n_raw
        stats.rows_dropped += n_raw - len(chunk)
    return chunk


def select_symbol(chunk: pd.DataFrame, symbol: str, venue: Optional[str] = None) -> pd.DataFrame:
    """Rows for ``symbol`` (case-insensitive), optionally one venue."""
    mask = _matches(chunk["coin"], symbol.upper(), str.upper)
    if venue is not None:
        mask &= _matches(chunk["venue"], venue.lower(), str.lower)
    return chunk.loc[mask]


def _matches(column: pd.Series, wanted: str, norm) -> np.ndarray:
    """``norm(str(value)) == wanted``, evaluated once per distinct value."""
    hits = [v for v in column.unique() if norm(str(v)) == wanted]
    return column.isin(hits).to_numpy()
