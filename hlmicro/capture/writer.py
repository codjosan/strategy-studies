"""Thread-safe CSV writer that rotates files at the UTC day boundary."""
from __future__ import annotations

import csv
import gzip
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

log = logging.getLogger(__name__)


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


class DailyCSVWriter:
    """Append rows to ``<directory>/<prefix>_YYYYMMDD.csv``.

    When the UTC date changes the current file is closed and, if ``gzip_closed``
    is set, compressed to ``.csv.gz``. Compression goes through a temporary
    file and an atomic rename, so a ``.csv.gz`` on disk is always complete; the
    plain file is removed only after the rename succeeds.
    """

    def __init__(self, directory: Path, prefix: str, header: Sequence[str],
                 gzip_closed: bool = False,
                 clock: Callable[[], str] = _utc_date) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.header = list(header)
        self.gzip_closed = gzip_closed
        self._clock = clock
        self._date: str | None = None
        self._fh = None
        self._csv = None
        self._lock = threading.Lock()

    def path_for(self, date: str) -> Path:
        return self.directory / f"{self.prefix}_{date}.csv"

    def write(self, row: Sequence) -> None:
        with self._lock:
            self._roll()
            self._csv.writerow(row)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
                self._csv = None
                self._date = None     # a later write reopens the file instead of failing

    def _roll(self) -> None:
        today = self._clock()
        if today == self._date:
            return
        previous = self._date
        if self._fh is not None:
            self._fh.close()
        if self.gzip_closed and previous is not None:
            compress(self.path_for(previous))
        path = self.path_for(today)
        is_new = not path.exists() or path.stat().st_size == 0
        self.directory.mkdir(parents=True, exist_ok=True)
        needs_newline = not is_new and not _ends_with_newline(path)
        self._fh = open(path, "a", newline="", buffering=1)
        if needs_newline:
            # The previous run stopped mid-row. Finish that line so the next row
            # starts on its own line; the fragment is dropped by the reader.
            self._fh.write("\n")
        self._csv = csv.writer(self._fh)
        if is_new:
            self._csv.writerow(self.header)
        self._date = today


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def compress(path: Path) -> Path | None:
    """Gzip ``path`` to ``path.gz`` atomically, then remove ``path``."""
    path = Path(path)
    if not path.exists():
        return None
    target = path.with_name(path.name + ".gz")
    tmp = path.with_name(path.name + ".gz.tmp")
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 16)
        os.replace(tmp, target)
        path.unlink()
        log.info("compressed %s", path.name)
        return target
    except OSError as exc:
        log.error("compression of %s failed: %s", path.name, exc)
        tmp.unlink(missing_ok=True)
        return None
