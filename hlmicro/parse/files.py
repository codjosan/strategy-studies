"""Discovery of daily capture files named ``<prefix>_YYYYMMDD.csv[.gz]``."""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable, Optional

_DATE_RE = re.compile(r"_(\d{8})\.csv(?:\.gz)?$")


def file_date(path: str | Path) -> Optional[str]:
    """``YYYYMMDD`` from a capture file name, or ``None``."""
    match = _DATE_RE.search(Path(path).name)
    return match.group(1) if match else None


def _select(paths: Iterable[Path], start: Optional[str], end: Optional[str],
            last_n: Optional[int]) -> list[Path]:
    by_date: dict[str, Path] = {}
    for path in sorted(paths):
        date = file_date(path)
        if date is None or (start and date < start) or (end and date > end):
            continue
        current = by_date.get(date)
        # A plain file and its .gz twin coexist while rotation is in progress or
        # after an interrupted rotation; the plain file is the complete one.
        if current is None or (current.suffix == ".gz" and path.suffix != ".gz"):
            by_date[date] = path
    selected = [by_date[d] for d in sorted(by_date)]
    if last_n:
        selected = selected[-last_n:]
    return selected


def discover(directory: str | Path, prefix: str, start: Optional[str] = None,
             end: Optional[str] = None, last_n: Optional[int] = None) -> list[Path]:
    """One file per date for ``prefix`` in ``directory``, oldest first.

    ``start``/``end`` are inclusive ``YYYYMMDD`` bounds; ``last_n`` keeps only
    the most recent dates after filtering.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{8}}\.csv(?:\.gz)?$")
    candidates = (p for p in Path(directory).glob(f"{prefix}_*.csv*") if pattern.match(p.name))
    return _select(candidates, start, end, last_n)


def discover_glob(pattern: str, start: Optional[str] = None, end: Optional[str] = None,
                  last_n: Optional[int] = None) -> list[Path]:
    """Like :func:`discover`, for an explicit glob pattern."""
    return _select((Path(p) for p in glob.glob(pattern)), start, end, last_n)
