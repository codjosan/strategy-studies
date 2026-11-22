"""Inventory of a capture directory: coverage and integrity per file family.

For each daily file family this reports the number of files and dates
covered, rows per coin, rows dropped as malformed, and files whose stream
ended early (typically a truncated gzip from an interrupted write).

Usage::

    python -m hlmicro inventory --capture-dir capture_data
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..cli import add_common_args, add_date_args, setup_logging
from ..config import default_capture_dir
from ..schemas import ALL_DAILY_PREFIXES
from .filecache import map_files
from .files import discover, file_date
from .reader import ReadStats, iter_chunks

log = logging.getLogger(__name__)

_NUMERIC = {"ts_ms": True, "px": True, "sz": True, "mid": True, "obi_ntl": True}


@dataclass
class FamilyReport:
    prefix: str
    files: int = 0
    first_date: str = ""
    last_date: str = ""
    missing_dates: list[str] = field(default_factory=list)
    rows_by_coin: Counter = field(default_factory=Counter)
    stats: ReadStats = field(default_factory=ReadStats)


def _expected_dates(first: str, last: str) -> list[str]:
    import pandas as pd
    return [d.strftime("%Y%m%d") for d in pd.date_range(first, last, freq="D")]


def _inventory_part(path: Path):
    stats = ReadStats()
    rows_by_coin: Counter = Counter()
    for chunk in iter_chunks(path, ["ts_ms", "coin", "px", "sz", "mid", "obi_ntl"],
                             numeric=list(_NUMERIC), required=["ts_ms", "coin"], stats=stats):
        cols = [c for c in ("px", "sz", "mid", "obi_ntl") if c in chunk]
        valid = chunk.dropna(subset=cols)
        stats.rows_dropped += len(chunk) - len(valid)
        rows_by_coin.update(valid["coin"].astype(str).str.upper().value_counts().to_dict())
    return rows_by_coin, stats


def inventory(directory: Path, start: str | None = None, end: str | None = None) -> list[FamilyReport]:
    reports = []
    for prefix in ALL_DAILY_PREFIXES:
        files = discover(directory, prefix, start, end)
        report = FamilyReport(prefix=prefix, files=len(files))
        if files:
            dates = [file_date(p) for p in files]
            report.first_date, report.last_date = dates[0], dates[-1]
            report.missing_dates = sorted(set(_expected_dates(dates[0], dates[-1])) - set(dates))
            for counts in map_files(_inventory_part, files, {}, None, report.stats):
                report.rows_by_coin.update(counts)
        reports.append(report)
    return reports


def render(reports: list[FamilyReport]) -> str:
    lines = [f"{'family':<18}{'files':>6}  {'dates':<19}{'rows':>12}{'dropped':>9}  coins"]
    for r in reports:
        span = f"{r.first_date}-{r.last_date}" if r.files else "-"
        total = sum(r.rows_by_coin.values())
        coins = " ".join(f"{c}:{n:,}" for c, n in sorted(r.rows_by_coin.items()))
        lines.append(f"{r.prefix:<18}{r.files:>6}  {span:<19}{total:>12,}"
                     f"{r.stats.rows_dropped:>9,}  {coins}")
        if r.missing_dates:
            lines.append(f"{'':<18}missing dates: {', '.join(r.missing_dates)}")
        if r.stats.files_truncated:
            lines.append(f"{'':<18}ended early: {', '.join(r.stats.files_truncated)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro inventory", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=default_capture_dir())
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.capture_dir.is_dir():
        parser.error(f"capture directory not found: {args.capture_dir}")
    print(render(inventory(args.capture_dir, args.start, args.end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
