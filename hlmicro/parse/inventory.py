"""Inventory of a capture directory: coverage and integrity per file family.

For each daily file family this reports the number of files and dates
covered, rows per coin, rows dropped as malformed, and files whose stream
ended early (typically a truncated gzip from an interrupted write).

It then runs health checks, because a silently dead or truncated feed
poisons a backtest and can feed gappy data to a live strategy:

* files whose stream ended early, and files with no valid rows
* missing calendar days
* intra-day gaps longer than ``--max-gap-min`` (books and trade tapes)
* low-volume days: fewer than ``--low-volume-frac`` of the family's median
  rows per day (the newest day is skipped, as it may still be in progress)
* with ``--stale-min``: families whose newest row is older than that, the
  freshness check for a running collector

With ``--check`` the command exits with status 1 when any check fails, so it
can guard a scheduled run: ``python -m hlmicro inventory --check || alert``.

Usage::

    python -m hlmicro inventory --capture-dir capture_data
    python -m hlmicro inventory --check --stale-min 15 --per-file
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cli import add_common_args, add_date_args, setup_logging
from ..config import default_capture_dir
from ..schemas import (ALL_DAILY_PREFIXES, BOOK_REST_PREFIX, BOOK_WS_PREFIX, CEX_TRADE_PREFIXES,
                       HL_TRADES_PREFIX)
from .files import discover, file_date
from .reader import ReadStats, iter_chunks

log = logging.getLogger(__name__)

_NUMERIC = {"ts_ms": True, "px": True, "sz": True, "mid": True, "obi_ntl": True}

# Families expected to have data every few seconds; liquidations are sparse by nature.
CONTINUOUS_PREFIXES = (BOOK_REST_PREFIX, BOOK_WS_PREFIX, HL_TRADES_PREFIX, *CEX_TRADE_PREFIXES.values())


@dataclass
class FileReport:
    name: str
    date: str
    rows: int = 0
    minutes: int = 0
    first_ts: int = 0
    last_ts: int = 0
    max_gap_min: int = 0

    @property
    def span_hours(self) -> float:
        return (self.last_ts - self.first_ts) / 3_600_000 if self.rows else 0.0


@dataclass
class FamilyReport:
    prefix: str
    files: int = 0
    first_date: str = ""
    last_date: str = ""
    missing_dates: list[str] = field(default_factory=list)
    rows_by_coin: Counter = field(default_factory=Counter)
    stats: ReadStats = field(default_factory=ReadStats)
    per_file: list[FileReport] = field(default_factory=list)

    @property
    def newest_ts(self) -> int:
        return max((f.last_ts for f in self.per_file if f.rows), default=0)


def _expected_dates(first: str, last: str) -> list[str]:
    import pandas as pd
    return [d.strftime("%Y%m%d") for d in pd.date_range(first, last, freq="D")]


def _scan_file(path: Path, report: FamilyReport) -> FileReport:
    out = FileReport(name=path.name, date=file_date(path) or "")
    minutes: list[np.ndarray] = []
    for chunk in iter_chunks(path, ["ts_ms", "coin", "px", "sz", "mid", "obi_ntl"],
                             numeric=list(_NUMERIC), required=["ts_ms", "coin"], stats=report.stats):
        cols = [c for c in ("px", "sz", "mid", "obi_ntl") if c in chunk]
        valid = chunk.dropna(subset=cols)
        report.stats.rows_dropped += len(chunk) - len(valid)
        report.rows_by_coin.update(valid["coin"].astype(str).str.upper().value_counts().to_dict())
        if valid.empty:
            continue
        ts = valid["ts_ms"].to_numpy(dtype=np.int64)
        out.first_ts = int(ts.min()) if not out.rows else min(out.first_ts, int(ts.min()))
        out.last_ts = max(out.last_ts, int(ts.max()))
        out.rows += len(valid)
        minutes.append(np.unique(ts // 60_000))
    if minutes:
        observed = np.unique(np.concatenate(minutes))
        out.minutes = len(observed)
        out.max_gap_min = int(np.diff(observed).max()) if len(observed) > 1 else 0
    return out


def inventory(directory: Path, start: str | None = None, end: str | None = None) -> list[FamilyReport]:
    reports = []
    for prefix in ALL_DAILY_PREFIXES:
        files = discover(directory, prefix, start, end)
        report = FamilyReport(prefix=prefix, files=len(files))
        if files:
            dates = [file_date(p) for p in files]
            report.first_date, report.last_date = dates[0], dates[-1]
            report.missing_dates = sorted(set(_expected_dates(dates[0], dates[-1])) - set(dates))
        report.per_file = [_scan_file(path, report) for path in files]
        reports.append(report)
    return reports


def health_problems(reports: list[FamilyReport], max_gap_min: int = 10, low_volume_frac: float = 0.5,
                    stale_min: float | None = None, now_ms: int | None = None) -> list[str]:
    """Human-readable list of failed health checks; empty when the capture is healthy."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    problems = []
    for r in reports:
        if not r.files:
            continue
        problems += [f"{r.prefix}: {name} ended early (truncated)" for name in r.stats.files_truncated]
        problems += [f"{r.prefix}: {d} missing" for d in r.missing_dates]
        problems += [f"{r.prefix}: {f.name} has no valid rows" for f in r.per_file if not f.rows]
        if r.prefix not in CONTINUOUS_PREFIXES:
            continue
        problems += [f"{r.prefix}: {f.date} has a {f.max_gap_min}-minute gap" for f in r.per_file
                     if f.rows and f.max_gap_min > max_gap_min]
        complete = [f for f in r.per_file if f.rows][:-1]      # newest day may still be filling
        if complete:
            median = float(np.median([f.rows for f in complete]))
            problems += [f"{r.prefix}: {f.date} low volume ({f.rows:,} rows vs median {median:,.0f})"
                         for f in complete if f.rows < low_volume_frac * median]
        if stale_min is not None and r.newest_ts:
            age = (now_ms - r.newest_ts) / 60_000
            if age > stale_min:
                problems.append(f"{r.prefix}: stale, newest row is {age:,.0f} minutes old")
    return problems


def render(reports: list[FamilyReport], problems: list[str] | None = None, per_file: bool = False) -> str:
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
    if per_file:
        lines += ["", f"{'file':<30}{'rows':>10}{'minutes':>9}{'span h':>8}{'max gap':>9}"]
        for r in reports:
            for f in r.per_file:
                lines.append(f"{f.name:<30}{f.rows:>10,}{f.minutes:>9,}{f.span_hours:>8.1f}{f.max_gap_min:>8}m")
    if problems is not None:
        lines += ["", "health:"]
        lines += [f"  FAIL  {p}" for p in problems] or ["  ok    no truncation, missing days, gaps, low volume"
                                                         " or staleness found"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro inventory", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--max-gap-min", type=int, default=10,
                        help="largest acceptable intra-day gap in books and trade tapes")
    parser.add_argument("--low-volume-frac", type=float, default=0.5,
                        help="flag days with fewer rows than this share of the median")
    parser.add_argument("--stale-min", type=float, default=None,
                        help="flag families whose newest row is older than this many minutes")
    parser.add_argument("--per-file", action="store_true", help="list rows, minutes and gaps per file")
    parser.add_argument("--check", action="store_true", help="exit with status 1 if any health check fails")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.capture_dir.is_dir():
        parser.error(f"capture directory not found: {args.capture_dir}")
    reports = inventory(args.capture_dir, args.start, args.end)
    problems = health_problems(reports, args.max_gap_min, args.low_volume_frac, args.stale_min)
    print(render(reports, problems, per_file=args.per_file))
    return 1 if args.check and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
