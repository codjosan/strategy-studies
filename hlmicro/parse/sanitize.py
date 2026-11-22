"""Sanitize capture files into a clean copy: order books and trade tapes.

Every daily file of the selected families is read, checked row by row, and
written under the same name to a separate output directory. The input is
never modified.

Families: order books ``obi_5s``, ``obi_ws``; trade tapes ``hl_trades``,
``okx_trades``, ``bybit_trades``; liquidations ``hl_liquidations``.

Applied to every family
-----------------------

* Windows line endings (``\\r\\n``) become ``\\n``; a truncated gzip stream is
  salvaged up to the break and an incomplete final line is dropped.
* Output has exactly the schema columns of :mod:`hlmicro.schemas`, in schema
  order. Any other input column is removed and listed in the report. This is
  how **account identifiers** (e.g. ``user_a``/``user_b`` wallet addresses in
  Hyperliquid tapes) are kept out of the clean data.
* Coin is upper-cased, venue lower-cased, surrounding spaces removed; rows
  are sorted by ``ts_ms`` (ties keep file order).

Rows **dropped** (first failing rule is counted):

=========================  ===================================================
``wrong_field_count``      more or fewer fields than the header (e.g. the
                           collector restarted onto a half-written line)
``missing_or_non_numeric`` an empty or unparseable numeric field
``bad_coin``               coin is not 1-20 letters/digits
``ts_out_of_range``        timestamp outside the file's UTC date +/- 1 day
``duplicate``              exact copy of an earlier row
=========================  ===================================================

Order books (``obi_*``), in addition:

=========================  ===================================================
``unknown_venue``          venue not in ``hlmicro.config.BOOK_VENUES``
``nonpositive_price``      mid, bid or ask <= 0
``crossed_or_locked``      bid >= ask
``mid_inconsistent``       mid != (bid + ask) / 2
``spread_inconsistent``    spread_bps != (ask - bid) / mid
``negative_depth``         a quantity or notional < 0
``imbalance_out_of_range`` obi_qty or obi_ntl outside [-1, 1]
``imbalance_inconsistent`` obi does not match its quantities/notionals
                           (tolerance allows for the CSV rounding)
``duplicate_key``          same timestamp, coin and venue as a later row
                           (the later write is kept)
=========================  ===================================================

Trade tapes and liquidations, in addition:

=========================  ===================================================
``bad_side``               side not one of B/A/S/BUY/SELL/BID/ASK; kept sides
                           are written canonically as ``B`` (buy) / ``A`` (sell)
``bad_tid``                trade id empty or malformed (digits for Hyperliquid)
``nonpositive_price``      px <= 0
``nonpositive_size``       sz <= 0
``duplicate_tid``          same coin and trade id as an earlier row. The
                           collector replays recent trades after a reconnect;
                           the first write is kept
=========================  ===================================================

Corrections specific to a family:

* Order books: OKX swap depth is converted from contracts to coins. The OKX
  books endpoint reports sizes in contracts, so ``bid/ask_qty_5`` and
  ``bid/ask_ntl_5`` are multiplied by the contract value
  (:data:`OKX_CONTRACT_VALUES`, override with ``--okx-contract-values``).
  Imbalances are ratios and do not change.
* Trade tapes: a trade whose timestamp falls on another UTC day than its file
  (written just after the file rotated at midnight) is moved to that day's
  file when that day is part of the run, and kept in place otherwise.

Market anomalies are **flagged, never removed**, because they are what
several studies look for:

* books: ``wide_spread`` (> ``--wide-spread-bps``), ``venue_deviation`` (mid
  further than ``--deviation-bps`` from the median of the venues in the same
  5-second bucket)
* tapes: ``price_jump`` (px further than ``--price-jump-bps`` from the median
  of the surrounding 100 trades of the coin, e.g. a sweep through the book),
  ``large_trade`` (notional above ``--large-trade-usd``)
* capture gaps longer than ``--gap-s`` per coin (and venue), and outages
  where the whole file recorded nothing

Outputs in ``--out``, per family: the cleaned day files,
``sanitize_report_<family>.csv`` (one row per file, counts per rule),
``sanitize_flags_<family>.csv`` and ``sanitize_gaps_<family>.csv``.
``SANITIZED.json`` records options, version and totals of every run. A
directory containing ``SANITIZED.json`` is refused as input, so nothing is
converted twice.

Usage::

    python -m hlmicro sanitize --capture-dir capture_data --out capture_data_clean
    python -m hlmicro sanitize --capture-dir capture_data --out capture_data_clean \\
        --families hl_trades --jobs 4
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import re
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..cli import add_common_args, add_date_args, setup_logging
from ..config import BOOK_VENUES, default_capture_dir
from ..schemas import (
    BOOK_REST_COLUMNS, BOOK_REST_PREFIX, BOOK_WS_COLUMNS, BOOK_WS_PREFIX, CEX_TRADE_PREFIXES,
    CEX_TRADES_COLUMNS, HL_LIQUIDATIONS_COLUMNS, HL_LIQUIDATIONS_PREFIX, HL_TRADES_COLUMNS,
    HL_TRADES_PREFIX,
)
from .filecache import jobs as n_jobs
from .files import discover, file_date
from .reader import _plausible_ts_range

log = logging.getLogger(__name__)

BOOK_FAMILIES = {BOOK_REST_PREFIX: BOOK_REST_COLUMNS, BOOK_WS_PREFIX: BOOK_WS_COLUMNS}
TAPE_FAMILIES = {HL_TRADES_PREFIX: HL_TRADES_COLUMNS, HL_LIQUIDATIONS_PREFIX: HL_LIQUIDATIONS_COLUMNS,
                 **{prefix: CEX_TRADES_COLUMNS for prefix in CEX_TRADE_PREFIXES.values()}}
FAMILIES = {**BOOK_FAMILIES, **TAPE_FAMILIES}
TAPE_VENUE = {HL_TRADES_PREFIX: "hyperliquid", HL_LIQUIDATIONS_PREFIX: "hyperliquid",
              **{prefix: venue for venue, prefix in CEX_TRADE_PREFIXES.items()}}
TEXT = ("coin", "venue", "side", "tid")
MANIFEST = "SANITIZED.json"
DAY_MS = 86_400_000

# Contract value (coins per contract) of OKX USDT-margined swaps. BTC, ETH, SOL,
# LINK and DOGE were checked against the captured data (converted OKX depth lands
# in the range of Binance/Bybit/Hyperliquid); HYPE was confirmed by the data owner.
OKX_CONTRACT_VALUES = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0, "DOGE": 1000.0, "LINK": 1.0, "HYPE": 1.0}

DROP_RULES = (
    "wrong_field_count", "missing_or_non_numeric", "bad_coin", "unknown_venue", "bad_side", "bad_tid",
    "ts_out_of_range", "nonpositive_price", "nonpositive_size", "crossed_or_locked", "mid_inconsistent",
    "spread_inconsistent", "negative_depth", "imbalance_out_of_range", "imbalance_inconsistent",
    "duplicate", "duplicate_key", "duplicate_tid",
)
_COIN_RE = r"^[A-Z0-9]{1,20}$"
BUY_LABELS = ["B", "BUY", "BID"]
SELL_LABELS = ["A", "S", "SELL", "ASK"]


@dataclass
class Options:
    okx_contract_values: dict[str, float] = field(default_factory=lambda: dict(OKX_CONTRACT_VALUES))
    convert_okx_units: bool = True
    wide_spread_bps: float = 50.0
    deviation_bps: float = 100.0
    price_jump_bps: float = 100.0
    large_trade_usd: float = 10_000_000.0
    gap_s: float = 60.0
    compression: str = "keep"          # keep | csv | gzip


@dataclass
class FileReport:
    file: str
    family: str = ""
    out_file: str = ""
    rows_in: int = 0
    rows_out: int = 0
    truncated: bool = False
    partial_last_line: bool = False
    header_repaired: bool = False
    line_endings_fixed: int = 0
    columns_removed: str = ""
    okx_rows_converted: int = 0
    okx_rows_unconverted: int = 0
    okx_coins_unconverted: str = ""
    rows_reordered: int = 0
    rows_moved_out: int = 0
    rows_moved_in: int = 0
    other_day_kept: int = 0
    wide_spread: int = 0
    venue_deviation: int = 0
    price_jump: int = 0
    large_trade: int = 0
    gaps: int = 0
    outages: int = 0
    dropped: dict = field(default_factory=dict)
    error: str = ""


FLAG_COLUMNS = ["file", "ts_ms", "coin", "venue", "flag", "value"]
GAP_COLUMNS = ["file", "coin", "venue", "start_ts_ms", "end_ts_ms", "gap_s"]


# --------------------------------------------------------------------------- reading / writing

def _read_lines(path: Path) -> tuple[bytes, bool, bool]:
    """Complete lines of a plain or gzip file.

    Returns ``(data, truncated, partial_last_line)``. A gzip stream that ends
    early is decompressed as far as it goes (zlib keeps every recoverable
    byte, unlike block-wise reading) and an incomplete final line is dropped.
    """
    raw = path.read_bytes()
    truncated = False
    if path.suffix == ".gz":
        parts, rest = [], raw
        while rest:
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                parts.append(d.decompress(rest))
            except zlib.error:
                truncated = True
                break
            if not d.eof:
                truncated = True
                break
            rest = d.unused_data             # concatenated gzip members
        data = b"".join(parts)
    else:
        data = raw
    cut = data.rfind(b"\n") + 1
    partial = bool(data[cut:].strip())
    return data[:cut], truncated, partial


def _parse(data: bytes, columns: tuple[str, ...], extra_ok: bool) -> tuple[pd.DataFrame, int, bool, list[str]]:
    """Rows as strings, schema columns in schema order.

    Returns ``(frame, n_wrong_field_count, header_repaired, removed_columns)``.
    With ``extra_ok`` the header may carry columns beyond the schema (they are
    read, then removed). A file whose first line is not a header is read with
    the schema names when it has the schema's field count (header lost).
    """
    lines = [ln for ln in data.split(b"\n") if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=list(columns)), 0, False, []
    header = [h.strip() for h in lines[0].decode("utf-8", "replace").split(",")]
    repaired = False
    if len(set(header)) == len(header) and set(columns) <= set(header) and (
            extra_ok or len(header) == len(columns)):
        body, names = lines[1:], header
    elif len(header) == len(columns) and not re.search(r"[A-Za-z_]{3,}", header[0]):
        body, names, repaired = lines, list(columns), True
    else:
        raise ValueError(f"unexpected header {header}")
    # Field counts are checked here, not by pandas: with a surplus field on the
    # first row pandas shifts or truncates columns instead of rejecting the row.
    n_sep = len(names) - 1
    good = [ln for ln in body if ln.count(b",") == n_sep]
    wrong = len(body) - len(good)
    frame = (pd.read_csv(io.BytesIO(b"\n".join(good)), header=None, names=names, dtype=str,
                         keep_default_na=False, engine="c")
             if good else pd.DataFrame(columns=names))
    removed = [c for c in names if c not in columns]
    return frame[list(columns)], wrong, repaired, removed


def _load(path: Path, columns: tuple[str, ...], extra_ok: bool, report: FileReport):
    data, report.truncated, report.partial_last_line = _read_lines(path)
    report.line_endings_fixed = data.count(b"\r\n")
    if report.line_endings_fixed:
        data = data.replace(b"\r\n", b"\n")
    s, wrong, report.header_repaired, removed = _parse(data, columns, extra_ok)
    report.columns_removed = " ".join(removed)
    report.rows_in = len(s) + wrong
    s = s.fillna("").apply(lambda col: col.str.strip())
    s["coin"] = s["coin"].str.upper()
    return s, Counter({"wrong_field_count": wrong}) if wrong else Counter()


def _target(path: Path, out_dir: Path, opts: Options) -> Path:
    gz = {"keep": path.suffix == ".gz", "csv": False, "gzip": True}[opts.compression]
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    return out_dir / (name + (".gz" if gz else ""))


def _write(s: pd.DataFrame, target: Path) -> None:
    """Write atomically (temporary file, then rename)."""
    tmp = target.with_name(target.name + ".tmp")
    text = s.to_csv(index=False, lineterminator="\n")
    if target.suffix == ".gz":
        with gzip.open(tmp, "wt", newline="") as fh:
            fh.write(text)
    else:
        with open(tmp, "w", newline="") as fh:
            fh.write(text)
    os.replace(tmp, target)


def _apply(rules: dict[str, np.ndarray], keep: np.ndarray, dropped: Counter) -> np.ndarray:
    for rule, mask in rules.items():
        hit = keep & mask
        if hit.any():
            dropped[rule] += int(hit.sum())
        keep &= ~mask
    return keep


def _gaps(name: str, ts: np.ndarray, coin: pd.Series, venue: pd.Series, gap_s: float) -> pd.DataFrame:
    """Gaps > gap_s per (coin, venue), and across the whole file (coin/venue 'ALL')."""
    rows, gap_ms = [], gap_s * 1000
    frame = pd.DataFrame({"ts": ts, "coin": coin.to_numpy(), "venue": venue.to_numpy()})
    for (c, v), t in frame.groupby(["coin", "venue"])["ts"]:
        t = np.sort(t.to_numpy())
        d = np.diff(t)
        rows += [(name, c, v, int(t[i]), int(t[i + 1]), round(d[i] / 1000, 1)) for i in np.flatnonzero(d > gap_ms)]
    t = np.sort(ts)
    d = np.diff(t)
    rows += [(name, "ALL", "ALL", int(t[i]), int(t[i + 1]), round(d[i] / 1000, 1)) for i in np.flatnonzero(d > gap_ms)]
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def _fmt(values: np.ndarray, decimals: int) -> np.ndarray:
    return np.char.mod(f"%.{decimals}f", values)


def _empty_outputs():
    return pd.DataFrame(columns=FLAG_COLUMNS), pd.DataFrame(columns=GAP_COLUMNS)


# --------------------------------------------------------------------------- order books

def _book_checks(s: pd.DataFrame, x: pd.DataFrame, path: Path) -> dict[str, np.ndarray]:
    """'Bad' mask per drop rule, on string (s) and numeric (x) frames, in rule order."""
    n = len(s)
    bad = {}
    bad["missing_or_non_numeric"] = x.isna().any(axis=1).to_numpy()
    bad["bad_coin"] = ~s["coin"].str.fullmatch(_COIN_RE).to_numpy(dtype=bool, na_value=False)
    bad["unknown_venue"] = ~s["venue"].isin(BOOK_VENUES).to_numpy()
    rng = _plausible_ts_range(path)
    ts = x["ts_ms"].to_numpy()
    bad["ts_out_of_range"] = ((ts < rng[0]) | (ts >= rng[1])) if rng else np.zeros(n, bool)
    mid, bid, ask = (x[c].to_numpy() for c in ("mid", "bid1_px", "ask1_px"))
    bad["nonpositive_price"] = (mid <= 0) | (bid <= 0) | (ask <= 0)
    bad["crossed_or_locked"] = bid >= ask
    with np.errstate(divide="ignore", invalid="ignore"):
        # prices are written with 6 decimals: allow that rounding
        bad["mid_inconsistent"] = np.abs(mid - (bid + ask) / 2) > 1e-6 + 1e-9 * np.abs(mid)
        spread = (ask - bid) / mid * 1e4
        bad["spread_inconsistent"] = np.abs(spread - x["spread_bps"].to_numpy()) > 0.006 + 0.011 / mid
        qb, qa, nb, na = (x[c].to_numpy() for c in ("bid_qty_5", "ask_qty_5", "bid_ntl_5", "ask_ntl_5"))
        bad["negative_depth"] = (qb < 0) | (qa < 0) | (nb < 0) | (na < 0)
        oq, on = x["obi_qty"].to_numpy(), x["obi_ntl"].to_numpy()
        bad["imbalance_out_of_range"] = (np.abs(oq) > 1) | (np.abs(on) > 1)
        # Imbalance is written with 4 decimals, quantities with 4 and notionals with 2;
        # the tolerance is the worst case of those roundings (d obi / d side <= 2 / total).
        eq = np.where(qb + qa > 0, (qb - qa) / (qb + qa), 0.0)
        en = np.where(nb + na > 0, (nb - na) / (nb + na), 0.0)
        tol_q = 1e-4 + np.where(qb + qa > 0, 2e-4 / (qb + qa), 0)
        tol_n = 1e-4 + np.where(nb + na > 0, 2e-2 / (nb + na), 0)
        bad["imbalance_inconsistent"] = (np.abs(eq - oq) > tol_q) | (np.abs(en - on) > tol_n)
    return {k: np.asarray(v, dtype=bool) for k, v in bad.items()}


def sanitize_book_file(path: Path, out_dir: Path, opts: Options):
    """Clean one daily order-book file. Returns (report, flags, gaps)."""
    path = Path(path)
    family = next(p for p in BOOK_FAMILIES if path.name.startswith(p + "_"))
    columns = BOOK_FAMILIES[family]
    report = FileReport(file=path.name, family=family)
    flags_empty, gaps_empty = _empty_outputs()
    try:
        s, dropped = _load(path, columns, extra_ok=False, report=report)
    except Exception as exc:                       # unreadable file: report and skip it
        report.error = f"{type(exc).__name__}: {exc}"
        log.error("%s: %s", path.name, report.error)
        return report, flags_empty, gaps_empty
    s["venue"] = s["venue"].str.lower()
    x = s[[c for c in columns if c not in TEXT]].apply(pd.to_numeric, errors="coerce")

    keep = _apply(_book_checks(s, x, path), np.ones(len(s), bool), dropped)
    s, x = s[keep], x[keep]

    dup = s.duplicated(keep="first").to_numpy()
    if dup.any():
        dropped["duplicate"] += int(dup.sum())
    s, x = s[~dup], x[~dup]
    ts_cols = ["ts_ms"] + (["recv_ts_ms"] if "recv_ts_ms" in columns else [])
    dup_key = x[ts_cols].assign(coin=s["coin"], venue=s["venue"]).duplicated(keep="last").to_numpy()
    if dup_key.any():
        dropped["duplicate_key"] += int(dup_key.sum())
    s, x = s[~dup_key], x[~dup_key]

    order = np.argsort(x["ts_ms"].to_numpy(), kind="stable")
    report.rows_reordered = int((order != np.arange(len(order))).sum())
    s, x = s.iloc[order].reset_index(drop=True), x.iloc[order].reset_index(drop=True)

    # OKX contracts -> coins
    if opts.convert_okx_units:
        okx = (s["venue"] == "okx").to_numpy()
        ct = s["coin"].map(opts.okx_contract_values).to_numpy(dtype=float)
        conv = okx & np.isfinite(ct)
        if conv.any():
            for col, dec in (("bid_qty_5", 4), ("ask_qty_5", 4), ("bid_ntl_5", 2), ("ask_ntl_5", 2)):
                vals = x.loc[conv, col].to_numpy() * ct[conv]
                x.loc[conv, col] = vals
                s.loc[conv, col] = _fmt(vals, dec)
        report.okx_rows_converted = int(conv.sum())
        unconv = okx & ~np.isfinite(ct)
        report.okx_rows_unconverted = int(unconv.sum())
        report.okx_coins_unconverted = " ".join(sorted(s.loc[unconv, "coin"].unique()))

    flags = []
    wide = x["spread_bps"].to_numpy() > opts.wide_spread_bps
    report.wide_spread = int(wide.sum())
    if wide.any():
        flags.append(pd.DataFrame({"file": path.name, "ts_ms": x.loc[wide, "ts_ms"].astype("int64"),
                                   "coin": s.loc[wide, "coin"], "venue": s.loc[wide, "venue"],
                                   "flag": "wide_spread", "value": x.loc[wide, "spread_bps"].round(2)}))
    grp = pd.DataFrame({"b": x["ts_ms"].astype("int64") // 5000, "coin": s["coin"], "mid": x["mid"]})
    med = grp.groupby(["b", "coin"])["mid"].transform("median")
    n_venues = grp.groupby(["b", "coin"])["mid"].transform("size")
    dev = ((x["mid"] / med - 1) * 1e4).to_numpy()
    off = (np.abs(dev) > opts.deviation_bps) & (n_venues.to_numpy() >= 3)
    report.venue_deviation = int(off.sum())
    if off.any():
        flags.append(pd.DataFrame({"file": path.name, "ts_ms": x.loc[off, "ts_ms"].astype("int64"),
                                   "coin": s.loc[off, "coin"], "venue": s.loc[off, "venue"],
                                   "flag": "venue_deviation", "value": np.round(dev[off], 1)}))
    gaps = _gaps(path.name, x["ts_ms"].astype("int64").to_numpy(), s["coin"], s["venue"], opts.gap_s)
    report.gaps = int((gaps["coin"] != "ALL").sum())
    report.outages = int((gaps["coin"] == "ALL").sum())

    target = _target(path, out_dir, opts)
    _write(s, target)
    report.out_file, report.rows_out, report.dropped = target.name, len(s), dict(dropped)
    flags = pd.concat(flags, ignore_index=True) if flags else flags_empty
    return report, flags, gaps


# --------------------------------------------------------------------------- trade tapes

def _tape_checks(s: pd.DataFrame, x: pd.DataFrame, path: Path, family: str) -> dict[str, np.ndarray]:
    n = len(s)
    bad = {}
    bad["missing_or_non_numeric"] = x.isna().any(axis=1).to_numpy()
    bad["bad_coin"] = ~s["coin"].str.fullmatch(_COIN_RE).to_numpy(dtype=bool, na_value=False)
    bad["bad_side"] = ~s["side"].isin(BUY_LABELS + SELL_LABELS).to_numpy()
    tid_re = r"^\d{1,20}$" if TAPE_VENUE[family] == "hyperliquid" else r"^[A-Za-z0-9_.:-]{1,64}$"
    bad["bad_tid"] = ~s["tid"].str.fullmatch(tid_re).to_numpy(dtype=bool, na_value=False)
    rng = _plausible_ts_range(path)
    ts = x["ts_ms"].to_numpy()
    bad["ts_out_of_range"] = ((ts < rng[0]) | (ts >= rng[1])) if rng else np.zeros(n, bool)
    bad["nonpositive_price"] = ~(x["px"].to_numpy() > 0)
    bad["nonpositive_size"] = ~(x["sz"].to_numpy() > 0)
    return {k: np.asarray(v, dtype=bool) for k, v in bad.items()}


def sanitize_tape_file(path: Path, out_dir: Path, opts: Options, run_days: frozenset = frozenset()):
    """Clean one daily trade-tape / liquidation file.

    Returns (report, flags, gaps, spill): ``spill`` holds rows that belong to
    another day of the run (column ``_day`` names it); they are not written here.
    """
    path = Path(path)
    family = next(p for p in TAPE_FAMILIES if path.name.startswith(p + "_"))
    columns = TAPE_FAMILIES[family]
    venue = TAPE_VENUE[family]
    report = FileReport(file=path.name, family=family)
    flags_empty, gaps_empty = _empty_outputs()
    no_spill = pd.DataFrame(columns=[*columns, "_day", "_src"])
    try:
        s, dropped = _load(path, columns, extra_ok=True, report=report)
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        log.error("%s: %s", path.name, report.error)
        return report, flags_empty, gaps_empty, no_spill
    s["side"] = s["side"].str.upper()
    num = [c for c in columns if c not in TEXT]
    x = s[num].apply(pd.to_numeric, errors="coerce")

    keep = _apply(_tape_checks(s, x, path, family), np.ones(len(s), bool), dropped)
    s, x = s[keep].copy(), x[keep]
    s["side"] = np.where(s["side"].isin(BUY_LABELS), "B", "A")

    dup = s.duplicated(keep="first").to_numpy()
    if dup.any():
        dropped["duplicate"] += int(dup.sum())
    s, x = s[~dup], x[~dup]
    dup_tid = s.duplicated(["coin", "tid"], keep="first").to_numpy()
    if dup_tid.any():
        dropped["duplicate_tid"] += int(dup_tid.sum())
    s, x = s[~dup_tid], x[~dup_tid]

    # rows of another UTC day
    day = file_date(path)
    d0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000) if day else None
    spill = no_spill
    if d0 is not None:
        ts = x["ts_ms"].to_numpy()
        other = (ts < d0) | (ts >= d0 + DAY_MS)
        if other.any():
            days = pd.to_datetime(ts[other], unit="ms", utc=True).strftime("%Y%m%d").to_numpy()
            movable = np.isin(days, list(run_days))
            idx = np.flatnonzero(other)
            move = np.zeros(len(s), bool)
            move[idx[movable]] = True
            report.other_day_kept = int((~movable).sum())
            report.rows_moved_out = int(move.sum())
            if move.any():
                spill = s[move].assign(_day=days[movable], _src=path.name)
                s, x = s[~move], x[~move]

    order = np.argsort(x["ts_ms"].to_numpy(), kind="stable")
    report.rows_reordered = int((order != np.arange(len(order))).sum())
    s, x = s.iloc[order].reset_index(drop=True), x.iloc[order].reset_index(drop=True)

    flags = []
    px = x["px"]
    med = px.groupby(s["coin"]).transform(
        lambda p: p.rolling(101, center=True, min_periods=21).median())
    dev = ((px / med - 1) * 1e4).to_numpy()
    jump = np.abs(np.nan_to_num(dev)) > opts.price_jump_bps
    report.price_jump = int(jump.sum())
    if jump.any():
        flags.append(pd.DataFrame({"file": path.name, "ts_ms": x.loc[jump, "ts_ms"].astype("int64"),
                                   "coin": s.loc[jump, "coin"], "venue": venue,
                                   "flag": "price_jump", "value": np.round(dev[jump], 1)}))
    notional = (px * x["sz"]).to_numpy()
    large = notional > opts.large_trade_usd
    report.large_trade = int(large.sum())
    if large.any():
        flags.append(pd.DataFrame({"file": path.name, "ts_ms": x.loc[large, "ts_ms"].astype("int64"),
                                   "coin": s.loc[large, "coin"], "venue": venue,
                                   "flag": "large_trade", "value": np.round(notional[large], 0)}))
    gaps = _gaps(path.name, x["ts_ms"].astype("int64").to_numpy(), s["coin"],
                 pd.Series(venue, index=s.index), opts.gap_s)
    report.gaps = int((gaps["coin"] != "ALL").sum())
    report.outages = int((gaps["coin"] == "ALL").sum())

    target = _target(path, out_dir, opts)
    _write(s, target)
    report.out_file, report.rows_out, report.dropped = target.name, len(s), dict(dropped)
    flags = pd.concat(flags, ignore_index=True) if flags else flags_empty
    return report, flags, gaps, spill


def _merge_spills(out_dir: Path, reports: dict[str, FileReport], spills: list[pd.DataFrame],
                  columns: tuple[str, ...]) -> None:
    """Append moved rows to the clean file of their own day (deduplicated, re-sorted)."""
    if not spills:
        return
    spill = pd.concat(spills, ignore_index=True)
    by_day = {file_date(r.out_file): r for r in reports.values() if r.out_file and not r.error}
    # A row whose own day could not be written goes back to the file it came from.
    home = spill["_day"].where(spill["_day"].isin(list(by_day)),
                               spill["_src"].map(lambda f: file_date(f)))
    for day, rows in spill.groupby(home):
        report = by_day[day]
        target = out_dir / report.out_file
        data, _, _ = _read_lines(target)
        current = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        merged = pd.concat([current, rows[list(columns)]], ignore_index=True)
        dup = merged.duplicated(["coin", "tid"], keep="first").to_numpy()
        if dup.any():
            report.dropped["duplicate_tid"] = report.dropped.get("duplicate_tid", 0) + int(dup.sum())
        merged = merged[~dup]
        merged = merged.iloc[np.argsort(merged["ts_ms"].astype("int64").to_numpy(), kind="stable")]
        _write(merged, target)
        report.rows_moved_in += len(rows) - int(dup.sum())
        report.rows_out = len(merged)


# --------------------------------------------------------------------------- directory

def _task(args):
    kind, path, out_dir, opts, run_days = args
    if kind == "book":
        return (*sanitize_book_file(path, out_dir, opts), None)
    return sanitize_tape_file(path, out_dir, opts, run_days)


def sanitize_dir(capture_dir: Path, out_dir: Path, opts: Options, families=tuple(FAMILIES),
                 start: str | None = None, end: str | None = None, jobs: int = 1) -> list[FileReport]:
    """Sanitize every file of ``families`` in ``capture_dir`` into ``out_dir``."""
    capture_dir, out_dir = Path(capture_dir), Path(out_dir)
    if (capture_dir / MANIFEST).exists():
        raise ValueError(f"{capture_dir} is already sanitized output ({MANIFEST} present); "
                         "sanitize the raw capture directory instead")
    if out_dir.exists() and capture_dir.resolve() == out_dir.resolve():
        raise ValueError("--out must differ from --capture-dir (the input is never modified)")
    by_family = {fam: discover(capture_dir, fam, start, end) for fam in families}
    by_family = {fam: files for fam, files in by_family.items() if files}
    if not by_family:
        raise ValueError(f"no {', '.join(families)} files in {capture_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for fam, files in by_family.items():
        kind = "book" if fam in BOOK_FAMILIES else "tape"
        run_days = frozenset(file_date(p) for p in files)
        tasks += [(kind, p, out_dir, opts, run_days) for p in files]
    log.info("sanitizing %d files (%s) from %s into %s, %d jobs", len(tasks),
             ", ".join(by_family), capture_dir, out_dir, jobs)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(_task, tasks))
    else:
        results = [_task(t) for t in tasks]

    all_reports: list[FileReport] = []
    manifest_path = out_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"tool": "hlmicro sanitize", "version": __version__})
    runs = manifest.setdefault("families", {})
    for fam in by_family:
        fam_results = [r for r in results if r[0].family == fam]
        reports = {r[0].file: r[0] for r in fam_results}
        if fam in TAPE_FAMILIES:
            _merge_spills(out_dir, reports, [r[3] for r in fam_results if len(r[3])], TAPE_FAMILIES[fam])
        for report in reports.values():
            log.info("%s: %d -> %d rows%s", report.file, report.rows_in, report.rows_out,
                     f"  ERROR {report.error}" if report.error else "")
        rows = []
        for r in reports.values():
            row = {k: v for k, v in asdict(r).items() if k != "dropped"}
            row.update({f"drop_{k}": r.dropped.get(k, 0) for k in DROP_RULES})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"sanitize_report_{fam}.csv", index=False)
        for kind, idx, cols in (("flags", 1, FLAG_COLUMNS), ("gaps", 2, GAP_COLUMNS)):
            frames = [r[idx] for r in fam_results if len(r[idx])]
            (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)).to_csv(
                out_dir / f"sanitize_{kind}_{fam}.csv", index=False)
        totals = Counter()
        for r in reports.values():
            totals.update(r.dropped)
        runs[fam] = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": __version__, "input_dir": str(capture_dir.resolve()),
            "files": len(reports), "rows_in": sum(r.rows_in for r in reports.values()),
            "rows_out": sum(r.rows_out for r in reports.values()), "dropped": dict(totals),
            "columns_removed": sorted({c for r in reports.values() for c in r.columns_removed.split()}),
            "errors": [r.file for r in reports.values() if r.error],
            "options": asdict(opts),
            **({"okx_units": "coins" if opts.convert_okx_units else "contracts (as captured)"}
               if fam in BOOK_FAMILIES else {}),
        }
        all_reports += list(reports.values())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return all_reports


def render(reports: list[FileReport], opts: Options) -> str:
    out = []
    for fam in dict.fromkeys(r.family for r in reports):
        rs = [r for r in reports if r.family == fam]
        rows_in, rows_out = sum(r.rows_in for r in rs), sum(r.rows_out for r in rs)
        moved = sum(r.rows_moved_in for r in rs)
        totals = Counter()
        for r in rs:
            totals.update(r.dropped)
        out += [f"{fam}: {len(rs)} files, {rows_in:,} rows in, {rows_out:,} rows out "
                f"({sum(totals.values()):,} dropped, {100 * sum(totals.values()) / max(1, rows_in):.4f}%)",
                "  dropped by rule:"]
        out += [f"    {k:<24}{totals[k]:>12,}" for k in DROP_RULES if totals[k]] or ["    none"]
        removed = sorted({c for r in rs for c in r.columns_removed.split()})
        crlf = sum(r.line_endings_fixed for r in rs)
        out += ["  corrected:",
                f"    rows re-sorted by ts_ms          {sum(r.rows_reordered for r in rs):>12,}"]
        if crlf:
            out.append(f"    CRLF line endings -> LF          {crlf:>12,}")
        if removed:
            out.append(f"    columns removed                  {', '.join(removed)}")
        if fam in BOOK_FAMILIES:
            out.append(f"    OKX rows contracts -> coins      {sum(r.okx_rows_converted for r in rs):>12,}")
            unconv = sum(r.okx_rows_unconverted for r in rs)
            if unconv:
                coins = sorted({c for r in rs for c in r.okx_coins_unconverted.split()})
                out.append(f"    OKX rows NOT converted           {unconv:>12,}   (no contract value for "
                           f"{', '.join(coins)}; pass --okx-contract-values COIN=VALUE)")
        else:
            out.append(f"    rows moved to their own day      {moved:>12,}")
            kept = sum(r.other_day_kept for r in rs)
            if kept:
                out.append(f"    other-day rows kept (day not in run) {kept:>8,}")
        out.append("  flagged (kept):")
        if fam in BOOK_FAMILIES:
            out += [f"    spread > {opts.wide_spread_bps:g} bps                {sum(r.wide_spread for r in rs):>12,}",
                    f"    mid > {opts.deviation_bps:g} bps from venues      {sum(r.venue_deviation for r in rs):>12,}"]
        else:
            out += [f"    px > {opts.price_jump_bps:g} bps from neighbours   {sum(r.price_jump for r in rs):>12,}",
                    f"    trades > ${opts.large_trade_usd:,.0f}          {sum(r.large_trade for r in rs):>12,}"]
        out += [f"    gaps > {opts.gap_s:g} s per coin             {sum(r.gaps for r in rs):>12,}",
                f"    outages (whole file) > {opts.gap_s:g} s     {sum(r.outages for r in rs):>12,}"]
        trunc = [r.file for r in rs if r.truncated or r.partial_last_line]
        if trunc:
            out.append(f"    files ending early (kept up to the break): {', '.join(trunc)}")
        errors = [f"{r.file} ({r.error})" for r in rs if r.error]
        if errors:
            out += ["  FILES NOT WRITTEN:"] + [f"    {e}" for e in errors]
        out.append("")
    return "\n".join(out).rstrip()


def _contract_values(text: str) -> dict[str, float]:
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        coin, _, value = item.partition("=")
        try:
            out[coin.strip().upper()] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected COIN=VALUE, got {item!r}") from None
        if out[coin.strip().upper()] <= 0:
            raise argparse.ArgumentTypeError(f"contract value must be positive: {item!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro sanitize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Options()
    parser.add_argument("--capture-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--out", type=Path, required=True, help="output directory (created)")
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help=f"comma-separated, from {', '.join(FAMILIES)} (default: all present)")
    parser.add_argument("--okx-contract-values", type=_contract_values, default={},
                        help="extra or overriding OKX contract values, e.g. XRP=100")
    parser.add_argument("--no-okx-units", action="store_true",
                        help="keep OKX depth in contracts, as captured")
    parser.add_argument("--wide-spread-bps", type=float, default=d.wide_spread_bps)
    parser.add_argument("--deviation-bps", type=float, default=d.deviation_bps)
    parser.add_argument("--price-jump-bps", type=float, default=d.price_jump_bps)
    parser.add_argument("--large-trade-usd", type=float, default=d.large_trade_usd)
    parser.add_argument("--gap-s", type=float, default=d.gap_s)
    parser.add_argument("--compression", choices=("keep", "csv", "gzip"), default=d.compression,
                        help="output files: same as input (default), plain, or gzip")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        parser.error(f"unknown families {unknown}; choose from {sorted(FAMILIES)}")
    opts = Options(okx_contract_values={**OKX_CONTRACT_VALUES, **args.okx_contract_values},
                   convert_okx_units=not args.no_okx_units, wide_spread_bps=args.wide_spread_bps,
                   deviation_bps=args.deviation_bps, price_jump_bps=args.price_jump_bps,
                   large_trade_usd=args.large_trade_usd, gap_s=args.gap_s, compression=args.compression)
    try:
        reports = sanitize_dir(args.capture_dir, args.out, opts, families, args.start, args.end,
                               n_jobs())
    except ValueError as exc:          # bad input choice; I/O errors propagate with a traceback
        parser.error(str(exc))
    print(render(reports, opts))
    print(f"\nwrote {args.out}/ (cleaned files, sanitize_report_<family>.csv, sanitize_flags_<family>.csv, "
          f"sanitize_gaps_<family>.csv, {MANIFEST})")
    return 1 if any(r.error for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
