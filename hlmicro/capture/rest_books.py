"""Poll top-of-book depth over REST on a fixed interval.

One row per (timestamp, coin, venue) is appended to ``obi_<interval>s_YYYYMMDD.csv``
with the columns in :data:`hlmicro.schemas.BOOK_REST_COLUMNS`.

Usage::

    python -m hlmicro capture-rest --symbols BTC,ETH,SOL --interval 5 --gzip
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from ..cli import add_common_args, csv_list, setup_logging
from ..config import default_capture_dir
from ..schemas import BOOK_REST_COLUMNS
from .book import DEFAULT_DEPTH, Level, book_metrics
from .writer import DailyCSVWriter

log = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
OKX_BOOKS_URL = "https://www.okx.com/api/v5/market/books"
BYBIT_BOOK_URL = "https://api.bybit.com/v5/market/orderbook"
USER_AGENT = "hlmicro-capture/1.0"
HTTP_TIMEOUT_S = 4.0

Book = tuple[list[Level], list[Level]]


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read())


def _post_json(url: str, payload: dict) -> object:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read())


def _levels(raw: list, depth: int) -> list[Level]:
    return [(float(level[0]), float(level[1])) for level in raw[:depth]]


def fetch_hyperliquid(symbol: str, depth: int = DEFAULT_DEPTH) -> Optional[Book]:
    data = _post_json(HL_INFO_URL, {"type": "l2Book", "coin": symbol})
    if not isinstance(data, dict) or len(data.get("levels", [])) < 2:
        return None
    bids, asks = data["levels"][0], data["levels"][1]
    return ([(float(x["px"]), float(x["sz"])) for x in bids[:depth]],
            [(float(x["px"]), float(x["sz"])) for x in asks[:depth]])


def fetch_binance(symbol: str, depth: int = DEFAULT_DEPTH) -> Optional[Book]:
    data = _get_json(f"{BINANCE_DEPTH_URL}?symbol={symbol}USDT&limit={depth}")
    if not isinstance(data, dict):
        return None
    return _levels(data.get("bids", []), depth), _levels(data.get("asks", []), depth)


def fetch_okx(symbol: str, depth: int = DEFAULT_DEPTH) -> Optional[Book]:
    data = _get_json(f"{OKX_BOOKS_URL}?instId={symbol}-USDT-SWAP&sz={depth}")
    if not isinstance(data, dict) or data.get("code") != "0" or not data.get("data"):
        return None
    book = data["data"][0]
    return _levels(book.get("bids", []), depth), _levels(book.get("asks", []), depth)


def fetch_bybit(symbol: str, depth: int = DEFAULT_DEPTH) -> Optional[Book]:
    data = _get_json(f"{BYBIT_BOOK_URL}?category=linear&symbol={symbol}USDT&limit={depth}")
    if not isinstance(data, dict) or data.get("retCode") != 0:
        return None
    result = data.get("result", {})
    return _levels(result.get("b", []), depth), _levels(result.get("a", []), depth)


FETCHERS: dict[str, Callable[[str, int], Optional[Book]]] = {
    "hyperliquid": fetch_hyperliquid,
    "binance": fetch_binance,
    "okx": fetch_okx,
    "bybit": fetch_bybit,
}


class Sampler:
    """Fetch one (symbol, venue) book and append it to the writer."""

    def __init__(self, writer: DailyCSVWriter, depth: int = DEFAULT_DEPTH) -> None:
        self.writer = writer
        self.depth = depth
        self.ok: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self._lock = threading.Lock()

    def sample(self, symbol: str, venue: str) -> None:
        try:
            book = FETCHERS[venue](symbol, self.depth)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError,
                KeyError, ValueError, TypeError, IndexError) as exc:
            log.debug("%s %s fetch failed: %s", venue, symbol, exc)
            book = None
        metrics = book_metrics(*book, depth=self.depth) if book else None
        if metrics is None:
            with self._lock:
                self.errors[venue] += 1
            return
        ts_ms = int(time.time() * 1000)
        self.writer.write([ts_ms, symbol, venue, *metrics.as_fields()])
        with self._lock:
            self.ok[venue] += 1

    def status(self) -> str:
        with self._lock:
            ok = " ".join(f"{k}={v}" for k, v in sorted(self.ok.items()))
            err = " ".join(f"{k}={v}" for k, v in sorted(self.errors.items()))
        return f"ok: {ok or '-'}  errors: {err or '-'}"


def run(symbols: list[str], venues: list[str], interval_s: float, out_dir: Path,
        gzip_closed: bool, stop: threading.Event) -> None:
    writer = DailyCSVWriter(out_dir, f"obi_{int(interval_s)}s", BOOK_REST_COLUMNS,
                            gzip_closed=gzip_closed)
    sampler = Sampler(writer)
    pool = ThreadPoolExecutor(max_workers=min(16, len(symbols) * len(venues)))
    next_tick = time.time() + interval_s
    last_status = time.time()
    n_ticks = 0
    try:
        while not stop.is_set():
            now = time.time()
            if now < next_tick:
                stop.wait(min(0.2, next_tick - now))
                continue
            for symbol in symbols:
                for venue in venues:
                    pool.submit(sampler.sample, symbol, venue)
            n_ticks += 1
            next_tick += interval_s
            if now > next_tick + 2 * interval_s:
                # Fell behind (e.g. host suspended): resynchronise instead of bursting.
                next_tick = now + interval_s
            if now - last_status >= 60:
                log.info("ticks=%d  %s", n_ticks, sampler.status())
                last_status = now
    finally:
        # Cancel queued samples but let in-flight ones (bounded by HTTP_TIMEOUT_S)
        # finish, so none of them writes to a closed file.
        pool.shutdown(wait=True, cancel_futures=True)
        writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro capture-rest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", type=csv_list(str.upper), default=["BTC", "ETH", "SOL"])
    parser.add_argument("--venues", type=csv_list(str.lower), default=list(FETCHERS))
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between snapshots (min 1)")
    parser.add_argument("--out-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--gzip", action="store_true", help="gzip each file when its day closes")
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    unknown = sorted(set(args.venues) - set(FETCHERS))
    if unknown:
        parser.error(f"unknown venues {unknown}; choose from {sorted(FETCHERS)}")
    interval = max(1.0, args.interval)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("REST book capture: symbols=%s venues=%s interval=%.0fs out=%s",
             args.symbols, args.venues, interval, args.out_dir)
    run(args.symbols, args.venues, interval, args.out_dir, args.gzip, stop)
    log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
