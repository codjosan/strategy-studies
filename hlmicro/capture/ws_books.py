"""Record every top-of-book update pushed by Hyperliquid (l2Book) and OKX (books5).

Rows go to ``obi_ws_YYYYMMDD.csv`` with the columns in
:data:`hlmicro.schemas.BOOK_WS_COLUMNS`; ``ts_ms`` is the exchange timestamp
and ``recv_ts_ms`` the local receive time.

Usage::

    python -m hlmicro capture-ws --symbols BTC,ETH,SOL --gzip

Requires the ``websocket-client`` package.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from collections import Counter
from pathlib import Path

from ..cli import add_common_args, csv_list, setup_logging
from ..config import default_capture_dir
from ..schemas import BOOK_WS_COLUMNS, BOOK_WS_PREFIX
from .book import DEFAULT_DEPTH, Level, book_metrics
from .writer import DailyCSVWriter

log = logging.getLogger(__name__)

HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
MAX_BACKOFF_S = 60.0


class BookStream(threading.Thread):
    """Base class: one websocket connection with reconnect and exponential backoff."""

    venue = ""
    url = ""
    ping_interval_s = 30

    def __init__(self, symbols: list[str], writer: DailyCSVWriter, stop: threading.Event,
                 counts: Counter, counts_lock: threading.Lock, depth: int = DEFAULT_DEPTH) -> None:
        super().__init__(daemon=True, name=f"{self.venue}-book")
        self.symbols = [s.upper() for s in symbols]
        self.writer = writer
        self.stop_event = stop
        self.counts = counts
        self.counts_lock = counts_lock
        self.depth = depth
        self._ws = None

    # Subclasses implement these two.
    def subscribe(self, ws) -> None:
        raise NotImplementedError

    def handle(self, message: dict, recv_ts_ms: int) -> None:
        raise NotImplementedError

    def emit(self, ts_ms: int, recv_ts_ms: int, coin: str,
             bids: list[Level], asks: list[Level]) -> None:
        metrics = book_metrics(bids, asks, depth=self.depth)
        if metrics is None:
            return
        self.writer.write([ts_ms, recv_ts_ms, coin, self.venue, *metrics.as_fields()])
        with self.counts_lock:
            self.counts[self.venue] += 1

    def _on_message(self, ws, raw: str) -> None:
        recv_ts_ms = int(time.time() * 1000)
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        try:
            self.handle(message, recv_ts_ms)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            log.debug("%s: unparseable message: %s", self.venue, exc)

    def run(self) -> None:
        import websocket  # imported lazily so the package works without it

        backoff = 2.0
        while not self.stop_event.is_set():
            opened = threading.Event()

            def on_open(ws) -> None:
                opened.set()
                self.subscribe(ws)

            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open=on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, err: log.warning("%s: %s", self.venue, err),
                    on_close=lambda ws, code, msg: log.info("%s closed (%s)", self.venue, code),
                )
                self._ws.run_forever(ping_interval=self.ping_interval_s, ping_timeout=10)
            except Exception as exc:  # network stack errors are not worth crashing for
                log.warning("%s: connection error: %s", self.venue, exc)
            if self.stop_event.is_set():
                break
            if opened.is_set():
                # The connection worked and later dropped: reconnect quickly
                # instead of carrying the backoff from earlier failures.
                backoff = 2.0
            delay = min(MAX_BACKOFF_S, backoff)
            log.info("%s: reconnecting in %.0fs", self.venue, delay)
            self.stop_event.wait(delay)
            backoff = min(MAX_BACKOFF_S, backoff * 1.8)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass


class HyperliquidBookStream(BookStream):
    venue = "hyperliquid"
    url = HL_WS_URL
    ping_interval_s = 30

    def subscribe(self, ws) -> None:
        for symbol in self.symbols:
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": "l2Book", "coin": symbol}}))
        log.info("hyperliquid: subscribed l2Book %s", self.symbols)

    def handle(self, message: dict, recv_ts_ms: int) -> None:
        if message.get("channel") != "l2Book":
            return
        data = message.get("data") or {}
        levels = data.get("levels", [])
        if len(levels) < 2:
            return
        bids = [(float(x["px"]), float(x["sz"])) for x in levels[0][:self.depth]]
        asks = [(float(x["px"]), float(x["sz"])) for x in levels[1][:self.depth]]
        self.emit(int(data.get("time", recv_ts_ms)), recv_ts_ms, data.get("coin", ""), bids, asks)


class OkxBookStream(BookStream):
    venue = "okx"
    url = OKX_WS_URL
    ping_interval_s = 25

    def subscribe(self, ws) -> None:
        args = [{"channel": "books5", "instId": f"{s}-USDT-SWAP"} for s in self.symbols]
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        log.info("okx: subscribed books5 %s", self.symbols)

    def handle(self, message: dict, recv_ts_ms: int) -> None:
        if message.get("arg", {}).get("channel") != "books5":
            return
        for book in message.get("data", []):
            coin = book.get("instId", "").split("-")[0]
            # OKX level format: [price, size, deprecated, order_count]
            bids = [(float(b[0]), float(b[1])) for b in book.get("bids", [])[:self.depth]]
            asks = [(float(a[0]), float(a[1])) for a in book.get("asks", [])[:self.depth]]
            self.emit(int(book.get("ts", recv_ts_ms)), recv_ts_ms, coin, bids, asks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro capture-ws", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", type=csv_list(str.upper), default=["BTC", "ETH", "SOL"])
    parser.add_argument("--out-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--gzip", action="store_true", help="gzip each file when its day closes")
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        import websocket  # noqa: F401
    except ImportError:
        parser.error("websocket capture needs the 'websocket-client' package")

    writer = DailyCSVWriter(args.out_dir, BOOK_WS_PREFIX, BOOK_WS_COLUMNS, gzip_closed=args.gzip)
    stop = threading.Event()
    counts: Counter[str] = Counter()
    lock = threading.Lock()
    streams = [cls(args.symbols, writer, stop, counts, lock)
               for cls in (HyperliquidBookStream, OkxBookStream)]

    def shutdown(*_):
        stop.set()
        for stream in streams:
            stream.close()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    log.info("websocket book capture: symbols=%s out=%s", args.symbols, args.out_dir)
    for stream in streams:
        stream.start()

    previous: Counter[str] = Counter()
    while not stop.is_set():
        stop.wait(60)
        with lock:
            current = Counter(counts)
        rates = " ".join(f"{k}={current[k]} (+{current[k] - previous[k]}/min)"
                         for k in sorted(current))
        log.info("updates: %s", rates or "none")
        previous = current

    for stream in streams:
        stream.join(timeout=5)
    writer.close()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
