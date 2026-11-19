import gzip

import pytest

from hlmicro.capture.book import book_metrics
from hlmicro.capture.writer import DailyCSVWriter, compress


def test_book_metrics_values():
    m = book_metrics([(99.0, 2.0), (98.0, 1.0)], [(101.0, 1.0), (102.0, 1.0)])
    assert m.mid == 100.0
    assert m.bid_qty == 3.0 and m.ask_qty == 2.0
    assert m.obi_qty == pytest.approx(0.2)
    bid_ntl, ask_ntl = 99 * 2 + 98, 101 + 102
    assert m.obi_ntl == pytest.approx((bid_ntl - ask_ntl) / (bid_ntl + ask_ntl))
    assert m.spread_bps == pytest.approx(200.0)


def test_book_metrics_uses_requested_depth():
    bids = [(100 - i, 1.0) for i in range(10)]
    asks = [(101 + i, 1.0) for i in range(10)]
    assert book_metrics(bids, asks, depth=5).bid_qty == 5.0


@pytest.mark.parametrize("bids,asks", [([], [(1.0, 1.0)]), ([(0.0, 1.0)], [(1.0, 1.0)])])
def test_book_metrics_rejects_invalid_books(bids, asks):
    assert book_metrics(bids, asks) is None


def test_writer_rotates_and_compresses(tmp_path):
    day = {"value": "20260101"}
    writer = DailyCSVWriter(tmp_path, "obi_5s", ["a", "b"], gzip_closed=True, clock=lambda: day["value"])
    writer.write([1, 2])
    day["value"] = "20260102"
    writer.write([3, 4])
    writer.close()
    assert not (tmp_path / "obi_5s_20260101.csv").exists()
    with gzip.open(tmp_path / "obi_5s_20260101.csv.gz", "rt") as fh:
        assert fh.read().splitlines() == ["a,b", "1,2"]
    assert (tmp_path / "obi_5s_20260102.csv").read_text().splitlines() == ["a,b", "3,4"]


def test_writer_appends_without_repeating_header(tmp_path):
    for _ in range(2):
        writer = DailyCSVWriter(tmp_path, "x", ["a"], clock=lambda: "20260101")
        writer.write([1])
        writer.close()
    assert (tmp_path / "x_20260101.csv").read_text().splitlines() == ["a", "1", "1"]


def test_compress_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "x_20260101.csv"
    path.write_text("a\n1\n")
    assert compress(path) == tmp_path / "x_20260101.csv.gz"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x_20260101.csv.gz"]


class _ListWriter:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(list(row))


def test_rest_sampler_writes_schema_row(monkeypatch):
    from hlmicro.capture import rest_books
    from hlmicro.schemas import BOOK_REST_COLUMNS

    monkeypatch.setitem(rest_books.FETCHERS, "okx",
                        lambda symbol, depth: ([(99.0, 1.0)], [(101.0, 3.0)]))
    monkeypatch.setitem(rest_books.FETCHERS, "bybit", lambda symbol, depth: None)
    writer = _ListWriter()
    sampler = rest_books.Sampler(writer)
    sampler.sample("BTC", "okx")
    sampler.sample("BTC", "bybit")
    assert len(writer.rows) == 1 and len(writer.rows[0]) == len(BOOK_REST_COLUMNS)
    assert writer.rows[0][1:5] == ["BTC", "okx", "100.000000", "99.000000"]
    assert sampler.ok["okx"] == 1 and sampler.errors["bybit"] == 1


def test_ws_streams_parse_exchange_messages():
    import threading
    from collections import Counter

    from hlmicro.capture.ws_books import HyperliquidBookStream, OkxBookStream
    from hlmicro.schemas import BOOK_WS_COLUMNS

    writer, counts = _ListWriter(), Counter()
    args = (["BTC"], writer, threading.Event(), counts, threading.Lock())
    HyperliquidBookStream(*args).handle(
        {"channel": "l2Book", "data": {"coin": "BTC", "time": 1000, "levels": [
            [{"px": "99", "sz": "2", "n": 1}], [{"px": "101", "sz": "2", "n": 1}]]}}, recv_ts_ms=1100)
    OkxBookStream(*args).handle(
        {"arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"},
         "data": [{"instId": "BTC-USDT-SWAP", "ts": "2000", "bids": [["99.5", "1", "0", "1"]],
                   "asks": [["100.5", "1", "0", "1"]]}]}, recv_ts_ms=2100)
    assert [r[:4] for r in writer.rows] == [[1000, 1100, "BTC", "hyperliquid"], [2000, 2100, "BTC", "okx"]]
    assert all(len(r) == len(BOOK_WS_COLUMNS) for r in writer.rows)
    assert counts == {"hyperliquid": 1, "okx": 1}
