import gzip

import numpy as np
import pandas as pd
import pytest

from hlmicro.parse import (
    aggressor_sign, bucket_venue_means, cross_venue_minutes, discover, iter_chunks,
    load_liquidations, load_tape, minute_grid, minute_liquidation_flow, tape_per_bucket,
    venue_mid_series,
)
from hlmicro.parse.reader import ReadStats


def test_discover_prefers_plain_file_and_filters_dates(tmp_path):
    for name in ["obi_5s_20260101.csv.gz", "obi_5s_20260102.csv", "obi_5s_20260102.csv.gz",
                 "obi_5s_20260103.csv", "obi_ws_20260101.csv", "obi_5s_notadate.csv"]:
        (tmp_path / name).write_text("ts_ms\n")
    assert [p.name for p in discover(tmp_path, "obi_5s")] == [
        "obi_5s_20260101.csv.gz", "obi_5s_20260102.csv", "obi_5s_20260103.csv"]
    assert [p.name for p in discover(tmp_path, "obi_5s", start="20260102", end="20260102")] == [
        "obi_5s_20260102.csv"]
    assert [p.name for p in discover(tmp_path, "obi_5s", last_n=1)] == ["obi_5s_20260103.csv"]


def test_aggressor_sign_handles_all_venue_conventions():
    sides = pd.Series(["B", "A", "S", "buy", "sell", "bid", "ask", "?"])
    assert aggressor_sign(sides).tolist() == [1, -1, -1, 1, -1, 1, -1, 0]


DAY0 = 1767225600000  # 2026-01-01T00:00:00Z


def _read_all(path, stats):
    return pd.concat(iter_chunks(path, ["ts_ms", "coin", "mid"], numeric=["ts_ms", "mid"],
                                 required=["ts_ms", "mid"], stats=stats))


def test_reader_drops_damaged_rows_and_incomplete_last_line(tmp_path):
    t = [DAY0 + i for i in range(8)]
    path = tmp_path / "obi_5s_20260101.csv"
    path.write_text(
        "ts_ms,coin,mid\n"
        f"{t[1]},BTC,10\n"
        f"{t[2]},BTC,abc\n"                    # non-numeric value
        f"{t[3]},BTC,10,99,99\n"               # surplus fields (half-written line + next row)
        f"{t[4]},BTC\n"                        # missing field
        f"17672256{t[5]},BTC,11\n"             # cut-off timestamp merged into the next row
        f"{t[6]},BTC,12\n"
        f"{t[7]},BT")                          # write in progress
    stats = ReadStats()
    rows = _read_all(path, stats)
    assert rows["ts_ms"].tolist() == [t[1], t[6]]
    # The surplus-field line is rejected by the CSV parser before row counting.
    assert stats.rows_read == 5 and stats.rows_dropped == 3


def test_reader_salvages_truncated_gzip(tmp_path):
    lines = ["ts_ms,coin,mid"] + [f"{DAY0 + i},BTC,{100 + i}" for i in range(20_000)]
    full = gzip.compress(("\n".join(lines) + "\n").encode())
    path = tmp_path / "obi_5s_20260101.csv.gz"
    path.write_bytes(full[: len(full) // 2])
    stats = ReadStats()
    rows = _read_all(path, stats)
    assert 5_000 < len(rows) < 20_000
    offsets = (rows["ts_ms"] - DAY0).tolist()
    assert offsets == list(range(len(rows)))           # every kept row is intact and in order
    assert (rows["mid"] == np.array(offsets) + 100).all()
    assert stats.files_truncated == [path.name]


def test_bucket_means_do_not_depend_on_chunk_size(capture_dir):
    files = discover(capture_dir, "obi_5s")
    small = bucket_venue_means(files, "BTC", 60_000, chunksize=7_777)
    large = bucket_venue_means(files, "BTC", 60_000)
    pd.testing.assert_frame_equal(small, large, check_exact=False, rtol=1e-12)


def test_minute_grid_is_continuous(capture_dir):
    grid = minute_grid(discover(capture_dir, "obi_5s"), "BTC")
    assert (np.diff(grid["minute_ts"]) == 1).all()
    assert grid["n_venues"].max() == 4
    assert grid["mid"].notna().all()


def test_cross_venue_minutes_uses_hyperliquid_mid(capture_dir):
    files = discover(capture_dir, "obi_5s")
    minutes = cross_venue_minutes(files, "SOL")
    hl = venue_mid_series(files, "SOL", "hyperliquid")
    first = minutes["ts_min"].iloc[0]
    in_minute = hl[(hl["ts_ms"] >= first) & (hl["ts_ms"] < first + 60_000)]["mid"].mean()
    assert minutes["mid_hl"].iloc[0] == pytest.approx(in_minute)


def test_tape_per_bucket_counts_hyperliquid_sells(capture_dir):
    flow = tape_per_bucket(discover(capture_dir, "hl_trades"), "BTC", 60_000, "hl")
    assert flow["hl_sell_ntl"].sum() > 0
    share = flow["hl_buy_ntl"].sum() / flow["hl_sum_ntl"].sum()
    assert 0.3 < share < 0.7


def test_load_tape_canonical_sides(capture_dir):
    tape = load_tape(discover(capture_dir, "okx_trades"), "BTC")
    assert set(tape["side"]) == {"B", "A"}
    assert (np.sign(tape["sign"]) == np.where(tape["side"] == "B", 1, -1)).all()
    assert "recv_ts_ms" in tape and tape["ts_ms"].is_monotonic_increasing


def test_liquidations_and_minute_flow(capture_dir):
    liqs = load_liquidations(discover(capture_dir, "hl_liquidations"), "SOL")
    assert len(liqs) and set(liqs["coin"]) == {"SOL"}
    flow = minute_liquidation_flow(liqs)
    assert flow["liq_count"].sum() == len(liqs)
    assert flow["long_liq_ntl"].sum() == pytest.approx(liqs.loc[liqs["side"] == "A", "notional"].sum())
