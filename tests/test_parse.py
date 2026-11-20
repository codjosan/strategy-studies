import gzip

import numpy as np
import pandas as pd
import pytest

from hlmicro.parse import (
    aggressor_sign, bucket_venue_means, cross_venue_flow, cross_venue_minutes, discover, iter_chunks,
    load_liquidations, load_tape, minute_grid, minute_liquidation_flow, tape_per_bucket,
    venue_mid_series,
)
from hlmicro.parse.inventory import health_problems, inventory, render
from hlmicro.parse.inventory import main as inventory_main
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


def test_synthetic_identity_columns_are_omitted(capture_dir):
    for prefix, forbidden in (
        ("hl_trades", {"user_a", "user_b"}),
        ("hl_liquidations", {"liq_user"}),
    ):
        path = discover(capture_dir, prefix)[0]
        assert not forbidden.intersection(pd.read_csv(path, nrows=0).columns)


def test_load_tape_canonical_sides(capture_dir):
    tape = load_tape(discover(capture_dir, "okx_trades"), "BTC")
    assert set(tape["side"]) == {"B", "A"}
    assert (np.sign(tape["sign"]) == np.where(tape["side"] == "B", 1, -1)).all()
    assert "recv_ts_ms" in tape and tape["ts_ms"].is_monotonic_increasing


def test_liquidations_and_minute_flow(capture_dir):
    liqs = load_liquidations(discover(capture_dir, "hl_liquidations"), "SOL")
    assert len(liqs) and set(liqs["coin"]) == {"SOL"}
    assert "liq_user" not in liqs
    flow = minute_liquidation_flow(liqs)
    assert flow["liq_count"].sum() == len(liqs)
    assert flow["long_liq_ntl"].sum() == pytest.approx(liqs.loc[liqs["side"] == "A", "notional"].sum())


def test_liquidation_loader_discards_account_identifier(tmp_path):
    path = tmp_path / "hl_liquidations_20260101.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("ts_ms,coin,px,sz,side,tid,liq_user\n")
        f.write(f"{DAY0},BTC,100,2,A,123,private-field\n")
    liqs = load_liquidations([path], "BTC")
    assert liqs["notional"].tolist() == [200.0]
    assert "liq_user" not in liqs


def test_cross_venue_flow_reads_every_side_convention(tmp_path):
    (tmp_path / "hl_trades_20260101.csv").write_text(
        "ts_ms,coin,px,sz,side,tid\n"
        f"{DAY0},BTC,100,3,B,1\n{DAY0 + 1},BTC,100,1,A,2\n"           # minute 0: (300 - 100) / 400
        f"{DAY0 + 60_000},BTC,100,1,A,3\n")                          # minute 1: -1
    (tmp_path / "okx_trades_20260101.csv").write_text(
        "ts_ms,recv_ts_ms,coin,px,sz,side,tid\n"
        f"{DAY0},{DAY0},BTC,100,1,buy,1\n{DAY0 + 5},{DAY0 + 5},BTC,100,1,sell,2\n")   # minute 0: 0
    flow = cross_venue_flow(tmp_path, "BTC", 60_000, venues=("hl", "okx", "bybit"))
    first = DAY0 // 60_000
    assert flow["bucket"].tolist() == [first, first + 1]
    assert flow["tfi_hl"].tolist() == [0.5, -1.0]
    assert flow["tfi_okx"].iloc[0] == 0.0 and np.isnan(flow["tfi_okx"].iloc[1])
    assert flow["xv_tfi"].tolist() == [0.25, -1.0]
    assert flow["n_venues"].tolist() == [2, 1]
    assert flow["total_ntl"].tolist() == [600.0, 100.0]


def _health_day(directory, date: str, rows: int, gap_at=None, truncate=False):
    """A 5-second book file for ``date``; optionally a 20-minute gap or a cut-off gzip."""
    t = int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)
    lines = ["ts_ms,coin,venue,mid,obi_ntl"]
    for i in range(rows):
        t += 5_000 + (20 * 60_000 if i == gap_at else 0)
        lines.append(f"{t},SOL,hyperliquid,100.0,0.1")
    raw = gzip.compress(("\n".join(lines) + "\n").encode())
    (directory / f"obi_5s_{date}.csv.gz").write_bytes(raw[:-40] if truncate else raw)
    return t


def test_inventory_health_checks(tmp_path):
    _health_day(tmp_path, "20260628", 3000)
    _health_day(tmp_path, "20260629", 3000, gap_at=1000)
    _health_day(tmp_path, "20260630", 3000, truncate=True)
    _health_day(tmp_path, "20260702", 200)                           # 20260701 missing; low volume
    newest = _health_day(tmp_path, "20260703", 300)                   # in progress: not low volume
    reports = inventory(tmp_path)
    problems = health_problems(reports, max_gap_min=10, stale_min=15, now_ms=newest + 5 * 60_000)
    assert len(problems) == 4
    assert set(problems[:3]) == {
        "obi_5s: obi_5s_20260630.csv.gz ended early (truncated)",
        "obi_5s: 20260701 missing",
        "obi_5s: 20260629 has a 20-minute gap",
    }
    assert problems[3].startswith("obi_5s: 20260702 low volume (200 rows")   # newest day not flagged
    assert any("stale" in p for p in health_problems(reports, stale_min=15, now_ms=newest + 60 * 60_000))
    assert "FAIL" in render(reports, problems)
    assert inventory_main(["--capture-dir", str(tmp_path), "--check"]) == 1
    assert inventory_main(["--capture-dir", str(tmp_path)]) == 0


def test_inventory_synthetic_capture_is_healthy(capture_dir):
    assert health_problems(inventory(capture_dir)) == []
