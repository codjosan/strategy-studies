import gzip
import json

import pandas as pd
import pytest

from hlmicro.parse import discover, minute_grid
from hlmicro.parse.sanitize import DROP_RULES, MANIFEST, Options, sanitize_dir, sanitize_book_file
from hlmicro.schemas import BOOK_REST_COLUMNS

DAY = "20260801"
T0 = 1785542400000          # 2026-08-01 00:00:00 UTC
HEADER = ",".join(BOOK_REST_COLUMNS)


def row(ts, coin="BTC", venue="binance", bid=100.0, ask=100.02, qb=3.0, qa=1.0, spread=None,
        obi_qty=None, obi_ntl=None, mid=None):
    mid = (bid + ask) / 2 if mid is None else mid
    nb, na = bid * qb, ask * qa
    oq = (qb - qa) / (qb + qa) if obi_qty is None else obi_qty
    on = (nb - na) / (nb + na) if obi_ntl is None else obi_ntl
    sp = (ask - bid) / mid * 1e4 if spread is None else spread
    return (f"{ts},{coin},{venue},{mid:.6f},{bid:.6f},{ask:.6f},{qb:.4f},{qa:.4f},"
            f"{nb:.2f},{na:.2f},{oq:.4f},{on:.4f},{sp:.2f}")


def write(path, lines, gz=False):
    text = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def test_every_drop_rule(tmp_path):
    good = [row(T0 + 5000 * i, venue=v) for i in range(3) for v in ("binance", "bybit", "hyperliquid")]
    bad = {
        "wrong_field_count": row(T0 + 1, venue="okx") + ",EXTRA",
        "missing_or_non_numeric": row(T0 + 2).replace("100.010000", "abc"),
        "bad_coin": row(T0 + 3, coin="B-TC"),
        "unknown_venue": row(T0 + 4, venue="deribit"),
        "ts_out_of_range": row(T0 - 3 * 86_400_000),
        "nonpositive_price": row(T0 + 6, bid=-1.0, ask=1.0, mid=0.0, spread=0.0),
        "crossed_or_locked": row(T0 + 7, bid=100.02, ask=100.0),
        "mid_inconsistent": row(T0 + 8, mid=100.5),
        "spread_inconsistent": row(T0 + 9, spread=9.99),
        "negative_depth": row(T0 + 10, qb=-1.0, qa=1.0, obi_qty=0.0, obi_ntl=0.0),
        "imbalance_out_of_range": row(T0 + 11, obi_qty=1.5),
        "imbalance_inconsistent": row(T0 + 12, obi_ntl=-0.3),
    }
    dup = row(T0 + 13)
    key_old, key_new = row(T0 + 14, qb=2.0), row(T0 + 14, qb=2.5)
    lines = [HEADER, bad["wrong_field_count"], *good[:4], *[v for k, v in bad.items() if k != "wrong_field_count"],
             dup, dup, key_old, key_new, *good[4:]]
    src = write(tmp_path / f"obi_5s_{DAY}.csv", lines)
    out = tmp_path / "out"
    out.mkdir()
    report, _, _ = sanitize_book_file(src, out, Options())
    expected = {k: 1 for k in bad} | {"duplicate": 1, "duplicate_key": 1}
    assert report.dropped == expected
    assert set(report.dropped) <= set(DROP_RULES)
    clean = pd.read_csv(out / src.name)
    assert len(clean) == len(good) + 2
    assert clean["ts_ms"].is_monotonic_increasing
    assert clean.loc[clean["ts_ms"] == T0 + 14, "bid_qty_5"].tolist() == [2.5]      # later write kept
    assert list(clean.columns) == list(BOOK_REST_COLUMNS)


def test_okx_contracts_converted_to_coins_and_unknown_coins_reported(tmp_path):
    lines = [HEADER, row(T0, venue="okx", qb=150.0, qa=50.0), row(T0 + 1, coin="XRP", venue="okx"),
             row(T0 + 2, venue="binance", qb=1.5, qa=0.5)]
    src = write(tmp_path / f"obi_5s_{DAY}.csv", lines)
    out = tmp_path / "out"
    out.mkdir()
    report, _, _ = sanitize_book_file(src, out, Options())
    clean = pd.read_csv(out / src.name).set_index(["coin", "venue"])
    assert clean.loc[("BTC", "okx"), "bid_qty_5"] == pytest.approx(1.5)            # 150 contracts x 0.01
    assert clean.loc[("BTC", "okx"), "bid_ntl_5"] == pytest.approx(150.0)
    assert clean.loc[("BTC", "okx"), "obi_ntl"] == clean.loc[("BTC", "binance"), "obi_ntl"]
    assert report.okx_rows_converted == 1 and report.okx_rows_unconverted == 1
    assert report.okx_coins_unconverted == "XRP"


def test_text_is_normalised_and_untouched_values_keep_their_text(tmp_path):
    line = row(T0, coin=" btc ", venue="Binance")
    src = write(tmp_path / f"obi_5s_{DAY}.csv", [HEADER, line])
    out = tmp_path / "out"
    out.mkdir()
    sanitize_book_file(src, out, Options())
    written = (out / src.name).read_text().splitlines()[1].split(",")
    assert written[1:3] == ["BTC", "binance"]
    assert written[3:] == line.split(",")[3:]


def test_truncated_gzip_and_lost_header(tmp_path):
    body = [row(T0 + 5000 * i) for i in range(200)]
    full = tmp_path / "full.gz"
    write(full, body, gz=True)                                     # no header line
    data = full.read_bytes()
    src = tmp_path / f"obi_5s_{DAY}.csv.gz"
    src.write_bytes(data[: len(data) * 2 // 3])
    out = tmp_path / "out"
    out.mkdir()
    report, _, _ = sanitize_book_file(src, out, Options())
    assert report.truncated and report.header_repaired
    assert 0 < report.rows_out < 200
    assert (out / src.name).exists()                               # compression kept


def test_flags_do_not_drop_rows(tmp_path):
    lines = [HEADER]
    for i in range(3):
        t = T0 + 5000 * i
        lines += [row(t, venue="binance"), row(t + 1, venue="bybit"), row(t + 2, venue="okx"),
                  row(t + 3, venue="hyperliquid", bid=103.0, ask=103.02)]      # ~300 bps off
    lines.append(row(T0 + 200_000, venue="binance", bid=100.0, ask=101.0))     # 100 bps spread, gap
    src = write(tmp_path / f"obi_5s_{DAY}.csv", lines)
    out = tmp_path / "out"
    out.mkdir()
    report, flags, gaps = sanitize_book_file(src, out, Options())
    assert report.rows_out == 13 and not report.dropped
    assert report.venue_deviation == 3 and report.wide_spread == 1
    assert set(flags["flag"]) == {"venue_deviation", "wide_spread"}
    assert report.outages == 1 and ((gaps["coin"] == "BTC") & (gaps["venue"] == "binance")).sum() == 1


def test_directory_run_is_readable_and_cannot_be_repeated_on_its_output(capture_dir, tmp_path):
    out = tmp_path / "clean"
    reports = sanitize_dir(capture_dir, out, Options(), jobs=2)
    assert reports and not any(r.error for r in reports)
    manifest = json.loads((out / MANIFEST).read_text())
    assert set(manifest["families"]) == {"obi_5s", "obi_ws", "hl_trades", "okx_trades", "bybit_trades",
                                         "hl_liquidations"}
    for fam, run in manifest["families"].items():
        assert run["rows_out"] == sum(r.rows_out for r in reports if r.family == fam)
        assert (out / f"sanitize_report_{fam}.csv").exists() and (out / f"sanitize_gaps_{fam}.csv").exists()
    raw = minute_grid(discover(capture_dir, "obi_5s"), "BTC")
    clean = minute_grid(discover(out, "obi_5s"), "BTC")
    assert len(clean) == len(raw)
    pd.testing.assert_series_equal(clean["mid"], raw["mid"])        # OKX units do not move mids
    with pytest.raises(ValueError, match="already sanitized"):
        sanitize_dir(out, tmp_path / "again", Options())


# ----------------------------------------------------------------------------- trade tapes

from hlmicro.parse import load_tape                                   # noqa: E402
from hlmicro.parse.sanitize import sanitize_tape_file                 # noqa: E402

TAPE_HEADER = "ts_ms,coin,px,sz,side,tid,user_a,user_b"
ADDR = "0x" + "ab" * 20


def trade(ts, coin="BTC", px="80000.0", sz="0.01", side="B", tid=None, extra=True):
    tid = tid if tid is not None else str(10**14 + ts % 10**9)
    base = f"{ts},{coin},{px},{sz},{side},{tid}"
    return base + (f",{ADDR},{ADDR[:-1]}c" if extra else "")


def test_tape_every_drop_rule_and_identifiers_removed(tmp_path):
    good = [trade(T0 + 1000 * i, px=str(80000 + i)) for i in range(30)]
    bad = {
        "wrong_field_count": trade(T0 + 31_001) + ",EXTRA",
        "missing_or_non_numeric": trade(T0 + 31_002, px="x"),
        "bad_coin": trade(T0 + 31_003, coin="B TC!"),
        "bad_side": trade(T0 + 31_004, side="Q"),
        "bad_tid": trade(T0 + 31_005, tid="12ab"),
        "ts_out_of_range": trade(T0 + 5 * 86_400_000),
        "nonpositive_price": trade(T0 + 31_007, px="0"),
        "nonpositive_size": trade(T0 + 31_008, sz="-1"),
    }
    replay = good[3]
    same_tid = trade(T0 + 31_010, tid=good[4].split(",")[5], px="80004.5")
    lines = [TAPE_HEADER, *good[:10], *bad.values(), *good[10:], replay, same_tid]
    src = write(tmp_path / f"hl_trades_{DAY}.csv", lines)
    out = tmp_path / "out"
    out.mkdir()
    report, _, _, _ = sanitize_tape_file(src, out, Options())
    assert report.dropped == {k: 1 for k in bad} | {"duplicate": 1, "duplicate_tid": 1}
    assert report.columns_removed == "user_a user_b"
    text = (out / src.name).read_text()
    assert "0x" not in text
    clean = pd.read_csv(out / src.name)
    assert list(clean.columns) == ["ts_ms", "coin", "px", "sz", "side", "tid"]
    assert len(clean) == 30 and clean["ts_ms"].is_monotonic_increasing


def test_tape_normalises_line_endings_sides_and_sorts_replays(tmp_path):
    rows = [trade(T0 + 1000 * i, side="buy" if i % 2 else "S") for i in range(10)]
    replayed = rows[5:8]                                   # reconnect replays 3 older trades
    lines = [TAPE_HEADER, *rows, *replayed, trade(T0 + 20_000)]
    src = tmp_path / f"hl_trades_{DAY}.csv.gz"
    with gzip.open(src, "wb") as fh:
        fh.write(("\r\n".join(lines) + "\r\n").encode())
    out = tmp_path / "out"
    out.mkdir()
    report, _, _, _ = sanitize_tape_file(src, out, Options())
    assert report.line_endings_fixed == len(lines)
    assert report.dropped == {"duplicate": 3}
    clean = pd.read_csv(out / src.name)
    assert set(clean["side"]) == {"A", "B"} and clean["ts_ms"].is_monotonic_increasing
    assert b"\r" not in gzip.open(out / src.name).read()


def test_tape_rows_of_another_day_move_to_their_own_file(tmp_path):
    d1, d2 = "20260801", "20260802"
    t1 = T0 + 86_400_000
    write(tmp_path / f"hl_trades_{d1}.csv", [TAPE_HEADER] + [trade(T0 + 1000 * i) for i in range(5)])
    late = trade(t1 - 200, tid="777")                   # last 0.2 s of day 1, written after rotation
    write(tmp_path / f"hl_trades_{d2}.csv", [TAPE_HEADER, late] + [trade(t1 + 1000 * i) for i in range(5)])
    out = tmp_path / "out"
    reports = {r.file: r for r in sanitize_dir(tmp_path, out, Options(), families=["hl_trades"])}
    c1 = pd.read_csv(out / f"hl_trades_{d1}.csv")
    c2 = pd.read_csv(out / f"hl_trades_{d2}.csv")
    assert 777 in c1["tid"].tolist() and 777 not in c2["tid"].tolist()
    assert len(c1) == 6 and len(c2) == 5 and c1["ts_ms"].is_monotonic_increasing
    assert reports[f"hl_trades_{d2}.csv"].rows_moved_out == 1
    assert reports[f"hl_trades_{d1}.csv"].rows_moved_in == 1
    # a day that is not part of the run keeps its rows in place
    only2 = sanitize_dir(tmp_path, tmp_path / "out2", Options(), families=["hl_trades"], start=d2)
    assert only2[0].other_day_kept == 1 and only2[0].rows_out == 6


def test_tape_flags_are_kept(tmp_path):
    lines = [TAPE_HEADER] + [trade(T0 + 1000 * i, px="100.0") for i in range(60)]
    lines.append(trade(T0 + 60_500, px="103.0"))          # 300 bps print (sweep)
    lines.append(trade(T0 + 61_000, px="100.0", sz="200000"))   # $20M block
    lines += [trade(T0 + 300_000 + 1000 * i, px="100.0") for i in range(60)]   # 4-minute gap before
    src = write(tmp_path / f"hl_trades_{DAY}.csv", lines)
    out = tmp_path / "out"
    out.mkdir()
    report, flags, gaps, _ = sanitize_tape_file(src, out, Options())
    assert not report.dropped and report.rows_out == 122
    assert report.price_jump == 1 and report.large_trade == 1
    assert set(flags["flag"]) == {"price_jump", "large_trade"}
    assert report.gaps == 1 and report.outages == 1


def test_sanitized_tape_loads_like_the_raw_one(capture_dir, tmp_path):
    out = tmp_path / "clean"
    sanitize_dir(capture_dir, out, Options(), families=["hl_trades"])
    raw = load_tape(discover(capture_dir, "hl_trades"), "BTC")
    clean = load_tape(discover(out, "hl_trades"), "BTC")
    pd.testing.assert_frame_equal(raw.reset_index(drop=True), clean.reset_index(drop=True))
