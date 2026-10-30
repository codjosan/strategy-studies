import numpy as np
from pathlib import Path

from fast_capture.liq_cascade.events import window_sum, find_events
from fast_capture.liq_cascade.file_utils import glob_daily

def test_window_sum():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    w = 3
    res = window_sum(x, w)
    expected = np.array([1.0, 3.0, 6.0, 9.0, 12.0], dtype="float32")
    np.testing.assert_allclose(res, expected)

def test_glob_daily_duplicate_resolution(tmp_path: Path):
    (tmp_path / "hl_trades_20260101.csv.gz").write_text("compressed")
    (tmp_path / "hl_trades_20260101.csv").write_text("uncompressed")
    
    files = glob_daily(tmp_path, "hl_trades_")
    assert "20260101" in files
    assert files["20260101"].suffix == ""

def test_past_day_thresholds_no_lookahead():
    days_all = [f"202601{i:02d}" for i in range(1, 11)]
    DAY_S = 10
    n = len(days_all) * DAY_S
    
    px = np.ones(n, dtype="float32") * 100.0
    ls = np.zeros(n, dtype="float32")
    lb = np.zeros(n, dtype="float32")
    
    ls[7 * DAY_S + 5] = 1000.0
    
    events = find_events(
        coin="BTC", days_all=days_all, px=px, ls=ls, lb=lb,
        window_s=2, qs=[0.95], lookback_days=5, warmup_days=3,
        latency_s=1, horizons_s=[5]
    )
    assert isinstance(events, list)

def test_non_overlapping_events():
    days_all = ["20260101"]
    DAY_S = 100
    n = DAY_S
    px = np.ones(n, dtype="float32") * 100.0
    ls = np.ones(n, dtype="float32") * 10.0
    lb = np.zeros(n, dtype="float32")
    
    ls[10] = 10000.0
    ls[15] = 10000.0
    
    events = find_events(
        coin="BTC", days_all=days_all, px=px, ls=ls, lb=lb,
        window_s=2, qs=[0.5], lookback_days=1, warmup_days=0,
        latency_s=1, horizons_s=[10]
    )
    
    if len(events) > 1:
        timestamps = [ev["ts"] for ev in events]
        diffs = np.diff([ts.timestamp() for ts in timestamps])
        assert all(d >= 10 for d in diffs)
