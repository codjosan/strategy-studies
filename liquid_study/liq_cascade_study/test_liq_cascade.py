import numpy as np
from pathlib import Path

from fast_capture.liq_cascade.events import window_sum, find_events
from fast_capture.liq_cascade.file_utils import DAY_S
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
    assert files["20260101"].name == "hl_trades_20260101.csv"

def _liq_days(n_days: int, seed: int = 0):
    """n_days of 1-second arrays: flat price, 300 background liquidation seconds per day."""
    rng = np.random.default_rng(seed)
    n = n_days * DAY_S
    px = np.full(n, 100.0, dtype="float32")
    ls = np.zeros(n, dtype="float32")
    lb = np.zeros(n, dtype="float32")
    for d in range(n_days):
        secs = d * DAY_S + rng.choice(np.arange(1000, DAY_S - 4000), 300, replace=False)
        ls[secs] = rng.uniform(1.0, 100.0, 300).astype("float32")
    days = [f"202601{i + 1:02d}" for i in range(n_days)]
    return days, px, ls, lb


def test_thresholds_use_only_earlier_days():
    days, px, ls, lb = _liq_days(8)
    burst = 7 * DAY_S + 50_000
    ls[burst] = 1e6
    events = find_events(coin="BTC", days_all=days, px=px, ls=ls, lb=lb, window_s=1, qs=[0.95],
                         lookback_days=5, warmup_days=3, latency_s=1, horizons_s=[60])
    # nothing before the warm-up is complete (days 0-2 have < 3 earlier days)
    assert all(e["day"] >= days[3] for e in events)
    burst_event = [e for e in events if e["liq_ntl"] == 1e6]
    assert len(burst_event) == 1
    # the day-7 threshold comes from days 2..6 only: the day-7 burst must not raise it
    pool = np.concatenate([ls[d * DAY_S:(d + 1) * DAY_S][ls[d * DAY_S:(d + 1) * DAY_S] > 0]
                           for d in range(2, 7)])
    assert burst_event[0]["thr"] == np.float64(np.quantile(pool, 0.95))


def test_no_threshold_without_history():
    days, px, ls, lb = _liq_days(1)
    events = find_events(coin="BTC", days_all=days, px=px, ls=ls, lb=lb, window_s=2, qs=[0.5],
                         lookback_days=1, warmup_days=0, latency_s=1, horizons_s=[10])
    assert events == []


def test_non_overlapping_events():
    days, px, ls, lb = _liq_days(2)
    ls[DAY_S + 20_000:DAY_S + 20_200] = 1e5        # 200 s of continuous bursts on day 1
    events = find_events(coin="BTC", days_all=days, px=px, ls=ls, lb=lb, window_s=2, qs=[0.5],
                         lookback_days=1, warmup_days=1, latency_s=1, horizons_s=[30, 60])
    assert len(events) >= 2
    times = sorted(e["ts"].timestamp() for e in events)
    assert all(b - a >= 60 for a, b in zip(times, times[1:]))
