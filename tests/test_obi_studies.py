"""The shared imbalance z-score simulator and the studies built on it."""
import numpy as np
import pandas as pd
import pytest

from hlmicro.validate import breakeven_exit, oos_split, signal_exit, threshold_cv, trend_gate
from hlmicro.validate.obi_signal import ExitSpec, SignalSpec, folds_by_entry, simulate

T0 = 28_000_000  # minute index


def _grid(mid_after: list[float], n: int = 200, spike_at: int = 100) -> pd.DataFrame:
    """Flat noisy imbalance with a long spike at ``spike_at``; mid 100 until the
    spike, then ``mid_after`` (held at its last value)."""
    rng = np.random.default_rng(0)
    obi = rng.normal(0, 0.01, n)
    obi[spike_at:spike_at + 5] = 1.0
    mid = np.full(n, 100.0)
    tail = np.array(mid_after + [mid_after[-1]] * (n - spike_at - len(mid_after)))
    mid[spike_at:] = tail[: n - spike_at]
    minutes = np.arange(n, dtype=np.int64) + T0
    return pd.DataFrame({"minute_ts": minutes, "ts_ms": minutes * 60_000, "obi_ntl": obi, "mid": mid})


SIGNAL = SignalSpec(threshold=2.5)


def test_entry_on_trigger_bar_or_next_bar():
    grid = _grid([100.0, 101.0])
    same = simulate(grid, SIGNAL, ExitSpec(hold_min=10, cooldown_min=500), fee_bps=0)
    nxt = simulate(grid, SIGNAL, ExitSpec(hold_min=10, cooldown_min=500, next_bar_entry=True), fee_bps=0)
    assert same[0]["side"] == "LONG" and same[0]["entry_ts_ms"] == (T0 + 100) * 60_000
    assert nxt[0]["entry_ts_ms"] == (T0 + 101) * 60_000 and nxt[0]["entry_px"] == 101.0
    assert same[0]["exit_reason"] == nxt[0]["exit_reason"] == "timeout"
    assert same[0]["minutes_held"] == 10


def test_exit_priority_and_target_on_net():
    falling = _grid([100.0, 100.1, 98.0])
    t = simulate(falling, SIGNAL, ExitSpec(hold_min=60, cooldown_min=500, stop_pct=-1.0, target_pct=0.05),
                 fee_bps=8.64)
    assert t[0]["exit_reason"] == "tp" and t[0]["minutes_held"] == 1        # +0.1% gross >= 0.05
    t = simulate(falling, SIGNAL, ExitSpec(hold_min=60, cooldown_min=500, stop_pct=-1.0, target_pct=0.05,
                                           target_on_net=True, target_label="breakeven"), fee_bps=8.64)
    assert t[0]["exit_reason"] == "stop" and t[0]["minutes_held"] == 2       # +0.1% gross is < 0.05 net
    assert t[0]["pnl_pct_net"] == pytest.approx(-2.0 - 0.0864)


def test_open_position_reported_at_end():
    grid = _grid([100.0, 100.2])
    t = simulate(grid, SIGNAL, ExitSpec(hold_min=10_000, cooldown_min=500, close_at_end=True), fee_bps=0)
    assert [x["exit_reason"] for x in t] == ["open_at_end"]
    assert t[0]["minutes_held"] == 99
    assert simulate(grid, SIGNAL, ExitSpec(hold_min=10_000, cooldown_min=500), fee_bps=0) == []


def test_extension_and_counterfactual():
    grid = _grid([100.0, 100.5])
    grid.loc[100:, "obi_ntl"] = np.linspace(1.0, 3.0, 100)          # z60 stays with the trade
    t = simulate(grid, SIGNAL, ExitSpec(hold_min=10, cooldown_min=500, extension_z=0.5, max_hold_min=30),
                 fee_bps=0)
    assert t[0]["exit_reason"] in ("timeout", "max_hold") and t[0]["extended"]
    assert t[0]["pnl_at_hold_net"] == pytest.approx(0.5)


def test_folds_by_entry():
    trades = [{"entry_ts_ms": x} for x in (0, 24, 25, 99, 100)]
    by = folds_by_entry(trades, 0, 100, 4)
    assert [len(by[k]) for k in range(4)] == [2, 1, 0, 2]


@pytest.fixture(scope="module")
def cv_result(capture_dir):
    return threshold_cv.run(threshold_cv.Config(capture_dir=capture_dir, symbol="BTC", thresholds=[2.0, 2.5],
                                                hold_min=30, cooldown_min=10, stop_pct=-1.0, n_folds=2))


def test_threshold_cv(cv_result, tmp_path):
    for thr, pooled in cv_result.pooled.items():
        assert pooled["n"] == sum(cv_result.cells[(thr, f["idx"])]["n"] for f in cv_result.folds)
    assert cv_result.ranking and cv_result.ranking == sorted(cv_result.ranking, key=lambda r: r["composite"])
    assert "RANKING" in threshold_cv.render(cv_result)
    frame = pd.read_csv(threshold_cv.write_outputs(cv_result, tmp_path))
    assert {"threshold", "fold_idx", "stop_exits", "mean"} <= set(frame.columns)


def test_threshold_cv_rejects_positive_stop(capture_dir):
    with pytest.raises(ValueError, match="negative"):
        threshold_cv.run(threshold_cv.Config(capture_dir=capture_dir, stop_pct=1.0))


def test_oos_split(capture_dir, tmp_path):
    res = oos_split.run(oos_split.Config(capture_dir=capture_dir, symbol="BTC", hold_min=30, cooldown_min=10,
                                         train_frac=0.5, threshold_sweep=True, sweep_thresholds=[2.0, 2.5]))
    assert res.windows["train"][1] < res.boundary_ms <= res.windows["test"][0]
    assert res.verdict in ("PASS", "MARGINAL", "FAIL", "no test trades", "no training trades")
    assert {t["period"] for t in res.trades} <= {"train", "test"}
    assert [s["threshold"] for s in res.sweep] == [2.0, 2.5]
    assert "VERDICT" in oos_split.render(res)
    assert oos_split.write_outputs(res, tmp_path).exists()
    with pytest.raises(ValueError, match="test window is empty"):
        oos_split.run(oos_split.Config(capture_dir=capture_dir, symbol="BTC", split_date="20300101"))


def test_breakeven_exit(capture_dir, tmp_path):
    res = breakeven_exit.run(breakeven_exit.Config(capture_dir=capture_dir, symbol="BTC", cooldown_min=10,
                                                   stop_levels=[-1.0], breakeven_levels=[0.0],
                                                   max_hold_levels=[30, 120], reference_hold_min=30))
    assert len(res.references) == 1 and len(res.sweep) == 2
    for row in res.sweep:
        for t in row.trades:
            if t["exit_reason"] == "breakeven":
                assert t["pnl_pct_net"] >= 0
            assert t["exit_reason"] == "open_at_end" or t["minutes_held"] <= row.max_hold
    assert "FAILURE-MODE FLAGS" in breakeven_exit.render(res)
    assert breakeven_exit.write_outputs(res, tmp_path).exists()


def test_breakeven_failure_flags():
    stats = {"n": 9, "n_open": 1, "winr": 80.0, "mean": 0.01, "skew": -2.0, "med_hold": 1000}
    flags = breakeven_exit.failure_flags(stats, reference_hold_min=240)
    assert [f.split(" (")[0] for f in flags] == ["CENSORING>=10%", "WIN-RATE TRAP", "NEG SKEW", "LOCKUP"]


def test_signal_exit(capture_dir, tmp_path):
    res = signal_exit.run(signal_exit.Config(capture_dir=capture_dir, symbol="BTC", hold_min=30, max_hold_min=60,
                                             cooldown_min=10, stop_pct=-0.5, reversal_levels=[0.0, 1.0],
                                             reversal_min_hold=10, extension_levels=[0.0, 1.0]))
    assert [(v.reversal_z, v.extension_z) for v in res.sweep] == [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    for v in res.sweep:
        for t in v.trades:
            assert t["minutes_held"] <= (60 if v.extension_z else 30)
            if t["exit_reason"] == "reversal":
                assert t["minutes_held"] >= 10
    assert "COUNTERFACTUAL" in signal_exit.render(res)
    assert signal_exit.write_outputs(res, tmp_path).exists()


def _trend_panel(strong: bool, n: int = 30000, seed: int = 7) -> pd.DataFrame:
    """Regime-switching drift and imbalance bursts; with ``strong`` the drift is
    large enough that a trend filter on longs must help."""
    rng = np.random.default_rng(seed)
    drift, i = np.zeros(n), 0
    while i < n:
        span = rng.integers(800, 1500)
        drift[i:i + span] = rng.choice([1, -1]) * (0.00045 if strong else 0.0)
        i += span
    mid = 100 * np.cumprod(1 + drift[:n] + rng.standard_normal(n) * 0.0018)
    obi, i = np.zeros(n), 0
    while i < n - 120:
        if rng.random() < 0.05:
            w = rng.integers(70, 110)
            obi[i:i + w] += rng.choice([-1, 1]) * 3.0
            i += w + rng.integers(60, 120)
        else:
            i += 1
    minutes = np.arange(n, dtype=np.int64) + T0
    return pd.DataFrame({"minute_ts": minutes, "ts_ms": minutes * 60_000, "obi_ntl": obi, "mid": mid})


@pytest.mark.parametrize("strong", [True, False])
def test_trend_gate_detects_planted_trend(strong):
    panel = _trend_panel(strong)
    cfg = trend_gate.Config(symbol="SYN", threshold=2.5, n_perm=2000)
    res = trend_gate.evaluate(panel, trend_gate.baseline_trades(panel, cfg), cfg)
    assert res.selected_window in cfg.sma_windows
    assert res.passed is strong
    assert "VERDICT" in trend_gate.render(res)


def test_trend_gate_runs_on_capture(capture_dir):
    res = trend_gate.run(trend_gate.Config(capture_dir=capture_dir, symbol="BTC", hold_min=30, cooldown_min=10,
                                           n_perm=200))
    assert res.venues == ["binance", "bybit", "hyperliquid", "okx"]
    assert "Trend gate" in trend_gate.render(res)
