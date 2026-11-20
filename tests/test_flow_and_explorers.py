"""Taker-flow studies, trigger comparison and the signal explorers."""
import numpy as np
import pandas as pd
import pytest

from hlmicro.validate import feature_explore, obi_configs, scalp_explore, tfi_signals


@pytest.fixture(scope="module")
def configs_result(capture_dir):
    return obi_configs.run(obi_configs.Config(capture_dir=capture_dir, symbols=["SOL", "BTC"]))


def test_obi_configs_trades(configs_result):
    ungated = [v for v in configs_result.variants if v.gate == "ungated"]
    assert {v.symbol for v in ungated} == {"SOL", "BTC"}
    for v in ungated:
        t = v.trades
        if t.empty:
            continue
        assert set(t["exit_reason"]) <= {"tp", "stop", "horizon"}
        assert (t["entry_ts_ms"] % 60_000 == 0).all()                    # priced at a minute close
        assert (t["entry_ts_ms"].to_numpy()[1:] > t["exit_ts_ms"].to_numpy()[:-1]).all()   # no overlap
        tp = t[t["exit_reason"] == "tp"]
        if v.trigger.tp_pct is None:
            assert tp.empty
        else:
            assert np.allclose(tp["gross_bps"], v.trigger.tp_pct * 100)
        assert (t["hold_min"] <= v.trigger.max_hold_min).all()


def test_obi_configs_tfi_gates(configs_result):
    btc = {v.gate: v for v in configs_result.variants if v.symbol == "BTC"}
    assert set(btc) == {"ungated", "soft", "strict"}
    soft, strict = btc["soft"].trades, btc["strict"].trades
    assert (np.sign(soft["tfi"]) == soft["side"]).all()
    assert (strict["tfi_z"].abs() >= 1.0).all() and len(strict) <= len(soft) <= len(btc["ungated"].trades)
    assert "Best variant per symbol" in obi_configs.render(configs_result)


def test_obi_configs_rejects_unknown_symbol(capture_dir):
    with pytest.raises(ValueError, match="no trigger presets"):
        obi_configs.run(obi_configs.Config(capture_dir=capture_dir, symbols=["DOGE"]))


def test_streak_triggers():
    flow = pd.DataFrame({"bucket": np.arange(10), "xv_tfi": [0.3] * 5 + [-0.3] * 5})
    trig = tfi_signals.streak_triggers(flow, n=3, floor=0.1, cooldown_min=0)
    assert trig.values.tolist() == [[2, 1], [7, -1]]


def test_zscore_triggers_respect_cooldown_per_direction():
    flow = pd.DataFrame({"bucket": np.arange(6), "xv_tfi_z": [3.0, 3.0, -3.0, 3.0, -3.0, 3.0]})
    trig = tfi_signals.zscore_triggers(flow, threshold=2.0, cooldown_min=3)
    assert trig.values.tolist() == [[0, 1], [2, -1], [3, 1]]


def test_tfi_signals(capture_dir):
    res = tfi_signals.run(tfi_signals.Config(capture_dir=capture_dir, symbols=["BTC", "ETH"],
                                             z_horizons_min=[5, 15], streak_horizons_min=[5], min_n=10))
    assert res.coverage["ETH"] is None and res.coverage["BTC"]["venues_per_minute"] > 2
    zs = [s for s in res.sweeps if s.signal == "zscore"]
    assert zs and all(s.response.empty or (s.response["ts_ms"] % 60_000 == 0).all() for s in zs)
    assert "Gate summary" in tfi_signals.render(res)


@pytest.fixture(scope="module")
def scalp_cfg(capture_dir):
    return scalp_explore.Config(capture_dir=capture_dir, symbol="BTC", holds_s=[60, 300], z_windows_s=[300],
                                imb_windows_s=[60], min_trades=20, realism_top_n=1, k_bps=[0.0])


def test_scalp_features(scalp_cfg):
    grid = scalp_explore.build_features(scalp_cfg)
    assert {"tfi_hl_60s", "tfi_okx_60s", "obi_mean_z300s", "liq_imbalance", "liq_spike_z",
            "hl_px_min"} <= set(grid.columns)
    # Hyperliquid labels sells "A": sell flow must register on every venue.
    for v in ("hl", "okx", "bybit"):
        assert (grid[f"tfi_{v}"] < 0).any() and (grid[f"{v}_sell_ntl"] > 0).any()
    assert (grid["long_liq_ntl"] > 0).any() and (grid["short_liq_ntl"] > 0).any()
    # A window with quiet ticks still has a flow imbalance.
    assert grid["tfi_hl_60s"].notna().mean() > 0.95


def test_scalp_explore(scalp_cfg, tmp_path):
    res = scalp_explore.run(scalp_cfg)
    assert not res.sweep.empty and set(res.sweep["fees"]) == {"taker", "maker", "zero"}
    if not res.top.empty:
        assert (res.top["sharpe"] > 0).all() and not res.realism.empty
        assert {"entry_fill_rate", "exit_maker_rate"} <= set(res.realism.columns)
    assert "Scalp explorer" in scalp_explore.render(res)
    assert scalp_explore.write_outputs(res, tmp_path)


def test_scalp_no_overlap():
    kept = scalp_explore.no_overlap(np.array([0, 1, 2, 6, 7, 10, 30]), hold_ticks=5)
    assert kept.tolist() == [0, 7, 30]


def test_feature_zscore_uses_preceding_window():
    z = feature_explore._zs(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 4)
    assert z.iloc[:4].tolist() == [0.0] * 4
    assert z.iloc[4] == pytest.approx((5 - 2.5) / np.std([1, 2, 3, 4]))
    assert feature_explore._zs(pd.Series([0.0] * 5 + [9.0]), 4).iloc[-1] == 0.0   # zero variance


def test_feature_evaluate_cooldown_and_split():
    n = 200
    side = np.ones(n, dtype=np.int8)
    mid = np.linspace(100, 120, n)
    cfg = feature_explore.Config(cooldown_min=10, fee_bps=0.0, min_trades=5)
    s = feature_explore.evaluate(side, mid, 5, cfg, train_end=100, validate_end=150)
    assert s["n"] == 14                                   # entries at 60, 70, ..., 190
    assert (s["train"]["n"], s["validate"]["n"], s["holdout"]["n"]) == (4, 5, 5)
    assert s["win_rate"] == 100.0
    nxt = feature_explore.evaluate(side, mid, 5, feature_explore.Config(cooldown_min=10, fee_bps=0.0,
                                                                       min_trades=5, next_bar_entry=True),
                                   train_end=100, validate_end=150)
    assert nxt["n"] == 14 and nxt["mean"] < s["mean"]     # later entry on a rising price


def test_feature_explore(capture_dir, tmp_path):
    res = feature_explore.run(feature_explore.Config(capture_dir=capture_dir, symbol="BTC", horizons_min=[5, 60],
                                                     cooldown_min=15))
    assert res.rows and all(r["stats"]["n"] >= 10 for r in res.rows)
    assert {r["group"] for r in res.rows} >= {"A", "H1", "H6", "2"}
    assert all(feature_explore.validated(r["stats"]) for r in res.validated())
    assert "HOLDOUT" in feature_explore.render(res)
    frame = pd.read_csv(feature_explore.write_outputs(res, tmp_path))
    assert {"label", "horizon_min", "train_sharpe", "holdout_mean", "validated"} <= set(frame.columns)
