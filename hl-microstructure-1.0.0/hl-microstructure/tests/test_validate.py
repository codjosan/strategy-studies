import numpy as np
import pandas as pd
import pytest

from hlmicro.config import FeeModel
from hlmicro.validate import cex_hl_lag, funding_settlement, hf_features, lead_lag, multi_gate, source_compare
from hlmicro.validate import tp_multifold
from hlmicro.validate.common import StepLookup, describe, rolling_zscore


def test_rolling_zscore_matches_definition():
    x = np.arange(10, dtype=float) ** 2
    z = rolling_zscore(x, 4)
    window = x[6:10]
    assert np.isnan(z[:3]).all()
    assert z[9] == pytest.approx((x[9] - window.mean()) / window.std(ddof=1))


def test_step_lookup_at_and_nearest():
    lookup = StepLookup(np.array([10, 20, 30]), np.array([1.0, 2.0, 3.0]), max_age_ms=5)
    assert np.isnan(lookup.at(9))
    assert lookup.at(20) == 2.0 and lookup.at(24) == 2.0
    assert np.isnan(lookup.at(26))                       # older than max_age
    assert lookup.nearest(25, 10) == 3.0                 # tie -> later observation
    assert np.isnan(lookup.nearest(100, 10))


def test_describe():
    s = describe([1.0, -1.0, 2.0])
    assert s["n"] == 3 and s["win_rate"] == pytest.approx(2 / 3)
    assert s["t"] == pytest.approx(s["mean"] / s["std"] * np.sqrt(3))
    assert describe([1.0])["sharpe"] == 0.0


def test_fee_model_round_trips():
    fees = FeeModel(taker_bps=4.0, maker_bps=1.0, half_spread_bps=0.5)
    assert fees.taker_round_trip_bps == 8.0
    assert fees.round_trip_with_spread_bps(fees.maker_bps, fees.taker_bps) == 6.0


@pytest.fixture(scope="module")
def tp_result(capture_dir):
    cfg = tp_multifold.Config(capture_dir=capture_dir, symbol="BTC", hold_min=30, cooldown_min=10,
                              tp_levels=[0.0, 0.2], stop_levels=[-1.0], n_folds=2)
    return tp_multifold.run(cfg)


def test_tp_multifold(tp_result, tmp_path):
    trades = tp_result.pooled[(-1.0, 0.0)][1]
    assert trades and all(t["exit_reason"] in ("stop", "tp", "timeout") for t in trades)
    assert all(t["minutes_held"] <= 30 for t in trades)
    entries = sorted(t["entry_ts_ms"] for t in trades)
    assert all(b > a for a, b in zip(entries, entries[1:]))
    assert "TAKE-PROFIT" in tp_multifold.render(tp_result)
    sweep, per_trade = tp_multifold.write_outputs(tp_result, tmp_path)
    assert list(pd.read_csv(per_trade).columns) == tp_multifold.TRADE_COLUMNS


def test_hf_features(capture_dir, tp_result, tmp_path):
    _, per_trade = tp_multifold.write_outputs(tp_result, tmp_path)
    res = hf_features.run(hf_features.Config(capture_dir=capture_dir, trades_csv=per_trade, symbol="BTC",
                                             trade_filter={"tp_pct": 0.0, "stop_pct": -1.0}))
    assert res.agreement is not None and res.agreement["corr"] > 0.5
    assert res.n_tested == 4
    assert "Feature tests" in hf_features.render(res)


def test_source_compare(capture_dir):
    res = source_compare.run(source_compare.Config(capture_dir=capture_dir, symbol="BTC", threshold=2.0,
                                                   hold_min=60, cooldown_min=10))
    assert res.shared_minutes > 2000
    assert res.rest is not None and res.ws is not None
    assert "classification" in source_compare.render(res)


def test_lead_lag(capture_dir):
    res = lead_lag.run(lead_lag.Config(capture_dir=capture_dir, symbol="BTC", days=None))
    assert res.cells
    for cell in res.cells:
        assert set(cell.summary["side"]) <= {cell.side}
    assert "Gate" in lead_lag.render(res)


def test_cex_hl_lag(capture_dir):
    res = cex_hl_lag.run(cex_hl_lag.Config(capture_dir=capture_dir, symbols=["SOL"], thresholds_bps=[10.0]))
    sweep = res.sweeps[0]
    assert sweep.n_triggers == sweep.n_up + sweep.n_down > 0
    first_up = sweep.response[sweep.response["direction"] == 1]
    assert (first_up.groupby("ts_ms")["horizon_s"].count() <= 5).all()
    assert "Gate summary" in cex_hl_lag.render(res)


def test_multi_gate_sell_rules_fire(capture_dir):
    res = multi_gate.run(multi_gate.Config(capture_dir=capture_dir, symbol="BTC", fills=False))
    assert res.trigger_counts["tick_speed_sell"] > 0
    assert res.trigger_counts["vwap_dev_sell"] > 0
    assert set(res.sweep["fees"]) == {"taker", "maker", "zero"}


def test_funding_settlement(capture_dir):
    res = funding_settlement.run(funding_settlement.Config(capture_dir=capture_dir, symbol="BTC"))
    assert len(res.events) >= 30
    assert {b["fee_bps"] for b in res.backtests} == {4.32, 1.44, 0.0}
    assert "Direction check" in funding_settlement.render(res)


def test_funding_settlement_unknown_venue_is_an_error(capture_dir):
    with pytest.raises(ValueError, match="no deribit mids"):
        funding_settlement.run(funding_settlement.Config(capture_dir=capture_dir, venue="deribit"))
