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


def test_folds_include_the_last_observation():
    grid = pd.DataFrame({"ts_ms": np.arange(10) * 60_000})
    assert sum(len(f["grid"]) for f in tp_multifold.split_folds(grid, 2)) == 10
    s = source_compare.summarize([{"entry_min": 0, "ret_net": 0.01}, {"entry_min": 100, "ret_net": 0.01}],
                                 2, 0, 100)
    assert not np.isnan(s["fold_means"]).any()


def test_hf_features_returns_are_fractions_for_small_percent_returns(capture_dir, tmp_path):
    trades = pd.DataFrame({"side": ["LONG", "SHORT"] * 15,
                           "entry_ts_ms": (np.arange(30) * 600_000 + 1_767_600_000_000),
                           "pnl_pct_net": np.linspace(-0.1, 0.1, 30)})       # all |x| < 0.2 %
    path = tmp_path / "small.csv"
    trades.to_csv(path, index=False)
    res = hf_features.run(hf_features.Config(capture_dir=capture_dir, trades_csv=path, symbol="BTC"))
    assert res.per_trade["pnl_pct_net"].abs().max() == pytest.approx(0.001)


def test_hf_features_unknown_filter_column_is_an_error(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"pnl_pct_net": [0.1], "tp_pct": [0.0]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="tpct"):
        hf_features.load_trades(path, "BTC", {"tpct": 0.0})


def test_multi_gate_k_bps_is_comma_separated(capture_dir, tmp_path, capsys):
    multi_gate.main(["--capture-dir", str(capture_dir), "--symbol", "BTC", "--k-bps", "0,0.5",
                     "--outdir", str(tmp_path)])
    assert "Passive-limit fills" in capsys.readouterr().out


def test_tp_multifold_enters_the_minute_after_the_signal(tp_result):
    trades = tp_result.pooled[(-1.0, 0.0)][1]
    assert trades
    grid = tp_multifold.minute_grid(tp_result.files, "BTC").set_index("ts_ms")
    for t in trades:
        assert t["entry_ts_ms"] == t["signal_ts_ms"] + 60_000
        assert t["entry_px"] == pytest.approx(grid.loc[t["entry_ts_ms"], "mid"])


def test_source_compare_stop_fills_at_the_observed_price():
    obi = np.r_[np.zeros(59), [0.0, 1.0], np.zeros(9)] + np.tile([0.01, -0.01], 35)
    prices = np.full(70, 100.0)
    prices[62] = 97.0                                         # gaps through a -1% stop
    cfg = source_compare.Config(threshold=2.0, stop_pct=-1.0, tp_pct=0.0, hold_min=5, fee_bps=0.0)
    trades = source_compare.backtest(np.arange(70) * 60_000, obi, prices, cfg)
    assert trades and trades[0]["ret_net"] == pytest.approx(-0.03)


def test_funding_settlement_books_the_funding_payment():
    flat = {"final_ret": 0.0, "post_ret": 0.0}
    events = [{"hour_ms": h * 3_600_000, "funding_rate": r, **flat}
              for h, r in enumerate(np.linspace(-1e-4, 1e-4, 40))]
    bt = funding_settlement.backtest(events, 10.0, 0.0, 1000.0)
    assert bt["price_total"] == 0.0
    assert bt["funding_total"] > 0                            # shorts receive +funding, longs receive -funding
    assert bt["gross_total"] == pytest.approx(bt["funding_total"])


def test_lead_lag_ignores_stale_hyperliquid_mids(capture_dir):
    def n_events(max_age):
        res = lead_lag.run(lead_lag.Config(capture_dir=capture_dir, symbol="BTC", days=None,
                                           max_quote_age_ms=max_age))
        return sum(int(c.summary["n"].max()) for c in res.cells if not c.summary.empty)
    unlimited, default, strict = n_events(None), n_events(5_000), n_events(200)
    assert unlimited >= default > strict          # ~2 s between synthetic updates: 200 ms drops most


def _simulate_reference(grid, cfg, stop_pct, tp_pct):
    """The original minute-by-minute loop of tp_multifold.simulate, kept as the reference."""
    obi, mid, ts_ms = grid["obi_ntl"].to_numpy(), grid["mid"].to_numpy(), grid["ts_ms"].to_numpy()
    z_primary = rolling_zscore(obi, cfg.primary_window_min)
    z_confirm = rolling_zscore(obi, cfg.confirm_window_min)
    fee_pct = cfg.fee_bps / 100.0
    trades, side, pending = [], None, None
    entry_idx = signal_idx = deadline = 0
    entry_px, last_exit, mae, mfe = 0.0, -10**9, 0.0, 0.0
    for i in range(max(cfg.primary_window_min, cfg.confirm_window_min), len(grid)):
        px = mid[i]
        if pending is not None:
            if px > 0:
                side, entry_idx, entry_px, deadline = pending, i, px, i + cfg.hold_min
                mae = mfe = 0.0
            pending = None
            continue
        if not px > 0:
            continue
        if side is not None:
            ret = (px - entry_px) / entry_px * 100
            if side == "SHORT":
                ret = -ret
            mae, mfe = min(mae, ret), max(mfe, ret)
            reason = None
            if stop_pct is not None and ret <= stop_pct:
                reason = "stop"
            elif tp_pct and ret >= tp_pct:
                reason = "tp"
            elif i >= deadline:
                reason = "timeout"
            if reason:
                trades.append({"side": side, "signal_ts_ms": int(ts_ms[signal_idx]),
                               "entry_ts_ms": int(ts_ms[entry_idx]), "exit_ts_ms": int(ts_ms[i]),
                               "entry_px": float(entry_px), "exit_px": float(px),
                               "minutes_held": i - entry_idx, "exit_reason": reason,
                               "ret_gross_pct": ret, "pnl_pct_net": ret - fee_pct,
                               "mae_pct": mae, "mfe_pct": mfe})
                side = None
                last_exit = i
        if side is None and i - last_exit >= cfg.cooldown_min:
            zp, zc = z_primary[i], z_confirm[i]
            if np.isnan(zp) or np.isnan(zc):
                continue
            if zp > cfg.threshold and zc > cfg.threshold:
                pending, signal_idx = "LONG", i
            elif zp < -cfg.threshold and zc < -cfg.threshold:
                pending, signal_idx = "SHORT", i
    return trades


@pytest.mark.parametrize("seed", range(12))
def test_fast_simulate_matches_the_minute_loop(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    obi = np.tanh(np.cumsum(rng.normal(0, 0.2, n)) * 0.1 + rng.normal(0, 0.3, n))
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    mid[rng.random(n) < 0.03] = np.nan                       # missing minutes
    mid[:5] = np.nan
    mid[rng.random(n) < 0.005] = 0.0                         # invalid prices
    grid = pd.DataFrame({"ts_ms": np.arange(n) * 60_000, "mid": mid, "obi_ntl": obi})
    cooldown = [0, 1, 10, 60][seed % 4]
    cfg = tp_multifold.Config(threshold=[1.0, 1.5, 2.0][seed % 3], hold_min=[5, 30, 240][seed % 3],
                              cooldown_min=cooldown)
    for stop, tp in ((-1.0, None), (None, 0.3), (-0.5, 0.5), (None, None)):
        fast = tp_multifold.simulate(grid, cfg, stop, tp)
        ref = _simulate_reference(grid, cfg, stop, tp)
        assert len(fast) == len(ref) and len(ref) > 0
        for a, b in zip(fast, ref):
            assert a == b


def test_multi_gate_wide_matches_pandas_pivot():
    rng = np.random.default_rng(0)
    book = pd.DataFrame({"bucket": np.repeat(np.arange(50), 3), "venue": np.tile(["okx", "binance", "hyperliquid"], 50),
                         "obi_ntl": rng.normal(size=150)}).sample(frac=0.8, random_state=1)
    pd.testing.assert_frame_equal(multi_gate._wide(book, "obi_ntl"),
                                  book.pivot(index="bucket", columns="venue", values="obi_ntl"),
                                  check_names=False)
