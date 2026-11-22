import numpy as np
import pandas as pd
import pytest

from hlmicro.validate import multi_gate, run_all, tp_multifold


@pytest.fixture(scope="module")
def outcome(capture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("run_all")
    return run_all.run_all(capture_dir, out, config={"tp-multifold": {"hold_min": 30, "cooldown_min": 10}}), out


def test_feasibility_and_outputs(outcome):
    (plan, runs, table), out = outcome
    assert plan.symbols == ["BTC", "SOL"]
    ran = {s for s, (ok, _) in plan.feasible.items() if ok}
    assert ran == {"tp-multifold", "cex-hl-lag", "multi-gate", "funding-settlement", "candidates"}
    assert all(not plan.feasible[s][0] and plan.feasible[s][1] for s in set(run_all.STUDIES) - ran)
    assert runs and not any(r.error for r in runs)
    assert set(table["study"]) == ran
    assert (out / "reports").is_dir() and len(list((out / "reports").glob("*.txt"))) == len(runs)


def test_leaderboard_numbers_match_each_study(outcome, capture_dir):
    (_, _, table), _ = outcome
    # tp-multifold: pooled mean of the study, in % -> bps
    res = tp_multifold.run(tp_multifold.Config(capture_dir=capture_dir, symbol="BTC", hold_min=30, cooldown_min=10))
    stats, trades = res.pooled[(-1.0, 0.0)]
    row = table[(table["study"] == "tp-multifold") & (table["symbol"] == "BTC")
                & table["params"].str.contains("stop -1% tp none")].iloc[0]
    assert row["n"] == stats["n"] and row["mean_net_bps"] == pytest.approx(stats["mean"] * 100)
    # multi-gate: the study's own taker-fee row
    gate = multi_gate.run(multi_gate.Config(capture_dir=capture_dir, symbol="BTC"))
    own = gate.sweep[(gate.sweep["fees"] == "taker") & (gate.sweep["exit"] == "fixed")].iloc[0]
    row = table[(table["study"] == "multi-gate") & (table["symbol"] == "BTC") & (table["strategy"] == own["rule"])
                & table["params"].str.startswith(f"hold {own['hold_s']}s")].iloc[0]
    assert row["n"] == own["n"] and row["mean_net_bps"] == pytest.approx(own["mean_bps"], abs=1e-3)


def test_ranking_and_verdicts():
    rng = np.random.default_rng(0)
    ts = np.arange(400) * 3_600_000
    make = lambda name, x: run_all.Strategy("s", "BTC", name, "", "", ts[:len(x)], x)
    strategies = [make("edge", rng.normal(5, 10, 400)),              # t ~ 10
                  make("noise", rng.normal(0.3, 10, 400)),
                  make("loser", rng.normal(-3, 10, 400)),
                  make("few", rng.normal(50, 1, 10))]
    strategies += [make(f"n{i}", rng.normal(0, 10, 400)) for i in range(40)]
    table = run_all.leaderboard(strategies, n_folds=4, min_trades=30)
    top = table.iloc[0]
    assert top["strategy"] == "edge" and top["verdict"] == "significant" and top["rank"] == 1
    assert table.set_index("strategy").loc["loser", "verdict"] == "negative"
    few = table.set_index("strategy").loc["few"]
    assert few["rank"] == 0 and few["verdict"] == "n<30"                # too few trades: not ranked
    assert (table["verdict"] == "significant").sum() == 1                # noise does not survive correction
