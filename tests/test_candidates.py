import numpy as np
import pandas as pd
import pytest

from hlmicro.validate import candidates as cd

DAYS = 6
N = DAYS * 17_280


def make_grid(symbol="ETH", seed=0, lag=2, n=N):
    """Random-walk market: CEX venues lead, HL follows `lag` ticks later; tape with bursts."""
    rng = np.random.default_rng(seed)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, n) + np.where(rng.random(n) < 0.002, rng.normal(0, 5e-3, n), 0)))
    mid = {"binance": base.copy(), "okx": base * 1.0001, "bybit": base * 0.9999,
           "hyperliquid": np.r_[np.full(lag, base[0]), base[:-lag]]}
    obi = {v: np.tanh(rng.normal(0, 1, n)) for v in mid}
    count = rng.poisson(5, n).astype(float) + np.where(rng.random(n) < 0.01, 40, 0)
    side = np.clip(rng.normal(0, 0.5, n) + np.sign(np.r_[0, np.diff(mid["hyperliquid"])]) * 0.3, -1, 1)
    ntl = count * 1000.0
    hl = mid["hyperliquid"]
    tape = {"buy_ntl": ntl * (1 + side) / 2, "sell_ntl": ntl * (1 - side) / 2, "count": count,
            "sum_ntl": ntl, "sum_sz": ntl / hl, "px_min": hl * (1 - 3e-4), "px_max": hl * (1 + 3e-4)}
    return cd.Grid(symbol, 1_767_225_600_000, mid, obi, tape)


def prefix(g: cd.Grid, t: int) -> cd.Grid:
    cut = lambda d: {k: v[:t] for k, v in d.items()}
    return cd.Grid(g.symbol, g.t0_ms, cut(g.mid), cut(g.obi), cut(g.tape) if g.tape else None)


def signals(monkeypatch, c, g, g2, params):
    """Entry signals (tick, side) a candidate hands to its execution model."""
    seen = []

    def record(mid, idx, sides, *args, **kwargs):
        seen.extend(zip(np.asarray(idx).tolist(), np.asarray(sides).tolist()))
        return []
    for fn in ("taker_fixed", "taker_until", "passive"):
        monkeypatch.setattr(cd, fn, record if fn != "passive" else
                            (lambda mid, lo, hi, idx, sides, *a, **k: record(mid, idx, sides)))
    c.fn(g, **params)
    return seen


@pytest.mark.parametrize("name", [n for n in cd.CANDIDATES if n != "pairs"])
def test_signals_do_not_look_ahead(name, monkeypatch):
    """Signals computed on data cut at tick t equal the full-data signals before t."""
    c = cd.CANDIDATES[name]
    g = make_grid("ETH", 0)
    for params in list(cd._combos(c.grid))[:6]:
        full = signals(monkeypatch, c, g, None, params)
        for t in (int(N * 0.55), int(N * 0.8), N - 1000):
            cut = [sig for sig in signals(monkeypatch, c, prefix(g, t), None, params) if sig[0] < t - 1]
            assert cut == [sig for sig in full if sig[0] < t - 1], f"{name} {params}: signal uses future data"


def test_pair_signal_does_not_look_ahead():
    a, b = make_grid("ETH", 0).mid["hyperliquid"], make_grid("BTC", 5).mid["hyperliquid"]
    full = cd.pair_z(a, b, 6)
    for t in (int(N * 0.55), int(N * 0.8)):
        cut = cd.pair_z(a[:t], b[:t], 6)
        np.testing.assert_array_equal(cut[:-1], full[:len(cut) - 1])


@pytest.mark.parametrize("name", list(cd.CANDIDATES))
def test_every_candidate_trades_on_a_lagged_market(name):
    c = cd.CANDIDATES[name]
    g, g2 = make_grid("ETH", 0), make_grid("BTC", 5)
    total = sum(len(c.fn(g, g2, **p) if c.pair else c.fn(g, **p)) for p in cd._combos(c.grid))
    assert total > 0


def test_costs_on_a_flat_price():
    mid = np.full(1000, 100.0)
    trades = cd.taker_fixed(mid, np.array([10, 500]), np.array([1, -1]), 50)
    assert [round(x, 10) for _, x in trades] == [-cd.TAKER_RT, -cd.TAKER_RT]
    trades = cd.passive(mid, mid * 0.999, mid * 1.001, np.array([10]), np.array([1]), 50, 0.0)
    assert trades[0][1] == pytest.approx(-2 * cd.MAKER)              # maker in and out at the mid


def test_pairs_leg_arithmetic():
    n = 60 * 24 * 3 * 12
    rng = np.random.default_rng(0)
    a = 100.0 * (1 + rng.normal(0, 1e-5, n))
    b = np.full(n, 50.0)
    m = np.arange(0, n, 12)
    # A jumps 2% above its usual ratio for a few hours, then returns to normal
    bump = (m >= len(m) // 2) & (m < len(m) // 2 + 12 * 180)
    a[m[bump][0]:m[bump][-1] + 12] *= 1.02
    ga = cd.Grid("A", 0, {"hyperliquid": a, **{v: a for v in cd.CEX}}, {}, None)
    gb = cd.Grid("B", 0, {"hyperliquid": b, **{v: b for v in cd.CEX}}, {}, None)
    trades = cd.pairs(ga, gb, lookback_h=6, z_entry=2, max_hold_h=24)
    assert trades, "no trade on a 2% dislocation"
    # entered short A / long B after the jump (at 102 / 50); exited when A is back at 100
    assert trades[0][1] == pytest.approx((1 - 1 / 1.02) * 1e4 - 2 * cd.TAKER_RT, abs=0.5)


def test_run_and_leaderboard_on_synthetic_capture(capture_dir):
    strategies, _ = cd.run(capture_dir, ["BTC", "SOL"], ["basis_reversion", "momentum"], "20260105", "20260106")
    assert strategies and {s.name for s in strategies} <= {"basis_reversion", "momentum"}
    table = cd.run_all.leaderboard(strategies, 4, 10)
    assert {"first_half_mean", "second_half_mean"} <= set(cd.halves(strategies, table).columns)


def test_context_features_do_not_look_ahead():
    """Features at ticks just before a cut must not change when the data after the cut is removed."""
    grids = {c: make_grid(c, seed) for c, seed in (("ETH", 0), ("BTC", 5), ("SOL", 7))}
    for t in (int(N * 0.6), int(N * 0.85)):
        pts = t - np.array([400, 50, 13, 7, 3, 2, 1])      # coint_z needs about 3 days of history: t >> 51,840
        full = cd.context_features(grids, "ETH", pts)
        assert full.notna().all().all()
        cut = {c: prefix(g, t) for c, g in grids.items()}
        pd.testing.assert_frame_equal(cd.context_features(cut, "ETH", pts), full)
