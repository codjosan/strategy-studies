#!/usr/bin/env python3
"""Correlation / cointegration study around the top-5 dip-buy strategies.

Strategies: leaderboard ranks 1-5 (multi-gate ``vwap_dev_buy``; ETH x4, BTC #5)
and the locked ``vol_norm_dip`` (3 sd below VWAP 10 min, 2 bp limit, hold 75 min,
6 coins).

Part 1, structure of the data (18 Jul - 27 Sep 2026, HL mids):
  * cross-venue: Engle-Granger cointegration of log HL mid vs log CEX median mid, per
    coin, and the half-life of the basis
  * cross-asset: Engle-Granger for all 15 coin pairs (15-min log prices), Johansen rank
    of the 6 coins, pair stability in the two halves, 5-min return correlations

Part 2, does it reinforce the signal? For every trade, at the signal tick and from
trailing data only:
  * basis_dev       HL vs CEX basis minus its trailing 1 h mean, bps (< 0: HL-only dip)
  * coint_z         z-score of the cointegration residual of the coin vs its partner
                    (BTC for alts, ETH for BTC), rolling OLS on the trailing 3 days of
                    1-min log prices (< 0: coin cheap vs partner)
  * market_15m      mean 15-min return of the other five coins, bps (< 0: market-wide dip)
  * idio_15m        own 15-min return minus market_15m, bps
  * corr_6h         trailing 6 h correlation of 1-min returns, coin vs partner
Each feature vs the trade's net return: Spearman rho, tercile means, and the same split
in each half of the period (a reinforcement should hold in both).

Part 3: correlation of the strategies' daily P&L.

Usage:
    PYTHONPATH=. python scripts/cointegration_study.py
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.tsa.stattools import coint  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import vwap_dev_sweep as vs  # noqa: E402
from hlmicro.validate import candidates as cd  # noqa: E402

COINS = ["BTC", "ETH", "SOL", "HYPE", "DOGE", "LINK"]
PARTNER = {"BTC": "ETH", "ETH": "BTC", "SOL": "BTC", "HYPE": "BTC", "DOGE": "BTC", "LINK": "BTC"}  # fixed a priori
START, END, TAPE_START, SPLIT = "20260718", "20260927", "20260825", "20260911"
TOP5 = [("#1 ETH passive 0.5bp 30m", "ETH", 1800, 0.5), ("#2 ETH passive 1bp 30m", "ETH", 1800, 1.0),
        ("#3 ETH passive 0.5bp 10m", "ETH", 600, 0.5), ("#4 ETH passive 0bp 30m", "ETH", 1800, 0.0),
        ("#5 BTC taker 10m", "BTC", 600, "taker")]
LOCKED = ("locked vol_norm_dip (6 coins)", 3.0, 75, 2.0, 10)
FEATURES = {"basis_dev": "HL vs CEX basis minus 1 h mean (bps); < 0 = HL-only dip",
            "coint_z": "cointegration residual z vs partner (trailing 3 d); < 0 = coin cheap vs partner",
            "market_15m": "mean 15-min return of the other 5 coins (bps); < 0 = market-wide dip",
            "idio_15m": "own 15-min return minus market_15m (bps); < 0 = coin-specific drop",
            "corr_6h": "trailing 6 h correlation of 1-min returns with the partner"}


def ffill(x: np.ndarray, limit: int) -> np.ndarray:
    return pd.Series(x).ffill(limit=limit).to_numpy()


def half_life(resid: np.ndarray) -> float:
    """Half-life (in samples) from an AR(1) fit of the residual."""
    r = resid[np.isfinite(resid)]
    x, y = r[:-1] - r[:-1].mean(), r[1:] - r[1:].mean()
    phi = (x @ y) / (x @ x)
    return np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--out", type=Path, default=Path("results/cointegration_study"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grids = {c: cd.build_grid(args.capture_dir, c, START, END) for c in COINS}
    n = grids["BTC"].n
    t0 = grids["BTC"].t0_ms
    hl = {c: ffill(grids[c].mid["hyperliquid"], 12) for c in COINS}                # 5 s, gaps up to 1 min
    cex = {c: ffill(np.nanmedian(np.vstack([grids[c].mid[v] for v in cd.CEX]), axis=0), 12) for c in COINS}
    minute = np.arange(0, n, 12)
    logm = {c: np.log(hl[c][minute]) for c in COINS}                                # 1-min log HL mids

    # ----------------------------------------------------------------- part 1: structure
    venue_rows = []
    for c in COINS:
        sub = np.arange(0, n, 12)                                                   # 1-min samples
        y, x = np.log(hl[c][sub]), np.log(cex[c][sub])
        ok = np.isfinite(y) & np.isfinite(x)
        tstat, p, _ = coint(y[ok], x[ok])
        beta = np.polyfit(x[ok], y[ok], 1)[0]
        basis5 = (hl[c] / cex[c] - 1) * 1e4
        hl_ticks = half_life(basis5[np.isfinite(basis5)][-200_000:])
        venue_rows.append({"coin": c, "EG t-stat": tstat, "p": p, "beta": beta,
                           "basis mean bps": np.nanmean(basis5), "basis sd bps": np.nanstd(basis5),
                           "half-life": f"{hl_ticks * 5:.0f} s" if np.isfinite(hl_ticks) else "none"})
    venue = pd.DataFrame(venue_rows)

    q15 = np.arange(0, n, 180)
    log15 = {c: np.log(hl[c][q15]) for c in COINS}
    split_q = (int(pd.Timestamp(SPLIT, tz="UTC").timestamp() * 1000) - t0) // (180 * 5000)
    mid_q = len(q15) // 2
    pair_rows = []
    for a, b in itertools.combinations(COINS, 2):
        ya, yb = log15[a], log15[b]
        ok = np.isfinite(ya) & np.isfinite(yb)
        tstat, p, _ = coint(ya[ok], yb[ok])
        beta, alpha = np.polyfit(yb[ok], ya[ok], 1)
        resid = ya - (alpha + beta * yb)
        p_h = []
        for sl in (slice(0, mid_q), slice(mid_q, None)):
            okh = ok.copy(); okh[:] = False; okh[sl] = ok[sl]
            p_h.append(coint(ya[okh], yb[okh])[1])
        pair_rows.append({"pair": f"{a}/{b}", "EG t-stat": tstat, "p": p, "beta": beta,
                          "half-life h": half_life(resid) * 0.25, "p 1st half": p_h[0], "p 2nd half": p_h[1],
                          "cointegrated (p<0.05) in both halves": "yes" if max(p_h) < 0.05 else "no"})
    pairs_tbl = pd.DataFrame(pair_rows).sort_values("p")
    mat = np.column_stack([log15[c] for c in COINS])
    mat = mat[np.isfinite(mat).all(axis=1)]
    jo = coint_johansen(mat, det_order=0, k_ar_diff=1)
    rank = int(sum(jo.lr1[i] > jo.cvt[i, 1] for i in range(len(COINS))))    # trace test, 95%
    johansen = {"trace stats": " ".join(f"{v:.1f}" for v in jo.lr1),
                "95% critical": " ".join(f"{v:.1f}" for v in jo.cvt[:, 1]),
                "cointegration rank (95%)": rank}
    r5 = pd.DataFrame({c: np.diff(np.log(hl[c][np.arange(0, n, 60)])) for c in COINS})
    corr = r5.corr()

    # ----------------------------------------------------------------- part 2: features per trade
    one = {c: np.r_[np.nan, np.diff(logm[c])] for c in COINS}                         # 1-min log returns

    def rolling_coint_z(c):
        """Residual z of log c on log partner, trailing 3 days of 1-min data (per minute)."""
        y, x = pd.Series(logm[c]), pd.Series(logm[PARTNER[c]])
        w = 3 * 1440
        mx, my = x.rolling(w, min_periods=w // 2).mean(), y.rolling(w, min_periods=w // 2).mean()
        cov = (x * y).rolling(w, min_periods=w // 2).mean() - mx * my
        var = (x * x).rolling(w, min_periods=w // 2).mean() - mx * mx
        beta = cov / var
        resid = y - (my + beta * (x - mx))
        sd = resid.rolling(w, min_periods=w // 2).std()
        return (resid / sd).to_numpy()

    cz = {c: rolling_coint_z(c) for c in COINS}
    corr6 = {c: pd.Series(one[c]).rolling(360, min_periods=180).corr(pd.Series(one[PARTNER[c]])).to_numpy()
             for c in COINS}
    basis = {}
    for c in COINS:
        bs = (grids[c].mid["hyperliquid"] / np.nanmedian(np.vstack([grids[c].mid[v] for v in cd.CEX]), axis=0) - 1) * 1e4
        basis[c] = bs - pd.Series(bs).rolling(720, min_periods=360).mean().to_numpy()
    r15 = {c: cd._ret(hl[c], 180) for c in COINS}

    def features(c, sig):
        m = sig // 12                                                               # minute of the signal tick
        other = [o for o in COINS if o != c]
        mkt = np.nanmean(np.vstack([r15[o][sig] for o in other]), axis=0)
        shared = cd.context_features(grids, c, sig)           # the definitions used by the forward test
        assert np.allclose(shared.market_15m, mkt, equal_nan=True) and np.allclose(shared.coint_z, cz[c][m], equal_nan=True)
        return pd.DataFrame({"basis_dev": basis[c][sig], "coint_z": cz[c][m], "market_15m": mkt,
                             "idio_15m": r15[c][sig] - mkt, "corr_6h": corr6[c][m]})

    strat_trades = {}
    tape_grid = {}
    for label, c, hold, ex in TOP5:
        if c not in tape_grid:
            tape_grid[c] = vs.features(args.capture_dir, c, TAPE_START, END)
        g = tape_grid[c]
        a = {"mid": g["mid"].to_numpy(np.float64), "lo": g["hl_px_min"].to_numpy(np.float64),
             "hi": g["hl_px_max"].to_numpy(np.float64)}
        e, x = vs.simulate(a, vs.signal_ticks(g, 300, 20, 0.0, 0.0), hold // 5, ex)
        ts = g["tick_ts"].to_numpy()[e] * 5000
        sig = (ts - t0) // 5000 - 1
        strat_trades[label] = pd.concat([pd.DataFrame({"coin": c, "ts": ts, "net": x}), features(c, sig)], axis=1)
    parts = []
    name, z, hold, k, vw = LOCKED
    for c in COINS:
        g = cd.build_grid(args.capture_dir, c, TAPE_START, END)
        rows = cd.vol_norm_dip(g, z, hold, k, vwap_min=vw)
        e = np.array([r[0] for r in rows], dtype=np.int64)
        ts = g.ts_ms(e)
        sig = (ts - t0) // 5000 - 1
        parts.append(pd.concat([pd.DataFrame({"coin": c, "ts": ts, "net": [r[1] for r in rows]}), features(c, sig)], axis=1))
    strat_trades[name] = pd.concat(parts, ignore_index=True)
    expect = {"#1 ETH passive 0.5bp 30m": (73, 7.57), "#5 BTC taker 10m": (33, 6.22), name: (139, 24.09)}
    for label, (n_e, m_e) in expect.items():
        t = strat_trades[label]
        assert len(t) == n_e and abs(t.net.mean() - m_e) < 0.01, f"{label}: {len(t)} {t.net.mean():.2f}"
    print("check: trades reproduce the leaderboard / validation numbers")

    split_ms = int(pd.Timestamp(SPLIT, tz="UTC").timestamp() * 1000)
    rows = []
    for label, t in strat_trades.items():
        for f in FEATURES:
            d = t[["net", f, "ts"]].dropna()
            if len(d) < 10:
                continue
            rho, p = stats.spearmanr(d[f], d.net)
            lo, hi = d[d[f] <= d[f].quantile(1 / 3)], d[d[f] >= d[f].quantile(2 / 3)]
            med = d[f].median()
            halves = []
            for part in (d[d.ts < split_ms], d[d.ts >= split_ms]):
                below, above = part[part[f] < med].net.mean(), part[part[f] >= med].net.mean()
                halves.append(below - above)
            rows.append({"strategy": label, "feature": f, "n": len(d), "spearman rho": rho, "p": p,
                         "low third mean": lo.net.mean(), "high third mean": hi.net.mean(),
                         "low - high": lo.net.mean() - hi.net.mean(),
                         "below-median minus above, 1st half": halves[0],
                         "same, 2nd half": halves[1],
                         "same sign both halves": "yes" if np.sign(halves[0]) == np.sign(halves[1]) else "no"})
    reinf = pd.DataFrame(rows)

    # filters with thresholds fixed at 0 (not tuned): does keeping only the reinforced trades help in each half?
    filt_rows = []
    filters = {"market_15m < 0 (market-wide dip)": lambda d: d.market_15m < 0,
               "coint_z < 0 (cheap vs partner)": lambda d: d.coint_z < 0,
               "both": lambda d: (d.market_15m < 0) & (d.coint_z < 0)}
    for label, t in strat_trades.items():
        d = t.dropna(subset=["market_15m", "coint_z"])
        for fname, f in {"no filter": lambda d: np.ones(len(d), bool), **filters}.items():
            keep = np.asarray(f(d))
            row = {"strategy": label, "filter": fname}
            for part, m in (("all", np.ones(len(d), bool)), ("1st half", (d.ts < split_ms).to_numpy()),
                            ("2nd half", (d.ts >= split_ms).to_numpy())):
                x = d.net.to_numpy()[keep & m]
                row[f"{part} n"] = len(x)
                row[f"{part} mean"] = x.mean() if len(x) else np.nan
                if part == "all":
                    row["all t"] = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan
            filt_rows.append(row)
    filt = pd.DataFrame(filt_rows)
    pd.concat([t.assign(strategy=label) for label, t in strat_trades.items()]).to_csv(args.out / "trades_features.csv", index=False)

    # ----------------------------------------------------------------- part 3: strategy P&L correlation
    daily = pd.DataFrame({label: t.assign(day=pd.to_datetime(t.ts, unit="ms").dt.date).groupby("day").net.sum()
                          for label, t in strat_trades.items()}).fillna(0.0)
    pnl_corr = daily.corr()

    # ----------------------------------------------------------------- report
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, m, title in ((axes[0], corr, "5-min return correlation (HL mids, 72 days)"),
                         (axes[1], pnl_corr, "daily P&L correlation of the strategies")):
        ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        labels = [s.split(" (")[0][:14] for s in m.columns]
        ax.set_xticks(range(len(m)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(m)), labels, fontsize=7)
        for (r, c_), v in np.ndenumerate(m.to_numpy()):
            ax.text(c_, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=9)
    chart_corr = b64(fig)
    show = [TOP5[0][0], TOP5[4][0], name]
    fig, axes = plt.subplots(len(show), len(FEATURES), figsize=(17, 3 * len(show)))
    for i, label in enumerate(show):
        t = strat_trades[label]
        for j, f in enumerate(FEATURES):
            ax = axes[i, j]
            ax.scatter(t[f], t.net, s=10, alpha=0.6)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{label[:26]}\n{f}", fontsize=8)
            ax.tick_params(labelsize=7)
    chart_scatter = b64(fig)
    for n_, c_ in (("correlations", chart_corr), ("features", chart_scatter)):
        (args.out / f"{n_}.png").write_bytes(base64.b64decode(c_))
    for name_, df in (("venue_cointegration", venue), ("pair_cointegration", pairs_tbl), ("reinforcement", reinf),
                      ("return_correlation", corr), ("strategy_pnl_correlation", pnl_corr)):
        df.to_csv(args.out / f"{name_}.csv", index=name_.endswith("correlation"))

    fmt = lambda v: f"{v:+.3f}" if isinstance(v, (float, np.floating)) else str(v)
    css = ("body{font:14px system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:12.5px;margin:8px 0 20px}td,th{border:1px solid #ddd;"
           "padding:4px 8px;text-align:right}th{background:#f4f4f4}td:first-child{text-align:left}"
           "h2{margin-top:28px;border-bottom:1px solid #eee}img{max-width:100%}")
    feat_desc = pd.Series(FEATURES).to_frame("meaning (all from data before the signal)")
    html = [f"<html><head><meta charset='utf-8'><title>Correlation / cointegration study</title><style>{css}</style></head><body>",
            "<h1>Correlation / cointegration study: top-5 strategies and locked vol_norm_dip</h1>",
            "<h2>Part 1a: HL vs CEX, same coin (Engle-Granger, 1-min log mids, 72 days)</h2>", venue.map(fmt).to_html(index=False),
            "<h2>Part 1b: coin pairs (Engle-Granger, 15-min log mids)</h2>", pairs_tbl.map(fmt).to_html(index=False),
            "<h3>Johansen, 6 coins</h3>", pd.Series(johansen).to_frame("value").to_html(),
            f"<img src='data:image/png;base64,{chart_corr}'>",
            "<h2>Part 2: does it reinforce the signal?</h2>", feat_desc.to_html(),
            reinf.map(fmt).to_html(index=False), f"<img src='data:image/png;base64,{chart_scatter}'>",
            "<h3>Filters with thresholds fixed at 0 (not tuned): kept trades per half</h3>",
            filt.map(fmt).to_html(index=False),
            "<h2>Part 3: strategy P&L correlation (daily)</h2>", pnl_corr.map(fmt).to_html(), "</body></html>"]
    (args.out / "report.html").write_text("\n".join(html))
    pd.set_option("display.width", 260)
    print(venue.map(fmt).to_string(index=False), "\n")
    print(pairs_tbl.map(fmt).to_string(index=False), "\n")
    print(johansen, "\n")
    print(corr.round(2).to_string(), "\n")
    print(reinf.map(fmt).to_string(index=False), "\n")
    print(filt.map(fmt).to_string(index=False), "\n")
    print(pnl_corr.round(2).to_string())
    print(f"\nwrote {args.out}/report.html and csv files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
