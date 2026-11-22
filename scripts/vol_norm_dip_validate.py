#!/usr/bin/env python3
"""Out-of-sample validation of the vol_norm_dip candidate (docs/NEW_STRATEGIES.md, step 1).

Pre-declared grid (fixed before this run):
    dip K in {2.5, 3, 3.5, 4} standard deviations, hold in {45, 60, 75, 90} min,
    VWAP window in {2, 5, 10} min, passive limit in {1, 2} bp; all 6 coins.
Trade-tape period 25 Aug - 27 Sep 2026, split in two equal halves at 11 Sep.

Checks:
1. the first-run setting (3 sd, 60 min, 1 bp, VWAP 5 min) reproduces the candidates leaderboard
2. selection on one half, judged on the other (both directions): per coin, and one common
   setting for all coins
3. robustness over the grid (share of settings positive in both halves)
4. for the common settings: day-bootstrap CI, outlier dependence, random-entry placebo,
   strict fills (one price increment through the limit), and a liquidation diagnostic
5. the setting locked for the forward test (rule: best worst-half pooled mean, n >= 60 per half)

Usage:
    PYTHONPATH=. python scripts/vol_norm_dip_validate.py
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from hlmicro.parse import discover, load_liquidations  # noqa: E402
from hlmicro.validate import candidates as cd  # noqa: E402

COINS = ["BTC", "ETH", "SOL", "HYPE", "DOGE", "LINK"]
GRID = {"z_min": [2.5, 3.0, 3.5, 4.0], "hold_min": [45, 60, 75, 90], "vwap_min": [2, 5, 10], "k": [1.0, 2.0]}
START, END, SPLIT = "20260825", "20260927", "20260911"
FIRST_RUN = {"z_min": 3.0, "hold_min": 60, "vwap_min": 5, "k": 1.0}
FIRST_RUN_EXPECT = {"ETH": (44, 19.18), "LINK": (29, 21.18), "HYPE": (32, 16.17)}


def label(p):
    return f"{p['z_min']:g} sd, hold {p['hold_min']}m, VWAP {p['vwap_min']}m, limit {p['k']:g}bp"


def stat(x):
    n = len(x)
    if n < 3:
        return {"n": n, "mean": np.nan, "t": np.nan, "win": np.nan}
    sd = x.std(ddof=1)
    return {"n": n, "mean": x.mean(), "t": x.mean() / sd * np.sqrt(n) if sd > 0 else np.nan, "win": (x > 0).mean()}


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--out", type=Path, default=Path("results/vol_norm_dip_validation"))
    ap.add_argument("--placebo-draws", type=int, default=500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    grids = {c: cd.build_grid(args.capture_dir, c, START, END) for c in COINS}
    split_tick = (int(pd.Timestamp(SPLIT, tz="UTC").timestamp() * 1000) - grids["ETH"].t0_ms) // cd.TICK_MS

    # liquidation notional per tick (diagnostic only)
    liq_files = discover(args.capture_dir, "hl_liquidations", START, END)
    liq5 = {}
    for c in COINS:
        arr = np.zeros(grids[c].n)
        if liq_files:
            lq = load_liquidations(liq_files, c)
            i = (lq["ts_ms"].to_numpy(np.int64) - grids[c].t0_ms) // cd.TICK_MS
            ok = (i >= 0) & (i < len(arr))
            np.add.at(arr, i[ok], lq["notional"].to_numpy()[ok])
        liq5[c] = cd._rolling(arr, 60, "sum", 1)          # trailing 5 minutes, USD

    def trades(c, p, through=0, idx=None):
        g = grids[c]
        if idx is None:
            idx = cd.vol_norm_signal(g, p["z_min"], p["vwap_min"])
        rows = cd.passive(g.mid["hyperliquid"], g.tape["px_min"], g.tape["px_max"], idx, np.ones(len(idx)),
                          p["hold_min"] * cd.TICKS_PER_MIN, p["k"], through)
        return np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows])

    # 1. regression against the first candidates run
    for c, (n, mean) in FIRST_RUN_EXPECT.items():
        e, x = trades(c, FIRST_RUN)
        assert len(x) == n and abs(x.mean() - mean) < 0.01, f"{c}: {len(x)} / {x.mean():.2f}, expected {n} / {mean}"
    print("check: the first-run setting reproduces the candidates leaderboard")

    # 2. the grid
    rows, store = [], {}
    for values in itertools.product(*GRID.values()):
        p = dict(zip(GRID, values))
        pooled = {"A": [], "B": [], "all": []}
        for c in COINS:
            e, x = trades(c, p)
            store[(label(p), c)] = (e, x)
            a, b = x[e < split_tick], x[e >= split_tick]
            r = {"setting": label(p), **p, "coin": c}
            for part, v in (("A", a), ("B", b), ("all", x)):
                r.update({f"{part}_{k}": val for k, val in stat(v).items()})
                pooled[part].append(v)
            rows.append(r)
        r = {"setting": label(p), **p, "coin": "ALL"}
        for part, vs in pooled.items():
            r.update({f"{part}_{k}": val for k, val in stat(np.concatenate(vs)).items()})
        rows.append(r)
    grid = pd.DataFrame(rows)
    grid.to_csv(args.out / "grid.csv", index=False)
    pooled = grid[grid.coin == "ALL"].copy()

    # selection on one half, judged on the other
    sel = []
    for trn, tst in (("A", "B"), ("B", "A")):
        ok = pooled[pooled[f"{trn}_n"] >= 60]
        pick = ok.sort_values(f"{trn}_mean", ascending=False).iloc[0]
        per = grid[(grid.setting == pick.setting) & (grid.coin != "ALL")]
        sel.append({"selection": f"common setting, chosen on {'first' if trn == 'A' else 'second'} half",
                    "setting": pick.setting, "train mean": pick[f"{trn}_mean"], "train n": pick[f"{trn}_n"],
                    "test mean": pick[f"{tst}_mean"], "test n": pick[f"{tst}_n"], "test t": pick[f"{tst}_t"],
                    "coins positive on test": f"{int((per[f'{tst}_mean'] > 0).sum())}/6",
                    "all settings, test mean (avg)": ok[f"{tst}_mean"].mean()})
        coin_test = []
        for c in COINS:
            okc = grid[(grid.coin == c) & (grid[f"{trn}_n"] >= 10)]
            if okc.empty:
                continue
            pc = okc.sort_values(f"{trn}_mean", ascending=False).iloc[0]
            coin_test.append((c, pc.setting, pc[f"{trn}_mean"], pc[f"{tst}_mean"], pc[f"{tst}_n"]))
        ct = pd.DataFrame(coin_test, columns=["coin", "setting", "train", "test", "test_n"])
        sel.append({"selection": f"best per coin, chosen on {'first' if trn == 'A' else 'second'} half",
                    "setting": "(differs by coin)", "train mean": ct.train.mean(), "train n": np.nan,
                    "test mean": np.average(ct.test, weights=ct.test_n), "test n": ct.test_n.sum(), "test t": np.nan,
                    "coins positive on test": f"{int((ct.test > 0).sum())}/{len(ct)}",
                    "all settings, test mean (avg)": np.nan})
    sel = pd.DataFrame(sel)

    # 3. robustness
    both = pooled[(pooled.A_n >= 60) & (pooled.B_n >= 60)]
    robust = {
        "settings (pooled over coins)": len(pooled),
        "positive in first half": f"{(pooled.A_mean > 0).mean():.0%}",
        "positive in second half": f"{(pooled.B_mean > 0).mean():.0%}",
        "positive in both halves (n>=60 each)": f"{((both.A_mean > 0) & (both.B_mean > 0)).sum()}/{len(both)}",
        "rank correlation first vs second half": f"{pooled.A_mean.corr(pooled.B_mean, method='spearman'):+.2f}",
    }
    heat = pooled.pivot_table(index="z_min", columns="hold_min", values="all_mean", aggfunc="mean")

    # 5. the locked setting: best worst-half pooled mean with n >= 60 in each half
    both = both.assign(worst=np.minimum(both.A_mean, both.B_mean))
    locked_row = both.sort_values("worst", ascending=False).iloc[0]
    locked = {k: (float(locked_row[k]) if k in ("z_min", "k") else int(locked_row[k])) for k in GRID}

    # 4. deep checks for first-run, both selected common settings, and the locked one
    focus = {"first-run setting": FIRST_RUN, "locked for forward test": locked}
    for s in sel.itertuples():
        if s.selection.startswith("common"):
            p = pooled[pooled.setting == s.setting].iloc[0]
            focus[s.selection] = {k: (float(p[k]) if k in ("z_min", "k") else int(p[k])) for k in GRID}
    deep, placebo_charts = [], {}
    for name, p in focus.items():
        xs, es, days, liqflag = [], [], [], []
        strict = []
        for c in COINS:
            e, x = trades(c, p)
            xs.append(x); es.append(e)
            days.append(pd.to_datetime(grids[c].ts_ms(e), unit="ms").strftime("%m%d").to_numpy())
            lq = liq5[c][e - 1] if len(e) else np.array([])
            thr = np.nanpercentile(liq5[c][liq5[c] > 0], 90) if (liq5[c] > 0).any() else np.inf
            liqflag.append(lq >= thr)
            strict.append(trades(c, p, through=1)[1])
        x = np.concatenate(xs); day = np.concatenate(days); lf = np.concatenate(liqflag); xs_strict = np.concatenate(strict)
        ud = np.unique(day)
        boot = [np.concatenate([x[day == d] for d in rng.choice(ud, len(ud))]).mean() for _ in range(3000)]
        # placebo: random entry ticks per coin, same execution, same number of trades per coin
        means = []
        valid = {c: np.flatnonzero(np.isfinite(grids[c].tape["px_min"]) & (grids[c].mid["hyperliquid"] > 0))
                 for c in COINS}
        for _ in range(args.placebo_draws):
            fake = []
            for c, xc in zip(COINS, xs):
                if not len(xc):
                    continue
                v = valid[c][:-(p["hold_min"] * cd.TICKS_PER_MIN + 3)]
                _, xf = trades(c, p, idx=np.sort(rng.choice(v, min(len(v), len(xc) * 40), replace=False)))
                fake.append(rng.choice(xf, len(xc), replace=len(xf) < len(xc)))
            means.append(np.concatenate(fake).mean())
        means = np.array(means)
        placebo_charts[name] = (means, x.mean())
        sd = x.std(ddof=1)
        deep.append({
            "setting": f"{name}: {label(p)}", "trades": len(x), "mean bps": x.mean(), "median": np.median(x),
            "t": x.mean() / sd * np.sqrt(len(x)), "p (mean>0)": stats.t.sf(x.mean() / sd * np.sqrt(len(x)), len(x) - 1),
            "95% CI (day bootstrap)": f"{np.percentile(boot, 2.5):+.1f} .. {np.percentile(boot, 97.5):+.1f}",
            "win": f"{(x > 0).mean():.0%}", "mean without best 3": np.sort(x)[:-3].mean(),
            "first / second half": f"{np.concatenate([xc[ec < split_tick] for xc, ec in zip(xs, es)]).mean():+.1f} / "
                                   f"{np.concatenate([xc[ec >= split_tick] for xc, ec in zip(xs, es)]).mean():+.1f}",
            "coins positive": f"{sum(xc.mean() > 0 for xc in xs if len(xc))}/{sum(len(xc) > 0 for xc in xs)}",
            "per coin (n: mean)": " ".join(f"{c} {len(xc)}:{xc.mean():+.0f}" for c, xc in zip(COINS, xs) if len(xc)),
            "strict fills": f"{xs_strict.mean():+.1f} (n {len(xs_strict)})",
            "placebo random entries": f"{means.mean():+.1f} (p = {(means >= x.mean()).mean():.3f})",
            "dips during liquidation burst / other": f"{x[lf].mean():+.1f} (n {lf.sum()}) / {x[~lf].mean():+.1f} (n {(~lf).sum()})",
        })
    deep = pd.DataFrame(deep)

    # lock file for the forward test
    lock = {"locked_on": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), "data_used_up_to": END,
            "vol_norm_dip": {**locked, "coins": COINS, "rule": "best worst-half pooled mean (n>=60 per half) "
                             "of the pre-declared grid"},
            "momentum": {"coins": ["LINK", "ETH", "BTC"], "lookback_h": 1, "k_sigma": 2, "hold_h": 1, "mode": "follow"},
            "vol_norm_dip_context_filter": {"declared_on": "2026-11-22", "base": "vol_norm_dip (locked setting)", "filter": "market_15m < 0 and coint_z < 0 at the signal tick (hlmicro.validate.candidates.context_features)", "why": "correlation/cointegration study: both reinforced the locked rule in each half of 25 Aug - 27 Sep (found on that data, so secondary to the primary rule)"},
            "pass_criteria": "pooled mean net > 0 with one-sided p < 0.05 on data strictly after data_used_up_to, "
                             "at least 30 trades; no parameter changes"}
    (args.out / "forward_lock.json").write_text(json.dumps(lock, indent=2))

    # charts
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(heat.to_numpy(), cmap="RdYlGn", vmin=-np.nanmax(np.abs(heat.to_numpy())),
                   vmax=np.nanmax(np.abs(heat.to_numpy())), aspect="auto")
    ax.set_xticks(range(len(heat.columns)), [f"{c}m" for c in heat.columns])
    ax.set_yticks(range(len(heat.index)), [f"{i:g} sd" for i in heat.index])
    for (r, c), v in np.ndenumerate(heat.to_numpy()):
        ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    ax.set(title="mean net bps, all coins pooled (avg over VWAP window and limit)", xlabel="hold", ylabel="dip")
    chart_heat = b64(fig)
    fig, axes = plt.subplots(1, len(placebo_charts), figsize=(4 * len(placebo_charts), 3.2))
    for ax, (name, (means, real)) in zip(np.atleast_1d(axes), placebo_charts.items()):
        ax.hist(means, bins=40, color="gray", alpha=0.7, label="random entries")
        ax.axvline(real, color="red", lw=2, label="strategy")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    chart_placebo = b64(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, p in focus.items():
        es_all, xs_all = [], []
        for c in COINS:
            e, x = trades(c, p)
            es_all.append(grids[c].ts_ms(e)); xs_all.append(x)
        order = np.argsort(np.concatenate(es_all))
        ax.plot(pd.to_datetime(np.concatenate(es_all)[order], unit="ms"), np.cumsum(np.concatenate(xs_all)[order]),
                label=f"{name}: {label(p)}")
    ax.axvline(pd.Timestamp(SPLIT), color="gray", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(title="cumulative net bps, all coins pooled", ylabel="bps")
    ax.legend(fontsize=7)
    chart_cum = b64(fig)
    for n_, c_ in (("heat", chart_heat), ("placebo", chart_placebo), ("cumulative", chart_cum)):
        (args.out / f"{n_}.png").write_bytes(base64.b64decode(c_))

    fmt = lambda v: f"{v:+.2f}" if isinstance(v, (float, np.floating)) else str(v)
    css = ("body{font:14px system-ui,sans-serif;max-width:1250px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:12.5px;margin:8px 0 20px}td,th{border:1px solid #ddd;"
           "padding:4px 8px;text-align:right}th{background:#f4f4f4}td:first-child{text-align:left}"
           "h2{margin-top:28px;border-bottom:1px solid #eee}img{max-width:100%}")
    html = [f"<html><head><meta charset='utf-8'><title>vol_norm_dip validation</title><style>{css}</style></head><body>",
            "<h1>vol_norm_dip: out-of-sample validation</h1>",
            f"<p>Pre-declared grid of {len(pooled)} settings x 6 coins; trade-tape period {START}-{END}, halves split at "
            f"{SPLIT}. Passive fills simulated (maker in, maker/taker out). Net bps per trade.</p>",
            f"<img src='data:image/png;base64,{chart_cum}'>",
            "<h2>Selection on one half, judged on the other</h2>", sel.map(fmt).to_html(index=False),
            "<h2>Robustness over the grid</h2>", pd.Series(robust).to_frame("value").to_html(),
            f"<img src='data:image/png;base64,{chart_heat}'>",
            "<h2>Detailed checks</h2>", deep.map(fmt).T.to_html(header=False),
            f"<img src='data:image/png;base64,{chart_placebo}'>",
            "<h2>Locked for the forward test</h2>", f"<pre>{json.dumps(lock, indent=2)}</pre>",
            "</body></html>"]
    (args.out / "report.html").write_text("\n".join(html))
    pd.set_option("display.width", 250)
    print(sel.map(fmt).to_string(index=False))
    print()
    print(pd.Series(robust).to_string())
    print()
    print(heat.round(1).to_string())
    print()
    print(deep.map(fmt).T.to_string(header=False))
    print(f"\nlocked: {label(locked)}\nwrote {args.out}/report.html, grid.csv, forward_lock.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
