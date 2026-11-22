#!/usr/bin/env python3
"""Deep-dive report on the multi-gate ``vwap_dev_buy`` strategies (ETH focus).

Reads the sweep written by ``scripts/vwap_dev_sweep.py`` and the capture data,
and writes ``REPORT.md``, ``report.html`` and charts to the sweep directory:

* statistics of the original top-5 and of the improved variants: confidence
  intervals (trade and day bootstrap), win rate with its interval, payoff,
  break-even win rate, outlier dependence, drawdown, weekly results
* out-of-sample selection test (choose on one half, score on the other)
* robustness: placebo (random entries), strict fills, neighbouring
  parameters, hold length, other coins

Usage:
    PYTHONPATH=. python scripts/vwap_dev_report.py
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import vwap_dev_sweep as vs  # noqa: E402

TICK_PX = {"ETH": 0.1, "BTC": 1.0, "SOL": 0.01, "HYPE": 0.001, "DOGE": 0.00001, "LINK": 0.001}

# (label, symbol, vwap window, dip, hold s, execution)
STRATS = [
    ("#1 original: ETH passive 0.5bp, 30m", "ETH", 300, 20, 1800, 0.5),
    ("#2 original: ETH passive 1bp, 30m", "ETH", 300, 20, 1800, 1.0),
    ("#3 original: ETH passive 0.5bp, 10m", "ETH", 300, 20, 600, 0.5),
    ("#4 original: ETH passive 0bp, 30m", "ETH", 300, 20, 1800, 0.0),
    ("#5 original: BTC taker, 10m", "BTC", 300, 20, 600, "taker"),
    ("IMPROVED A: ETH passive 1bp, 60m", "ETH", 300, 20, 3600, 1.0),
    ("IMPROVED B: ETH passive 2bp, 60m", "ETH", 300, 20, 3600, 2.0),
]


def arrays(g):
    return {"mid": g["mid"].to_numpy(np.float64), "lo": g["hl_px_min"].to_numpy(np.float64),
            "hi": g["hl_px_max"].to_numpy(np.float64)}


def trades(feats, sym, w, dev, hold, ex, strict=False):
    g = feats[sym]
    e, x = vs.simulate(arrays(g), vs.signal_ticks(g, w, dev, 0.0, 0.0), hold // vs.TICK_S, ex,
                       through_ticks=1 if strict else 0, tick_px=TICK_PX[sym])
    ts = pd.to_datetime(g["tick_ts"].to_numpy()[e] * vs.TICK_S * 1000, unit="ms", utc=True)
    return pd.DataFrame({"ts": ts, "net_bps": x})


def full_stats(t: pd.DataFrame, rng) -> dict:
    x = t["net_bps"].to_numpy()
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    day = t["ts"].dt.strftime("%m%d").to_numpy()
    days = np.unique(day)
    by_day = {d: x[day == d] for d in days}
    boot = [np.concatenate([by_day[d] for d in rng.choice(days, len(days))]).mean() for _ in range(5000)]
    w = int((x > 0).sum())
    ci_w = stats.binomtest(w, n).proportion_ci(0.95, method="wilson")
    wins, losses = x[x > 0], x[x <= 0]
    cum = np.cumsum(x)
    return {
        "trades": n, "trades/day": n / 34, "mean bps": x.mean(), "median bps": np.median(x),
        "95% CI (t)": f"{x.mean() - stats.t.ppf(.975, n - 1) * se:+.1f} .. {x.mean() + stats.t.ppf(.975, n - 1) * se:+.1f}",
        "95% CI (day bootstrap)": f"{np.percentile(boot, 2.5):+.1f} .. {np.percentile(boot, 97.5):+.1f}",
        "t": x.mean() / se, "p (mean > 0)": stats.t.sf(x.mean() / se, n - 1),
        "win rate": f"{100 * w / n:.0f}% ({100 * ci_w.low:.0f}-{100 * ci_w.high:.0f}%)",
        "p (win rate > 50%)": stats.binomtest(w, n, 0.5, alternative="greater").pvalue,
        "avg win / avg loss": f"{wins.mean():+.1f} / {losses.mean():+.1f}",
        "break-even win rate": f"{100 * -losses.mean() / (wins.mean() - losses.mean()):.0f}%",
        "profit factor": wins.sum() / -losses.sum(),
        "best / worst": f"{x.max():+.0f} / {x.min():+.0f}",
        "mean without best 3": np.sort(x)[:-3].mean(),
        "share of total from best 3": f"{100 * np.sort(x)[-3:].sum() / x.sum():.0f}%",
        "total bps": x.sum(), "max drawdown bps": (np.maximum.accumulate(cum) - cum).max(),
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/vwap_dev_sweep"))
    ap.add_argument("--split", default="20260914")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    out = args.sweep_dir
    feats = {s: vs.features(args.capture_dir, s, "20260825", "20260927") for s in TICK_PX}
    split = pd.Timestamp(args.split, tz="UTC")
    charts, md = {}, []

    # 1. statistics table
    tr = {label: trades(feats, s, w, d, h, ex) for label, s, w, d, h, ex in STRATS}
    table = pd.DataFrame({label: full_stats(t, rng) for label, t in tr.items()})
    for label, t in tr.items():
        a, b = t[t.ts < split].net_bps, t[t.ts >= split].net_bps
        table.loc["first half / second half mean", label] = f"{a.mean():+.1f} (n {len(a)}) / {b.mean():+.1f} (n {len(b)})"
        wk = t.set_index("ts").net_bps.resample("7D", origin=pd.Timestamp("2026-08-25", tz="UTC")).mean()
        table.loc["weekly means", label] = " ".join(f"{v:+.0f}" for v in wk.dropna())
        s_t = trades(feats, *[v for v in STRATS if v[0] == label][0][1:], strict=True)
        table.loc["mean with strict fills", label] = (f"{s_t.net_bps.mean():+.1f} (n {len(s_t)})"
                                                      if label.find("taker") < 0 else "n/a (taker)")

    # 2. placebo
    plac = {}
    for label, s, w, d, h, ex in STRATS:
        if label.startswith(("#1", "IMPROVED")):
            a = arrays(feats[s])
            valid = np.flatnonzero(np.isfinite(a["mid"]) & (a["mid"] > 0))[:-(h // 5) - 2]
            n = len(tr[label])
            means = []
            for _ in range(2000):
                # random entry ticks over the whole period; then n of the resulting trades at random
                # (not the first n, which would all come from the start of the period)
                _, xf = vs.simulate(a, np.sort(rng.choice(valid, min(len(valid), n * 40), replace=False)), h // 5, ex)
                means.append(rng.choice(xf, n, replace=len(xf) < n).mean())
            plac[label] = np.array(means)
            table.loc["placebo: random entries, same execution", label] = (
                f"{np.mean(means):+.1f} (p = {(np.array(means) >= tr[label].net_bps.mean()).mean():.3f})")

    table = table.fillna("-")

    # 3. out-of-sample selection from the sweep
    sw = pd.read_csv(out / "sweep_ETH.csv")
    oos = []
    for trn, tst in (("A", "B"), ("B", "A")):
        ok = sw[sw[f"{trn}_n"] >= 20]
        for crit in (f"{trn}_mean", f"{trn}_t"):
            top = ok.sort_values(crit, ascending=False).head(10)
            oos.append({"chosen on": trn, "by": crit.split("_")[1], "train mean (top 10)": top[f"{trn}_mean"].mean(),
                        "test mean (top 10)": top[f"{tst}_mean"].mean(),
                        "positive on test": f"{(top[f'{tst}_mean'] > 0).sum()}/10",
                        "test mean, all configs": ok[f"{tst}_mean"].mean()})
    oos = pd.DataFrame(oos)

    # 4. neighbourhood and hold length, other coins
    neigh = pd.DataFrame({f"dip {d} bps": [trades(feats, "ETH", w, d, 3600, 1.0).net_bps.mean() for w in (120, 300, 600, 900)]
                          for d in (15, 20, 25, 30)}, index=[f"VWAP {w}s" for w in (120, 300, 600, 900)])
    holds = pd.DataFrame({f"passive {k}": [trades(feats, "ETH", 300, 20, h, k).net_bps.mean() for h in (600, 1800, 3600, 5400, 7200)]
                          for k in (0.5, 1.0, 2.0)}, index=[f"{h // 60} min" for h in (600, 1800, 3600, 5400, 7200)])
    coins = []
    for s in TICK_PX:
        t = trades(feats, s, 300, 20, 3600, 1.0)
        x = t.net_bps
        coins.append({"coin": s, "n": len(x), "mean bps": x.mean(), "median": x.median(),
                      "win": f"{100 * (x > 0).mean():.0f}%", "t": x.mean() / x.std(ddof=1) * np.sqrt(len(x))})
    coins = pd.DataFrame(coins)

    # charts
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label in (STRATS[0][0], STRATS[1][0], STRATS[5][0], STRATS[6][0]):
        t = tr[label]
        ax.plot(t.ts, t.net_bps.cumsum(), marker=".", label=label)
    ax.axvline(split, color="gray", ls="--", lw=1)
    ax.text(split, ax.get_ylim()[1] * 0.95, " out-of-sample half", color="gray", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(title="Cumulative net return per trade (bps), ETH", ylabel="bps")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=8)
    charts["cumulative"] = fig_to_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, label in zip(axes, (STRATS[0][0], STRATS[5][0])):
        x = tr[label].net_bps
        ax.hist(x, bins=25, color="#4C72B0", alpha=0.8)
        ax.axvline(x.mean(), color="red", label=f"mean {x.mean():+.1f}")
        ax.axvline(x.median(), color="green", ls="--", label=f"median {x.median():+.1f}")
        ax.set(title=label, xlabel="net bps per trade")
        ax.legend(fontsize=8)
    charts["distribution"] = fig_to_b64(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, label in zip(axes, plac):
        ax.hist(plac[label], bins=40, color="gray", alpha=0.7, label="random entries")
        ax.axvline(tr[label].net_bps.mean(), color="red", lw=2, label="strategy")
        ax.set(title=label, xlabel="mean net bps")
        ax.legend(fontsize=7)
    charts["placebo"] = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ok = sw[(sw.A_n >= 20) & (sw.B_n >= 20)]
    ax.scatter(ok.A_mean, ok.B_mean, s=6, alpha=0.35)
    for label, s, w, d, h, ex in STRATS[:4] + STRATS[5:]:
        m = sw[(sw.vwap_window_s == w) & (sw.dev_bps == d) & (sw.tfi_min == 0) & (sw.obi_min == 0) & (sw.hold_s == h)
               & (sw["exec"].astype(str) == (f"passive {ex:g}" if ex != "taker" else "taker"))]
        if len(m):
            ax.scatter(m.A_mean, m.B_mean, s=40, label=label.split(":")[0])
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set(xlabel="mean, first half (25 Aug-13 Sep)", ylabel="mean, second half (14-27 Sep)",
           title=f"ETH sweep: {len(ok)} configs, rank corr {ok.A_mean.corr(ok.B_mean, method='spearman'):+.2f}")
    ax.legend(fontsize=7)
    charts["halves"] = fig_to_b64(fig)

    for name, b64 in charts.items():
        (out / f"{name}.png").write_bytes(base64.b64decode(b64))

    fmt = lambda v: f"{v:+.2f}" if isinstance(v, (float, np.floating)) else str(v)
    tab_md = table.map(fmt).to_markdown() if hasattr(table, "map") else table.to_string()
    body = {
        "stats": table.map(fmt), "oos": oos.round(1), "neigh": neigh.round(1), "holds": holds.round(1),
        "coins": coins.round(2),
    }
    md = ["# vwap_dev_buy deep dive", "", "## Statistics", "", tab_md, "",
          "## Out-of-sample selection (ETH sweep)", "", oos.round(1).to_markdown(index=False), "",
          "## Neighbouring parameters (hold 60m, passive 1 bp, mean net bps)", "", neigh.round(1).to_markdown(), "",
          "## Hold length (ETH, VWAP 300s, dip 20 bps, mean net bps)", "", holds.round(1).to_markdown(), "",
          "## Same rule on every coin (hold 60m, passive 1 bp)", "", coins.round(2).to_markdown(index=False), "",
          "## Charts", ""] + [f"![{n}]({n}.png)" for n in charts]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    css = ("body{font:14px system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:12.5px;margin:8px 0 20px}td,th{border:1px solid #ddd;padding:4px 8px;"
           "text-align:right}th{background:#f4f4f4}td:first-child,th:first-child{text-align:left}"
           "h2{margin-top:28px;border-bottom:1px solid #eee}img{max-width:100%}")
    html = [f"<html><head><meta charset='utf-8'><title>vwap_dev_buy deep dive</title><style>{css}</style></head><body>",
            "<h1>vwap_dev_buy deep dive</h1><p>ETH and BTC, 25 Aug - 27 Sep 2026, capture_data_clean. "
            "Net bps per trade after fees; passive fills simulated.</p>",
            f"<img src='data:image/png;base64,{charts['cumulative']}'>",
            "<h2>Statistics</h2>", body["stats"].to_html(),
            f"<img src='data:image/png;base64,{charts['distribution']}'>",
            "<h2>Placebo: does the entry signal matter?</h2>",
            f"<img src='data:image/png;base64,{charts['placebo']}'>",
            "<h2>Out-of-sample selection (choose on one half, score on the other)</h2>", body["oos"].to_html(index=False),
            f"<img src='data:image/png;base64,{charts['halves']}'>",
            "<h2>Neighbouring parameters (hold 60m, passive 1 bp)</h2>", body["neigh"].to_html(),
            "<h2>Hold length (VWAP 300s, dip 20 bps)</h2>", body["holds"].to_html(),
            "<h2>Same rule on every coin (hold 60m, passive 1 bp)</h2>", body["coins"].to_html(index=False),
            "</body></html>"]
    (out / "report.html").write_text("\n".join(html))
    print(table.map(fmt).to_string())
    print("\n", oos.round(1).to_string(index=False))
    print(f"\nwrote {out}/REPORT.md, {out}/report.html and charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
