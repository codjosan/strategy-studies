#!/usr/bin/env python3
"""Take-profit / stop-loss sweep for leaderboard strategies, entries held fixed.

Strategy sets (``--set``):

* ``top5``  - leaderboard ranks 1-5 (multi-gate ``vwap_dev_buy``, ETH and BTC)
* ``next5`` - leaderboard ranks 6-10 (multi-gate ``vwap_dev_buy`` ETH/SOL, and the
  tp-multifold HYPE order-book-imbalance strategy)

Entry rule and execution stay exactly as in the leaderboard; only the exits are
swept over TP, SL in {none, 10, 15, 20, 30, 40, 60, 80, 100, 150} bps. Each
strategy keeps its own study's exit mechanics:

* multi-gate, checked every 5-second tick from the tick after entry:
  stop-loss when the tick's mid <= entry x (1 - SL), filled at that observed mid
  (can be worse than the level), taker fee; take-profit = resting limit sell at
  entry x (1 + TP), filled at the TP price when the tape trades there, maker fee;
  both in one tick -> stop first; neither -> the original time exit.
* tp-multifold, checked every minute on the minute mid (the study's own rule):
  stop when the return <= -SL, take-profit when >= TP, both exited at the
  observed mid (a TP can fill above its level), taker round trip; otherwise the
  time exit after ``hold_min``. Its leaderboard version already has TP 150 /
  SL 100: that cell is its baseline.

The leaderboard setting of every strategy is reproduced exactly before the
sweep (checked). Parameters are chosen on one half of the period and scored on
the other, in both directions.

Usage:
    PYTHONPATH=. python scripts/vwap_dev_tpsl.py --set top5
    PYTHONPATH=. python scripts/vwap_dev_tpsl.py --set next5
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import vwap_dev_sweep as vs  # noqa: E402
from hlmicro.config import DEFAULT_FEES  # noqa: E402
from hlmicro.validate import tp_multifold  # noqa: E402

LEVELS = [None, 10, 15, 20, 30, 40, 60, 80, 100, 150]        # bps; None = no TP / no SL
MAKER, TAKER = DEFAULT_FEES.maker_bps, DEFAULT_FEES.taker_bps
START, END = "20260825", "20260927"


def simulate(a: dict, idx: np.ndarray, hold: int, execution, tp_bps=None, sl_bps=None):
    """Per trade: (entry tick index, exit tick index, net bps, exit reason)."""
    mid, lo, hi = a["mid"], a["lo"], a["hi"]
    n = len(mid)
    rows, busy = [], -1
    for i in idx:
        if i <= busy:
            continue
        if execution == "taker":
            e = i + 1
            if e + hold >= n or not (mid[e] > 0 and mid[e + hold] > 0):
                continue
            entry_px, entry_fee, ref = mid[e], TAKER, e + hold
        else:
            k = float(execution)
            if not mid[i] > 0 or i + 1 >= n:
                continue
            post = mid[i] * (1 - k * 1e-4)
            if not (np.isfinite(lo[i + 1]) and lo[i + 1] <= post):
                continue
            e = i + 1
            ref = e + hold
            if ref + 1 >= n or not mid[ref] > 0:
                continue
            entry_px, entry_fee = post, MAKER
        # TP / SL scan over ticks e+1 .. ref
        exit_idx = None
        if tp_bps is not None or sl_bps is not None:
            w_mid, w_hi = mid[e + 1:ref + 1], hi[e + 1:ref + 1]
            with np.errstate(invalid="ignore"):
                sl_hit = w_mid <= entry_px * (1 - sl_bps * 1e-4) if sl_bps is not None else np.zeros(len(w_mid), bool)
                tp_hit = w_hi >= entry_px * (1 + tp_bps * 1e-4) if tp_bps is not None else np.zeros(len(w_mid), bool)
            j_sl = int(np.argmax(sl_hit)) if sl_hit.any() else None
            j_tp = int(np.argmax(tp_hit)) if tp_hit.any() else None
            if j_sl is not None and (j_tp is None or j_sl <= j_tp):
                exit_idx = e + 1 + j_sl
                gross, fee, reason = (mid[exit_idx] / entry_px - 1) * 1e4, entry_fee + TAKER, "stop"
            elif j_tp is not None:
                exit_idx = e + 1 + j_tp
                gross, fee, reason = tp_bps * 1.0, entry_fee + MAKER, "tp"
        if exit_idx is None:                     # original time exit
            if execution == "taker":
                exit_idx = ref
                gross, fee, reason = (mid[ref] / entry_px - 1) * 1e4, 2 * TAKER, "time"
            else:
                post_x = mid[ref] * (1 + k * 1e-4)
                exit_idx = ref + 1
                if np.isfinite(hi[ref + 1]) and hi[ref + 1] >= post_x:
                    gross, fee, reason = (post_x / entry_px - 1) * 1e4, entry_fee + MAKER, "time"
                else:
                    if not mid[exit_idx] > 0:
                        continue
                    gross, fee, reason = (mid[exit_idx] / entry_px - 1) * 1e4, entry_fee + TAKER, "time"
        rows.append((e, exit_idx, gross - fee, reason))
        busy = exit_idx
    return rows


@dataclass
class Gate:
    """multi-gate vwap_dev_buy (VWAP 300 s, dip 20 bps, flow >= 0, imbalance >= 0)."""
    label: str
    symbol: str
    hold_s: int
    execution: object                      # "taker" or passive offset in bps
    rank: str = ""
    baseline: tuple = (None, None)         # (TP, SL) of the leaderboard version
    expect: tuple | None = None            # (n, mean) on the leaderboard, checked

    def prepare(self, feats, capture_dir):
        g = feats[self.symbol]
        self.a = {"mid": g["mid"].to_numpy(np.float64), "lo": g["hl_px_min"].to_numpy(np.float64),
                  "hi": g["hl_px_max"].to_numpy(np.float64)}
        self.ticks = g["tick_ts"].to_numpy()
        self.idx = vs.signal_ticks(g, 300, 20, 0.0, 0.0)
        _, ref = vs.simulate(self.a, self.idx, self.hold_s // vs.TICK_S, self.execution)
        mine = np.array([r[2] for r in simulate(self.a, self.idx, self.hold_s // vs.TICK_S, self.execution)])
        assert len(mine) == len(ref) and np.allclose(mine, ref), f"{self.label}: not reproduced"

    def trades(self, tp, sl) -> pd.DataFrame:
        rows = simulate(self.a, self.idx, self.hold_s // vs.TICK_S, self.execution, tp, sl)
        e = np.array([r[0] for r in rows], dtype=np.int64)
        return pd.DataFrame({"ts_ms": self.ticks[e] * vs.TICK_S * 1000 if len(e) else np.zeros(0, np.int64),
                             "net": [r[2] for r in rows], "reason": [r[3] for r in rows]})


@dataclass
class TpMultifold:
    """tp-multifold dual-window OBI z-score, its own minute-level stop/TP rules."""
    label: str
    symbol: str
    rank: str = ""
    threshold: float = 2.5
    hold_min: int = 240
    cooldown_min: int = 60
    baseline: tuple = (150, 100)
    expect: tuple | None = None

    def prepare(self, feats, capture_dir):
        cfg = tp_multifold.Config(capture_dir=capture_dir, symbol=self.symbol, threshold=self.threshold,
                                  hold_min=self.hold_min, cooldown_min=self.cooldown_min,
                                  stop_levels=[0.0] + [-v / 100 for v in LEVELS[1:]],
                                  tp_levels=[0.0] + [v / 100 for v in LEVELS[1:]])
        self.pooled = tp_multifold.run(cfg).pooled

    def trades(self, tp, sl) -> pd.DataFrame:
        key = (-sl / 100 if sl else 0.0, tp / 100 if tp else 0.0)
        _, trades = self.pooled[key]
        reason = {"stop": "stop", "tp": "tp", "timeout": "time"}
        return pd.DataFrame({"ts_ms": [t["entry_ts_ms"] for t in trades],
                             "net": [t["pnl_pct_net"] * 100 for t in trades],
                             "reason": [reason[t["exit_reason"]] for t in trades]})


SETS = {
    "top5": [
        Gate("#1 ETH passive 0.5bp, 30m", "ETH", 1800, 0.5, "1", expect=(73, 7.566)),
        Gate("#2 ETH passive 1bp, 30m", "ETH", 1800, 1.0, "2", expect=(65, 8.179)),
        Gate("#3 ETH passive 0.5bp, 10m", "ETH", 600, 0.5, "3", expect=(81, 5.464)),
        Gate("#4 ETH passive 0bp, 30m", "ETH", 1800, 0.0, "4", expect=(91, 4.960)),
        Gate("#5 BTC taker, 10m", "BTC", 600, "taker", "5", expect=(33, 6.225)),
        Gate("ref: ETH passive 1bp, 60m", "ETH", 3600, 1.0, "ref"),
        Gate("ref: ETH passive 2bp, 60m", "ETH", 3600, 2.0, "ref"),
    ],
    "next5": [
        Gate("#6 ETH passive 1bp, 10m", "ETH", 600, 1.0, "6", expect=(69, 4.563)),
        Gate("#7 ETH passive 0bp, 10m", "ETH", 600, 0.0, "7", expect=(100, 2.902)),
        Gate("#8 SOL passive 1bp, 30m", "SOL", 1800, 1.0, "8", expect=(108, 4.406)),
        TpMultifold("#9 HYPE OBI z-score, 4h (tp-multifold)", "HYPE", "9", expect=(279, 3.367)),
        Gate("#10 SOL passive 0.5bp, 30m", "SOL", 1800, 0.5, "10", expect=(111, 3.931)),
    ],
}


def stat(x: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n}
    sd = x.std(ddof=1)
    return {"n": n, "mean": x.mean(), "median": np.median(x), "win": (x > 0).mean(),
            "t": x.mean() / sd * np.sqrt(n) if sd > 0 else np.nan, "ex_top3": np.sort(x)[:-3].mean(),
            "worst": x.min()}


def lv(v):
    return "none" if v is None else v


def sweep(spec, split_ms: int) -> pd.DataFrame:
    out = []
    for tp, sl in itertools.product(LEVELS, LEVELS):
        t = spec.trades(tp, sl)
        x = t["net"].to_numpy()
        first = t["ts_ms"].to_numpy() < split_ms
        rec = {"strategy": spec.label, "tp_bps": lv(tp), "sl_bps": lv(sl),
               "exit_tp": (t.reason == "tp").mean(), "exit_stop": (t.reason == "stop").mean(),
               "exit_time": (t.reason == "time").mean()}
        for part, m in (("all", np.ones(len(x), bool)), ("A", first), ("B", ~first)):
            rec.update({f"{part}_{k}": v for k, v in stat(x[m]).items()})
        out.append(rec)
    return pd.DataFrame(out)


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap(ax, df, value, title, baseline=(None, None)):
    levels = ["none"] + [str(v) for v in LEVELS[1:]]
    piv = df.assign(tp=df.tp_bps.astype(str), sl=df.sl_bps.astype(str)).pivot(index="sl", columns="tp", values=value)
    vals = piv.reindex(index=levels, columns=levels).to_numpy(float)
    lim = np.nanmax(np.abs(vals)) or 1
    ax.imshow(vals, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(levels)), levels, fontsize=7)
    ax.set_yticks(range(len(levels)), levels, fontsize=7)
    ax.set_xlabel("take-profit (bps)", fontsize=8)
    ax.set_ylabel("stop-loss (bps)", fontsize=8)
    ax.set_title(title, fontsize=9)
    for (r, c), v in np.ndenumerate(vals):
        if np.isfinite(v):
            ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=6)
    b = (levels.index(str(lv(baseline[1]))), levels.index(str(lv(baseline[0]))))
    ax.add_patch(plt.Rectangle((b[1] - 0.5, b[0] - 0.5), 1, 1, fill=False, lw=2, ec="black"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="top5")
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--split", default="20260914")
    ap.add_argument("--out", type=Path, default=None, help="default results/tpsl_<set>")
    ap.add_argument("--min-half", type=int, default=20, help="minimum trades in the selection half")
    args = ap.parse_args()
    out = args.out or Path(f"results/tpsl_{args.set}")
    out.mkdir(parents=True, exist_ok=True)
    split_ms = int(pd.Timestamp(args.split, tz="UTC").timestamp() * 1000)
    specs = SETS[args.set]
    feats = {s: vs.features(args.capture_dir, s, START, END)
             for s in sorted({sp.symbol for sp in specs if isinstance(sp, Gate)})}
    for sp in specs:
        sp.prepare(feats, args.capture_dir)
        if sp.expect:
            t = sp.trades(*sp.baseline)
            assert len(t) == sp.expect[0] and abs(t.net.mean() - sp.expect[1]) < 0.01, \
                f"{sp.label}: {len(t)} trades, mean {t.net.mean():.3f}; leaderboard {sp.expect}"
    print("check: the leaderboard setting of every strategy is reproduced exactly")

    results = pd.concat([sweep(sp, split_ms) for sp in specs], ignore_index=True)
    results.to_csv(out / "tpsl_sweep.csv", index=False)
    cell = lambda r, tp, sl: r[(r.tp_bps.astype(str) == str(lv(tp))) & (r.sl_bps.astype(str) == str(lv(sl)))].iloc[0]

    summary, oos = [], []
    for sp in specs:
        r = results[results.strategy == sp.label]
        base = cell(r, *sp.baseline)
        none = cell(r, None, None)
        ok = r[r.all_n >= 30]
        best = ok.sort_values("all_mean", ascending=False).iloc[0]
        best_t = ok.sort_values("all_t", ascending=False).iloc[0]
        summary.append({"strategy": sp.label, "leaderboard exits": f"TP {lv(sp.baseline[0])} / SL {lv(sp.baseline[1])}",
                        "baseline mean": base.all_mean, "baseline win": base.all_win, "baseline t": base.all_t,
                        "no-exit mean": none.all_mean,
                        "best TP/SL (in-sample)": f"TP {best.tp_bps} / SL {best.sl_bps}", "best mean": best.all_mean,
                        "best win": best.all_win, "best t": best.all_t,
                        "best-t TP/SL": f"TP {best_t.tp_bps} / SL {best_t.sl_bps} (t {best_t.all_t:+.2f})",
                        "combos beating baseline": f"{(ok.all_mean > base.all_mean).sum()}/{len(ok) - 1}"})
        for trn, tst in (("A", "B"), ("B", "A")):
            min_n = args.min_half
            cand = r[r[f"{trn}_n"] >= min_n]
            if cand.empty:                       # small strategies: relax, and say so
                min_n = 10
                cand = r[r[f"{trn}_n"] >= min_n]
            pick = cand.sort_values(f"{trn}_mean", ascending=False).iloc[0]
            oos.append({"strategy": sp.label, "chosen on": f"{trn} (n>={min_n})",
                        "TP/SL": f"TP {pick.tp_bps} / SL {pick.sl_bps}", "train mean": pick[f"{trn}_mean"],
                        "test mean": pick[f"{tst}_mean"], "baseline test mean": base[f"{tst}_mean"],
                        "test n": pick[f"{tst}_n"],
                        "improves on test": "yes" if pick[f"{tst}_mean"] > base[f"{tst}_mean"] + 1e-9 else "no"})
    summary, oos = pd.DataFrame(summary), pd.DataFrame(oos)

    # each exit on its own, relative to no exit at all (average over the set's ranked strategies)
    ranked = [sp for sp in specs if sp.rank != "ref"]
    marg = {}
    for dim, other in (("tp_bps", "sl_bps"), ("sl_bps", "tp_bps")):
        d = {}
        for h, name in (("A", "first half"), ("B", "second half"), ("all", "full period")):
            per = []
            for sp in ranked:
                r = results[(results.strategy == sp.label) & (results[other].astype(str) == "none")]
                none = cell(results[results.strategy == sp.label], None, None)[f"{h}_mean"]
                per.append(pd.Series(r[f"{h}_mean"].to_numpy() - none, index=r[dim].astype(str).to_numpy()))
            d[name] = pd.concat(per, axis=1).mean(axis=1)
        marg[dim] = pd.DataFrame(d).reindex(["none"] + [str(v) for v in LEVELS[1:]])

    # why: trades hit by an exit, against the same trades with no exit
    why = []
    for sp in ranked:
        held = sp.trades(None, None).set_index("ts_ms").net
        for tp, sl in ((20, None), (40, None), (80, None), (None, 30), (None, 60), (None, 100)):
            t = sp.trades(tp, sl)
            hit = t[t.reason.isin(["tp", "stop"]) & t.ts_ms.isin(held.index)]
            extra = t[~t.ts_ms.isin(held.index)]
            why.append({"strategy": sp.label, "exit": f"TP {tp}" if tp else f"SL {sl}", "trades exited": len(hit),
                        "mean with the exit": hit.net.mean(), "same trades, no exit": held.loc[hit.ts_ms].mean(),
                        "extra trades": len(extra), "extra trades mean": extra.net.mean() if len(extra) else np.nan,
                        "strategy mean": t.net.mean(), "no-exit mean": held.mean()})
    why = pd.DataFrame(why)

    # tp-multifold strategies: does the best corner continue past the grid edge?
    extended = {}
    for sp in specs:
        if isinstance(sp, TpMultifold):
            tps, sls = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 0.0], [-0.6, -0.8, -1.0, -1.5, -2.0, 0.0]
            res = tp_multifold.run(tp_multifold.Config(
                capture_dir=args.capture_dir, symbol=sp.symbol, threshold=sp.threshold, hold_min=sp.hold_min,
                cooldown_min=sp.cooldown_min, stop_levels=sls, tp_levels=tps)).pooled
            cells = {}
            for (st, tpv), (_, tr) in res.items():
                x = np.array([t["pnl_pct_net"] * 100 for t in tr])
                ts = np.array([t["entry_ts_ms"] for t in tr])
                cells[(f"SL {-st * 100:.0f}" if st else "SL none", f"TP {tpv * 100:.0f}" if tpv else "TP none")] = \
                    f"{x.mean():+.1f} ({x[ts < split_ms].mean():+.1f} | {x[ts >= split_ms].mean():+.1f})"
            m = pd.Series(cells).unstack()
            extended[sp.label] = m.loc[[f"SL {-v * 100:.0f}" for v in sls if v] + ["SL none"],
                                       [f"TP {v * 100:.0f}" for v in tps if v] + ["TP none"]]

    charts = {}
    for sp in specs:
        r = results[results.strategy == sp.label]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        heatmap(axes[0], r, "all_mean", f"{sp.label}: mean net bps (full period)", sp.baseline)
        ts = pd.to_datetime(sp.trades(None, None).ts_ms, unit="ms", utc=True)
        split_day = pd.Timestamp(args.split, tz="UTC")
        heatmap(axes[1], r, "A_mean", f"first half ({ts.min():%d %b} - {split_day - pd.Timedelta(days=1):%d %b})",
                sp.baseline)
        heatmap(axes[2], r, "B_mean", f"second half ({split_day:%d %b} - {ts.max():%d %b})", sp.baseline)
        charts[sp.label] = b64(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    fig.suptitle(f"Average over {', '.join(sp.label.split()[0] for sp in ranked)}", fontsize=10)
    for ax, dim, title in zip(axes, ("tp_bps", "sl_bps"), ("take-profit alone (no stop)", "stop-loss alone (no TP)")):
        m = marg[dim].drop(index="none")
        x = np.arange(len(m))
        for j, col in enumerate(m.columns):
            ax.bar(x + (j - 1) * 0.27, m[col], 0.27, label=col)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, m.index)
        ax.set(title=title, xlabel="level (bps)", ylabel="change in mean vs no exit (bps)")
        ax.legend(fontsize=7)
    charts["marginal"] = b64(fig)

    fmt = lambda v: f"{v:+.2f}" if isinstance(v, (float, np.floating)) else str(v)
    css = ("body{font:14px system-ui,sans-serif;max-width:1250px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:12.5px;margin:8px 0 20px}td,th{border:1px solid #ddd;"
           "padding:4px 8px;text-align:right}th{background:#f4f4f4}td:first-child{text-align:left}"
           "h2{margin-top:28px;border-bottom:1px solid #eee}img{max-width:100%}")
    html = [f"<html><head><meta charset='utf-8'><title>TP/SL sweep - {args.set}</title><style>{css}</style></head><body>",
            f"<h1>Take-profit / stop-loss sweep: {args.set}</h1>",
            "<p>Entries and execution as on the leaderboard; only the exits change. multi-gate: TP = resting limit "
            "(maker), SL = observed 5-second mid (taker), stop first if both in one tick. tp-multifold: its own rule, "
            "both exits at the observed minute mid, taker. Otherwise the original time exit. Net bps per trade after "
            "fees; multi-gate 25 Aug - 27 Sep 2026 (trade-tape period), tp-multifold 18 Jul - 27 Sep (all order-book data); halves split at 14 Sep. Black square in the heatmaps: leaderboard setting.</p>",
            "<h2>Summary</h2>", summary.map(fmt).to_html(index=False),
            "<h2>Out-of-sample: TP/SL chosen on one half, scored on the other</h2>", oos.map(fmt).to_html(index=False),
            "<h2>Why: trades hit by an exit, against the same trades with no exit</h2>", why.map(fmt).to_html(index=False),
            "<h2>Each exit on its own</h2>", f"<img src='data:image/png;base64,{charts['marginal']}'>",
            "<h3>take-profit only</h3>", marg["tp_bps"].map(fmt).to_html(),
            "<h3>stop-loss only</h3>", marg["sl_bps"].map(fmt).to_html(), "<h2>Heatmaps</h2>"]
    html += [f"<h3>{sp.label}</h3><img src='data:image/png;base64,{charts[sp.label]}'>" for sp in specs]
    for label, m in extended.items():
        html += [f"<h2>{label}: wider take-profit and stop levels</h2>",
                 "<p>Mean net bps, full period (first half | second half).</p>", m.to_html()]
    html.append("</body></html>")
    (out / "report.html").write_text("\n".join(html))
    for name, data in charts.items():
        safe = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        (out / f"{safe}.png").write_bytes(base64.b64decode(data))
    pd.set_option("display.width", 260)
    print(summary.map(fmt).to_string(index=False))
    print()
    print(oos.map(fmt).to_string(index=False))
    print()
    print(why.map(fmt).to_string(index=False))
    for dim in ("tp_bps", "sl_bps"):
        print(f"\n{dim} alone, change in mean vs no exit:")
        print(marg[dim].map(fmt).to_string())
    print(f"\nwrote {out}/tpsl_sweep.csv, {out}/report.html and charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
