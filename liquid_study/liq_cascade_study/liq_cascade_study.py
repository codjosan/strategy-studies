#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fast_capture.liq_cascade.file_utils import glob_daily
from fast_capture.liq_cascade.extract import extract_day
from fast_capture.liq_cascade.events import load_coin_grid, find_events
from fast_capture.liq_cascade.report import print_grid, print_verdict

DEFAULT_SYMBOLS = "BTC,ETH,SOL,HYPE,DOGE,LINK"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path,
                   default=Path.home() / "fast_capture/capture_data")
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--windows", default="60,300", help="burst windows, seconds")
    p.add_argument("--quantiles", default="0.95,0.99,0.999",
                   help="burst threshold: quantile of the coin's trailing non-zero windows")
    p.add_argument("--horizons-min", default="1,5,15,30,60")
    p.add_argument("--latency-s", type=int, default=2)
    p.add_argument("--fee-bps-rt", type=float, default=8.64,
                   help="taker fee round trip (4.32 bps × 2, same as hlmicro FeeModel)")
    p.add_argument("--slip-bps-rt", type=float, default=2.0,
                   help="extra slippage round trip (spreads widen in cascades)")
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=7)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--skip-today", action="store_true")
    p.add_argument("--redo", action="store_true", help="rebuild the per-day cache")
    p.add_argument("--outdir", type=Path, default=Path("fast_capture/liq_cascade"))
    args = p.parse_args()

    symbols = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    windows = sorted({int(x) for x in args.windows.split(",")})
    qs = sorted({float(x) for x in args.quantiles.split(",")})
    horizons_s = sorted({int(float(x) * 60) for x in args.horizons_min.split(",")})
    cost = args.fee_bps_rt + args.slip_bps_rt
    cache = args.outdir / "daily"
    cache.mkdir(parents=True, exist_ok=True)

    tfiles = glob_daily(args.capture_dir, "hl_trades_")
    lfiles = glob_daily(args.capture_dir, "hl_liquidations_")
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    days = [d for d in tfiles
            if (args.since is None or d >= args.since)
            and (args.until is None or d <= args.until)
            and not (args.skip_today and d == today)]
    if not days:
        print("No hl_trades files in range")
        return
    days_all = [d.strftime("%Y%m%d") for d in
                pd.date_range(pd.to_datetime(days[0]), pd.to_datetime(days[-1]), freq="D")]

    print(f"Liquidation cascade study  —  {', '.join(symbols)}")
    print(f"  Days: {len(days)} with trades  ({days[0]} → {days[-1]}),  "
          f"liquidation files from {min(lfiles) if lfiles else 'n/a'}")
    print(f"  Windows: {windows}s   thresholds (trailing {args.lookback_days}d quantile): {qs}")
    print(f"  Horizons: {[h // 60 for h in horizons_s]} min   latency: {args.latency_s}s   "
          f"cost: {args.fee_bps_rt:.2f} fee + {args.slip_bps_rt:.2f} slip = {cost:.2f} bps RT")
    print()

    todo = [d for d in days if args.redo or d == today or not (cache / f"sec_{d}.npz").exists()]
    print(f"Stage 1 — per-day 1s extract: {len(days) - len(todo)} cached, {len(todo)} to build")
    for d in todo:
        msg = extract_day(d, tfiles.get(d), lfiles.get(d), symbols, cache / f"sec_{d}.npz")
        print(msg, flush=True)
        gc.collect()

    print("\nStage 2 — finding bursts per coin")
    all_events = []
    for c in symbols:
        px, ls, lb = load_coin_grid(days_all, cache, c)
        n_before = len(all_events)
        for w in windows:
            all_events += find_events(c, days_all, px, ls, lb, w, qs,
                                      args.lookback_days, args.warmup_days,
                                      args.latency_s, horizons_s)
            gc.collect()
        print(f"  {c:<5} events: {len(all_events) - n_before:>6,}   "
              f"liq notional total: ${(ls.sum() + lb.sum()) / 1e6:,.1f}M", flush=True)
        del px, ls, lb
        gc.collect()

    ev = pd.DataFrame(all_events)
    ev_path = args.outdir / "cascade_events.csv"
    ev.to_csv(ev_path, index=False)
    if ev.empty:
        print("\nNo events found (not enough liquidation history?)")
        return

    med_pre = ev["pre_move_bps"].median()
    print(f"\n  Sanity: median price move during bursts, in the cascade's direction: "
          f"{med_pre:+.1f} bps")
    if med_pre < 0:
        print("  ⚠ Price moved AGAINST the assumed cascade direction — convention inverted.")
    ev_days = sorted(ev["day"].unique())
    fold_of = {d: min(args.n_folds - 1, i * args.n_folds // len(ev_days))
               for i, d in enumerate(ev_days)}

    print()
    print("=" * 118)
    print(f"NET RETURN PER EVENT, bps, after {cost:.2f} bps round trip")
    print("=" * 118)

    summary = []
    for w in windows:
        for q in qs:
            sub = ev[(ev["window_s"] == w) & (ev["q"] == q)]
            if sub.empty:
                continue
            pm = sub["pre_move_bps"]
            rows = print_grid(sub, horizons_s, cost, fold_of,
                              f"[ALL coins]  window={w}s  threshold=q{q}  "
                              f"events={len(sub)}  median pre-move with cascade="
                              f"{pm.median():+.1f} bps")
            summary += [{"window_s": w, "q": q, "coin": "ALL", **r} for r in rows]

    w_mid = windows[-1] if len(windows) > 1 else windows[0]
    q_mid = qs[len(qs) // 2]
    print()
    print("=" * 118)
    print(f"BY COIN  (window={w_mid}s, q{q_mid})")
    print("=" * 118)
    for c in symbols:
        sub = ev[(ev["window_s"] == w_mid) & (ev["q"] == q_mid) & (ev["coin"] == c)]
        if len(sub) < 10:
            print(f"\n  [{c}] only {len(sub)} events — skipped")
            continue
        rows = print_grid(sub, horizons_s, cost, fold_of, f"[{c}]  events={len(sub)}")
        summary += [{"window_s": w_mid, "q": q_mid, "coin": c, **r} for r in rows]

    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.outdir / "cascade_summary.csv", index=False)

    print_verdict(sdf, cost, len(sdf))
    print(f"Wrote {ev_path}, {args.outdir}/cascade_summary.csv")

if __name__ == "__main__":
    main()
