#!/usr/bin/env python3
"""Forward test of the locked strategies on data captured after the lock.

Reads ``forward_lock.json`` (written by ``scripts/vol_norm_dip_validate.py``) and
evaluates, with no parameter changes:

* vol_norm_dip on every locked coin (needs obi_5s and hl_trades)
* momentum "follow 1 h move >= 2 sd, hold 1 h" on LINK, ETH and BTC (needs obi_5s)

Only trades entered strictly after ``data_used_up_to`` count. Earlier days are
loaded only as warm-up for the trailing statistics (24 h for vol_norm_dip,
7 days for momentum). Pass criteria are the ones in the lock file.

Usage:
    PYTHONPATH=. python scripts/forward_test.py
    PYTHONPATH=. python scripts/forward_test.py --pretend-lock-date 20260920   # mechanics check only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hlmicro.parse import discover, file_date
from hlmicro.validate import candidates as cd


def evaluate(name, parts):
    x = np.concatenate([p for p in parts.values() if len(p)]) if any(len(p) for p in parts.values()) else np.array([])
    n = len(x)
    line = {"strategy": name, "trades": n,
            "per coin": " ".join(f"{c} {len(v)}:{v.mean():+.1f}" for c, v in parts.items() if len(v))}
    if n >= 2:
        t = x.mean() / x.std(ddof=1) * np.sqrt(n)
        p = stats.t.sf(t, n - 1)
        line.update({"mean bps": round(x.mean(), 2), "t": round(t, 2), "p (mean>0)": round(p, 4),
                     "PASS" if (n >= 30 and x.mean() > 0 and p < 0.05) else "not yet / fail":
                         "n>=30, mean>0, p<0.05"})
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, default=Path("capture_data_clean"))
    ap.add_argument("--lock", type=Path, default=Path("results/vol_norm_dip_validation/forward_lock.json"))
    ap.add_argument("--pretend-lock-date", default=None,
                    help="testing the mechanics only: treat this date as the lock date (that data is NOT unseen)")
    args = ap.parse_args()
    lock = json.loads(args.lock.read_text())
    cutoff = args.pretend_lock_date or lock["data_used_up_to"]
    books = discover(args.capture_dir, "obi_5s")
    after = [p for p in books if file_date(p) > cutoff]
    print(f"lock: {args.lock} (locked {lock['locked_on']}, data used up to {lock['data_used_up_to']})")
    if args.pretend_lock_date:
        print(f"PRETEND MODE: evaluating after {cutoff}; these days were used in development, so this is "
              f"a mechanics check, not evidence")
    if not after:
        print(f"\nno obi_5s files after {cutoff} in {args.capture_dir}: nothing to forward-test yet.")
        print("Add newer captures (obi_5s, and hl_trades for vol_norm_dip), sanitize them into the capture "
              "directory, and run this script again.")
        return 0
    first, last = file_date(after[0]), file_date(after[-1])
    cut_ms = int((pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    warm = (pd.Timestamp(first) - pd.Timedelta(days=8)).strftime("%Y%m%d")
    print(f"forward period: {first} .. {last} ({len(after)} days); warm-up from {warm}\n")
    results = []

    v = lock["vol_norm_dip"]
    tapes_after = [p for p in discover(args.capture_dir, "hl_trades") if file_date(p) > cutoff]
    parts, filtered = {}, {}
    if tapes_after:
        grids = {c: cd.build_grid(args.capture_dir, c, warm, last) for c in v["coins"]}
        for c, g in grids.items():
            if g.tape is None:
                continue
            rows = cd.vol_norm_dip(g, v["z_min"], v["hold_min"], v["k"], vwap_min=v["vwap_min"])
            e = np.array([r[0] for r in rows], dtype=np.int64)
            x = np.array([r[1] for r in rows])
            after_lock = g.ts_ms(e) >= cut_ms if len(e) else np.zeros(0, bool)
            parts[c] = x[after_lock]
            if "vol_norm_dip_context_filter" in lock and len(e):
                ctx = cd.context_features(grids, c, e - 1)
                keep = ((ctx.market_15m < 0) & (ctx.coint_z < 0)).to_numpy()
                filtered[c] = x[after_lock & keep]
        results.append(evaluate(f"vol_norm_dip {v['z_min']:g} sd, hold {v['hold_min']}m, VWAP {v['vwap_min']}m, "
                                f"limit {v['k']:g}bp", parts))
        if "vol_norm_dip_context_filter" in lock:
            results.append(evaluate("secondary: same, only when market_15m < 0 and coint_z < 0", filtered))
    else:
        results.append({"strategy": "vol_norm_dip", "trades": 0, "note": f"no hl_trades after {cutoff}"})

    m = lock["momentum"]
    parts = {}
    for c in m["coins"]:
        g = cd.build_grid(args.capture_dir, c, warm, last)
        rows = cd.momentum(g, m["lookback_h"], m["k_sigma"], m["hold_h"], m["mode"])
        e = np.array([r[0] for r in rows], dtype=np.int64)
        x = np.array([r[1] for r in rows])
        parts[c] = x[g.ts_ms(e) >= cut_ms] if len(e) else x
    results.append(evaluate(f"momentum {m['mode']} {m['lookback_h']}h >= {m['k_sigma']} sd, hold {m['hold_h']}h "
                            f"({', '.join(m['coins'])})", parts))
    for r in results:
        print("  " + "\n    ".join(f"{k}: {val}" for k, val in r.items()))
    print(f"\npass criteria: {lock['pass_criteria']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
