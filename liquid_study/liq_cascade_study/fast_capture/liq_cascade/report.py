from __future__ import annotations

import pandas as pd
import numpy as np
from .statistics import cell_stats

def print_grid(ev: pd.DataFrame, horizons_s: list[int], cost: float, fold_of: dict,
               title: str) -> list[dict]:
    rows = []
    print(f"\n  {title}")
    print(f"    {'dir':<7}{'horizon':>8}{'n':>6}{'days':>6}{'mean':>9}{'median':>9}"
          f"{'win%':>7}{'95% CI (day bootstrap)':>26}  folds+  fold means")
    for sign, name in [(1, "FADE"), (-1, "FOLLOW")]:
        for h in horizons_s:
            col = f"fade_{h // 60}m"
            s = cell_stats(ev, col, sign, cost, fold_of)
            if s["n"] == 0:
                continue
            ci = (f"[{s['ci_lo']:+7.1f}, {s['ci_hi']:+7.1f}]"
                  if np.isfinite(s["ci_lo"]) else "   (too few days)")
            print(f"    {name:<7}{h // 60:>7}m{s['n']:>6}{s['days']:>6}{s['mean']:>+9.1f}"
                  f"{s['median']:>+9.1f}{s['win']:>7.1f}{ci:>26}  {s['folds_pos']:>6}  "
                  f"{s['fold_means']}")
            rows.append({"dir": name, "h_min": h // 60, **s})
    return rows

def print_verdict(sdf: pd.DataFrame, cost: float, n_tested: int) -> None:
    print()
    print("=" * 118)
    print("VERDICT")
    print("=" * 118)
    ok = sdf[(sdf["n"] >= 30) & (sdf["ci_lo"] > 0)] if len(sdf) else sdf
    if len(ok):
        ok = ok[ok["folds_pos"].map(lambda s: int(s.split("/")[0]) / max(1, int(s.split("/")[1]))) >= 0.75]
    if ok.empty:
        print(f"  None of the {n_tested} (setting, coin, direction, horizon) cells has a "
              f"day-bootstrap CI above 0 net of {cost:.1f} bps with ≥75% of folds positive.")
        print("  → Liquidation cascades do not pay for taker execution at this latency.")
    else:
        print(f"  {len(ok)} of {n_tested} cells pass (CI > 0 net, ≥75% folds positive, n ≥ 30):")
        for _, r in ok.sort_values("mean", ascending=False).head(15).iterrows():
            print(f"    {r['coin']:<5} W={int(r['window_s']):>3}s q{r['q']:<6} {r['dir']:<6} "
                  f"{int(r['h_min']):>2}m  mean={r['mean']:+.1f}bps  CI=[{r['ci_lo']:+.1f},"
                  f"{r['ci_hi']:+.1f}]  win={r['win']:.0f}%  folds+={r['folds_pos']}  n={int(r['n'])}")
        print(f"  With {n_tested} cells tested, a couple can pass by luck. Believe it only if the")
        print("  neighbouring thresholds/horizons point the same way and it holds with "
              "--latency-s 5 --slip-bps-rt 4.")
