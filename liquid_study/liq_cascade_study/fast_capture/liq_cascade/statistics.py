from __future__ import annotations

import numpy as np
import pandas as pd

def day_bootstrap_ci(vals: np.ndarray, days: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    ud, inv = np.unique(days, return_inverse=True)
    if len(ud) < 5:
        return (np.nan, np.nan)
    s = np.bincount(inv, weights=vals)
    c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    means = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def cell_stats(ev: pd.DataFrame, col: str, sign: int, cost: float,
               fold_of: dict) -> dict:
    x = ev[[col, "day"]].dropna()
    if len(x) == 0:
        return {"n": 0}
    v = sign * x[col].to_numpy() - cost
    lo, hi = day_bootstrap_ci(v, x["day"].to_numpy())
    f = x["day"].map(fold_of).to_numpy()
    fm = [v[f == k].mean() for k in sorted(set(f))]
    return {"n": len(v), "days": x["day"].nunique(), "mean": v.mean(),
            "median": float(np.median(v)), "win": 100 * (v > 0).mean(),
            "ci_lo": lo, "ci_hi": hi,
            "folds_pos": f"{sum(m > 0 for m in fm)}/{len(fm)}",
            "fold_means": " ".join(f"{m:+.1f}" for m in fm)}
