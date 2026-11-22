"""Primitives shared by the studies: z-scores, as-of lookups and summary statistics."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def rolling_zscore(values: Sequence[float] | np.ndarray | pd.Series, window: int,
                   min_periods: Optional[int] = None) -> np.ndarray:
    """Trailing z-score of each value against the preceding ``window`` values
    (inclusive). Undefined (NaN) until ``min_periods`` observations exist or
    where the window has zero variance."""
    series = pd.Series(np.asarray(values, dtype=float))
    roll = series.rolling(window, min_periods=window if min_periods is None else min_periods)
    std = roll.std().replace(0, np.nan)
    return ((series - roll.mean()) / std).to_numpy()


class StepLookup:
    """As-of lookup on a sorted time series."""

    def __init__(self, ts: np.ndarray, values: np.ndarray, max_age_ms: Optional[int] = None):
        self.ts = np.asarray(ts, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        self.max_age_ms = max_age_ms

    def at(self, t):
        """Last value at or before ``t``; NaN before the first observation or
        when the last observation is older than ``max_age_ms``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.int64))
        idx = np.searchsorted(self.ts, t_arr, side="right") - 1
        out = np.full(t_arr.shape, np.nan)
        ok = idx >= 0
        if self.max_age_ms is not None:
            ok &= (t_arr - self.ts[np.clip(idx, 0, None)]) <= self.max_age_ms
        out[ok] = self.values[idx[ok]]
        return float(out[0]) if np.ndim(t) == 0 else out

    def nearest(self, t, max_skew_ms: int):
        """Observation closest to ``t`` (the later one on a tie); NaN if none
        lies within ``max_skew_ms``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.int64))
        n = len(self.ts)
        out = np.full(t_arr.shape, np.nan)
        if n == 0:
            return float(out[0]) if np.ndim(t) == 0 else out
        after = np.searchsorted(self.ts, t_arr, side="left")
        before = after - 1
        d_after = np.where(after < n, self.ts[np.clip(after, 0, n - 1)] - t_arr, np.iinfo(np.int64).max)
        d_before = np.where(before >= 0, t_arr - self.ts[np.clip(before, 0, n - 1)], np.iinfo(np.int64).max)
        use_after = d_after <= d_before
        idx = np.where(use_after, after, before)
        dist = np.where(use_after, d_after, d_before)
        ok = dist <= max_skew_ms
        out[ok] = self.values[idx[ok]]
        return float(out[0]) if np.ndim(t) == 0 else out


def describe(returns: Sequence[float] | np.ndarray, ddof: int = 1) -> dict:
    """n, mean, std, per-trade Sharpe (mean/std), t-statistic, win rate, total.

    With fewer than two observations (or zero dispersion) Sharpe and t are 0.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sharpe": np.nan, "t": np.nan,
                "win_rate": np.nan, "total": 0.0}
    mean = float(x.mean())
    std = float(x.std(ddof=ddof)) if n > ddof else 0.0
    sharpe = mean / std if std > 0 else 0.0
    return {"n": n, "mean": mean, "std": std, "sharpe": sharpe,
            "t": sharpe * np.sqrt(n) if std > 0 else 0.0,
            "win_rate": float((x > 0).mean()), "total": float(x.sum())}


def fmt_signed(value: float, digits: int = 2, width: int = 0) -> str:
    """``+1.23`` style; ``na`` for NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "na".rjust(width)
    return f"{value:+.{digits}f}".rjust(width)


def rule(width: int = 90, char: str = "=") -> str:
    return char * width
