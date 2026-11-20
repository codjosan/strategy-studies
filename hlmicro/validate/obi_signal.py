"""The order-book imbalance z-score signal and its minute-bar position simulator.

Signal: trailing z-scores of cross-venue mean order-book imbalance over a
primary window (default 60 minutes) and, optionally, a confirmation window
(default 30 minutes). Long when every z-score is above ``+threshold``, short
when every one is below ``-threshold``.

The simulator holds at most one position. Each minute in a position it
checks exits in a fixed priority order:

1. ``stop``      gross return at or below ``stop_pct``
2. ``reversal``  primary z-score against the trade at least ``reversal_z``,
                 once ``reversal_min_hold`` minutes have passed
3. target        return (gross, or net of cost) at or above ``target_pct``;
                 reported as ``target_label`` (``tp`` by default)
4. ``timeout``   ``hold_min`` reached, unless the extension rule keeps the
                 position open (primary z-score with the trade at least
                 ``extension_z``), in which case ``max_hold`` ends it

After an exit no entry is allowed for ``cooldown_min`` minutes. By default
a position is entered at the mid of the trigger minute (the convention of
``tp-multifold``); ``next_bar_entry`` enters at the following minute's mid
instead, so the entry price is strictly later than the information used.

Studies built on this module: ``tp-multifold``, ``threshold-cv``,
``oos-split``, ``breakeven-exit``, ``signal-exit`` and ``trend-gate``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .common import describe, rolling_zscore


@dataclass(frozen=True)
class SignalSpec:
    threshold: float = 2.5
    primary_window_min: int = 60
    confirm_window_min: Optional[int] = 30    # None: primary window only

    @property
    def warmup(self) -> int:
        return max(self.primary_window_min, self.confirm_window_min or 0)

    def describe(self) -> str:
        windows = f"z{self.primary_window_min}"
        if self.confirm_window_min:
            windows += f" and z{self.confirm_window_min}"
        return f"{windows} beyond +/-{self.threshold}"


@dataclass(frozen=True)
class ExitSpec:
    hold_min: int = 240
    cooldown_min: int = 60
    stop_pct: Optional[float] = None          # negative, e.g. -1.0; None disables
    target_pct: Optional[float] = None        # None disables (0 is a valid level)
    target_on_net: bool = False               # compare the target with net instead of gross return
    target_label: str = "tp"
    reversal_z: float = 0.0                   # 0 disables
    reversal_min_hold: Optional[int] = None   # default: hold_min
    extension_z: float = 0.0                  # 0 disables
    max_hold_min: Optional[int] = None        # cap while extending; default: hold_min
    next_bar_entry: bool = False
    close_at_end: bool = False                # report a position open at the end as "open_at_end"


def signal_zscores(obi: np.ndarray, spec: SignalSpec) -> tuple[np.ndarray, Optional[np.ndarray]]:
    primary = rolling_zscore(obi, spec.primary_window_min)
    confirm = rolling_zscore(obi, spec.confirm_window_min) if spec.confirm_window_min else None
    return primary, confirm


def _direction(zp: float, zc: Optional[float], threshold: float) -> Optional[str]:
    if np.isnan(zp) or (zc is not None and np.isnan(zc)):
        return None
    if zp > threshold and (zc is None or zc > threshold):
        return "LONG"
    if zp < -threshold and (zc is None or zc < -threshold):
        return "SHORT"
    return None


def simulate(grid: pd.DataFrame, signal: SignalSpec, exits: ExitSpec, fee_bps: float) -> list[dict]:
    """Single-position backtest on a minute grid with ``obi_ntl``, ``mid`` and
    ``ts_ms`` columns. Returns one dict per trade; returns are in percent and
    ``fee_bps`` is charged once per round trip."""
    obi = grid["obi_ntl"].to_numpy()
    mid = grid["mid"].to_numpy()
    ts_ms = grid["ts_ms"].to_numpy()
    n = len(grid)
    z_primary, z_confirm = signal_zscores(obi, signal)
    cost_pct = fee_bps / 100.0
    reversal_min_hold = exits.hold_min if exits.reversal_min_hold is None else exits.reversal_min_hold
    max_hold = exits.hold_min if exits.max_hold_min is None else exits.max_hold_min

    trades: list[dict] = []
    side: Optional[str] = None
    entry_idx = 0
    entry_px = 0.0
    last_exit = -10**9
    mae = mfe = 0.0

    def record(i: int, px: float, ret: float, reason: str, z_now: float) -> None:
        held = i - entry_idx
        cf_idx = entry_idx + exits.hold_min
        cf_net = np.nan
        if cf_idx < n and mid[cf_idx] > 0:
            cf = (mid[cf_idx] - entry_px) / entry_px * 100
            cf_net = (-cf if side == "SHORT" else cf) - cost_pct
        trades.append({
            "side": side, "entry_ts_ms": int(ts_ms[entry_idx]), "exit_ts_ms": int(ts_ms[i]),
            "entry_px": float(entry_px), "exit_px": float(px), "minutes_held": int(held),
            "exit_reason": reason, "ret_gross_pct": ret, "pnl_pct_net": ret - cost_pct,
            "mae_pct": mae, "mfe_pct": mfe,
            "z_primary_entry": float(z_primary[entry_idx]),
            "z_confirm_entry": float(z_confirm[entry_idx]) if z_confirm is not None else np.nan,
            "z_primary_exit": z_now, "extended": bool(held > exits.hold_min),
            "pnl_at_hold_net": float(cf_net),
        })

    for i in range(signal.warmup, n):
        px = mid[i]
        if not px > 0:
            continue
        if side is not None:
            ret = (px - entry_px) / entry_px * 100
            if side == "SHORT":
                ret = -ret
            mae, mfe = min(mae, ret), max(mfe, ret)
            held = i - entry_idx
            z_now = 0.0 if np.isnan(z_primary[i]) else float(z_primary[i])
            z_with = -z_now if side == "SHORT" else z_now
            target_ret = ret - cost_pct if exits.target_on_net else ret
            reason = None
            if exits.stop_pct is not None and ret <= exits.stop_pct:
                reason = "stop"
            elif exits.reversal_z > 0 and -z_with >= exits.reversal_z and held >= reversal_min_hold:
                reason = "reversal"
            elif exits.target_pct is not None and target_ret >= exits.target_pct:
                reason = exits.target_label
            elif held >= exits.hold_min:
                extending = exits.extension_z > 0 and z_with >= exits.extension_z and held < max_hold
                if not extending:
                    reason = "max_hold" if exits.extension_z > 0 and held >= max_hold else "timeout"
            if reason:
                record(i, px, ret, reason, z_now)
                side = None
                last_exit = i
        if side is None and i - last_exit >= exits.cooldown_min:
            zc = z_confirm[i] if z_confirm is not None else None
            direction = _direction(z_primary[i], zc, signal.threshold)
            if direction is None:
                continue
            j = i + 1 if exits.next_bar_entry else i
            if j >= n or not mid[j] > 0:
                continue
            side, entry_idx, entry_px = direction, j, mid[j]
            mae = mfe = 0.0

    if side is not None and exits.close_at_end:
        px = mid[n - 1]
        ret = (px - entry_px) / entry_px * 100
        if side == "SHORT":
            ret = -ret
        z_now = 0.0 if np.isnan(z_primary[n - 1]) else float(z_primary[n - 1])
        record(n - 1, px, ret, "open_at_end", z_now)
    return trades


def split_folds(grid: pd.DataFrame, n_folds: int) -> list[dict]:
    """Equal-duration chronological folds over the grid's time span."""
    first, last = int(grid["ts_ms"].iloc[0]), int(grid["ts_ms"].iloc[-1])
    edges = np.linspace(first, last, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        start, end = int(edges[k]), int(edges[k + 1])
        part = grid[(grid["ts_ms"] >= start) & (grid["ts_ms"] < end)].reset_index(drop=True)
        if len(part):
            folds.append({"idx": k, "start_ms": start, "end_ms": end,
                          "days": (end - start) / 86_400_000, "grid": part})
    return folds


def folds_by_entry(trades: list[dict], first_ms: int, last_ms: int, n_folds: int) -> dict[int, list[dict]]:
    """Trades assigned to equal-duration folds by entry time (exits may spill
    into the next fold); entries at or after the last edge go to the last fold."""
    span = last_ms - first_ms
    edges = [first_ms + int(span * k / n_folds) for k in range(n_folds + 1)]
    out: dict[int, list[dict]] = {k: [] for k in range(n_folds)}
    for t in trades:
        k = next((k for k in range(n_folds) if edges[k] <= t["entry_ts_ms"] < edges[k + 1]), n_folds - 1)
        out[k].append(t)
    return out


def trade_stats(trades: list[dict]) -> dict:
    """n, win rate (%), mean, per-trade Sharpe, t-statistic and sum of net PnL%."""
    s = describe([t["pnl_pct_net"] for t in trades], ddof=1)
    return {"n": s["n"], "winr": s["win_rate"] * 100 if s["n"] else np.nan,
            "mean": s["mean"], "sharpe": s["sharpe"], "tstat": s["t"], "sum": s["total"]}


def fold_mean_cells(by_fold: dict[int, list[dict]]) -> tuple[list[tuple[int, float]], int, float]:
    """Per-fold (n, mean net PnL%), folds with a positive mean, mean of means."""
    cells, means = [], []
    for k in sorted(by_fold):
        pnl = [t["pnl_pct_net"] for t in by_fold[k]]
        if pnl:
            cells.append((len(pnl), float(np.mean(pnl))))
            means.append(cells[-1][1])
        else:
            cells.append((0, np.nan))
    return cells, sum(m > 0 for m in means), float(np.mean(means)) if means else np.nan


def render_fold_row(label: str, cells: list[tuple[int, float]], positive: int, mom: float,
                    width: int = 44) -> str:
    parts = [f"{'-':>15}" if n == 0 else f"{m:>+7.3f}% (n={n:>3})" for n, m in cells]
    mom_s = "na" if np.isnan(mom) else f"{mom:+.3f}%"
    return f"  {label:<{width}}" + "".join(f"{p:>17}" for p in parts) + f"  {positive}/{len(cells)}  {mom_s:>8}"
