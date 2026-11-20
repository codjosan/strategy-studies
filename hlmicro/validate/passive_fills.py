"""Passive limit-order fills on a tick grid, checked against the traded range.

Used by ``multi-gate`` and ``scalp-explore`` to test whether an edge measured
at the mid survives realistic maker execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeeModel


def simulate_passive_fills(grid: pd.DataFrame, idx: np.ndarray, hold_ticks: int, side: int,
                           k_bps: float, entry_patience: int, exit_patience: int,
                           fees: FeeModel) -> pd.DataFrame:
    """Passive limit ``k_bps`` inside the mid at entry and exit.

    The entry limit is posted at the end of the trigger tick and fills when
    the Hyperliquid tape (``hl_px_min``/``hl_px_max``) trades through it within
    ``entry_patience`` ticks; otherwise the trade is skipped. The exit limit is
    posted ``hold_ticks`` after the fill; if it does not fill within
    ``exit_patience`` ticks the position is closed at the mid with taker fees.
    Positions never overlap.

    ``attrs`` carries ``n_triggers``, ``n_in_position`` (triggers skipped
    because a position was open), ``n_filled``, ``n_exit_maker`` and
    ``n_exit_taker``.
    """
    if "hl_px_min" not in grid or "hl_px_max" not in grid:
        raise ValueError("fill simulation needs the Hyperliquid trade tape")
    mid, lo, hi, ts = (grid[c].to_numpy() for c in ("mid", "hl_px_min", "hl_px_max", "tick_ts"))
    n = len(grid)
    maker, taker = fees.maker_bps / 1e4, fees.taker_bps / 1e4
    rows, busy_until = [], -1
    counts = {"n_triggers": 0, "n_in_position": 0, "n_filled": 0, "n_exit_maker": 0, "n_exit_taker": 0}
    for i in idx:
        counts["n_triggers"] += 1
        if i <= busy_until:
            counts["n_in_position"] += 1
            continue
        if not (np.isfinite(mid[i]) and mid[i] > 0):
            continue
        offset = k_bps * 1e-4 * mid[i]
        post_entry = mid[i] - offset if side > 0 else mid[i] + offset
        entry = next((k for k in range(i + 1, min(i + 1 + entry_patience, n))
                      if (side > 0 and np.isfinite(lo[k]) and lo[k] <= post_entry)
                      or (side < 0 and np.isfinite(hi[k]) and hi[k] >= post_entry)), None)
        if entry is None:
            continue
        counts["n_filled"] += 1
        ref = entry + hold_ticks
        if ref >= n or not (np.isfinite(mid[ref]) and mid[ref] > 0):
            continue
        offset = k_bps * 1e-4 * mid[ref]
        post_exit = mid[ref] + offset if side > 0 else mid[ref] - offset
        exit_ = next((k for k in range(ref + 1, min(ref + 1 + exit_patience, n))
                      if (side > 0 and np.isfinite(hi[k]) and hi[k] >= post_exit)
                      or (side < 0 and np.isfinite(lo[k]) and lo[k] <= post_exit)), None)
        exit_px, exit_maker = post_exit, True
        if exit_ is None:
            exit_ = min(ref + exit_patience, n - 1)
            if not (np.isfinite(mid[exit_]) and mid[exit_] > 0):
                continue
            exit_px, exit_maker = mid[exit_], False
            counts["n_exit_taker"] += 1
        else:
            counts["n_exit_maker"] += 1
        gross = side * (exit_px - post_entry) / post_entry
        rows.append({"entry_tick": int(ts[entry]), "exit_tick": int(ts[exit_]),
                     "entry_px": float(post_entry), "exit_px": float(exit_px), "gross_ret": float(gross),
                     "net_ret": float(gross - (maker + (maker if exit_maker else taker))),
                     "exit_was_maker": exit_maker, "entry_delay_ticks": int(entry - i)})
        busy_until = exit_
    frame = pd.DataFrame(rows)
    frame.attrs.update(counts)
    return frame
