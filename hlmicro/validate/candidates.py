"""Candidate strategies drafted from the capture data, tested in the run-all framework.

Seven hypotheses, each with a small parameter grid fixed in advance (see
``docs/NEW_STRATEGIES.md`` for the reasoning behind each one):

====================  =========================================================  ===============
name                  idea                                                       data
====================  =========================================================  ===============
basis_reversion       HL mid strays from the CEX consensus, then converges       obi_5s
passive_leadlag       follow a sharp Binance move on HL with a maker order        obi_5s + tape
flow_fade             fade a selling / buying climax on the HL tape              obi_5s + tape
vol_norm_dip          VWAP dip-buy with the threshold scaled to volatility       obi_5s + tape
momentum              1 h / 4 h breakout vs. recent volatility: follow or fade   obi_5s
funding_proxy         short/long into the hourly settlement by HL premium        obi_5s
pairs                 ratio mean reversion ETH/BTC, SOL/ETH, SOL/BTC             obi_5s
====================  =========================================================  ===============

Common rules:

* One aligned 5-second grid per coin: per-venue mid and imbalance from
  ``obi_5s``, HL trade flow from ``hl_trades`` (NaN where no tape exists).
* Signals use only data up to and including the signal tick. Entry is at the
  next tick; positions never overlap.
* Costs:
  * taker: mid fills and the 8.64 bps round trip of ``FeeModel`` (two legs
    for ``pairs``)
  * passive: a limit k bps inside the mid, filled if the HL tape trades
    through it within one tick, maker fee; the exit limit falls back to taker
    at the mid (the multi-gate fill model)
* Every (candidate, parameters, coin) is one strategy and is scored with
  ``run_all.leaderboard``. It also gets an out-of-sample view: the mean in
  the first and second half of its own data period.

Usage::

    python -m hlmicro candidates --capture-dir capture_data_clean
    python -m hlmicro candidates --only basis_reversion,pairs --symbols ETH,BTC
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, default_capture_dir
from ..parse import discover, file_date, tape_per_bucket
from ..parse.books import bucket_venue_means
from ..schemas import BOOK_REST_PREFIX, HL_TRADES_PREFIX
from . import run_all
from .common import rule

log = logging.getLogger(__name__)

TICK_MS = 5_000
TICKS_PER_MIN = 12
TAKER_RT = DEFAULT_FEES.taker_round_trip_bps
MAKER, TAKER = DEFAULT_FEES.maker_bps, DEFAULT_FEES.taker_bps
CEX = ("binance", "okx", "bybit")


# --------------------------------------------------------------------------- data

@dataclass
class Grid:
    symbol: str
    t0_ms: int
    mid: dict                 # venue -> array (NaN where no snapshot)
    obi: dict
    tape: dict | None         # buy_ntl, sell_ntl, count, sum_ntl, sum_sz, px_min, px_max (NaN without tape)

    @property
    def n(self) -> int:
        return len(self.mid[HL_VENUE])

    def ts_ms(self, idx) -> np.ndarray:
        return self.t0_ms + np.asarray(idx, dtype=np.int64) * TICK_MS


def build_grid(capture_dir: Path, symbol: str, start: str, end: str) -> Grid:
    """5-second grid from ``start`` 00:00 to the end of ``end`` (UTC), identical for every coin."""
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    n = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000 - t0) // TICK_MS
    books = discover(capture_dir, BOOK_REST_PREFIX, start, end)
    book = bucket_venue_means(books, symbol, TICK_MS, ("mid", "obi_ntl"))
    mid, obi = {}, {}
    idx_all = book["bucket"].to_numpy(np.int64) - t0 // TICK_MS
    for venue in (HL_VENUE, *CEX):
        m = (book["venue"] == venue).to_numpy() & (idx_all >= 0) & (idx_all < n)
        mid[venue] = np.full(n, np.nan)
        obi[venue] = np.full(n, np.nan)
        mid[venue][idx_all[m]] = book["mid"].to_numpy()[m]
        obi[venue][idx_all[m]] = book["obi_ntl"].to_numpy()[m]
    tapes = discover(capture_dir, HL_TRADES_PREFIX, start, end)
    tape = None
    t = tape_per_bucket(tapes, symbol, TICK_MS, "hl") if tapes else pd.DataFrame()
    if not t.empty:
        i = t["bucket"].to_numpy(np.int64) - t0 // TICK_MS
        ok = (i >= 0) & (i < n)
        tape = {}
        for col in ("buy_ntl", "sell_ntl", "count", "sum_ntl", "sum_sz", "px_min", "px_max"):
            arr = np.full(n, np.nan)
            arr[i[ok]] = t[f"hl_{col}"].to_numpy(np.float64)[ok]
            tape[col] = arr
        # ticks inside the tape period without trades: zero flow (not missing)
        first, last = i[ok].min(), i[ok].max()
        for col in ("buy_ntl", "sell_ntl", "count", "sum_ntl", "sum_sz"):
            seg = tape[col][first:last + 1]
            seg[np.isnan(seg)] = 0.0
    return Grid(symbol, t0, mid, obi, tape)


def _rolling(x: np.ndarray, window: int, fn: str, min_periods: int | None = None) -> np.ndarray:
    s = pd.Series(x).rolling(window, min_periods=min_periods or window)
    return getattr(s, fn)().to_numpy()


def _basis(g: Grid) -> np.ndarray:
    """HL mid vs the median CEX mid, bps (NaN when HL or every CEX is missing)."""
    stack = np.vstack([g.mid[v] for v in CEX])
    with np.errstate(invalid="ignore", divide="ignore"):
        cex = np.nanmedian(np.where(np.isnan(stack).all(axis=0), 0, stack), axis=0)
        cex[np.isnan(stack).all(axis=0)] = np.nan
        return (g.mid[HL_VENUE] / cex - 1) * 1e4


def _edges(mask: np.ndarray) -> np.ndarray:
    """Ticks where ``mask`` turns true (NaN comparisons are already false)."""
    fire = np.zeros(len(mask), bool)
    fire[1:] = mask[1:] & ~mask[:-1]
    return np.flatnonzero(fire)


def _ret(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[lag:] = (x[lag:] / x[:-lag] - 1) * 1e4
    return out


# --------------------------------------------------------------------------- market context

PARTNER = {"BTC": "ETH", "ETH": "BTC", "SOL": "BTC", "HYPE": "BTC", "DOGE": "BTC", "LINK": "BTC"}


def context_features(grids: dict, coin: str, sig: np.ndarray) -> pd.DataFrame:
    """Market context at signal ticks, from trailing data only (scripts/cointegration_study.py).

    * ``market_15m``: mean 15-min HL return of the other coins in ``grids``, bps
      (< 0: market-wide dip)
    * ``coint_z``: z-score of the residual of log(coin) on log(partner), rolling OLS and
      std over the trailing 3 days of 1-min HL mids (< 0: coin cheap vs its partner;
      partner fixed in :data:`PARTNER`)
    """
    hl = {c: pd.Series(g.mid[HL_VENUE]).ffill(limit=12).to_numpy() for c, g in grids.items()}
    r15 = {c: _ret(hl[c], 180) for c in grids}
    other = [c for c in grids if c != coin]
    with np.errstate(invalid="ignore"):
        stack = np.vstack([r15[o][sig] for o in other])
        cnt = np.isfinite(stack).sum(axis=0)
        market = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
    minute = np.arange(0, len(hl[coin]), TICKS_PER_MIN)
    y, x = pd.Series(np.log(hl[coin][minute])), pd.Series(np.log(hl[PARTNER[coin]][minute]))
    w = 3 * 1440
    mx, my = x.rolling(w, min_periods=w // 2).mean(), y.rolling(w, min_periods=w // 2).mean()
    beta = ((x * y).rolling(w, min_periods=w // 2).mean() - mx * my) / \
           ((x * x).rolling(w, min_periods=w // 2).mean() - mx * mx)
    resid = y - (my + beta * (x - mx))
    cz = (resid / resid.rolling(w, min_periods=w // 2).std()).to_numpy()
    return pd.DataFrame({"market_15m": market, "coint_z": cz[sig // TICKS_PER_MIN]})


# --------------------------------------------------------------------------- execution models

def taker_fixed(mid, idx, sides, hold, cost=TAKER_RT):
    """Enter at the next tick's mid, exit ``hold`` ticks later at the mid."""
    out, busy, n = [], -1, len(mid)
    for i, s in zip(idx, sides):
        e, x = i + 1, i + 1 + hold
        if i <= busy or x >= n or not (mid[e] > 0 and mid[x] > 0):
            continue
        out.append((e, s * (mid[x] / mid[e] - 1) * 1e4 - cost))
        busy = x
    return out


def taker_until(mid, idx, sides, max_hold, done: Callable[[int, int], np.ndarray], cost=TAKER_RT):
    """As :func:`taker_fixed`, but exit at the first tick where ``done(side, e)`` is true
    (an array over ticks e+1 .. e+max_hold), else after ``max_hold``."""
    out, busy, n = [], -1, len(mid)
    for i, s in zip(idx, sides):
        e = i + 1
        if i <= busy or e + max_hold >= n or not mid[e] > 0:
            continue
        hit = done(s, e)
        x = e + 1 + int(np.argmax(hit)) if hit.any() else e + max_hold
        while x > e and not mid[x] > 0:
            x -= 1
        if x == e:
            continue
        out.append((e, s * (mid[x] / mid[e] - 1) * 1e4 - cost))
        busy = x
    return out


def hl_tick(price: float) -> float:
    """Hyperliquid price increment: at most 5 significant figures, at most 6 decimals."""
    return max(10.0 ** (np.floor(np.log10(price)) - 4), 1e-6)


def passive(mid, lo, hi, idx, sides, hold, k, through_ticks: int = 0):
    """Limit k bps inside the mid; filled if the next tick trades at (``through_ticks`` = 0)
    or that many price increments through it; exit limit k bps inside at ``hold`` with one
    tick of patience (same fill rule), else taker at the mid."""
    out, busy, n = [], -1, len(mid)
    for i, s in zip(idx, sides):
        if i <= busy or not mid[i] > 0 or i + hold + 2 >= n:
            continue
        post = mid[i] * (1 - s * k * 1e-4)
        margin = through_ticks * hl_tick(mid[i])
        filled = lo[i + 1] <= post - margin if s > 0 else hi[i + 1] >= post + margin
        if not (np.isfinite(lo[i + 1]) and filled):
            continue
        e = i + 1
        ref = e + hold
        if not mid[ref] > 0:
            continue
        post_x = mid[ref] * (1 + s * k * 1e-4)
        exit_fill = hi[ref + 1] >= post_x + margin if s > 0 else lo[ref + 1] <= post_x - margin
        if np.isfinite(hi[ref + 1]) and exit_fill:
            gross, fee = s * (post_x / post - 1) * 1e4, 2 * MAKER
        elif mid[ref + 1] > 0:
            gross, fee = s * (mid[ref + 1] / post - 1) * 1e4, MAKER + TAKER
        else:
            continue
        out.append((e, gross - fee))
        busy = ref + 1
    return out


def _execute(g: Grid, idx, sides, hold, execution):
    if execution == "taker":
        return taker_fixed(g.mid[HL_VENUE], idx, sides, hold)
    return passive(g.mid[HL_VENUE], g.tape["px_min"], g.tape["px_max"], idx, sides, hold, float(execution))


# --------------------------------------------------------------------------- candidates

def basis_reversion(g: Grid, dev_bps, hold_min, exit_rule):
    """HL vs CEX-median basis, minus its trailing 1 h mean. Buy HL when it is dev_bps cheap,
    sell when dev_bps rich. Exit after hold_min, or (``revert``) when the basis is back at
    its mean, at most hold_min."""
    hl = g.mid[HL_VENUE]
    basis = _basis(g)
    dev = basis - _rolling(basis, 720, "mean", 360)
    with np.errstate(invalid="ignore"):
        lo_, hi_ = dev <= -dev_bps, dev >= dev_bps
    idx = np.concatenate([_edges(lo_), _edges(hi_)])
    sides = np.concatenate([np.ones(len(_edges(lo_))), -np.ones(len(_edges(hi_)))])
    order = np.argsort(idx, kind="stable")
    idx, sides = idx[order], sides[order]
    hold = hold_min * TICKS_PER_MIN
    if exit_rule == "time":
        return taker_fixed(hl, idx, sides, hold)
    return taker_until(hl, idx, sides, hold, lambda s, e: (s * dev[e + 1:e + hold + 1]) >= 0)


def passive_leadlag(g: Grid, move_bps, window_s, hold_s, k):
    """Binance moved >= move_bps in window_s while HL moved less than half of it: post a
    maker order on HL in the move's direction; time exit."""
    w = window_s // 5
    lead, follow = _ret(g.mid["binance"], w), _ret(g.mid[HL_VENUE], w)
    with np.errstate(invalid="ignore"):
        up = (lead >= move_bps) & (follow < lead / 2)
        down = (lead <= -move_bps) & (follow > lead / 2)
    iu, idn = _edges(up), _edges(down)
    idx = np.concatenate([iu, idn])
    sides = np.concatenate([np.ones(len(iu)), -np.ones(len(idn))])
    order = np.argsort(idx, kind="stable")
    return passive(g.mid[HL_VENUE], g.tape["px_min"], g.tape["px_max"], idx[order], sides[order], hold_s // 5, k)


def _flow(g: Grid):
    buy = _rolling(g.tape["buy_ntl"], 12, "sum")
    sell = _rolling(g.tape["sell_ntl"], 12, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        tfi = (buy - sell) / (buy + sell)
    count = g.tape["count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = _rolling(count, 180, "std")
        cz = (count - _rolling(count, 180, "mean")) / np.where(sd > 0, sd, np.nan)
    return tfi, cz


def flow_fade(g: Grid, tfi_min, move_bps, hold_min, execution):
    """Trade burst (count z >= 2) that is one-sided (|TFI 60 s| >= tfi_min) and has moved the
    price >= move_bps in 60 s: take the other side."""
    tfi, cz = _flow(g)
    r = _ret(g.mid[HL_VENUE], 12)
    with np.errstate(invalid="ignore"):
        sell_climax = (tfi <= -tfi_min) & (cz >= 2) & (r <= -move_bps)
        buy_climax = (tfi >= tfi_min) & (cz >= 2) & (r >= move_bps)
    il, ish = _edges(sell_climax), _edges(buy_climax)
    idx = np.concatenate([il, ish])
    sides = np.concatenate([np.ones(len(il)), -np.ones(len(ish))])
    order = np.argsort(idx, kind="stable")
    return _execute(g, idx[order], sides[order], hold_min * TICKS_PER_MIN, execution)


def vol_norm_signal(g: Grid, z_min, vwap_min=5):
    """Signal ticks of :func:`vol_norm_dip`."""
    w = vwap_min * TICKS_PER_MIN
    ntl = _rolling(g.tape["sum_ntl"], w, "sum")
    sz = _rolling(g.tape["sum_sz"], w, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = ntl / np.where(sz > 0, sz, np.nan)
        dev = (g.mid[HL_VENUE] - vwap) / vwap * 1e4
    sd = _rolling(dev, 17_280, "std", 8_640)
    tfi, _ = _flow(g)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = (dev / sd <= -z_min) & (tfi >= 0) & (g.obi[HL_VENUE] >= 0)
    return _edges(sig)


def vol_norm_dip(g: Grid, z_min, hold_min, k, vwap_min=5, through_ticks=0):
    """vwap_dev_buy with the dip measured in standard deviations of the coin's own VWAP
    deviation over the trailing 24 h (flow >= 0 and HL imbalance >= 0, as the original)."""
    idx = vol_norm_signal(g, z_min, vwap_min)
    return passive(g.mid[HL_VENUE], g.tape["px_min"], g.tape["px_max"], idx, np.ones(len(idx)),
                   hold_min * TICKS_PER_MIN, float(k), through_ticks)


def momentum(g: Grid, lookback_h, k_sigma, hold_h, mode):
    """At each full hour: the HL return over lookback_h against k_sigma x its trailing 7-day
    standard deviation (hourly samples). ``follow`` trades in the move's direction,
    ``fade`` against it; time exit after hold_h."""
    mid = g.mid[HL_VENUE]
    hour = 720
    points = np.arange(hour, g.n, hour)                  # grid starts at 00:00 UTC
    lb = lookback_h * hour
    r = _ret(mid, lb)[points]
    sd = pd.Series(r).rolling(168, min_periods=72).std().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        up, down = r >= k_sigma * sd, r <= -k_sigma * sd
    sign = 1 if mode == "follow" else -1
    idx = np.concatenate([points[up], points[down]])
    sides = np.concatenate([np.full(up.sum(), sign), np.full(down.sum(), -sign)])
    order = np.argsort(idx, kind="stable")
    return taker_fixed(mid, idx[order], sides[order], hold_h * hour)


def funding_proxy(g: Grid, pct, lead_min):
    """HL funding follows the HL premium. Before each hourly settlement, rank the last
    hour's mean basis (HL vs CEX median) within the trailing 7 days: short the top pct%,
    long the bottom pct%, from lead_min before the hour to 5 min after."""
    hl = g.mid[HL_VENUE]
    basis = _basis(g)
    hour, lead = 720, lead_min * TICKS_PER_MIN
    sig_ticks = np.arange(hour, g.n, hour) - lead - 1  # signal tick; entry next tick = T - lead
    def window_mean(t):
        w = basis[max(0, t - hour):t + 1]
        ok = np.isfinite(w)
        return w[ok].mean() if t > 0 and ok.any() else np.nan
    prem = np.array([window_mean(t) for t in sig_ticks])
    s = pd.Series(prem)
    hi_q = s.rolling(168, min_periods=72).quantile(1 - pct / 100).shift(1).to_numpy()
    lo_q = s.rolling(168, min_periods=72).quantile(pct / 100).shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        short, long_ = prem >= hi_q, prem <= lo_q
    idx = np.concatenate([sig_ticks[short], sig_ticks[long_]])
    sides = np.concatenate([-np.ones(short.sum()), np.ones(long_.sum())])
    order = np.argsort(idx, kind="stable")
    return taker_fixed(hl, idx[order], sides[order], lead + 5 * TICKS_PER_MIN)


def pair_z(a: np.ndarray, b: np.ndarray, lookback_h: int) -> np.ndarray:
    """z-score of log(A/B) per minute against the trailing lookback_h mean and std."""
    m = np.arange(0, len(a), TICKS_PER_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.log(a[m] / b[m])
    s = pd.Series(ratio)
    L = lookback_h * 60
    mean = s.rolling(L, min_periods=L // 2).mean().shift(1).to_numpy()
    sd = s.rolling(L, min_periods=L // 2).std().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ratio - mean) / np.where(sd > 0, sd, np.nan)      # flat window: no signal


def pairs(ga: Grid, gb: Grid, lookback_h, z_entry, max_hold_h):
    """log(A/B) on minute samples against its trailing mean and std: z >= z_entry -> short A,
    long B; z <= -z_entry -> long A, short B. Exit when z crosses 0 or after max_hold_h.
    Two taker legs: cost 2 x 8.64 bps on one leg's notional."""
    a, b = ga.mid[HL_VENUE], gb.mid[HL_VENUE]
    m = np.arange(0, ga.n, TICKS_PER_MIN)
    z = pair_z(a, b, lookback_h)
    out, busy, hold = [], -1, max_hold_h * 60
    with np.errstate(invalid="ignore"):
        hi_, lo_ = z >= z_entry, z <= -z_entry
    for j in np.sort(np.concatenate([_edges(hi_), _edges(lo_)])):
        e = j + 1
        if j <= busy or e + hold >= len(m):
            continue
        side = -1.0 if z[j] > 0 else 1.0                # +1: long A / short B
        seg = side * z[e + 1:e + hold + 1]
        x = e + 1 + int(np.argmax(seg >= 0)) if (seg >= 0).any() else e + hold
        ra = a[m[x]] / a[m[e]] - 1
        rb = b[m[x]] / b[m[e]] - 1
        if not np.isfinite(ra + rb):
            continue
        out.append((m[e], side * (ra - rb) * 1e4 - 2 * TAKER_RT))
        busy = x
    return out


@dataclass
class Candidate:
    name: str
    fn: Callable
    grid: dict
    needs_tape: bool
    cost: str
    describe: Callable[[dict], str]
    cost_bps: float = float("nan")
    pair: bool = False


CANDIDATES = {c.name: c for c in [
    Candidate("basis_reversion", basis_reversion,
              {"dev_bps": [5, 10, 20], "hold_min": [1, 5, 15], "exit_rule": ["time", "revert"]}, False,
              f"taker {TAKER_RT:.2f} bps RT", lambda p: f"dev {p['dev_bps']}bps vs 1h mean, "
              f"{'exit at mean, max ' if p['exit_rule'] == 'revert' else 'hold '}{p['hold_min']}m", TAKER_RT),
    Candidate("passive_leadlag", passive_leadlag,
              {"move_bps": [10, 15, 20], "window_s": [30, 60], "hold_s": [30, 60, 300], "k": [0.0, 1.0]}, True,
              "maker in, maker/taker out (simulated)",
              lambda p: f"binance >= {p['move_bps']}bps/{p['window_s']}s, limit {p['k']:g}bp, hold {p['hold_s']}s"),
    Candidate("flow_fade", flow_fade,
              {"tfi_min": [0.5, 0.7], "move_bps": [15, 30], "hold_min": [5, 15, 30], "execution": ["taker", 1.0]},
              True, "taker 8.64 bps RT or simulated passive 1bp",
              lambda p: f"|TFI|>={p['tfi_min']}, move>={p['move_bps']}bps/60s, hold {p['hold_min']}m, "
                        f"{'taker' if p['execution'] == 'taker' else 'passive 1bp'}"),
    Candidate("vol_norm_dip", vol_norm_dip, {"z_min": [2, 3, 4], "hold_min": [30, 60], "k": [1.0, 2.0]}, True,
              "maker in, maker/taker out (simulated)",
              lambda p: f"dip <= -{p['z_min']} sd below VWAP {p.get('vwap_min', 5)}m, hold {p['hold_min']}m, "
                        f"limit {p['k']:g}bp"),
    Candidate("momentum", momentum,
              {"lookback_h": [1, 4], "k_sigma": [1, 2], "hold_h": [1, 4], "mode": ["follow", "fade"]}, False,
              f"taker {TAKER_RT:.2f} bps RT",
              lambda p: f"{p['mode']} {p['lookback_h']}h move >= {p['k_sigma']} sd, hold {p['hold_h']}h", TAKER_RT),
    Candidate("funding_proxy", funding_proxy, {"pct": [10, 20], "lead_min": [5, 10]}, False,
              f"taker {TAKER_RT:.2f} bps RT",
              lambda p: f"premium top/bottom {p['pct']}% (7d), T-{p['lead_min']}m..T+5m", TAKER_RT),
    Candidate("pairs", pairs, {"lookback_h": [6, 24], "z_entry": [2, 3], "max_hold_h": [4, 24]}, False,
              f"2 taker legs, {2 * TAKER_RT:.2f} bps",
              lambda p: f"z>={p['z_entry']} vs {p['lookback_h']}h, exit at mean, max {p['max_hold_h']}h",
              2 * TAKER_RT, pair=True),
]}
PAIRS = [("ETH", "BTC"), ("SOL", "ETH"), ("SOL", "BTC")]


# --------------------------------------------------------------------------- run

def _combos(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run(capture_dir: Path, symbols: list[str], only: list[str] | None, start: str, end: str):
    """All candidate strategies as ``run_all.Strategy`` plus a run log."""
    chosen = [CANDIDATES[c] for c in (only or CANDIDATES)]
    need = set(symbols)
    if any(c.pair for c in chosen):
        need |= {s for p in PAIRS for s in p if set(p) <= set(symbols) or not symbols}
    grids, log_rows = {}, []
    for sym in sorted(need):
        t = time.time()
        grids[sym] = build_grid(capture_dir, sym, start, end)
        log_rows.append(("grid", sym, time.time() - t, 0))
    strategies = []
    for c in chosen:
        t = time.time()
        targets = [p for p in PAIRS if p[0] in grids and p[1] in grids] if c.pair else \
            [s for s in symbols if s in grids]
        n_before = len(strategies)
        for target in targets:
            g = grids[target] if not c.pair else None
            if c.needs_tape and (g is None or g.tape is None):
                continue
            for params in _combos(c.grid):
                trades = c.fn(grids[target[0]], grids[target[1]], **params) if c.pair else c.fn(g, **params)
                if not trades:
                    continue
                idx = np.array([tr[0] for tr in trades], dtype=np.int64)
                ref = grids[target[0]] if c.pair else g
                strategies.append(run_all.Strategy(
                    "candidates", "/".join(target) if c.pair else target, c.name, c.describe(params), c.cost,
                    ref.ts_ms(idx), np.array([tr[1] for tr in trades]), c.cost_bps))
        log_rows.append((c.name, len(targets), time.time() - t, len(strategies) - n_before))
    return strategies, log_rows


def halves(strategies: list, table: pd.DataFrame) -> pd.DataFrame:
    """Mean in the first and second half of each strategy's own trading period."""
    a, b, na, nb = [], [], [], []
    by_key = {(s.symbol, s.name, s.params): s for s in strategies}
    for _, r in table.iterrows():
        s = by_key[(r["symbol"], r["strategy"], r["params"])]
        mid_t = s.ts_ms.min() + (s.ts_ms.max() - s.ts_ms.min()) / 2
        first = s.ts_ms < mid_t
        a.append(s.net_bps[first].mean() if first.any() else np.nan)
        b.append(s.net_bps[~first].mean() if (~first).any() else np.nan)
        na.append(int(first.sum()))
        nb.append(int((~first).sum()))
    return table.assign(first_half_mean=a, second_half_mean=b, n_first=na, n_second=nb,
                        both_halves_positive=(np.array(a) > 0) & (np.array(b) > 0))


def render(table: pd.DataFrame, log_rows, top: int, min_trades: int) -> str:
    out = [rule(120), "Candidate strategies", rule(120)]
    for name, target, secs, n in log_rows:
        if name != "grid":
            out.append(f"  {name:<18} {target} targets  {n:>4} strategies  {secs:6.1f}s")
    elig = table[table["eligible"]]
    counts = elig["verdict"].value_counts()
    out += ["", f"{len(table)} strategies, {len(elig)} with n >= {min_trades} ranked: "
                f"significant {counts.get('significant', 0)}, promising {counts.get('promising', 0)}, "
                f"weak {counts.get('weak', 0)}, negative {counts.get('negative', 0)}; "
                f"positive in both halves: {int(elig['both_halves_positive'].sum())}", "",
            "Per candidate (ranked strategies):"]
    for name, grp in elig.groupby("strategy"):
        best = grp.iloc[0]
        out.append(f"  {name:<18} {len(grp):>4} ranked | median mean {grp['mean_net_bps'].median():+6.1f} bps, "
                   f"gross {grp['mean_gross_bps'].median():+6.1f} | best: {best['symbol']} {best['params']} "
                   f"mean {best['mean_net_bps']:+.1f}, t {best['t']:+.2f}, halves "
                   f"{best['first_half_mean']:+.1f}/{best['second_half_mean']:+.1f}")
    out += ["", rule(120, "-"), f"TOP {top} BY t (net bps per trade; halves = own period split in two)", rule(120, "-"),
            f"  {'#':>3} {'candidate':<16} {'sym':<8} {'parameters':<52} {'n':>5} {'mean':>7} {'gross':>6} "
            f"{'win%':>5} {'t':>6} {'1st/2nd half':>14}  verdict"]
    for _, r in elig.head(top).iterrows():
        gross = "" if pd.isna(r["mean_gross_bps"]) else f"{r['mean_gross_bps']:+.1f}"
        out.append(f"  {int(r['rank']):>3} {r['strategy']:<16} {r['symbol']:<8} {r['params'][:52]:<52} {int(r['n']):>5} "
                   f"{r['mean_net_bps']:>+7.2f} {gross:>6} {100 * r['win_rate']:>5.1f} {r['t']:>+6.2f} "
                   f"{r['first_half_mean']:>+6.1f}/{r['second_half_mean']:<+7.1f} {r['verdict']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro candidates", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--symbols", type=csv_list(str.upper), default=["BTC", "ETH", "SOL", "HYPE", "DOGE", "LINK"])
    parser.add_argument("--only", type=csv_list(str), default=None, help=f"subset of {', '.join(CANDIDATES)}")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--outdir", type=Path, default=None, help="default results/candidates/<timestamp>")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.verbose:
        logging.getLogger("hlmicro").setLevel(logging.WARNING)
    if args.only and set(args.only) - set(CANDIDATES):
        parser.error(f"unknown candidates {sorted(set(args.only) - set(CANDIDATES))}")
    books = discover(args.capture_dir, BOOK_REST_PREFIX, args.start, args.end)
    if not books:
        parser.error(f"no {BOOK_REST_PREFIX} files in {args.capture_dir}")
    start, end = args.start or file_date(books[0]), args.end or file_date(books[-1])
    strategies, log_rows = run(args.capture_dir, args.symbols, args.only, start, end)
    table = run_all.leaderboard(strategies, n_folds=4, min_trades=args.min_trades)
    if table.empty:
        print("no candidate produced trades")
        return 1
    table = halves(strategies, table)
    text = render(table, log_rows, args.top, args.min_trades)
    print(text)
    outdir = args.outdir or Path("results/candidates") / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "strategies.csv", index=False)
    (outdir / "leaderboard.md").write_text("# Candidate strategies\n\n```\n" + text + "\n```\n")
    print(f"\nwrote {outdir}/leaderboard.md and {outdir}/strategies.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
