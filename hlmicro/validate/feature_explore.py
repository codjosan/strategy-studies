"""Single- and two-feature signal discovery on a one-minute grid, with a
train / validate / holdout split.

Builds per-minute features from the 5-second books (cross-venue and
per-venue imbalance, spread) and the Hyperliquid trade tape, grouped by
hypothesis:

=====  =====================================================================
A      levels and z-scores of imbalance, venue consensus, flow, volume
H1     imbalance velocity and acceleration
H2     trade-size regime (mean trade size, minutes with a large trade)
H3     Hyperliquid imbalance against the other venues
H4     aggression: trade-count imbalance, longest same-side run
H5     spread widening
H6     short-window (5-30 minute) imbalance z-scores and 1-3 minute velocity
=====  =====================================================================

Z-scores here compare each minute with the preceding window (population
standard deviation); undefined values are 0. The large-trade cut-off is the
95th percentile of trade notional over the whole sample.

Each rule is ``feature > +thr`` / ``< -thr``, traded with (``follow``) or
against (``fade``) the feature, held for a fixed horizon from the trigger
minute's mid (``--next-bar-entry``: the following minute's), with at most one
entry per ``cooldown_min``. Positions may overlap when the horizon exceeds
the cooldown. Two-feature rules require a base rule and a confirming rule to
agree in direction.

A rule is validated when the training and validation segments each have
n >= 10, win rate >= 53%, positive mean and Sharpe >= 0.10. The holdout is
reported but never used for selection. Many rules are tested, so expect
false positives; the holdout section is the honest estimate.

Usage::

    python -m hlmicro feature-explore --symbol SOL
    python -m hlmicro feature-explore --symbol SOL --horizons 1,3,5,10,15,30 --cooldown-min 15
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, default_capture_dir
from ..parse import bucket_venue_means, discover, load_tape
from ..schemas import BOOK_REST_PREFIX, HL_TRADES_PREFIX
from .common import rule

log = logging.getLogger(__name__)

MINUTE_MS = 60_000
WARMUP_MIN = 60
OTHER_VENUES = ("binance", "okx", "bybit")


def _both(group: str, feature: str, thresholds: tuple) -> list[tuple]:
    return [(group, feature, thresholds, "follow"), (group, feature, thresholds, "fade")]


SINGLE_SPECS: list[tuple[str, str, tuple, str]] = [
    *_both("A", "obi_ntl_avg", (0.2, 0.3, 0.4, 0.5)),
    *_both("A", "obi_zscore_15m", (1.0, 1.5, 2.0, 2.5)),
    *_both("A", "obi_zscore_60m", (1.0, 1.5, 2.0, 2.5)),
    *_both("A", "hl_obi_ntl", (0.2, 0.3, 0.4, 0.5)),
    *_both("A", "obi_consensus", (1.5, 2.5, 3.5)),
    *_both("A", "flow_imb_5m", (0.3, 0.5, 0.7)),
    *_both("A", "flow_imb_15m", (0.3, 0.5, 0.7)),
    *_both("A", "vol_z_15m", (1.5, 2.0, 2.5)),
    *_both("H1", "obi_velocity_5m", (0.1, 0.2, 0.3, 0.4)),
    *_both("H1", "obi_velocity_15m", (0.2, 0.3, 0.4, 0.5)),
    *_both("H1", "obi_velocity_30m", (0.2, 0.4, 0.6)),
    ("H1", "hl_obi_velocity_5m", (0.1, 0.2, 0.3), "follow"),
    ("H1", "hl_obi_velocity_15m", (0.2, 0.3, 0.4), "follow"),
    *_both("H1", "obi_acceleration_5m", (0.1, 0.2, 0.3)),
    *_both("H1", "obi_velocity_5m_z", (1.5, 2.0, 2.5)),
    *_both("H1", "obi_velocity_15m_z", (1.5, 2.0, 2.5)),
    *_both("H2", "mean_trade_sz_z_15m", (1.5, 2.0, 2.5)),
    *_both("H2", "mean_trade_sz_z_60m", (1.5, 2.0, 2.5)),
    *_both("H2", "large_count_5m", (2, 3, 5)),
    ("H2", "large_count_15m", (3, 5, 10), "follow"),
    *_both("H2", "large_count_z_60m", (1.5, 2.0, 2.5)),
    *_both("H3", "hl_vs_avg3_obi_diff", (0.1, 0.2, 0.3, 0.4)),
    *_both("H3", "hl_vs_binance_diff", (0.1, 0.2, 0.3, 0.4)),
    *_both("H3", "hl_vs_avg3_diff_z_60m", (1.5, 2.0, 2.5)),
    *_both("H3", "hl_vs_binance_diff_z_60m", (1.5, 2.0, 2.5)),
    *_both("H3", "max_pairwise_z_60m", (1.5, 2.0, 2.5)),
    *_both("H4", "count_imb_5m", (0.3, 0.5, 0.7)),
    *_both("H4", "count_imb_15m", (0.3, 0.5, 0.7)),
    *_both("H4", "count_imb_z_60m", (1.5, 2.0, 2.5)),
    ("H4", "longest_streak_5m", (3, 5, 8, 12), "follow"),
    *_both("H4", "longest_streak_z_60m", (1.5, 2.0, 2.5)),
    *_both("H5", "spread_bps_z_15m", (1.5, 2.0, 2.5)),
    *_both("H5", "spread_bps_z_60m", (1.5, 2.0, 2.5)),
    *_both("H5", "spread_widening_ratio", (1.5, 2.0, 3.0)),
    *_both("H6", "obi_zscore_5m", (2.0, 2.5, 3.0)),
    *_both("H6", "obi_zscore_10m", (1.5, 2.0, 2.5, 3.0)),
    *_both("H6", "obi_zscore_20m", (1.5, 2.0, 2.5, 3.0)),
    *_both("H6", "obi_zscore_30m", (1.5, 2.0, 2.5, 3.0)),
    *_both("H6", "hl_obi_zs_5m", (2.0, 2.5, 3.0)),
    *_both("H6", "hl_obi_zs_10m", (1.5, 2.0, 2.5)),
    *_both("H6", "hl_obi_zs_30m", (1.5, 2.0, 2.5)),
    *_both("H6", "obi_velocity_1m_z", (2.0, 2.5, 3.0)),
    *_both("H6", "obi_velocity_3m_z", (2.0, 2.5, 3.0)),
]

# (feature, threshold, direction). Bases on short z-scores are also tested at short horizons.
BASES = [("obi_zscore_60m", 2.0, "follow"), ("obi_zscore_60m", 2.5, "follow"),
         ("obi_velocity_15m_z", 2.0, "follow"), ("hl_vs_avg3_diff_z_60m", 2.0, "follow"),
         ("hl_vs_avg3_diff_z_60m", 2.0, "fade"), ("count_imb_z_60m", 2.0, "follow"),
         ("large_count_z_60m", 2.0, "follow"), ("obi_zscore_30m", 2.0, "follow"),
         ("obi_zscore_30m", 2.5, "follow"), ("obi_zscore_15m", 2.0, "follow"),
         ("obi_zscore_15m", 2.5, "follow")]
SHORT_BASES = {"obi_zscore_30m", "obi_zscore_15m"}
BASE_HORIZONS, SHORT_BASE_HORIZONS = (60, 120, 240), (5, 15, 30, 60, 120)
CONFIRMERS = [("obi_consensus", 1.5, "follow"), ("flow_imb_5m", 0.3, "follow"), ("flow_imb_5m", 0.3, "fade"),
              ("vol_z_15m", 1.5, "follow"), ("count_imb_5m", 0.3, "follow"), ("spread_bps_z_15m", 1.5, "follow"),
              ("obi_zscore_5m", 1.5, "follow"), ("obi_zscore_5m", 2.0, "follow"),
              ("obi_zscore_10m", 1.5, "follow"), ("obi_zscore_10m", 2.0, "follow"),
              ("obi_zscore_15m", 1.5, "follow"), ("obi_velocity_1m_z", 2.0, "follow")]


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    horizons_min: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15, 30, 60, 120, 240])
    cooldown_min: int = 60
    train_frac: float = 0.50
    validate_frac: float = 0.25
    min_trades: int = 10
    min_n: int = 20
    top: int = 30
    next_bar_entry: bool = False
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None


# --- rolling helpers: 0 where undefined, as in the original explorer ---------

def _zs(x: pd.Series, window: int) -> pd.Series:
    """Z-score against the preceding ``window`` values (population std)."""
    mean = x.rolling(window).mean().shift(1)
    sd = x.rolling(window).std(ddof=0).shift(1)
    return ((x - mean) / sd.where(sd > 0)).fillna(0.0)


def _diff(x: pd.Series, lag: int) -> pd.Series:
    return (x - x.shift(lag)).fillna(0.0)


def _rsum(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).sum().fillna(0.0)


def _rimb(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    sa, sb = a.rolling(window).sum(), b.rolling(window).sum()
    total = sa + sb
    return ((sa - sb) / total.where(total > 0)).fillna(0.0)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den.abs() > 1e-9)).fillna(0.0)


# --- loading ----------------------------------------------------------------

def _book(files: list[Path], symbol: str) -> pd.DataFrame:
    per_venue = bucket_venue_means(files, symbol, MINUTE_MS, ("mid", "obi_ntl", "spread_bps"))
    if per_venue.empty:
        return pd.DataFrame()
    full = pd.RangeIndex(int(per_venue["bucket"].min()), int(per_venue["bucket"].max()) + 1, name="minute")
    wide = {c: per_venue.pivot(index="bucket", columns="venue", values=c).reindex(full)
            for c in ("mid", "obi_ntl", "spread_bps")}
    obi = wide["obi_ntl"]
    out = pd.DataFrame(index=full)
    out["mid"] = wide["mid"].mean(axis=1).fillna(0.0)
    out["obi_ntl_avg"] = obi.mean(axis=1).fillna(0.0)
    out["obi_consensus"] = ((obi > 0.2).sum(axis=1) - (obi < -0.2).sum(axis=1)).astype(float)
    out["spread_bps"] = wide["spread_bps"].mean(axis=1).fillna(0.0)
    out["obi_disp"] = (obi.max(axis=1) - obi.min(axis=1)).fillna(0.0)
    for v in (HL_VENUE, *OTHER_VENUES):
        out[f"{v}_obi"] = obi[v].fillna(0.0) if v in obi else 0.0
    return out


def _trades(files: list[Path], symbol: str, index: pd.Index) -> tuple[pd.DataFrame, float]:
    """Per-minute flow aggregates and the 95th-percentile trade notional."""
    parts, notionals = [], []
    for path in files:
        tape = load_tape([path], symbol)
        if tape.empty:
            continue
        minute = tape["ts_ms"].to_numpy() // MINUTE_MS
        ntl = (tape["px"] * tape["sz"]).to_numpy()
        sign = tape["sign"].to_numpy()
        new_run = np.ones(len(tape), dtype=bool)
        new_run[1:] = (sign[1:] != sign[:-1]) | (minute[1:] != minute[:-1])
        run_id = np.cumsum(new_run)
        run_len = np.bincount(run_id)[run_id]       # length of the same-side run each trade belongs to
        parts.append(pd.DataFrame({
            "minute": minute, "buy_vol": np.where(sign > 0, ntl, 0.0), "sell_vol": np.where(sign < 0, ntl, 0.0),
            "buy_count": (sign > 0).astype(float), "sell_count": (sign < 0).astype(float),
            "ntl": ntl, "run": run_len,
        }).groupby("minute").agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
                                 buy_count=("buy_count", "sum"), sell_count=("sell_count", "sum"),
                                 n_trades=("ntl", "size"), sum_ntl=("ntl", "sum"), max_ntl=("ntl", "max"),
                                 longest_streak=("run", "max")))
        notionals.append(ntl.astype(np.float32))
    cols = ["buy_vol", "sell_vol", "buy_count", "sell_count", "n_trades", "sum_ntl", "max_ntl", "longest_streak"]
    if not parts:
        return pd.DataFrame(0.0, index=index, columns=cols), np.inf
    # A minute split across two daily files: add the flows, keep the maxima.
    agg = pd.concat(parts).groupby(level=0).agg({c: ("max" if c in ("max_ntl", "longest_streak") else "sum")
                                                for c in cols})
    return agg.reindex(index, fill_value=0.0), float(np.percentile(np.concatenate(notionals), 95))


def build_features(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Minute grid with ``mid`` and every feature used by the specs."""
    book_files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not book_files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    b = _book(book_files, cfg.symbol)
    if b.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    trade_files = discover(cfg.capture_dir, HL_TRADES_PREFIX, cfg.start, cfg.end)
    t, large_thr = _trades(trade_files, cfg.symbol, b.index)

    f = pd.DataFrame(index=b.index)
    f["mid"] = b["mid"]
    obi, hl = b["obi_ntl_avg"], b[f"{HL_VENUE}_obi"]
    f["obi_ntl_avg"], f["obi_consensus"], f["hl_obi_ntl"], f["obi_disp"] = obi, b["obi_consensus"], hl, b["obi_disp"]
    for w in (5, 10, 15, 20, 30, 60):
        f[f"obi_zscore_{w}m"] = _zs(obi, w)
    for w in (5, 10, 15, 30, 60):
        f[f"hl_obi_zs_{w}m"] = _zs(hl, w)
    for lag in (1, 3, 5, 15, 30):
        f[f"obi_velocity_{lag}m"] = _diff(obi, lag)
    f["hl_obi_velocity_5m"], f["hl_obi_velocity_15m"] = _diff(hl, 5), _diff(hl, 15)
    f["obi_acceleration_5m"] = _diff(f["obi_velocity_5m"], 5)
    f["obi_velocity_5m_z"], f["obi_velocity_15m_z"] = _zs(f["obi_velocity_5m"], 60), _zs(f["obi_velocity_15m"], 60)
    f["obi_velocity_1m_z"], f["obi_velocity_3m_z"] = _zs(f["obi_velocity_1m"], 30), _zs(f["obi_velocity_3m"], 30)

    mean_sz = _ratio(t["sum_ntl"], t["n_trades"])
    f["mean_trade_sz_z_15m"], f["mean_trade_sz_z_60m"] = _zs(mean_sz, 15), _zs(mean_sz, 60)
    large = (t["max_ntl"] > large_thr).astype(float)
    f["large_count_5m"], f["large_count_15m"], f["large_count_z_60m"] = _rsum(large, 5), _rsum(large, 15), _zs(large, 60)

    others = b[[f"{v}_obi" for v in OTHER_VENUES]]
    avg3 = others.where(others != 0).mean(axis=1).fillna(0.0)
    f["hl_vs_avg3_obi_diff"] = hl - avg3
    f["hl_vs_binance_diff"] = hl - b["binance_obi"]
    f["hl_vs_avg3_diff_z_60m"] = _zs(f["hl_vs_avg3_obi_diff"], 60)
    f["hl_vs_binance_diff_z_60m"] = _zs(f["hl_vs_binance_diff"], 60)
    venues = b[[f"{v}_obi" for v in (HL_VENUE, *OTHER_VENUES)]].replace(0.0, np.nan)
    pairwise = (venues.max(axis=1) - venues.min(axis=1)).where(venues.count(axis=1) >= 2).fillna(0.0)
    f["max_pairwise_z_60m"] = _zs(pairwise, 60)

    count_imb = _ratio(t["buy_count"] - t["sell_count"], t["buy_count"] + t["sell_count"])
    f["count_imb_5m"], f["count_imb_15m"] = _rimb(t["buy_count"], t["sell_count"], 5), _rimb(t["buy_count"], t["sell_count"], 15)
    f["count_imb_z_60m"] = _zs(count_imb, 60)
    f["longest_streak_5m"] = _rsum(t["longest_streak"], 5) / 5
    f["longest_streak_z_60m"] = _zs(t["longest_streak"], 60)

    spread = b["spread_bps"]
    f["spread_bps_z_15m"], f["spread_bps_z_60m"] = _zs(spread, 15), _zs(spread, 60)
    f["spread_widening_ratio"] = _ratio(spread, _rsum(spread, 30) / 30)

    f["flow_imb_5m"] = _rimb(t["buy_vol"], t["sell_vol"], 5)
    f["flow_imb_15m"] = _rimb(t["buy_vol"], t["sell_vol"], 15)
    f["vol_z_15m"] = _zs(t["buy_vol"] + t["sell_vol"], 15)
    info = {"minutes": len(f), "minutes_with_mid": int((f["mid"] > 0).sum()), "trade_files": len(trade_files),
            "trades": int(t["n_trades"].sum()), "large_trade_usd": large_thr}
    return f.reset_index(drop=True), info


# --- evaluation ---------------------------------------------------------------

def side_of(features: pd.DataFrame, feature: str, thr: float, direction: str) -> np.ndarray:
    v = features[feature].to_numpy()
    side = np.where(v > thr, 1, np.where(v < -thr, -1, 0)).astype(np.int8)
    return side if direction == "follow" else -side


def agree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where((a != 0) & (a == b), a, 0).astype(np.int8)


def _segment(pnl: np.ndarray) -> Optional[dict]:
    if not len(pnl):
        return None
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return {"n": len(pnl), "win_rate": 100 * float((pnl > 0).mean()), "mean": float(pnl.mean()),
            "sharpe": float(pnl.mean() / sd) if sd > 0 else 0.0, "sum": float(pnl.sum())}


def evaluate(side: np.ndarray, mid: np.ndarray, horizon: int, cfg: Config,
             train_end: int, validate_end: int) -> Optional[dict]:
    """Greedy entries with a cooldown, fixed-horizon exits; stats overall and per segment."""
    n = len(mid)
    shift = 1 if cfg.next_bar_entry else 0
    idx = np.arange(n)
    entry, exit_ = idx + shift, idx + shift + horizon
    ok = (side != 0) & (idx >= WARMUP_MIN) & (exit_ < n)
    cand = idx[ok]
    cand = cand[(mid[entry[cand]] > 0) & (mid[exit_[cand]] > 0)]
    chosen, j = [], 0
    while j < len(cand):
        chosen.append(cand[j])
        j = np.searchsorted(cand, cand[j] + cfg.cooldown_min, side="left")
    if len(chosen) < cfg.min_trades:
        return None
    i = np.asarray(chosen)
    pnl = side[i] * (mid[exit_[i]] / mid[entry[i]] - 1) * 100 - cfg.fee_bps / 100
    out = _segment(pnl)
    out["train"] = _segment(pnl[i < train_end])
    out["validate"] = _segment(pnl[(i >= train_end) & (i < validate_end)])
    out["holdout"] = _segment(pnl[i >= validate_end])
    return out


def validated(stats: dict) -> bool:
    """Training and validation each: n >= 10, win >= 53%, mean > 0, Sharpe >= 0.10."""
    for seg in (stats.get("train"), stats.get("validate")):
        if not seg or seg["n"] < 10 or seg["win_rate"] < 53 or seg["mean"] <= 0 or seg["sharpe"] < 0.10:
            return False
    return True


@dataclass
class Result:
    config: Config
    info: dict
    train_end: int
    validate_end: int
    rows: list[dict]     # label, group, horizon_min, stats

    def ranked(self) -> list[dict]:
        rows = [r for r in self.rows if r["stats"]["n"] >= self.config.min_n]
        return sorted(rows, key=lambda r: -r["stats"]["sharpe"])

    def validated(self) -> list[dict]:
        rows = [r for r in self.ranked() if validated(r["stats"])]
        return sorted(rows, key=lambda r: -r["stats"]["validate"]["sharpe"])


def run(cfg: Config) -> Result:
    features, info = build_features(cfg)
    mid = features["mid"].to_numpy(np.float64)
    if info["minutes_with_mid"] < 100:
        raise ValueError(f"only {info['minutes_with_mid']} minutes with a mid price")
    n = len(features)
    train_end = int(n * cfg.train_frac)
    validate_end = int(n * (cfg.train_frac + cfg.validate_frac))
    horizons = set(cfg.horizons_min)
    rows = []
    for group, feature, thresholds, direction in SINGLE_SPECS:
        for thr in thresholds:
            side = side_of(features, feature, thr, direction)
            for h in cfg.horizons_min:
                stats = evaluate(side, mid, h, cfg, train_end, validate_end)
                if stats:
                    rows.append({"label": f"[{group}] {feature} {direction} thr={thr:g}", "group": group,
                                 "horizon_min": h, "stats": stats})
    for base in BASES:
        b_side = side_of(features, *base)
        for conf in CONFIRMERS:
            side = agree(b_side, side_of(features, *conf))
            for h in (SHORT_BASE_HORIZONS if base[0] in SHORT_BASES else BASE_HORIZONS):
                if h not in horizons:
                    continue
                stats = evaluate(side, mid, h, cfg, train_end, validate_end)
                if stats:
                    label = f"[2] ({base[0]} {base[2]} @{base[1]:g}) AND ({conf[0]} {conf[2]} @{conf[1]:g})"
                    rows.append({"label": label, "group": "2", "horizon_min": h, "stats": stats})
    return Result(cfg, info, train_end, validate_end, rows)


GROUP_NAMES = {"A": "A: imbalance / flow", "H1": "H1: imbalance velocity", "H2": "H2: trade-size regime",
               "H3": "H3: cross-venue divergence", "H4": "H4: aggression", "H5": "H5: spread widening",
               "H6": "H6: short-window imbalance", "2": "two-feature rules"}


def _seg(seg: Optional[dict]) -> str:
    return f"{'(empty)':>15}" if seg is None else f"n={seg['n']:>3} w={seg['win_rate']:>4.0f}%"


def _seg_full(seg: Optional[dict]) -> str:
    return ("(no trades)" if seg is None else
            f"n={seg['n']:>3} win={seg['win_rate']:>4.1f}% mean={seg['mean']:>+6.3f}% sharpe={seg['sharpe']:>+5.3f}")


def _line(rank: int, r: dict, mark: bool) -> str:
    s = r["stats"]
    tag = " *" if mark and validated(s) else ""
    return (f"{rank:<5}{r['label'][:70]:<70}{r['horizon_min']:>5}m{s['n']:>5}{s['win_rate']:>6.1f}%"
            f"{s['mean']:>+8.3f}{s['sharpe']:>+8.3f}  {_seg(s['train'])}  {_seg(s['validate'])}  "
            f"{_seg(s['holdout'])}{tag}")


def render(result: Result) -> str:
    cfg, info = result.config, result.info
    n = info["minutes"]
    header = (f"{'rank':<5}{'rule':<70}{'hzn':>6}{'n':>5}{'win%':>7}{'mean%':>8}{'sharpe':>8}  "
              f"{'TRAIN':>15}  {'VALIDATE':>15}  {'HOLDOUT':>15}")
    out = [f"Feature explorer - {cfg.symbol}",
           f"  grid: {n:,} minutes ({info['minutes_with_mid']:,} with a mid); {info['trades']:,} trades in "
           f"{info['trade_files']} files; large trade >= ${info['large_trade_usd']:,.0f}",
           f"  split: train {result.train_end / 1440:.1f}d / validate {(result.validate_end - result.train_end) / 1440:.1f}d"
           f" / holdout {(n - result.validate_end) / 1440:.1f}d   cooldown={cfg.cooldown_min}m   "
           f"cost={cfg.fee_bps:.2f}bps round trip" + ("   entry=next bar" if cfg.next_bar_entry else ""),
           f"  rules with >= {cfg.min_trades} trades: {len(result.rows):,}; with n >= {cfg.min_n}: {len(result.ranked()):,}",
           "", rule(160), f"TOP {cfg.top} BY OVERALL SHARPE (* = validated)", rule(160), header]
    out += [_line(i, r, True) for i, r in enumerate(result.ranked()[:cfg.top], 1)]

    good = result.validated()
    out += ["", rule(160), "VALIDATED RULES (by validation Sharpe; holdout not used for selection)", rule(160)]
    if good:
        out += [header] + [_line(i, r, False) for i, r in enumerate(good[:20], 1)]
    else:
        out.append("  none: no rule had n >= 10, win >= 53%, mean > 0 and Sharpe >= 0.10 in both training "
                   "and validation")

    out += ["", rule(160), "HOLDOUT for the top 5 validated rules (the out-of-sample estimate)", rule(160)]
    for i, r in enumerate(good[:5], 1):
        s, hold = r["stats"], r["stats"]["holdout"]
        verdict = ("inconclusive (too few trades)" if not hold or hold["n"] < 5 else
                   "holds out of sample" if hold["mean"] > 0 and hold["win_rate"] >= 50 else "degrades out of sample")
        out += [f"  #{i} {r['label']}  h={r['horizon_min']}m", f"     train:    {_seg_full(s['train'])}",
                f"     validate: {_seg_full(s['validate'])}", f"     holdout:  {_seg_full(hold)}  -> {verdict}"]
    if not good:
        out.append("  no validated rules")

    out += ["", "  validated rules by hypothesis:"]
    counts = pd.Series([r["group"] for r in good]).value_counts()
    out += [f"    {GROUP_NAMES.get(g, g):<30}{c:>4}" for g, c in counts.items()] or ["    none"]
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in result.rows:
        s = r["stats"]
        row = {"label": r["label"], "group": r["group"], "horizon_min": r["horizon_min"],
               **{k: s[k] for k in ("n", "win_rate", "mean", "sharpe", "sum")}, "validated": validated(s)}
        for seg in ("train", "validate", "holdout"):
            for k in ("n", "win_rate", "mean", "sharpe"):
                row[f"{seg}_{k}"] = s[seg][k] if s[seg] else None
        rows.append(row)
    path = outdir / f"feature_explore_{result.config.symbol.lower()}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro feature-explore", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--horizons", type=csv_list(int), default=d.horizons_min, help="minutes")
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min,
                        help="minutes between entries; use ~15 for short horizons")
    parser.add_argument("--train-frac", type=float, default=d.train_frac)
    parser.add_argument("--validate-frac", type=float, default=d.validate_frac)
    parser.add_argument("--min-n", type=int, default=d.min_n, help="minimum trades to be ranked")
    parser.add_argument("--top", type=int, default=d.top)
    parser.add_argument("--next-bar-entry", action="store_true",
                        help="enter at the minute after the trigger instead of the trigger minute")
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--outdir", type=Path, default=Path("results/feature_explore"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, horizons_min=args.horizons,
                 cooldown_min=args.cooldown_min, train_frac=args.train_frac, validate_frac=args.validate_frac,
                 min_n=args.min_n, top=args.top, next_bar_entry=args.next_bar_entry, fee_bps=args.fee_bps,
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    print(f"\nwrote {write_outputs(result, args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
