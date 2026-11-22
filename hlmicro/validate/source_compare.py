"""Does websocket-sampled order-book imbalance trade better than 5-second polling?

The same dual-window z-score signal and exit rules run twice over the same
minutes: once driven by imbalance from ``obi_5s`` (REST polling) and once by
imbalance from ``obi_ws`` (websocket updates). Returns for both use the common
Hyperliquid mid from the 5-second data, and only minutes present in both
sources are used, so any difference is attributable to the sampling alone.

Entries are taken one minute after the signal (no look-ahead); exits on stop,
take-profit or timeout, followed by a cooldown. A take-profit is a resting
limit and fills at its level; a stop fills at the observed minute price, which
can be beyond the stop level when the price moved through it.

Usage::

    python -m hlmicro source-compare --symbol BTC --threshold 2.75 --tp 1.5 --stop=-1.0
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, setup_logging
from ..config import DEFAULT_FEES, default_capture_dir
from ..parse import cross_venue_minutes, discover, discover_glob
from ..schemas import BOOK_REST_PREFIX, BOOK_WS_PREFIX
from .common import describe, rolling_zscore, rule

log = logging.getLogger(__name__)


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    ws_glob: str | None = None
    symbol: str = "BTC"
    threshold: float = 2.5
    tp_pct: float = 0.0
    stop_pct: float = -1.0
    hold_min: int = 240
    cooldown_min: int = 60
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    n_folds: int = 4
    min_overlap_minutes: int = 500
    start: str | None = None
    end: str | None = None
    # Material-improvement criteria for the websocket source.
    material_ratio: float = 1.5
    material_t: float = 1.5
    material_folds: int = 3


def backtest(minutes: np.ndarray, obi: np.ndarray, prices: np.ndarray, cfg: Config) -> list[dict]:
    """Position backtest on aligned per-minute arrays (``minutes`` in ms)."""
    n = len(minutes)
    z60 = rolling_zscore(obi, 60, 20)
    z30 = rolling_zscore(obi, 30, 15)
    stop = cfg.stop_pct / 100.0
    tp = cfg.tp_pct / 100.0
    fee = cfg.fee_bps / 1e4
    cooldown_ms = cfg.cooldown_min * 60_000
    trades: list[dict] = []
    next_allowed = 0
    for i in range(n - 1):
        if minutes[i] < next_allowed or np.isnan(z60[i]) or np.isnan(z30[i]):
            continue
        if z60[i] > cfg.threshold and z30[i] > cfg.threshold:
            side = 1.0
        elif z60[i] < -cfg.threshold and z30[i] < -cfg.threshold:
            side = -1.0
        else:
            continue
        entry = i + 1
        entry_px = prices[entry]
        if np.isnan(entry_px) or entry_px <= 0:
            continue
        exit_ret = None
        exit_min = minutes[entry]
        last = min(cfg.hold_min, n - 1 - entry)
        for h in range(1, last + 1):
            px = prices[entry + h]
            if np.isnan(px):
                continue
            ret = (px / entry_px - 1.0) * side
            exit_min = minutes[entry + h]
            if tp and ret >= tp:
                exit_ret = tp
                break
            if ret <= stop:
                exit_ret = ret        # the price actually seen, not the stop level
                break
        if exit_ret is None:
            px = prices[entry + last]
            exit_ret = (px / entry_px - 1.0) * side if not np.isnan(px) else 0.0
        trades.append({"entry_min": int(minutes[entry]), "side": int(side), "ret_net": exit_ret - fee})
        next_allowed = exit_min + cooldown_ms
    return trades


def summarize(trades: list[dict], n_folds: int, start_ms: int, end_ms: int) -> dict | None:
    if not trades:
        return None
    frame = pd.DataFrame(trades)
    s = describe(frame["ret_net"].to_numpy(), ddof=1)
    if s["std"] == 0 or s["n"] < 2:
        s["sharpe"] = s["t"] = np.nan
    edges = np.linspace(start_ms, end_ms, n_folds + 1)
    fold_means = []
    for k in range(n_folds):
        upper = (frame["entry_min"] <= edges[k + 1]) if k == n_folds - 1 else (frame["entry_min"] < edges[k + 1])
        part = frame[(frame["entry_min"] >= edges[k]) & upper]
        fold_means.append(part["ret_net"].mean() if len(part) else np.nan)
    s["fold_means"] = fold_means
    s["folds_positive"] = sum(1 for m in fold_means if not np.isnan(m) and m > 0)
    return s


@dataclass
class Result:
    config: Config
    shared_minutes: int
    start_ms: int
    end_ms: int
    rest: dict | None
    ws: dict | None


def run(cfg: Config) -> Result:
    rest_files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    ws_files = (discover_glob(cfg.ws_glob, cfg.start, cfg.end) if cfg.ws_glob
                else discover(cfg.capture_dir, BOOK_WS_PREFIX, cfg.start, cfg.end))
    if not rest_files or not ws_files:
        raise FileNotFoundError("need both obi_5s and obi_ws files")
    log.info("loading %d REST and %d websocket files", len(rest_files), len(ws_files))
    rest = cross_venue_minutes(rest_files, cfg.symbol)
    ws = cross_venue_minutes(ws_files, cfg.symbol)
    merged = (rest.merge(ws[["ts_min", "avg_obi"]], on="ts_min", how="inner", suffixes=("_rest", "_ws"))
              .sort_values("ts_min").reset_index(drop=True))
    if len(merged) < cfg.min_overlap_minutes:
        raise ValueError(f"only {len(merged)} overlapping minutes (need {cfg.min_overlap_minutes})")
    minutes = merged["ts_min"].to_numpy()
    price = merged["mid_hl"].to_numpy()
    start, end = int(minutes[0]), int(minutes[-1])
    return Result(
        config=cfg, shared_minutes=len(merged), start_ms=start, end_ms=end,
        rest=summarize(backtest(minutes, merged["avg_obi_rest"].to_numpy(), price, cfg), cfg.n_folds, start, end),
        ws=summarize(backtest(minutes, merged["avg_obi_ws"].to_numpy(), price, cfg), cfg.n_folds, start, end),
    )


def classify(rest: dict, ws: dict, cfg: Config) -> str:
    material = (ws["mean"] > rest["mean"] * cfg.material_ratio and ws["t"] > cfg.material_t
                and ws["folds_positive"] >= cfg.material_folds)
    modest = (ws["mean"] > rest["mean"] + 0.0002 and ws["t"] >= rest["t"]
              and ws["folds_positive"] >= rest["folds_positive"])
    if material:
        return (f"websocket better (material): mean > {cfg.material_ratio}x REST, "
                f"t > {cfg.material_t}, folds+ >= {cfg.material_folds}")
    if modest:
        return "websocket better (modest): mean +2bps or more, t and folds+ not lower"
    return "websocket not better than REST"


def _day(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def render(result: Result) -> str:
    cfg = result.config
    tp = f"+{cfg.tp_pct}%" if cfg.tp_pct else "none"
    out = [
        rule(84), f"Imbalance sampling comparison - {cfg.symbol}", rule(84),
        f"  window: {_day(result.start_ms)} .. {_day(result.end_ms)} "
        f"({(result.end_ms - result.start_ms) / 86_400_000:.1f}d, {result.shared_minutes:,} shared minutes)",
        f"  config: threshold=+/-{cfg.threshold}  hold={cfg.hold_min}m  stop={cfg.stop_pct}%  tp={tp}  "
        f"cooldown={cfg.cooldown_min}m  cost={cfg.fee_bps:.2f}bps round trip",
        "",
    ]
    for name, s in (("REST 5s sampling", result.rest), ("websocket sampling", result.ws)):
        if s is None:
            out += [f"  {name}: no trades", ""]
            continue
        folds = "  ".join(f"f{k}={m * 100:+.3f}%" if not np.isnan(m) else f"f{k}=na"
                          for k, m in enumerate(s["fold_means"]))
        out += [f"  {name}",
                f"    n={s['n']:<4} mean={s['mean'] * 100:+.4f}%  t={s['t']:+.2f}  "
                f"win={s['win_rate'] * 100:.1f}%  sharpe={s['sharpe']:+.3f}  "
                f"folds+={s['folds_positive']}/{cfg.n_folds}",
                f"    {folds}", ""]
    out += [rule(84), "  Result", rule(84)]
    if result.rest and result.ws:
        r, w = result.rest, result.ws
        out += [f"  mean per trade:  REST {r['mean'] * 100:+.4f}%   websocket {w['mean'] * 100:+.4f}%   "
                f"difference {(w['mean'] - r['mean']) * 100:+.4f}pp",
                f"  t-statistic:     REST {r['t']:+.2f}   websocket {w['t']:+.2f}",
                f"  trades:          REST {r['n']}   websocket {w['n']}",
                f"  classification:  {classify(r, w, cfg)}"]
    else:
        out.append("  comparison needs trades from both sources")
    out.append("  (in-sample on the overlap window)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro source-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--ws-glob", default=None, help="websocket files (default: obi_ws in capture dir)")
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=d.threshold)
    parser.add_argument("--tp", type=float, default=d.tp_pct, help="take-profit %%, 0 = none")
    parser.add_argument("--stop", type=float, default=d.stop_pct, help="stop %% (negative)")
    parser.add_argument("--hold", type=int, default=d.hold_min, help="minutes")
    parser.add_argument("--cooldown", type=int, default=d.cooldown_min, help="minutes")
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, ws_glob=args.ws_glob, symbol=args.symbol,
                 threshold=args.threshold, tp_pct=args.tp, stop_pct=args.stop, hold_min=args.hold,
                 cooldown_min=args.cooldown, fee_bps=args.fee_bps, n_folds=args.n_folds,
                 start=args.start, end=args.end)
    print(render(run(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
