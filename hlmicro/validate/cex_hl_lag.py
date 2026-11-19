"""Does Hyperliquid follow sharp moves on a leading CEX with a tradeable delay?

A trigger fires when the leader venue's mid (default Binance, from 5-second
polling) moves at least ``threshold_bps`` within ``window_s`` seconds, with a
per-direction cooldown. At each trigger a position is notionally taken on
Hyperliquid in the move's direction and its return measured at each horizon:

    lag_capture_bps = direction x Hyperliquid return over the horizon

Gate, per horizon: median capture >= ``gate_bps``, n >= ``min_n``, and the
median positive in both chronological halves.

Usage::

    python -m hlmicro cex-hl-lag --symbols SOL,ETH --threshold-bps 10,15,20
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, FeeModel, default_capture_dir
from ..parse import discover, venue_mid_series
from ..schemas import BOOK_REST_PREFIX
from .common import StepLookup, rule

log = logging.getLogger(__name__)


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbols: list[str] = field(default_factory=lambda: ["SOL", "HYPE", "ETH", "BTC"])
    leader: str = "binance"
    move_windows_s: list[int] = field(default_factory=lambda: [30, 60])
    thresholds_bps: list[float] = field(default_factory=lambda: [10.0, 15.0, 20.0])
    horizons_s: list[int] = field(default_factory=lambda: [10, 30, 60, 120, 300])
    cooldown_s: int = 300
    max_quote_age_ms: int = 30_000
    gate_bps: float = 15.0
    min_n: int = 30
    min_n_halves: int = 20
    fees: FeeModel = DEFAULT_FEES
    start: str | None = None
    end: str | None = None

    @property
    def cost_taker_bps(self) -> float:
        return self.fees.round_trip_with_spread_bps(self.fees.taker_bps, self.fees.taker_bps)

    @property
    def cost_mixed_bps(self) -> float:
        return self.fees.round_trip_with_spread_bps(self.fees.maker_bps, self.fees.taker_bps)

    @property
    def cost_maker_bps(self) -> float:
        return self.fees.round_trip_with_spread_bps(self.fees.maker_bps, self.fees.maker_bps)


def detect_moves(leader: pd.DataFrame, window_s: int, threshold_bps: float,
                 cooldown_s: int) -> pd.DataFrame:
    """Triggers where the leader mid moved >= ``threshold_bps`` since the first
    sample inside the trailing window. At most one trigger per direction per
    ``cooldown_s``."""
    cols = ["ts_ms", "direction", "move_bps", "px_leader"]
    if leader.empty:
        return pd.DataFrame(columns=cols)
    ts = leader["ts_ms"].to_numpy(np.int64)
    mid = leader["mid"].to_numpy(np.float64)
    start = np.searchsorted(ts, ts - window_s * 1000, side="left")
    idx = np.arange(len(ts))
    ok = (start < idx) & (mid[np.minimum(start, len(ts) - 1)] > 0)
    move = np.full(len(ts), np.nan)
    move[ok] = (mid[ok] - mid[start[ok]]) / mid[start[ok]] * 1e4
    candidates = np.flatnonzero(np.abs(np.nan_to_num(move)) >= threshold_bps)

    last = {1: -10**15, -1: -10**15}
    cooldown_ms = cooldown_s * 1000
    rows = []
    for i in candidates:
        direction = 1 if move[i] > 0 else -1
        if ts[i] - last[direction] < cooldown_ms:
            continue
        last[direction] = ts[i]
        rows.append((int(ts[i]), direction, float(move[i]), float(mid[i])))
    return pd.DataFrame(rows, columns=cols)


def measure_response(triggers: pd.DataFrame, follower: StepLookup, horizons_s: list[int]) -> pd.DataFrame:
    """One row per (trigger, horizon) with the follower's signed return in bps."""
    cols = ["ts_ms", "direction", "move_bps", "horizon_s", "lag_capture_bps"]
    if triggers.empty:
        return pd.DataFrame(columns=cols)
    t = triggers["ts_ms"].to_numpy(np.int64)
    px0 = follower.at(t)
    keep = ~np.isnan(px0) & (px0 > 0)
    t, px0 = t[keep], px0[keep]
    direction = triggers["direction"].to_numpy()[keep]
    move = triggers["move_bps"].to_numpy()[keep]
    h = np.asarray(horizons_s, dtype=np.int64)
    pxh = follower.at((t[:, None] + h[None, :] * 1000).ravel()).reshape(len(t), len(h))
    capture = direction[:, None] * (pxh - px0[:, None]) / px0[:, None] * 1e4
    frame = pd.DataFrame({
        "ts_ms": np.repeat(t, len(h)), "direction": np.repeat(direction, len(h)),
        "move_bps": np.repeat(move, len(h)), "horizon_s": np.tile(h, len(t)),
        "lag_capture_bps": capture.ravel(),
    })
    return frame[~np.isnan(pxh.ravel())].reset_index(drop=True)


def summarize(response: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in response.groupby("horizon_s"):
        x = group["lag_capture_bps"].dropna().to_numpy()
        if not len(x):
            continue
        std = x.std(ddof=1) if len(x) > 1 else np.nan
        rows.append({"horizon_s": int(horizon), "n": len(x), "mean": x.mean(), "median": np.median(x),
                     "p25": np.percentile(x, 25), "p75": np.percentile(x, 75), "std": std,
                     "sharpe": x.mean() / std if std and std > 0 else np.nan,
                     "pct_pos": (x > 0).mean() * 100})
    return pd.DataFrame(rows)


def halves(response: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for horizon, group in response.groupby("horizon_s"):
        x = group.sort_values("ts_ms", kind="stable")["lag_capture_bps"].dropna().to_numpy()
        if len(x) < min_n:
            continue
        cut = len(x) // 2
        m1, m2 = float(np.median(x[:cut])), float(np.median(x[cut:]))
        rows.append({"horizon_s": int(horizon), "n_first": cut, "n_second": len(x) - cut,
                     "median_first": m1, "median_second": m2, "positive_both": m1 > 0 and m2 > 0})
    return pd.DataFrame(rows, columns=["horizon_s", "n_first", "n_second", "median_first",
                                       "median_second", "positive_both"])


@dataclass
class Sweep:
    symbol: str
    window_s: int
    threshold_bps: float
    n_triggers: int
    n_up: int
    n_down: int
    response: pd.DataFrame
    summary: pd.DataFrame
    halves: pd.DataFrame

    def gate_rows(self, cfg: Config) -> list[dict]:
        out = []
        for _, r in self.summary.iterrows():
            h = self.halves[self.halves["horizon_s"] == r["horizon_s"]]
            stable = bool(h["positive_both"].iloc[0]) if not h.empty else False
            if r["median"] >= cfg.gate_bps and r["n"] >= cfg.min_n and stable:
                out.append({"horizon_s": int(r["horizon_s"]), "median": r["median"], "n": int(r["n"])})
        return out


@dataclass
class Result:
    config: Config
    coverage: dict
    sweeps: list[Sweep]


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    coverage, sweeps = {}, []
    for symbol in cfg.symbols:
        leader = venue_mid_series(files, symbol, cfg.leader)
        follower = venue_mid_series(files, symbol, HL_VENUE)
        if leader.empty or follower.empty:
            coverage[symbol] = None
            continue
        coverage[symbol] = {"leader_rows": len(leader), "follower_rows": len(follower),
                            "days": (follower["ts_ms"].max() - follower["ts_ms"].min()) / 86_400_000}
        lookup = StepLookup(follower["ts_ms"].to_numpy(), follower["mid"].to_numpy(), cfg.max_quote_age_ms)
        for window in cfg.move_windows_s:
            for threshold in cfg.thresholds_bps:
                triggers = detect_moves(leader, window, threshold, cfg.cooldown_s)
                response = measure_response(triggers, lookup, cfg.horizons_s)
                sweeps.append(Sweep(symbol, window, threshold, len(triggers),
                                    int((triggers["direction"] == 1).sum()),
                                    int((triggers["direction"] == -1).sum()),
                                    response, summarize(response), halves(response, cfg.min_n_halves)))
    return Result(cfg, coverage, sweeps)


def render(result: Result, direction_breakdown: bool = False) -> str:
    cfg = result.config
    out = [rule(), f"{cfg.leader}-to-Hyperliquid repricing lag", rule(),
           f"  symbols: {cfg.symbols}   windows: {cfg.move_windows_s}s   thresholds: {cfg.thresholds_bps} bps",
           f"  horizons: {cfg.horizons_s}s   cooldown: {cfg.cooldown_s}s",
           f"  costs (round trip incl. spread): taker {cfg.cost_taker_bps:.2f}  "
           f"maker-in/taker-out {cfg.cost_mixed_bps:.2f}  maker {cfg.cost_maker_bps:.2f} bps",
           f"  gate: median >= {cfg.gate_bps} bps, n >= {cfg.min_n}, median positive in both halves",
           "  (* = passes gate; halves column: Y = positive in both)"]
    for symbol, cov in result.coverage.items():
        out += ["", rule(), f"  {symbol}", rule()]
        if cov is None:
            out.append(f"  no {cfg.leader} or Hyperliquid data")
            continue
        out.append(f"  {cfg.leader}: {cov['leader_rows']:,} rows   Hyperliquid: {cov['follower_rows']:,} rows   "
                   f"coverage {cov['days']:.1f} days")
        for sweep in (s for s in result.sweeps if s.symbol == symbol):
            if sweep.n_triggers == 0:
                out.append(f"\n  window={sweep.window_s}s threshold={sweep.threshold_bps:.0f}bps: no triggers")
                continue
            if sweep.summary.empty:
                continue
            passing = {g["horizon_s"] for g in sweep.gate_rows(cfg)}
            out += [f"\n  window={sweep.window_s}s  threshold={sweep.threshold_bps:.0f}bps  "
                    f"triggers={sweep.n_triggers} (up {sweep.n_up}, down {sweep.n_down})",
                    f"    {'hzn':>6}{'n':>6}{'mean':>8}{'median':>8}{'p25':>8}{'p75':>8}{'pos%':>7}"
                    f"{'sharpe':>8}{'net@taker':>11}{'net@mixed':>11}{'halves':>8}"]
            for _, r in sweep.summary.iterrows():
                h = sweep.halves[sweep.halves["horizon_s"] == r["horizon_s"]]
                stable = "Y" if (not h.empty and bool(h["positive_both"].iloc[0])) else "N"
                mark = "*" if int(r["horizon_s"]) in passing else " "
                out.append(f"   {mark}{int(r['horizon_s']):>5}s{int(r['n']):>6}{r['mean']:>+8.2f}{r['median']:>+8.2f}"
                           f"{r['p25']:>+8.2f}{r['p75']:>+8.2f}{r['pct_pos']:>6.1f}%{r['sharpe']:>+8.3f}"
                           f"{r['median'] - cfg.cost_taker_bps:>+11.2f}{r['median'] - cfg.cost_mixed_bps:>+11.2f}"
                           f"{stable:>8}")
            if not sweep.halves.empty:
                out.append("    halves: " + "  ".join(
                    f"{int(h['horizon_s'])}s {h['median_first']:+.2f}/{h['median_second']:+.2f}"
                    for _, h in sweep.halves.iterrows()))
            if direction_breakdown and passing:
                for d, label in ((1, "up moves (long HL)"), (-1, "down moves (short HL)")):
                    part = sweep.response[sweep.response["direction"] == d]
                    if part.empty:
                        continue
                    out.append(f"    {label}: {part['ts_ms'].nunique()} triggers")
                    for _, r in summarize(part).iterrows():
                        out.append(f"      {int(r['horizon_s']):>5}s n={int(r['n']):>4} mean={r['mean']:+.2f} "
                                   f"median={r['median']:+.2f} pos={r['pct_pos']:.1f}%")
    out += ["", rule(), "  Gate summary", rule()]
    passed = [(s, g) for s in result.sweeps for g in s.gate_rows(cfg)]
    if not passed:
        out.append("  FAIL  no configuration passes the gate")
    for s, g in passed:
        out.append(f"  PASS  {s.symbol} window={s.window_s}s threshold={s.threshold_bps:.0f}bps "
                   f"horizon={g['horizon_s']}s n={g['n']} median={g['median']:+.2f} "
                   f"net@taker={g['median'] - cfg.cost_taker_bps:+.2f} net@mixed={g['median'] - cfg.cost_mixed_bps:+.2f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro cex-hl-lag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbols", type=csv_list(str.upper), default=d.symbols)
    parser.add_argument("--leader", default=d.leader, help="venue whose moves trigger events")
    parser.add_argument("--move-window-s", type=csv_list(int), default=d.move_windows_s)
    parser.add_argument("--threshold-bps", type=csv_list(float), default=d.thresholds_bps)
    parser.add_argument("--horizons-s", type=csv_list(int), default=d.horizons_s)
    parser.add_argument("--cooldown-s", type=int, default=d.cooldown_s)
    parser.add_argument("--gate-bps", type=float, default=d.gate_bps)
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--maker-bps", type=float, default=d.fees.maker_bps)
    parser.add_argument("--half-spread-bps", type=float, default=d.fees.half_spread_bps)
    parser.add_argument("--direction-breakdown", action="store_true",
                        help="split passing configurations into up and down moves")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbols=args.symbols, leader=args.leader.lower(),
                 move_windows_s=args.move_window_s, thresholds_bps=args.threshold_bps,
                 horizons_s=args.horizons_s, cooldown_s=args.cooldown_s, gate_bps=args.gate_bps,
                 fees=FeeModel(taker_bps=args.taker_bps, maker_bps=args.maker_bps,
                               half_spread_bps=args.half_spread_bps),
                 start=args.start, end=args.end)
    print(render(run(cfg), direction_breakdown=args.direction_breakdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
