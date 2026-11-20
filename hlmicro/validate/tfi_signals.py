"""Taker-flow imbalance (TFI) as a standalone directional signal.

Hypothesis: persistent aggressive flow (the executed side of the market, as
opposed to resting book shape) leads short-horizon price, especially on less
liquid coins where book imbalance is easier to spoof than prints.

Per minute and venue (Hyperliquid, OKX, Bybit), TFI = (buy - sell) / (buy +
sell) by notional; cross-venue TFI is the mean over venues that traded. Two
signal families, each following the flow direction:

``zscore``
    The cross-venue TFI z-score over ``z_window_min`` minutes crosses
    +/-threshold. One trigger per direction per ``zscore_cooldown_min``.
``streak``
    ``N`` consecutive minutes of same-sign cross-venue TFI, each with |TFI| at
    least ``floor``. The streak restarts after each trigger.

Each trigger is priced at the Hyperliquid mid at the close of the trigger
minute, when its flow is known, and its signed return measured at each
horizon. Gate per horizon: median >= ``gate_bps``, n >= ``min_n``, and the
median positive in both chronological halves.

Usage::

    python -m hlmicro tfi-signals --symbols SOL,HYPE
    python -m hlmicro tfi-signals --symbols SOL --signals streak --streak-n 10,15,20
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
from ..parse import cross_venue_flow, discover, venue_mid_series
from ..schemas import BOOK_REST_PREFIX
from .common import StepLookup, horizon_halves, horizon_summary, rolling_zscore, rule

log = logging.getLogger(__name__)

MINUTE_MS = 60_000
SIGNALS = ("zscore", "streak")


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbols: list[str] = field(default_factory=lambda: ["SOL", "HYPE", "ETH", "BTC"])
    signals: list[str] = field(default_factory=lambda: list(SIGNALS))
    flow_venues: list[str] = field(default_factory=lambda: ["hl", "okx", "bybit"])
    z_window_min: int = 60
    z_thresholds: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    z_horizons_min: list[int] = field(default_factory=lambda: [15, 30, 60, 240])
    zscore_cooldown_min: int = 60
    streak_n: list[int] = field(default_factory=lambda: [10, 15, 20, 30])
    streak_floors: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    streak_horizons_min: list[int] = field(default_factory=lambda: [15, 30, 60, 240])
    streak_cooldown_min: int = 60
    min_streak_triggers: int = 5
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


def zscore_triggers(flow: pd.DataFrame, threshold: float, cooldown_min: int) -> pd.DataFrame:
    """Minutes where |TFI z| >= threshold, at most one per direction per cooldown."""
    z = flow["xv_tfi_z"].to_numpy()
    minutes = flow["bucket"].to_numpy(np.int64)
    last = {1: -10**15, -1: -10**15}
    rows = []
    for i in np.flatnonzero(np.abs(np.nan_to_num(z)) >= threshold):
        d = 1 if z[i] > 0 else -1
        if minutes[i] - last[d] < cooldown_min:
            continue
        last[d] = minutes[i]
        rows.append((int(minutes[i]), d))
    return pd.DataFrame(rows, columns=["bucket", "direction"])


def streak_triggers(flow: pd.DataFrame, n: int, floor: float, cooldown_min: int) -> pd.DataFrame:
    """Minutes completing ``n`` consecutive same-sign minutes with |TFI| >= floor
    (consecutive rows of minutes that traded)."""
    tfi = flow["xv_tfi"].to_numpy()
    minutes = flow["bucket"].to_numpy(np.int64)
    last = {1: -10**15, -1: -10**15}
    rows, sign, length = [], 0, 0
    for i, v in enumerate(tfi):
        if np.isnan(v) or abs(v) < floor:
            sign, length = 0, 0
            continue
        s = 1 if v > 0 else -1
        sign, length = (s, length + 1) if s == sign else (s, 1)
        if length < n or minutes[i] - last[sign] < cooldown_min:
            continue
        last[sign] = minutes[i]
        rows.append((int(minutes[i]), sign))
        length = 0
    return pd.DataFrame(rows, columns=["bucket", "direction"])


def measure(triggers: pd.DataFrame, lookup: StepLookup, horizons_min: list[int]) -> pd.DataFrame:
    """Signed Hyperliquid return in bps from the close of each trigger minute."""
    cols = ["ts_ms", "direction", "horizon_min", "pnl_bps"]
    if triggers.empty:
        return pd.DataFrame(columns=cols)
    t = (triggers["bucket"].to_numpy(np.int64) + 1) * MINUTE_MS
    px0 = lookup.at(t)
    keep = ~np.isnan(px0) & (px0 > 0)
    t, px0, d = t[keep], px0[keep], triggers["direction"].to_numpy()[keep]
    h = np.asarray(horizons_min, dtype=np.int64)
    pxh = lookup.at((t[:, None] + h[None, :] * MINUTE_MS).ravel()).reshape(len(t), len(h))
    pnl = d[:, None] * (pxh - px0[:, None]) / px0[:, None] * 1e4
    frame = pd.DataFrame({"ts_ms": np.repeat(t, len(h)), "direction": np.repeat(d, len(h)),
                          "horizon_min": np.tile(h, len(t)), "pnl_bps": pnl.ravel()})
    return frame[~np.isnan(pxh.ravel())].reset_index(drop=True)


@dataclass
class Sweep:
    symbol: str
    signal: str
    params: dict
    n_triggers: int
    response: pd.DataFrame
    summary: pd.DataFrame
    halves: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{self.signal} " + " ".join(f"{k}={v:g}" for k, v in self.params.items())

    def gate_rows(self, cfg: Config) -> list[dict]:
        out = []
        for _, r in self.summary.iterrows():
            h = self.halves[self.halves["horizon_min"] == r["horizon_min"]]
            stable = bool(h["positive_both"].iloc[0]) if not h.empty else False
            if r["median"] >= cfg.gate_bps and r["n"] >= cfg.min_n and stable:
                out.append({"horizon_min": int(r["horizon_min"]), "median": r["median"], "n": int(r["n"]),
                            "pct_pos": r["pct_pos"], "sharpe": r["sharpe"]})
        return out


@dataclass
class Result:
    config: Config
    coverage: dict
    sweeps: list[Sweep]


def _sweep(symbol, signal, params, triggers, lookup, horizons, cfg) -> Sweep:
    response = measure(triggers, lookup, horizons)
    return Sweep(symbol, signal, params, len(triggers), response,
                 horizon_summary(response, "pnl_bps", "horizon_min"),
                 horizon_halves(response, "pnl_bps", "horizon_min", cfg.min_n_halves))


def run(cfg: Config) -> Result:
    unknown = set(cfg.signals) - set(SIGNALS)
    if unknown:
        raise ValueError(f"unknown signals {sorted(unknown)}; choose from {SIGNALS}")
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    coverage, sweeps = {}, []
    for symbol in cfg.symbols:
        flow = cross_venue_flow(cfg.capture_dir, symbol, MINUTE_MS, cfg.flow_venues, cfg.start, cfg.end)
        mids = venue_mid_series(files, symbol, HL_VENUE)
        if flow.empty or mids.empty:
            coverage[symbol] = None
            continue
        flow["xv_tfi_z"] = rolling_zscore(flow["xv_tfi"].to_numpy(), cfg.z_window_min,
                                          min_periods=max(10, cfg.z_window_min // 4))
        coverage[symbol] = {"minutes": int(flow["xv_tfi"].notna().sum()),
                            "days": (flow["bucket"].max() - flow["bucket"].min()) / 1440,
                            "venues_per_minute": float(flow["n_venues"].mean()), "mid_rows": len(mids)}
        lookup = StepLookup(mids["ts_ms"].to_numpy(), mids["mid"].to_numpy(), cfg.max_quote_age_ms)
        if "zscore" in cfg.signals:
            for thr in cfg.z_thresholds:
                trig = zscore_triggers(flow, thr, cfg.zscore_cooldown_min)
                sweeps.append(_sweep(symbol, "zscore", {"threshold": thr}, trig, lookup, cfg.z_horizons_min, cfg))
        if "streak" in cfg.signals:
            for n in cfg.streak_n:
                for floor in cfg.streak_floors:
                    trig = streak_triggers(flow, n, floor, cfg.streak_cooldown_min)
                    if len(trig) < cfg.min_streak_triggers:
                        continue
                    sweeps.append(_sweep(symbol, "streak", {"n": n, "floor": floor}, trig, lookup,
                                         cfg.streak_horizons_min, cfg))
    return Result(cfg, coverage, sweeps)


def render(result: Result) -> str:
    cfg = result.config
    out = [rule(), "Taker-flow imbalance signals", rule(),
           f"  symbols: {cfg.symbols}   signals: {cfg.signals}   flow venues: {cfg.flow_venues}",
           f"  costs (round trip incl. spread): taker {cfg.cost_taker_bps:.2f}  "
           f"maker-in/taker-out {cfg.cost_mixed_bps:.2f} bps",
           f"  gate: median >= {cfg.gate_bps} bps, n >= {cfg.min_n}, median positive in both halves",
           "  (* = passes gate; halves column: Y = positive in both)"]
    for symbol, cov in result.coverage.items():
        out += ["", rule(), f"  {symbol}", rule()]
        if cov is None:
            out.append("  no trade-tape or Hyperliquid mid data")
            continue
        out.append(f"  flow: {cov['minutes']:,} minutes over {cov['days']:.1f} days, "
                   f"{cov['venues_per_minute']:.2f} venues per minute   Hyperliquid mids: {cov['mid_rows']:,}")
        for sweep in (s for s in result.sweeps if s.symbol == symbol):
            if sweep.n_triggers == 0:
                out.append(f"\n  {sweep.label}: no triggers")
                continue
            if sweep.summary.empty:
                continue
            passing = {g["horizon_min"] for g in sweep.gate_rows(cfg)}
            out += [f"\n  {sweep.label}  triggers={sweep.n_triggers}",
                    f"    {'hzn':>6}{'n':>6}{'mean':>8}{'median':>8}{'p25':>8}{'p75':>8}{'pos%':>7}"
                    f"{'sharpe':>8}{'net@taker':>11}{'net@mixed':>11}{'halves':>8}"]
            for _, r in sweep.summary.iterrows():
                h = sweep.halves[sweep.halves["horizon_min"] == r["horizon_min"]]
                stable = "Y" if (not h.empty and bool(h["positive_both"].iloc[0])) else "N"
                mark = "*" if int(r["horizon_min"]) in passing else " "
                sharpe = "na" if np.isnan(r["sharpe"]) else f"{r['sharpe']:+.3f}"
                out.append(f"   {mark}{int(r['horizon_min']):>4}m{int(r['n']):>6}{r['mean']:>+8.2f}{r['median']:>+8.2f}"
                           f"{r['p25']:>+8.2f}{r['p75']:>+8.2f}{r['pct_pos']:>6.1f}%{sharpe:>8}"
                           f"{r['median'] - cfg.cost_taker_bps:>+11.2f}{r['median'] - cfg.cost_mixed_bps:>+11.2f}"
                           f"{stable:>8}")
            if not sweep.halves.empty:
                out.append("    halves: " + "  ".join(
                    f"{int(h['horizon_min'])}m {h['median_first']:+.2f}/{h['median_second']:+.2f}"
                    for _, h in sweep.halves.iterrows()))
    out += ["", rule(), "  Gate summary", rule()]
    passed = [(s, g) for s in result.sweeps for g in s.gate_rows(cfg)]
    if not passed:
        out.append("  FAIL  no configuration passes the gate")
    for s, g in passed:
        out.append(f"  PASS  {s.symbol} {s.label} horizon={g['horizon_min']}m n={g['n']} "
                   f"median={g['median']:+.2f} net@taker={g['median'] - cfg.cost_taker_bps:+.2f} "
                   f"net@mixed={g['median'] - cfg.cost_mixed_bps:+.2f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro tfi-signals", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbols", type=csv_list(str.upper), default=d.symbols)
    parser.add_argument("--signals", type=csv_list(str.lower), default=d.signals,
                        help=f"subset of {','.join(SIGNALS)}")
    parser.add_argument("--flow-venues", type=csv_list(str.lower), default=d.flow_venues)
    parser.add_argument("--z-window-min", type=int, default=d.z_window_min)
    parser.add_argument("--z-thresholds", type=csv_list(float), default=d.z_thresholds)
    parser.add_argument("--z-horizons-min", type=csv_list(int), default=d.z_horizons_min)
    parser.add_argument("--streak-n", type=csv_list(int), default=d.streak_n)
    parser.add_argument("--streak-floors", type=csv_list(float), default=d.streak_floors)
    parser.add_argument("--streak-horizons-min", type=csv_list(int), default=d.streak_horizons_min)
    parser.add_argument("--gate-bps", type=float, default=d.gate_bps)
    parser.add_argument("--min-n", type=int, default=d.min_n)
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--maker-bps", type=float, default=d.fees.maker_bps)
    parser.add_argument("--half-spread-bps", type=float, default=d.fees.half_spread_bps)
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbols=args.symbols, signals=args.signals,
                 flow_venues=args.flow_venues, z_window_min=args.z_window_min, z_thresholds=args.z_thresholds,
                 z_horizons_min=args.z_horizons_min, streak_n=args.streak_n, streak_floors=args.streak_floors,
                 streak_horizons_min=args.streak_horizons_min, gate_bps=args.gate_bps, min_n=args.min_n,
                 fees=FeeModel(taker_bps=args.taker_bps, maker_bps=args.maker_bps,
                               half_spread_bps=args.half_spread_bps),
                 start=args.start, end=args.end)
    print(render(run(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
