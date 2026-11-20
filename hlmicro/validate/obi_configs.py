"""Imbalance trigger configurations compared under identical exits, with and
without a taker-flow (TFI) gate.

Every configuration for a symbol uses the same exit logic, so differences
between them come from the trigger alone:

* trigger: per-minute cross-venue mean imbalance; z-scores over one window
  (``single``) or two windows that must agree in sign (``dual``), each beyond
  the threshold. Z-scores are taken over minutes with data (no gap filling).
* entry: the Hyperliquid mid at the close of the trigger minute, when that
  minute's mean imbalance is known. One position at a time.
* exit: first touch of the stop or take-profit on the 5-second Hyperliquid
  mid (filled at the level), else the last mid within ``max_hold_min``.

Each configuration is reported ungated and with two TFI gates computed from
the trigger minute's trades on Hyperliquid, OKX and Bybit: ``soft`` (TFI
sign agrees with the trade) and ``strict`` (sign agrees and |TFI z| >= 1).

Gate, per variant: median net of taker cost >= ``gate_net_bps``, n >=
``min_n``, and the median positive in both chronological halves.
``bps/day`` = median net per trade x trades / days covered.

Usage::

    python -m hlmicro obi-configs
    python -m hlmicro obi-configs --symbols SOL,HYPE
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
from ..config import DEFAULT_FEES, HL_VENUE, FeeModel, default_capture_dir
from ..parse import cross_venue_flow, cross_venue_minutes, discover, venue_mid_series
from ..schemas import BOOK_REST_PREFIX
from .common import StepLookup, rolling_zscore, rule

log = logging.getLogger(__name__)

MINUTE_MS = 60_000


@dataclass(frozen=True)
class Trigger:
    label: str
    windows_min: tuple[int, ...]     # one window (single) or two (dual, same sign)
    threshold: float
    stop_pct: Optional[float] = -1.0
    tp_pct: Optional[float] = 1.0
    max_hold_min: int = 240
    reference: bool = False          # the configuration currently traded

    def describe(self) -> str:
        kind = "dual" if len(self.windows_min) > 1 else "single"
        stop = "none" if self.stop_pct is None else f"{self.stop_pct:+.1f}%"
        tp = "none" if self.tp_pct is None else f"+{self.tp_pct:.1f}%"
        return (f"{kind} z{'+'.join(map(str, self.windows_min))} beyond +/-{self.threshold}  "
                f"stop={stop}  TP={tp}  hold<={self.max_hold_min}m")


PRESETS: dict[str, list[Trigger]] = {
    "SOL": [Trigger("dual_60+30_t2.5", (60, 30), 2.5, reference=True),
            Trigger("single_60_t2.5", (60,), 2.5),
            Trigger("single_60_t2.0", (60,), 2.0)],
    "BTC": [Trigger("single_60_t2.75", (60,), 2.75, tp_pct=None, reference=True)],
    "HYPE": [Trigger("dual_60+30_t2.5", (60, 30), 2.5),
             Trigger("single_60_t2.0", (60,), 2.0)],
}


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbols: list[str] = field(default_factory=lambda: ["SOL", "BTC"])
    flow_venues: list[str] = field(default_factory=lambda: ["hl", "okx", "bybit"])
    tfi_z_window_min: int = 60
    strict_tfi_z: float = 1.0
    max_quote_age_ms: int = 30_000
    gate_net_bps: float = 0.0
    min_n: int = 20
    min_half_n: int = 5
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


def _zscore(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_zscore(values, window, min_periods=max(10, window // 4))


def direction(trigger: Trigger, z: dict[int, float]) -> int:
    values = [z[w] for w in trigger.windows_min]
    if any(np.isnan(v) or abs(v) < trigger.threshold for v in values):
        return 0
    if len({v > 0 for v in values}) > 1:
        return 0
    return 1 if values[0] > 0 else -1


def simulate_exit(t_entry: int, side: int, px_entry: float, trigger: Trigger,
                  ts: np.ndarray, mid: np.ndarray) -> Optional[tuple[int, float, str]]:
    """First stop or take-profit touch within the hold window, else the last mid."""
    i0 = np.searchsorted(ts, t_entry, side="left")
    i1 = np.searchsorted(ts, t_entry + trigger.max_hold_min * MINUTE_MS, side="right")
    if i1 <= i0:
        return None
    ret_bps = side * (mid[i0:i1] - px_entry) / px_entry * 1e4
    tp_hit = np.flatnonzero(ret_bps >= trigger.tp_pct * 100) if trigger.tp_pct is not None else []
    stop_hit = np.flatnonzero(ret_bps <= trigger.stop_pct * 100) if trigger.stop_pct is not None else []
    tp_i = tp_hit[0] if len(tp_hit) else None
    stop_i = stop_hit[0] if len(stop_hit) else None
    if tp_i is not None and (stop_i is None or tp_i < stop_i):
        return int(ts[i0 + tp_i]), px_entry * (1 + side * trigger.tp_pct / 100), "tp"
    if stop_i is not None:
        return int(ts[i0 + stop_i]), px_entry * (1 + side * trigger.stop_pct / 100), "stop"
    return int(ts[i1 - 1]), float(mid[i1 - 1]), "horizon"


def generate_trades(trigger: Trigger, obi: pd.DataFrame, flow: pd.DataFrame,
                    mids: pd.DataFrame, max_quote_age_ms: int) -> pd.DataFrame:
    """One row per trade: entry/exit time and price, reason, gross PnL in bps,
    and the trigger minute's cross-venue TFI and its z-score."""
    cols = ["entry_ts_ms", "exit_ts_ms", "side", "entry_px", "exit_px", "exit_reason",
            "hold_min", "gross_bps", "tfi", "tfi_z"]
    if obi.empty or mids.empty:
        return pd.DataFrame(columns=cols)
    ts, mid = mids["ts_ms"].to_numpy(np.int64), mids["mid"].to_numpy(np.float64)
    lookup = StepLookup(ts, mid, max_quote_age_ms)
    flow_by_minute = {}
    if not flow.empty:
        flow_by_minute = dict(zip(flow["bucket"].to_numpy(np.int64),
                                  zip(flow["xv_tfi"].to_numpy(), flow["xv_tfi_z"].to_numpy())))
    z = {w: obi[f"z_{w}"].to_numpy() for w in trigger.windows_min}
    minutes = obi["ts_min"].to_numpy(np.int64)
    rows, busy_until = [], -1
    for i, minute_start in enumerate(minutes):
        decided = int(minute_start) + MINUTE_MS
        if decided <= busy_until:
            continue
        side = direction(trigger, {w: z[w][i] for w in trigger.windows_min})
        if side == 0:
            continue
        px = lookup.at(decided)
        if not px > 0:
            continue
        exit_ = simulate_exit(decided, side, px, trigger, ts, mid)
        if exit_ is None:
            continue
        t_exit, px_exit, reason = exit_
        tfi, tfi_z = flow_by_minute.get(int(minute_start) // MINUTE_MS, (np.nan, np.nan))
        rows.append((decided, t_exit, side, px, px_exit, reason, (t_exit - decided) / MINUTE_MS,
                     side * (px_exit - px) / px * 1e4, tfi, tfi_z))
        busy_until = t_exit
    return pd.DataFrame(rows, columns=cols)


def apply_gate(trades: pd.DataFrame, min_abs_z: Optional[float]) -> pd.DataFrame:
    """Trades whose trigger-minute TFI agrees in sign (and, if given, |z| >= min_abs_z)."""
    keep = np.sign(trades["tfi"]) == trades["side"]
    if min_abs_z is not None:
        keep &= trades["tfi_z"].abs() >= min_abs_z
    return trades[keep.fillna(False)]


def summarize(trades: pd.DataFrame, days: float, cfg: Config) -> dict:
    if trades.empty:
        return {"n": 0}
    pnl = trades.sort_values("entry_ts_ms")["gross_bps"].to_numpy()
    n = len(pnl)
    half = n // 2
    m1 = float(np.median(pnl[:half])) if half >= cfg.min_half_n else np.nan
    m2 = float(np.median(pnl[half:])) if n - half >= cfg.min_half_n else np.nan
    std = float(pnl.std(ddof=1)) if n > 1 else np.nan
    median = float(np.median(pnl))
    reasons = trades["exit_reason"].value_counts()
    s = {"n": n, "n_tp": int(reasons.get("tp", 0)), "n_stop": int(reasons.get("stop", 0)),
         "n_horizon": int(reasons.get("horizon", 0)), "median": median, "mean": float(pnl.mean()),
         "p25": float(np.percentile(pnl, 25)), "p75": float(np.percentile(pnl, 75)),
         "win_pct": 100 * float((pnl > 0).mean()), "sharpe": float(pnl.mean() / std) if std and std > 0 else np.nan,
         "hold_avg": float(trades["hold_min"].mean()), "median_first": m1, "median_second": m2,
         "halves_positive": bool(m1 > 0 and m2 > 0),
         "net_taker": median - cfg.cost_taker_bps, "net_mixed": median - cfg.cost_mixed_bps}
    s["bps_per_day_taker"] = s["net_taker"] * n / days if days > 0 else np.nan
    s["bps_per_day_mixed"] = s["net_mixed"] * n / days if days > 0 else np.nan
    s["apr_taker_pct"] = s["bps_per_day_taker"] * 365 / 100
    s["passed"] = bool(s["net_taker"] >= cfg.gate_net_bps and n >= cfg.min_n and s["halves_positive"])
    return s


@dataclass
class Variant:
    symbol: str
    trigger: Trigger
    gate: str            # "ungated" / "soft" / "strict"
    trades: pd.DataFrame
    stats: dict


@dataclass
class Result:
    config: Config
    coverage: dict       # symbol -> {"days", "minutes", "flow_minutes"} or None
    variants: list[Variant]


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    unknown = [s for s in cfg.symbols if s not in PRESETS]
    if unknown:
        raise ValueError(f"no trigger presets for {unknown}; known: {sorted(PRESETS)}")
    coverage, variants = {}, []
    for symbol in cfg.symbols:
        triggers = PRESETS[symbol]
        obi = cross_venue_minutes(files, symbol)
        mids = venue_mid_series(files, symbol, HL_VENUE)
        if obi.empty or mids.empty:
            coverage[symbol] = None
            continue
        for w in sorted({w for t in triggers for w in t.windows_min}):
            obi[f"z_{w}"] = _zscore(obi["avg_obi"].to_numpy(), w)
        flow = cross_venue_flow(cfg.capture_dir, symbol, MINUTE_MS, cfg.flow_venues, cfg.start, cfg.end)
        if not flow.empty:
            flow["xv_tfi_z"] = _zscore(flow["xv_tfi"].to_numpy(), cfg.tfi_z_window_min)
        days = (obi["ts_min"].max() - obi["ts_min"].min()) / 86_400_000
        coverage[symbol] = {"days": days, "minutes": len(obi), "flow_minutes": len(flow),
                            "mid_rows": len(mids)}
        for trigger in triggers:
            trades = generate_trades(trigger, obi, flow, mids, cfg.max_quote_age_ms)
            gates = [("ungated", trades)]
            if not flow.empty and not trades.empty:
                gates += [("soft", apply_gate(trades, None)), ("strict", apply_gate(trades, cfg.strict_tfi_z))]
            for gate, part in gates:
                variants.append(Variant(symbol, trigger, gate, part, summarize(part, days, cfg)))
    return Result(cfg, coverage, variants)


def _block(v: Variant) -> list[str]:
    s = v.stats
    head = f"    {v.gate:<8}"
    if s["n"] == 0:
        return [f"{head}no trades"]
    halves = ("na" if np.isnan(s["median_first"]) or np.isnan(s["median_second"])
              else f"{s['median_first']:+.2f}/{s['median_second']:+.2f}")
    return [f"{head}n={s['n']:<4} tp/stop/horizon={s['n_tp']}/{s['n_stop']}/{s['n_horizon']}  "
            f"avg hold {s['hold_avg']:.0f}m  median={s['median']:+.2f}  mean={s['mean']:+.2f}  "
            f"win={s['win_pct']:.1f}%  sharpe={s['sharpe']:+.3f}",
            f"{'':<12}net/trade: taker {s['net_taker']:+.2f}  mixed {s['net_mixed']:+.2f}   "
            f"bps/day: taker {s['bps_per_day_taker']:+.2f} ({s['apr_taker_pct']:+.1f}% APR at 1x)  "
            f"mixed {s['bps_per_day_mixed']:+.2f}   halves {halves}" + ("   * PASS" if s["passed"] else "")]


def render(result: Result) -> str:
    cfg = result.config
    out = [rule(110), "Imbalance trigger configurations under identical exits", rule(110),
           f"  costs (round trip incl. spread): taker {cfg.cost_taker_bps:.2f}  maker-in/taker-out "
           f"{cfg.cost_mixed_bps:.2f}  maker {cfg.cost_maker_bps:.2f} bps",
           f"  gate: median net of taker cost >= {cfg.gate_net_bps} bps, n >= {cfg.min_n}, "
           "median positive in both halves",
           f"  TFI gates: soft = sign agrees; strict = sign agrees and |z| >= {cfg.strict_tfi_z}"]
    for symbol, cov in result.coverage.items():
        out += ["", rule(110), f"  {symbol}", rule(110)]
        if cov is None:
            out.append("  no imbalance or Hyperliquid mid data")
            continue
        out.append(f"  {cov['minutes']:,} minutes over {cov['days']:.1f} days   "
                   f"flow minutes: {cov['flow_minutes']:,}" + ("" if cov["flow_minutes"] else " (TFI gates skipped)"))
        for trigger in PRESETS[symbol]:
            tag = "  (reference)" if trigger.reference else ""
            out += ["", f"  {trigger.label}{tag}: {trigger.describe()}"]
            for v in (v for v in result.variants if v.symbol == symbol and v.trigger == trigger):
                out += _block(v)

    out += ["", rule(110), "  Summary  (* passes gate, R = reference configuration ungated)", rule(110),
            f"  {'sym':<5} {'configuration':<20} {'gate':<8}{'n':>5}{'median':>9}{'net@tk':>9}{'bps/day':>10}"
            f"{'APR%':>9}{'win%':>7}{'halves':>8}"]
    for v in result.variants:
        s = v.stats
        if s["n"] == 0:
            continue
        mark = "*" if s["passed"] else " "
        ref = "R" if v.trigger.reference and v.gate == "ungated" else " "
        out.append(f" {mark}{ref}{v.symbol:<4} {v.trigger.label:<20} {v.gate:<8}{s['n']:>5}{s['median']:>+9.2f}"
                   f"{s['net_taker']:>+9.2f}{s['bps_per_day_taker']:>+10.2f}{s['apr_taker_pct']:>+9.1f}"
                   f"{s['win_pct']:>6.1f}%{'Y' if s['halves_positive'] else 'N':>8}")

    out += ["", "  Best variant per symbol (highest bps/day among variants positive in both halves):"]
    for symbol in result.coverage:
        rows = [v for v in result.variants if v.symbol == symbol and v.stats["n"]]
        stable = [v for v in rows if v.stats["halves_positive"]]
        if not stable:
            out.append(f"    {symbol}: none")
            continue
        best = max(stable, key=lambda v: v.stats["bps_per_day_taker"])
        line = (f"    {symbol}: {best.trigger.label} / {best.gate}  {best.stats['bps_per_day_taker']:+.2f} bps/day "
                f"({best.stats['apr_taker_pct']:+.1f}% APR at 1x, n={best.stats['n']})")
        ref = next((v for v in rows if v.trigger.reference and v.gate == "ungated"), None)
        if ref is not None and ref is not best:
            line += f"   vs reference {best.stats['bps_per_day_taker'] - ref.stats['bps_per_day_taker']:+.2f} bps/day"
        out.append(line)
    return "\n".join(out)


def write_outputs(result: Result, outdir: Path) -> tuple[Path, Path] | None:
    if not result.variants:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "obi_configs_summary.csv"
    pd.DataFrame([{"symbol": v.symbol, "trigger": v.trigger.label, "reference": v.trigger.reference,
                   "gate": v.gate, **v.stats} for v in result.variants]).to_csv(summary, index=False)
    trades = outdir / "obi_configs_trades.csv"
    frames = [v.trades.assign(symbol=v.symbol, trigger=v.trigger.label)
              for v in result.variants if v.gate == "ungated" and not v.trades.empty]
    (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(trades, index=False)
    return summary, trades


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro obi-configs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbols", type=csv_list(str.upper), default=d.symbols,
                        help=f"symbols with presets: {', '.join(PRESETS)}")
    parser.add_argument("--flow-venues", type=csv_list(str.lower), default=d.flow_venues)
    parser.add_argument("--strict-tfi-z", type=float, default=d.strict_tfi_z)
    parser.add_argument("--gate-net-bps", type=float, default=d.gate_net_bps)
    parser.add_argument("--min-n", type=int, default=d.min_n)
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--maker-bps", type=float, default=d.fees.maker_bps)
    parser.add_argument("--half-spread-bps", type=float, default=d.fees.half_spread_bps)
    parser.add_argument("--outdir", type=Path, default=Path("results/obi_configs"))
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbols=args.symbols, flow_venues=args.flow_venues,
                 strict_tfi_z=args.strict_tfi_z, gate_net_bps=args.gate_net_bps, min_n=args.min_n,
                 fees=FeeModel(taker_bps=args.taker_bps, maker_bps=args.maker_bps,
                               half_spread_bps=args.half_spread_bps),
                 start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    paths = write_outputs(result, args.outdir)
    if paths:
        print(f"\nwrote {paths[0]}\nwrote {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
