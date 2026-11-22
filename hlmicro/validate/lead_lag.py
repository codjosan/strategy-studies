"""Event study: how the Hyperliquid mid responds to large OKX/Bybit prints.

Events are CEX trades above a notional threshold that arrive while the
Hyperliquid mid has been stable (move below ``stability_bps`` over the
preceding ``stability_window_ms``). For each event and lag L, signed by the
CEX aggressor side:

* ``residual_at_lag``     = HL mid(T+L) - HL mid(T), in bps
* ``capturable_from_lag`` = residual at the final lag - residual at L, i.e.
  the move still available to a trade entered at T+L

Gate (per venue, threshold and side): capturable at the target lag is at
least the round-trip taker cost, n >= ``min_events``, and both chronological
halves have the same sign.

The Hyperliquid mid comes from websocket book updates (``obi_ws``). A mid older
than ``max_quote_age_ms`` counts as missing, so a gap in the websocket
recording (disconnect, collector restart) cannot pass as a stable price, and
no residual is measured against a stale one.

Usage::

    python -m hlmicro lead-lag --symbol BTC --days 7 --plot-out btc_lead_lag.png
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
from ..parse import discover, load_tape, venue_mid_series
from ..schemas import BOOK_WS_PREFIX, CEX_TRADE_PREFIXES
from .common import StepLookup, rule

log = logging.getLogger(__name__)

NOTIONAL_THRESHOLDS = {
    "BTC": [50_000, 100_000, 250_000],
    "ETH": [25_000, 50_000, 100_000],
    "SOL": [10_000, 25_000, 50_000],
    "HYPE": [10_000, 25_000, 50_000],
    "DOGE": [10_000, 25_000, 50_000],
    "LINK": [5_000, 10_000, 25_000],
}
VENUE_LABELS = {"okx": "OKX", "bybit": "Bybit"}


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    days: int | None = 3
    start: str | None = None
    end: str | None = None
    lags_ms: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000])
    target_lag_ms: int = 500
    thresholds: list[float] | None = None
    stability_window_ms: int = 5_000
    stability_bps: float = 0.5
    min_events: int = 30
    min_events_to_report: int = 5
    max_quote_age_ms: int | None = 5_000     # None = no limit
    fees: FeeModel = DEFAULT_FEES

    @property
    def round_trip_cost_bps(self) -> float:
        return self.fees.round_trip_with_spread_bps(self.fees.taker_bps, self.fees.taker_bps)

    @property
    def settlement_lag_ms(self) -> int:
        return max(self.lags_ms)


def detect_events(trades: pd.DataFrame, threshold: float, hl: StepLookup, cfg: Config) -> pd.DataFrame:
    """Trades at or above ``threshold`` notional with a stable HL mid beforehand."""
    if trades.empty:
        return trades
    frame = trades.assign(notional=trades["px"] * trades["sz"])
    frame = frame[frame["notional"] >= threshold]
    if frame.empty:
        return frame
    t = frame["ts_ms"].to_numpy()
    mid_now = hl.at(t)
    mid_before = hl.at(t - cfg.stability_window_ms)
    valid = np.isfinite(mid_now) & np.isfinite(mid_before) & (mid_now > 0)
    move_bps = np.full(len(t), np.inf)
    move_bps[valid] = np.abs(mid_now[valid] - mid_before[valid]) / mid_now[valid] * 1e4
    keep = valid & (move_bps < cfg.stability_bps)
    return frame[keep].assign(hl_mid_at_event=mid_now[keep])


def event_study(events: pd.DataFrame, hl: StepLookup, cfg: Config) -> pd.DataFrame:
    """One row per (event, lag) with residual and capturable move in bps."""
    if events.empty:
        return pd.DataFrame()
    t = events["ts_ms"].to_numpy()
    mid_t = events["hl_mid_at_event"].to_numpy()
    sign = events["sign"].to_numpy()
    residual = {lag: (hl.at(t + lag) - mid_t) / mid_t * 1e4 * sign
                for lag in sorted(set(cfg.lags_ms) | {cfg.settlement_lag_ms})}
    final = residual[cfg.settlement_lag_ms]
    study = pd.concat([pd.DataFrame({
        "lag_ms": lag, "event_ts_ms": t, "side": events["side"].to_numpy(),
        "notional": events["notional"].to_numpy(), "venue": events["venue"].to_numpy(),
        "residual_bps": residual[lag], "capturable_bps": final - residual[lag],
    }) for lag in cfg.lags_ms], ignore_index=True)
    return study.dropna(subset=["residual_bps", "capturable_bps"])


def summarize(study: pd.DataFrame) -> pd.DataFrame:
    return study.groupby(["side", "lag_ms"]).agg(
        n=("residual_bps", "size"),
        residual_mean=("residual_bps", "mean"), residual_median=("residual_bps", "median"),
        capturable_mean=("capturable_bps", "mean"), capturable_median=("capturable_bps", "median"),
    ).reset_index()


def halves(study: pd.DataFrame, lag_ms: int) -> dict | None:
    """Mean capturable move in the first and second half of events (by time)."""
    at_lag = study[study["lag_ms"] == lag_ms]
    if at_lag.empty:
        return None
    cut = at_lag["event_ts_ms"].median()
    first = at_lag[at_lag["event_ts_ms"] < cut]["capturable_bps"]
    second = at_lag[at_lag["event_ts_ms"] >= cut]["capturable_bps"]
    c1 = first.mean() if len(first) else np.nan
    c2 = second.mean() if len(second) else np.nan
    return {"n_first": len(first), "n_second": len(second), "cap_first": c1, "cap_second": c2,
            "sign_stable": bool(not np.isnan(c1) and not np.isnan(c2) and np.sign(c1) == np.sign(c2))}


@dataclass
class Cell:
    venue: str
    threshold: float
    side: str
    summary: pd.DataFrame
    halves: dict | None

    def at_target(self, lag_ms: int) -> pd.Series | None:
        row = self.summary[self.summary["lag_ms"] == lag_ms]
        return None if row.empty else row.iloc[0]


@dataclass
class Result:
    config: Config
    n_trades: dict
    hl_rows: int
    hl_hours: float
    clock: dict
    cells: list[Cell]
    skipped: list[tuple[str, float, int]]

    def passing(self) -> list[Cell]:
        cfg = self.config
        out = []
        for cell in self.cells:
            row = cell.at_target(cfg.target_lag_ms)
            if (row is not None and row["capturable_mean"] >= cfg.round_trip_cost_bps
                    and row["n"] >= cfg.min_events and cell.halves and cell.halves["sign_stable"]):
                out.append(cell)
        return out


def run(cfg: Config) -> Result:
    last_n = None if (cfg.start or cfg.end) else cfg.days
    tapes = {}
    for venue, prefix in CEX_TRADE_PREFIXES.items():
        files = discover(cfg.capture_dir, prefix, cfg.start, cfg.end, last_n)
        tapes[venue] = load_tape(files, cfg.symbol).assign(venue=VENUE_LABELS[venue])
    hl_files = discover(cfg.capture_dir, BOOK_WS_PREFIX, cfg.start, cfg.end, last_n)
    hl_mid = venue_mid_series(hl_files, cfg.symbol, HL_VENUE)
    if hl_mid.empty:
        raise ValueError(f"no Hyperliquid websocket mids for {cfg.symbol}")
    if all(t.empty for t in tapes.values()):
        raise ValueError(f"no CEX trades for {cfg.symbol}")
    hl = StepLookup(hl_mid["ts_ms"].to_numpy(), hl_mid["mid"].to_numpy(), cfg.max_quote_age_ms)

    clock = {}
    for venue, tape in tapes.items():
        if "recv_ts_ms" in tape and not tape.empty:
            drift = (pd.to_numeric(tape["recv_ts_ms"], errors="coerce") - tape["ts_ms"]).dropna()
            if len(drift):
                clock[VENUE_LABELS[venue]] = {"n": len(drift), "mean": drift.mean(), "median": drift.median(),
                                              "p95": drift.quantile(0.95), "p99": drift.quantile(0.99)}

    thresholds = cfg.thresholds or NOTIONAL_THRESHOLDS.get(cfg.symbol)
    if not thresholds:
        raise ValueError(f"no notional thresholds for {cfg.symbol}; pass --thresholds")
    cells, skipped = [], []
    for venue, tape in tapes.items():
        if tape.empty:
            continue
        for threshold in thresholds:
            events = detect_events(tape, threshold, hl, cfg)
            if len(events) < cfg.min_events_to_report:
                skipped.append((VENUE_LABELS[venue], threshold, len(events)))
                continue
            study = event_study(events, hl, cfg)
            summary = summarize(study)
            for side in ("B", "A"):
                cells.append(Cell(VENUE_LABELS[venue], threshold, side,
                                  summary[summary["side"] == side].reset_index(drop=True),
                                  halves(study[study["side"] == side], cfg.target_lag_ms)))
    return Result(config=cfg, n_trades={VENUE_LABELS[v]: len(t) for v, t in tapes.items()},
                  hl_rows=len(hl_mid),
                  hl_hours=(hl_mid["ts_ms"].iloc[-1] - hl_mid["ts_ms"].iloc[0]) / 3_600_000,
                  clock=clock, cells=cells, skipped=skipped)


def render(result: Result) -> str:
    cfg = result.config
    cost = cfg.round_trip_cost_bps
    window = (f"{cfg.start or 'first'} .. {cfg.end or 'last'}" if (cfg.start or cfg.end)
              else f"last {cfg.days} day(s)")
    out = [rule(), f"Lead-lag event study - {cfg.symbol}", rule(),
           f"  capture: {cfg.capture_dir}   window: {window}",
           f"  cost: 2 x ({cfg.fees.taker_bps:.2f} taker + {cfg.fees.half_spread_bps:.2f} half-spread) "
           f"= {cost:.2f} bps round trip",
           f"  HL mid max age: {f'{cfg.max_quote_age_ms} ms' if cfg.max_quote_age_ms else 'no limit'}",
           "  trades: " + "  ".join(f"{k} {v:,}" for k, v in result.n_trades.items())
           + f"   HL mid updates: {result.hl_rows:,} ({result.hl_hours:.1f}h)", "",
           "  clock: receive time minus exchange time (ms)"]
    for venue, c in result.clock.items():
        out.append(f"    {venue:<6} n={c['n']:,}  mean={c['mean']:.0f}  median={c['median']:.0f}  "
                   f"p95={c['p95']:.0f}  p99={c['p99']:.0f}")
    out += ["", rule(), f"  HL response to CEX prints (HL stable < {cfg.stability_bps} bps over "
            f"prior {cfg.stability_window_ms} ms; * = meets cost and n at {cfg.target_lag_ms} ms)", rule()]
    for venue, threshold, n in result.skipped:
        out.append(f"  {venue} >= ${threshold:,.0f}: {n} events, not reported")
    by_key: dict = {}
    for cell in result.cells:
        by_key.setdefault((cell.venue, cell.threshold), []).append(cell)
    for (venue, threshold), cells in by_key.items():
        out += ["", f"  {venue} prints >= ${threshold:,.0f}",
                f"    {'lag_ms':>7}{'side':>5}{'n':>6}{'resid_mean':>12}{'resid_med':>11}"
                f"{'cap_mean':>10}{'cap_med':>9}{'net':>9}"]
        rows = pd.concat([c.summary for c in cells]).sort_values(["side", "lag_ms"])
        for _, r in rows.iterrows():
            net = r["capturable_mean"] - cost
            mark = "*" if (net >= 0 and r["n"] >= cfg.min_events and r["lag_ms"] == cfg.target_lag_ms) else " "
            out.append(f"   {mark}{int(r['lag_ms']):>6}{r['side']:>5}{int(r['n']):>6}"
                       f"{r['residual_mean']:>+12.2f}{r['residual_median']:>+11.2f}"
                       f"{r['capturable_mean']:>+10.2f}{r['capturable_median']:>+9.2f}{net:>+9.2f}")
    out += ["", rule(), f"  Chronological halves at {cfg.target_lag_ms} ms", rule(),
            f"  {'venue':>7}{'threshold':>11}{'side':>5}{'n1':>6}{'n2':>6}{'cap1':>9}{'cap2':>9}  same sign"]
    for cell in result.cells:
        h = cell.halves
        if h:
            out.append(f"  {cell.venue:>7}{cell.threshold:>11,.0f}{cell.side:>5}{h['n_first']:>6}{h['n_second']:>6}"
                       f"{h['cap_first']:>+9.2f}{h['cap_second']:>+9.2f}  {'yes' if h['sign_stable'] else 'no'}")
    out += ["", rule(), "  Gate", rule(),
            f"  capturable at {cfg.target_lag_ms} ms >= {cost:.2f} bps, n >= {cfg.min_events}, halves same sign"]
    passing = result.passing()
    if passing:
        for cell in passing:
            r = cell.at_target(cfg.target_lag_ms)
            out.append(f"  PASS  {cell.venue} >= ${cell.threshold:,.0f} side {cell.side}: "
                       f"capturable {r['capturable_mean']:+.2f} bps, n={int(r['n'])}, "
                       f"net {r['capturable_mean'] - cost:+.2f} bps")
    else:
        best = max(((c, c.at_target(cfg.target_lag_ms)) for c in result.cells
                    if c.at_target(cfg.target_lag_ms) is not None),
                   key=lambda pair: pair[1]["capturable_mean"], default=None)
        if best is None:
            out.append("  FAIL  no events at the target lag")
        else:
            cell, r = best
            out.append(f"  FAIL  no configuration passes; best {cell.venue} >= ${cell.threshold:,.0f} "
                       f"side {cell.side}: capturable {r['capturable_mean']:+.2f} bps "
                       f"(cost {cost:.2f}), n={int(r['n'])}")
    return "\n".join(out)


def plot(result: Result, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = result.config
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [("B", "residual_mean", "buyer-initiated, residual"),
              ("A", "residual_mean", "seller-initiated, residual"),
              ("B", "capturable_mean", "buyer-initiated, capturable"),
              ("A", "capturable_mean", "seller-initiated, capturable")]
    for ax, (side, column, title) in zip(axes.flat, panels):
        for cell in result.cells:
            if cell.side == side and not cell.summary.empty:
                ax.plot(cell.summary["lag_ms"], cell.summary[column], marker="o",
                        label=f"{cell.venue} ${cell.threshold / 1000:.0f}k (n~{int(cell.summary['n'].mean())})")
        ax.set(xscale="log", xlabel="lag after CEX print (ms)", ylabel="bps, signed by side",
               title=f"{cfg.symbol}: {title}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(cfg.target_lag_ms, color="red", lw=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro lead-lag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--days", type=int, default=d.days,
                        help="most recent N days (ignored when --start/--end are given)")
    parser.add_argument("--thresholds", type=csv_list(float), default=None,
                        help="notional thresholds in USD (default: per-symbol table)")
    parser.add_argument("--taker-bps", type=float, default=d.fees.taker_bps)
    parser.add_argument("--half-spread-bps", type=float, default=d.fees.half_spread_bps)
    parser.add_argument("--max-quote-age-ms", type=int, default=d.max_quote_age_ms,
                        help="treat a Hyperliquid mid older than this as missing; 0 = no limit")
    parser.add_argument("--plot-out", type=Path, default=None, help="save a PNG of the response curves")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, days=args.days,
                 start=args.start, end=args.end, thresholds=args.thresholds,
                 max_quote_age_ms=args.max_quote_age_ms or None,
                 fees=FeeModel(taker_bps=args.taker_bps, half_spread_bps=args.half_spread_bps))
    result = run(cfg)
    print(render(result))
    if args.plot_out:
        plot(result, args.plot_out)
        print(f"\nwrote {args.plot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
