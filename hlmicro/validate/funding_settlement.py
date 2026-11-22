"""Price behaviour around Hyperliquid's hourly funding settlement.

Hypothesis: when funding is extreme, leveraged traders close positions before
settlement to avoid paying it (selling ahead of high positive funding, buying
ahead of high negative funding), and the pressure reverses afterwards.

For each settlement hour T with price coverage, returns (in %) are measured
over four windows: early T-15m..T-5m, final T-5m..T, post T..T+5m and late
T+5m..T+15m, then grouped by funding-rate quantile. A simple rule is
backtested: short at T-5m when funding is in the top ``p``%, long when in the
bottom ``p``%, exit at T+5m. The position is open at settlement T, so it
collects (or pays) that hour's funding: longs pay ``rate`` and shorts receive
it, as a fraction of notional.

Prices are the mid of one venue from 5-second book snapshots. Funding comes
from a CSV (``--funding-csv``, default ``hl_funding_<COIN>.csv`` in the
capture directory when present) or from the public API.

Usage::

    python -m hlmicro funding-settlement --symbol BTC
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, default_capture_dir
from ..parse import discover, fetch_funding_history, funding_by_hour, load_funding_csv, venue_mid_series
from ..schemas import BOOK_REST_PREFIX, FUNDING_FILE_TEMPLATE
from .common import StepLookup, rule

log = logging.getLogger(__name__)

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
WINDOWS = {"early": (-15, -5), "final": (-5, 0), "post": (0, 5), "late": (5, 15)}


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "BTC"
    venue: str = HL_VENUE
    funding_csv: Path | None = None
    fees_bps_per_side: list[float] = field(
        default_factory=lambda: [DEFAULT_FEES.taker_bps, DEFAULT_FEES.maker_bps, 0.0])
    percentiles: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0])
    capital: float = 1000.0
    max_skew_ms: int = 30_000
    min_events: int = 30
    start: str | None = None
    end: str | None = None


def percentile(values, p: float) -> float:
    """Linear-interpolated percentile, ``p`` in [0, 100]."""
    return float(np.percentile(np.asarray(values, dtype=float), p)) if len(values) else 0.0


def mean_sem_t(values) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "t": 0.0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "sem": sem, "t": mean / sem if sem > 0 else 0.0}


def settlement_events(prices: StepLookup, funding: dict[int, float], max_skew_ms: int) -> list[dict]:
    """Per settlement hour: funding rate and window returns in percent."""
    if len(prices.ts) == 0:
        return []
    first, last = int(prices.ts[0]), int(prices.ts[-1])
    events = []
    offsets = (-15, -5, 0, 5, 15)
    for hour, rate in sorted(funding.items()):
        if hour < first + 15 * MINUTE_MS or hour > last - 15 * MINUTE_MS:
            continue
        px = dict(zip(offsets, prices.nearest(np.array([hour + m * MINUTE_MS for m in offsets]), max_skew_ms)))
        if any(np.isnan(v) for v in px.values()):
            continue
        event = {"hour_ms": hour, "funding_rate": rate}
        for name, (a, b) in WINDOWS.items():
            event[f"{name}_ret"] = (px[b] / px[a] - 1) * 100
        events.append(event)
    return events


def bucket_events(events: list[dict]) -> dict:
    rates = [e["funding_rate"] for e in events]
    q = {p: percentile(rates, p) for p in (10, 25, 50, 75, 90)}
    buckets = {
        "top_10%": [e for e in events if e["funding_rate"] >= q[90]],
        "top_25%": [e for e in events if e["funding_rate"] >= q[75]],
        "middle": [e for e in events if q[25] < e["funding_rate"] < q[75]],
        "bot_25%": [e for e in events if e["funding_rate"] <= q[25]],
        "bot_10%": [e for e in events if e["funding_rate"] <= q[10]],
    }
    stats = {name: {w: mean_sem_t([e[f"{w}_ret"] for e in evts]) for w in WINDOWS}
             for name, evts in buckets.items() if evts}
    return {"quantiles": q, "min": min(rates), "max": max(rates),
            "sizes": {k: len(v) for k, v in buckets.items()}, "stats": stats}


def backtest(events: list[dict], pct: float, fee_bps: float, capital: float) -> dict | None:
    """Short top ``pct``% funding, long bottom ``pct``%, from T-5m to T+5m."""
    rates = sorted(e["funding_rate"] for e in events)
    if len(rates) < 20:
        return None
    high, low = percentile(rates, 100 - pct), percentile(rates, pct)
    round_trip_pct = 2 * fee_bps / 100
    trades = []
    for e in events:
        if e["funding_rate"] >= high:
            direction = -1
        elif e["funding_rate"] <= low:
            direction = 1
        else:
            continue
        held = ((1 + e["final_ret"] / 100) * (1 + e["post_ret"] / 100) - 1) * 100   # T-5m .. T+5m
        price = direction * held
        funding = -direction * e["funding_rate"] * 100       # % of notional, settled at T
        gross = price + funding
        trades.append({"hour_ms": e["hour_ms"], "price": price, "funding": funding, "gross": gross,
                       "net": gross - round_trip_pct})
    if not trades:
        return None
    net = [t["net"] for t in trades]
    span_h = (events[-1]["hour_ms"] - events[0]["hour_ms"]) / HOUR_MS
    total_net = sum(net)
    s = mean_sem_t(net)
    return {"pct": pct, "high": high, "low": low, "fee_bps": fee_bps, "n": len(trades),
            "win_rate": sum(1 for x in net if x > 0) / len(net) * 100,
            "price_total": sum(t["price"] for t in trades),
            "funding_total": sum(t["funding"] for t in trades),
            "gross_total": sum(t["gross"] for t in trades), "net_total": total_net,
            "pnl_usd": total_net / 100 * capital, "apr_pct": total_net / span_h * 24 * 365,
            "mean": s["mean"], "sem": s["sem"], "t": s["t"], "trades": trades}


@dataclass
class Result:
    config: Config
    n_prices: int
    price_span_days: float
    n_funding_hours: int
    events: list[dict]
    buckets: dict
    backtests: list[dict]


def _funding(cfg: Config, start_ms: int, end_ms: int) -> dict[int, float]:
    path = cfg.funding_csv or cfg.capture_dir / FUNDING_FILE_TEMPLATE.format(coin=cfg.symbol)
    if path.exists():
        log.info("funding from %s", path)
        return funding_by_hour(load_funding_csv(path))
    log.info("funding from the Hyperliquid API")
    return funding_by_hour(fetch_funding_history(cfg.symbol, start_ms, end_ms))


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    prices = venue_mid_series(files, cfg.symbol, cfg.venue)
    if prices.empty:
        raise ValueError(f"no {cfg.venue} mids for {cfg.symbol} in {len(files)} {BOOK_REST_PREFIX} files")
    lookup = StepLookup(prices["ts_ms"].to_numpy(), prices["mid"].to_numpy())
    first, last = int(prices["ts_ms"].iloc[0]), int(prices["ts_ms"].iloc[-1])
    funding = _funding(cfg, first - HOUR_MS, last + HOUR_MS)
    events = settlement_events(lookup, funding, cfg.max_skew_ms)
    if len(events) < cfg.min_events:
        raise ValueError(f"only {len(events)} settlement hours with full coverage (need {cfg.min_events})")
    tests = [b for fee in cfg.fees_bps_per_side for p in cfg.percentiles
             if (b := backtest(events, p, fee, cfg.capital)) is not None]
    return Result(cfg, len(prices), (last - first) / 86_400_000, len(funding), events,
                  bucket_events(events), tests)


def render(result: Result) -> str:
    cfg = result.config
    b = result.buckets
    q = b["quantiles"]
    out = [rule(80, "="), f"Funding settlement study - {cfg.symbol} ({cfg.venue} mid)", rule(80, "="),
           f"  prices: {result.n_prices:,} over {result.price_span_days:.2f} days   "
           f"funding hours: {result.n_funding_hours}   events with full coverage: {len(result.events)}", "",
           f"  funding rate quantiles (%): p10={q[10] * 100:+.5f}  p25={q[25] * 100:+.5f}  "
           f"p50={q[50] * 100:+.5f}  p75={q[75] * 100:+.5f}  p90={q[90] * 100:+.5f}",
           f"  range (%): {b['min'] * 100:+.5f} .. {b['max'] * 100:+.5f}",
           "  bucket sizes: " + "  ".join(f"{k}={v}" for k, v in b["sizes"].items()), "",
           rule(120, "-"), "  Mean return (%) +/- SEM by funding bucket and window, t in brackets", rule(120, "-"),
           f"  {'bucket':<10}{'n':>5}  " + "".join(f"{w + ' ' + str(a) + '..' + str(z) + 'm':>27}"
                                                  for w, (a, z) in WINDOWS.items())]
    for name, stats in b["stats"].items():
        cells = "".join(f"{s['mean']:>+11.4f} +/-{s['sem']:.4f} [{s['t']:+5.2f}]" for s in stats.values())
        out.append(f"  {name:<10}{b['sizes'][name]:>5}  {cells}")

    if "top_10%" in b["stats"] and "bot_10%" in b["stats"]:
        top, bot = b["stats"]["top_10%"], b["stats"]["bot_10%"]
        checks = [("top 10%: final window negative", top["final"]["t"] < -1),
                  ("top 10%: post window positive", top["post"]["t"] > 1),
                  ("bottom 10%: final window positive", bot["final"]["t"] > 1),
                  ("bottom 10%: post window negative", bot["post"]["t"] < -1)]
        out += ["", "  Direction check (|t| > 1 in the predicted direction):"]
        out += [f"    {'yes' if ok else 'no ':<4}{label}" for label, ok in checks]

    out += ["", rule(80, "="), "  Backtest: short top p%, long bottom p%, T-5m to T+5m (gross = price + funding)",
            rule(80, "=")]
    for r in result.backtests:
        out += [f"  fee {r['fee_bps']:.2f} bps/side, p={r['pct']:.0f}%  "
                f"(funding >= {r['high'] * 100:+.4f}% or <= {r['low'] * 100:+.4f}%)",
                f"    n={r['n']}  win={r['win_rate']:.1f}%  price={r['price_total']:+.3f}%  "
                f"funding={r['funding_total']:+.3f}%  gross={r['gross_total']:+.3f}%  "
                f"net={r['net_total']:+.3f}%  mean={r['mean']:+.4f}%  SEM={r['sem']:.4f}%  "
                f"t={r['t']:+.2f}  annualised={r['apr_pct']:+.2f}%"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro funding-settlement", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--venue", type=str.lower, default=d.venue, help="price source venue")
    parser.add_argument("--funding-csv", type=Path, default=None)
    parser.add_argument("--fee-bps", type=csv_list(float), default=d.fees_bps_per_side,
                        help="per-side fees to test, comma-separated")
    parser.add_argument("--percentiles", type=csv_list(float), default=d.percentiles)
    parser.add_argument("--capital", type=float, default=d.capital)
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, venue=args.venue,
                 funding_csv=args.funding_csv, fees_bps_per_side=args.fee_bps,
                 percentiles=args.percentiles, capital=args.capital, start=args.start, end=args.end)
    print(render(run(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
