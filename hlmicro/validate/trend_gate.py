"""Trend gate on the long side of the imbalance z-score signal.

Hypothesis: suppressing long entries taken against the prevailing trend
improves the long book. A long is kept only when the Hyperliquid mid at entry
is at or above its trailing ``N``-minute simple moving average (past and
current minutes only; while the average is undefined the trade is kept).
Shorts are never touched.

The baseline is the signal of ``tp-multifold`` with a stop and take-profit,
driven by the cross-venue mean imbalance (each venue forward-filled) with PnL
on the Hyperliquid mid. ``N`` is chosen on the first ``train_frac`` of the
time span by the uplift in mean long PnL; it is then evaluated once on the
rest. Significance comes from a permutation test that shuffles the trend
label among test longs.

PASS requires all of: gated longs beat all longs, gated longs are positive,
the gated book (gated longs plus all shorts) beats the baseline book, and the
permutation p-value is below ``alpha`` (default 0.05 / 4, one test per
candidate ``N``).

Usage::

    python -m hlmicro trend-gate --symbol SOL --stop-pct=-1.0 --tp-pct 1.0
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, HL_VENUE, default_capture_dir
from ..parse import bucket_venue_means, discover
from ..schemas import BOOK_REST_PREFIX
from .common import rule
from .obi_signal import ExitSpec, SignalSpec, simulate

log = logging.getLogger(__name__)

MINUTE_MS = 60_000


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    symbol: str = "SOL"
    threshold: float | None = None      # default: 2.75 for BTC, 2.5 otherwise
    stop_pct: float = -1.0
    tp_pct: float = 1.0
    hold_min: int = 240
    cooldown_min: int = 60
    sma_windows: list[int] = field(default_factory=lambda: [60, 120, 240, 480])
    train_frac: float = 0.60
    min_train_longs: int = 8
    n_perm: int = 5000
    perm_seed: int = 13
    alpha: float = 0.0125
    fee_bps: float = DEFAULT_FEES.taker_round_trip_bps
    start: str | None = None
    end: str | None = None

    @property
    def effective_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return 2.75 if self.symbol.upper() == "BTC" else 2.5


def load_panel(files: list[Path], symbol: str) -> pd.DataFrame:
    """Continuous minute grid with the cross-venue mean imbalance (each venue
    forward-filled) and the forward-filled Hyperliquid mid.

    Columns: ``minute_ts``, ``ts_ms``, ``obi_ntl``, ``mid``.
    """
    per_venue = bucket_venue_means(files, symbol, MINUTE_MS)
    if per_venue.empty:
        return pd.DataFrame(columns=["minute_ts", "ts_ms", "obi_ntl", "mid"])
    full = pd.RangeIndex(int(per_venue["bucket"].min()), int(per_venue["bucket"].max()) + 1)
    obi = per_venue.pivot(index="bucket", columns="venue", values="obi_ntl").reindex(full).ffill()
    mid = per_venue.pivot(index="bucket", columns="venue", values="mid").reindex(full).ffill()
    if HL_VENUE not in mid:
        raise ValueError(f"no {HL_VENUE} quotes for {symbol}")
    minutes = full.to_numpy(dtype=np.int64)
    panel = pd.DataFrame({"minute_ts": minutes, "ts_ms": minutes * MINUTE_MS,
                          "obi_ntl": obi.mean(axis=1).to_numpy(), "mid": mid[HL_VENUE].to_numpy()})
    panel.attrs["venues"] = sorted(obi.columns)
    return panel


def baseline_trades(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    exits = ExitSpec(hold_min=cfg.hold_min, cooldown_min=cfg.cooldown_min, stop_pct=cfg.stop_pct,
                     target_pct=cfg.tp_pct if cfg.tp_pct > 0 else None)
    trades = simulate(panel, SignalSpec(cfg.effective_threshold, 60, 30), exits, cfg.fee_bps)
    return pd.DataFrame({"entry_ts_ms": [t["entry_ts_ms"] for t in trades],
                         "dir": [1 if t["side"] == "LONG" else -1 for t in trades],
                         "net": [t["pnl_pct_net"] for t in trades]})


def trend_up(panel: pd.DataFrame, entry_ts_ms: np.ndarray, window: int) -> np.ndarray:
    """Mid at or above its trailing ``window``-minute SMA at each entry minute;
    True where the SMA is undefined."""
    sma = panel["mid"].rolling(window, min_periods=window).mean().to_numpy()
    mid = panel["mid"].to_numpy()
    pos = np.searchsorted(panel["minute_ts"].to_numpy(), np.asarray(entry_ts_ms, dtype=np.int64) // MINUTE_MS)
    pos = np.clip(pos, 0, len(panel) - 1)
    return np.where(np.isnan(sma[pos]), True, mid[pos] >= sma[pos])


def cohort(x: pd.Series) -> dict:
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "win": np.nan, "t": np.nan}
    sd = x.std()
    return {"n": n, "mean": float(x.mean()), "win": 100 * float((x > 0).mean()),
            "t": float(x.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else np.nan}


def permutation_p(net: np.ndarray, keep: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Observed uplift of the kept subset's mean over the whole set, and the
    one-sided p-value under random relabelling."""
    base = net.mean()
    k = int(keep.sum())
    observed = net[keep].mean() - base
    if k == 0 or k == len(net):
        return float(observed), np.nan
    null = np.array([net[rng.choice(len(net), k, replace=False)].mean() - base for _ in range(n_perm)])
    return float(observed), float((np.sum(null >= observed) + 1) / (n_perm + 1))


@dataclass
class Result:
    config: Config
    minutes: int
    span_days: float
    venues: list[str]
    n_train: int
    n_test: int
    selected_window: int | None
    train_uplift: float
    cohorts: dict = field(default_factory=dict)
    uplift: float = np.nan
    p_value: float = np.nan
    retention_pct: float = np.nan
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def evaluate(panel: pd.DataFrame, trades: pd.DataFrame, cfg: Config) -> Result:
    first, last = int(panel["minute_ts"].iloc[0]), int(panel["minute_ts"].iloc[-1])
    cut = int(first + (last - first) * cfg.train_frac) * MINUTE_MS
    train, test = trades[trades["entry_ts_ms"] < cut], trades[trades["entry_ts_ms"] >= cut].copy()
    train_long = train[train["dir"] == 1]
    best, best_up = None, -np.inf
    for window in cfg.sma_windows:
        up = trend_up(panel, train_long["entry_ts_ms"].to_numpy(), window)
        if up.sum() < cfg.min_train_longs:
            continue
        uplift = train_long["net"][up].mean() - train_long["net"].mean()
        if uplift > best_up:
            best, best_up = window, uplift
    result = Result(cfg, len(panel), (last - first) / 1440, panel.attrs.get("venues", []),
                    len(train), len(test), best, best_up if best else np.nan)
    if best is None:
        return result

    test["trend_up"] = trend_up(panel, test["entry_ts_ms"].to_numpy(), best)
    longs, shorts = test[test["dir"] == 1], test[test["dir"] == -1]
    gated = longs[longs["trend_up"]]
    gated_book = pd.concat([gated["net"], shorts["net"]])
    result.cohorts = {
        "test longs, all": cohort(longs["net"]),
        f"test longs, uptrend (N={best})": cohort(gated["net"]),
        "test longs, suppressed": cohort(longs[~longs["trend_up"]]["net"]),
        "test shorts (untouched)": cohort(shorts["net"]),
        "test book, baseline": cohort(test["net"]),
        "test book, gated": cohort(gated_book),
    }
    if len(longs):
        rng = np.random.default_rng(cfg.perm_seed)
        result.uplift, result.p_value = permutation_p(longs["net"].to_numpy(), longs["trend_up"].to_numpy(),
                                                      cfg.n_perm, rng)
        result.retention_pct = 100 * float(longs["trend_up"].mean())
    g, b = cohort(gated["net"])["mean"], cohort(longs["net"])["mean"]
    result.checks = {
        "gated longs beat all longs": bool(g > b),
        "gated longs positive": bool(g > 0),
        "gated book beats baseline book": bool(cohort(gated_book)["mean"] > cohort(test["net"])["mean"]),
        f"permutation p < {cfg.alpha}": bool(result.p_value < cfg.alpha),
    }
    return result


def run(cfg: Config) -> Result:
    files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    if not files:
        raise FileNotFoundError(f"no {BOOK_REST_PREFIX} files in {cfg.capture_dir}")
    panel = load_panel(files, cfg.symbol)
    if panel.empty:
        raise ValueError(f"no rows for {cfg.symbol}")
    return evaluate(panel, baseline_trades(panel, cfg), cfg)


def render(result: Result) -> str:
    cfg = result.config
    tp = f"+{cfg.tp_pct:.1f}%" if cfg.tp_pct > 0 else "none"
    out = [f"Trend gate on longs - {cfg.symbol}",
           f"  grid: {result.minutes:,} minutes over {result.span_days:.1f} days   "
           f"venues: {', '.join(result.venues)}",
           f"  baseline: z60 and z30 beyond +/-{cfg.effective_threshold}  stop={cfg.stop_pct:+.1f}%  TP={tp}  "
           f"hold={cfg.hold_min}m  cooldown={cfg.cooldown_min}m  cost={cfg.fee_bps:.2f}bps round trip",
           f"  trades: train {result.n_train}  test {result.n_test}   SMA windows tried: {cfg.sma_windows}",
           ""]
    if result.selected_window is None:
        out.append(f"  fewer than {cfg.min_train_longs} training longs in an uptrend for every window: "
                   "underpowered, no evaluation")
        return "\n".join(out)
    out += [f"  selected N={result.selected_window} on training data "
            f"(long uplift {result.train_uplift:+.4f}pp)", "", rule(90),
            f"  {'cohort':<34}{'n':>5}{'mean pp':>10}{'win%':>8}{'t':>8}", rule(90)]
    for label, c in result.cohorts.items():
        if c["n"] == 0:
            out.append(f"  {label:<34}{0:>5}")
            continue
        t = "na" if np.isnan(c["t"]) else f"{c['t']:+.2f}"
        out.append(f"  {label:<34}{c['n']:>5}{c['mean']:>+10.4f}{c['win']:>7.1f}%{t:>8}")
    p = "na" if np.isnan(result.p_value) else f"{result.p_value:.4f}"
    out += ["", f"  long uplift {result.uplift:+.4f}pp  permutation p={p}  "
                f"longs kept {result.retention_pct:.0f}%", "", "  checks:"]
    out += [f"    {'ok  ' if ok else 'fail'}  {name}" for name, ok in result.checks.items()]
    out.append(f"\n  VERDICT: {'PASS' if result.passed else 'FAIL'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro trend-gate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--symbol", type=str.upper, default=d.symbol)
    parser.add_argument("--threshold", type=float, default=None,
                        help="z-score threshold (default 2.75 for BTC, 2.5 otherwise)")
    parser.add_argument("--stop-pct", type=float, default=d.stop_pct, help="write --stop-pct=-1.0")
    parser.add_argument("--tp-pct", type=float, default=d.tp_pct, help="take-profit; 0 disables")
    parser.add_argument("--hold-min", type=int, default=d.hold_min)
    parser.add_argument("--cooldown-min", type=int, default=d.cooldown_min)
    parser.add_argument("--sma-windows", type=csv_list(int), default=d.sma_windows)
    parser.add_argument("--train-frac", type=float, default=d.train_frac)
    parser.add_argument("--n-perm", type=int, default=d.n_perm)
    parser.add_argument("--alpha", type=float, default=d.alpha)
    parser.add_argument("--fee-bps", type=float, default=d.fee_bps, help="round-trip cost")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, symbol=args.symbol, threshold=args.threshold,
                 stop_pct=args.stop_pct, tp_pct=args.tp_pct, hold_min=args.hold_min,
                 cooldown_min=args.cooldown_min, sma_windows=args.sma_windows, train_frac=args.train_frac,
                 n_perm=args.n_perm, alpha=args.alpha, fee_bps=args.fee_bps, start=args.start, end=args.end)
    print(render(run(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
