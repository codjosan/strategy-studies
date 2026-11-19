"""Within-minute order-book features from websocket data, tested against trade outcomes.

Part A - agreement. Per-minute imbalance from websocket updates is compared
with the same quantity from 5-second polling: correlation, and agreement of
threshold crossings of their 60-minute z-scores.

Part B - features. For each trade in a per-trade CSV (``tp_trades_<symbol>.csv``
from ``tp-multifold``), four features of the entry minute are tested against
the trade's net return (Spearman rank correlation, Bonferroni-adjusted):

* ``obi_intra_std``    mean across venues of the within-minute std of imbalance
* ``obi_range``        mean across venues of the within-minute range
* ``obi_drift_signed`` within-minute drift (last - first), signed by trade side
* ``n_updates``        number of book updates in the minute

The per-trade features are written to ``hf_features_<symbol>.csv`` next to the
trades file.

Usage::

    python -m hlmicro hf-features --symbol BTC \\
        --trades-csv results/tp_multifold/tp_trades_btc.csv --filter tp_pct=0,stop_pct=-1.0
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..cli import add_common_args, add_date_args, setup_logging
from ..config import default_capture_dir
from ..parse import cross_venue_minutes, discover, discover_glob, intra_minute_book_stats
from ..schemas import BOOK_REST_PREFIX, BOOK_WS_PREFIX
from .common import rolling_zscore, rule

log = logging.getLogger(__name__)

FEATURES = {
    "obi_intra_std": "within-minute std of imbalance (mean over venues)",
    "obi_range": "within-minute range of imbalance (mean over venues)",
    "obi_drift_signed": "within-minute imbalance drift, signed by trade side",
    "n_updates": "book updates in the entry minute",
}
RETURN_COLUMNS = ("pnl_pct_net", "return_pct", "ret_net_pct", "pnl_pct", "ret_gross_pct")


@dataclass
class Config:
    capture_dir: Path = field(default_factory=default_capture_dir)
    ws_glob: str | None = None
    trades_csv: Path = Path("results/tp_multifold/tp_trades_btc.csv")
    symbol: str = "BTC"
    trade_filter: dict = field(default_factory=dict)
    threshold: float = 2.5
    alpha: float = 0.05
    min_trades: int = 20
    chunksize: int = 1_000_000
    start: str | None = None
    end: str | None = None


def parse_filter(text: str) -> dict:
    """``"tp_pct=1.5,stop_pct=-1.0"`` -> ``{"tp_pct": 1.5, "stop_pct": -1.0}``."""
    out: dict = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value.strip()
    return out


def load_trades(path: Path, symbol: str, trade_filter: dict) -> tuple[pd.DataFrame, str]:
    trades = pd.read_csv(path)
    for key, value in trade_filter.items():
        if key in trades:
            trades = trades[trades[key] == value]
    if "symbol" in trades:
        trades = trades[trades["symbol"].astype(str).str.upper() == symbol.upper()]
    return_col = next((c for c in RETURN_COLUMNS if c in trades), None)
    if return_col is None:
        raise KeyError(f"no return column in {path} (expected one of {RETURN_COLUMNS})")
    return trades.reset_index(drop=True), return_col


def agreement(ws_minutes: pd.DataFrame, rest_minutes: pd.DataFrame, threshold: float) -> dict | None:
    merged = ws_minutes.merge(rest_minutes[["ts_min", "avg_obi"]], on="ts_min", how="inner")
    if merged[["ws_obi_mean", "avg_obi"]].dropna().shape[0] <= 100:
        return None
    pairs = merged.dropna(subset=["ws_obi_mean", "avg_obi"])
    corr = float(np.corrcoef(pairs["ws_obi_mean"], pairs["avg_obi"])[0, 1])
    ordered = merged.sort_values("ts_min")
    z = pd.DataFrame({"ws": rolling_zscore(ordered["ws_obi_mean"], 60, 20),
                      "rest": rolling_zscore(ordered["avg_obi"], 60, 20)}).dropna()
    fire_ws, fire_rest = z["ws"].abs() >= threshold, z["rest"].abs() >= threshold
    both = int((fire_ws & fire_rest).sum())
    only_ws = int((fire_ws & ~fire_rest).sum())
    only_rest = int((~fire_ws & fire_rest).sum())
    either = both + only_ws + only_rest
    return {"corr": corr, "all_minute_agreement": float((fire_ws == fire_rest).mean()),
            "fire_agreement": both / either if either else float("nan"),
            "both": both, "only_ws": only_ws, "only_rest": only_rest}


def univariate(frame: pd.DataFrame, return_col: str, min_n: int) -> tuple[list[dict], int]:
    rows, tested = [], 0
    for feature in FEATURES:
        sub = frame[[feature, return_col]].dropna()
        if len(sub) < min_n:
            rows.append({"feature": feature, "n": len(sub), "skip": f"n<{min_n}"})
            continue
        rho, p = stats.spearmanr(sub[feature], sub[return_col])
        try:
            quartile = pd.qcut(sub[feature], 4, labels=False, duplicates="drop")
            means = sub.groupby(quartile)[return_col].mean()
            q1, q4 = means.iloc[0] * 100, means.iloc[-1] * 100
        except ValueError:
            q1 = q4 = np.nan
        rows.append({"feature": feature, "n": len(sub), "rho": float(rho), "p": float(p),
                     "q1": q1, "q4": q4, "q4_q1": q4 - q1, "skip": None})
        tested += 1
    for row in rows:
        row["p_bonf"] = None if row["skip"] else min(1.0, row["p"] * max(1, tested))
    rows.sort(key=lambda r: (r["p_bonf"] is None, r["p_bonf"] or 1.0))
    return rows, tested


@dataclass
class Result:
    config: Config
    n_trades: int
    n_joined: int
    ws_minutes: int
    agreement: dict | None
    features: list[dict]
    n_tested: int
    per_trade: pd.DataFrame
    return_col: str


def run(cfg: Config) -> Result:
    ws_files = (discover_glob(cfg.ws_glob, cfg.start, cfg.end) if cfg.ws_glob
                else discover(cfg.capture_dir, BOOK_WS_PREFIX, cfg.start, cfg.end))
    if not ws_files:
        raise FileNotFoundError("no obi_ws files")
    log.info("streaming %d websocket files", len(ws_files))
    ws = intra_minute_book_stats(ws_files, cfg.symbol, chunksize=cfg.chunksize)
    if ws.empty:
        raise ValueError(f"no websocket rows for {cfg.symbol}")

    rest_files = discover(cfg.capture_dir, BOOK_REST_PREFIX, cfg.start, cfg.end)
    agree = agreement(ws, cross_venue_minutes(rest_files, cfg.symbol), cfg.threshold) if rest_files else None

    trades, return_col = load_trades(cfg.trades_csv, cfg.symbol, cfg.trade_filter)
    if len(trades) < cfg.min_trades:
        raise ValueError(f"only {len(trades)} trades after filtering (need {cfg.min_trades})")
    trades["ts_min"] = (trades["entry_ts_ms"] // 60_000) * 60_000
    joined = trades.merge(ws, on="ts_min", how="left")
    side = np.where(joined["side"].astype(str).str.upper() == "LONG", 1.0, -1.0)
    joined["obi_drift_signed"] = joined["obi_drift"] * side
    if np.nanmax(np.abs(joined[return_col].astype(float))) > 0.2:   # percent -> fraction
        joined[return_col] = joined[return_col] / 100.0
    features, tested = univariate(joined, return_col, cfg.min_trades)
    keep = ["ts_min", "side", return_col, *FEATURES]
    return Result(config=cfg, n_trades=len(trades), n_joined=int(joined["n_updates"].notna().sum()),
                  ws_minutes=len(ws), agreement=agree, features=features, n_tested=tested,
                  per_trade=joined[keep], return_col=return_col)


def render(result: Result) -> str:
    cfg = result.config
    label = ",".join(f"{k}={v}" for k, v in cfg.trade_filter.items()) or "no filter"
    out = [rule(88), f"Within-minute book features - {cfg.symbol} ({label})", rule(88),
           f"  trades: {result.n_trades}   with websocket features: {result.n_joined}   "
           f"websocket minutes: {result.ws_minutes:,}", "",
           "-" * 88, "  A. Agreement: websocket vs 5s per-minute imbalance", "-" * 88]
    a = result.agreement
    if a is None:
        out.append("  not enough overlapping minutes")
    else:
        out += [f"  correlation:                        {a['corr']:.4f}",
                f"  agreement, all minutes @ {cfg.threshold:.2f}:     {a['all_minute_agreement'] * 100:.1f}%",
                f"  agreement, firing minutes @ {cfg.threshold:.2f}:  {a['fire_agreement'] * 100:.1f}%  "
                f"(both {a['both']}, websocket only {a['only_ws']}, 5s only {a['only_rest']})"]
    alpha_adj = cfg.alpha / max(1, result.n_tested)
    out += ["", "-" * 88, "  B. Feature tests (Spearman vs net return, Bonferroni)", "-" * 88,
            f"  features tested: {result.n_tested}   significance: p < {alpha_adj:.4f}", "",
            f"  {'feature':<18}{'n':>5}{'rho':>9}{'p':>10}{'p_bonf':>10}{'Q1%':>10}{'Q4%':>10}{'Q4-Q1':>9}"]
    for r in result.features:
        if r["skip"]:
            out.append(f"  {r['feature']:<18}{r['n']:>5}   skipped ({r['skip']})")
            continue
        q = [f"{v:.4f}" if not pd.isna(v) else "nan" for v in (r["q1"], r["q4"], r["q4_q1"])]
        flag = "  *" if r["p_bonf"] < cfg.alpha else ""
        out.append(f"  {r['feature']:<18}{r['n']:>5}{r['rho']:>9.3f}{r['p']:>10.4f}{r['p_bonf']:>10.4f}"
                   f"{q[0]:>10}{q[1]:>10}{q[2]:>9}{flag}")
    passed = sum(1 for r in result.features if r["p_bonf"] is not None and r["p_bonf"] < cfg.alpha)
    out += ["", f"  features significant after correction: {passed} of {result.n_tested}", "",
            "  definitions:"] + [f"    {k:<18} {v}" for k, v in FEATURES.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro hf-features", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--capture-dir", type=Path, default=d.capture_dir)
    parser.add_argument("--ws-glob", default=None, help="websocket files (default: obi_ws in capture dir)")
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--symbol", type=str.upper, required=True)
    parser.add_argument("--filter", default="", help="trade filter, e.g. tp_pct=0,stop_pct=-1.0")
    parser.add_argument("--threshold", type=float, default=d.threshold,
                        help="z threshold for the agreement diagnostic")
    parser.add_argument("--chunksize", type=int, default=d.chunksize)
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config(capture_dir=args.capture_dir, ws_glob=args.ws_glob, trades_csv=args.trades_csv,
                 symbol=args.symbol, trade_filter=parse_filter(args.filter), threshold=args.threshold,
                 chunksize=args.chunksize, start=args.start, end=args.end)
    result = run(cfg)
    print(render(result))
    out_path = cfg.trades_csv.parent / f"hf_features_{cfg.symbol.lower()}.csv"
    result.per_trade.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
