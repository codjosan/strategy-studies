"""Run every study the capture data supports and rank the strategies they test.

1. **Feasibility.** Each study's data needs are checked against the capture
   directory. Studies without their data are skipped, with the reason.
2. **Runs.** Feasible studies run with their normal code. The per-coin
   studies (``tp-multifold``, ``multi-gate``, ``funding-settlement``) run once
   per symbol, and ``cex-hl-lag`` runs once for all symbols. Each study's full
   report is saved.
3. **Leaderboard.** Every configuration a study tests (e.g. one stop/TP cell
   of ``tp-multifold``, one rule and hold of ``multi-gate``) becomes one
   strategy. Its per-trade net returns are scored the same way for every
   study.

Common cost model: mid-price fills and the taker round trip of
:class:`hlmicro.config.FeeModel` (2 x 4.32 = 8.64 bps), with no spread or
slippage. The exception is the passive-fill strategies of ``multi-gate``,
which use their own simulated maker/taker fills and fees.

Scoring per strategy:

* ``n``, trades per day, mean and median net return (bps), win rate
* ``t``: mean / standard error. Strategies are ranked by ``t``, because a
  high mean from few or very noisy trades is not evidence.
* ``folds+``: the trades are split into ``--folds`` equal time periods; the
  number of periods with a positive mean, out of the periods that had trades
* verdict:

  ============  ============================================================
  significant   mean > 0, one-sided p < 0.05 **after Bonferroni correction**
                for the number of strategies ranked, and >= 75% of periods
                positive
  promising     mean > 0, t >= 2 and >= 75% of periods positive (can be luck
                when many strategies are tested)
  weak          mean > 0 but not promising
  negative      mean <= 0 after costs
  ============  ============================================================

Everything is in-sample on the capture period: the leaderboard says which
ideas deserve an out-of-sample test, not which ones to trade.

Outputs in ``--outdir`` (default ``results/run_all/<timestamp>``):
``leaderboard.md`` (feasibility, runs, top strategies), ``strategies.csv``
(every strategy with all metrics) and ``reports/`` (each study's own report).

Variants: ``--config variants.json`` maps a study to a dict, or a list of
dicts, of its ``Config`` fields, e.g.::

    {"tp-multifold": [{}, {"hold_min": 30, "cooldown_min": 10}],
     "multi-gate": {"tick_s": 10}}

Usage::

    python -m hlmicro run-all
    python -m hlmicro run-all --symbols BTC,ETH --top 20 --config variants.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..cli import add_common_args, add_date_args, csv_list, setup_logging
from ..config import DEFAULT_FEES, default_capture_dir
from ..parse import discover, file_date, iter_chunks
from ..parse.funding import HL_INFO_URL
from ..schemas import (BOOK_REST_PREFIX, BOOK_WS_PREFIX, CEX_TRADE_PREFIXES, FUNDING_FILE_TEMPLATE,
                       HL_LIQUIDATIONS_PREFIX, HL_TRADES_PREFIX)
from . import cex_hl_lag, funding_settlement, multi_gate, tp_multifold
from .common import rule

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
TAKER_RT_BPS = DEFAULT_FEES.taker_round_trip_bps
STUDIES = ("tp-multifold", "cex-hl-lag", "multi-gate", "funding-settlement", "candidates",
           "source-compare", "hf-features", "lead-lag", "liq-cascade")


# --------------------------------------------------------------------------- strategies and scoring

@dataclass
class Strategy:
    study: str
    symbol: str
    name: str
    params: str
    cost: str
    ts_ms: np.ndarray           # entry time of each trade
    net_bps: np.ndarray         # net return of each trade, bps
    cost_bps: float = float("nan")   # fixed round-trip cost per trade; NaN when it varies (simulated fills)


def score(s: Strategy, n_folds: int) -> dict:
    x = np.asarray(s.net_bps, dtype=float)
    t_ms = np.asarray(s.ts_ms, dtype=np.int64)
    ok = np.isfinite(x)
    x, t_ms = x[ok], t_ms[ok]
    n = len(x)
    row = {"study": s.study, "symbol": s.symbol, "strategy": s.name, "params": s.params, "cost": s.cost, "n": n}
    if n == 0:
        return row
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    t = mean / std * math.sqrt(n) if std > 0 else 0.0
    span_days = max((t_ms.max() - t_ms.min()) / DAY_MS, 1.0)
    edges = np.linspace(t_ms.min(), t_ms.max() + 1, n_folds + 1)
    fold = np.clip(np.searchsorted(edges, t_ms, side="right") - 1, 0, n_folds - 1)
    fold_means = [x[fold == k].mean() for k in range(n_folds) if (fold == k).any()]
    row.update({
        "first_trade": pd.Timestamp(int(t_ms.min()), unit="ms", tz="UTC").strftime("%Y-%m-%d"),
        "last_trade": pd.Timestamp(int(t_ms.max()), unit="ms", tz="UTC").strftime("%Y-%m-%d"),
        "trades_per_day": n / span_days,
        "mean_net_bps": mean, "mean_gross_bps": mean + s.cost_bps, "median_net_bps": float(np.median(x)),
        "std_bps": std,
        "win_rate": float((x > 0).mean()), "total_net_bps": float(x.sum()), "t": t,
        "p_one_sided": float(sps.t.sf(t, n - 1)) if n > 1 and std > 0 else 1.0,
        "folds_pos": sum(m > 0 for m in fold_means), "folds_with_trades": len(fold_means),
        "fold_means_bps": " ".join(f"{m:+.1f}" for m in fold_means),
    })
    return row


def leaderboard(strategies: list[Strategy], n_folds: int, min_trades: int) -> pd.DataFrame:
    rows = [score(s, n_folds) for s in strategies]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["eligible"] = table["n"] >= min_trades
    n_ranked = int(table["eligible"].sum())
    table["p_bonferroni"] = np.minimum(1.0, table.get("p_one_sided", 1.0) * max(1, n_ranked))
    stable = table["folds_pos"] >= np.ceil(0.75 * table["folds_with_trades"])
    positive = table["mean_net_bps"] > 0
    table["verdict"] = np.select(
        [~table["eligible"], ~positive,
         positive & (table["p_bonferroni"] < 0.05) & stable,
         positive & (table["t"] >= 2) & stable],
        [f"n<{min_trades}", "negative", "significant", "promising"], default="weak")
    table = table.sort_values(["eligible", "t"], ascending=[False, False]).reset_index(drop=True)
    table.insert(0, "rank", np.where(table["eligible"], np.arange(1, len(table) + 1), 0))
    return table


# --------------------------------------------------------------------------- feasibility

def _book_coverage(files: list[Path]) -> tuple[list[str], set[str]]:
    """Coins and venues in the REST book files (read from two sample days)."""
    coins, venues = set(), set()
    for path in {files[0], files[-1]}:
        for chunk in iter_chunks(path, ["coin", "venue"], chunksize=500_000):
            coins |= set(chunk["coin"].astype(str).str.upper())
            venues |= set(chunk["venue"].astype(str).str.lower())
    return sorted(coins), venues


def _api_reachable(timeout: float = 4.0) -> bool:
    try:
        req = urllib.request.Request(HL_INFO_URL, data=b'{"type":"meta"}',
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).close()
        return True
    except Exception:
        return False


@dataclass
class Plan:
    symbols: list[str]
    feasible: dict = field(default_factory=dict)      # study -> (bool, note)
    tape_range: tuple | None = None
    book_range: tuple | None = None
    funding_symbols: list[str] = field(default_factory=list)


def plan(capture_dir: Path, symbols: list[str] | None, start, end) -> Plan:
    books = discover(capture_dir, BOOK_REST_PREFIX, start, end)
    ws = discover(capture_dir, BOOK_WS_PREFIX, start, end)
    tapes = discover(capture_dir, HL_TRADES_PREFIX, start, end)
    cex = [p for prefix in CEX_TRADE_PREFIXES.values() for p in discover(capture_dir, prefix, start, end)]
    liqs = discover(capture_dir, HL_LIQUIDATIONS_PREFIX, start, end)
    coins, venues = _book_coverage(books) if books else ([], set())
    p = Plan(symbols=[s.upper() for s in symbols] if symbols else coins)
    if books:
        p.book_range = (file_date(books[0]), file_date(books[-1]))
    f = p.feasible
    span = lambda fs: f"{len(fs)} days {file_date(fs[0])}..{file_date(fs[-1])}" if fs else ""
    f["tp-multifold"] = (bool(books), f"obi_5s: {span(books)}" if books else "needs obi_5s files")
    lead = "binance"
    f["cex-hl-lag"] = ((bool(books) and {lead, "hyperliquid"} <= venues),
                       f"{lead} -> hyperliquid mids from obi_5s" if books else "needs obi_5s files")
    if books and tapes:
        p.tape_range = (file_date(tapes[0]), file_date(tapes[-1]))
        f["multi-gate"] = (True, f"obi_5s + hl_trades, restricted to the tape period {span(tapes)}")
    else:
        f["multi-gate"] = (False, "needs obi_5s (or obi_ws) and hl_trades files")
    f["candidates"] = (bool(books), "7 drafted strategies (docs/NEW_STRATEGIES.md); tape-based ones on the "
                       "hl_trades period" if books else "needs obi_5s files")
    p.funding_symbols = [s for s in p.symbols if (capture_dir / FUNDING_FILE_TEMPLATE.format(coin=s)).exists()]
    if books and not p.funding_symbols and _api_reachable():
        p.funding_symbols = list(p.symbols)
    f["funding-settlement"] = (bool(books and p.funding_symbols),
                               f"funding for {', '.join(p.funding_symbols)}" if books and p.funding_symbols else
                               "needs hl_funding_<COIN>.csv in the capture dir, or access to the Hyperliquid API")
    f["source-compare"] = (False, "websocket-vs-REST comparison; needs obi_ws files") if not ws else \
        (False, "compares data sources rather than testing a strategy; run it directly")
    f["hf-features"] = (False, "needs obi_ws files" if not ws else "feature diagnostic, not a strategy; run it directly")
    f["lead-lag"] = (False, "needs obi_ws and okx_trades/bybit_trades files" if not (ws and cex)
                     else "event study without a tradable exit rule; run it directly")
    f["liq-cascade"] = (False, "needs hl_liquidations files" if not liqs
                        else "separate script in liquid_study/, not yet a hlmicro command")
    return p


# --------------------------------------------------------------------------- study adapters

def _variants(config: dict, study: str) -> list[dict]:
    v = config.get(study, [{}])
    return v if isinstance(v, list) else [v]


def _label(overrides: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides.items()) or "default"


def run_tp(capture_dir, symbol, overrides, start, end):
    cfg = tp_multifold.Config(capture_dir=capture_dir, symbol=symbol, start=start, end=end, **overrides)
    result = tp_multifold.run(cfg)
    out = []
    for (stop, tp), (_, trades) in result.pooled.items():
        if not trades:
            continue
        out.append(Strategy(
            "tp-multifold", symbol, "OBI dual z-score",
            f"z>{cfg.threshold} hold {cfg.hold_min}m cooldown {cfg.cooldown_min}m "
            f"stop {'%g%%' % stop if stop < 0 else 'none'} tp {'%g%%' % tp if tp > 0 else 'none'}",
            f"taker {cfg.fee_bps:.2f} bps RT, mid fills",
            np.array([t["entry_ts_ms"] for t in trades]), np.array([t["pnl_pct_net"] * 100 for t in trades]),
            cfg.fee_bps))
    return tp_multifold.render(result), out


def run_cex(capture_dir, symbols, overrides, start, end):
    cfg = cex_hl_lag.Config(capture_dir=capture_dir, symbols=symbols, start=start, end=end, **overrides)
    result = cex_hl_lag.run(cfg)
    cost = cfg.fees.taker_round_trip_bps
    out = []
    for sw in result.sweeps:
        for h, g in sw.response.groupby("horizon_s"):
            out.append(Strategy(
                "cex-hl-lag", sw.symbol, f"follow {cfg.leader} move on HL",
                f"move>={sw.threshold_bps:g}bps/{sw.window_s}s hold {int(h)}s cooldown {cfg.cooldown_s}s",
                f"taker {cost:.2f} bps RT, mid fills",
                g["ts_ms"].to_numpy(), g["lag_capture_bps"].to_numpy() - cost, cost))
    return cex_hl_lag.render(result), out


def run_gate(capture_dir, symbol, overrides, start, end):
    cfg = multi_gate.Config(capture_dir=capture_dir, symbol=symbol, start=start, end=end, **overrides)
    result = multi_gate.run(cfg)
    tick_ms = cfg.tick_s * 1000
    out = []
    for key, frame in result.trades.items():
        ts = frame["entry_tick"].to_numpy(np.int64) * tick_ms
        if key[0] == "fill":
            _, rule_name, hold, k = key
            out.append(Strategy("multi-gate", symbol, rule_name,
                                f"passive limit {k:g}bps inside mid, hold {hold}s, tick {cfg.tick_s}s",
                                f"simulated fills: maker {cfg.fees.maker_bps:.2f} in, maker/taker out",
                                ts, frame["net_ret"].to_numpy() * 1e4))
        else:
            kind, rule_name, hold = key
            exit_label = "exit at VWAP (failsafe %ds)" % cfg.vwap_failsafe_s if kind == "vwap" else f"hold {hold}s"
            out.append(Strategy("multi-gate", symbol, rule_name, f"{exit_label}, tick {cfg.tick_s}s",
                                f"taker {cfg.fees.taker_round_trip_bps:.2f} bps RT, mid fills",
                                ts, frame["gross_ret"].to_numpy() * 1e4 - cfg.fees.taker_round_trip_bps,
                                cfg.fees.taker_round_trip_bps))
    return multi_gate.render(result), out


def run_candidates(capture_dir, symbols, overrides, start, end, books_range):
    from . import candidates                   # imported here: candidates builds on this module
    s0, e0 = books_range
    only = overrides.get("only")
    strategies, log_rows = candidates.run(capture_dir, symbols, only, start or s0, end or e0)
    report = "\n".join(f"{name:<18} {target} targets  {n} strategies  {secs:.1f}s"
                       for name, target, secs, n in log_rows if name != "grid")
    return report, strategies


def run_funding(capture_dir, symbol, overrides, start, end):
    cfg = funding_settlement.Config(capture_dir=capture_dir, symbol=symbol, start=start, end=end,
                                    fees_bps_per_side=[DEFAULT_FEES.taker_bps], **overrides)
    result = funding_settlement.run(cfg)
    out = []
    for b in result.backtests:
        out.append(Strategy("funding-settlement", symbol, "fade extreme funding around settlement",
                            f"top/bottom {b['pct']:g}% funding, T-5m..T+5m, incl. funding payment",
                            f"taker {2 * b['fee_bps']:.2f} bps RT, mid fills",
                            np.array([t["hour_ms"] for t in b["trades"]]),
                            np.array([t["net"] * 100 for t in b["trades"]]), 2 * b["fee_bps"]))
    return funding_settlement.render(result), out


# --------------------------------------------------------------------------- orchestration

@dataclass
class RunLog:
    study: str
    symbol: str
    variant: str
    seconds: float
    strategies: int
    error: str = ""
    report: str = ""


def run_all(capture_dir: Path, outdir: Path, symbols=None, studies=None, config=None, start=None, end=None,
            n_folds: int = 4, min_trades: int = 30):
    config = config or {}
    p = plan(capture_dir, symbols, start, end)
    wanted = [s for s in STUDIES if (studies is None or s in studies)]
    jobs = []
    for study in wanted:
        if not p.feasible[study][0]:
            continue
        for ov in _variants(config, study):
            if study == "tp-multifold":
                jobs += [(study, sym, ov, run_tp, (capture_dir, sym, ov, start, end)) for sym in p.symbols]
            elif study == "cex-hl-lag":
                jobs.append((study, ",".join(p.symbols), ov, run_cex, (capture_dir, p.symbols, ov, start, end)))
            elif study == "multi-gate":
                s0, e0 = p.tape_range
                s0, e0 = max(s0, start or s0), min(e0, end or e0)
                jobs += [(study, sym, ov, run_gate, (capture_dir, sym, ov, s0, e0)) for sym in p.symbols]
            elif study == "candidates":
                jobs.append((study, ",".join(p.symbols), ov, run_candidates,
                             (capture_dir, p.symbols, ov, start, end, p.book_range)))
            elif study == "funding-settlement":
                jobs += [(study, sym, ov, run_funding, (capture_dir, sym, ov, start, end)) for sym in p.funding_symbols]
    reports_dir = outdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    strategies, runs = [], []
    for i, (study, sym, ov, fn, args) in enumerate(jobs, 1):
        t0 = time.time()
        label = _label(ov)
        try:
            report, found = fn(*args)
            error = ""
        except Exception as exc:                         # one failed run does not stop the others
            report, found, error = "", [], f"{type(exc).__name__}: {exc}"
            log.warning("%s %s failed: %s", study, sym, error)
        for s in found:
            if label != "default":
                s.params += f" [{label}]"
        strategies += found
        name = f"{study}_{sym.replace(',', '-')}" + ("" if label == "default" else "_" + label.replace("=", "-").replace(",", "_"))
        path = reports_dir / f"{name}.txt"
        if report:
            path.write_text(report + "\n")
        runs.append(RunLog(study, sym, label, time.time() - t0, len(found), error,
                           str(path.relative_to(outdir)) if report else ""))
        print(f"  [{i}/{len(jobs)}] {study:<18} {sym:<26} {label:<22} {time.time() - t0:6.1f}s  "
              f"{len(found):>3} strategies{'  FAILED: ' + error if error else ''}", flush=True)
    table = leaderboard(strategies, n_folds, min_trades)
    return p, runs, table


def _fmt_table(table: pd.DataFrame, top: int) -> str:
    view = table[table["eligible"]].head(top)
    if view.empty:
        return "  (no strategy has enough trades)"
    lines = [f"  {'#':>3} {'study':<12} {'sym':<5} {'strategy / parameters':<72} {'n':>5} {'/day':>5} "
             f"{'mean':>7} {'gross':>6} {'win%':>5} {'t':>6} {'folds+':>6}  verdict"]
    for _, r in view.iterrows():
        desc = f"{r['strategy']}: {r['params']}"
        desc = desc if len(desc) <= 72 else desc[:69] + "..."
        lines.append(f"  {int(r['rank']):>3} {r['study'][:12]:<12} {r['symbol']:<5} {desc:<72} {int(r['n']):>5} "
                     f"{r['trades_per_day']:>5.1f} {r['mean_net_bps']:>+7.2f} "
                     f"{'' if pd.isna(r['mean_gross_bps']) else format(r['mean_gross_bps'], '+.1f'):>6} "
                     f"{100 * r['win_rate']:>5.1f} "
                     f"{r['t']:>+6.2f} {int(r['folds_pos']):>3}/{int(r['folds_with_trades'])}  {r['verdict']}")
    return "\n".join(lines)


def render(p: Plan, runs: list[RunLog], table: pd.DataFrame, top: int, capture_dir: Path, min_trades: int) -> str:
    out = [rule(120), f"All feasible studies - {capture_dir}", rule(120), "", "Feasibility:"]
    for study in STUDIES:
        ok, note = p.feasible[study]
        out.append(f"  {'RUN ' if ok else 'skip'}  {study:<20} {note}")
    total = sum(r.seconds for r in runs)
    failed = [r for r in runs if r.error]
    out += ["", f"Runs: {len(runs)} ({total:.0f}s), {len(failed)} failed"]
    out += [f"  FAILED {r.study} {r.symbol} {r.variant}: {r.error}" for r in failed]
    if table.empty:
        out.append("\nNo strategies were produced.")
        return "\n".join(out)
    elig = table[table["eligible"]]
    counts = elig["verdict"].value_counts()
    out += ["", f"Strategies: {len(table)} tested, {len(elig)} with n >= {min_trades} ranked. "
                f"Significant: {counts.get('significant', 0)}, promising: {counts.get('promising', 0)}, "
                f"weak: {counts.get('weak', 0)}, negative after costs: {counts.get('negative', 0)}",
            f"Cost: mid fills + taker {TAKER_RT_BPS:.2f} bps round trip (multi-gate passive fills: simulated). "
            f"Ranked by t = mean / standard error.", "",
            f"Before costs: {int((elig['mean_gross_bps'] > 0).sum())} of {int(elig['mean_gross_bps'].notna().sum())} "
            f"fixed-cost strategies have a positive mean; "
            f"{int((elig['mean_gross_bps'] > TAKER_RT_BPS).sum())} clear the {TAKER_RT_BPS:.2f} bps taker round trip.",
            "",
            rule(120, "-"), f"TOP {min(top, len(elig))} BY t-STATISTIC (net bps per trade; gross = before costs)",
            rule(120, "-"),
            _fmt_table(table, top)]
    best = elig.sort_values("t", ascending=False).groupby("study").head(1)
    out += ["", "Best per study:"]
    out += [f"  {r['study']:<18} {r['symbol']:<5} {r['strategy']}: {r['params']}  "
            f"mean {r['mean_net_bps']:+.2f} bps, t {r['t']:+.2f}, n {int(r['n'])}, {r['verdict']}"
            for _, r in best.iterrows()]
    n_sig = counts.get("significant", 0)
    out += ["", "Reading this:",
            f"  * {len(elig)} strategies were ranked; at that count a handful reach t >= 2 by chance alone.",
            "    'significant' survives a Bonferroni correction for all of them; 'promising' does not.",
            "  * All results are in-sample on the capture period. Re-test anything promising on new data",
            "    (e.g. --start after today) before trusting it.",
            "  * " + ("No strategy is significant after correction and costs." if not n_sig else
                      f"{n_sig} strateg{'y is' if n_sig == 1 else 'ies are'} significant after correction; "
                      "check the fold means for one-off periods.")]
    return "\n".join(out)


def _markdown(text: str, table: pd.DataFrame, runs: list[RunLog], top: int) -> str:
    md = ["# Strategy leaderboard", "", "```", text, "```", "", "## Top strategies", ""]
    cols = ["rank", "study", "symbol", "strategy", "params", "n", "trades_per_day", "mean_net_bps", "mean_gross_bps",
            "median_net_bps", "win_rate", "t", "p_bonferroni", "folds_pos", "folds_with_trades",
            "fold_means_bps", "verdict", "cost"]
    view = table[table["eligible"]].head(top) if not table.empty else table
    if not view.empty:
        md.append(view[cols].to_markdown(index=False, floatfmt=".3g") if _has_tabulate() else
                  "```\n" + view[cols].to_string(index=False) + "\n```")
    md += ["", "## Study reports", ""] + [f"- `{r.report}` ({r.seconds:.0f}s)" for r in runs if r.report]
    return "\n".join(md) + "\n"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro run-all", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, default=default_capture_dir())
    parser.add_argument("--symbols", type=csv_list(str.upper), default=None,
                        help="default: every coin in the obi_5s files")
    parser.add_argument("--studies", type=csv_list(str.lower), default=None, help=f"subset of {', '.join(STUDIES)}")
    parser.add_argument("--config", type=Path, default=None, help="JSON with per-study Config overrides/variants")
    parser.add_argument("--top", type=int, default=15, help="strategies to list")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--folds", type=int, default=4, help="time periods for the stability check")
    parser.add_argument("--outdir", type=Path, default=None, help="default results/run_all/<timestamp>")
    add_date_args(parser)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.verbose:                       # keep the progress list readable; -v shows study logs
        logging.getLogger("hlmicro").setLevel(logging.WARNING)
    if args.studies:
        unknown = sorted(set(args.studies) - set(STUDIES))
        if unknown:
            parser.error(f"unknown studies {unknown}; choose from {', '.join(STUDIES)}")
    if not args.capture_dir.is_dir():
        parser.error(f"capture directory not found: {args.capture_dir}")
    config = json.loads(args.config.read_text()) if args.config else {}
    outdir = args.outdir or Path("results/run_all") / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"capture dir: {args.capture_dir}   output: {outdir}", flush=True)
    p, runs, table = run_all(args.capture_dir, outdir, args.symbols, args.studies, config,
                             args.start, args.end, args.folds, args.min_trades)
    text = render(p, runs, table, args.top, args.capture_dir, args.min_trades)
    print("\n" + text)
    if not table.empty:
        table.to_csv(outdir / "strategies.csv", index=False)
    (outdir / "leaderboard.md").write_text(_markdown(text, table, runs, args.top))
    print(f"\nwrote {outdir}/leaderboard.md, {outdir}/strategies.csv and {outdir}/reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
