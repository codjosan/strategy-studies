"""Deterministic synthetic capture data in the exact on-disk schemas.

Generates order-book snapshots (REST and websocket), Hyperliquid and CEX trade
tapes, liquidation prints and hourly funding, so every command can be run end
to end without real capture files. The test suite uses the same generator.

The price process has calm/active volatility regimes and occasional jumps;
Hyperliquid quotes lag the CEX venues by ``hl_lag_s`` seconds, and order-book
imbalance follows a persistent latent factor with occasional shocks. None of
this is calibrated to real markets: it exists to exercise code paths, not to
produce meaningful trading results.

Usage::

    python -m hlmicro synth --out sample_data --days 4
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .cli import add_common_args, csv_list, setup_logging
from .schemas import (
    BOOK_REST_COLUMNS, BOOK_REST_PREFIX, BOOK_WS_COLUMNS, BOOK_WS_PREFIX,
    CEX_TRADE_PREFIXES, CEX_TRADES_COLUMNS, FUNDING_FILE_TEMPLATE,
    HL_LIQUIDATIONS_COLUMNS, HL_LIQUIDATIONS_PREFIX, HL_TRADES_COLUMNS, HL_TRADES_PREFIX,
)

log = logging.getLogger(__name__)

DAY_MS = 86_400_000

# Per-coin market parameters: price tick on each venue, typical book depth
# (USD, top 5 levels) and typical trade notional (USD).
_COIN_PARAMS = {
    "BTC": {"tick": {"hyperliquid": 1.0, "binance": 0.1, "okx": 0.1, "bybit": 0.1},
            "depth_usd": 1_500_000.0, "hl_trade_usd": 3_000.0, "sz_decimals": 5},
    "SOL": {"tick": {"hyperliquid": 0.001, "binance": 0.01, "okx": 0.01, "bybit": 0.01},
            "depth_usd": 300_000.0, "hl_trade_usd": 1_000.0, "sz_decimals": 2},
}
_SPREAD_BPS = {"hyperliquid": 0.3, "binance": 0.1, "okx": 0.2, "bybit": 0.1}


@dataclass
class SyntheticConfig:
    start: str = "2026-01-05"
    days: int = 4
    coins: dict[str, float] = field(default_factory=lambda: {"BTC": 60_000.0, "SOL": 150.0})
    ws_coins: tuple[str, ...] = ("BTC",)
    cex_tape_coins: tuple[str, ...] = ("BTC",)
    book_venues: tuple[str, ...] = ("hyperliquid", "binance", "okx", "bybit")
    ws_venues: tuple[str, ...] = ("hyperliquid", "okx")
    hl_lag_s: int = 10
    poll_interval_s: int = 5
    ws_mean_interval_s: float = 2.0
    trade_mean_interval_s: float = 2.0
    liquidation_bursts_per_day: int = 6
    gzip_closed_days: bool = True
    seed: int = 7


class _Market:
    """One coin's latent price and imbalance paths at 1-second resolution."""

    def __init__(self, rng: np.random.Generator, n_seconds: int, price0: float, lag_s: int):
        self.n = n_seconds
        self.lag_s = lag_s
        switches = rng.random(n_seconds) < 1 / 1200          # ~20-minute regimes
        active = (np.cumsum(switches) % 2).astype(bool)
        vol = np.where(active, 1.5e-4, 0.15e-4)
        shocks = np.where(rng.random(n_seconds) < 1 / 4000,
                          rng.choice([-2.5, 2.5], n_seconds), 0.0)
        self.latent = lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.03, n_seconds) + shocks)
        returns = rng.standard_normal(n_seconds) * vol + 0.05e-4 * np.tanh(self.latent)
        jumps = rng.random(n_seconds) < 1 / 3600
        returns[jumps] += rng.normal(0, 25e-4, int(jumps.sum()))
        self.price = price0 * np.exp(np.cumsum(returns))

    def _sec(self, t_ms: np.ndarray, lag_s: int = 0) -> np.ndarray:
        return np.clip(t_ms // 1000 - lag_s, 0, self.n - 1).astype(np.int64)

    def price_at(self, t_ms: np.ndarray, venue: str) -> np.ndarray:
        lag = self.lag_s if venue == "hyperliquid" else 0
        return self.price[self._sec(t_ms, lag)]

    def imbalance_at(self, t_ms: np.ndarray, noise: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(self.latent[self._sec(t_ms)] + noise), -0.999, 0.999)


def _quantize(x: np.ndarray, tick: float) -> np.ndarray:
    return np.round(x / tick) * tick


def _book_frame(rng, market: _Market, coin: str, venue: str, ts: np.ndarray,
                obi_noise_sd: float) -> pd.DataFrame:
    params = _COIN_PARAMS.get(coin, _COIN_PARAMS["SOL"])
    mid = _quantize(market.price_at(ts, venue), params["tick"].get(venue, 0.01))
    spread_bps = _SPREAD_BPS.get(venue, 0.2) * (1 + 0.5 * rng.random(len(ts)))
    obi = market.imbalance_at(ts, rng.normal(0, obi_noise_sd, len(ts)))
    depth = params["depth_usd"] * rng.lognormal(0, 0.3, len(ts))
    bid_ntl = depth * (1 + obi) / 2
    ask_ntl = depth * (1 - obi) / 2
    bid_qty = bid_ntl / mid
    ask_qty = ask_ntl / mid
    return pd.DataFrame({
        "ts_ms": ts,
        "coin": coin,
        "venue": venue,
        "mid": np.round(mid, 6),
        "bid1_px": np.round(mid * (1 - spread_bps / 2e4), 6),
        "ask1_px": np.round(mid * (1 + spread_bps / 2e4), 6),
        "bid_qty_5": np.round(bid_qty, 4),
        "ask_qty_5": np.round(ask_qty, 4),
        "bid_ntl_5": np.round(bid_ntl, 2),
        "ask_ntl_5": np.round(ask_ntl, 2),
        "obi_qty": np.round((bid_qty - ask_qty) / (bid_qty + ask_qty), 4),
        "obi_ntl": np.round(obi, 4),
        "spread_bps": np.round(spread_bps, 2),
    })


def _event_times(rng, n_seconds: int, mean_interval_s: float) -> np.ndarray:
    """Strictly increasing Poisson event times in ms over the window."""
    n = int(n_seconds / mean_interval_s * 1.2) + 10
    gaps = np.maximum(1, np.ceil(rng.exponential(mean_interval_s * 1000, n))).astype(np.int64)
    t = np.cumsum(gaps)
    return t[t < n_seconds * 1000]


def _addresses(rng, n: int) -> np.ndarray:
    return np.array(["0x" + "".join(rng.choice(list("0123456789abcdef"), 40)) for _ in range(n)])


def _write_daily(frame: pd.DataFrame, out: Path, prefix: str, columns, start_ms: int,
                 days: int, gzip_closed: bool, day_key: str = "ts_ms") -> None:
    day_index = (frame[day_key].to_numpy() - start_ms) // DAY_MS
    for d in range(days):
        part = frame.loc[day_index == d, list(columns)]
        date = pd.Timestamp(start_ms + d * DAY_MS, unit="ms", tz="UTC").strftime("%Y%m%d")
        compressed = gzip_closed and d < days - 1
        name = f"{prefix}_{date}.csv" + (".gz" if compressed else "")
        part.to_csv(out / name, index=False, compression="gzip" if compressed else None)


def generate(out_dir: Path, config: SyntheticConfig | None = None) -> Path:
    """Write a complete synthetic capture directory and return its path."""
    cfg = config or SyntheticConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    start_ms = int(pd.Timestamp(cfg.start, tz="UTC").timestamp() * 1000)
    n_seconds = cfg.days * 86_400
    users = _addresses(rng, 64)

    books, ws_books, hl_trades, cex_trades, liqs = [], [], [], {k: [] for k in CEX_TRADE_PREFIXES}, []
    for coin, price0 in cfg.coins.items():
        market = _Market(rng, n_seconds, price0, cfg.hl_lag_s)
        params = _COIN_PARAMS.get(coin, _COIN_PARAMS["SOL"])

        ticks = np.arange(0, n_seconds * 1000, cfg.poll_interval_s * 1000, dtype=np.int64)
        for venue in cfg.book_venues:
            ts = ticks + rng.integers(50, 900, len(ticks))
            books.append(_book_frame(rng, market, coin, venue, ts, 0.15))

        if coin in cfg.ws_coins:
            for venue in cfg.ws_venues:
                ts = _event_times(rng, n_seconds, cfg.ws_mean_interval_s)
                frame = _book_frame(rng, market, coin, venue, ts, 0.25)
                frame.insert(1, "recv_ts_ms", ts + rng.integers(40, 400, len(ts)))
                ws_books.append(frame)

        ts = _event_times(rng, n_seconds, cfg.trade_mean_interval_s)
        mid = market.price_at(ts, "hyperliquid")
        buy = rng.random(len(ts)) < 0.5 + 0.25 * np.tanh(market.latent[market._sec(ts)])
        px = _quantize(mid * np.where(buy, 1 + 0.15e-4, 1 - 0.15e-4), params["tick"]["hyperliquid"])
        notional = params["hl_trade_usd"] * rng.lognormal(0, 1.4, len(ts))
        a_idx, b_idx = rng.integers(0, len(users), (2, len(ts)))
        hl_trades.append(pd.DataFrame({
            "ts_ms": ts, "coin": coin, "px": np.round(px, 6),
            "sz": np.maximum(np.round(notional / px, params["sz_decimals"]), 10.0 ** -params["sz_decimals"]),
            "side": np.where(buy, "B", "A"),
            "tid": np.cumsum(rng.integers(1, 1000, len(ts))) + 10**12,
            "user_a": users[a_idx], "user_b": users[b_idx],
        }))

        if coin in cfg.cex_tape_coins:
            for venue in CEX_TRADE_PREFIXES:
                ts = _event_times(rng, n_seconds, cfg.trade_mean_interval_s)
                px = market.price_at(ts, venue)
                ahead = market.price[np.clip(ts // 1000 + 5, 0, n_seconds - 1)] - market.price[ts // 1000]
                informed = rng.random(len(ts)) < 0.65
                buy = np.where(informed, ahead > 0, rng.random(len(ts)) < 0.5)
                notional = 5_000.0 * rng.lognormal(0, 1.6, len(ts))
                cex_trades[venue].append(pd.DataFrame({
                    "ts_ms": ts, "recv_ts_ms": ts + rng.integers(20, 250, len(ts)),
                    "coin": coin, "px": np.round(_quantize(px, params["tick"][venue]), 6),
                    "sz": np.round(notional / px, 4), "side": np.where(buy, "B", "A"),
                    "tid": np.cumsum(rng.integers(1, 50, len(ts))) + 10**9,
                }))

        n_bursts = cfg.liquidation_bursts_per_day * cfg.days
        for t0 in np.sort(rng.integers(0, n_seconds - 60, n_bursts)) * 1000:
            n_prints = int(rng.integers(3, 16))
            ts = np.sort(t0 + rng.integers(0, 8000, n_prints))
            side = "A" if rng.random() < 0.5 else "B"
            sign = -1 if side == "A" else 1
            px = market.price_at(ts, "hyperliquid") * (1 + sign * 5e-4)
            accounts = users[rng.integers(0, len(users), int(rng.integers(1, 6)))]
            liqs.append(pd.DataFrame({
                "ts_ms": ts, "coin": coin, "px": np.round(px, 6),
                "sz": np.round(rng.uniform(5_000, 80_000, n_prints) / px, params["sz_decimals"]),
                "side": side, "tid": rng.integers(10**13, 10**14, n_prints),
                "liq_user": accounts[rng.integers(0, len(accounts), n_prints)],
            }))

        hours = np.arange(0, n_seconds * 1000, 3_600_000, dtype=np.int64) + start_ms
        pd.DataFrame({"time": hours, "coin": coin,
                      "fundingRate": np.round(rng.normal(1.25e-5, 2.5e-5, len(hours)), 10)}
                     ).to_csv(out / FUNDING_FILE_TEMPLATE.format(coin=coin), index=False)

    def finish(frames: list[pd.DataFrame], sort_key: str) -> pd.DataFrame:
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.sort_values(sort_key, kind="stable").reset_index(drop=True)
        frame["ts_ms"] = frame["ts_ms"] + start_ms
        if "recv_ts_ms" in frame:
            frame["recv_ts_ms"] = frame["recv_ts_ms"] + start_ms
        return frame

    _write_daily(finish(books, "ts_ms"), out, BOOK_REST_PREFIX, BOOK_REST_COLUMNS,
                 start_ms, cfg.days, cfg.gzip_closed_days)
    if ws_books:
        _write_daily(finish(ws_books, "recv_ts_ms"), out, BOOK_WS_PREFIX, BOOK_WS_COLUMNS,
                     start_ms, cfg.days, cfg.gzip_closed_days, day_key="recv_ts_ms")
    _write_daily(finish(hl_trades, "ts_ms"), out, HL_TRADES_PREFIX, HL_TRADES_COLUMNS,
                 start_ms, cfg.days, cfg.gzip_closed_days)
    for venue, frames in cex_trades.items():
        if frames:
            _write_daily(finish(frames, "recv_ts_ms"), out, CEX_TRADE_PREFIXES[venue],
                         CEX_TRADES_COLUMNS, start_ms, cfg.days, cfg.gzip_closed_days)
    if liqs:
        _write_daily(finish(liqs, "ts_ms"), out, HL_LIQUIDATIONS_PREFIX, HL_LIQUIDATIONS_COLUMNS,
                     start_ms, cfg.days, cfg.gzip_closed_days)
    log.info("synthetic capture written to %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hlmicro synth", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("sample_data"))
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--start", default="2026-01-05", help="first day, YYYY-MM-DD")
    parser.add_argument("--coins", type=csv_list(str.upper), default=["BTC", "SOL"])
    parser.add_argument("--seed", type=int, default=7)
    add_common_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    prices = {"BTC": 60_000.0, "SOL": 150.0}
    cfg = SyntheticConfig(start=args.start, days=args.days, seed=args.seed,
                          coins={c: prices.get(c, 100.0) for c in args.coins})
    generate(args.out, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
