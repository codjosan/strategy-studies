"""On-disk file families and their column layouts.

Every capture file is a daily CSV named ``<prefix>_YYYYMMDD.csv``; files from
completed days may be gzipped (``.csv.gz``).
"""
from __future__ import annotations

# Order-book snapshots polled over REST (``obi_<interval>s``, normally ``obi_5s``).
BOOK_REST_PREFIX = "obi_5s"
BOOK_REST_COLUMNS = (
    "ts_ms", "coin", "venue", "mid", "bid1_px", "ask1_px",
    "bid_qty_5", "ask_qty_5", "bid_ntl_5", "ask_ntl_5",
    "obi_qty", "obi_ntl", "spread_bps",
)

# Order-book updates pushed over websocket. ``recv_ts_ms`` is local receive time.
BOOK_WS_PREFIX = "obi_ws"
BOOK_WS_COLUMNS = (
    "ts_ms", "recv_ts_ms", "coin", "venue", "mid", "bid1_px", "ask1_px",
    "bid_qty_5", "ask_qty_5", "bid_ntl_5", "ask_ntl_5",
    "obi_qty", "obi_ntl", "spread_bps",
)

# Trade tapes. Hyperliquid labels the aggressor side "B" (buy) / "A" (sell).
HL_TRADES_PREFIX = "hl_trades"
HL_TRADES_COLUMNS = ("ts_ms", "coin", "px", "sz", "side", "tid", "user_a", "user_b")

CEX_TRADE_PREFIXES = {"okx": "okx_trades", "bybit": "bybit_trades"}
CEX_TRADES_COLUMNS = ("ts_ms", "recv_ts_ms", "coin", "px", "sz", "side", "tid")

# Liquidation prints. side "A" = liquidated long (forced sell),
# side "B" = liquidated short (forced buy).
HL_LIQUIDATIONS_PREFIX = "hl_liquidations"
HL_LIQUIDATIONS_COLUMNS = ("ts_ms", "coin", "px", "sz", "side", "tid", "liq_user")

# Hourly funding history (one file per coin, not daily).
FUNDING_FILE_TEMPLATE = "hl_funding_{coin}.csv"
FUNDING_COLUMNS = ("time", "coin", "fundingRate")

ALL_DAILY_PREFIXES = (
    BOOK_REST_PREFIX,
    BOOK_WS_PREFIX,
    HL_TRADES_PREFIX,
    *CEX_TRADE_PREFIXES.values(),
    HL_LIQUIDATIONS_PREFIX,
)
