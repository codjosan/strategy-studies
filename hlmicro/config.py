"""Shared constants and the cost model used by every study."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HL_VENUE = "hyperliquid"
BOOK_VENUES = ("hyperliquid", "binance", "okx", "bybit")

CAPTURE_DIR_ENV = "HLMICRO_CAPTURE_DIR"


def default_capture_dir() -> Path:
    """Capture directory from ``$HLMICRO_CAPTURE_DIR``, else ``./capture_data``."""
    return Path(os.environ.get(CAPTURE_DIR_ENV, "capture_data"))


@dataclass(frozen=True)
class FeeModel:
    """Per-side exchange fees and quoted half-spread, in basis points.

    Defaults are Hyperliquid's base-tier perpetual fees with the referral
    discount applied.
    """

    taker_bps: float = 4.32
    maker_bps: float = 1.44
    half_spread_bps: float = 0.14

    @property
    def taker_round_trip_bps(self) -> float:
        """Taker in, taker out; fees only."""
        return 2.0 * self.taker_bps

    @property
    def maker_round_trip_bps(self) -> float:
        """Maker in, maker out; fees only."""
        return 2.0 * self.maker_bps

    def round_trip_with_spread_bps(self, entry_fee_bps: float, exit_fee_bps: float) -> float:
        """Round-trip cost including half the quoted spread on each leg."""
        return entry_fee_bps + exit_fee_bps + 2.0 * self.half_spread_bps


DEFAULT_FEES = FeeModel()
