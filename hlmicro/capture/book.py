"""Top-of-book metrics shared by the REST and websocket collectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

Level = tuple[float, float]  # (price, size)

DEFAULT_DEPTH = 5


@dataclass(frozen=True)
class BookMetrics:
    mid: float
    bid1_px: float
    ask1_px: float
    bid_qty: float
    ask_qty: float
    bid_ntl: float
    ask_ntl: float
    obi_qty: float
    obi_ntl: float
    spread_bps: float

    def as_fields(self) -> list[str]:
        """Values in schema order, formatted for CSV output."""
        return [
            f"{self.mid:.6f}", f"{self.bid1_px:.6f}", f"{self.ask1_px:.6f}",
            f"{self.bid_qty:.4f}", f"{self.ask_qty:.4f}",
            f"{self.bid_ntl:.2f}", f"{self.ask_ntl:.2f}",
            f"{self.obi_qty:.4f}", f"{self.obi_ntl:.4f}",
            f"{self.spread_bps:.2f}",
        ]


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def book_metrics(bids: Sequence[Level], asks: Sequence[Level],
                 depth: int = DEFAULT_DEPTH) -> BookMetrics | None:
    """Imbalance, depth and spread over the top ``depth`` levels.

    ``bids`` and ``asks`` are best-first. Returns ``None`` for an empty or
    crossed-to-zero book so that no row is written for it.
    """
    bids = list(bids)[:depth]
    asks = list(asks)[:depth]
    if not bids or not asks:
        return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    bid_qty = sum(q for _, q in bids)
    ask_qty = sum(q for _, q in asks)
    bid_ntl = sum(p * q for p, q in bids)
    ask_ntl = sum(p * q for p, q in asks)
    return BookMetrics(
        mid=mid,
        bid1_px=best_bid,
        ask1_px=best_ask,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        bid_ntl=bid_ntl,
        ask_ntl=ask_ntl,
        obi_qty=_imbalance(bid_qty, ask_qty),
        obi_ntl=_imbalance(bid_ntl, ask_ntl),
        spread_bps=(best_ask - best_bid) / mid * 1e4,
    )
