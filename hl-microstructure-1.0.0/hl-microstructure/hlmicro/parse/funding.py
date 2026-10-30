"""Hourly Hyperliquid funding rates: from the public API or a saved CSV."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HOUR_MS = 3_600_000


def _post(payload: dict, timeout: float) -> object:
    req = urllib.request.Request(
        HL_INFO_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "hlmicro/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_funding_history(coin: str, start_ms: int, end_ms: int, *, timeout: float = 15.0,
                          max_pages: int = 50, pause_s: float = 0.05) -> pd.DataFrame:
    """Funding history from the ``fundingHistory`` endpoint, paging by time.

    Returns columns ``time`` (ms) and ``funding_rate``, in API order.
    """
    samples: list[dict] = []
    cursor = start_ms
    for _ in range(max_pages):
        try:
            page = _post({"type": "fundingHistory", "coin": coin,
                          "startTime": cursor, "endTime": end_ms}, timeout)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("funding fetch stopped: %s", exc)
            break
        if not isinstance(page, list) or not page:
            break
        samples.extend(page)
        last = max(int(s.get("time", 0)) for s in page)
        if last <= cursor or last >= end_ms - HOUR_MS:
            break
        cursor = last + 1
        time.sleep(pause_s)
    rows = []
    for s in samples:
        try:
            rows.append((int(s["time"]), float(s["fundingRate"])))
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows, columns=["time", "funding_rate"])


def load_funding_csv(path: str | Path) -> pd.DataFrame:
    """Funding history saved as CSV with ``time`` (ms) and ``fundingRate`` or
    ``funding_rate`` columns."""
    frame = pd.read_csv(path)
    rate_col = "fundingRate" if "fundingRate" in frame else "funding_rate"
    out = pd.DataFrame({"time": pd.to_numeric(frame["time"], errors="coerce"),
                        "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce")})
    out = out.dropna()
    out["time"] = out["time"].astype("int64")
    return out


def funding_by_hour(funding: pd.DataFrame) -> dict[int, float]:
    """``{hour_start_ms: rate}``; when an hour appears twice the later row wins."""
    out: dict[int, float] = {}
    for t, rate in zip(funding["time"].to_numpy(), funding["funding_rate"].to_numpy()):
        out[int(t) // HOUR_MS * HOUR_MS] = float(rate)
    return out
