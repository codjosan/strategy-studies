from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

CHUNK = 200_000
DAY_S = 86_400
FULL_COLS = {
    "hl_trades_": ["ts_ms", "coin", "px", "sz", "side", "tid"],
    "hl_liquidations_": ["ts_ms", "coin", "px", "sz", "side", "tid"],
}

def glob_daily(directory: Path, prefix: str) -> dict[str, Path]:
    seen: dict[str, Path] = {}
    for fp in sorted(directory.glob(f"{prefix}*.csv*")):
        stem = fp.name.replace(".csv.gz", "").replace(".csv", "")
        d = stem.split("_")[-1]
        if not (d.isdigit() and len(d) == 8):
            continue
        if d in seen:
            if seen[d].suffix == ".gz" and fp.suffix != ".gz":
                seen[d] = fp
        else:
            seen[d] = fp
    return {d: seen[d] for d in sorted(seen)}

def _compression(fp: Path) -> str | None:
    return "gzip" if fp.suffix == ".gz" else None

def _read_args(fp: Path, usecols: list[str]) -> dict:
    try:
        header = list(pd.read_csv(fp, compression=_compression(fp), nrows=0).columns)
    except Exception:
        header = []
    if all(c in header for c in usecols):
        return {"usecols": usecols}
    full = next((v for k, v in FULL_COLS.items() if fp.name.startswith(k)), None)
    if full is None:
        return {"usecols": usecols}
    print(f"    WARN: {fp.name}: unexpected header — reading with the known schema",
          file=sys.stderr)
    return {"header": None, "names": full, "usecols": usecols}

def _chunks(fp: Path, usecols: list[str], numeric: list[str], symbols: set[str]):
    try:
        for ch in pd.read_csv(fp, compression=_compression(fp), chunksize=CHUNK,
                              on_bad_lines="skip", dtype=str, **_read_args(fp, usecols)):
            ch = ch[ch["coin"].isin(symbols)]
            if ch.empty:
                continue
            for c in numeric:
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
            ch = ch.dropna(subset=numeric)
            if ch.empty:
                continue
            yield ch
    except Exception as e:
        print(f"    WARN: {fp.name}: {type(e).__name__}: {e} (kept rows read so far)",
              file=sys.stderr)

def day_start_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
               .timestamp() * 1000)
