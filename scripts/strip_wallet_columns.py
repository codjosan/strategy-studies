#!/usr/bin/env python3
"""Remove wallet-address columns from raw capture files, in place.

Everything else stays byte for byte as captured: rows, their order,
duplicates and line endings. Only the address columns (header name in
ADDRESS_COLUMNS, or every sampled value an 0x address) are cut out. Use
``hlmicro sanitize`` for cleaning; this script only removes identifiers from
the raw record.

Safety:

* every removed value must be an 0x address (40 hex digits) or empty, and
  every line must have the header's field count, otherwise the file is left
  untouched and reported
* files are rewritten through a temporary file and an atomic rename; gzip
  files stay gzip (same inner file name and timestamp)
* after writing, the file is read back: no address may remain, the line
  count must match, and a hash of the kept content must equal the one
  computed while writing

Usage::

    python scripts/strip_wallet_columns.py capture_data ../trade_tape_Sep_aug [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ADDRESS_COLUMNS = {"user", "users", "user_a", "user_b", "liq_user", "buyer", "seller", "address", "wallet", "hash"}
ADDRESS = re.compile(rb"0x[0-9a-fA-F]{40}")
FIELD = re.compile(rb"^(0x[0-9a-fA-F]{40})?$")
SAMPLE = 1000


def _read(path: Path, gz: bool | None = None) -> bytes:
    gz = path.suffix == ".gz" if gz is None else gz
    with (gzip.open(path, "rb") if gz else open(path, "rb")) as fh:
        return fh.read()


def _address_columns(header: list[bytes], sample: list[list[bytes]]) -> list[int]:
    cols = []
    for i, name in enumerate(header):
        values = [row[i] for row in sample if len(row) == len(header)]
        by_name = name.strip().lower().decode("utf-8", "replace") in ADDRESS_COLUMNS
        by_value = bool(values) and all(ADDRESS.fullmatch(v) for v in values)
        if by_name or by_value:
            cols.append(i)
    return cols


def strip(data: bytes) -> tuple[bytes, list[str], int]:
    """Return (new_data, removed_column_names, n_lines). Raises ValueError if unsafe."""
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
        trailing = True
    else:
        trailing = False
    split = lambda ln: (ln[:-1] if ln.endswith(b"\r") else ln).split(b",")
    header = split(lines[0])
    cols = _address_columns(header, [split(ln) for ln in lines[1:SAMPLE + 1]])
    if not cols:
        return data, [], len(lines)
    keep = [i for i in range(len(header)) if i not in cols]
    out = []
    for n, ln in enumerate(lines):
        cr = ln.endswith(b"\r")
        fields = split(ln)
        if len(fields) != len(header):
            raise ValueError(f"line {n + 1} has {len(fields)} fields, header has {len(header)}")
        if n and not all(FIELD.match(fields[i]) for i in cols):
            raise ValueError(f"line {n + 1}: a removed field is not a wallet address")
        out.append(b",".join(fields[i] for i in keep) + (b"\r" if cr else b""))
    new = b"\n".join(out) + (b"\n" if trailing else b"")
    return new, [header[i].decode() for i in cols], len(lines)


def process(path: Path, dry_run: bool) -> str:
    data = _read(path)
    if not ADDRESS.search(data):
        return f"{path.name}: no addresses, unchanged"
    new, removed, n_lines = strip(data)
    if ADDRESS.search(new):
        raise ValueError("addresses remain outside the address columns; not rewritten")
    digest = hashlib.sha256(new).hexdigest()
    if dry_run:
        return f"{path.name}: would remove {', '.join(removed)} ({n_lines:,} lines)"
    tmp = path.with_name(path.name + ".strip.tmp")
    if path.suffix == ".gz":
        mtime = os.stat(path).st_mtime
        with open(path, "rb") as fh:                          # keep the inner name/mtime of the original
            head = fh.read(10 + 512)
        inner = path.name[:-3]
        if head[3] & 0x08 and not head[3] & 0x04:              # FNAME set, no FEXTRA before it
            inner = head[10:].split(b"\0", 1)[0].decode("latin-1") or inner
        gz_mtime = int.from_bytes(head[4:8], "little")
        with open(tmp, "wb") as raw, gzip.GzipFile(filename=inner, mode="wb", fileobj=raw,
                                                   compresslevel=9, mtime=gz_mtime) as fh:
            fh.write(new)
    else:
        mtime = os.stat(path).st_mtime
        tmp.write_bytes(new)
    check = _read(tmp, gz=path.suffix == ".gz")
    if ADDRESS.search(check) or hashlib.sha256(check).hexdigest() != digest:
        tmp.unlink()
        raise ValueError("verification of the rewritten file failed; original kept")
    os.replace(tmp, path)
    os.utime(path, (mtime, mtime))
    return f"{path.name}: removed {', '.join(removed)} ({n_lines:,} lines, verified)"


def _run(args):
    path, dry = args
    try:
        return True, process(path, dry)
    except Exception as exc:
        return False, f"{path.name}: NOT CHANGED - {exc}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    files = []
    for p in args.paths:
        files += sorted(p.glob("*.csv")) + sorted(p.glob("*.csv.gz")) if p.is_dir() else [p]
    with ProcessPoolExecutor(args.jobs) as ex:
        results = list(ex.map(_run, [(f, args.dry_run) for f in files]))
    for _, msg in results:
        print(msg)
    failed = [m for ok, m in results if not ok]
    print(f"\n{len(files)} files, {len(failed)} not changed because of an error")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
