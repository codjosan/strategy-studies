"""Per-file loader results, computed in parallel and cached on disk.

Every loader reduces each daily file on its own (one symbol, aggregated or
filtered) and then combines the per-file parts. :func:`map_files` runs that
per-file step:

* **cache**: the part of each file is stored under ``$HLMICRO_CACHE_DIR``
  (default ``~/.cache/hlmicro``). The key covers the file's resolved path,
  size and modification time, the loader and its arguments, and a hash of the
  parse-layer source code. A changed file, a file still being written, or a
  code change therefore never reuses a stale result. Entries live in one
  folder per code version; folders of other versions unused for 7 days are
  removed automatically. ``HLMICRO_NO_CACHE=1``
  (or ``--no-cache``) turns the cache off; ``python -m hlmicro cache`` shows
  or clears it.
* **parallel**: files without a cached part are processed in up to
  ``$HLMICRO_JOBS`` worker processes (default: up to 4 CPUs; ``--jobs``).

Per-file parts are combined in file order, so results do not depend on the
number of workers or on whether a part came from the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

log = logging.getLogger(__name__)

CACHE_ENV = "HLMICRO_CACHE_DIR"
NO_CACHE_ENV = "HLMICRO_NO_CACHE"
JOBS_ENV = "HLMICRO_JOBS"
_FORMAT = 1
_SOURCES = ("reader.py", "books.py", "trades.py", "liquidations.py", "inventory.py", "filecache.py")


def cache_dir() -> Optional[Path]:
    if os.environ.get(NO_CACHE_ENV, "").strip() not in ("", "0"):
        return None
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "hlmicro")


def jobs() -> int:
    try:
        return max(1, int(os.environ[JOBS_ENV]))
    except (KeyError, ValueError):
        return max(1, min(4, os.cpu_count() or 1))


_code_hash: Optional[str] = None


def _code_version() -> str:
    global _code_hash
    if _code_hash is None:
        h = hashlib.sha256(str(_FORMAT).encode())
        here = Path(__file__).parent
        for name in _SOURCES:
            h.update((here / name).read_bytes())
        _code_hash = h.hexdigest()
    return _code_hash


def _key(fn: Callable, path: Path, params: dict) -> Optional[str]:
    try:
        st = path.stat()
    except OSError:
        return None
    blob = json.dumps([_code_version(), fn.__module__, fn.__qualname__, str(path.resolve()),
                       st.st_size, st.st_mtime_ns, params], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _version_dir(root: Path) -> Path:
    return root / _code_version()[:16]


def _load(root: Path, key: str):
    path = _version_dir(root) / key[:2] / f"{key}.pkl"
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:                       # corrupt entry: recompute
        log.debug("cache entry %s unreadable (%s)", path.name, exc)
        return None


_pruned = False


def _prune(root: Path, max_age_days: float = 7.0) -> None:
    """Remove cache folders of other code versions not used for ``max_age_days``."""
    global _pruned
    if _pruned:
        return
    _pruned = True
    import shutil
    import time
    current = _version_dir(root).name
    for d in root.iterdir() if root.exists() else ():
        if d.is_dir() and d.name != current and time.time() - d.stat().st_mtime > max_age_days * 86400:
            shutil.rmtree(d, ignore_errors=True)


def _store(root: Path, key: str, value) -> None:
    _prune(root)
    path = _version_dir(root) / key[:2] / f"{key}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        os.utime(path.parent.parent)             # marks this code version's folder as in use
    except OSError as exc:                         # read-only or full disk: just do not cache
        log.debug("cannot write cache entry (%s)", exc)


def _call(args):
    fn, path, params, extra = args
    return fn(path, **params, **extra)


def map_files(fn: Callable, files: Sequence[Path], params: dict, extra: Optional[dict] = None,
              stats=None) -> list:
    """``[fn(path, **params, **extra)[0] for path in files]`` with caching and parallelism.

    ``fn`` is a module-level function returning ``(part, ReadStats)``.
    ``params`` determine the result and are part of the cache key; ``extra``
    (e.g. chunk size) only affects how the file is read. Row accounting of
    every file, cached or not, is added to ``stats``.
    """
    extra = extra or {}
    files = [Path(f) for f in files]
    root = cache_dir()
    keys = [_key(fn, p, params) if root else None for p in files]
    results: list = [None] * len(files)
    todo = []
    for i, (path, key) in enumerate(zip(files, keys)):
        hit = _load(root, key) if key else None
        if hit is not None:
            results[i] = hit
        else:
            todo.append(i)
    if todo:
        n = min(jobs(), len(todo))
        args = [(fn, files[i], params, extra) for i in todo]
        if n > 1:
            ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
            with ProcessPoolExecutor(n, mp_context=ctx) as ex:
                computed = list(ex.map(_call, args))
        else:
            computed = [_call(a) for a in args]
        for i, value in zip(todo, computed):
            results[i] = value
            if keys[i]:
                _store(root, keys[i], value)
    log.debug("%s: %d files, %d from cache", fn.__qualname__, len(files), len(files) - len(todo))
    parts = []
    for part, file_stats in results:
        if stats is not None:
            stats.rows_read += file_stats.rows_read
            stats.rows_dropped += file_stats.rows_dropped
            stats.files_truncated += [f for f in file_stats.files_truncated if f not in stats.files_truncated]
        parts.append(part)
    return parts


def cache_info() -> tuple[Optional[Path], int, int]:
    root = cache_dir()
    if root is None or not root.exists():
        return root, 0, 0
    entries = list(root.glob("*/*/*.pkl"))
    return root, len(entries), sum(e.stat().st_size for e in entries)


def clear_cache() -> int:
    import shutil
    root = cache_dir()
    if root is None or not root.exists():
        return 0
    n = len(list(root.glob("**/*.pkl")))
    for d in root.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="hlmicro cache", description="Show or clear the loader cache.")
    parser.add_argument("--clear", action="store_true", help="delete every cached entry")
    args = parser.parse_args(argv)
    if args.clear:
        print(f"removed {clear_cache()} cached entries")
        return 0
    root, n, size = cache_info()
    if root is None:
        print(f"cache disabled (${NO_CACHE_ENV} is set)")
    else:
        print(f"cache: {root}\n  entries: {n:,}\n  size: {size / 1e6:,.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
