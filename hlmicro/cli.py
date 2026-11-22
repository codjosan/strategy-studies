"""Small helpers shared by the command-line entry points."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable, TypeVar

T = TypeVar("T")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def csv_list(cast: Callable[[str], T]) -> Callable[[str], list[T]]:
    """argparse ``type=`` for comma-separated values, e.g. ``csv_list(float)``."""
    def parse(text: str) -> list[T]:
        try:
            return [cast(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid list {text!r}: {exc}") from exc
    return parse


class _EnvAction(argparse.Action):
    """Store the option in an environment variable, read by :mod:`hlmicro.parse.filecache`
    (worker processes inherit it)."""

    def __init__(self, option_strings, dest, env: str, value=None, **kwargs):
        self.env, self.value = env, value
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = self.value if self.value is not None else values
        os.environ[self.env] = str(value)
        setattr(namespace, self.dest, value)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--jobs", type=int, action=_EnvAction, env="HLMICRO_JOBS", default=None,
                        help="worker processes for loading files (default: up to 4 CPUs)")
    parser.add_argument("--no-cache", nargs=0, action=_EnvAction, env="HLMICRO_NO_CACHE", value="1",
                        default=None, help="do not read or write the per-file loader cache")


def add_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", default=None, help="first capture date, YYYYMMDD")
    parser.add_argument("--end", default=None, help="last capture date, YYYYMMDD")
