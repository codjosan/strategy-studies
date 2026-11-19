"""Small helpers shared by the command-line entry points."""
from __future__ import annotations

import argparse
import logging
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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def add_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", default=None, help="first capture date, YYYYMMDD")
    parser.add_argument("--end", default=None, help="last capture date, YYYYMMDD")
