"""Command-line entry point: ``python -m hlmicro <command> [options]``."""
from __future__ import annotations

import importlib
import sys

from . import __version__

COMMANDS = {
    # capture
    "capture-rest": ("hlmicro.capture.rest_books", "poll order books over REST"),
    "capture-ws": ("hlmicro.capture.ws_books", "record websocket order-book updates"),
    # parse
    "inventory": ("hlmicro.parse.inventory", "coverage and integrity of a capture directory"),
    "synth": ("hlmicro.synthetic", "write a synthetic capture directory"),
    "sanitize": ("hlmicro.parse.sanitize", "write a cleaned copy of capture files"),
    "cache": ("hlmicro.parse.filecache", "show or clear the per-file loader cache"),
    # validate
    "tp-multifold": ("hlmicro.validate.tp_multifold", "stop/TP grid across chronological folds"),
    "source-compare": ("hlmicro.validate.source_compare", "websocket vs 5s imbalance as a signal"),
    "hf-features": ("hlmicro.validate.hf_features", "within-minute book features vs trade outcomes"),
    "lead-lag": ("hlmicro.validate.lead_lag", "HL response to large CEX prints"),
    "cex-hl-lag": ("hlmicro.validate.cex_hl_lag", "HL repricing after CEX moves"),
    "multi-gate": ("hlmicro.validate.multi_gate", "short-horizon AND-gate rules"),
    "funding-settlement": ("hlmicro.validate.funding_settlement", "returns around funding settlement"),
    "candidates": ("hlmicro.validate.candidates", "seven drafted strategies, scored like run-all"),
    # all studies
    "run-all": ("hlmicro.validate.run_all", "run every feasible study and rank the strategies"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"hlmicro {__version__}", "", "usage: python -m hlmicro <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run a command with --help for its options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] == "--version":
        print(__version__)
        return 0
    command = COMMANDS.get(argv[0])
    if command is None:
        print(f"unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return importlib.import_module(command[0]).main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
