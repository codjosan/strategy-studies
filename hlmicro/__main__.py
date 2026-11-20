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
    "inventory": ("hlmicro.parse.inventory", "coverage, integrity and health of a capture directory"),
    "synth": ("hlmicro.synthetic", "write a synthetic capture directory"),
    # validate: imbalance z-score signal
    "tp-multifold": ("hlmicro.validate.tp_multifold", "stop/TP grid across chronological folds"),
    "threshold-cv": ("hlmicro.validate.threshold_cv", "threshold selection across chronological folds"),
    "oos-split": ("hlmicro.validate.oos_split", "fixed configuration, train vs later data"),
    "breakeven-exit": ("hlmicro.validate.breakeven_exit", "breakeven exit vs timeout, with failure flags"),
    "signal-exit": ("hlmicro.validate.signal_exit", "z-score reversal and extension exits"),
    "trend-gate": ("hlmicro.validate.trend_gate", "SMA trend filter on long entries"),
    "obi-configs": ("hlmicro.validate.obi_configs", "trigger variants under identical exits, TFI gates"),
    "source-compare": ("hlmicro.validate.source_compare", "websocket vs 5s imbalance as a signal"),
    "hf-features": ("hlmicro.validate.hf_features", "within-minute book features vs trade outcomes"),
    # validate: flow, cross-venue and event studies
    "tfi-signals": ("hlmicro.validate.tfi_signals", "taker-flow imbalance z-score and streak signals"),
    "lead-lag": ("hlmicro.validate.lead_lag", "HL response to large CEX prints"),
    "cex-hl-lag": ("hlmicro.validate.cex_hl_lag", "HL repricing after CEX moves"),
    "funding-settlement": ("hlmicro.validate.funding_settlement", "returns around funding settlement"),
    # validate: short-horizon rules and signal discovery
    "multi-gate": ("hlmicro.validate.multi_gate", "short-horizon AND-gate rules"),
    "scalp-explore": ("hlmicro.validate.scalp_explore", "tick-resolution rule sweep with fill realism"),
    "feature-explore": ("hlmicro.validate.feature_explore", "minute-grid feature discovery, 3-way split"),
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
