"""Hyperliquid market-microstructure research toolkit.

The package is organised as a three-stage pipeline:

``hlmicro.capture``
    Collectors that write daily order-book snapshot files.
``hlmicro.parse``
    Chunked, fault-tolerant loaders for order-book, trade-tape, liquidation
    and funding data.
``hlmicro.validate``
    Event studies and backtests built on the parse layer, each with explicit
    pass/fail criteria.
"""

__version__ = "1.1.0"
