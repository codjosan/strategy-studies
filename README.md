# hlmicro

Capture, parsing and validation tools for market-microstructure research on
Hyperliquid perpetuals, with cross-venue reference data from Binance, OKX and
Bybit.

The package is a three-stage pipeline:

```
capture  ->  daily CSV files  ->  parse  ->  validate
```

| Stage | Module | Purpose |
|---|---|---|
| capture | `hlmicro.capture` | Collectors for top-of-book snapshots (REST polling and websocket push) writing day-rotated CSV files |
| parse | `hlmicro.parse` | Chunked, fault-tolerant loaders for order books, trade tapes, liquidations and funding |
| validate | `hlmicro.validate` | Event studies and backtests with explicit, pre-stated pass/fail criteria |

Every study runs from a single command and is also importable: each module
exposes a `Config` dataclass, `run(config) -> Result` and `render(result) -> str`.

## Project history

The nine original research scripts predate this repository and were imported
into Git in the initial commit on 30 October 2026; ten more were integrated in
1.1.0. The [CHANGELOG](./CHANGELOG.md) maps every script to its command and
lists the ones deliberately left out, with reasons.

## Data privacy

Do not commit real capture files or wallet/account identifiers. The parser
discards account identifier fields from its outputs, and `hlmicro synth`
creates made-up data for examples and tests. Generated captures and results
are excluded by `.gitignore`.

## Install

```bash
pip install -e ".[capture,plot,dev]"
```

Python 3.10+, with numpy, pandas and scipy. `websocket-client` is needed only
for websocket capture, `matplotlib` only for plots.

## Quick start with synthetic data

No capture data is needed to try the pipeline:

```bash
python -m hlmicro synth --out sample_data --days 4
export HLMICRO_CAPTURE_DIR=sample_data

python -m hlmicro inventory
python -m hlmicro tp-multifold --symbol BTC --hold-min 30 --cooldown-min 10
python -m hlmicro hf-features --symbol BTC \
    --trades-csv results/tp_multifold/tp_trades_btc.csv --filter tp_pct=0,stop_pct=-1.0
python -m hlmicro source-compare --symbol BTC --threshold 2.5
python -m hlmicro lead-lag --symbol BTC
python -m hlmicro cex-hl-lag --symbols BTC,SOL
python -m hlmicro multi-gate --symbol BTC
python -m hlmicro funding-settlement --symbol BTC

python -m hlmicro threshold-cv --symbol BTC --hold-min 30 --cooldown-min 10 --stop-pct=-1.0
python -m hlmicro oos-split --symbol BTC --hold-min 30 --cooldown-min 10 --threshold-sweep
python -m hlmicro breakeven-exit --symbol BTC --cooldown-min 10 --reference-hold-min 30 \
    --max-hold-levels 30,120,480
python -m hlmicro signal-exit --symbol BTC --hold-min 30 --max-hold-min 60 --cooldown-min 10
python -m hlmicro obi-configs --symbols SOL,BTC
python -m hlmicro tfi-signals --symbols BTC --z-horizons-min 5,15,30
python -m hlmicro scalp-explore --symbol BTC
python -m hlmicro feature-explore --symbol BTC --cooldown-min 15
python -m hlmicro inventory --check
```

The synthetic generator writes every file family in the production schemas
(including gzipped closed days). Its price and order-flow processes are not
calibrated to real markets, so study results on it are not meaningful; it
exists to exercise the code.

`python -m hlmicro` with no arguments lists all commands; each takes `--help`.
Lists of negative numbers must be attached with `=`, e.g. `--stop-levels=-2.5,-1.0`.

## Commands

| Command | Stage | Description |
|---|---|---|
| `capture-rest` | capture | Poll top-5 depth on Hyperliquid, Binance, OKX and Bybit every N seconds |
| `capture-ws` | capture | Record every Hyperliquid `l2Book` and OKX `books5` update |
| `inventory` | parse | Files, date coverage, rows per coin, malformed rows and truncated files per family; health checks for gaps, missing and low-volume days and staleness (`--check` exits 1 on failure) |
| `synth` | parse | Write a synthetic capture directory |
| `tp-multifold` | validate | Stop/take-profit grid for the dual-window imbalance z-score signal across chronological folds |
| `threshold-cv` | validate | Threshold selection for the same signal across folds, optionally with a stop, ranked by fold consistency |
| `oos-split` | validate | A fixed configuration on a training window and on the data after it, with a pass/fail verdict |
| `breakeven-exit` | validate | Breakeven exit against the timeout exit, flagging censoring, win-rate trap, negative skew and capital lockup |
| `signal-exit` | validate | Exits on z-score reversal and holds extended while the z-score persists, with the reversal counterfactual |
| `trend-gate` | validate | SMA trend filter on long entries, chosen on training data and tested with a permutation test |
| `obi-configs` | validate | Imbalance trigger variants under identical exits, ungated and with taker-flow (TFI) gates |
| `source-compare` | validate | Same signal driven by websocket vs 5-second imbalance over identical minutes |
| `hf-features` | validate | Within-minute book features (update rate, imbalance volatility, drift) tested against trade outcomes |
| `tfi-signals` | validate | Cross-venue taker-flow imbalance z-score and streak signals as an event study |
| `lead-lag` | validate | Event study of the Hyperliquid mid after large OKX/Bybit prints |
| `cex-hl-lag` | validate | Hyperliquid repricing after sharp moves on a leading venue |
| `funding-settlement` | validate | Returns around hourly funding settlement, bucketed by funding rate |
| `multi-gate` | validate | Short-horizon AND-gate rules on flow, imbalance and VWAP deviation, with passive-fill simulation |
| `scalp-explore` | validate | Tick-resolution sweep of single-feature rules under three fee regimes, with passive-fill realism on the best |
| `feature-explore` | validate | Single- and two-feature rules on a minute grid with a train / validate / holdout split |

The imbalance z-score studies share one simulator (`hlmicro.validate.obi_signal`).
By default they enter at the trigger minute's mean mid, as `tp-multifold` does;
`--next-bar-entry` enters a minute later, so the entry price is strictly after
the information used (`oos-split` does this by default). `obi-configs` and
`tfi-signals` price entries at the close of the trigger minute.

## Data files

Daily files are named `<prefix>_YYYYMMDD.csv`; closed days may be gzipped
(`.csv.gz`). Timestamps are Unix milliseconds, UTC.

| Prefix | Written by | Columns |
|---|---|---|
| `obi_5s` | `capture-rest` | `ts_ms, coin, venue, mid, bid1_px, ask1_px, bid_qty_5, ask_qty_5, bid_ntl_5, ask_ntl_5, obi_qty, obi_ntl, spread_bps` |
| `obi_ws` | `capture-ws` | as `obi_5s`, plus `recv_ts_ms` (local receive time) after `ts_ms` |
| `hl_trades` | trade-tape collector | `ts_ms, coin, px, sz, side, tid` |
| `okx_trades`, `bybit_trades` | trade-tape collector | `ts_ms, recv_ts_ms, coin, px, sz, side, tid` |
| `hl_liquidations` | trade-tape collector | `ts_ms, coin, px, sz, side, tid` |
| `hl_funding_<COIN>.csv` | saved API history | `time, coin, fundingRate` (not daily) |

Order-book imbalance is `(bid - ask) / (bid + ask)` over the top five levels,
by quantity (`obi_qty`) and by notional (`obi_ntl`). Trade side is the
aggressor: Hyperliquid writes `B` (buy) and `A` (sell); the parse layer also
accepts `S`, `buy`/`sell` and `bid`/`ask`. For liquidations, `A` is a
liquidated long (forced sell) and `B` a liquidated short (forced buy).

## How files are read

All loaders go through `hlmicro.parse.reader.iter_chunks`:

* files are streamed in chunks and reduced to one symbol before aggregation;
  per-bucket means are accumulated as sums and counts, so results do not
  depend on chunk boundaries
* rows with non-numeric values, missing fields or surplus fields are dropped
  individually, and timestamps outside the file's date (+/- one day) are rejected
* a truncated gzip stream (interrupted write, full disk) yields every complete
  row before the break instead of failing the whole day; an incomplete final
  line (a write in progress) is ignored
* when a date has both `.csv` and `.csv.gz`, the plain file is used

`inventory` reports what was dropped or salvaged per file family.

## Costs

Fees come from `hlmicro.config.FeeModel` (per side: taker 4.32 bps, maker
1.44 bps; half-spread 0.14 bps). Every study accepts fee overrides on the
command line.

## Tests

```bash
pytest
```

The suite generates two days of synthetic data and covers the capture
formatting and rotation, every loader (including damaged and truncated
files), the inventory health checks, the shared simulators, and an
end-to-end run of each study.

## Layout

```
hlmicro/
  capture/   book.py (metrics), writer.py (day-rotating CSV), rest_books.py, ws_books.py
  parse/     reader.py, files.py, books.py, trades.py, liquidations.py, funding.py, inventory.py
  validate/  common.py, obi_signal.py (shared signal + simulator), passive_fills.py,
             tp_multifold.py, threshold_cv.py, oos_split.py, breakeven_exit.py, signal_exit.py,
             trend_gate.py, obi_configs.py, source_compare.py, hf_features.py, tfi_signals.py,
             lead_lag.py, cex_hl_lag.py, funding_settlement.py, multi_gate.py,
             scalp_explore.py, feature_explore.py
  config.py  schemas.py  synthetic.py  cli.py  __main__.py
tests/
```
