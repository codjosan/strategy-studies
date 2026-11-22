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
into Git in the initial commit on 30 October 2026. The [CHANGELOG](./CHANGELOG.md)
records how they evolved into this package.

## Data privacy

Do not commit real capture files or wallet/account identifiers. The parser
discards account identifier fields from its outputs, and `hlmicro synth`
creates made-up data for examples and tests. Generated captures and results
are excluded by `.gitignore`.

Some trade tapes arrive with wallet-address columns (e.g. `user_a`, `user_b`).
`hlmicro sanitize` leaves them out of the clean copy. To remove them from the
raw files too, keeping everything else byte for byte:

```bash
python scripts/strip_wallet_columns.py capture_data --dry-run
python scripts/strip_wallet_columns.py capture_data
```

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
| `inventory` | parse | Files, date coverage, rows per coin, malformed rows and truncated files per family |
| `synth` | parse | Write a synthetic capture directory |
| `cache` | parse | Show the size of the loader cache, or clear it (`--clear`) |
| `sanitize` | parse | Write a cleaned copy of book, trade-tape and liquidation files, with a report of every dropped, corrected and flagged row |
| `tp-multifold` | validate | Stop/take-profit grid for the dual-window imbalance z-score signal across chronological folds |
| `source-compare` | validate | Same signal driven by websocket vs 5-second imbalance over identical minutes |
| `hf-features` | validate | Within-minute book features (update rate, imbalance volatility, drift) tested against trade outcomes |
| `lead-lag` | validate | Event study of the Hyperliquid mid after large OKX/Bybit prints |
| `cex-hl-lag` | validate | Hyperliquid repricing after sharp moves on a leading venue |
| `multi-gate` | validate | Short-horizon AND-gate rules on flow, imbalance and VWAP deviation, with passive-fill simulation |
| `funding-settlement` | validate | Returns around hourly funding settlement, bucketed by funding rate |
| `candidates` | validate | Seven drafted strategies (basis reversion, passive lead-lag, flow fade, volatility-scaled dip, momentum, funding proxy, pairs), scored like `run-all`; see [docs/NEW_STRATEGIES.md](docs/NEW_STRATEGIES.md) |
| `run-all` | validate | Run every study the capture data supports and rank all tested strategies on one cost model |

## Running all studies at once

```bash
python -m hlmicro run-all --capture-dir capture_data_clean
python -m hlmicro run-all --capture-dir capture_data_clean --symbols BTC,ETH --top 20 \
    --config scripts/run_all_variants.example.json
```

`run-all` works in three steps:

1. It checks which studies the capture directory can feed. Studies without
   their data are skipped with the reason, e.g. `lead-lag` needs `obi_ws` and
   OKX/Bybit tapes, and `funding-settlement` needs `hl_funding_<COIN>.csv` or
   the API.
2. It runs the feasible ones: `tp-multifold`, `multi-gate` and
   `funding-settlement` once per coin, `cex-hl-lag` once for all coins.
   `multi-gate` is limited to the period with trade tapes.
3. Every configuration those studies test becomes one strategy, scored on its
   per-trade net returns under one cost model: mid fills and the 8.64 bps
   taker round trip. `multi-gate`'s passive-fill variants keep their simulated
   maker/taker fills.

Strategies are ranked by t-statistic (mean / standard error) and checked for
stability across 4 time periods. `significant` means the result survives a
Bonferroni correction for every strategy ranked; `promising` (t >= 2) does
not. Results are in-sample.

Output in `results/run_all/<timestamp>/`: `leaderboard.md`, `strategies.csv`
(every strategy with all metrics, including the mean before costs) and each
study's own report under `reports/`. `--config` takes a JSON file of
per-study `Config` overrides. A list runs several variants; see
`scripts/run_all_variants.example.json`.

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
* rows with surplus fields are found by counting fields on the raw bytes, so
  only the columns a study needs are parsed

### Speed: parallel loading and the cache

Each loader reduces every daily file on its own (one symbol, aggregated) and
then combines the per-file results:

* Files are processed in parallel, on up to 4 CPUs by default. Change this
  with `--jobs N` on any command, or `HLMICRO_JOBS`.
* Each file's reduced result is cached in `~/.cache/hlmicro`
  (`HLMICRO_CACHE_DIR`). Later runs, including other commands that need the
  same data, skip reading that file.
* The cache key includes the file's path, size and modification time, the
  loader's arguments and the parse code itself. A file that changes, such as
  today's file still being written, or a code update is therefore never
  served stale.
* `--no-cache` (or `HLMICRO_NO_CACHE=1`) turns the cache off.
  `python -m hlmicro cache` shows its size and `python -m hlmicro cache --clear`
  empties it.

Results are identical with and without the cache and for any number of
workers. The tests check this.

`inventory` reports what was dropped or salvaged per file family.

## Sanitizing capture data

`hlmicro sanitize` reads the raw capture directory and writes a cleaned copy of
every daily file to a separate directory. The input is never changed. It
handles order books (`obi_5s`, `obi_ws`), trade tapes (`hl_trades`,
`okx_trades`, `bybit_trades`) and `hl_liquidations`.

```bash
python -m hlmicro sanitize --capture-dir capture_data --out capture_data_clean
export HLMICRO_CAPTURE_DIR=capture_data_clean
```

* **Dropped**: rows that cannot be a correct record. These are wrong field
  counts, missing or non-numeric values, timestamps outside the file's day,
  and duplicates.
  * Books also lose unknown venues, non-positive or crossed prices, and rows
    whose mid, spread or imbalance contradicts their own prices and sizes.
  * Tapes also lose unknown side labels, malformed trade ids, non-positive
    price or size, and trades replayed after a reconnect (same coin and trade
    id).
* **Corrected**:
  * line endings normalised to `\n`
  * output limited to the schema columns, which removes the wallet addresses
    (`user_a`/`user_b`) some tapes carry
  * coin, venue and side names normalised (sides become `B`/`A`)
  * rows sorted by time
  * OKX book depth converted from contracts to coins (see below)
  * a trade written into the next day's file just after midnight is moved to
    its own day
* **Flagged, kept**: real market or collector events. Deleting them would bias
  the studies.
  * books: spreads above 50 bps, mids more than 100 bps from the other venues
  * tapes: prints more than 100 bps from the surrounding trades (sweeps),
    trades above $10M
  * both: capture gaps and outages

Per family the output directory holds `sanitize_report_<family>.csv` (per
file and rule), `sanitize_flags_<family>.csv` and `sanitize_gaps_<family>.csv`.
`SANITIZED.json` records every run. A directory with `SANITIZED.json` is
refused as input, so nothing is converted twice. Run
`python -m hlmicro sanitize --help` for the full rule list.

**OKX units.** The collectors store OKX `bid/ask_qty_5` and `bid/ask_ntl_5` in
contracts, not coins. Mid, spread and imbalance are unaffected.
`sanitize` multiplies the OKX depth columns by the contract value (BTC 0.01,
ETH 0.1, SOL 1, LINK 1, DOGE 1000, HYPE 1; add others with
`--okx-contract-values COIN=VALUE`). OKX rows of coins without a known value
are kept as captured and listed in the summary.

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
files), and an end-to-end run of each study.

## Layout

```
hlmicro/
  capture/   book.py (metrics), writer.py (day-rotating CSV), rest_books.py, ws_books.py
  parse/     reader.py, files.py, books.py, trades.py, liquidations.py, funding.py, inventory.py,
             sanitize.py, filecache.py
  validate/  common.py, tp_multifold.py, source_compare.py, hf_features.py, lead_lag.py,
             cex_hl_lag.py, multi_gate.py, funding_settlement.py
  config.py  schemas.py  synthetic.py  cli.py  __main__.py
tests/
```
