# Changelog

## Unreleased

### Performance

Measured on the July-September capture (72 days `obi_5s`, 34 days `hl_trades`),
4 CPUs, outputs byte-identical to the previous version:

| command | before | first run | later runs |
|---|---:|---:|---:|
| `tp-multifold --symbol BTC --hold-min 30 --cooldown-min 10` | 66 s | 15-19 s | 1.2-1.8 s |
| `cex-hl-lag --symbols BTC,SOL` | 284 s | 45 s | 5 s |
| `multi-gate --symbol BTC` | 204 s | 41-59 s | 9 s |
| `inventory` | 122 s | 35 s | 1 s |

* New `hlmicro.parse.filecache`: every loader reduces each file on its own, in
  parallel worker processes (up to 4 CPUs), and caches that per-file result
  in `~/.cache/hlmicro`. The cache key covers the file's path, size and
  modification time, the loader's arguments and the parse code, so changed
  files or code are never served stale. Every command accepts `--jobs` and
  `--no-cache`, and `hlmicro cache [--clear]` shows or empties the cache.
* Symbol and venue matching normalises each distinct value once instead of
  every row.
* Venue mids are cached per symbol for all venues, so `cex-hl-lag` and
  `funding-settlement` read each file once per symbol.
* Per-file tick aggregates are combined on integer keys, and `multi-gate`
  pivots with numpy, instead of a MultiIndex groupby and `DataFrame.pivot`.
* `tp-multifold`'s backtest jumps from signal to signal and finds each exit
  with numpy instead of stepping through every minute. A test keeps the old
  loop as a reference and checks identical trades on random grids.
* Fix: a row with a surplus field on the first data line made pandas shift
  every column and lose the whole file. That row is now skipped.

### Added

* `scripts/cointegration_study.py`: correlation / cointegration study of the
  top-5 strategies and the locked vol_norm_dip. It tests HL-vs-CEX and
  cross-coin cointegration, measures the market context at each trade's
  signal, and compares the strategies' P&L correlation. Needs the optional
  `stats` extra (`statsmodels`).
* `candidates.context_features` (market-wide 15-min move, cointegration
  residual vs a fixed partner coin), with a look-ahead test. The forward test
  evaluates the secondary hypothesis "vol_norm_dip only when market_15m < 0
  and coint_z < 0", registered in `forward_lock.json`.
* `scripts/vol_norm_dip_validate.py`: out-of-sample validation of
  `vol_norm_dip` on a pre-declared grid. It locks the forward-test setting in
  `forward_lock.json`.
* `scripts/forward_test.py`: evaluates the locked strategies on captures
  after the lock date, with no parameter changes.
* `candidates`: `vol_norm_dip` takes the VWAP window as a parameter; passive
  fills can require trading through the limit (`through_ticks`), with the
  Hyperliquid price increment from `hl_tick`.
* `scripts/strip_wallet_columns.py` also recognises `liq_user`.
* `hlmicro candidates` (`hlmicro/validate/candidates.py`): seven new
  strategy hypotheses drafted from the capture data, with fixed parameter
  grids. They are scored with the run-all leaderboard plus a half-period
  check, and are part of `run-all`. A test checks that no candidate's signals
  use future data. The design and first results are in
  `docs/NEW_STRATEGIES.md`.
* `scripts/vwap_dev_tpsl.py --set top5|next5`: take-profit / stop-loss sweep
  (10 x 10 levels) for leaderboard ranks 1-5 or 6-10, with entry and execution
  held fixed and each study's own exit mechanics (multi-gate and
  tp-multifold). It checks that each leaderboard setting is reproduced
  exactly, selects on one half and scores on the other, and compares exited
  trades with the same trades held to the end. Output goes to
  `results/tpsl_<set>/`.
* `scripts/vwap_dev_sweep.py` and `scripts/vwap_dev_report.py`: a parameter
  sweep of the `multi-gate` `vwap_dev_buy` rule with train/test halves in both
  directions, and a deep-dive report. The report covers confidence intervals,
  win-rate intervals, outlier dependence, placebo random entries, strict fills,
  neighbouring parameters and other coins.
* `hlmicro run-all`: runs every study the capture data supports and writes a
  leaderboard of all tested strategies. The strategies are scored on one
  cost model and ranked by t-statistic, with a time-stability check and a
  Bonferroni-corrected verdict. Each study's own report is saved alongside.
  `multi-gate` and `funding-settlement` now keep their per-trade results
  (`Result.trades`, `backtests[i]["trades"]`) for this.
* `hlmicro sanitize`: writes a cleaned copy of order-book (`obi_5s`, `obi_ws`),
  trade-tape (`hl_trades`, `okx_trades`, `bybit_trades`) and `hl_liquidations`
  files, plus per-family reports (see README, "Sanitizing capture data").
  * Converts OKX depth from contracts to coins.
  * Removes columns outside the schema, such as wallet addresses in tapes.
  * Drops trades replayed after a reconnect.
  * Moves trades written into the next day's file back to their own day.
* `scripts/strip_wallet_columns.py`: removes wallet-address columns from raw
  capture files in place, leaving every other byte unchanged. It verifies
  each file after rewriting it.

### Fixes

* **Collector restart after a crash mid-row.** The daily writer appended the
  first new row onto the half-written line, which made one merged row with too
  many fields, and the reader dropped it. The writer now ends the broken line
  first, so only the fragment is lost.
* A `DailyCSVWriter` written to after `close()` raised "I/O operation on closed
  file". It now reopens the file. `capture-rest` also waits for in-flight
  fetches before closing the writer at shutdown.
* `capture-ws` reset its reconnect delay only at startup, so after a few drops
  every reconnect waited the full 60 s. The delay now resets once a connection
  has opened.
* `tp-multifold` and `source-compare`: the last fold left out the final minute
  or trade, because the upper edge of every fold was exclusive.
* `hf-features` divided returns by 100 only when some |return| was above 0.2,
  so small-return trade sets stayed in percent. Q1/Q4 were then 100x too large.
  The return columns are always percent and are now always converted. A
  `--filter` on a column that does not exist used to be ignored silently. It
  is now an error.
* `multi-gate`: `--k-bps` takes a comma-separated list like every other list
  option (it used to take space-separated values). `fill_rate` is now fills
  divided by orders actually posted. Triggers that arrived while a position
  was open no longer count.
* `funding-settlement`: the T-5m..T+5m trade return compounds its two windows
  instead of adding the percentages.
* Liquidation-cascade study: `find_events` crashed (np.concatenate of an empty
  list) on days with no earlier history when `--warmup-days 0`. Liquidation
  rows with a side other than `A`/`B` were counted as forced buys; they are now
  ignored. Two of its tests could not pass or tested nothing (they assumed
  10-second days); they now check thresholds without look-ahead and non-overlap
  on real 86,400-second days.

### Behaviour changes

These change study results. Regenerate any tables made with 1.0.0.

* `tp-multifold` enters at the mid of the minute **after** the signal minute.
  It used to use the same minute whose imbalance produced the signal, which
  was look-ahead. Per-trade output gains `signal_ts_ms`.
  `hf-features` takes its features from that signal minute (the last minute
  known at decision time). For older trade files without the column it falls
  back to `entry_ts_ms`.
* `source-compare` records a stopped trade at the price actually observed, not
  at the stop level. A minute that gaps through the stop now shows the full
  loss. Take-profits still fill at their level (resting limit).
* `funding-settlement` books the funding payment the position collects or pays
  at settlement T: shorts receive and longs pay `rate` x notional. The
  backtest now reports price, funding and gross (= price + funding) separately.
  The top/bottom percentile cut-offs are still computed over the whole sample.
* `lead-lag` treats a Hyperliquid mid older than 5 s as missing
  (`--max-quote-age-ms`, 0 = no limit). Websocket gaps no longer count as a
  stable price, and residuals are no longer measured against stale mids.
* Liquidation-cascade study: the default taker fee is 8.64 bps round trip
  (`--fee-bps-rt`), the same as `FeeModel`; it was 9.0. Slippage (2.0 bps)
  is unchanged. The CSVs in `liquid_study/fast_capture/liq_cascade/` were made
  with 11.0 bps total and predate this change.

## 1.0.0

Nine standalone research scripts restructured into one package with separate
capture, parse and validate layers, a single command-line entry point, a
synthetic data generator and a test suite.

| Previous script | Now |
|---|---|
| `obi_capture.py` | `hlmicro capture-rest` |
| `obi_ws_capture.py` | `hlmicro capture-ws` |
| `obi_tp_multifold.py` (v1.1) | `hlmicro tp-multifold` |
| `obi_source_compare.py` | `hlmicro source-compare` |
| `obi_hf_features.py` | `hlmicro hf-features` |
| `lead_lag_analysis.py` | `hlmicro lead-lag` |
| `cex_hl_lag_analysis.py` | `hlmicro cex-hl-lag` |
| `scalp_multi_gate.py` | `hlmicro multi-gate` |
| `funding_flip_scalp_test.py` | `hlmicro funding-settlement` |



### Fixes

* Personal account identifiers are no longer included in generated trade and
  liquidation files or loaded into liquidation results.
* **Trade-side labels in `multi-gate`.** Sell volume was counted only for side
  `S`, but the Hyperliquid tape labels sells `A`. Sell flow was therefore always
  zero: trade-flow imbalance was +1 whenever any trade printed, the sell rules
  could never fire, and the buy rules' flow condition was always met. On the
  synthetic data the previous version produced 1267 / 0 / 463 / 0 triggers for
  the four rules, against 511 / 567 / 287 / 314 now. Results produced with the
  previous version should be regenerated.
* **Price loading in `funding-settlement`.** Only uncompressed files were read,
  and the venue filter used `hl` while the data uses `hyperliquid`, so prices
  silently fell back to Binance for the current day only. All files are now
  read, the venue defaults to `hyperliquid`, and an unknown venue is an error.
* **Damaged files.** A truncated gzip previously cost the whole day, and a single
  non-numeric value made some loaders skip the entire file. Rows are now
  dropped individually and a truncated stream keeps every complete row before
  the break. On a damaged copy of the synthetic data, minutes recovered rose
  from 2880 to 5183 (5-second book) and from 4320 to 5179 (websocket book), out
  of 5760.
* **Surplus-field rows.** When only some columns were requested, pandas accepted
  rows with too many fields, which is what a collector restarting after a
  half-written line produces. These rows are now rejected, as are timestamps
  outside the file's own date.
* Zero-variance z-score windows give NaN instead of +/-inf (`source-compare`).
* Unrecognised trade-side labels are dropped instead of being counted as sells
  (`lead-lag`).
* Dates with both `.csv` and `.csv.gz` are read once (plain file preferred)
  by every loader; previously some loaders read both.

### Behaviour changes

* **Fees** come from one `FeeModel`: per side taker 4.32 bps, maker 1.44 bps,
  half-spread 0.14 bps. Previous defaults differed by script (8.6 bps round
  trip; 6.72 bps taker; 4.5 / 1.5 bps). Use `--fee-bps`, `--taker-bps`,
  `--maker-bps` or `--half-spread-bps` to reproduce earlier tables.
* `tp-multifold`: the stop used for the MAE/MFE diagnostic and the take-profit
  comparison is `--reference-stop` (default -2.5).
* `multi-gate`: rules are named `tick_speed_buy/sell` and `vwap_dev_buy/sell`;
  output columns renamed (`rule`, `hold_s`, `fees`, `mean_bps`,
  `sharpe_first_half`, `sharpe_second_half`, `ticks_held`).
* `funding-settlement` reads funding from `hl_funding_<COIN>.csv` when present
  (or `--funding-csv`) and from the API otherwise, so runs are reproducible
  offline.
* Reports state gate criteria and PASS/FAIL results.
* Output goes to `results/<study>/` under the working directory; the capture
  directory comes from `--capture-dir` or `$HLMICRO_CAPTURE_DIR`.
* `hf-features` no longer takes column-name overrides.
* The REST collector skips empty or invalid books instead of writing rows with a
  zero mid; day files are compressed through a temporary file and an atomic
  rename.
