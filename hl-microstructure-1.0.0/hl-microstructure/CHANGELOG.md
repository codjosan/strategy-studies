# Changelog

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

### Verification

Each study was run against its standalone predecessor on the same synthetic
capture data (4 days, 2 coins, all file families). Trades, events, fold
statistics and output files match. The largest numeric difference is about 1e-7
relative, from parsing as float64 where some scripts used float32; no trade or
event changed as a result. The exceptions are the intentional changes below.

### Fixes

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
