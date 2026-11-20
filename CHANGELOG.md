# Changelog

## 1.1.0

Ten research scripts from the `hl-micro-strategies` collection brought into
the package, on the shared parse layer, cost model and test fixtures.

| Previous script | Now |
|---|---|
| `obi_multifold_cv.py` (v2) | `hlmicro threshold-cv` |
| `obi_oos_validator.py` | `hlmicro oos-split` |
| `obi_breakeven_exit.py` | `hlmicro breakeven-exit` |
| `obi_signal_conditional_exit.py` | `hlmicro signal-exit` |
| `obi_trend_gate_test.py` | `hlmicro trend-gate` |
| `obi_config_comparison.py` (v3) | `hlmicro obi-configs` |
| `tfi_signal_analysis.py` (signals 1 and 3) | `hlmicro tfi-signals` |
| `scalp_explorer_tick.py` | `hlmicro scalp-explore` |
| `fast_feature_explorer_v3.py` | `hlmicro feature-explore` |
| `capture_health.py` | `hlmicro inventory` health checks and `--check` |

### Not imported

| Script | Reason |
|---|---|
| `obi_tp_multifold.py` | Already `hlmicro tp-multifold` (1.0.0). |
| `scalp_multi_gate.py` | Already `hlmicro multi-gate` (1.0.0); this copy predates the trade-side fix. |
| `live_config_tfi_backtest.py` | Superseded by `obi_config_comparison.py`, whose first configuration per symbol is the same live configuration. |
| `tfi_signal_analysis.py`, signal 2 | Imbalance with a TFI gate at a fixed horizon; `obi-configs` runs the same gate with stops, take-profit and one position at a time. |
| `scalp_feature_explorer.py`, `scalp_fill_realism.py` | Minute-resolution predecessors of `scalp_explorer_tick.py`. The one tested feature the tick version lacked, `liq_spike_z`, is in `scalp-explore`. |
| `liquidation_extractor.py` | Identifies liquidations from counterparty wallet addresses, which this repository does not store. Its burst heuristic predates the `hl_liquidations` feed, which the liquidation-cascade study uses. |
| `strategy_explorer.py` | Its liquidation strategy needs the extractor's output; the 4-hour EMA trend strategy is outside the microstructure scope and, by its own warning, needs 200+ four-hour bars; the funding strategy reads a CSV from an external engine. |
| `fast_trigger_analyzer.py`, `fast_trigger_alt_analyzer.py`, `regime_drift_backtester.py` | Analyse `trigger_snapshots.csv` from a separate regime/trigger system; nothing in this repository produces or documents that file. |

### Shared code

* `hlmicro.validate.obi_signal`: one implementation of the imbalance z-score
  signal and its minute-bar simulator (stop, reversal, target on gross or net
  return, timeout with extension, cooldown, optional next-bar entry, positions
  open at the end). `tp-multifold` now uses it; its outputs on the synthetic
  data are byte-identical to 1.0.0.
* `hlmicro.validate.passive_fills`: the passive-limit fill simulation, shared
  by `multi-gate` (outputs unchanged) and `scalp-explore`.
* `hlmicro.parse.cross_venue_flow`: taker-flow imbalance per bucket on each
  venue and across venues.

### Fixes relative to the imported scripts

* **Trade-side labels.** The scalp scripts counted sells only for side `S`,
  but Hyperliquid labels them `A`, the defect fixed in `multi-gate` in 1.0.0:
  every TFI feature was +1 whenever a trade printed, and long liquidations
  (side `A`) were never counted. The TFI scripts accepted only `B`/`A` and
  silently dropped trades labelled `buy`/`sell`. All flow now goes through
  `aggressor_sign`.
* **Entry timing in `obi-configs` and `tfi-signals`.** Entries were priced at
  the opening of the trigger minute, before that minute's mean imbalance or
  flow was known. They are now priced at the minute's close.
* **Rolling flow windows in `scalp-explore`.** One tick without trades made
  the whole window undefined, which at 5-second ticks covered most 15-minute
  windows. Ticks without trades now count as zero flow.
* **Low-volume check.** `capture_health.py` compared the day in progress with
  full days, so a scheduled check failed every morning. The newest day is now
  left out of the comparison and of the median.
* **Permutation test in `trend-gate`** uses a generator seeded per run; the
  script's module-level generator made results depend on call order.
* Zero-variance z-score windows give NaN instead of +/-inf (`scalp-explore`).

### Behaviour changes relative to the imported scripts

* **Fees** come from `FeeModel` (per side: taker 4.32 bps, maker 1.44 bps;
  half-spread 0.14 bps). The scripts used 8.6 bps round trip (imbalance
  studies, feature explorer) or 13.72 / 7.15 / 0.58 bps (TFI studies). Every
  command takes fee overrides.
* Stops are negative percentages everywhere (`--stop-pct=-1.0`);
  `obi_config_comparison.py` used positive magnitudes.
* `feature-explore`: burst features (from the extractor's output) removed;
  liquidation features and `obi_qty_avg`, built but never used by any rule,
  are no longer computed; the documented but unimplemented three-feature stage
  is gone; the longest same-side run counts every trade in the minute (was the
  first 1,000); the large-trade cut-off is the exact 95th percentile (was
  estimated from every 100th trade).
* `threshold-cv`, `breakeven-exit`, `signal-exit` and `feature-explore` take
  `--next-bar-entry`; `oos-split` keeps next-bar entry by default
  (`--same-bar-entry` reverts).
* `signal-exit` reports take-profit exits as `tp` (was `TP`).
* Output goes to `results/<study>/`; file names no longer carry a run date.

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
