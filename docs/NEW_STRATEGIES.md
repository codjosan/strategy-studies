# Candidate strategies for the capture dataset

Seven new strategy hypotheses designed for the data we have, implemented in
`hlmicro/validate/candidates.py` and tested in the run-all framework: one
cost model, t-statistic ranking, Bonferroni verdicts, and half-period checks.

```bash
python -m hlmicro candidates --capture-dir capture_data_clean          # candidates only
python -m hlmicro run-all   --capture-dir capture_data_clean           # together with every study
python -m hlmicro candidates --only vol_norm_dip,momentum --symbols ETH,BTC,LINK
```

## What the dataset allows

| Data | Period | Used for |
|---|---|---|
| `obi_5s`: mid, top-of-book, depth, imbalance for 6 coins x 4 venues | 18 Jul - 27 Sep 2026 (72 days) | all candidates |
| `hl_trades`: Hyperliquid tape, aggressor side | 25 Aug - 27 Sep 2026 (34 days) | flow, VWAP and passive fills |
| Not available: funding history, liquidations, websocket books, CEX tapes | | |

## Where the ideas come from

| Earlier finding | Candidate |
|---|---|
| The only robust edge was ETH `vwap_dev_buy`: buying a 20 bps dip below the 5-min VWAP. It failed on other coins, where a fixed 20 bps means a very different move. | **vol_norm_dip** |
| `cex-hl-lag`: HL follows Binance. Gross +3.7 bps median, +7.2 on ETH, but a taker pays 8.64. | **passive_leadlag**, **basis_reversion** |
| Following trade bursts (`tick_speed`) and OBI momentum lost money. | **flow_fade** (the opposite trade) |
| Nothing tested relative value, multi-hour trends or the hourly funding cycle. | **pairs**, **momentum**, **funding_proxy** |

## Common rules

* One aligned 5-second grid per coin. Signals use data up to the signal tick
  only; entry is at the next tick; positions never overlap. A test checks
  that signals computed on truncated data equal the full-data signals, and it
  catches a planted 5-tick leak.
* Costs, as everywhere in the framework:
  * taker: mid fills and an 8.64 bps round trip (two legs for pairs)
  * passive: a limit inside the mid, filled when the HL tape trades through it
    within one tick, maker fee; the exit falls back to taker at the mid
* Parameter grids are small and were fixed before the data was run. Every
  (candidate, parameters, coin) is one strategy in the leaderboard.

## The candidates

### 1. basis_reversion: HL vs CEX dislocations converge
* **Hypothesis.** When the HL mid strays from the median Binance/OKX/Bybit mid
  beyond its usual offset, the gap closes (the lag seen in `cex-hl-lag`).
* **Rule.** basis = HL / median(CEX) - 1. dev = basis - trailing 1 h mean.
  Buy HL when dev <= -D, sell when dev >= +D (first tick). Exit after H, or
  when dev returns to 0 (at most H).
* **Grid.** D in {5, 10, 20} bps; H in {1, 5, 15} min; exit in {time, revert}. Taker.
* **Supports it.** Positive net mean in both halves, on more than one coin.

### 2. passive_leadlag: the lead-lag edge with maker execution
* **Hypothesis.** The +3.7 bps lag capture clears a maker cost (2.88 bps)
  but not a taker cost.
* **Rule.** Binance moved >= X bps over W while HL moved less than half of
  that. Post an HL limit in the move's direction at the mid (or k bps
  inside). Exit after H.
* **Grid.** X in {10, 15, 20} bps; W in {30, 60} s; H in {30, 60, 300} s; k in {0, 1} bp.
* **Main risk.** Adverse selection: a resting order fills mostly when HL does
  *not* follow.

### 3. flow_fade: fade one-sided trade bursts
* **Hypothesis.** A burst of aggressive selling (buying) that has already
  moved the price is an exhaustion point, so the move partly reverts.
* **Rule.** Trade count z-score (15 min) >= 2, 60 s flow imbalance
  <= -T (>= +T) and price down (up) >= D bps in 60 s: buy (sell). Exit after H.
* **Grid.** T in {0.5, 0.7}; D in {15, 30} bps; H in {5, 15, 30} min; taker or passive 1 bp.

### 4. vol_norm_dip: the ETH dip-buy, scaled to each coin
* **Hypothesis.** `vwap_dev_buy` works when the dip is large *for that coin*.
  A threshold in standard deviations should carry across coins where 20 bps
  did not.
* **Rule.** dev = (mid - VWAP 5 min) / VWAP, divided by the trailing 24 h
  standard deviation of dev. Buy when that is <= -K, with flow >= 0 and HL
  imbalance >= 0 (as the original). Passive limit k bp. Exit after H.
* **Grid.** K in {2, 3, 4}; H in {30, 60} min; k in {1, 2} bp. Long only.

### 5. momentum: hourly breakouts, follow or fade
* **Hypothesis.** Large multi-hour moves continue (or revert) over the next hours.
* **Rule.** At each full hour: the L-hour return against k x its trailing
  7-day standard deviation (hourly samples). `follow` trades in the move's
  direction, `fade` against it. Exit after H.
* **Grid.** L in {1, 4} h; k in {1, 2}; H in {1, 4} h; mode in {follow, fade}. Taker.

### 6. funding_proxy: trading into the hourly settlement
* **Hypothesis.** HL funding follows the HL premium. Before a settlement with
  a high premium, longs close to avoid paying funding, and that selling
  reverses afterwards. We have no funding history, so the last hour's mean
  HL-vs-CEX basis stands in for the funding rate.
* **Rule.** Rank that premium within its trailing 7 days: short the top p%,
  long the bottom p%, from T - lead to T + 5 min.
* **Grid.** p in {10, 20}; lead in {5, 10} min. Taker.
* **Caveat.** A proxy. The funding CSVs (or API access) would allow the
  real `funding-settlement` study.

### 7. pairs: ratio mean reversion
* **Hypothesis.** Large deviations of log(A/B) from its recent mean revert.
* **Rule.** z of log(A/B) on minute samples against the trailing L-hour mean
  and std. Short A / long B when z >= Z, the reverse when z <= -Z. Exit when
  z crosses 0, at most H.
* **Grid.** L in {6, 24} h; Z in {2, 3}; H in {4, 24} h; pairs ETH/BTC,
  SOL/ETH, SOL/BTC. Two taker legs (17.28 bps).

## First run (capture_data_clean, 6 coins)

684 strategies, 586 with at least 30 trades ranked. **None is significant or
promising** after costs and the correction for multiple tests.

| candidate | ranked | median net bps | median gross bps | positive in both halves | best (t) |
|---|---:|---:|---:|---:|---|
| **vol_norm_dip** | 39 | **+1.6** | (passive) | **11** | HYPE 3 sd/30m +29.8 (1.78); ETH 3 sd/60m +19.2 (1.76) |
| momentum | 96 | -8.6 | 0.0 | 4 | LINK follow 1h >= 2 sd +18.8 (1.52) |
| basis_reversion | 102 | -6.3 | +2.4 | 5 | HYPE dev 20, exit at mean +4.3 (0.94) |
| passive_leadlag | 157 | -7.7 | (passive) | 4 | BTC 10 bps/60 s, 300 s +11.6 (1.70), first half -1.4 |
| flow_fade | 144 | -8.3 | (mixed) | 1 | BTC +8.2 (0.79) |
| funding_proxy | 24 | -7.7 | +0.9 | 0 | all negative |
| pairs | 24 | -16.0 | +1.3 | 0 | ETH/BTC +3.8 (0.24) |

What this says:

* **vol_norm_dip is the lead worth pursuing.**
  * It is the only candidate with a positive median.
  * It takes 11 of the top 13 places and has 11 configurations positive in
    both halves.
  * At 3 sd / 60 min / 1 bp it is positive on 4 of 6 coins: ETH +19, LINK
    +21, HYPE +16, BTC +5 bps. The fixed-20 bps rule was positive only on
    ETH, SOL and HYPE (HYPE +2.7), and negative on DOGE and LINK.
  * Samples are small (29-44 trades per coin).
* **momentum (follow a >= 2 sd 1-hour move, hold 1 h)** works on LINK, ETH
  and BTC (+7 to +19 bps net, positive in both halves) but loses badly on
  DOGE and HYPE. It may depend on the coin, or be noise.
* **basis_reversion** does converge (up to +13 bps gross), but below the
  taker cost. It only becomes viable with cheaper execution or larger
  dislocations.
* **passive_leadlag** confirms the adverse-selection risk: maker orders fill
  mainly when HL does not follow.
* **funding_proxy** and **pairs** are not viable here: a weak proxy, and the
  cost of two legs.

## Step 1: vol_norm_dip validation (`scripts/vol_norm_dip_validate.py`)

* **Grid.** Fixed before the run: dip {2.5, 3, 3.5, 4} sd x hold {45, 60, 75,
  90} min x VWAP {2, 5, 10} min x limit {1, 2} bp, for all 6 coins.
* **Halves.** The 34-day tape period is split in two equal halves at 11 Sep.
* **Output.** `results/vol_norm_dip_validation/report.html`.

| check | result |
|---|---|
| common setting chosen on the **first** half, scored on the second | **3 sd, 75 min, VWAP 10 min, 2 bp: +24.9 -> +23.4 bps (t 2.24), 6/6 coins positive** |
| common setting chosen on the second half, scored on the first | 4 sd, 90 min, VWAP 2 min, 1 bp: +53.3 -> +0.5 (4/6 coins positive) |
| best setting per coin instead of one common setting | +6.3 and +0.2 out of sample: per-coin tuning overfits |
| grid robustness | rank correlation of the halves +0.45; deeper dips steadily better (2.5 sd about +1 to +6, 4 sd about +25 to +38) |
| second half vs first | 97% of settings positive vs 60%: the second half favoured dip-buying in general |

**Locked setting.** Chosen by a rule fixed in advance: the best worst-half
pooled mean, with at least 60 trades per half. It is **3 sd below the 10-min
VWAP, passive 2 bp limit, hold 75 min, all 6 coins**, the same setting the
first-half selection picked.

| | locked | first-run setting (3 sd, 60 min, VWAP 5, 1 bp) |
|---|---|---|
| trades / mean / median | 139 / **+24.1** / +28.3 bps | 208 / +8.8 / +12.5 |
| t, p | **3.47, 0.0003** | 1.46, 0.07 |
| 95% CI (day bootstrap) | **+8.3 .. +37.3** | -6.4 .. +23.2 |
| coins positive | **6/6** (+16 to +37 each) | 4/6 |
| without the best 3 trades | +18.4 | +5.1 |
| strict fills (one increment through) | +24.2 | +11.7 |
| placebo, random entries | -3.8 (p < 0.002) | -4.6 (p = 0.002) |
| dips during a liquidation burst / others | +44.5 (n 14) / +21.8 | +28.3 (n 18) / +7.0 |
| without the best day / best 3 days | +20.1 (t 3.02) / +15.4 (t 2.25) | |
| positive days | 23 of 31 | |

**What it means.** This is the strongest result so far. It holds out of
sample in one direction, is positive on every coin, beats random entries and
survives strict fills. Its t of 3.47 also passes a Bonferroni correction for
the 96-setting grid. It has not passed the same test in the other direction,
and it still rests on 34 days. The forward test below is the real check.

## Correlation / cointegration study (`scripts/cointegration_study.py`)

Output: `results/cointegration_study/report.html`. Strategies: leaderboard
top 5 and the locked vol_norm_dip.

**Structure of the data (72 days):**

* **HL vs CEX, same coin.** Cointegrated for every coin (Engle-Granger p ~ 0,
  hedge ratio 1.00). The basis is 2-4 bps with a 3-5 bps sd; half-life 30-112 s.
* **Coin pairs.** Highly correlated (5-min returns 0.50-0.85; HYPE the least),
  but not stably cointegrated:
  * 5 of 15 pairs pass at 5% over the full period, none in both halves
  * half-lives of 30-125 hours
  * Johansen finds at most one cointegrating vector, barely

**Does it reinforce the signal?** Features at the signal tick, trailing data
only:

| feature | top-5 (ETH 30 m variants) | locked vol_norm_dip |
|---|---|---|
| market-wide dip (other coins' 15-min return < 0) | weak, one half only | **reinforces**: low third +48 vs high third +10 bps, rho -0.29, p 0.001, both halves |
| coin cheap vs partner (3-day cointegration residual z) | weak, one half only | **reinforces**: +30 vs +14, rho -0.18, p 0.04, both halves |
| high correlation with partner (6 h) | none | reinforces: +36 vs +9, rho +0.20, p 0.02 |
| coin-specific drop (own minus market) | none | opposite: coin-only drops revert less |
| HL-only dip (basis vs CEX) | none | none |

The dip-buy works when the whole correlated market sells off and the coin is
cheap against its long-run partner. A coin falling on its own (possibly news)
reverts less. This agrees with the liquidation diagnostic.

**Filters with thresholds fixed at 0 (not tuned), locked vol_norm_dip:**

| filter | n | mean | t | first / second half |
|---|---|---|---|---|
| none | 138 | +24.0 | 3.43 | +24.9 / +23.2 |
| market_15m < 0 | 118 | +27.1 | 3.55 | +29.4 / +25.3 |
| coint_z < 0 | 84 | +30.7 | 3.47 | +32.1 / +29.7 |
| both | 72 | **+34.8** | 3.63 | **+39.3 / +30.9** |

* **Top 5: not reinforced.** The filters help in one half and hurt in the
  other, and #3 and #5 get worse.
* **Strategy P&L.** The four ETH top-5 variants are one bet (daily P&L
  correlation 0.93-0.99). vol_norm_dip is only weakly correlated with them
  (0.04-0.37).
* **Forward test.** The filters were found on this data, so the filtered
  rule is registered as a **secondary hypothesis** in `forward_lock.json`
  (the primary rule is unchanged). `scripts/forward_test.py` reports both.
  The features are computed by
  `hlmicro.validate.candidates.context_features`, with a look-ahead test
  that catches a planted 1-tick leak.

## Step 2: momentum on LINK/ETH/BTC: waiting for new data

Every upload ends on 27 Sep 2026, including the liquidation files, and every
day up to then was already used. So there is no unseen data for this test
yet. Both strategies are therefore **locked** in
`results/vol_norm_dip_validation/forward_lock.json`, and
`scripts/forward_test.py` evaluates them on any captures after 27 Sep, with
no parameter changes:

* **vol_norm_dip** with the locked setting on 6 coins: about 4 trades a day,
  so 30 trades (the minimum) take about 8 days of captures.
* **momentum**, follow a 1 h move >= 2 sd, hold 1 h, on LINK/ETH/BTC: about
  2.6 trades a day, so about 12 days.
* **Pass.** Pooled mean > 0 with one-sided p < 0.05 and at least 30 trades.

```bash
python -m hlmicro sanitize --capture-dir capture_data --out capture_data_clean   # after adding new files
PYTHONPATH=. python scripts/forward_test.py
```

## Liquidation data (26 Aug - 27 Sep 2026)

* **Integrity.** The uploaded `hl_liquidations` files are consistent: every
  print is also in the HL trade tape with the same trade id, price, size,
  side and time. Liquidations are 3-15% of traded notional. The `liq_user`
  column was empty in all 642,823 rows and is removed by `sanitize`.
* **The liquidation-cascade study** (`liquid_study/`, fee 8.64 + 2 bps
  slippage) finds nothing that pays: fading or following bursts loses 5-19
  bps at every horizon. No (setting, coin, direction, horizon) cell passes.
  Its sanity check (price moves with the cascade during bursts, +15.9 bps)
  holds. Output: `results/liq_cascade/report.txt`.
* **Link to vol_norm_dip.** Dips during a liquidation burst rebound more.
  That diagnostic is not part of the locked rule.

## Adding a candidate

Write a function `fn(grid, **params) -> [(entry_tick, net_bps), ...]` that
builds its signals with `_edges`, `_ret`, `_rolling` and `_basis`, and
executes with `taker_fixed`, `taker_until` or `passive`. Then register it:

```python
Candidate("my_idea", my_idea, {"param": [1, 2, 3]}, needs_tape=False,
          cost=f"taker {TAKER_RT:.2f} bps RT", describe=lambda p: f"param {p['param']}", cost_bps=TAKER_RT)
```

`tests/test_candidates.py` runs every registered candidate through the
look-ahead test automatically.
