# Selective 15-minute direction research

The pipeline predicts whether each asset's final 15-minute close exceeds its
anchor open. Its objective is **accuracy among triggered windows**, subject to
at least 25% window coverage. This is symmetric green/red directional accuracy,
not the positive-class precision metric from a classification report. BTC, ETH,
and SOL have separate models and coverage constraints; a window means an
**asset/candle pair**, not a portfolio trade across all three assets.

## Run

From the `tradematic` checkout root:

```sh
uv sync --python 3.11
uv run python -m pytest -q
uv run ruff check backend/research
uv run python -m backend.research.run \
  --coverage-buffer 0.03 \
  --invalid-rows quarantine \
  --output backend/research/artifacts/experiment_001
```

The buffer is an optional, predeclared reserve: this command tunes for at least
28% coverage and still judges test eligibility at 25%. Omit it to optimize at
exactly the 25% validation floor. It cannot guarantee future coverage. No
threshold is refitted to test scores or labels. A frozen policy below 25% on a
test fold is explicitly **ineligible**. Increasing the buffer after seeing test
results would require a fresh, untouched test period.

On this Mac, LightGBM needs the existing OpenMP runtime in `/opt/anaconda3/lib`:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib uv run python -m pytest -q
DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib uv run python -m backend.research.run \
  --coverage-buffer 0.03 --invalid-rows quarantine \
  --output backend/research/artifacts/experiment_001
```

That environment setting only applies to the command. For another machine,
follow the [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).
No system package install is performed by this project. If a sandbox cannot
access the default uv cache, use `uv --cache-dir /private/tmp/tradematic-uv-cache`
in place of `uv` after syncing the environment.

Defaults: three expanding walk-forward folds, 28 test days each, 14 calibration
days, 14 threshold-tuning days, and a 120-minute gap between partitions. Each
fold fits 30 LightGBM classifiers (one per asset and horizon), with 160 trees,
15 leaves, minimum 80 observations per leaf, and L2 regularization. Hyperparameters
are fixed in advance. `--calibration isotonic` selects isotonic instead of Platt
calibration; this is an experiment choice, not an automatically selected test winner.
`--features-only` saves the audited data and snapshots without fitting.

## Observation contract and target

Input rows must be **start-stamped** minute bars. `dt` is parsed as UTC; naive
timestamps are assumed UTC. `dt` is authoritative, and `minute` must agree. The
provided `hour` can be a timestamp and is not assumed to be an integer. Invalid
OHLCV and missing symbols fail loudly by default. `--invalid-rows quarantine`
turns those whole bars into missing minutes, with counts and timestamps in the
audit. It does not repair their prices. Wrong nonempty symbols, off-grid times,
and duplicate timestamps always fail. The provided files contain one truncated
BTC OHLCV row and one SOL row with a missing symbol, so the example command
enables quarantine explicitly.

For a window starting at \(t_0\), row offset \(j=0,\ldots,14\) covers
\([t_0+j,t_0+j+1)\). That row's OHLCV becomes available at \(t_0+j+1\).
The loader exposes both `elapsed=j` and **`k=j+1` completed minutes**. Thus:

| Decision | Last observable row | Remaining minutes |
| --- | --- | --- |
| k=0 | Anchor open only; no minute OHLCV | 15 |
| k=1 | Offset 0, the first minute | 14 |
| k=10 | Offset 9, the tenth minute | 5 |
| Label | Offset 14 final close | 0 |

Using offset 10's close for a decision at k=10 would leak one minute. The final
label is \(Y=\mathbf1[C_{14}>O_0]\), available at \(t_0+15\).
Exact dojis are excluded explicitly; missing target minutes exclude the candle
from the evaluable universe. Counts are preserved in `data_audit.json` and
`window_audit.parquet`. Every remaining test candle counts in coverage, including
candles without ready features. Missing minutes are reindexed as NaN, never
filled with future or previous prices. The universe uses the common date span
of all three assets; BTC's extra tail outside ETH/SOL history is not evaluated.

Features receive bars only, never final closes, labels, or future completeness
flags. The separate label join happens in training. A streaming consumer can
call `build_features` on a prefix ending inside the current window; it does not
need that window's eventual label. At k=0, this implementation abstains and
reports zero actions; no fictitious anchor-only Brownian probability is trained.

## Mathematical blueprint

Write \(d_k=C_{k-1}/O_0-1\), \(r_t=\log(C_t/C_{t-1})\), and
\(\hat\sigma_0=\operatorname{sd}(r_{t_0-60:t_0-1})\), using at least 40
observations. This volatility is calculated from **pre-window data** and frozen
at the anchor. It is an online causal statistic, not a scaler fit over the
full dataset. A floor of \(10^{-7}\) avoids division by zero. No global imputation,
standardization, or volatility fitting uses validation or test rows.

| Family | Implemented statistic | Continuation / fade interpretation to test |
| --- | --- | --- |
| Anchor distances | \(z_k=d_k/(\hat\sigma_0\sqrt{k})\), \(b_k=d_k/(\hat\sigma_0\sqrt{15-k})\) | Large \(\lvert b_k\rvert\) means a reversal must cover more historical volatility units; the essential baseline edge. |
| Excursions | \(u_k=\max_{j<k}H_j/O_0-1\), \(a_k=1-\min_{j<k}L_j/O_0\); log ratio \(\log[(u_k+\epsilon)/(a_k+\epsilon)]\), balance \((u_k-a_k)/(u_k+a_k+\epsilon)\) | One-sided excursion plus price near its extreme supports continuation; a large excursion followed by retracement suggests fading. |
| Occupancy / chop | \(k^{-1}\sum_{j<k}\mathbf1[C_j>O_0]\), midpoint occupancy, nonzero sign transitions | Persistent occupancy and few crossings indicate a clean drive; repeated crossings indicate open-pinned chop. Zero closes carry the last nonzero sign for crossing counts. |
| Weighted geometry | Fit \(d_j=\alpha j+\beta j^2\), through \(d_0=0\), weighted by bar volume; velocity \(\alpha\), curvature \(2\beta\), endpoint velocity \(\alpha+2k\beta\), divided by \(\hat\sigma_0\) | Acceleration in the displacement direction supports continuation; deceleration/opposite curvature suggests exhaustion. k=1 uses linear displacement and zero curvature. |
| Path efficiency | Signed \(\log(C_t/C_{t-L})/\sum_{i=t-L+1}^{t}|r_i|\), L=5,10,15,30; analogous anchor-window statistic | High signed efficiency identifies directed paths; low absolute efficiency identifies two-way churn. |
| Variance ratio | \(\operatorname{Var}_{60}(\log(C_t/C_{t-5}))/(5\operatorname{Var}_{60}(r_t))\) | Values above/below one proxy persistence/reversion. Overlapping returns and a short sample mean this is not an inferential Lo–MacKinlay test. |
| Tail / wick signature | Trailing 15-return skewness and excess kurtosis; \(\log[\sum V(H-\max(O,C))/O\,/\,\sum V(\min(O,C)-L)/O]\) | Heavy upper wicks during a positive drive suggest rejected highs; lower wicks during a negative drive suggest rejection of lows. High kurtosis alone has no direction. |
| Previous extremum timing | Earliest high/low minute in the **previous complete** 15m candle | Late extremes may continue or exhaust; let training interactions with present displacement distinguish them. Incomplete previous candles yield NaN. |
| Cross-asset alignment | Same-side flag; sum/dispersion of \(b_k\); each peer's \(b_k\); peer mean minus own \(b_k\) | Synchronized moves can reinforce continuation. A lagging asset may catch up, or reflect asset-specific weakness; alignment alone does not establish causality. |
| Beta residuals | \(\beta_{ab,t}=\operatorname{Cov}_{120}(r_a,r_b)/\operatorname{Var}_{120}(r_b)\), shifted one bar; \((R_{a,L}-\beta R_{b,L})/(\hat\sigma_a\sqrt L)\), L=1,3,5; lagged peer returns | A negative residual with positive peer displacement is a candidate catch-up signal; persistent negative residuals may indicate genuine divergence. Betas require at least 60 prior pairs and are clipped to ±5. |
| Volume synchrony | Trailing prior-60 log-dollar-volume z-score; count above 2; entropy of cross-sectional positive-z shares | Common volume ignition can confirm a market-wide drive; concentrated ignition may identify a leader or an idiosyncratic shock. |
| Flow / pressure | \(\delta_j=V_j(2C_j-H_j-L_j)/(H_j-L_j+\epsilon)\); cumulative delta, delta/volume, trailing-15 delta and difference from pre-window delta fraction | Flow aligned with displacement supports continuation; price/flow disagreement suggests fading or absorption. Flat bars contribute zero signed flow. |
| Impact | Mean \(\lvert r\rvert/(VC)\) over ten bars; \((I_+-I_-)/(I_++I_-+\epsilon)\) | Direction-dependent thinness can amplify a move or reveal fragile price impact. Dollar volume makes the three asset scales more comparable. Zero-volume impacts remain missing. |
| Clock structure | UTC day/week sine/cosine; quarter-hour position; top-of-hour flag; DST-aware US-open clock; historical CME-close clock; 00/08/16 UTC funding-clock distance | Liquidity regime interactions, without claiming that a time flag causes continuation or fading. |

Rolling efficiency, variance ratio, skewness, and wick pressure are also frozen
from just before the window as separate pre-window signatures. Other rolling
statistics update using only information available at the decision time.

The signed-volume feature is a **CLV pressure proxy**, not measured aggressive
order flow or genuine VPIN. Amihud and impact asymmetry are OHLCV proxies, not
estimated structural Kyle lambda. Minute OHLCV cannot reveal event order,
spread, queue depth, aggressor identity, or within-minute crossings.

Time flags are deliberately clock proxies rather than full exchange calendars.
US-open flags respect New York DST around 09:30 weekdays, but not holidays or
special sessions ([NYSE hours](https://www.nyse.com/trade/hours-calendars)).
The historical CME 16:00 Chicago close flag is disabled after the
[May 29, 2026 transition to 24/7 crypto trading](https://www.cmegroup.com/media-room/press-releases/2026/6/01/cme_group_announceslaunchof247cryptocurrencyfuturesandoptionstra.html).
Funding flags describe conventional eight-hour boundaries, not actual Coinbase
spot funding events or a venue-specific perpetual schedule.

## Baseline, training, and stopping

Under zero-drift Gaussian independent future increments,
\[
P(Y=1\mid\mathcal F_{t_0+k})\approx\Phi(b_k).
\]
This is a remaining-increment Gaussian/barrier-distance proxy, often informally
called a Brownian-bridge baseline. A literal bridge conditions on a known
endpoint, which is not available here. The probability of **ever crossing**
the anchor before expiry is also different from the terminal-color probability.
The baseline always follows current displacement and ranks \(\lvert b_k\rvert\).
Native distances are retained for exact ranking to avoid Gaussian CDF saturation.

Each fold has four ordered partitions:

1. **Train:** fit asset/horizon-specific LightGBM models on ready snapshots.
2. **Calibration:** fit Platt logistic regression on raw model log odds, or
   isotonic regression on raw probabilities. Models are not retrained on this
   partition. This separation follows the
   [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration).
3. **Policy tuning:** learn confidence cutoffs and choose the fixed horizon.
4. **Test:** replay the frozen models, calibrators, horizon, and thresholds once.

Each partition retains only windows whose labels finish before its end; windows
crossing boundaries are purged. A 120-minute gap exceeds the 15-minute target
horizon. Past prices from embargo periods can be used for causal test features;
their labels are never used to fit the preceding partition. Later folds may
train on earlier test outcomes after those labels become historical. All three
assets share time boundaries, preventing the same timestamp's labels entering
training for another asset while it is tested.

For calibrated probability \(p_k\), confidence is \(q_k=\max(p_k,1-p_k)\).
At fixed k, enumerate every distinct tuning confidence cutoff, admitting all
ties, and maximize \(\sum\mathbf1[\hat Y=Y,A]/\sum\mathbf1[A]\) subject to
\(\sum\mathbf1[A]\ge\lceil cN\rceil\). Evaluate each k=1..10 independently.
Choose \(k^*\) on tuning precision, breaking ties toward earlier k. The test
frontier is descriptive; its winner is not retroactively selected as \(k^*\).

The dynamic policy uses
\[
T=\inf\{k\in\{1,\ldots,10\}:q_k\ge\tau_k\},
\quad A=\mathbf1[T\le10].
\]
If the set is empty, abstain. A window locks at its first passage. An unused
minute has threshold infinity (serialized as JSON `null`). The threshold vector
is optimized on **window** outcomes and first-passage coverage, using a bounded
25-quantile grid, three initializations, and up to three coordinate passes.
This is a local search, not a proof of globally optimal stopping. A dynamic
policy may correctly degenerate to a single fixed horizon if earlier action
does not improve its objective. The distance baseline receives the same fixed
and dynamic policy search, so dynamic gains are not credited merely for having
more chances to cross a threshold.

Thresholds learned from a finite historical distribution cannot guarantee 25%
coverage under an arbitrary future distribution. A requirement to guarantee it
would need an explicit online quota/fallback policy that changes the requested
abstention rule, or retrospective test-distribution cutoffs. Neither is silently
added. The code reports failures rather than calling them valid results.

## Outputs and interpretation

All generated outputs are ignored by git; the source and uv lockfile are portable.
Existing experiment directories are never overwritten.

| Output | Meaning |
| --- | --- |
| `report.md` | Frozen-policy results, retrospective identical-25% frontier, limitations |
| `policy_metrics.csv` | All fixed k, validation-selected k*, dynamic, baseline; pooled and per-fold coverage eligibility |
| `matched_25_frontier.csv` | Same ceil(25% × windows) count for ML and distance at every fixed k |
| `precision_differences.csv` | ML-minus-distance precision and seven-day block-bootstrap descriptive 95% interval |
| `trigger_minute_metrics.csv` | Actual triggered-window accuracy and frequency at k=0..10; zero at k=0 |
| `calibration_metrics.csv` | Test Brier score, log loss, all-ready directional accuracy by asset/fold/k |
| `shap_triggered.csv` | TreeSHAP summaries from up to 1,000 triggered windows per selected policy/asset/fold |
| `decisions.parquet`, `oos_probabilities.parquet` | Replayable per-window actions and per-minute test scores |
| `matched_25_decisions.parquet` | Retrospective ranked diagnostic selections, explicitly separate from actions |
| `features.parquet`, `window_audit.parquet` | Causal snapshots and separate target/data-quality universe |
| `models/`, `policies.json` | Fold-specific estimators, calibrators, fixed selections, and learned stopping thresholds |
| `manifest.json`, `folds.json`, `data_audit.json` | Parameters, versions, input/source SHA-256s, split bounds, missing/doji counts |

Exact test top-25% ranking is useful for measuring discrimination at identical
coverage, but its cutoff knows the test score distribution. It is **not a
deployable policy**. It uses no test outcomes for ranking and is never used to
repair frozen coverage. Bootstrap comparisons of frozen policies may have
different coverage; consult the exact-25% rows for count-matched comparisons.
Confidence ties in diagnostic rankings use chronological order, whereas causal
thresholds admit all equal-confidence windows.

The seven-day circular moving-block bootstrap resamples the same calendar-day
blocks for both policies, carrying daily hits and trigger counts. It respects
short-term serial dependence better than treating 1m snapshots or the three
assets as independent trials. Intervals remain descriptive, unadjusted for
multiple horizons/assets, and sensitive to block length. Positive absolute
precision, or a favorable unadjusted interval at one cherry-picked k, does not
establish alpha. Require valid coverage, improvement over distance at matched
coverage, fold stability, and new untouched replication.

Native [LightGBM TreeSHAP](https://lightgbm.readthedocs.io/en/stable/Python-API.html)
explains the underlying model's raw log odds. A separate Platt/isotonic mapping
does not preserve that additive interpretation on calibrated probability.
Samples are drawn without reference to correctness and include winners and
losers. No feature-importance ranking establishes economic causation.

The tests perturb every future OHLCV value across all assets and compare every
available feature; they also replay truncated prefixes, remove minutes, inject
dojis/duplicates, test threshold ties and first-passage locking, and check purge
and embargo boundaries. They do not validate live bar arrival latency. Labels
describe candle color relative to the old anchor; a correct late prediction
does not imply a profitable trade from the current price.

## SOL signals over Kalshi quotes

Replay the existing frozen out-of-sample SOL actions without retraining:

```sh
uv run python -m backend.research.kalshi_overlay \
  --research backend/research/artifacts/walk_forward_v1_audited \
  --candles backend/data/kalshi/candles_SOLUSD.csv \
  --markets backend/data/kalshi/markets_SOLUSD.csv \
  --output backend/research/artifacts/kalshi_sol_overlay_v1
```

`markets_SOLUSD.csv` supplies actual settlement outcomes and UTC market opens;
Coinbase candle labels are not substituted for Kalshi results. Green signals
buy YES at `yes_ask`; red signals buy NO at `no_ask`. Positions are held to
settlement. The default is one contract with a 0.07 taker-fee coefficient and
conservative whole-cent rounding per order. `--quantity`, `--fee-rate`, and
`--fee-rounding` make these assumptions explicit; the CSV does not establish
the historical fee schedule or account-specific charges.

To expose fee-rounding sensitivity, repeat the command with
`--fee-rounding 0.0001` and a fresh output directory, such as
`backend/research/artifacts/kalshi_sol_overlay_fine_fees`. This compares the
same frozen actions and quotes under finer fee rounding; it does not establish
which historical/account fee schedule applied. Read the gross result and both
fee scenarios together.

The overlay compares ML and distance fixed-selected/dynamic policies. It
reports all signals and a separately labeled positive-proxy-EV filter, with
0/1-minute execution-delay and 0/1/2-cent adverse-price sensitivities. The EV
filter compares the existing Coinbase probability to the Kalshi ask plus fees;
it is not a probability calibrated for Kalshi, and it can reduce coverage below
25%. Rejected signals never trigger a second time. Retrospective top-25%
rankings are excluded from the trading overlay.

Exact timestamp joins never fill missing quotes or use a later quote at the
original signal time. Results are hypothetical quote-level settlement
markouts: these files lack resting sizes, queue history, bar publication
latency, and confirmed fills. Reports preserve gross/net P&L, fee assumptions,
settlement-label differences, execution coverage, missing quotes, and skip
reasons. `summary.csv` includes per-fold results and seven-day bootstrap
intervals; `aligned_decisions.parquet` is the full audit trail.
