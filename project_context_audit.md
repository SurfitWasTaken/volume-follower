# Volume Spike Directional Prediction Pipeline — Full Project Context Audit

---

## 1. PROJECT OVERVIEW

**What it does:** A backtesting pipeline that detects abnormal volume spikes on forex OHLCV candles and tests whether following the spike direction (long on bullish, short on bearish) produces a statistically significant edge over the unconditional base rate — after accounting for transaction costs, multiple-testing correction, and temporal stability.

**Target instrument / asset class:** EUR/USD forex (primary), with a cross-currency extension using GBP/USD as a secondary signal. Config supports SPX500, NAS100, and XAU CFDs but these are not actively tested. Data sourced from **OANDA v20 REST API (practice account)** — tick volume only, not real exchange volume.

**Core alpha hypothesis:** Volume spikes reveal informed-flow activity. A candle with volume statistically anomalous relative to recent history, filtered by candlestick conviction (body ratio, wick asymmetry), session timing (London/NY opens), news exclusion, and trend context, may predict directional continuation over the next K candles.

**Intended execution context:** Research / backtesting engine. The pipeline is explicitly **not** a live trading system. Previous conversations explored holdout validation and live-trading readiness planning, but no execution or order management code exists.

---

## 2. CURRENT IMPLEMENTATION STATUS

| Component | Status | What Is Implemented | What Is Missing / Deferred |
|---|---|---|---|
| **Data Ingestion** | ✅ Complete | OANDA v20 API fetcher with chunked pagination, CSV cache layer, max-history fetch (up to 5yr), density validation, data quality checks (NaN, duplicates, zero-volume, OHLC violations, gap detection), IS/OOS time split | No streaming/live data feed. Relies on practice-account tick volume (~0.7–0.85 correlation with real volume). |
| **Signal Generation** | ✅ Complete | 4 variants: A (rolling Z-score), B (volume multiplier), C (percentile rank), D (adaptive session-normalised Z-score). Minimum gap enforcement (10 candles) to prevent overlapping outcomes. Expanding-window session normalisation (no look-ahead). | No ML/ensemble signal weighting. |
| **Signal Filtering** | ✅ Complete | 7 cumulative filters: direction classification, body-ratio, wick asymmetry, session window (London/NY/both), news exclusion (hardcoded calendar), trend context (MA50/MA200), range position (breakout vs mid). Cross-currency EUR-driven filter. | News calendar is hardcoded (static list of NFP/CPI/FOMC/ECB events through early 2026) — no live calendar integration. |
| **Outcome Calculation** | ✅ Complete | ATR-based TP/SL (1.5 ATR target, 1.0 ATR stop), MFE/MAE, hold-to-close returns, per-trade cost modelling (OANDA spread + widening factor). Entry at next candle's open (no look-ahead). Supports adaptive position sizing via cross-currency features. | No slippage model beyond spread. |
| **Statistical Tests** | ✅ Complete | Session-matched base rate, one-sided binomial test, Cohen's h effect size, Benjamini-Hochberg FDR correction, Bonferroni correction, block bootstrap CIs (preserves serial correlation), walk-forward stability assessment, permutation test (volume-shuffle), stationarity check (half-split comparison). | No HAC/Newey-West standard errors explicitly. Block bootstrap partially addresses this. |
| **Risk Management** | 🔧 Partial | Red flag system with 8 checks: insufficient N, suspicious Sharpe, WR CI below base rate, non-stationary edge, permutation test failure, trivial effect size, extreme WR shift. Cost viability pre-flight gate. | No live risk controls (drawdown kill-switch, position limits, emergency unwind). No exposure limits. |
| **Performance Analytics** | ✅ Complete | Summary tables (CSV + markdown), equity curves with drawdown shading, return distribution overlays, walk-forward Sharpe box plots, permutation test histograms, MFE/MAE scatter, session heatmaps, structured conclusion reports. All n-gated (min 100 for charts, 150 for reportable results). | No real-time dashboarding. Previous conversations explored a candlestick visualisation dashboard (separate system). |
| **Pre-Commitment / Anti-P-Hacking** | ✅ Complete | SHA-256 hash of full test matrix locked before data loading. Verification at run end detects contamination. Pipeline checkpoint logging with JSON audit trail. | No external audit signing. |
| **Cross-Currency Extension** | ✅ Complete | EUR-move scoring (EUR/USD − GBP/USD return), rolling correlation regime detection (HIGH/MED/LOW), GBP/USD volume Z-score, adaptive position sizing by regime and EUR-move percentile. Parameter sweep grid. | CC extension results show only 4 signals on H1 with critical red flags — effectively non-viable. |
| **Execution / OMS** | ❌ Not Started | None | No order management, execution routing, fill simulation, or broker integration. |
| **Deployment / Infra** | 🔧 Partial | `.gitignore`, `.env` pattern for API keys (from security audit conversation), `requirements.txt`. | No CI/CD, containerisation, or monitoring infrastructure. |

---

## 3. TECHNICAL IMPLEMENTATION DETAILS

### a) Data Layer

- **Source:** OANDA v20 REST API (practice environment)
- **Frequency:** M1, M5, M15, H1, H4 (primary runs use M5 and H1)
- **Fields:** Open, High, Low, Close, Volume (tick volume, not real volume)
- **Data range:** 2023-01-01 to 2026-03-01 (~3 years)
- **Cleaning:** Automatic validation for NaN counts, duplicate timestamps, zero-volume candles, OHLC constraint violations (H < max(O,C), L > min(O,C)), and gap detection
- **Storage:** CSV cache in `data/` directory (e.g., `EUR_USD_M5.csv`)
- **IS/OOS Split:** 2023-01-01 → 2025-03-01 (in-sample), 2025-03-01 → 2026-03-01 (out-of-sample), with 50-candle blackout buffer at boundary

> [!WARNING]
> OANDA tick volume is a proxy with ~0.7–0.85 correlation to real exchange volume. The entire alpha hypothesis rests on volume information content that may be partially absent from this data source.

### b) Signal / Alpha Model

**Four detection variants, all operating on the `volume` series:**

| Variant | Formula | Default Params |
|---|---|---|
| **A — Z-Score** | Fire when `(V_t − μ) / σ > z_threshold` over rolling `lookback` candles (current excluded) | lookback=20, z_threshold=2.0 |
| **B — Multiplier** | Fire when `V_t > multiplier × rolling_mean(lookback)` | lookback=20, multiplier=2.5 |
| **C — Percentile** | Fire when `rank(V_t) / window_size > percentile_threshold / 100` in rolling window | lookback=20, percentile=95 |
| **D — Adaptive** | Session-normalise volume by expanding hour-of-day mean, then apply Z-Score | lookback=20, z_threshold=2.0 |

- **Session normalisation:** Available for A/B/C as `_sn` suffix. Uses expanding-window (not rolling) hour-of-day mean — verified look-ahead-free.
- **Minimum signal gap:** 10 candles between successive signals — prevents overlapping outcomes and serial correlation.
- **Direction:** Classified by candle close vs open. Bullish (close > open) → +1, Bearish → −1.
- **Lookback windows:** All default to 20 candles. Justified as ≈1 trading day on M5 (20 candles × 5 min = 100 min ≈ session segment).
- **Feature engineering:** Cross-currency features include EUR-move score (`EUR/USD return − GBP/USD return`), rolling 20-candle pair correlation (.shift(1) to prevent look-ahead), GBP/USD volume Z-score.

### c) Position Sizing & Portfolio Construction

- **Methodology:** Regime-based multipliers from cross-currency features:
  - HIGH_CORR regime: 1.5× base
  - MED_CORR regime: 1.0× base
  - LOW_CORR regime: 0.5× (or skip entirely if `cc_skip_low_corr_regime=True`)
  - Additional scaling by EUR-move-score percentile rank (expanding window, look-ahead-free)
  - Penalty for high GBP/USD volume at signal time (0.5× if Z > 2.0)
- **Constraints:** Position size capped at [0.25, 2.0]× base size
- **Rebalancing:** Per-signal sizing; no portfolio-level rebalancing (single-instrument, single-position model)
- **Leverage limits:** Not implemented — this is a backtesting engine, not a position management system

### d) Execution Model

- **Order types:** None implemented — backtesting only
- **Entry assumption:** Next candle's open after signal candle (one-bar delay, no look-ahead)
- **Slippage model:** Spread-only (OANDA typical spreads in pips: EUR/USD=1.0, GBP/USD=1.4). A `spread_widening_factor` of 1.5× is applied during session opens. `slippage_atr_fraction` is set to 0.0 (cost fully embedded in spread).
- **Fill rate:** 100% assumed (market orders implied)
- **Latency:** Not modelled

### e) Risk Controls

**Backtesting-level (implemented):**
- Cost viability gate: rejects instrument×timeframe combos where break-even win rate ≥ 60%
- Red flag system: 8 automatic checks with CRITICAL/WARNING severity levels
- Pre-commitment hash verification to detect mid-run test matrix modification
- Sample size estimation with hard halt at < 150 signals

**Live-trading-level (NOT implemented):**
- No drawdown kill-switch
- No gross/net exposure limits
- No correlation or concentration risk controls
- No emergency unwind procedures

---

## 4. VALIDATION & ROBUSTNESS ASSESSMENT

### A. Data Integrity Checks

| Check | Verdict | Explanation |
|---|---|---|
| Look-ahead bias audit | **PASS** | Entry is at next candle's open (shift +1). All rolling windows exclude current bar. Session normalisation uses expanding (not rolling) window. Cross-currency correlation uses `.shift(1)`. Synthetic data test validates no false positives on pure GBM noise. |
| Survivorship bias | **N/A** | Single instrument (EUR/USD) — no universe selection issue. |
| Point-in-time correctness | **PASS (partial)** | OHLCV data is point-in-time by construction. However, the news calendar is hardcoded rather than sourced from historical records, introducing potential point-in-time errors for event timestamps. |
| Corporate actions handling | **N/A** | Forex pair — no splits, dividends, or delistings. |

### B. Statistical Validity

| Check | Verdict | Explanation |
|---|---|---|
| IS vs OOS split documented | **PASS** | IS: 2023-01-01 → 2025-03-01 (2yr), OOS: 2025-03-01 → 2026-03-01 (1yr). 50-candle blackout buffer. |
| Walk-forward validation | **PASS (code exists)** | Walk-forward with 18-month train / 3-month test windows. Minimum 6 windows required. However, results were not generated in available runs due to insufficient signal counts. |
| Overfitting assessment | **PASS** | 189 tests run with FDR correction. Pre-commitment hash prevents post-hoc test selection. **0/189 tests significant after correction.** |
| Multiple testing correction | **PASS** | Benjamini-Hochberg FDR correction applied across all tests. Bonferroni also available. |
| t-statistics / p-values | **PASS** | Block bootstrap (block_size=20) used for CIs, preserving serial correlation structure. Binomial test for win rate significance. |
| Factor attribution | **NOT TESTED** | No decomposition against known risk premia (momentum, value, etc.). Not relevant for high-frequency forex, but would matter if deployed on equity CFDs. |

### C. Strategy Robustness

| Check | Verdict | Explanation |
|---|---|---|
| Parameter sensitivity | **NOT TESTED** | No systematic ±20% perturbation of hyperparameters. The CC extension does include a 4-parameter sweep (2×2×2×2 = 16 combos), but not for the core signal parameters. |
| Regime sensitivity | **PARTIAL** | Stationarity check compares first-half vs second-half win rates. Walk-forward tests rolling stability. No explicit bull/bear/high-vol/low-vol regime segmentation. |
| Drawdown analysis | **PARTIAL** | Equity curves with drawdown shading are generated. No explicit max drawdown / duration / recovery metrics reported in summary tables. |
| Stress testing | **NOT TESTED** | No 2008/2020/flash-crash scenarios. Data range (2023–2026) covers modest vol regimes only. |
| Capacity analysis | **NOT TESTED** | No market impact modelling. Not relevant at retail scale, but blocks institutional deployment. |

### D. Execution Realism

| Check | Verdict | Explanation |
|---|---|---|
| Transaction costs modelled | **PASS** | Bid-ask spread with 1.5× widening factor, instrument-specific spreads. No flat commission (correct for OANDA). |
| Slippage model validated | **FAIL** | `slippage_atr_fraction = 0.0`. No market impact beyond spread. For M5 timeframe, spread-only model is reasonable for retail sizes. |
| Turnover vs gross alpha | **PASS** | Mean Sharpe (gross): −0.507, Mean Sharpe (net): −4.829 on M5. Costs destroy any gross edge entirely. |

### E. Risk-Adjusted Performance

**Main run: EUR/USD M5, 189 tests**

| Metric | In-Sample (Best Config) | All Configs Average | Benchmark (Base Rate) |
|---|---|---|---|
| Annualized Return | N/A (measured in ATR units) | avg_return_net ≈ −0.7 ATR/trade | — |
| Annualized Volatility | — | — | — |
| Sharpe Ratio (net) | −1.153 (best by net Sharpe was B_sn K=20 at 1.12, but corrected p=1.0) | −4.829 | — |
| Sortino Ratio | Not computed | — | — |
| Max Drawdown | Not explicitly computed | — | — |
| Calmar Ratio | Not computed | — | — |
| Hit Rate (best) | 64.2% (B_sn, K=20, n=81) | — | 64.9% (session-matched) |
| Profit Factor | Not computed | — | — |

> [!CAUTION]
> **No edge was found.** 0/189 tests were significant after FDR correction. The best win rate (64.2%) was below the session-matched base rate (64.9%), yielding a **negative** win rate excess. Transaction costs further destroy any marginal gross returns.

**Baseline H1 run:** 22 signals, 50% win rate vs 40% base rate, Sharpe 10.86 — triggered 3 CRITICAL red flags (insufficient N, extreme Sharpe on small sample). **Statistically meaningless.**

**CC Extension H1 run:** 4 signals, 75% win rate — triggered 4 red flags including 3 CRITICAL. **Entirely meaningless.**

---

## 5. KNOWN ISSUES, RISKS & OPEN QUESTIONS

> **[SEVERITY: Critical]** — **No tradeable edge found.** Across 189 fully corrected tests on M5, zero results are statistically significant. The core alpha hypothesis (volume spikes predict direction) is not supported by the available data. Any continuation of this project requires a fundamental rethink of the signal, not parameter tuning.

> **[SEVERITY: Critical]** — **OANDA tick volume is not real volume.** The entire thesis requires volume to carry information about institutional flow. OANDA tick volume (count of price changes) has only ~0.7–0.85 correlation with real exchange volume. The hypothesis may be valid on real futures/exchange data but untestable on this data source.

> **[SEVERITY: High]** — **H1 timeframe is severely underpowered.** The primary config uses H1 as the "locked" primary timeframe, but only produces 22–38 signals over 2 years — far below the 150 minimum. The n-gate correctly blocks reporting, but the pipeline defaults to H1 despite knowing it cannot produce meaningful results.

> **[SEVERITY: High]** — **News calendar is static and hardcoded.** The `signal_filter.py` hardcodes ~200 event timestamps (NFP, CPI, FOMC, ECB) from 2023–2026. Any events not in the list are silently missed. In live trading, this would require constant manual maintenance or calendar API integration.

> **[SEVERITY: Medium]** — **No out-of-sample results generated.** The IS/OOS split exists in code and config, but the pipeline runs only in-sample. Previous conversations discussed holdout validation but no OOS results are present in the current output.

> **[SEVERITY: Medium]** — **Cross-currency extension adds complexity without value.** The CC-extension (EUR-move scoring, regime detection, adaptive sizing) produced only 4 signals on H1 and is entirely non-viable. This adds ~400 lines of code and a 16-combination parameter sweep for no demonstrated benefit.

> **[SEVERITY: Medium]** — **Variant D produces identical results to A_sn.** In the M5 summary table, variants A_sn and D produce byte-identical results across all metrics. This is because Variant D's implementation is functionally equivalent to A with session normalisation. One should be removed to reduce the test matrix.

> **[SEVERITY: Low]** — **Sharpe ratio calculation uses `√252` annualisation.** This assumes daily returns. For M5 data with variable signals per day, the annualisation factor may be misleading (though it's consistently applied).

> **[SEVERITY: Low]** — **`requirements.txt` is minimal.** Only lists `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`. No version pinning, no `oandapyV20` or API client dependency listed.

---

## 6. NEXT STEPS (PRIORITIZED)

1. **Validate on real exchange volume data** — Obtain actual CME EUR/USD futures volume data and re-run the pipeline. This is the single highest-priority action because the entire hypothesis may be valid but untestable on OANDA tick volume.

2. **Kill the H1 timeframe as primary** — Change `primary_timeframe` from H1 to M5 (which produces thousands of signals) or remove the H1 focus entirely. The current config creates false expectations about H1 viability.

3. **Remove duplicate Variant D** — A_sn and D produce identical results. Remove Variant D to cut the test matrix by ~14% (189 → ~163 tests) and eliminate confusion.

4. **Run explicit out-of-sample validation** — The OOS split exists in code but was never used in the actual pipeline output. Run the best M5 configuration on the held-out 2025-03 to 2026-03 period.

5. **Explore alternative alpha hypotheses** — Given the null result, consider: (a) volume spike + mean reversion rather than continuation, (b) volume as a filter for existing signals rather than the signal itself, (c) volume spike clustering / regime detection as information.

6. **Integrate a live news calendar** — Replace the hardcoded event list with an API-based calendar (e.g., ForexFactory, Investing.com calendar API) for any future live deployment.

7. **Add parameter sensitivity analysis** — Systematically perturb z_threshold (±20%), lookback (±30%), body_ratio, and wick_ratio to quantify robustness. Current pipeline runs a fixed parameter set.

8. **Compute proper risk metrics** — Add max drawdown, drawdown duration, Calmar ratio, Sortino ratio, and profit factor to the summary table. These exist conceptually (MFE/MAE are computed) but aren't surfaced.

9. **Pin dependency versions** — Update `requirements.txt` with exact versions and include `oandapyV20` (or whatever API client is used).

10. **Consider the project concluded if futures data also shows no edge** — If CME EUR/USD futures volume also fails to produce a significant signal after FDR correction, the hypothesis should be formally abandoned rather than subjected to further parameter search.

---

## 7. DEPLOYMENT READINESS VERDICT

| Dimension | Assessment |
|---|---|
| **Current stage** | Research prototype (backtesting engine complete, no edge found) |
| **Institutional grade?** | **No** — No edge, no execution layer, no live risk controls, tick volume only |
| **Estimated time to production-ready** | Not applicable — cannot deploy a strategy with no demonstrated edge. If an edge were found on real volume data: 3–6 months for execution layer, risk management, live monitoring, and paper-trade validation. |

**One sentence summary of the biggest gap:** The pipeline is methodologically rigorous and architecturally sound, but the fundamental alpha hypothesis (volume spikes predict direction) is **not supported by the data** — making all downstream deployment work premature.
