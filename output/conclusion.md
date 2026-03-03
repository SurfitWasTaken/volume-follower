# Volume Spike Signal — Conclusion

## 1. Statistical Significance

| Metric | Value |
|--------|-------|
| Total tests conducted | 189 |
| Significant at α=0.05 (raw) | 0 |
| Significant after FDR correction | 0 |
| Edge exists (after correction)? | **NO** |

## 2. Best Signal Configuration

| Parameter | Value |
|-----------|-------|
| Instrument | EUR_USD |
| Timeframe | M5 |
| Variant | B_sn |
| Filters | direction+body_ratio+wick+session_both+news |
| K (forward window) | 20 |
| n signals | 81 |
| Win rate | 0.6420 |
| Base rate | 0.6488 |
| Win rate excess | -0.0068 |
| p-value (corrected) | 1.0000 |
| Sharpe (gross) | 1.116 |
| Sharpe (net of costs) | -1.153 |
| MFE/MAE ratio | 1.151 |

## 3. Transaction Cost Impact

- Mean Sharpe (gross): -0.507
- Mean Sharpe (net):   -4.829
- Edge survives costs? **NO**

## 4. Temporal Stability


## 5. Permutation Test


## 6. Caveats

- OANDA tick volume is a proxy for real volume (correlation ≈ 0.7–0.85).
- CFD instruments (SPX500, NAS100, XAU) may have different microstructure than exchange-traded futures.
- Hardcoded news calendar may miss some events.
- Past edge does not guarantee future edge — microstructure strategies are sensitive to regime change.
- Each timeframe is tested independently; results should NOT be pooled across timeframes.

## 7. Recommended Next Steps

- [ ] Validate on exchange data (real volume) if edge found
- [ ] Live paper-trade for ≥3 months before capital deployment
- [ ] Monitor for edge decay with monthly Sharpe tracking
- [ ] Investigate economic mechanism of any discovered edge