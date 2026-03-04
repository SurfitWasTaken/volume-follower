from __future__ import annotations
import pandas as pd
import numpy as np

def compute_position_size(
    is_eur_driven: bool,
    corr_regime: str,
    eur_move_score: float,
    gbpusd_vol_zscore: float,
    base_size: float,
    config: dict,
    percentile_rank: float
) -> float:
    """
    Returns a position size scalar relative to base_size.
    """
    if not is_eur_driven:
        return 0.0

    if corr_regime == 'LOW_CORR' and config.get("cc_skip_low_corr_regime", True):
        return 0.0

    # Base multiplier by regime
    if corr_regime == "HIGH_CORR":
        mult = config.get("cc_size_high_corr_mult", 1.5)
    elif corr_regime == "MED_CORR":
        mult = config.get("cc_size_med_corr_mult", 1.0)
    else:
        mult = config.get("cc_size_low_corr_mult", 0.5)

    # Scale by |eur_move_score| percentile rank within the rolling distribution
    if percentile_rank >= 0.90:
        mult *= config.get("cc_size_high_eur_move_mult", 1.2)
    elif percentile_rank <= 0.50:
        mult *= config.get("cc_size_low_eur_move_mult", 0.8)

    # Penalise for anomalous GBP/USD volume
    if gbpusd_vol_zscore > 2.0:
        mult *= config.get("cc_size_gbpusd_vol_penalty", 0.5)

    final_size = base_size * mult
    
    # Cap and floor
    max_size = base_size * config.get("cc_size_max_mult", 2.0)
    min_size = base_size * config.get("cc_size_min_mult", 0.25)
    
    if final_size > 0.0:
        final_size = min(max_size, max(min_size, final_size))
        
    return final_size

def compute_position_sizes(
    features_df: pd.DataFrame,
    config: dict,
    base_size: float = 1.0
) -> pd.Series:
    """
    Given the cross-currency features DataFrame, compute the position size scalar 
    for each signal. Uses an expanding window over |eur_move_score| to calculate
    percentile ranks to prevent look-ahead bias.
    """
    if len(features_df) == 0:
        return pd.Series(dtype=float)
        
    # We need the *historical* percentile rank of |eur_move_score| for each row.
    # To prevent lookahead, we compute the rank of row `i` based on rows `0` to `i` (inclusive).
    abs_scores = features_df["eur_move_score"].abs()
    
    # Using an expanding window rank. Pandas expanding().rank(pct=True) is perfect for this.
    percentiles = abs_scores.expanding(min_periods=1).rank(pct=True)
    
    sizes = []
    for idx, row in features_df.iterrows():
        pct = percentiles.loc[idx]
        size = compute_position_size(
            is_eur_driven=row["is_eur_driven"],
            corr_regime=row["corr_regime"],
            eur_move_score=row["eur_move_score"],
            gbpusd_vol_zscore=row["gbpusd_vol_zscore"],
            base_size=base_size,
            config=config,
            percentile_rank=pct
        )
        sizes.append(size)
        
    return pd.Series(sizes, index=features_df.index)
