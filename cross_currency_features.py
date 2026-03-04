from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_cross_currency_features(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    signal_timestamps: pd.DatetimeIndex,
    config: dict
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by signal_timestamps with columns:
    [eur_move_score, rolling_corr, gbpusd_vol_zscore, is_eur_driven, corr_regime]
    """
    if len(signal_timestamps) == 0:
        return pd.DataFrame(columns=[
            "eur_move_score", "rolling_corr", "gbpusd_vol_zscore", 
            "is_eur_driven", "corr_regime"
        ])

    cc_corr_lookback = config.get("cc_corr_lookback", 20)
    min_eur_move_score = config.get("cc_min_eur_move_score", 0.0)
    max_gbpusd_vol_z = config.get("cc_max_gbpusd_vol_z", 1.5)
    
    # Pre-calculate rolling stats for the whole series to avoid slow loops
    # 1. Returns for correlation
    eurusd_returns = primary_df["close"].pct_change()
    gbpusd_returns = secondary_df["close"].pct_change()

    # The correlation window must use indices [t - cc_corr_lookback : t-1] inclusive.
    # We compute rolling correlation of length `cc_corr_lookback`, 
    # and then `.shift(1)` to ensure the correlation at `t` only uses data *up to* `t-1`.
    rolling_corr = eurusd_returns.rolling(window=cc_corr_lookback, min_periods=cc_corr_lookback).corr(gbpusd_returns)
    rolling_corr = rolling_corr.shift(1) # Strict look-ahead constraint

    # 3. GBP/USD Volume Z-score
    vol_lookback = config.get("lookback", 20)
    vol_mean = secondary_df["volume"].rolling(window=vol_lookback, min_periods=vol_lookback).mean().shift(1)
    vol_std = secondary_df["volume"].rolling(window=vol_lookback, min_periods=vol_lookback).std().shift(1)
    
    # Calculate exactly at t without including t in the rolling mean/std
    gbpusd_vol_zscore = (secondary_df["volume"] - vol_mean) / (vol_std + 1e-9)

    # Calculate returns for EUR Move Score on signal candles
    results = []
    
    for t in signal_timestamps:
        if t not in primary_df.index or t not in secondary_df.index:
            continue
            
        prim_row = primary_df.loc[t]
        sec_row = secondary_df.loc[t]
        
        # 1. EUR Move Score (close - open) / open
        eurusd_return_t = (prim_row["close"] - prim_row["open"]) / prim_row["open"]
        gbpusd_return_t = (sec_row["close"] - sec_row["open"]) / sec_row["open"]
        eur_move_score = eurusd_return_t - gbpusd_return_t
        
        # 2. Rolling Pair Correlation
        corr_t = rolling_corr.loc[t]
        
        # 3. GBP/USD Volume Z-score
        vol_z_t = gbpusd_vol_zscore.loc[t]
        
        # 4. Divergence Classification (Boolean)
        is_bullish = prim_row["close"] > prim_row["open"]
        
        is_eur_driven = False
        if is_bullish:
            if eur_move_score > min_eur_move_score and vol_z_t < max_gbpusd_vol_z:
                is_eur_driven = True
        else: # Bearish signal
            if eur_move_score < -min_eur_move_score and vol_z_t < max_gbpusd_vol_z:
                is_eur_driven = True
                
        # 5. Correlation Regime
        if pd.isna(corr_t):
            corr_regime = "LOW_CORR" # Default safely if not enough history
        elif corr_t >= 0.7:
            corr_regime = "HIGH_CORR"
        elif corr_t >= 0.4:
            corr_regime = "MED_CORR"
        else:
            corr_regime = "LOW_CORR"
            
        results.append({
            "time": t,
            "eur_move_score": float(eur_move_score),
            "rolling_corr": float(corr_t),
            "gbpusd_vol_zscore": float(vol_z_t),
            "is_eur_driven": bool(is_eur_driven),
            "corr_regime": corr_regime
        })
        
    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out.set_index("time", inplace=True)
    else:
        df_out = pd.DataFrame(columns=[
            "eur_move_score", "rolling_corr", "gbpusd_vol_zscore", 
            "is_eur_driven", "corr_regime"
        ])
        
    return df_out
