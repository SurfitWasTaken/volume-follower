from __future__ import annotations
"""
Synthetic OHLCV data generator for testing the pipeline when real data
is unavailable.

Produces a realistic DataFrame with:
- GBM-based price process
- Intraday U-shaped volume seasonality + random spikes
- Proper OHLC relationships
"""

import logging
import numpy as np
import pandas as pd
from config import CONFIG

logger = logging.getLogger(__name__)


def generate_synthetic_ohlcv(
    n_candles: int = 200_000,
    timeframe_minutes: int = 5,
    seed: int | None = None,
    start_price: float = 1.1000,
    daily_vol: float = 0.008,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame that mimics forex intraday data.

    Parameters
    ----------
    n_candles : int
        Number of candles to generate.
    timeframe_minutes : int
        Candle period in minutes (1, 5, or 15).
    seed : int or None
        Random seed; uses CONFIG default if None.
    start_price : float
        Starting mid-price.
    daily_vol : float
        Annualised-equivalent daily volatility (as fraction of price).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.  DatetimeIndex in UTC.
    """
    if seed is None:
        seed = CONFIG["random_seed"]
    rng = np.random.default_rng(seed)

    candles_per_day = int(24 * 60 / timeframe_minutes)
    per_candle_vol = daily_vol / np.sqrt(candles_per_day)

    logger.info(
        "Generating %d synthetic candles (tf=%dmin, σ_candle=%.6f)",
        n_candles, timeframe_minutes, per_candle_vol,
    )

    # --- Price process (GBM) ---
    returns = rng.normal(0, per_candle_vol, size=n_candles)
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Shift to get open prices (open_t = close_{t-1})
    open_prices = np.empty_like(close_prices)
    open_prices[0] = start_price
    open_prices[1:] = close_prices[:-1]

    # Intra-candle high/low: add noise around O/C
    candle_range = np.abs(close_prices - open_prices) + rng.exponential(
        per_candle_vol * start_price * 0.3, size=n_candles
    )
    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)

    high_prices = max_oc + rng.uniform(0, 1, n_candles) * candle_range * 0.5
    low_prices = min_oc - rng.uniform(0, 1, n_candles) * candle_range * 0.5

    # Ensure OHLC constraints
    high_prices = np.maximum(high_prices, max_oc)
    low_prices = np.minimum(low_prices, min_oc)

    # --- Volume process (U-shaped intraday + random spikes) ---
    # Create time index
    start_time = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    freq = f"{timeframe_minutes}min"
    timestamps = pd.date_range(start=start_time, periods=n_candles, freq=freq)

    # Remove weekends (Sat=5, Sun=6)
    weekday_mask = timestamps.weekday < 5
    timestamps = timestamps[weekday_mask][:n_candles]
    # If we ran out of weekday candles, regenerate with more
    if len(timestamps) < n_candles:
        timestamps = pd.date_range(
            start=start_time,
            periods=int(n_candles * 1.5),
            freq=freq,
        )
        timestamps = timestamps[timestamps.weekday < 5][:n_candles]

    n_actual = min(len(timestamps), n_candles)

    # Intraday volume profile: U-shape based on hour of day
    hours = timestamps[:n_actual].hour + timestamps[:n_actual].minute / 60.0
    # U-shape: high at 8-9 (London), 13-15 (NY), low at 0-6 and 20-24
    volume_profile = (
        1.0
        + 2.0 * np.exp(-0.5 * ((hours - 8.5) / 1.0) ** 2)   # London peak
        + 2.5 * np.exp(-0.5 * ((hours - 14.0) / 1.2) ** 2)   # NY peak
        + 0.5 * np.exp(-0.5 * ((hours - 21.0) / 0.8) ** 2)   # Asian modest
    )

    base_volume = rng.poisson(lam=500, size=n_actual).astype(float)
    volume = (base_volume * volume_profile.to_numpy()).astype(int)

    # Inject random volume spikes (~2% of candles)
    spike_mask = rng.random(n_actual) < 0.02
    volume[spike_mask] = (volume[spike_mask] * rng.uniform(3, 8, spike_mask.sum())).astype(int)

    # Clip volumes
    volume = np.clip(volume, 1, None)

    df = pd.DataFrame(
        {
            "open": open_prices[:n_actual],
            "high": high_prices[:n_actual],
            "low": low_prices[:n_actual],
            "close": close_prices[:n_actual],
            "volume": volume,
        },
        index=timestamps[:n_actual],
    )
    df.index.name = "time"

    logger.info(
        "Synthetic data generated: %d candles from %s to %s",
        len(df), df.index[0], df.index[-1],
    )
    return df
