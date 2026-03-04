from __future__ import annotations
import logging
import pandas as pd
from data_loader import DataLoader

logger = logging.getLogger(__name__)

def load_secondary(config: dict, primary_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Loads secondary cross-currency pairs matching the primary_df's date range,
    and identically aligning it to primary_df's index.
    
    Parameters
    ----------
    config: dict
        The global `CONFIG` object.
    primary_df: pd.DataFrame
        The loaded primary dataframe with UTC DatetimeIndex.
    timeframe: str
        The timeframe matching the primary DF.

    Returns
    -------
    pd.DataFrame
        The strictly aligned secondary dataframe.
    """
    instrument = config.get("cc_secondary_instrument", "GBP_USD")

    # Safety bounds from the primary DF
    if len(primary_df) == 0:
        return primary_df.copy()

    start_date = "2023-01-01" # Will use config wide or DF min
    end_date = "2026-03-01"

    if len(primary_df) > 0:
        start_date = str(primary_df.index.min().date())
        # pad the end date slightly
        end_date = str((primary_df.index.max() + pd.Timedelta(days=1)).date())
    
    logger.info("CC Extension: Fetching secondary instrument %s %s aligned to primary", instrument, timeframe)

    loader = DataLoader()
    secondary_df_raw = loader.fetch_from_oanda(
        instrument=instrument,
        granularity=timeframe,
        start=start_date,
        end=end_date,
    )

    if len(secondary_df_raw) == 0:
        logger.warning("CC Extension: Failed to fetch any secondary data for %s.", instrument)
        return pd.DataFrame(index=primary_df.index, columns=["open", "high", "low", "close", "volume"])

    # Align it exactly to primary_df:
    # 1. Reindex to primary_df's index.
    secondary_aligned = secondary_df_raw.reindex(primary_df.index)

    missing_primary = (secondary_aligned['close'].isna() & ~primary_df['close'].isna()).sum()
    if missing_primary > 0:
        logger.warning(
            "CC Extension: Secondary %s is missing %d candles that exist in primary.", 
            instrument, missing_primary
        )

    # 2. To avoid over-extrapolating we only forward fill gaps of length <= 2.
    # We do not backward fill.
    secondary_aligned = secondary_aligned.ffill(limit=2)

    return secondary_aligned
