from __future__ import annotations
"""
Signal detection module — identifies volume spike candles.

Four configurable variants:
  A — Rolling Z-Score
  B — Volume Multiplier
  C — Percentile Rank
  D — Adaptive (session-hour normalised Z-Score)

Each variant can optionally be pre-adjusted for intraday volume seasonality
(session_normalised=True).

After detection, a minimum signal gap is enforced to prevent overlapping
outcome windows and serial correlation between signals, which would
invalidate standard statistical tests.
"""

import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Detect volume-spike candles using various statistical methods."""

    # ----- public API ----------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        variant: str,
        session_normalised: bool = False,
        min_signal_gap: int | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Run a detection variant and return a boolean mask.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.
        variant : str
            One of 'A', 'B', 'C', 'D'.
        session_normalised : bool
            If True, pre-normalise volume by hour-of-day average profile
            before applying the threshold (except Variant D which does
            this internally).
        min_signal_gap : int or None
            Minimum number of candles between consecutive signals.
            Uses CONFIG default if None.  Set to 0 to disable.
        **kwargs
            Override CONFIG defaults (lookback, z_threshold, etc.).

        Returns
        -------
        pd.Series[bool]  — True where a volume spike is detected.
        """
        vol = df["volume"].astype(float).copy()

        if session_normalised and variant != "D":
            vol = self._session_normalise(vol, df.index)

        dispatch = {
            "A": self._variant_a_zscore,
            "B": self._variant_b_multiplier,
            "C": self._variant_c_percentile,
            "D": self._variant_d_adaptive,
        }
        func = dispatch.get(variant.upper())
        if func is None:
            raise ValueError(f"Unknown variant '{variant}'. Choose A/B/C/D.")

        if variant.upper() == "D":
            signals = func(df, vol, **kwargs)
        else:
            signals = func(vol, **kwargs)

        # ── Enforce minimum gap between signals ──────────────────────────
        gap = min_signal_gap if min_signal_gap is not None else CONFIG["min_signal_gap"]
        if gap > 0:
            signals = self._enforce_min_gap(signals, gap)

        n_sig = signals.sum()
        logger.info(
            "Variant %s (session_norm=%s, gap=%d): %d signals / %d candles (%.2f%%)",
            variant, session_normalised, gap, n_sig, len(df),
            100 * n_sig / max(len(df), 1),
        )
        return signals

    # ----- Minimum gap enforcement ----------------------------------------

    @staticmethod
    def _enforce_min_gap(signals: pd.Series, gap: int) -> pd.Series:
        """
        Remove signals that fire within `gap` candles of a previous signal.

        The *first* signal in any cluster is kept; subsequent ones within
        the cooldown window are dropped.
        """
        mask = signals.copy()
        indices = np.where(mask.values)[0]
        keep = []
        last_kept = -gap - 1  # ensures the first signal is always kept
        for idx in indices:
            if idx - last_kept >= gap:
                keep.append(idx)
                last_kept = idx

        new_mask = pd.Series(False, index=signals.index)
        if keep:
            new_mask.iloc[keep] = True

        dropped = len(indices) - len(keep)
        if dropped > 0:
            logger.debug("  Gap filter removed %d/%d overlapping signals.", dropped, len(indices))
        return new_mask

    # ----- Variant A: Rolling Z-Score ------------------------------------

    @staticmethod
    def _variant_a_zscore(
        vol: pd.Series,
        lookback: int | None = None,
        z_threshold: float | None = None,
        **_kw,
    ) -> pd.Series:
        """
        Fire when (V_t − μ) / σ > z_threshold over a rolling window
        of the *preceding* `lookback` candles (current candle excluded).
        """
        lookback = lookback or CONFIG["lookback"]
        z_threshold = z_threshold if z_threshold is not None else CONFIG["z_threshold"]

        # Shift(1) excludes the current candle from the window
        roll_mean = vol.shift(1).rolling(window=lookback, min_periods=lookback).mean()
        roll_std = vol.shift(1).rolling(window=lookback, min_periods=lookback).std(ddof=1)

        z = (vol - roll_mean) / roll_std.replace(0, np.nan)
        return (z > z_threshold).fillna(False)

    # ----- Variant B: Volume Multiplier ----------------------------------

    @staticmethod
    def _variant_b_multiplier(
        vol: pd.Series,
        lookback: int | None = None,
        multiplier: float | None = None,
        **_kw,
    ) -> pd.Series:
        """Fire when V_t > multiplier × rolling mean of preceding candles."""
        lookback = lookback or CONFIG["lookback"]
        multiplier = multiplier if multiplier is not None else CONFIG["multiplier"]

        roll_mean = vol.shift(1).rolling(window=lookback, min_periods=lookback).mean()
        return (vol > multiplier * roll_mean).fillna(False)

    # ----- Variant C: Percentile Rank ------------------------------------

    @staticmethod
    def _variant_c_percentile(
        vol: pd.Series,
        lookback: int | None = None,
        percentile_threshold: float | None = None,
        **_kw,
    ) -> pd.Series:
        """Fire when V_t exceeds the `percentile_threshold` of the rolling window."""
        lookback = lookback or CONFIG["lookback"]
        pct = percentile_threshold if percentile_threshold is not None else CONFIG["percentile_threshold"]

        def _rank_pct(window):
            """Rank the last element within the preceding window."""
            preceding = window.values[:-1]
            current = window.values[-1]
            if len(preceding) < lookback:
                return np.nan
            return (preceding < current).sum() / len(preceding) * 100

        # Window = lookback + 1 so that we can separate current from history
        rank = vol.rolling(window=lookback + 1, min_periods=lookback + 1).apply(
            _rank_pct, raw=False
        )
        return (rank >= pct).fillna(False)

    # ----- Variant D: Adaptive (VWAP-normalised) -------------------------

    def _variant_d_adaptive(
        self,
        df: pd.DataFrame,
        vol: pd.Series,
        lookback: int | None = None,
        z_threshold: float | None = None,
        **_kw,
    ) -> pd.Series:
        """
        Normalise volume by session-hour baseline, then apply Z-Score.

        This controls for the structural U-shaped volume profile so that
        signals at session open are not spuriously generated.

        NOTE: The session-hour profile is computed with an expanding window
        on prior data only (no look-ahead).  For walk-forward analysis,
        the caller should pass only the training slice so the profile
        is not contaminated by future data.
        """
        lookback = lookback or CONFIG["lookback"]
        z_threshold = z_threshold if z_threshold is not None else CONFIG["z_threshold"]

        norm_vol = self._session_normalise(vol, df.index)
        return self._variant_a_zscore(norm_vol, lookback=lookback, z_threshold=z_threshold)

    # ----- Session normalisation helper ----------------------------------

    @staticmethod
    def _session_normalise(vol: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """
        Divide volume by the expanding mean volume for that hour-of-day,
        producing a de-seasonalised series.

        Uses an *expanding* window so each candle only uses historical data
        (no look-ahead).
        """
        hour = index.hour
        day_of_week = index.weekday

        # Create a composite key: (day_of_week, hour)
        group_key = day_of_week * 100 + hour

        vol_with_key = pd.DataFrame(
            {"vol": vol.values, "key": group_key.values}, index=vol.index,
        )

        # Expanding mean per group (shift to avoid including current candle)
        expanding_mean = vol_with_key.groupby("key")["vol"].transform(
            lambda x: x.shift(1).expanding(min_periods=10).mean()
        )

        # Avoid division by zero
        expanding_mean = expanding_mean.replace(0, np.nan)
        normalised = vol / expanding_mean

        # Fill early NaNs (insufficient history) with raw volume — these
        # won't produce signals anyway because of rolling window requirements
        normalised = normalised.fillna(vol)

        return normalised
