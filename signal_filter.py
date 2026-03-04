from __future__ import annotations
"""
Signal filtering module — cumulative post-detection filters.

Filters 1–7 are applied in sequence.  Each filter narrows the signal set
and assigns/refines the predicted direction.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Hard-coded high-impact macro event dates (UTC).
# Covers FOMC, NFP, CPI, ECB decisions 2023-01 → 2026-06.
# Each entry is a (date_str, time_str, event) tuple.
# This is intentionally a curated subset; a production system would
# pull from a live calendar.
# ──────────────────────────────────────────────────────────────────────────

HIGH_IMPACT_EVENTS: list[tuple[str, str, str]] = [
    # ---- 2023 ----
    ("2023-01-06", "13:30", "NFP"), ("2023-01-12", "13:30", "CPI"),
    ("2023-02-01", "19:00", "FOMC"), ("2023-02-03", "13:30", "NFP"),
    ("2023-02-14", "13:30", "CPI"), ("2023-03-10", "13:30", "NFP"),
    ("2023-03-14", "12:30", "CPI"), ("2023-03-16", "13:15", "ECB"),
    ("2023-03-22", "18:00", "FOMC"), ("2023-04-07", "12:30", "NFP"),
    ("2023-04-12", "12:30", "CPI"), ("2023-05-03", "18:00", "FOMC"),
    ("2023-05-04", "12:15", "ECB"), ("2023-05-05", "12:30", "NFP"),
    ("2023-05-10", "12:30", "CPI"), ("2023-06-02", "12:30", "NFP"),
    ("2023-06-13", "12:30", "CPI"), ("2023-06-14", "18:00", "FOMC"),
    ("2023-06-15", "12:15", "ECB"), ("2023-07-07", "12:30", "NFP"),
    ("2023-07-12", "12:30", "CPI"), ("2023-07-26", "18:00", "FOMC"),
    ("2023-07-27", "12:15", "ECB"), ("2023-08-04", "12:30", "NFP"),
    ("2023-08-10", "12:30", "CPI"), ("2023-09-01", "12:30", "NFP"),
    ("2023-09-13", "12:30", "CPI"), ("2023-09-14", "12:15", "ECB"),
    ("2023-09-20", "18:00", "FOMC"), ("2023-10-06", "12:30", "NFP"),
    ("2023-10-12", "12:30", "CPI"), ("2023-10-26", "12:15", "ECB"),
    ("2023-11-01", "18:00", "FOMC"), ("2023-11-03", "12:30", "NFP"),
    ("2023-11-14", "13:30", "CPI"), ("2023-12-08", "13:30", "NFP"),
    ("2023-12-12", "13:30", "CPI"), ("2023-12-13", "19:00", "FOMC"),
    ("2023-12-14", "13:15", "ECB"),
    # ---- 2024 ----
    ("2024-01-05", "13:30", "NFP"), ("2024-01-11", "13:30", "CPI"),
    ("2024-01-25", "13:15", "ECB"), ("2024-01-31", "19:00", "FOMC"),
    ("2024-02-02", "13:30", "NFP"), ("2024-02-13", "13:30", "CPI"),
    ("2024-03-08", "13:30", "NFP"), ("2024-03-12", "12:30", "CPI"),
    ("2024-03-20", "18:00", "FOMC"), ("2024-04-05", "12:30", "NFP"),
    ("2024-04-10", "12:30", "CPI"), ("2024-04-11", "12:15", "ECB"),
    ("2024-05-01", "18:00", "FOMC"), ("2024-05-03", "12:30", "NFP"),
    ("2024-05-15", "12:30", "CPI"), ("2024-06-06", "12:15", "ECB"),
    ("2024-06-07", "12:30", "NFP"), ("2024-06-12", "12:30", "FOMC+CPI"),
    ("2024-07-05", "12:30", "NFP"), ("2024-07-11", "12:30", "CPI"),
    ("2024-07-31", "18:00", "FOMC"), ("2024-08-02", "12:30", "NFP"),
    ("2024-08-14", "12:30", "CPI"), ("2024-09-06", "12:30", "NFP"),
    ("2024-09-11", "12:30", "CPI"), ("2024-09-12", "12:15", "ECB"),
    ("2024-09-18", "18:00", "FOMC"), ("2024-10-04", "12:30", "NFP"),
    ("2024-10-10", "12:30", "CPI"), ("2024-10-17", "12:15", "ECB"),
    ("2024-11-01", "12:30", "NFP"), ("2024-11-07", "19:00", "FOMC"),
    ("2024-11-13", "13:30", "CPI"), ("2024-12-06", "13:30", "NFP"),
    ("2024-12-11", "13:30", "CPI"), ("2024-12-12", "13:15", "ECB"),
    ("2024-12-18", "19:00", "FOMC"),
    # ---- 2025 ----
    ("2025-01-10", "13:30", "NFP"), ("2025-01-15", "13:30", "CPI"),
    ("2025-01-29", "19:00", "FOMC"), ("2025-01-30", "13:15", "ECB"),
    ("2025-02-07", "13:30", "NFP"), ("2025-02-12", "13:30", "CPI"),
    ("2025-03-07", "13:30", "NFP"), ("2025-03-12", "12:30", "CPI"),
    ("2025-03-19", "18:00", "FOMC"), ("2025-04-04", "12:30", "NFP"),
    ("2025-04-10", "12:30", "CPI"), ("2025-04-17", "12:15", "ECB"),
    ("2025-05-02", "12:30", "NFP"), ("2025-05-07", "18:00", "FOMC"),
    ("2025-05-13", "12:30", "CPI"), ("2025-06-05", "12:15", "ECB"),
    ("2025-06-06", "12:30", "NFP"), ("2025-06-11", "12:30", "CPI"),
    ("2025-06-18", "18:00", "FOMC"), ("2025-07-03", "12:30", "NFP"),
    ("2025-07-10", "12:30", "CPI"), ("2025-07-30", "18:00", "FOMC"),
    ("2025-08-01", "12:30", "NFP"), ("2025-08-12", "12:30", "CPI"),
    ("2025-09-05", "12:30", "NFP"), ("2025-09-10", "12:30", "CPI"),
    ("2025-09-11", "12:15", "ECB"), ("2025-09-17", "18:00", "FOMC"),
    ("2025-10-03", "12:30", "NFP"), ("2025-10-14", "12:30", "CPI"),
    ("2025-10-30", "12:15", "ECB"), ("2025-11-05", "19:00", "FOMC"),
    ("2025-11-07", "13:30", "NFP"), ("2025-11-12", "13:30", "CPI"),
    ("2025-12-05", "13:30", "NFP"), ("2025-12-10", "13:30", "CPI"),
    ("2025-12-11", "13:15", "ECB"), ("2025-12-17", "19:00", "FOMC"),
    # ---- 2026 (partial) ----
    ("2026-01-09", "13:30", "NFP"), ("2026-01-14", "13:30", "CPI"),
    ("2026-01-28", "19:00", "FOMC"), ("2026-02-06", "13:30", "NFP"),
    ("2026-02-11", "13:30", "CPI"), ("2026-03-06", "13:30", "NFP"),
]


def _build_event_timestamps() -> pd.DatetimeIndex:
    """Convert the event list into a UTC DatetimeIndex."""
    stamps = []
    for date_s, time_s, _ in HIGH_IMPACT_EVENTS:
        stamps.append(pd.Timestamp(f"{date_s} {time_s}", tz="UTC"))
    return pd.DatetimeIndex(stamps)


class SignalFilter:
    """Apply cumulative post-detection filters to volume-spike signals."""

    def __init__(self):
        self._event_times = _build_event_timestamps()

    # ----- Master apply method -------------------------------------------

    def apply(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        filters: list[str],
        **kwargs,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Apply a list of named filters to the signal mask.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        signals : pd.Series[bool]
            Initial spike detections.
        filters : list[str]
            Ordered list of filter names to apply cumulatively:
            'direction', 'body_ratio', 'wick', 'session_london',
            'session_newyork', 'session_both', 'news', 'trend',
            'range_position'.
        **kwargs
            Override defaults for any filter parameter.

        Returns
        -------
        (filtered_signals, directions)
            - filtered_signals: pd.Series[bool]
            - directions: pd.Series[int]  (+1 = long, -1 = short, 0 = no signal)
        """
        mask = signals.copy()
        directions = pd.Series(0, index=df.index, dtype=int)

        for f in filters:
            before = mask.sum()
            if f == "direction":
                mask, directions = self.filter_direction(df, mask)
            elif f == "body_ratio":
                mask = self.filter_body_ratio(df, mask, **kwargs)
            elif f == "wick":
                mask, directions = self.filter_wick_asymmetry(df, mask, directions, **kwargs)
            elif f.startswith("session"):
                windows = self._parse_session_arg(f)
                mask = self.filter_session_window(df, mask, windows=windows)
            elif f == "news":
                mask = self.filter_news_exclusion(df, mask, **kwargs)
            elif f == "trend":
                mask, directions = self.filter_trend_context(df, mask, directions, **kwargs)
            elif f == "range_position":
                mask = self.filter_range_position(df, mask, **kwargs)
            elif f == "cc_eur_driven":
                mask = self.filter_cc_eur_driven(df, mask, **kwargs)
            else:
                logger.warning("Unknown filter '%s' — skipped.", f)
                continue

            after = mask.sum()
            logger.info("  Filter %-18s : %d → %d signals (−%d)", f, before, after, before - after)

        # Zero out directions for removed signals
        directions = directions.where(mask, 0)
        return mask, directions

    # ----- Filter 1: Candle direction ------------------------------------

    @staticmethod
    def filter_direction(
        df: pd.DataFrame, signals: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Classify each spike candle as bullish (+1) or bearish (-1)."""
        directions = pd.Series(0, index=df.index, dtype=int)
        bullish = df["close"] > df["open"]
        bearish = df["close"] < df["open"]
        directions[signals & bullish] = 1
        directions[signals & bearish] = -1

        # Remove doji-like candles where close == open exactly
        mask = signals & (directions != 0)
        return mask, directions

    # ----- Filter 2: Body-to-range ratio ---------------------------------

    @staticmethod
    def filter_body_ratio(
        df: pd.DataFrame,
        signals: pd.Series,
        min_body_ratio: float | None = None,
        **_kwargs,
    ) -> pd.Series:
        """Keep only candles with body_ratio > threshold."""
        min_br = min_body_ratio if min_body_ratio is not None else CONFIG["min_body_ratio"]
        body = (df["close"] - df["open"]).abs()
        rng = df["high"] - df["low"]
        ratio = body / rng.replace(0, np.nan)
        return signals & (ratio > min_br).fillna(False)

    # ----- Filter 3: Wick asymmetry -------------------------------------

    @staticmethod
    def filter_wick_asymmetry(
        df: pd.DataFrame,
        signals: pd.Series,
        directions: pd.Series,
        wick_ratio: float | None = None,
        **_kwargs,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Reject signals with excessive rejection wicks.

        For bullish spikes: reject if upper_wick > wick_ratio × body.
        For bearish spikes: reject if lower_wick > wick_ratio × body.
        """
        wr = wick_ratio if wick_ratio is not None else CONFIG["wick_ratio"]
        body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        reject_bull = (directions == 1) & (upper_wick > wr * body)
        reject_bear = (directions == -1) & (lower_wick > wr * body)

        mask = signals & ~reject_bull & ~reject_bear
        return mask, directions

    # ----- Filter 4: Session window -------------------------------------

    @staticmethod
    def filter_session_window(
        df: pd.DataFrame,
        signals: pd.Series,
        windows: list[tuple[int, int, int, int]] | None = None,
    ) -> pd.Series:
        """Keep signals only within specified (hour, min, hour, min) windows."""
        if windows is None:
            windows = list(CONFIG["session_windows"].values())

        idx = df.index
        time_minutes = idx.hour * 60 + idx.minute

        in_window = pd.Series(False, index=idx)
        for h_start, m_start, h_end, m_end in windows:
            start_mins = h_start * 60 + m_start
            end_mins = h_end * 60 + m_end
            in_window |= (time_minutes >= start_mins) & (time_minutes < end_mins)

        return signals & in_window

    # ----- Filter 5: News exclusion -------------------------------------

    def filter_news_exclusion(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        news_buffer_minutes: int | None = None,
        **_kwargs,
    ) -> pd.Series:
        """Exclude signals within ±buffer minutes of high-impact events."""
        buf = news_buffer_minutes if news_buffer_minutes is not None else CONFIG["news_buffer_minutes"]
        buf_td = pd.Timedelta(minutes=buf)

        near_event = pd.Series(False, index=df.index)
        for evt_time in self._event_times:
            near_event |= (df.index >= evt_time - buf_td) & (df.index <= evt_time + buf_td)

        return signals & ~near_event

    # ----- Filter 6: Trend context (50 / 200 MA) -------------------------

    @staticmethod
    def filter_trend_context(
        df: pd.DataFrame,
        signals: pd.Series,
        directions: pd.Series,
        ma_short: int | None = None,
        ma_long: int | None = None,
        mode: Literal["with_trend", "counter_trend", "all"] = "with_trend",
        **_kwargs,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Keep only signals aligned with (or against) the prevailing trend.

        Trend classification:
          uptrend   : close > MA50 > MA200
          downtrend : close < MA50 < MA200
          mixed     : everything else
        """
        ma_s = ma_short or CONFIG["ma_short"]
        ma_l = ma_long or CONFIG["ma_long"]

        ma50 = df["close"].rolling(ma_s, min_periods=ma_s).mean()
        ma200 = df["close"].rolling(ma_l, min_periods=ma_l).mean()

        uptrend = (df["close"] > ma50) & (ma50 > ma200)
        downtrend = (df["close"] < ma50) & (ma50 < ma200)

        if mode == "with_trend":
            keep = (
                (uptrend & (directions == 1))
                | (downtrend & (directions == -1))
            )
        elif mode == "counter_trend":
            keep = (
                (downtrend & (directions == 1))
                | (uptrend & (directions == -1))
            )
        else:
            keep = pd.Series(True, index=df.index)

        mask = signals & keep
        return mask, directions

    # ----- Filter 7: Prior range position ---------------------------------

    @staticmethod
    def filter_range_position(
        df: pd.DataFrame,
        signals: pd.Series,
        range_lookback: int | None = None,
        position: Literal["breakout", "mid", "all"] = "breakout",
        **_kwargs,
    ) -> pd.Series:
        """
        Classify signal candle position relative to the prior N-candle range.

        breakout : close above prior high (bullish) or below prior low (bearish)
        mid      : close within the middle 60% of the prior range
        """
        rl = range_lookback or CONFIG["range_lookback"]

        prior_high = df["high"].shift(1).rolling(rl, min_periods=rl).max()
        prior_low = df["low"].shift(1).rolling(rl, min_periods=rl).min()

        if position == "breakout":
            keep = (df["close"] > prior_high) | (df["close"] < prior_low)
        elif position == "mid":
            rng = prior_high - prior_low
            mid_low = prior_low + 0.2 * rng
            mid_high = prior_high - 0.2 * rng
            keep = (df["close"] >= mid_low) & (df["close"] <= mid_high)
        else:
            keep = pd.Series(True, index=df.index)

        return signals & keep.fillna(False)

    # ----- Filter 8: Cross currency confirmation --------------------------

    @staticmethod
    def filter_cc_eur_driven(
        df: pd.DataFrame,
        signals: pd.Series,
        cc_features: pd.DataFrame | None = None,
        **_kwargs,
    ) -> pd.Series:
        """
        Retain only signals classified as EUR-driven.
        Also drops signals in LOW_CORR regime if configured to do so.
        """
        if not CONFIG.get("cc_enabled", False):
            return signals

        if cc_features is None or cc_features.empty:
            return signals & False
            
        # Ensure cc_features indices match the signal times exactly
        is_eur_driven = cc_features["is_eur_driven"].reindex(df.index).fillna(False).astype(bool)
        
        skip_low = CONFIG.get("cc_skip_low_corr_regime", True)
        if skip_low:
            not_low_corr = (cc_features["corr_regime"] != "LOW_CORR").reindex(df.index).fillna(False)
            keep = is_eur_driven & not_low_corr
        else:
            keep = is_eur_driven
            
        return signals & keep

    # ----- Helpers -------------------------------------------------------

    @staticmethod
    def _parse_session_arg(name: str) -> list[tuple[int, int, int, int]]:
        """Map filter names like 'session_london' to window tuples."""
        windows_map = CONFIG["session_windows"]
        if "london" in name and "newyork" not in name and "both" not in name:
            return [windows_map["london"]]
        elif "newyork" in name and "london" not in name and "both" not in name:
            return [windows_map["newyork"]]
        else:  # 'session_both' or generic
            return list(windows_map.values())
