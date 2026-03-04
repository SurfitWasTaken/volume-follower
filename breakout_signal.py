"""
Breakout Signal Module — Volume-Gated Breakout Filter Hypothesis.

Tests whether volume spikes identify elevated-volatility regimes where
a T+1 price breakout has higher predictive power than unconditional
breakouts.

Signal flow:
    T   : Volume spike detected (Variant B, regime gate only)
    T+1 : Breakout confirmation — close[T+1] > high[T] → LONG
                                   close[T+1] < low[T]  → SHORT
                                   otherwise → SKIP
    T+2 : Entry at open (one-bar delay, no look-ahead)

By placing signals at T+1, the existing OutcomeCalculator with
entry_pos = sig_pos + 1 enters at T+2's open automatically.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import CONFIG
from outcome_calculator import OutcomeCalculator

logger = logging.getLogger(__name__)


def detect_breakout_signals(
    df: pd.DataFrame,
    spike_signals: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    For each volume spike at T, check candle T+1 for a price breakout.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    spike_signals : pd.Series[bool]
        Boolean mask of volume spike detections (at time T).

    Returns
    -------
    (breakout_signals, directions)
        breakout_signals : pd.Series[bool] — True at T+1 for valid breakouts
        directions : pd.Series[int] — +1 (long) / -1 (short) / 0 (no signal)

    Look-ahead audit:
        - high[T], low[T] are known at T close ✓
        - close[T+1] is known at T+1 close ✓
        - Signal placed at T+1, entry at T+2 open ✓
    """
    breakout_signals = pd.Series(False, index=df.index, dtype=bool)
    directions = pd.Series(0, index=df.index, dtype=int)

    pos_map = {t: i for i, t in enumerate(df.index)}

    spike_indices = df.index[spike_signals]
    n_long = 0
    n_short = 0
    n_ambiguous = 0

    for spike_time in spike_indices:
        t_pos = pos_map[spike_time]
        t1_pos = t_pos + 1

        if t1_pos >= len(df):
            continue

        spike_high = df.iloc[t_pos]["high"]
        spike_low = df.iloc[t_pos]["low"]
        t1_close = df.iloc[t1_pos]["close"]
        t1_time = df.index[t1_pos]

        if t1_close > spike_high:
            # Breakout above spike high → LONG
            breakout_signals.iloc[t1_pos] = True
            directions.iloc[t1_pos] = 1
            n_long += 1
        elif t1_close < spike_low:
            # Breakout below spike low → SHORT
            breakout_signals.iloc[t1_pos] = True
            directions.iloc[t1_pos] = -1
            n_short += 1
        else:
            # Ambiguous — T+1 closed within spike range
            n_ambiguous += 1

    total = n_long + n_short + n_ambiguous
    logger.info(
        "Breakout filter: %d spikes → %d long + %d short + %d ambiguous "
        "(%.1f%% tradeable)",
        total, n_long, n_short, n_ambiguous,
        100 * (n_long + n_short) / max(total, 1),
    )
    return breakout_signals, directions


def compute_unconditional_breakout_rate(
    df: pd.DataFrame,
    K: int,
    session_hours: set[int] | None = None,
    n_samples: int = 10_000,
) -> dict:
    """
    Compute the unconditional breakout win rate on ALL M5 candles.

    For every candle T where close[T+1] breaks above high[T] or below
    low[T], enter at T+2 open and evaluate with same TP/SL as the
    main pipeline. This is the base rate for the breakout filter test.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset.
    K : int
        Forward window for outcome evaluation.
    session_hours : set[int] | None
        If provided, only consider candles T where T's hour is in this set.
    n_samples : int
        Max samples to evaluate (random subset if too many breakouts).

    Returns
    -------
    dict with keys: n_breakouts, n_long, n_short, win_rate, n_evaluated
    """
    oc = OutcomeCalculator()
    atr = oc.compute_atr(df)

    min_move = CONFIG["min_move_atr"]
    stop_dist = CONFIG["stop_distance_atr"]

    pos_map = {t: i for i, t in enumerate(df.index)}

    # Find all breakouts
    breakout_indices = []
    breakout_dirs = []

    for i in range(len(df) - 2):  # need T, T+1, T+2
        t_time = df.index[i]

        # Session filter
        if session_hours is not None and t_time.hour not in session_hours:
            continue

        spike_high = df.iloc[i]["high"]
        spike_low = df.iloc[i]["low"]
        t1_close = df.iloc[i + 1]["close"]

        if t1_close > spike_high:
            breakout_indices.append(i + 1)  # signal at T+1
            breakout_dirs.append(1)
        elif t1_close < spike_low:
            breakout_indices.append(i + 1)  # signal at T+1
            breakout_dirs.append(-1)

    n_total = len(breakout_indices)
    n_long = sum(1 for d in breakout_dirs if d == 1)
    n_short = n_total - n_long

    if n_total == 0:
        return {
            "n_breakouts": 0, "n_long": 0, "n_short": 0,
            "win_rate": np.nan, "n_evaluated": 0,
        }

    # Subsample if too many
    rng = np.random.RandomState(CONFIG["random_seed"])
    if n_total > n_samples:
        idx = rng.choice(n_total, n_samples, replace=False)
        breakout_indices = [breakout_indices[j] for j in idx]
        breakout_dirs = [breakout_dirs[j] for j in idx]

    # Evaluate each breakout
    wins = 0
    evaluated = 0

    for sig_pos, direction in zip(breakout_indices, breakout_dirs):
        entry_pos = sig_pos + 1  # T+2
        if entry_pos >= len(df):
            continue

        atr_val = atr.iloc[sig_pos]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        entry_price = df.iloc[entry_pos]["open"]
        end_pos = min(entry_pos + K, len(df))
        if entry_pos >= end_pos:
            continue

        tp_level = min_move * atr_val
        sl_level = stop_dist * atr_val

        won = oc._evaluate_tp_sl(
            df, entry_pos, end_pos, entry_price, direction,
            tp_level, sl_level,
        )

        if won == 1.0:
            wins += 1
        evaluated += 1

    win_rate = wins / evaluated if evaluated > 0 else np.nan

    logger.info(
        "Unconditional breakout rate (K=%d, session=%s): %.4f "
        "(%d/%d evaluated, %d total breakouts, %d long / %d short)",
        K, session_hours, win_rate, wins, evaluated,
        n_total, n_long, n_short,
    )
    return {
        "n_breakouts": n_total,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": win_rate,
        "n_evaluated": evaluated,
    }
