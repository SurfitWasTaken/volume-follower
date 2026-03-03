from __future__ import annotations
"""
Outcome calculation module.

For each detected signal, compute forward-looking performance metrics:
- Directional accuracy (ATR-based TP / SL)
- Maximum Favourable / Adverse Excursion (MFE / MAE)
- Hold-to-close return
- Time-to-peak
- Transaction cost adjustment (OANDA spread-based, not commission-based)
"""

import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class OutcomeCalculator:
    """Compute forward-looking outcomes for volume-spike signals."""

    # ----- ATR calculation ------------------------------------------------

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int | None = None) -> pd.Series:
        """
        Compute Average True Range (Wilder smoothing) using only prior data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        period : int
            ATR lookback period.  Default from CONFIG.

        Returns
        -------
        pd.Series   ATR values aligned to df.index.
        """
        period = period or CONFIG["atr_period"]

        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Wilder / EMA-style smoothing
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return atr

    # ----- Main outcome computation --------------------------------------

    def compute_outcomes(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        directions: pd.Series,
        K_values: list[int] | None = None,
        atr_period: int | None = None,
        min_move_atr: float | None = None,
        stop_atr: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute forward-looking trading outcomes for every signal.

        IMPORTANT: Entry is assumed at the NEXT candle's open after the
        signal candle — never the signal candle itself.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV dataset.
        signals : pd.Series[bool]
            Signal mask.
        directions : pd.Series[int]
            +1 (long) or -1 (short) for each signal.
        K_values : list[int]
            Forward windows to evaluate.
        atr_period, min_move_atr, stop_atr : overrides.

        Returns
        -------
        pd.DataFrame  with one row per signal and columns including:
            signal_time, entry_time, entry_price, direction, atr,
            and per-K columns: win_{K}, mfe_{K}, mae_{K}, htc_return_{K},
            time_to_peak_{K}.
        """
        K_values = K_values or CONFIG["K_values"]
        atr_period = atr_period or CONFIG["atr_period"]
        min_move = min_move_atr if min_move_atr is not None else CONFIG["min_move_atr"]
        stop_dist = stop_atr if stop_atr is not None else CONFIG["stop_distance_atr"]

        atr = self.compute_atr(df, atr_period)

        signal_indices = df.index[signals]
        pos_map = {t: i for i, t in enumerate(df.index)}

        records = []
        for sig_time in signal_indices:
            sig_pos = pos_map[sig_time]
            entry_pos = sig_pos + 1  # next candle
            if entry_pos >= len(df):
                continue

            entry_price = df.iloc[entry_pos]["open"]
            direction = directions.loc[sig_time]
            if direction == 0:
                continue

            atr_at_signal = atr.iloc[sig_pos]
            if pd.isna(atr_at_signal) or atr_at_signal <= 0:
                continue

            rec = {
                "signal_time": sig_time,
                "entry_time": df.index[entry_pos],
                "entry_price": entry_price,
                "direction": direction,
                "atr": atr_at_signal,
                "hour": sig_time.hour,
                "day_of_week": sig_time.weekday(),
            }

            for K in K_values:
                end_pos = min(entry_pos + K, len(df))
                if entry_pos >= end_pos:
                    rec[f"win_{K}"] = np.nan
                    rec[f"mfe_{K}"] = np.nan
                    rec[f"mae_{K}"] = np.nan
                    rec[f"mfe_mae_ratio_{K}"] = np.nan
                    rec[f"htc_return_{K}"] = np.nan
                    rec[f"time_to_peak_{K}"] = np.nan
                    continue

                fwd = df.iloc[entry_pos:end_pos]

                if direction == 1:  # long
                    fwd_moves = fwd["high"] - entry_price
                    adv_moves = entry_price - fwd["low"]
                    mfe = fwd_moves.max()
                    mae = adv_moves.max()
                    peak_idx = fwd_moves.idxmax()
                    exit_close = fwd["close"].iloc[-1]
                    htc = exit_close - entry_price
                else:  # short
                    fwd_moves = entry_price - fwd["low"]
                    adv_moves = fwd["high"] - entry_price
                    mfe = fwd_moves.max()
                    mae = adv_moves.max()
                    peak_idx = fwd_moves.idxmax()
                    exit_close = fwd["close"].iloc[-1]
                    htc = entry_price - exit_close

                # Normalise to ATR units
                mfe_atr = mfe / atr_at_signal
                mae_atr = mae / atr_at_signal
                htc_atr = htc / atr_at_signal

                # Win / loss determination (bar-by-bar)
                tp_level = min_move * atr_at_signal
                sl_level = stop_dist * atr_at_signal

                won = self._evaluate_tp_sl(
                    df, entry_pos, end_pos, entry_price, direction,
                    tp_level, sl_level,
                )

                time_to_peak = (
                    pos_map.get(peak_idx, entry_pos) - entry_pos
                    if peak_idx is not None else K
                )

                rec[f"win_{K}"] = won
                rec[f"mfe_{K}"] = mfe_atr
                rec[f"mae_{K}"] = mae_atr
                rec[f"mfe_mae_ratio_{K}"] = (
                    mfe_atr / mae_atr if mae_atr > 0 else np.inf
                )
                rec[f"htc_return_{K}"] = htc_atr
                rec[f"time_to_peak_{K}"] = time_to_peak

            records.append(rec)

        outcomes = pd.DataFrame(records)
        logger.info("Computed outcomes for %d signals.", len(outcomes))
        return outcomes

    # ----- TP / SL evaluation (bar-by-bar) --------------------------------

    @staticmethod
    def _evaluate_tp_sl(
        df: pd.DataFrame,
        entry_pos: int,
        end_pos: int,
        entry_price: float,
        direction: int,
        tp_dist: float,
        sl_dist: float,
    ) -> float:
        """
        Walk forward bar-by-bar.  Return 1.0 if TP is hit first,
        0.0 if SL is hit first, and 0.5 if neither is hit (undecided).
        """
        for i in range(entry_pos, end_pos):
            bar = df.iloc[i]
            if direction == 1:
                if bar["high"] >= entry_price + tp_dist:
                    if bar["low"] <= entry_price - sl_dist:
                        return 0.0  # both hit → assume worst case
                    return 1.0
                if bar["low"] <= entry_price - sl_dist:
                    return 0.0
            else:
                if bar["low"] <= entry_price - tp_dist:
                    if bar["high"] >= entry_price + sl_dist:
                        return 0.0
                    return 1.0
                if bar["high"] >= entry_price + sl_dist:
                    return 0.0
        return 0.5  # neither hit within K candles

    # ----- Transaction cost model (OANDA spread-based) --------------------

    @staticmethod
    def apply_costs(
        outcomes: pd.DataFrame,
        instrument: str,
        K: int = 10,
    ) -> pd.DataFrame:
        """
        Subtract OANDA-appropriate transaction costs from returns.

        OANDA has NO per-trade commission.  All costs are embedded in
        the bid-ask spread.  We model:
          1. Instrument-specific typical spread (in price units).
          2. A widening factor for session-open entries (fast markets).
          3. Round-trip = 1× spread (entry) + 1× spread (exit) = 2× spread.

        The cost is expressed in ATR units for comparability.

        Parameters
        ----------
        outcomes : pd.DataFrame  (must have 'atr', 'hour', f'htc_return_{K}')
        instrument : str         OANDA instrument code (e.g. 'EUR_USD')
        K : int                  forward window

        Returns
        -------
        Same DataFrame with added column f'htc_return_net_{K}'.
        """
        inst_cfg = CONFIG["instruments"].get(instrument, {})
        spread_pips = inst_cfg.get("spread_pips", 1.0)
        pip_value = inst_cfg.get("pip", 0.0001)
        widen = CONFIG["spread_widening_factor"]

        # Base spread in price units
        base_spread = spread_pips * pip_value

        # Widen spread during session-open hours
        session_hours = set()
        for h_start, m_start, h_end, _m_end in CONFIG["session_windows"].values():
            session_hours.update(range(h_start, h_end + 1))

        in_session = outcomes["hour"].isin(session_hours)
        spread = pd.Series(base_spread, index=outcomes.index)
        spread[in_session] = base_spread * widen

        # Round-trip cost in ATR units
        atr = outcomes["atr"]
        cost_atr = (2 * spread) / atr

        htc_col = f"htc_return_{K}"
        outcomes[f"htc_return_net_{K}"] = outcomes[htc_col] - cost_atr
        outcomes[f"cost_atr_{K}"] = cost_atr

        logger.info(
            "Costs applied (K=%d, %s): mean spread=%.1f pips, "
            "mean cost=%.4f ATR",
            K, instrument, spread.mean() / pip_value, cost_atr.mean(),
        )
        return outcomes
