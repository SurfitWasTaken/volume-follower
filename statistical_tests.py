from __future__ import annotations
"""
Statistical inference module.

Implements all required tests with methodological corrections:

1. Session-matched base rate (not global random) — controls for session effect
2. One-sided binomial test with Bonferroni correction
3. Benjamini-Hochberg FDR correction across the full test matrix
4. Block bootstrap (not IID) — handles serial correlation
5. Walk-forward temporal stability check (NOT optimisation)
6. Permutation test (shuffle volume, preserve price structure)
7. Stationarity / half-split consistency check
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import CONFIG

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Statistical inference for volume-spike signal evaluation."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed or CONFIG["random_seed"])

    # ─── 1. Session-matched base rate ─────────────────────────────────────

    def compute_base_rate(
        self,
        df: pd.DataFrame,
        K: int,
        min_move_atr: float | None = None,
        stop_atr: float | None = None,
        atr_period: int | None = None,
        session_hours: set[int] | None = None,
    ) -> float:
        """
        Compute the unconditional directional hit rate for candles within
        the SAME session window as the signal — NOT a global average.

        This is the correct counterfactual: "what fraction of candles in
        this session window hit TP before SL, regardless of volume?"

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        K : int
            Forward window.
        min_move_atr : float
            TP distance in ATR units.
        stop_atr : float
            SL distance in ATR units.
        atr_period : int
            ATR lookback.
        session_hours : set[int] or None
            If provided, only sample candles whose hour is in this set.
            If None, use all candles (global baseline).

        Returns
        -------
        float  — base rate (0–1).
        """
        from outcome_calculator import OutcomeCalculator

        min_move_atr = min_move_atr if min_move_atr is not None else CONFIG["min_move_atr"]
        stop_atr = stop_atr if stop_atr is not None else CONFIG["stop_distance_atr"]
        atr_period = atr_period or CONFIG["atr_period"]

        oc = OutcomeCalculator()
        atr = oc.compute_atr(df, atr_period)

        # Filter to session hours if provided
        if session_hours is not None:
            mask = df.index.hour.isin(session_hours)
        else:
            mask = pd.Series(True, index=df.index)

        eligible = df.index[mask]

        # Sample up to 5000 random candles for efficiency
        n_sample = min(5000, len(eligible))
        if n_sample == 0:
            logger.warning("No candles in session window for base rate.")
            return 0.5

        sample_times = self.rng.choice(eligible, size=n_sample, replace=False)

        wins = 0
        valid = 0
        pos_map = {t: i for i, t in enumerate(df.index)}
        for t in sample_times:
            sig_pos = pos_map[t]
            entry_pos = sig_pos + 1
            end_pos = min(entry_pos + K, len(df))
            if entry_pos >= end_pos:
                continue

            atr_val = atr.iloc[sig_pos]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            entry_price = df.iloc[entry_pos]["open"]
            tp = min_move_atr * atr_val
            sl = stop_atr * atr_val

            # Test BOTH directions and average (since null = no directional info)
            for direction in [1, -1]:
                result = oc._evaluate_tp_sl(df, entry_pos, end_pos, entry_price, direction, tp, sl)
                wins += result
                valid += 1

        base_rate = wins / valid if valid > 0 else 0.5
        logger.info(
            "Base rate (K=%d, session_hours=%s): %.4f (%d samples)",
            K, session_hours, base_rate, valid,
        )
        return base_rate

    # ─── 2. Binomial test ─────────────────────────────────────────────────

    @staticmethod
    def binomial_test(wins: int, n: int, base_rate: float) -> float:
        """
        One-sided binomial test.
        H₀: p ≤ base_rate.  H₁: p > base_rate.

        Returns p-value.
        """
        if n == 0:
            return 1.0
        result = sp_stats.binomtest(wins, n, base_rate, alternative="greater")
        return result.pvalue

    # ─── 2b. Bonferroni correction ────────────────────────────────────────

    @staticmethod
    def bonferroni_correction(p_values: np.ndarray, n_tests: int | None = None) -> np.ndarray:
        """Bonferroni-correct an array of p-values."""
        n = n_tests or len(p_values)
        return np.minimum(np.asarray(p_values) * n, 1.0)

    # ─── 3. Benjamini-Hochberg FDR ────────────────────────────────────────

    @staticmethod
    def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction.

        Returns array of adjusted p-values.
        """
        p = np.asarray(p_values, dtype=float)
        n = len(p)
        if n == 0:
            return p

        order = np.argsort(p)
        ranked = np.empty_like(p)
        ranked[order] = np.arange(1, n + 1)

        adjusted = p * n / ranked
        # Enforce monotonicity (from largest rank down)
        adjusted_sorted = adjusted[np.argsort(-ranked)]
        for i in range(1, len(adjusted_sorted)):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i - 1])
        adjusted = adjusted_sorted[np.argsort(np.argsort(-ranked))]

        return np.minimum(adjusted, 1.0)

    # ─── 4. Block bootstrap CI ────────────────────────────────────────────

    def bootstrap_ci(
        self,
        values: np.ndarray,
        statistic: Callable = np.mean,
        n_resamples: int | None = None,
        block_size: int | None = None,
        ci: float = 0.95,
    ) -> tuple[float, float, float]:
        """
        Compute a confidence interval using block bootstrap.

        Uses contiguous blocks of `block_size` observations to preserve
        serial correlation structure.  This prevents artificially narrow
        CIs that result from IID bootstrap on correlated data.

        Parameters
        ----------
        values : array-like
            Observed metric values (one per signal, in temporal order).
        statistic : callable
            Aggregation function (np.mean, np.median, etc.).
        n_resamples : int
            Number of bootstrap resamples.
        block_size : int
            Size of contiguous blocks to resample.
        ci : float
            Confidence level (default 0.95 → 95% CI).

        Returns
        -------
        (point_estimate, ci_lower, ci_upper)
        """
        n_resamples = n_resamples or CONFIG["bootstrap_n"]
        block_size = block_size or CONFIG["bootstrap_block_size"]
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        n = len(values)

        if n == 0:
            return np.nan, np.nan, np.nan

        point = float(statistic(values))

        if n < block_size:
            # Fall back to IID bootstrap if too few observations
            block_size = 1

        n_blocks = int(np.ceil(n / block_size))
        boot_stats = np.empty(n_resamples)

        for b in range(n_resamples):
            # Draw random block start indices
            starts = self.rng.integers(0, n - block_size + 1, size=n_blocks)
            sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
            boot_stats[b] = statistic(sample)

        alpha = 1 - ci
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        logger.debug("Bootstrap CI: %.4f [%.4f, %.4f]", point, lo, hi)
        return point, lo, hi

    # ─── 5. Walk-forward stability check ──────────────────────────────────

    @staticmethod
    def walk_forward(
        df: pd.DataFrame,
        signal_func: Callable,
        filter_func: Callable,
        outcome_func: Callable,
        K: int = 10,
        train_months: int | None = None,
        test_months: int | None = None,
    ) -> list[dict]:
        """
        Rolling walk-forward temporal stability check.

        This is explicitly NOT a walk-forward optimisation — all parameters
        are fixed from CONFIG.  The purpose is to assess whether the signal's
        edge is temporally stable or concentrated in a few lucky windows.

        For each fold:
        1. Train window: generate session-hour profile (for Variant D) on
           training data only.
        2. Test window: apply signal + filters + outcomes on test data.
        3. Record Sharpe ratio and win rate for the test window.

        Parameters
        ----------
        df : pd.DataFrame
            Full in-sample OHLCV data.
        signal_func : callable(df) → signals
        filter_func : callable(df, signals) → (filtered_signals, directions)
        outcome_func : callable(df, signals, dirs) → outcomes_df
        K : int
            Forward window for outcomes.
        train_months, test_months : int

        Returns
        -------
        list[dict]  — one dict per fold with 'fold', 'start', 'end',
                       'n_signals', 'win_rate', 'sharpe', 'avg_return'.
        """
        train_m = train_months or CONFIG["walk_forward_train_months"]
        test_m = test_months or CONFIG["walk_forward_test_months"]

        results = []
        start = df.index.min()
        end = df.index.max()

        fold = 0
        cursor = start + pd.DateOffset(months=train_m)

        while cursor + pd.DateOffset(months=test_m) <= end:
            test_start = cursor
            test_end = cursor + pd.DateOffset(months=test_m)
            train_start = cursor - pd.DateOffset(months=train_m)

            df_train = df[train_start:test_start]
            df_test = df[test_start:test_end]

            if len(df_test) < 100:
                cursor += pd.DateOffset(months=1)
                fold += 1
                continue

            # Generate signals on test data
            # (Parameters are fixed — nothing is "trained" here)
            try:
                signals = signal_func(df_test)
                filtered, directions = filter_func(df_test, signals)
                outcomes = outcome_func(df_test, filtered, directions)
            except Exception as e:
                logger.warning("Walk-forward fold %d failed: %s", fold, e)
                cursor += pd.DateOffset(months=1)
                fold += 1
                continue

            win_col = f"win_{K}"
            htc_col = f"htc_return_{K}"

            if len(outcomes) == 0 or win_col not in outcomes.columns:
                results.append({
                    "fold": fold, "start": str(test_start.date()),
                    "end": str(test_end.date()), "n_signals": 0,
                    "win_rate": np.nan, "sharpe": np.nan, "avg_return": np.nan,
                })
            else:
                wins = outcomes[win_col].dropna()
                returns = outcomes[htc_col].dropna()
                wr = (wins == 1.0).mean() if len(wins) > 0 else np.nan
                sr = (
                    returns.mean() / returns.std() * np.sqrt(252)
                    if len(returns) > 1 and returns.std() > 0 else np.nan
                )
                results.append({
                    "fold": fold, "start": str(test_start.date()),
                    "end": str(test_end.date()), "n_signals": len(outcomes),
                    "win_rate": wr, "sharpe": sr, "avg_return": returns.mean(),
                })

            cursor += pd.DateOffset(months=1)
            fold += 1

        logger.info("Walk-forward: %d folds completed.", len(results))
        return results

    # ─── 6. Permutation test ─────────────────────────────────────────────

    def permutation_test(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
        filter_func: Callable,
        outcome_func: Callable,
        K: int = 10,
        n_perms: int | None = None,
    ) -> dict:
        """
        Shuffle the volume column (breaking volume-price relationship) and
        recompute strategy Sharpe.  The true Sharpe should sit in the top
        5th percentile if the signal has genuine information content.

        Parameters
        ----------
        df : pd.DataFrame
        signal_func, filter_func, outcome_func : callables
        K : int
        n_perms : int

        Returns
        -------
        dict with 'true_sharpe', 'perm_sharpes', 'empirical_p_value'.
        """
        n_perms = n_perms or CONFIG["permutation_n"]

        # True Sharpe
        signals = signal_func(df)
        filtered, directions = filter_func(df, signals)
        outcomes = outcome_func(df, filtered, directions)
        htc_col = f"htc_return_{K}"

        if len(outcomes) == 0 or htc_col not in outcomes.columns:
            return {"true_sharpe": np.nan, "perm_sharpes": [], "empirical_p_value": 1.0}

        returns = outcomes[htc_col].dropna()
        true_sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if len(returns) > 1 and returns.std() > 0 else 0.0
        )

        # Permutation distribution
        perm_sharpes = []
        for i in range(n_perms):
            df_perm = df.copy()
            df_perm["volume"] = self.rng.permutation(df_perm["volume"].values)

            try:
                sig_p = signal_func(df_perm)
                filt_p, dir_p = filter_func(df_perm, sig_p)
                out_p = outcome_func(df_perm, filt_p, dir_p)

                if len(out_p) > 0 and htc_col in out_p.columns:
                    r = out_p[htc_col].dropna()
                    sr = r.mean() / r.std() * np.sqrt(252) if len(r) > 1 and r.std() > 0 else 0.0
                else:
                    sr = 0.0
            except Exception:
                sr = 0.0
            perm_sharpes.append(sr)

            if (i + 1) % 100 == 0:
                logger.debug("  Permutation %d/%d", i + 1, n_perms)

        perm_arr = np.array(perm_sharpes)
        emp_p = (perm_arr >= true_sharpe).mean()

        logger.info(
            "Permutation test: true Sharpe=%.3f, emp. p=%.4f, "
            "perm median=%.3f",
            true_sharpe, emp_p, np.median(perm_arr),
        )
        return {
            "true_sharpe": true_sharpe,
            "perm_sharpes": perm_sharpes,
            "empirical_p_value": float(emp_p),
        }

    # ─── 7. Stationarity / half-split check ──────────────────────────────

    @staticmethod
    def stationarity_check(
        outcomes: pd.DataFrame,
        K: int = 10,
    ) -> dict:
        """
        Split outcomes into first-half and second-half (by signal_time).
        Compare Sharpe ratios and win rates across halves.

        A large divergence suggests the edge is decaying or non-stationary.
        """
        if len(outcomes) < 10:
            return {"first_half_sharpe": np.nan, "second_half_sharpe": np.nan,
                    "sharpe_diff": np.nan, "first_wr": np.nan, "second_wr": np.nan}

        mid = len(outcomes) // 2
        h1 = outcomes.iloc[:mid]
        h2 = outcomes.iloc[mid:]

        htc = f"htc_return_{K}"
        win = f"win_{K}"

        def _sharpe(s):
            s = s.dropna()
            return s.mean() / s.std() * np.sqrt(252) if len(s) > 1 and s.std() > 0 else np.nan

        s1 = _sharpe(h1[htc]) if htc in h1.columns else np.nan
        s2 = _sharpe(h2[htc]) if htc in h2.columns else np.nan

        wr1 = (h1[win] == 1.0).mean() if win in h1.columns else np.nan
        wr2 = (h2[win] == 1.0).mean() if win in h2.columns else np.nan

        result = {
            "first_half_sharpe": s1,
            "second_half_sharpe": s2,
            "sharpe_diff": abs(s1 - s2) if not (np.isnan(s1) or np.isnan(s2)) else np.nan,
            "first_wr": wr1,
            "second_wr": wr2,
        }
        logger.info("Stationarity: H1 Sharpe=%.3f WR=%.3f | H2 Sharpe=%.3f WR=%.3f",
                     s1, wr1, s2, wr2)
        return result
