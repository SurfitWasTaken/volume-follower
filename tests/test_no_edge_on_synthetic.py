"""
Synthetic data validation test — V2 Pipeline Overhaul.

Runs the full pipeline on pure GBM noise data (no signal embedded).
The test PASSES if and only if 0 statistically significant results
are found.  If any "edge" appears on pure noise, there's a bug.
"""

import sys
import os
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from synthetic_data import generate_synthetic_ohlcv
from signal_generator import SignalGenerator
from signal_filter import SignalFilter
from outcome_calculator import OutcomeCalculator
from statistical_tests import StatisticalTests


def test_no_edge_on_synthetic():
    """
    Run signal detection → filtering → outcome → stats on 150k
    synthetic GBM candles.  Assert that no significant edge is found.

    This validates that the pipeline does not produce false positives
    on data known to contain no tradeable signal.
    """
    np.random.seed(42)

    # Generate pure noise
    df = generate_synthetic_ohlcv(
        n_candles=150_000,
        timeframe_minutes=5,
        seed=42,
    )

    assert len(df) > 10_000, f"Synthetic data too short: {len(df)}"

    sig_gen = SignalGenerator()
    sig_filt = SignalFilter()
    oc = OutcomeCalculator()
    st = StatisticalTests(seed=42)

    # Test each variant with the most permissive and most restrictive filter ladders
    VARIANTS = ["A", "B", "C", "D"]
    K_VALUES = [5, 10, 20]

    filter_ladders = [
        [],  # No filters besides direction
        ["body_ratio", "wick", "session_both", "news", "trend", "range_position"],  # Full stack
    ]

    significant_results = []
    all_p_values = []

    for variant in VARIANTS:
        for sn in [False, True]:
            if variant == "D" and sn:
                continue

            signals = sig_gen.detect(df, variant, session_normalised=sn)

            for filters in filter_ladders:
                all_filters = ["direction"] + [f for f in filters if f != "direction"]
                filtered, directions = sig_filt.apply(df, signals, all_filters)
                n_signals = int(filtered.sum())

                if n_signals < 5:
                    continue

                for K in K_VALUES:
                    outcomes = oc.compute_outcomes(
                        df, filtered, directions, K_values=[K],
                    )

                    win_col = f"win_{K}"
                    wins = outcomes[win_col].dropna()
                    n_valid = len(wins)
                    n_wins = int((wins == 1.0).sum())

                    if n_valid < 5:
                        continue

                    base_rate = st.compute_base_rate(df, K)
                    p_val = st.binomial_test(n_wins, n_valid, base_rate)
                    all_p_values.append(p_val)

    # FDR correction
    if len(all_p_values) > 0:
        corrected = st.benjamini_hochberg(np.array(all_p_values))
        sig_count = (corrected < CONFIG["significance_level"]).sum()

        # The test passes if no significant results are found
        assert sig_count == 0, (
            f"PIPELINE BUG: {sig_count}/{len(corrected)} significant results "
            f"found on pure noise data! Minimum corrected p-value: "
            f"{corrected.min():.6f}. This indicates a look-ahead bias or "
            f"methodological error in the pipeline."
        )

    # Verify we actually ran enough tests
    assert len(all_p_values) >= 10, (
        f"Only {len(all_p_values)} tests ran — expected at least 10. "
        f"Check that signal generation is producing signals on synthetic data."
    )


def test_synthetic_data_properties():
    """Verify synthetic data has the expected properties."""
    df = generate_synthetic_ohlcv(n_candles=10_000, seed=123)

    assert len(df) == 10_000
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()

    # No look-ahead: close should be random walk
    returns = df["close"].pct_change().dropna()
    # Mean return should be approximately 0 (within 3 std errors)
    se = returns.std() / np.sqrt(len(returns))
    assert abs(returns.mean()) < 3 * se, (
        f"Synthetic returns have non-zero mean: {returns.mean():.6f} "
        f"(3σ = {3 * se:.6f})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
