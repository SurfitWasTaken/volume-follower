#!/usr/bin/env python3
"""
Volume-Gated Breakout Filter — Run Script

Tests whether volume spikes (Variant B) identify elevated-volatility
regimes where a T+1 price breakout has higher predictive power than
unconditional breakouts.

Test matrix: 4 K values × 4 session configs = 16 tests
Fixed: Variant B, body_ratio + wick filters on spike candle

Output: output/breakout_filter/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from data_loader import DataLoader
from signal_generator import SignalGenerator
from signal_filter import SignalFilter
from outcome_calculator import OutcomeCalculator
from statistical_tests import StatisticalTests
from red_flag_checker import check_all_results
from breakout_signal import detect_breakout_signals, compute_unconditional_breakout_rate

# ── Setup ────────────────────────────────────────────────────────────────

OUTPUT_DIR = "output/breakout_filter"
INSTRUMENT = "EUR_USD"
TIMEFRAME = "M5"
K_VALUES = [3, 5, 10, 20]

SESSION_CONFIGS = {
    "none": None,
    "london": set(range(8, 10)),     # 08:00-09:30 UTC
    "newyork": set(range(13, 16)),   # 13:30-15:00 UTC
    "both": set(range(8, 10)) | set(range(13, 16)),
}


def _setup_logging():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s"
    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(OUTPUT_DIR) / "pipeline.log", mode="w"),
        ],
    )


def _sharpe(arr):
    if len(arr) > 1 and arr.std() > 0:
        return arr.mean() / arr.std() * np.sqrt(252)
    return 0.0


def main():
    _setup_logging()
    log = logging.getLogger(__name__)
    np.random.seed(CONFIG["random_seed"])
    t0 = time.time()

    # ── Disable mean-reversion mode if it was left on ────────────────
    CONFIG["mean_reversion"] = False
    CONFIG["cc_enabled"] = False

    log.info("=" * 80)
    log.info("VOLUME-GATED BREAKOUT FILTER HYPOTHESIS TEST")
    log.info("  Volume spike → T+1 breakout confirmation → T+2 entry")
    log.info("  Variant B (multiplier=2.5, lookback=20)")
    log.info("  EUR_USD M5, IS period only")
    log.info("=" * 80)

    # ── Load data ────────────────────────────────────────────────────
    loader = DataLoader()
    df_full = loader.fetch_max_history(INSTRUMENT, TIMEFRAME)
    loader.validate_data(df_full)
    df_is, _ = loader.split_data(df_full)

    # Apply blackout buffer
    buffer = CONFIG["blackout_buffer_candles"]
    if len(df_is) > buffer:
        df_is = df_is.iloc[:-buffer]

    log.info("IS data: %d candles (%s → %s)",
             len(df_is), df_is.index[0], df_is.index[-1])

    # ── Step 1: Detect volume spikes (Variant B) ─────────────────────
    sig_gen = SignalGenerator()
    spikes = sig_gen.detect(df_is, "B", session_normalised=False)
    n_spikes_raw = int(spikes.sum())
    log.info("Volume spikes detected (Variant B): %d", n_spikes_raw)

    # ── Step 2: Apply body_ratio + wick filters to spike candle ──────
    sig_filt = SignalFilter()
    # We need a temporary direction classification for the filters
    # (body_ratio and wick operate on the candle shape, not direction)
    # Use the original direction filter first, then apply body/wick
    spike_filtered, spike_dirs = sig_filt.apply(
        df_is, spikes, ["direction", "body_ratio", "wick"]
    )
    n_spikes_filtered = int(spike_filtered.sum())
    log.info("After body_ratio + wick filter: %d spikes", n_spikes_filtered)

    # ── Step 3: Apply breakout detection on T+1 ──────────────────────
    breakout_signals, breakout_dirs = detect_breakout_signals(
        df_is, spike_filtered
    )
    n_breakouts = int(breakout_signals.sum())
    log.info("After breakout filter: %d tradeable signals (%.1f%% of filtered spikes)",
             n_breakouts, 100 * n_breakouts / max(n_spikes_filtered, 1))

    # ── Step 4: Compute unconditional breakout rates ─────────────────
    log.info("=" * 80)
    log.info("COMPUTING UNCONDITIONAL BREAKOUT RATES")
    log.info("=" * 80)

    unconditional_rates = {}
    for session_name, session_hours in SESSION_CONFIGS.items():
        for K in K_VALUES:
            key = f"{session_name}_K{K}"
            unconditional_rates[key] = compute_unconditional_breakout_rate(
                df_is, K, session_hours=session_hours,
            )

    # ── Step 5: Run test matrix ──────────────────────────────────────
    log.info("=" * 80)
    log.info("RUNNING TEST MATRIX: %d tests", len(K_VALUES) * len(SESSION_CONFIGS))
    log.info("=" * 80)

    oc = OutcomeCalculator()
    st = StatisticalTests()
    all_results = []

    for session_name, session_hours in SESSION_CONFIGS.items():
        # Apply session filter to breakout signals (T+1 candle time)
        if session_hours is not None:
            session_mask = breakout_signals & df_is.index.to_series().apply(
                lambda t: t.hour in session_hours
            )
        else:
            session_mask = breakout_signals.copy()

        # Apply session filter to directions
        session_dirs = breakout_dirs.where(session_mask, 0)
        n_session = int(session_mask.sum())

        for K in K_VALUES:
            log.info("─── Session=%s, K=%d, n=%d ───", session_name, K, n_session)

            if n_session < 5:
                log.info("  Skipping — only %d signals", n_session)
                result = _empty_result(session_name, K, n_session)
                all_results.append(result)
                continue

            # Compute outcomes (entry at T+2 because signal is at T+1)
            outcomes = oc.compute_outcomes(
                df_is, session_mask, session_dirs, K_values=[K]
            )
            outcomes = oc.apply_costs(outcomes, INSTRUMENT, K=K)

            win_col = f"win_{K}"
            htc_col = f"htc_return_{K}"
            htc_net_col = f"htc_return_net_{K}"

            wins = outcomes[win_col].dropna()
            n_valid = len(wins)
            n_wins = int((wins == 1.0).sum())
            win_rate = n_wins / n_valid if n_valid > 0 else np.nan

            # Base rate: unconditional breakout win rate
            uncond_key = f"{session_name}_K{K}"
            base_rate = unconditional_rates[uncond_key]["win_rate"]
            uncond_n = unconditional_rates[uncond_key]["n_evaluated"]

            # Binomial test against unconditional breakout rate
            p_raw = st.binomial_test(n_wins, n_valid, base_rate) if not np.isnan(base_rate) else 1.0

            # Bootstrap CIs
            try:
                wr_point, wr_lo, wr_hi = st.bootstrap_ci(
                    (wins == 1.0).astype(float).values, statistic=np.mean,
                )
            except Exception:
                wr_point, wr_lo, wr_hi = win_rate, np.nan, np.nan

            gross_returns = outcomes[htc_col].dropna().values
            net_returns = outcomes[htc_net_col].dropna().values if htc_net_col in outcomes.columns else gross_returns

            sharpe_gross = _sharpe(gross_returns)
            sharpe_net = _sharpe(net_returns)

            try:
                _, sr_lo, sr_hi = st.bootstrap_ci(net_returns, statistic=_sharpe)
            except Exception:
                sr_lo, sr_hi = np.nan, np.nan

            # Cohen's h
            try:
                h_val = 2 * (np.arcsin(np.sqrt(win_rate)) - np.arcsin(np.sqrt(base_rate)))
            except Exception:
                h_val = np.nan

            # MFE / MAE
            mfe_col = f"mfe_{K}"
            mae_col = f"mae_{K}"
            avg_mfe = outcomes[mfe_col].mean() if mfe_col in outcomes.columns else np.nan
            avg_mae = outcomes[mae_col].mean() if mae_col in outcomes.columns else np.nan
            ratio = avg_mfe / avg_mae if avg_mae and avg_mae > 0 else np.nan

            # Sample warning
            n_min = CONFIG.get("n_min_signals", 150)
            n_min_chart = CONFIG.get("n_min_signals_chart", 100)
            if n_valid < n_min_chart:
                sample_warning = "INSUFFICIENT"
            elif n_valid < n_min:
                sample_warning = "UNDERPOWERED"
            else:
                sample_warning = ""

            result = {
                "instrument": INSTRUMENT,
                "timeframe": TIMEFRAME,
                "variant": "B_breakout",
                "filters_applied": f"body_ratio+wick+breakout+session_{session_name}",
                "K": K,
                "n_signals": n_valid,
                "win_rate": win_rate,
                "base_rate": base_rate,
                "base_rate_n": uncond_n,
                "win_rate_excess": win_rate - base_rate if not np.isnan(win_rate) and not np.isnan(base_rate) else np.nan,
                "p_value_raw": p_raw,
                "p_value_corrected": np.nan,  # filled after FDR
                "cohens_h": h_val,
                "sample_warning": sample_warning,
                "avg_return_gross": float(np.nanmean(gross_returns)) if len(gross_returns) > 0 else np.nan,
                "avg_return_net": float(np.nanmean(net_returns)) if len(net_returns) > 0 else np.nan,
                "sharpe_gross": sharpe_gross,
                "sharpe_net": sharpe_net,
                "avg_mfe_atr": avg_mfe,
                "avg_mae_atr": avg_mae,
                "mfe_mae_ratio": ratio,
                "win_rate_ci_lo": wr_lo,
                "win_rate_ci_hi": wr_hi,
                "sharpe_ci_lo": sr_lo,
                "sharpe_ci_hi": sr_hi,
            }
            all_results.append(result)

    # ── Step 6: FDR correction + Red flags ───────────────────────────
    log.info("=" * 80)
    log.info("FDR CORRECTION + RED FLAGS (%d tests)", len(all_results))
    log.info("=" * 80)

    raw_pvals = np.array([r["p_value_raw"] for r in all_results])
    corrected = st.benjamini_hochberg(raw_pvals)
    for i, r in enumerate(all_results):
        r["p_value_corrected"] = corrected[i]

    red_flag_reports = check_all_results(all_results)

    # ── Step 7: Save results ─────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Summary table → %s  (%d rows)", summary_path, len(summary_df))

    # Save unconditional rates
    uncond_path = Path(OUTPUT_DIR) / "unconditional_breakout_rates.json"
    # Convert numpy types to Python types for JSON serialization
    uncond_serializable = {}
    for k, v in unconditional_rates.items():
        uncond_serializable[k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv
            for kk, vv in v.items()
        }
    uncond_path.write_text(json.dumps(uncond_serializable, indent=2, default=str))
    log.info("Unconditional rates → %s", uncond_path)

    # ── Step 8: Print results ────────────────────────────────────────
    log.info("=" * 80)
    log.info("RESULTS SUMMARY")
    log.info("=" * 80)

    sig_level = CONFIG["significance_level"]
    sig_count = (summary_df["p_value_corrected"] < sig_level).sum()
    log.info("%d / %d tests significant after FDR correction (α=%.2f)",
             sig_count, len(summary_df), sig_level)

    # Signal funnel
    log.info("")
    log.info("SIGNAL FUNNEL:")
    log.info("  Raw volume spikes (Variant B):     %d", n_spikes_raw)
    log.info("  After body_ratio + wick filter:     %d", n_spikes_filtered)
    log.info("  After T+1 breakout filter:          %d (%.1f%% tradeable)",
             n_breakouts, 100 * n_breakouts / max(n_spikes_filtered, 1))

    # Top results
    valid_results = summary_df.dropna(subset=["sharpe_net"])
    if not valid_results.empty:
        top = valid_results.nlargest(5, "sharpe_net")
        log.info("")
        log.info("TOP 5 BY NET SHARPE:")
        for _, row in top.iterrows():
            log.info(
                "  %s K=%d n=%d WR=%.3f base=%.3f excess=%.3f "
                "p_raw=%.4f p_fdr=%.4f h=%.3f Sharpe_net=%.3f",
                row["filters_applied"], row["K"], row["n_signals"],
                row["win_rate"], row["base_rate"], row["win_rate_excess"],
                row["p_value_raw"], row["p_value_corrected"],
                row["cohens_h"], row["sharpe_net"],
            )

    # Volume-gated vs unconditional comparison
    log.info("")
    log.info("VOLUME-GATED vs UNCONDITIONAL BREAKOUT COMPARISON:")
    for session_name in SESSION_CONFIGS:
        for K in K_VALUES:
            key = f"{session_name}_K{K}"
            uncond = unconditional_rates[key]
            gated_row = summary_df[
                (summary_df["filters_applied"].str.contains(f"session_{session_name}"))
                & (summary_df["K"] == K)
            ]
            if not gated_row.empty and not np.isnan(gated_row.iloc[0]["win_rate"]):
                gated_wr = gated_row.iloc[0]["win_rate"]
                uncond_wr = uncond["win_rate"]
                delta = gated_wr - uncond_wr if not np.isnan(uncond_wr) else np.nan
                log.info(
                    "  %s K=%d: gated=%.4f uncond=%.4f delta=%+.4f "
                    "(volume adds %s)",
                    session_name, K, gated_wr, uncond_wr,
                    delta if not np.isnan(delta) else 0.0,
                    "value" if delta and delta > 0.01 else "nothing",
                )

    # Red flag summary
    n_reportable = sum(1 for r in red_flag_reports if r.is_reportable)
    n_with_flags = sum(1 for r in red_flag_reports if r.n_flags > 0)
    n_clean = sum(
        1 for r in red_flag_reports
        if r.n_critical == 0 and r.n_flags == 0
    )
    log.info("")
    log.info("RED FLAGS: %d/%d have flags, %d reportable, %d clean (no flags)",
             n_with_flags, len(red_flag_reports), n_reportable, n_clean)

    log.info("")
    log.info("PIPELINE COMPLETE (%.0fs total)", time.time() - t0)


def _empty_result(session_name, K, n_signals):
    return {
        "instrument": INSTRUMENT, "timeframe": TIMEFRAME,
        "variant": "B_breakout",
        "filters_applied": f"body_ratio+wick+breakout+session_{session_name}",
        "K": K, "n_signals": n_signals,
        "win_rate": np.nan, "base_rate": np.nan, "base_rate_n": 0,
        "win_rate_excess": np.nan,
        "p_value_raw": 1.0, "p_value_corrected": np.nan,
        "cohens_h": np.nan, "sample_warning": "INSUFFICIENT",
        "avg_return_gross": np.nan, "avg_return_net": np.nan,
        "sharpe_gross": np.nan, "sharpe_net": np.nan,
        "avg_mfe_atr": np.nan, "avg_mae_atr": np.nan,
        "mfe_mae_ratio": np.nan,
        "win_rate_ci_lo": np.nan, "win_rate_ci_hi": np.nan,
        "sharpe_ci_lo": np.nan, "sharpe_ci_hi": np.nan,
    }


if __name__ == "__main__":
    main()
