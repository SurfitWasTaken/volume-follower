from __future__ import annotations
"""
Volume Spike Directional Prediction — Main Pipeline (V2 Overhaul)

Entry point: run_full_analysis()

V2 ARCHITECTURE:
    Stage 0: Pre-commitment lock (hash test matrix)
    Stage 1: Cost viability pre-flight
    Stage 2: Data loading + density validation
    Stage 3: Sample size estimation (HALT if insufficient)
    Stage 4-6: Signal → Filter → Outcome → Stats (per-test loop)
    Stage 7: Red flag assessment → CHECKPOINT
    Stage 8: Walk-forward (only for permutation-passing results)
    Stage 9: Reporting → pre-commitment verify → output
"""

import argparse
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add project root to path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from data_loader import DataLoader
from synthetic_data import generate_synthetic_ohlcv
from signal_generator import SignalGenerator
from signal_filter import SignalFilter
from outcome_calculator import OutcomeCalculator
from statistical_tests import StatisticalTests, InsufficientDataError
from performance_reporter import PerformanceReporter
from pre_commitment_log import lock_pre_commitment, verify_pre_commitment
from cost_viability_analyser import compute_cost_viability
from sample_size_estimator import estimate_sample_size
from red_flag_checker import check_red_flags, check_all_results

from cross_currency_loader import load_secondary
from cross_currency_features import compute_cross_currency_features
from adaptive_position_sizer import compute_position_sizes

# ── Logging setup ───────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s"
    ldir = log_dir or CONFIG["output_dir"]
    Path(ldir).mkdir(parents=True, exist_ok=True)

    # Remove existing handlers to avoid duplicates during dual runs
    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(ldir) / "pipeline.log", mode="w"),
        ],
    )

logger = logging.getLogger(__name__)

# ── Timeframe helpers ───────────────────────────────────────────────────

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}


def _session_hours_from_filters(filters: list[str]) -> set[int] | None:
    """Extract session hours from filter list for base-rate matching."""
    for f in filters:
        if f.startswith("session"):
            if "london" in f:
                return set(range(8, 10))
            elif "newyork" in f:
                return set(range(13, 16))
            else:  # both
                return set(range(8, 10)) | set(range(13, 16))
    return None  # no session filter → global base rate


# ── Test matrix definition ──────────────────────────────────────────────

# Pre-committed: EVERY combination is run.  No conditional expansion.
VARIANTS = ["A", "B", "C", "D"]
SESSION_NORM_OPTIONS = [False, True]  # True ignored for D (built-in)

# Cumulative filter ladders to test (each is additive)
FILTER_LADDERS = [
    [],                                              # volume only
    ["body_ratio"],                                  # + conviction
    ["body_ratio", "wick"],                          # + wick rejection
    ["body_ratio", "wick", "session_both"],           # + session window
    ["body_ratio", "wick", "session_london"],         # London only
    ["body_ratio", "wick", "session_newyork"],        # NY only
    ["body_ratio", "wick", "session_both", "news"],   # + news exclusion
    ["body_ratio", "wick", "session_both", "news", "trend"],  # + trend
    ["body_ratio", "wick", "session_both", "news", "trend", "range_position"],  # full stack
]

K_TO_TEST = [5, 10, 20]  # focused subset; expand if compute allows


# ── Pipeline checkpoint logging ─────────────────────────────────────────

def _write_checkpoint(stage: str, status: str, details: dict | None = None,
                      output_dir: str | None = None) -> None:
    """Append a checkpoint entry to pipeline_log.json."""
    out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pipeline_log.json"

    log_data = {}
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except json.JSONDecodeError:
            log_data = {}

    if "checkpoints" not in log_data:
        log_data["checkpoints"] = []

    log_data["checkpoints"].append({
        "stage": stage,
        "status": status,
        "details": details or {},
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    })

    log_path.write_text(json.dumps(log_data, indent=2, default=str))
    logger.info("CHECKPOINT [%s]: %s", stage, status)


# ── Core pipeline: single test ──────────────────────────────────────────

def run_single_test(
    df: pd.DataFrame,
    instrument: str,
    timeframe: str,
    variant: str,
    session_normalised: bool,
    filters: list[str],
    K: int,
    cc_features: pd.DataFrame | None = None,
    position_sizes: pd.Series | None = None,
) -> dict:
    """
    Run one complete test: detect → filter → outcomes → stats.
    Returns a dict suitable for the summary table.
    """
    sig_gen = SignalGenerator()
    sig_filt = SignalFilter()
    oc = OutcomeCalculator()
    st = StatisticalTests()

    # ── Detect ───────────────────────────────────────────────────────
    signals = sig_gen.detect(df, variant, session_normalised=session_normalised)

    # ── Direction always applied first ───────────────────────────────
    all_filters = ["direction"] + [f for f in filters if f != "direction"]
    filtered, directions = sig_filt.apply(df, signals, all_filters, cc_features=cc_features)

    n_signals = int(filtered.sum())
    n_min_chart = CONFIG.get("n_min_signals_chart", 100)

    if n_signals < 5:
        logger.info(
            "Skipping %s %s variant=%s filters=%s K=%d — only %d signals.",
            instrument, timeframe, variant, filters, K, n_signals,
        )
        return _empty_result(instrument, timeframe, variant, session_normalised,
                             all_filters, K, n_signals)

    # ── Outcomes ─────────────────────────────────────────────────────
    outcomes = oc.compute_outcomes(df, filtered, directions, K_values=[K], position_sizes=position_sizes)
    outcomes = oc.apply_costs(outcomes, instrument, K=K)

    win_col = f"win_{K}"
    htc_col = f"htc_return_{K}"
    htc_net_col = f"htc_return_net_{K}"

    wins = outcomes[win_col].dropna()
    n_valid = len(wins)
    n_wins = int((wins == 1.0).sum())
    win_rate = n_wins / n_valid if n_valid > 0 else np.nan

    # ── Session-matched base rate ────────────────────────────────────
    sess_hours = _session_hours_from_filters(all_filters)
    base_rate = st.compute_base_rate(df, K, session_hours=sess_hours)

    # ── Binomial test (SECONDARY) ────────────────────────────────────
    p_raw = st.binomial_test(n_wins, n_valid, base_rate)

    # ── Bootstrap CIs (block) ────────────────────────────────────────
    try:
        wr_point, wr_lo, wr_hi = st.bootstrap_ci(
            (wins == 1.0).astype(float).values, statistic=np.mean,
        )
    except Exception:
        wr_point, wr_lo, wr_hi = win_rate, np.nan, np.nan

    gross_returns = outcomes[htc_col].dropna().values
    net_returns = outcomes[htc_net_col].dropna().values if htc_net_col in outcomes.columns else gross_returns

    def _sharpe(arr):
        return arr.mean() / arr.std() * np.sqrt(252) if len(arr) > 1 and arr.std() > 0 else 0.0

    sharpe_gross = _sharpe(gross_returns)
    sharpe_net = _sharpe(net_returns)

    try:
        _, sr_lo, sr_hi = st.bootstrap_ci(net_returns, statistic=_sharpe)
    except Exception:
        sr_lo, sr_hi = np.nan, np.nan

    # ── Cohen's h ────────────────────────────────────────────────────
    try:
        h_val = 2 * (np.arcsin(np.sqrt(win_rate)) - np.arcsin(np.sqrt(base_rate)))
    except Exception:
        h_val = np.nan

    # ── MFE / MAE ────────────────────────────────────────────────────
    mfe_col = f"mfe_{K}"
    mae_col = f"mae_{K}"
    avg_mfe = outcomes[mfe_col].mean() if mfe_col in outcomes.columns else np.nan
    avg_mae = outcomes[mae_col].mean() if mae_col in outcomes.columns else np.nan
    ratio = avg_mfe / avg_mae if avg_mae and avg_mae > 0 else np.nan

    # ── Sample warning ───────────────────────────────────────────────
    n_min = CONFIG.get("n_min_signals", 150)
    if n_valid < n_min_chart:
        sample_warning = "INSUFFICIENT"
    elif n_valid < n_min:
        sample_warning = "UNDERPOWERED"
    else:
        sample_warning = ""

    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "variant": f"{variant}{'_sn' if session_normalised else ''}",
        "filters_applied": "+".join(all_filters),
        "K": K,
        "n_signals": n_valid,
        "win_rate": win_rate,
        "base_rate": base_rate,
        "win_rate_excess": win_rate - base_rate if not np.isnan(win_rate) else np.nan,
        "p_value_raw": p_raw,
        "p_value_corrected": np.nan,  # filled in later across full matrix
        "cohens_h": h_val,
        "sample_warning": sample_warning,
        "avg_return_gross": float(np.nanmean(gross_returns)),
        "avg_return_net": float(np.nanmean(net_returns)),
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


def _empty_result(instrument, timeframe, variant, session_normalised,
                  all_filters, K, n_signals):
    """Return a result dict for skipped tests (insufficient signals)."""
    return {
        "instrument": instrument, "timeframe": timeframe,
        "variant": f"{variant}{'_sn' if session_normalised else ''}",
        "filters_applied": "+".join(all_filters),
        "K": K, "n_signals": n_signals,
        "win_rate": np.nan, "base_rate": np.nan,
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


# ── Core execution logic ────────────────────────────────────────────────

def _execute_run(
    instruments: list[str],
    timeframes: list[str],
    use_synthetic: bool,
    skip_expensive: bool,
    pre_commitment_hash: str | None = None,
):
    """
    Execute the full staged pipeline for one configuration.
    
    Stages 0-3 (preflight) are run ONCE.
    Stages 4-9 (backtest) run the test matrix.
    """
    reporter = PerformanceReporter()
    st = StatisticalTests()
    loader = DataLoader()
    checkpoint_warnings: list[str] = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1: COST VIABILITY PRE-FLIGHT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 80)
    logger.info("STAGE 1: COST VIABILITY PRE-FLIGHT")
    logger.info("=" * 80)

    try:
        viability_df = compute_cost_viability(instruments, timeframes, loader)
        non_viable = viability_df[~viability_df["viable"]]

        if len(non_viable) > 0 and not CONFIG.get("force_run_insufficient_data", False):
            for _, row in non_viable.iterrows():
                warning = (
                    f"COST NON-VIABLE: {row['instrument']} {row['timeframe']} "
                    f"(BE WR = {row.get('break_even_win_rate', 'N/A')})"
                )
                checkpoint_warnings.append(warning)
                logger.warning(warning)

            # Filter to viable only
            viable_set = set(
                (row["instrument"], row["timeframe"])
                for _, row in viability_df[viability_df["viable"]].iterrows()
            )
            if not viable_set:
                logger.error("NO viable instrument×timeframe combinations. Halting.")
                _write_checkpoint("STAGE_1_COST_VIABILITY", "HALT_ALL_NON_VIABLE")
                reporter.write_insufficient_result(
                    "ALL", "ALL", 0,
                    reason="No instrument×timeframe combinations are cost-viable."
                )
                return

            _write_checkpoint("STAGE_1_COST_VIABILITY", "PASS_PARTIAL",
                              {"viable": len(viable_set), "non_viable": len(non_viable)})
        else:
            viable_set = set(product(instruments, timeframes))
            _write_checkpoint("STAGE_1_COST_VIABILITY", "PASS")
    except Exception as e:
        logger.warning("Cost viability check failed: %s — proceeding with all combos.", e)
        viable_set = set(product(instruments, timeframes))
        _write_checkpoint("STAGE_1_COST_VIABILITY", "SKIPPED", {"error": str(e)})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGES 2-8: PER INSTRUMENT×TIMEFRAME TEST MATRIX
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    all_results: list[dict] = []
    best_sharpe = -np.inf
    best_key = ""

    total_tests = (
        len(instruments) * len(timeframes)
        * len(VARIANTS) * len(SESSION_NORM_OPTIONS)
        * len(FILTER_LADDERS) * len(K_TO_TEST)
    )
    logger.info("=" * 80)
    logger.info("FULL TEST MATRIX: %d tests pre-committed", total_tests)
    logger.info("=" * 80)

    test_count = 0
    t0 = time.time()

    for instrument in instruments:
        for timeframe in timeframes:
            if (instrument, timeframe) not in viable_set:
                logger.info("Skipping non-viable %s %s", instrument, timeframe)
                continue

            # ── STAGE 2: DATA LOADING + DENSITY ──────────────────────
            logger.info("─── STAGE 2: Loading %s %s ───", instrument, timeframe)

            if use_synthetic:
                df_full = generate_synthetic_ohlcv(
                    n_candles=150_000,
                    timeframe_minutes=TF_MINUTES.get(timeframe, 60),
                )
            else:
                try:
                    df_full = loader.fetch_max_history(instrument, timeframe)
                    if len(df_full) < 1000:
                        raise ValueError(f"Insufficient data: only {len(df_full)} candles.")
                    loader.validate_data(df_full)
                except Exception as e:
                    logger.warning(
                        "Failed to load data for %s %s: %s — falling back to synthetic.",
                        instrument, timeframe, e,
                    )
                    df_full = generate_synthetic_ohlcv(
                        n_candles=150_000,
                        timeframe_minutes=TF_MINUTES.get(timeframe, 60),
                    )

            if len(df_full) == 0:
                logger.warning("No data for %s %s — skipping.", instrument, timeframe)
                continue

            # Density validation
            density_results = loader.validate_density(df_full, timeframe)
            loader.write_data_summary(df_full, instrument, timeframe, density_results)
            _write_checkpoint(f"STAGE_2_DATA_{instrument}_{timeframe}", "PASS",
                              {"candles": len(df_full)})

            # Split in-sample / OOS
            df_is, df_oos = loader.split_data(df_full)

            # ── STAGE 3: SAMPLE SIZE ESTIMATION ──────────────────────
            logger.info("─── STAGE 3: Sample size estimation %s %s ───", instrument, timeframe)

            sig_gen = SignalGenerator()
            sig_filt = SignalFilter()
            pilot_result = estimate_sample_size(
                df_full, instrument, timeframe,
                sig_gen, sig_filt,
                VARIANTS, SESSION_NORM_OPTIONS, FILTER_LADDERS,
            )

            if pilot_result["status"] == "INSUFFICIENT":
                if not CONFIG.get("force_run_insufficient_data", False):
                    warning = (
                        f"INSUFFICIENT SIGNALS: {instrument} {timeframe} — "
                        f"estimated {pilot_result['estimated_total_signals']} signals "
                        f"(need {pilot_result['n_min_required']})"
                    )
                    checkpoint_warnings.append(warning)
                    logger.warning(warning)
                    _write_checkpoint(
                        f"STAGE_3_SAMPLE_{instrument}_{timeframe}", "WARNING",
                        pilot_result,
                    )
                    # Continue anyway but flag all results
                else:
                    logger.info("force_run_insufficient_data=True — continuing despite insufficient estimate.")
            else:
                _write_checkpoint(
                    f"STAGE_3_SAMPLE_{instrument}_{timeframe}", "PASS",
                    {"estimated_signals": pilot_result["estimated_total_signals"]},
                )

            # ── Cross Currency Processing ────────────────────────────
            cc_features = None
            if CONFIG.get("cc_enabled", False):
                try:
                    secondary_df = load_secondary(CONFIG, df_is, timeframe)
                    cc_features = compute_cross_currency_features(
                        df_is, secondary_df, df_is.index, CONFIG,
                    )
                except Exception as e:
                    logger.warning("CC feature computation failed: %s", e)

            # Discard signals in the blackout buffer zone
            buffer = CONFIG["blackout_buffer_candles"]
            if len(df_is) > buffer:
                df_is = df_is.iloc[:-buffer]
                if cc_features is not None:
                    cc_features = cc_features.loc[cc_features.index <= df_is.index[-1]]

            if len(df_is) < 500:
                logger.warning(
                    "In-sample too short for %s %s (%d candles) — skipping.",
                    instrument, timeframe, len(df_is),
                )
                continue

            # ── STAGES 4-6: SIGNAL → FILTER → OUTCOME → STATS ───────
            for variant in VARIANTS:
                for session_norm in SESSION_NORM_OPTIONS:
                    if variant == "D" and session_norm:
                        continue

                    for filters in FILTER_LADDERS:
                        for K in K_TO_TEST:
                            test_count += 1
                            if test_count % 50 == 0:
                                elapsed = time.time() - t0
                                logger.info(
                                    "Progress: %d / %d tests  (%.0fs elapsed)",
                                    test_count, total_tests, elapsed,
                                )

                            # Position sizing
                            position_sizes = None
                            if CONFIG.get("cc_enabled", False) and cc_features is not None:
                                position_sizes = compute_position_sizes(
                                    cc_features, CONFIG, base_size=1.0,
                                )

                            result = run_single_test(
                                df_is, instrument, timeframe,
                                variant, session_norm, filters, K,
                                cc_features=cc_features,
                                position_sizes=position_sizes,
                            )
                            all_results.append(result)

                            # Track best for deep analysis
                            sn = result.get("sharpe_net", np.nan)
                            if not np.isnan(sn) and sn > best_sharpe:
                                best_sharpe = sn
                                best_key = (
                                    f"{instrument}_{timeframe}_{variant}_"
                                    f"{'+'.join(filters)}_K{K}"
                                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 7: FDR CORRECTION + RED FLAG ASSESSMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 80)
    logger.info("STAGE 7: FDR correction + Red Flag assessment across %d tests.", len(all_results))
    logger.info("=" * 80)

    # FDR correction
    raw_pvals = np.array([r["p_value_raw"] for r in all_results])
    corrected = st.benjamini_hochberg(raw_pvals)
    for i, r in enumerate(all_results):
        r["p_value_corrected"] = corrected[i]

    # Red flag assessment
    red_flag_reports = check_all_results(all_results)

    _write_checkpoint("STAGE_7_RED_FLAGS", "COMPLETE", {
        "total_results": len(all_results),
        "results_with_critical_flags": sum(1 for r in red_flag_reports if r.n_critical > 0),
        "results_reportable": sum(1 for r in red_flag_reports if r.is_reportable),
    })

    # ── Build summary table ──────────────────────────────────────────
    summary_df = reporter.summary_table(all_results)
    summary_path = Path(CONFIG["output_dir"]) / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary table → %s  (%d rows)", summary_path, len(summary_df))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 8: DEEP ANALYSIS (PERMUTATION + WALK-FORWARD)
    # Only for the best variant, if it has enough signals
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    perm_result = None
    wf_results = None
    stationarity_result = None
    cc_validity = None

    if not summary_df.empty and best_sharpe > -np.inf:
        logger.info("=" * 80)
        logger.info("STAGE 8: Deep analysis on best variant: %s", best_key)
        logger.info("=" * 80)

        best_row = summary_df.loc[summary_df["sharpe_net"].idxmax()] if not summary_df["sharpe_net"].isna().all() else summary_df.iloc[0]
        inst = best_row["instrument"]
        tf = best_row["timeframe"]
        var_raw = best_row["variant"]
        var = var_raw.replace("_sn", "")
        sn = "_sn" in var_raw
        filt_str = best_row["filters_applied"]
        filt_list = filt_str.split("+") if isinstance(filt_str, str) else []
        K_best = int(best_row["K"])
        n_best = int(best_row["n_signals"])

        # Check if deep analysis is warranted
        n_min_chart = CONFIG.get("n_min_signals_chart", 100)

        if n_best < n_min_chart:
            logger.warning(
                "Best result has only %d signals (min %d for deep analysis). "
                "Writing RESULT_INVALID file instead.",
                n_best, n_min_chart,
            )
            reporter.write_insufficient_result(inst, tf, n_best)
            _write_checkpoint("STAGE_8_DEEP_ANALYSIS", "SKIPPED_INSUFFICIENT_N",
                              {"n_signals": n_best, "min_required": n_min_chart})
        else:
            # Reload data
            if use_synthetic:
                df_deep = generate_synthetic_ohlcv(
                    n_candles=150_000, timeframe_minutes=TF_MINUTES.get(tf, 60),
                )
            else:
                df_deep = loader.fetch_max_history(inst, tf)

            df_is_deep, _ = loader.split_data(df_deep)
            buf = CONFIG["blackout_buffer_candles"]
            if len(df_is_deep) > buf:
                df_is_deep = df_is_deep.iloc[:-buf]

            sig_gen = SignalGenerator()
            sig_filt = SignalFilter()
            oc = OutcomeCalculator()

            def _signal_func(d):
                return sig_gen.detect(d, var, session_normalised=sn)

            def _filter_func(d, s):
                return sig_filt.apply(d, s, filt_list)

            def _outcome_func(d, s, dirs):
                return oc.compute_outcomes(d, s, dirs, K_values=[K_best])

            # Permutation test (PRIMARY gate)
            if not skip_expensive:
                logger.info("Running permutation test (%d perms)...", CONFIG["permutation_n"])
                perm_result = st.permutation_test(
                    df_is_deep, _signal_func, _filter_func, _outcome_func, K=K_best,
                )
            else:
                logger.info("Skipping permutation test (skip_expensive=True).")

            # Walk-forward stability
            if not skip_expensive:
                logger.info("Running walk-forward analysis...")
                wf_results = st.walk_forward(
                    df_is_deep, _signal_func, _filter_func, _outcome_func, K=K_best,
                )

            # Stationarity check
            signals_all = _signal_func(df_is_deep)
            filt_all, dir_all = _filter_func(df_is_deep, signals_all)
            outcomes_all = _outcome_func(df_is_deep, filt_all, dir_all)
            stationarity_result = st.stationarity_check(outcomes_all, K=K_best)

            # CC Validity
            if CONFIG.get("cc_enabled", False):
                try:
                    secondary_df = load_secondary(CONFIG, df_is_deep, tf)
                    cc_feats = compute_cross_currency_features(
                        df_is_deep, secondary_df, df_is_deep.index, CONFIG,
                    )
                    cc_validity = st.cross_currency_validity_test(
                        primary_df=df_is_deep,
                        secondary_df=secondary_df,
                        signals=signals_all,
                        directions=dir_all,
                        cc_features=cc_feats,
                        K=K_best,
                    )
                except Exception as e:
                    logger.warning("CC validity test failed: %s", e)

            # ── Plots (n-gated by reporter) ──────────────────────────
            htc_net = f"htc_return_net_{K_best}"
            htc_gross = f"htc_return_{K_best}"

            if htc_net in outcomes_all.columns:
                eq_net = outcomes_all[htc_net].dropna().cumsum().reset_index(drop=True)
                eq_gross = outcomes_all[htc_gross].dropna().cumsum().reset_index(drop=True)
                reporter.plot_equity_curves(
                    {"Gross": eq_gross, "Net of costs": eq_net},
                    title=f"Equity Curve: {inst} {tf} {var_raw} K={K_best}",
                )

            # Return distribution
            from outcome_calculator import OutcomeCalculator as OC2
            atr_all = OC2.compute_atr(df_is_deep)
            fwd_close = df_is_deep["close"].shift(-K_best)
            all_returns = ((fwd_close - df_is_deep["open"].shift(-1)) / atr_all).dropna().values
            sig_returns = outcomes_all[htc_gross].dropna().values if htc_gross in outcomes_all.columns else np.array([])

            if len(sig_returns) > 0 and len(all_returns) > 0:
                reporter.plot_distribution(sig_returns, all_returns, K_best)

            if wf_results:
                reporter.plot_walk_forward_sharpe(wf_results)
            if perm_result:
                reporter.plot_permutation_test(perm_result)

            reporter.plot_mfe_mae(outcomes_all, K_best)
            reporter.plot_session_heatmap(outcomes_all, K_best)

            _write_checkpoint("STAGE_8_DEEP_ANALYSIS", "COMPLETE")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 9: REPORTING + PRE-COMMITMENT VERIFICATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("=" * 80)
    logger.info("STAGE 9: Final reporting + pre-commitment verification")
    logger.info("=" * 80)

    # Pre-commitment verification
    if pre_commitment_hash:
        is_clean = verify_pre_commitment(
            instruments, timeframes, VARIANTS, SESSION_NORM_OPTIONS,
            FILTER_LADDERS, K_TO_TEST, pre_commitment_hash,
        )
        if not is_clean:
            checkpoint_warnings.append(
                "PRE-COMMITMENT CONTAMINATED: test matrix was modified mid-run!"
            )

    # Conclusion (with red flags and checkpoint warnings)
    reporter.write_conclusion(
        summary_df,
        perm_result=perm_result,
        wf_results=wf_results,
        stationarity=stationarity_result,
        cc_validity=cc_validity,
        red_flag_reports=red_flag_reports,
        checkpoint_warnings=checkpoint_warnings,
    )

    # CC Summary
    if CONFIG.get("cc_enabled", False) and cc_validity is not None:
        reporter.write_cc_summary(
            classification_summary=pd.DataFrame([{"Placeholder": "NYI"}]),
            regime_summary=pd.DataFrame([{"Placeholder": "NYI"}]),
            catch_up_stats=cc_validity,
        )

    _write_checkpoint("STAGE_9_REPORTING", "COMPLETE")

    # ── Print top results ────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE  (%.0fs total)", time.time() - t0)
    logger.info("=" * 80)

    top = summary_df.nsmallest(10, "p_value_corrected")
    logger.info("Top 10 by corrected p-value:\n%s", top.to_string(index=False))

    sig_level = CONFIG["significance_level"]
    sig_count = (summary_df["p_value_corrected"] < sig_level).sum()
    logger.info(
        "\n%d / %d tests significant after FDR correction (α=%.2f).",
        sig_count, len(summary_df), sig_level,
    )

    # Red flag summary
    n_reportable = sum(1 for r in red_flag_reports if r.is_reportable)
    n_with_flags = sum(1 for r in red_flag_reports if r.n_flags > 0)
    logger.info(
        "Red flags: %d/%d results have flags, %d are reportable.",
        n_with_flags, len(red_flag_reports), n_reportable,
    )


# ── Full analysis entry point ───────────────────────────────────────────

def run_full_analysis(
    instruments: list[str] | None = None,
    timeframes: list[str] | None = None,
    use_synthetic: bool = False,
    skip_expensive: bool = False,
) -> None:
    """
    Execute the complete pipeline end-to-end.

    V2: Staged architecture with preflight checks, n-gates, and red flags.
    """
    np.random.seed(CONFIG["random_seed"])

    instruments = instruments or list(CONFIG["instruments"].keys())
    timeframes = timeframes or CONFIG["timeframes"]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 0: PRE-COMMITMENT LOCK
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    runs_to_execute = []
    base_out = Path(CONFIG["output_dir"])

    # 1. Baseline Run
    runs_to_execute.append({
        "name": "baseline",
        "out_dir": base_out / "baseline",
        "cc_enabled": False,
        "cc_params": {}
    })

    # 2. Cross-Currency Extension Grids
    if CONFIG.get("cc_enabled", True):
        import hashlib
        sweep_grid = CONFIG.get("cc_param_sweep", {})
        keys = list(sweep_grid.keys())
        values = list(sweep_grid.values())

        combinations = [dict(zip(keys, v)) for v in product(*values)]

        grid_str = json.dumps(sweep_grid, sort_keys=True)
        grid_hash = hashlib.sha256(grid_str.encode()).hexdigest()[:8]

        if skip_expensive:
            logger.info("skip_expensive=True: Truncating for fast validation.")
            combinations = combinations[:1]
            global FILTER_LADDERS, K_TO_TEST, VARIANTS, SESSION_NORM_OPTIONS
            FILTER_LADDERS = [["body_ratio", "wick", "session_both", "news"]]
            K_TO_TEST = [10]
            VARIANTS = ["A"]
            SESSION_NORM_OPTIONS = [True]

        for i, params in enumerate(combinations):
            runs_to_execute.append({
                "name": f"cc_ext_{grid_hash}_{i}",
                "out_dir": base_out / "cc_extension" / f"run_{i}",
                "cc_enabled": True,
                "cc_params": params
            })

    for run in runs_to_execute:
        # Override config per run
        CONFIG["cc_enabled"] = run["cc_enabled"]
        for k, v in run["cc_params"].items():
            CONFIG[k] = v

        CONFIG["output_dir"] = str(run["out_dir"])
        _setup_logging(log_dir=str(run["out_dir"]))

        Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("STARTING RUN: %s", run["name"])
        if run["cc_params"]:
            logger.info("CC PARAMS: %s", run["cc_params"])
        logger.info("=" * 80)

        # Pre-commitment lock
        pre_commitment_hash = lock_pre_commitment(
            instruments, timeframes, VARIANTS, SESSION_NORM_OPTIONS,
            FILTER_LADDERS, K_TO_TEST,
        )
        _write_checkpoint("STAGE_0_PRE_COMMITMENT", "LOCKED",
                          {"hash": pre_commitment_hash[:16]})

        _execute_run(
            instruments, timeframes, use_synthetic, skip_expensive,
            pre_commitment_hash=pre_commitment_hash,
        )


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Volume Spike Directional Prediction Pipeline (V2)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of OANDA API.",
    )
    parser.add_argument(
        "--instrument", type=str, nargs="+", default=None,
        help="Instrument(s) to test, e.g. EUR_USD GBP_USD",
    )
    parser.add_argument(
        "--timeframe", type=str, nargs="+", default=None,
        help="Timeframe(s) to test, e.g. M5 M15 H1",
    )
    parser.add_argument(
        "--skip-expensive", action="store_true",
        help="Skip permutation test and walk-forward (for debugging).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    run_full_analysis(
        instruments=args.instrument,
        timeframes=args.timeframe,
        use_synthetic=args.synthetic,
        skip_expensive=args.skip_expensive,
    )


if __name__ == "__main__":
    main()
