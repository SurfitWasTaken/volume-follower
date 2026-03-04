from __future__ import annotations
"""
Volume Spike Directional Prediction — Main Pipeline

Entry point: run_full_analysis()

Executes the FULL pre-committed test matrix across all instruments,
timeframes, signal variants, and filter combinations.  No conditional
expansion — the entire matrix is run and FDR correction is applied
across all tests simultaneously.

Architecture:
    DataLoader → SignalGenerator → SignalFilter → OutcomeCalculator
    → StatisticalTests → PerformanceReporter
"""

import argparse
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
from statistical_tests import StatisticalTests
from performance_reporter import PerformanceReporter

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

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}


def _session_hours_from_filters(filters: list[str]) -> set[int] | None:
    """Extract session hours from filter list for base-rate matching."""
    for f in filters:
        if f.startswith("session"):
            windows = CONFIG["session_windows"]
            if "london" in f:
                return set(range(8, 10))
            elif "newyork" in f:
                return set(range(13, 16))
            else:  # both
                return set(range(8, 10)) | set(range(13, 16))
    return None  # no session filter → global base rate


# ── Core pipeline functions ─────────────────────────────────────────────

def load_or_generate_data(
    instrument: str,
    timeframe: str,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """Load real OANDA data or fall back to synthetic."""
    loader = DataLoader()

    if use_synthetic:
        logger.info("Using synthetic data for %s %s.", instrument, timeframe)
        return generate_synthetic_ohlcv(
            n_candles=150_000,
            timeframe_minutes=TF_MINUTES.get(timeframe, 5),
        )

    try:
        df = loader.fetch_from_oanda(
            instrument=instrument,
            granularity=timeframe,
            start=CONFIG["data_start"],
            end=CONFIG["data_end"],
        )
        if len(df) < 1000:
            raise ValueError(f"Insufficient data: only {len(df)} candles.")
        loader.validate_data(df)
        return df
    except Exception as e:
        logger.warning(
            "Failed to load real data for %s %s: %s — falling back to synthetic.",
            instrument, timeframe, e,
        )
        return generate_synthetic_ohlcv(
            n_candles=150_000,
            timeframe_minutes=TF_MINUTES.get(timeframe, 5),
        )


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
    if n_signals < 5:
        logger.info(
            "Skipping %s %s variant=%s filters=%s K=%d — only %d signals.",
            instrument, timeframe, variant, filters, K, n_signals,
        )
        return {
            "instrument": instrument, "timeframe": timeframe,
            "variant": f"{variant}{'_sn' if session_normalised else ''}",
            "filters_applied": "+".join(all_filters),
            "K": K, "n_signals": n_signals,
            "win_rate": np.nan, "base_rate": np.nan,
            "win_rate_excess": np.nan,
            "p_value_raw": 1.0, "p_value_corrected": np.nan,
            "cohens_h": np.nan, "sample_warning": "INSUFFICIENT" if n_signals < 150 else "",
            "avg_return_gross": np.nan, "avg_return_net": np.nan,
            "sharpe_gross": np.nan, "sharpe_net": np.nan,
            "avg_mfe_atr": np.nan, "avg_mae_atr": np.nan,
            "mfe_mae_ratio": np.nan,
            "win_rate_ci_lo": np.nan, "win_rate_ci_hi": np.nan,
            "sharpe_ci_lo": np.nan, "sharpe_ci_hi": np.nan,
        }

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
    base_rate = st.compute_base_rate(
        df, K, session_hours=sess_hours,
    )

    # ── Binomial test ────────────────────────────────────────────────
    p_raw = st.binomial_test(n_wins, n_valid, base_rate)

    # ── Bootstrap CIs (block) ────────────────────────────────────────
    wr_point, wr_lo, wr_hi = st.bootstrap_ci(
        (wins == 1.0).astype(float).values, statistic=np.mean,
    )

    gross_returns = outcomes[htc_col].dropna().values
    net_returns = outcomes[htc_net_col].dropna().values if htc_net_col in outcomes.columns else gross_returns

    def _sharpe(arr):
        return arr.mean() / arr.std() * np.sqrt(252) if len(arr) > 1 and arr.std() > 0 else 0.0

    sharpe_gross = _sharpe(gross_returns)
    sharpe_net = _sharpe(net_returns)

    _, sr_lo, sr_hi = st.bootstrap_ci(net_returns, statistic=_sharpe)

    # ── MFE / MAE ────────────────────────────────────────────────────
    mfe_col = f"mfe_{K}"
    mae_col = f"mae_{K}"
    avg_mfe = outcomes[mfe_col].mean() if mfe_col in outcomes.columns else np.nan
    avg_mae = outcomes[mae_col].mean() if mae_col in outcomes.columns else np.nan
    ratio = avg_mfe / avg_mae if avg_mae and avg_mae > 0 else np.nan

    # --- Handle Signal Reduction from Adaptive Sizing ---
    # n_signals is primarily signals after BOOLEAN filtering.
    # But CC logic can set p_size to 0.0, which acts as a late-stage filter.
    n_effective = n_valid
    
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "variant": f"{variant}{'_sn' if session_normalised else ''}",
        "filters_applied": "+".join(all_filters),
        "K": K,
        "n_signals": n_effective,
        "win_rate": win_rate,
        "base_rate": base_rate,
        "win_rate_excess": win_rate - base_rate if not np.isnan(win_rate) else np.nan,
        "p_value_raw": p_raw,
        "p_value_corrected": np.nan,  # filled in later across full matrix
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


# ── Test matrix definition ──────────────────────────────────────────────

# Pre-committed: EVERY combination is run.  No conditional expansion.
VARIANTS = ["A", "B", "C", "D"]
SESSION_NORM_OPTIONS = [False, True]  # True ignored for D (built-in)

# Cumulative filter ladders to test (each is additive)
FILTER_LADDERS = [
    [],                                              # volume only (directionless handled by direction=always)
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


# ── Full analysis ───────────────────────────────────────────────────────

def run_full_analysis(
    instruments: list[str] | None = None,
    timeframes: list[str] | None = None,
    use_synthetic: bool = False,
    skip_expensive: bool = False,
) -> None:
    """
    Execute the complete pipeline end-to-end.

    Parameters
    ----------
    instruments : list[str] or None
        Override CONFIG instruments.  Pass e.g. ['EUR_USD'] for a fast run.
    timeframes : list[str] or None
        Override CONFIG timeframes.
    use_synthetic : bool
        If True, use synthetic data instead of OANDA.
    skip_expensive : bool
        If True, skip permutation test and walk-forward (for quick debugging).
    """
    np.random.seed(CONFIG["random_seed"])

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    _setup_logging()

    instruments = instruments or list(CONFIG["instruments"].keys())
    timeframes = timeframes or CONFIG["timeframes"]

# ── Core execution logic ────────────────────────────────────────────────

def _execute_run(
    instruments: list[str],
    timeframes: list[str],
    use_synthetic: bool,
    skip_expensive: bool,
):
    reporter = PerformanceReporter()
    st = StatisticalTests()

    all_results: list[dict] = []
    best_sharpe = -np.inf
    best_key = ""
    cc_validity = None

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
            logger.info("─── Loading %s %s ───", instrument, timeframe)
            df_full = load_or_generate_data(instrument, timeframe, use_synthetic)

            if len(df_full) == 0:
                logger.warning("No data for %s %s — skipping.", instrument, timeframe)
                continue

            # Split in-sample / OOS
            loader = DataLoader()
            df_is, df_oos = loader.split_data(df_full)

            # --- Cross Currency Processing ---
            cc_features = None
            if CONFIG.get("cc_enabled", False):
                secondary_df = load_secondary(CONFIG, df_is, timeframe)
                # Compute features for all potential signals blindly (to allow testing)
                dummy_signals = pd.Series(True, index=df_is.index)
                cc_features = compute_cross_currency_features(df_is, secondary_df, df_is.index, CONFIG)

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

            for variant in VARIANTS:
                for session_norm in SESSION_NORM_OPTIONS:
                    # Variant D has built-in session normalisation
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

                            # 1. Feature injection
                            filtered_signals = None
                            
                            # 2. Position sizing
                            position_sizes = None
                            if CONFIG.get("cc_enabled", False) and cc_features is not None:
                                position_sizes = compute_position_sizes(cc_features, CONFIG, base_size=1.0)

                            result = run_single_test(
                                df_is, instrument, timeframe,
                                variant, session_norm, filters, K,
                                cc_features=cc_features,
                                position_sizes=position_sizes,
                            )
                            all_results.append(result)

                            # Track best for deep analysis
                            if result["sharpe_net"] is not np.nan and result["sharpe_net"] > best_sharpe:
                                best_sharpe = result["sharpe_net"]
                                best_key = f"{instrument}_{timeframe}_{variant}_{'+'.join(filters)}_K{K}"

    # ── FDR correction across the ENTIRE matrix ──────────────────────
    logger.info("Applying Benjamini-Hochberg FDR across %d tests.", len(all_results))
    raw_pvals = np.array([r["p_value_raw"] for r in all_results])
    corrected = st.benjamini_hochberg(raw_pvals)
    for i, r in enumerate(all_results):
        r["p_value_corrected"] = corrected[i]

    # ── Build summary table ──────────────────────────────────────────
    summary_df = reporter.summary_table(all_results)
    summary_path = Path(CONFIG["output_dir"]) / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary table → %s  (%d rows)", summary_path, len(summary_df))

    # ── Deep analysis on best variant ────────────────────────────────
    sig_level = CONFIG["significance_level"]
    significant = summary_df[summary_df["p_value_corrected"] < sig_level]

    perm_result = None
    wf_results = None
    stationarity_result = None
    cc_validity = None

    if not summary_df.empty:
        logger.info("=" * 80)
        logger.info("Deep analysis on best variant: %s", best_key)
        logger.info("=" * 80)

        # Identify the best row's parameters
        best_row = summary_df.loc[summary_df["sharpe_net"].idxmax()] if not summary_df["sharpe_net"].isna().all() else summary_df.iloc[0]
        inst = best_row["instrument"]
        tf = best_row["timeframe"]
        var_raw = best_row["variant"]
        var = var_raw.replace("_sn", "")
        sn = "_sn" in var_raw
        filt_str = best_row["filters_applied"]
        filt_list = filt_str.split("+") if isinstance(filt_str, str) else []
        K_best = int(best_row["K"])

        # Reload data
        df_deep = load_or_generate_data(inst, tf, use_synthetic)
        df_is_deep, _ = DataLoader().split_data(df_deep)
        buf = CONFIG["blackout_buffer_candles"]
        if len(df_is_deep) > buf:
            df_is_deep = df_is_deep.iloc[:-buf]

        # Create closures for walk-forward / permutation
        sig_gen = SignalGenerator()
        sig_filt = SignalFilter()
        oc = OutcomeCalculator()

        def _signal_func(d):
            return sig_gen.detect(d, var, session_normalised=sn)

        def _filter_func(d, s):
            return sig_filt.apply(d, s, filt_list)

        def _outcome_func(d, s, dirs):
            return oc.compute_outcomes(d, s, dirs, K_values=[K_best])

        # Permutation test
        logger.info("Running permutation test (%d perms)...", CONFIG["permutation_n"])
        perm_result = st.permutation_test(
            df_is_deep, _signal_func, _filter_func, _outcome_func, K=K_best,
        )

        # Walk-forward stability
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
        cc_validity = None
        if CONFIG.get("cc_enabled", False) and cc_features is not None:
            secondary_df = load_secondary(CONFIG, df_is_deep, tf)
            cc_validity = st.cross_currency_validity_test(
                primary_df=df_is_deep,
                secondary_df=secondary_df,
                signals=signals_all,
                directions=dir_all,
                cc_features=cc_features,
                K=K_best
            )

        # ── Plots ────────────────────────────────────────────────────
        # Equity curve
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
        # Compute all-candle forward returns for comparison
        from outcome_calculator import OutcomeCalculator as OC2
        atr_all = OC2.compute_atr(df_is_deep)
        fwd_close = df_is_deep["close"].shift(-K_best)
        all_returns = ((fwd_close - df_is_deep["open"].shift(-1)) / atr_all).dropna().values
        sig_returns = outcomes_all[htc_gross].dropna().values if htc_gross in outcomes_all.columns else np.array([])

        if len(sig_returns) > 0 and len(all_returns) > 0:
            reporter.plot_distribution(sig_returns, all_returns, K_best)

        # Walk-forward
        if wf_results:
            reporter.plot_walk_forward_sharpe(wf_results)

        # Permutation
        if perm_result:
            reporter.plot_permutation_test(perm_result)

        # MFE/MAE
        reporter.plot_mfe_mae(outcomes_all, K_best)

        # Session heatmap
        reporter.plot_session_heatmap(outcomes_all, K_best)

    elif len(significant) == 0:
        logger.info("=" * 80)
        logger.info("NO statistically significant results after FDR correction.")
        logger.info("Skipping deep analysis.  Generating conclusion.")
        logger.info("=" * 80)

    # ── Conclusion ───────────────────────────────────────────────────
    reporter.write_conclusion(
        summary_df,
        perm_result=perm_result,
        wf_results=wf_results,
        stationarity=stationarity_result,
        cc_validity=cc_validity,
    )

    # ── CC Specific Summary ──────────────────────────────────────────
    if CONFIG.get("cc_enabled", False) and cc_validity is not None:
        # Generate classification breakdown for reporting
        if best_sharpe != -np.inf:
             # Logic to generate classification summary if needed
             pass
        reporter.write_cc_summary(
             classification_summary=pd.DataFrame([{"Placeholder": "NYI"}]),
             regime_summary=pd.DataFrame([{"Placeholder": "NYI"}]),
             catch_up_stats=cc_validity
        )


    # ── Print top results ────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE  (%.0fs total)", time.time() - t0)
    logger.info("=" * 80)

    top = summary_df.nsmallest(10, "p_value_corrected")
    logger.info("Top 10 by corrected p-value:\n%s", top.to_string(index=False))

    sig_count = (summary_df["p_value_corrected"] < sig_level).sum()
    logger.info(
        "\n%d / %d tests significant after FDR correction (α=%.2f).",
        sig_count, len(summary_df), sig_level,
    )


def run_full_analysis(
    instruments: list[str] | None = None,
    timeframes: list[str] | None = None,
    use_synthetic: bool = False,
    skip_expensive: bool = False,
) -> None:
    np.random.seed(CONFIG["random_seed"])

    instruments = instruments or list(CONFIG["instruments"].keys())
    timeframes = timeframes or CONFIG["timeframes"]

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
        import json
        from itertools import product
        
        sweep_grid = CONFIG.get("cc_param_sweep", {})
        keys = list(sweep_grid.keys())
        values = list(sweep_grid.values())
        
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        grid_str = json.dumps(sweep_grid, sort_keys=True)
        grid_hash = hashlib.sha256(grid_str.encode()).hexdigest()[:8]
        
        if skip_expensive:
            logger.info("skip_expensive=True: Truncating combinations and test matrix for fast validation.")
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
        
        logger.info("=" * 80)
        logger.info("STARTING RUN: %s", run["name"])
        if run["cc_params"]:
            logger.info("CC PARAMS: %s", run["cc_params"])
        logger.info("=" * 80)
        
        _execute_run(instruments, timeframes, use_synthetic, skip_expensive)


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Volume Spike Directional Prediction Pipeline",
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
        help="Timeframe(s) to test, e.g. M5 M15",
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
