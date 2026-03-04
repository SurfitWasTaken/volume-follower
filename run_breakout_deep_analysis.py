#!/usr/bin/env python3
"""
Breakout Filter Deep Analysis — Tasks A, B, C

Task A: Expand sample (full history + relaxed body_ratio)
Task B: Cost structure diagnostic (London K=3, Both K=3)
Task C: OOS validation (London K=3, Both K=3)

Output: output/breakout_deep/
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

OUTPUT_DIR = "output/breakout_deep"
INSTRUMENT = "EUR_USD"
TIMEFRAME = "M5"

LONDON_HOURS = set(range(8, 10))
BOTH_HOURS = set(range(8, 10)) | set(range(13, 16))

def _setup_logging():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s"
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(OUTPUT_DIR) / "pipeline.log", mode="w"),
        ],
    )

log = logging.getLogger(__name__)


def _sharpe(arr):
    if len(arr) > 1 and np.std(arr) > 0:
        return np.mean(arr) / np.std(arr) * np.sqrt(252)
    return 0.0


def run_breakout_pipeline(
    df: pd.DataFrame,
    body_ratio_threshold: float,
    session_hours: set[int] | None,
    K: int,
    apply_costs: bool = True,
    label: str = "",
) -> dict:
    """Run the breakout pipeline on given data and return result dict."""
    sig_gen = SignalGenerator()
    sig_filt = SignalFilter()
    oc = OutcomeCalculator()
    st = StatisticalTests()

    # Detect spikes (Variant B)
    spikes = sig_gen.detect(df, "B", session_normalised=False)
    n_spikes_raw = int(spikes.sum())

    # Apply body_ratio + wick filters (with custom threshold)
    old_br = CONFIG["min_body_ratio"]
    CONFIG["min_body_ratio"] = body_ratio_threshold
    spike_filtered, _ = sig_filt.apply(df, spikes, ["direction", "body_ratio", "wick"])
    CONFIG["min_body_ratio"] = old_br  # restore
    n_spikes_filtered = int(spike_filtered.sum())

    # Breakout detection
    breakout_signals, breakout_dirs = detect_breakout_signals(df, spike_filtered)
    n_breakouts_total = int(breakout_signals.sum())

    # Session filter on breakout signals (T+1 candle time)
    if session_hours is not None:
        session_mask = breakout_signals & df.index.to_series().apply(
            lambda t: t.hour in session_hours
        )
    else:
        session_mask = breakout_signals.copy()

    session_dirs = breakout_dirs.where(session_mask, 0)
    n_session = int(session_mask.sum())

    if n_session < 3:
        return {
            "label": label, "n_spikes_raw": n_spikes_raw,
            "n_spikes_filtered": n_spikes_filtered,
            "n_breakouts_total": n_breakouts_total,
            "n_session": n_session, "n_valid": 0,
            "win_rate": np.nan, "base_rate": np.nan,
            "win_rate_excess": np.nan, "p_raw": 1.0,
            "sharpe_gross": np.nan, "sharpe_net": np.nan,
            "mean_gross_return_atr": np.nan, "mean_net_return_atr": np.nan,
            "mean_cost_atr": np.nan, "outcomes": None,
        }

    # Outcomes
    outcomes = oc.compute_outcomes(df, session_mask, session_dirs, K_values=[K])
    if apply_costs:
        outcomes = oc.apply_costs(outcomes, INSTRUMENT, K=K)

    win_col = f"win_{K}"
    htc_col = f"htc_return_{K}"
    htc_net_col = f"htc_return_net_{K}"
    cost_col = f"cost_atr_{K}"

    wins = outcomes[win_col].dropna()
    n_valid = len(wins)
    n_wins = int((wins == 1.0).sum())
    win_rate = n_wins / n_valid if n_valid > 0 else np.nan

    # Base rate (unconditional breakout rate)
    uncond = compute_unconditional_breakout_rate(df, K, session_hours=session_hours)
    base_rate = uncond["win_rate"]

    p_raw = st.binomial_test(n_wins, n_valid, base_rate) if not np.isnan(base_rate) else 1.0

    gross_returns = outcomes[htc_col].dropna().values
    net_returns = outcomes[htc_net_col].dropna().values if htc_net_col in outcomes.columns else gross_returns

    sharpe_gross = _sharpe(gross_returns)
    sharpe_net = _sharpe(net_returns) if apply_costs else np.nan

    mean_cost_atr = outcomes[cost_col].mean() if cost_col in outcomes.columns else np.nan

    return {
        "label": label,
        "n_spikes_raw": n_spikes_raw,
        "n_spikes_filtered": n_spikes_filtered,
        "n_breakouts_total": n_breakouts_total,
        "n_session": n_session,
        "n_valid": n_valid,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "base_rate": base_rate,
        "base_rate_n": uncond["n_evaluated"],
        "win_rate_excess": win_rate - base_rate if not np.isnan(win_rate) and not np.isnan(base_rate) else np.nan,
        "p_raw": p_raw,
        "sharpe_gross": sharpe_gross,
        "sharpe_net": sharpe_net,
        "mean_gross_return_atr": float(np.nanmean(gross_returns)) if len(gross_returns) > 0 else np.nan,
        "mean_net_return_atr": float(np.nanmean(net_returns)) if len(net_returns) > 0 else np.nan,
        "mean_cost_atr": mean_cost_atr,
        "outcomes": outcomes,
    }


def task_b_cost_diagnostic(df_is: pd.DataFrame):
    """Task B: Cost structure diagnostic for London K=3 and Both K=3."""
    log.info("=" * 80)
    log.info("TASK B: COST STRUCTURE DIAGNOSTIC")
    log.info("=" * 80)

    inst_cfg = CONFIG["instruments"][INSTRUMENT]
    pip_value = inst_cfg["pip"]  # 0.0001
    spread_pips = inst_cfg["spread_pips"]  # 1.0

    results = {}

    for session_name, session_hours in [("london", LONDON_HOURS), ("both", BOTH_HOURS)]:
        K = 3
        log.info("─── %s K=%d ───", session_name, K)

        # Run with costs
        res = run_breakout_pipeline(
            df_is, body_ratio_threshold=0.5,
            session_hours=session_hours, K=K,
            apply_costs=True, label=f"{session_name}_K{K}_costs",
        )

        outcomes = res["outcomes"]
        if outcomes is None or len(outcomes) == 0:
            log.warning("No outcomes for %s K=%d — skipping", session_name, K)
            continue

        htc_col = f"htc_return_{K}"
        cost_col = f"cost_atr_{K}"
        htc_net_col = f"htc_return_net_{K}"

        # ATR stats
        mean_atr = outcomes["atr"].mean()
        mean_gross_atr = outcomes[htc_col].dropna().mean()
        mean_cost_atr = outcomes[cost_col].mean() if cost_col in outcomes.columns else np.nan
        mean_net_atr = outcomes[htc_net_col].dropna().mean() if htc_net_col in outcomes.columns else np.nan

        # Convert to pips
        mean_gross_pips = mean_gross_atr * mean_atr / pip_value
        mean_cost_pips = mean_cost_atr * mean_atr / pip_value if not np.isnan(mean_cost_atr) else np.nan

        # TP distance in pips
        tp_atr = CONFIG["min_move_atr"]  # 1.5
        tp_pips = tp_atr * mean_atr / pip_value

        # Cost as % of TP
        cost_pct_of_tp = (mean_cost_atr / tp_atr * 100) if not np.isnan(mean_cost_atr) else np.nan

        # Break-even spread (pips) — find spread where net Sharpe = 0
        # Net return = gross return - cost
        # cost_atr = 2 * spread_price / atr
        # We want: mean(gross_return - 2*spread*widen/atr) / std(...) = 0
        # => mean(gross_return) = mean(2*spread*widen/atr)
        # For simplicity: find spread where mean gross = mean cost
        gross_returns = outcomes[htc_col].dropna().values
        if len(gross_returns) > 0 and not np.isnan(mean_cost_atr):
            # Current cost per unit of spread_pips:
            # cost_atr_per_pip = mean_cost_atr / spread_pips (linear relationship)
            cost_per_spread_pip = mean_cost_atr / spread_pips
            # Break-even: gross = cost => cost_per_pip * be_spread = gross
            be_spread = mean_gross_atr / cost_per_spread_pip if cost_per_spread_pip > 0 else np.nan
        else:
            be_spread = np.nan

        # Break-even win rate at current costs
        # With 1.5 ATR TP and 1.0 ATR SL (after costs):
        # TP_net = TP - cost = 1.5 - cost_atr (in ATR units per trade)
        # SL_net = SL + cost = 1.0 + cost_atr
        # Break-even: WR * TP_net = (1-WR) * SL_net
        # WR = SL_net / (TP_net + SL_net)
        tp_net = tp_atr - mean_cost_atr if not np.isnan(mean_cost_atr) else tp_atr
        sl_net = CONFIG["stop_distance_atr"] + (mean_cost_atr if not np.isnan(mean_cost_atr) else 0)
        be_wr = sl_net / (tp_net + sl_net) if (tp_net + sl_net) > 0 else np.nan

        diag = {
            "session": session_name,
            "K": K,
            "n": res["n_valid"],
            "win_rate": res["win_rate"],
            "mean_atr_pips": mean_atr / pip_value,
            "mean_gross_return_pips": mean_gross_pips,
            "mean_gross_return_atr": mean_gross_atr,
            "mean_cost_pips": mean_cost_pips,
            "mean_cost_atr": mean_cost_atr,
            "cost_pct_of_tp": cost_pct_of_tp,
            "break_even_spread_pips": be_spread,
            "break_even_wr": be_wr,
            "tp_distance_pips": tp_pips,
            "sl_distance_pips": CONFIG["stop_distance_atr"] * mean_atr / pip_value,
            "sharpe_gross": res["sharpe_gross"],
            "sharpe_net": res["sharpe_net"],
        }
        results[session_name] = diag

        log.info(
            "  Mean gross return:       %.4f ATR (%.1f pips)",
            mean_gross_atr, mean_gross_pips,
        )
        log.info(
            "  Mean cost per trade:     %.4f ATR (%.1f pips)",
            mean_cost_atr, mean_cost_pips,
        )
        log.info(
            "  TP distance:             %.4f ATR (%.1f pips)",
            tp_atr, tp_pips,
        )
        log.info(
            "  Cost as %% of TP:         %.1f%%",
            cost_pct_of_tp,
        )
        log.info(
            "  Break-even spread:       %.2f pips (current: %.1f pips)",
            be_spread, spread_pips,
        )
        log.info(
            "  Break-even WR (w/ costs): %.1f%% (actual WR: %.1f%%)",
            be_wr * 100, res["win_rate"] * 100,
        )
        log.info(
            "  Sharpe gross: %.3f  |  Sharpe net: %.3f",
            res["sharpe_gross"], res["sharpe_net"],
        )

    # Zero-cost run for London K=3
    log.info("")
    log.info("─── London K=3 ZERO-COST (gross edge isolation) ───")
    # Temporarily zero out spreads
    old_spread = CONFIG["instruments"][INSTRUMENT]["spread_pips"]
    CONFIG["instruments"][INSTRUMENT]["spread_pips"] = 0.0
    res_zero = run_breakout_pipeline(
        df_is, body_ratio_threshold=0.5,
        session_hours=LONDON_HOURS, K=3,
        apply_costs=True, label="london_K3_zerocost",
    )
    CONFIG["instruments"][INSTRUMENT]["spread_pips"] = old_spread  # restore

    if res_zero["outcomes"] is not None:
        log.info(
            "  Zero-cost Sharpe: %.3f  (vs net: %.3f)",
            res_zero["sharpe_gross"],
            results.get("london", {}).get("sharpe_net", np.nan),
        )
        results["london_zerocost"] = {
            "sharpe_gross": res_zero["sharpe_gross"],
            "win_rate": res_zero["win_rate"],
            "mean_gross_return_atr": res_zero["mean_gross_return_atr"],
        }

    return results


def task_a_expanded_sample(df_is: pd.DataFrame):
    """Task A: Expand sample with original and relaxed body_ratio."""
    log.info("=" * 80)
    log.info("TASK A: EXPANDED SAMPLE + RELAXED BODY_RATIO")
    log.info("=" * 80)

    data_start = df_is.index.min()
    data_end = df_is.index.max()
    log.info("IS data range: %s → %s (%d candles)",
             data_start, data_end, len(df_is))
    log.info("NOTE: OANDA M5 cache starts at 2023-01-01 — cannot expand further")

    K_VALUES = [3, 5, 10, 20]
    sessions = {
        "london": LONDON_HOURS,
        "newyork": set(range(13, 16)),
        "both": BOTH_HOURS,
        "none": None,
    }

    results = {}
    st = StatisticalTests()

    for br_label, br_threshold in [("original_0.50", 0.50), ("relaxed_0.375", 0.375)]:
        log.info("")
        log.info("═══ Body ratio: %s (threshold=%.3f) ═══", br_label, br_threshold)

        variant_results = []

        for session_name, session_hours in sessions.items():
            for K in K_VALUES:
                res = run_breakout_pipeline(
                    df_is, body_ratio_threshold=br_threshold,
                    session_hours=session_hours, K=K,
                    apply_costs=True,
                    label=f"{br_label}_{session_name}_K{K}",
                )
                variant_results.append(res)

                if session_name in ("london", "both"):
                    log.info(
                        "  %s K=%d: n=%d WR=%.3f base=%.3f excess=%+.3f p=%.4f",
                        session_name, K,
                        res["n_valid"],
                        res["win_rate"] if not np.isnan(res["win_rate"]) else 0,
                        res["base_rate"] if not np.isnan(res["base_rate"]) else 0,
                        res["win_rate_excess"] if not np.isnan(res.get("win_rate_excess", np.nan)) else 0,
                        res["p_raw"],
                    )

        # FDR correction across all tests in this variant
        raw_pvals = np.array([r["p_raw"] for r in variant_results])
        corrected = st.benjamini_hochberg(raw_pvals)
        for i, r in enumerate(variant_results):
            r["p_fdr"] = corrected[i]

        sig_count = (corrected < 0.05).sum()
        log.info("  FDR significant: %d / %d", sig_count, len(variant_results))

        results[br_label] = variant_results

        # Report London signal count
        london_results = [r for r in variant_results if "london" in r["label"]]
        for lr in london_results:
            K_val = int(lr["label"].split("_K")[-1])
            if lr["n_valid"] >= 100:
                log.info(
                    "  >>> London K=%d has n=%d (≥100) — QUALIFIES for full stats",
                    K_val, lr["n_valid"],
                )

    return results


def task_c_oos_validation(df_oos: pd.DataFrame):
    """Task C: OOS validation for London K=3 and Both K=3."""
    log.info("=" * 80)
    log.info("TASK C: OUT-OF-SAMPLE VALIDATION")
    log.info("=" * 80)

    oos_start = df_oos.index.min()
    oos_end = df_oos.index.max()
    log.info("OOS data range: %s → %s (%d candles)", oos_start, oos_end, len(df_oos))

    results = {}

    for session_name, session_hours in [("london", LONDON_HOURS), ("both", BOTH_HOURS)]:
        K = 3
        log.info("─── OOS: %s K=%d ───", session_name, K)

        res = run_breakout_pipeline(
            df_oos, body_ratio_threshold=0.5,
            session_hours=session_hours, K=K,
            apply_costs=True, label=f"oos_{session_name}_K{K}",
        )

        below_20 = res["n_valid"] < 20
        flag = "⚠ DIRECTIONAL ONLY (n < 20)" if below_20 else ""

        log.info(
            "  n=%d WR=%.3f base=%.3f excess=%+.3f p_raw=%.4f Sharpe_net=%.3f %s",
            res["n_valid"],
            res["win_rate"] if not np.isnan(res["win_rate"]) else 0,
            res["base_rate"] if not np.isnan(res["base_rate"]) else 0,
            res["win_rate_excess"] if not np.isnan(res.get("win_rate_excess", np.nan)) else 0,
            res["p_raw"],
            res["sharpe_net"] if not np.isnan(res.get("sharpe_net", np.nan)) else 0,
            flag,
        )
        results[session_name] = res

    return results


def main():
    _setup_logging()
    np.random.seed(CONFIG["random_seed"])
    t0 = time.time()

    CONFIG["mean_reversion"] = False
    CONFIG["cc_enabled"] = False

    log.info("=" * 80)
    log.info("BREAKOUT FILTER DEEP ANALYSIS — Tasks A, B, C")
    log.info("=" * 80)

    # Load data
    loader = DataLoader()
    df_full = loader.fetch_max_history(INSTRUMENT, TIMEFRAME)
    loader.validate_data(df_full)

    df_is, df_oos = loader.split_data(df_full)

    # Apply blackout buffer to IS only
    buffer = CONFIG["blackout_buffer_candles"]
    if len(df_is) > buffer:
        df_is = df_is.iloc[:-buffer]

    log.info("IS: %d candles (%s → %s)",
             len(df_is), df_is.index[0], df_is.index[-1])
    log.info("OOS: %d candles (%s → %s)",
             len(df_oos), df_oos.index[0], df_oos.index[-1])

    # Task B first (most critical)
    cost_results = task_b_cost_diagnostic(df_is)

    # Task A
    expanded_results = task_a_expanded_sample(df_is)

    # Task C
    oos_results = task_c_oos_validation(df_oos)

    # Save all results
    output = {
        "task_b_cost_diagnostic": {},
        "task_a_expanded_sample": {},
        "task_c_oos_validation": {},
    }

    for k, v in cost_results.items():
        output["task_b_cost_diagnostic"][k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items()
        }

    for br_label, variant_results in expanded_results.items():
        output["task_a_expanded_sample"][br_label] = [
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in r.items() if k != "outcomes"}
            for r in variant_results
        ]

    for k, v in oos_results.items():
        output["task_c_oos_validation"][k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items() if kk != "outcomes"
        }

    results_path = Path(OUTPUT_DIR) / "deep_analysis_results.json"
    results_path.write_text(json.dumps(output, indent=2, default=str))
    log.info("Results saved → %s", results_path)

    log.info("")
    log.info("PIPELINE COMPLETE (%.0fs total)", time.time() - t0)


if __name__ == "__main__":
    main()
