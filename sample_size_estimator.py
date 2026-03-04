from __future__ import annotations
"""
Sample size estimation module.

Runs a pilot analysis on the first 6 months of data to estimate
expected signal frequency post-filtering.  Determines whether the
available history is sufficient to reach the minimum signal threshold
(n_min_signals from CONFIG, default 150).

If insufficient, the pipeline halts unless force_run is set.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


def estimate_sample_size(
    df: pd.DataFrame,
    instrument: str,
    timeframe: str,
    signal_generator,
    signal_filter,
    variants: list[str],
    session_norm_options: list[bool],
    filter_ladders: list[list[str]],
    output_dir: str | None = None,
) -> dict:
    """
    Run a pilot signal count on the first 6 months of data.

    Parameters
    ----------
    df : pd.DataFrame
        Full available data (will be sliced to first 6 months internally).
    instrument, timeframe : str
    signal_generator : SignalGenerator instance
    signal_filter : SignalFilter instance
    variants, session_norm_options, filter_ladders : test matrix components

    Returns
    -------
    dict with keys:
        instrument, timeframe,
        pilot_candles, pilot_months,
        raw_signals_per_year (dict by variant),
        filtered_signals_per_year (dict by variant+filter combo),
        min_signals_per_year (worst case),
        available_years, estimated_total_signals,
        min_history_needed_years,
        status (SUFFICIENT / INSUFFICIENT)
    """
    n_min = CONFIG.get("n_min_signals", 150)

    # Slice to first 6 months for pilot
    start = df.index.min()
    pilot_end = start + pd.DateOffset(months=6)
    df_pilot = df[df.index < pilot_end]

    if len(df_pilot) < 100:
        logger.warning(
            "Pilot data too short for %s %s: %d candles",
            instrument, timeframe, len(df_pilot),
        )
        return _insufficient_result(instrument, timeframe, 0, 0, n_min)

    pilot_months = (df_pilot.index.max() - df_pilot.index.min()).days / 30.44
    if pilot_months < 1:
        pilot_months = 1.0

    # Available history
    total_months = (df.index.max() - df.index.min()).days / 30.44
    available_years = total_months / 12.0

    raw_signals = {}
    filtered_signals = {}

    for var in variants:
        for sn in session_norm_options:
            if var == "D" and sn:
                continue
            label = f"{var}{'_sn' if sn else ''}"

            # Raw signals (no filtering except direction)
            try:
                signals = signal_generator.detect(df_pilot, var, session_normalised=sn)
                n_raw = int(signals.sum())
                raw_per_year = n_raw / pilot_months * 12
                raw_signals[label] = {
                    "n_pilot": n_raw,
                    "per_year": round(raw_per_year, 1),
                }

                # Apply each filter ladder
                for filters in filter_ladders:
                    all_filters = ["direction"] + [f for f in filters if f != "direction"]
                    filt_key = f"{label}|{'+'.join(all_filters)}"

                    try:
                        filtered_sigs, _ = signal_filter.apply(
                            df_pilot, signals, all_filters,
                        )
                        n_filt = int(filtered_sigs.sum())
                        filt_per_year = n_filt / pilot_months * 12

                        filtered_signals[filt_key] = {
                            "n_pilot": n_filt,
                            "per_year": round(filt_per_year, 1),
                            "cascade": "+".join(all_filters),
                        }
                    except Exception as e:
                        logger.debug("Filter cascade failed for %s: %s", filt_key, e)

            except Exception as e:
                logger.warning("Pilot signal generation failed for %s: %s", label, e)

    # Find worst-case (most filtered) signal rate
    all_rates = [v["per_year"] for v in filtered_signals.values() if v["per_year"] > 0]
    if not all_rates:
        return _insufficient_result(instrument, timeframe, len(df_pilot), available_years, n_min)

    min_rate = min(all_rates)
    max_rate = max(all_rates)
    median_rate = float(np.median(all_rates))

    estimated_total = median_rate * available_years
    min_history_needed = n_min / max(median_rate, 0.01)

    status = "SUFFICIENT" if estimated_total >= n_min else "INSUFFICIENT"

    result = {
        "instrument": instrument,
        "timeframe": timeframe,
        "pilot_candles": len(df_pilot),
        "pilot_months": round(pilot_months, 1),
        "raw_signals": raw_signals,
        "filtered_signals_per_year_min": min_rate,
        "filtered_signals_per_year_max": max_rate,
        "filtered_signals_per_year_median": median_rate,
        "available_years": round(available_years, 1),
        "estimated_total_signals": round(estimated_total),
        "min_history_needed_years": round(min_history_needed, 1),
        "n_min_required": n_min,
        "status": status,
    }

    # Write report
    _write_report(result, output_dir)

    logger.info(
        "Sample size estimate %s %s: median %.1f signals/yr × %.1f yr = ~%d signals → %s",
        instrument, timeframe, median_rate, available_years,
        estimated_total, status,
    )

    return result


def _insufficient_result(inst, tf, n_candles, avail_years, n_min):
    return {
        "instrument": inst, "timeframe": tf,
        "pilot_candles": n_candles, "pilot_months": 0,
        "raw_signals": {}, "filtered_signals_per_year_min": 0,
        "filtered_signals_per_year_max": 0,
        "filtered_signals_per_year_median": 0,
        "available_years": round(avail_years, 1),
        "estimated_total_signals": 0,
        "min_history_needed_years": float("inf"),
        "n_min_required": n_min,
        "status": "INSUFFICIENT",
    }


def _write_report(result: dict, output_dir: str | None = None) -> None:
    out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Sample Size Estimate: {result['instrument']} {result['timeframe']}\n",
        f"Pilot window: {result['pilot_candles']} candles ({result['pilot_months']:.1f} months)\n",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Signal frequency (min, post-filter) | {result['filtered_signals_per_year_min']:.1f} / year |",
        f"| Signal frequency (median, post-filter) | {result['filtered_signals_per_year_median']:.1f} / year |",
        f"| Signal frequency (max, post-filter) | {result['filtered_signals_per_year_max']:.1f} / year |",
        f"| Available history | {result['available_years']:.1f} years |",
        f"| Estimated total signals | {result['estimated_total_signals']} |",
        f"| Minimum required | {result['n_min_required']} |",
        f"| Min history needed (at median rate) | {result['min_history_needed_years']:.1f} years |",
        "",
        f"## STATUS: **{result['status']}**\n",
    ]

    if result["status"] == "INSUFFICIENT":
        lines.extend([
            "> [!CAUTION]",
            "> The available data does not contain enough signals for statistically",
            "> reportable results. Either fetch more history, relax filters, or",
            f"> accept that results from {result['estimated_total_signals']} signals",
            "> cannot support any statistical claims.",
        ])

    path = out / "sample_size_estimate.md"
    path.write_text("\n".join(lines))

    # Also write JSON for machine consumption
    json_path = out / "sample_size_estimate.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
