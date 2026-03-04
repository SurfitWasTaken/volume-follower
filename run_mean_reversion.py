#!/usr/bin/env python3
"""
Mean-Reversion Variant — Run Script

Tests the inverse hypothesis: volume spikes predict short-term
mean reversion (bullish spike → short, bearish spike → long).

Uses the SAME locked test matrix, parameters, filters, and
statistical corrections as the original continuation run.
Only the trade direction is flipped.

Output: output/mean_reversion/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from main import run_full_analysis

if __name__ == "__main__":
    # ── Activate mean-reversion mode ─────────────────────────────────
    CONFIG["mean_reversion"] = True

    # ── Output to dedicated directory ────────────────────────────────
    CONFIG["output_dir"] = "output/mean_reversion"

    # ── Baseline only (no CC extension for this hypothesis test) ─────
    CONFIG["cc_enabled"] = False

    # ── Force run even if sample size estimation warns ───────────────
    CONFIG["force_run_insufficient_data"] = True

    print("=" * 80)
    print("MEAN-REVERSION VARIANT: Flipping trade directions")
    print("  Bullish volume spike → SHORT")
    print("  Bearish volume spike → LONG")
    print("=" * 80)

    run_full_analysis(
        instruments=["EUR_USD"],
        timeframes=["M5"],
        use_synthetic=False,
        skip_expensive=True,  # Skip permutation/walk-forward on first pass
    )

    print("\n" + "=" * 80)
    print("DONE. Results in output/mean_reversion/")
    print("=" * 80)
