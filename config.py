from __future__ import annotations
"""
Configuration module for Volume Spike Directional Prediction Pipeline.

All tuneable parameters are centralised here. No magic numbers should appear
anywhere else in the codebase — reference CONFIG[...] instead.
"""

CONFIG = {
    # ─── Random seed for reproducibility ──────────────────────────────────
    "random_seed": 42,

    # ─── Instruments ──────────────────────────────────────────────────────
    # Each entry: display_name, instrument_type, typical_spread_pips
    # Spreads are OANDA practice-account typical values.
    "instruments": {
        "EUR_USD":     {"name": "EUR/USD",       "type": "forex", "pip": 0.0001, "spread_pips": 1.0},
        "GBP_USD":     {"name": "GBP/USD",       "type": "forex", "pip": 0.0001, "spread_pips": 1.4},
        "SPX500_USD":  {"name": "S&P 500 CFD",   "type": "cfd",   "pip": 0.1,    "spread_pips": 5.0},
        "NAS100_USD":  {"name": "Nasdaq 100 CFD","type": "cfd",   "pip": 0.1,    "spread_pips": 10.0},
        "XAU_USD":     {"name": "Gold CFD",      "type": "cfd",   "pip": 0.01,   "spread_pips": 3.0},
    },

    # ─── Timeframes ───────────────────────────────────────────────────────
    "timeframes": ["M1", "M5", "M15"],

    # ─── Data date ranges ─────────────────────────────────────────────────
    "data_start": "2023-01-01",
    "data_end": "2026-03-01",
    "in_sample_end": "2025-03-01",       # 2 years in-sample
    "out_of_sample_start": "2025-03-01", # 1 year held-out OOS
    # Blackout buffer: signals whose outcome window would bleed across
    # the IS/OOS boundary are discarded.  Set to max(K_values).
    "blackout_buffer_candles": 50,

    # ─── Signal detection — Variant A (Rolling Z-Score) ───────────────────
    "lookback": 20,
    "z_threshold": 2.0,

    # ─── Signal detection — Variant B (Volume Multiplier) ─────────────────
    "multiplier": 2.5,

    # ─── Signal detection — Variant C (Percentile Rank) ───────────────────
    "percentile_threshold": 95,

    # ─── Signal detection — Variant D (Adaptive / VWAP-normalised) ────────
    # Uses session-hour volume profile normalisation; no extra params needed.

    # ─── Signal independence ──────────────────────────────────────────────
    # Minimum gap (in candles) between successive signals to avoid
    # overlapping outcome windows and serial correlation.
    "min_signal_gap": 10,

    # ─── Signal filters ───────────────────────────────────────────────────
    "min_body_ratio": 0.5,       # Filter 2 — Body-to-Range
    "wick_ratio": 1.0,           # Filter 3 — Wick Asymmetry

    # Filter 4 — Session windows (UTC: start_hour, start_min, end_hour, end_min)
    "session_windows": {
        "london":  (8, 0, 9, 30),   # 08:00 – 09:30 UTC
        "newyork": (13, 30, 15, 0),  # 13:30 – 15:00 UTC
    },

    # Filter 5 — News exclusion buffer (minutes)
    "news_buffer_minutes": 15,

    # Filter 6 — Trend context
    "ma_short": 50,
    "ma_long": 200,

    # Filter 7 — Prior range position
    "range_lookback": 20,

    # ─── Outcome calculation ──────────────────────────────────────────────
    "K_values": [3, 5, 10, 20, 50],
    "atr_period": 14,
    "min_move_atr": 0.5,         # Profit target in ATR units
    "stop_distance_atr": 1.0,    # Stop loss in ATR units

    # ─── Transaction costs (OANDA spread-based model) ─────────────────────
    # No per-trade commission on OANDA.  Costs are modelled purely via
    # bid-ask spread + a widening factor for fast markets.
    "spread_widening_factor": 1.5,    # multiply spread during session opens
    "slippage_atr_fraction": 0.0,     # additional beyond spread (set 0 since spread covers it)

    # ─── Statistical inference ────────────────────────────────────────────
    "bootstrap_n": 10_000,
    "bootstrap_block_size": 20,        # block bootstrap to handle serial correlation
    "permutation_n": 1_000,
    "walk_forward_train_months": 6,
    "walk_forward_test_months": 1,
    "significance_level": 0.05,

    # ─── Output ───────────────────────────────────────────────────────────
    "output_dir": "output",
    "data_dir": "data",
}
