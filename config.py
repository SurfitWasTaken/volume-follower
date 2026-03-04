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
    "timeframes": ["M1", "M5", "M15", "H1", "H4"],

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
    "min_move_atr": 1.5,         # Profit target in ATR units
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

    # ─── CROSS CURRENCY EXTENSION ─────────────────────────────────────────
    "cc_enabled": True,                    # Master switch for the entire extension
    "cc_secondary_instrument": "GBP_USD",  # Secondary instrument to load
    "cc_corr_lookback": 20,                # Rolling correlation lookback (candles)
    "cc_min_eur_move_score": 0.0,          # Min |eur_move_score| to classify as EUR-driven
    "cc_max_gbpusd_vol_z": 2.5,            # Max GBP/USD vol Z-score for EUR-driven classification
    "cc_skip_low_corr_regime": True,       # Skip signals in LOW_CORR regime
    "cc_size_high_corr_mult": 1.5,         # Position multiplier for HIGH_CORR regime
    "cc_size_med_corr_mult": 1.0,          # Position multiplier for MED_CORR regime
    "cc_size_low_corr_mult": 0.5,          # Position multiplier for LOW_CORR regime
    "cc_size_high_eur_move_mult": 1.2,     # Additional multiplier for top-decile eur_move_score
    "cc_size_low_eur_move_mult": 0.8,      # Multiplier for below-median eur_move_score
    "cc_size_gbpusd_vol_penalty": 0.5,     # Penalty multiplier for high GBP/USD vol at signal time
    "cc_size_max_mult": 2.0,               # Maximum position size multiplier
    "cc_size_min_mult": 0.25,              # Minimum position size multiplier (if trade taken)

    "cc_param_sweep": {
        "cc_corr_lookback": [10, 20],
        "cc_min_eur_move_score": [0.0, 0.0001],
        "cc_max_gbpusd_vol_z": [2.0, 3.0],
        "cc_skip_low_corr_regime": [True, False]
    },

    # ─── V2 OVERHAUL: PRE-COMMITMENT PARAMETERS ─────────────────────────
    # These must be set before any data is seen.  Changing them after
    # pipeline start triggers a contamination flag.
    "primary_timeframe": "H1",            # LOCKED.  Change requires re-running preflight.
    "secondary_timeframes": [],           # Validation only, never used for hypothesis selection.
    "max_history_years": 5,               # Fetch up to 5 years of history (OANDA practice limit).
    "n_min_signals": 150,                 # Hard minimum for reportable results.
    "n_min_signals_chart": 100,           # Hard minimum to generate any chart.
    "force_run_insufficient_data": False, # Override gate (flags ALL outputs as WARNING).
    "pre_commitment_hash": None,          # Auto-populated by pre_commitment_log.py.

    # ─── V2 OVERHAUL: WALK-FORWARD PARAMETERS ───────────────────────────
    "wf_train_months": 18,                # Training window for walk-forward.
    "wf_test_months": 3,                  # Test window for walk-forward.
    "wf_min_windows": 6,                  # Minimum walk-forward windows required.

    # ─── V2 OVERHAUL: RED FLAG THRESHOLDS ────────────────────────────────
    "max_reportable_sharpe": 5.0,         # Sharpe above this triggers automatic review.
    "min_cohens_h": 0.05,                 # Effect size minimum for economic significance.
    "max_stationarity_delta": 0.10,       # Max win rate difference between time halves.
    "cost_viability_max_be_wr": 0.60,     # Max break-even WR for a timeframe to be viable.
}
