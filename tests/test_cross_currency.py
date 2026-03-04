import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_currency_features import compute_cross_currency_features
from adaptive_position_sizer import compute_position_sizes
from config import CONFIG

class TestCrossCurrency(unittest.TestCase):

    def setUp(self):
        self.mock_config = CONFIG.copy()
        self.mock_config.update({
            "cc_corr_lookback": 5,
            "cc_min_eur_move_score": 0.0001,
            "cc_max_gbpusd_vol_z": 1.0,
            "cc_skip_low_corr_regime": True,
            "cc_size_high_corr_mult": 1.5,
            "cc_size_med_corr_mult": 1.0,
            "cc_size_low_corr_mult": 0.5,
            "cc_size_high_eur_move_mult": 1.2,
            "cc_size_low_eur_move_mult": 0.8,
            "cc_size_gbpusd_vol_penalty": 0.5,
            "lookback": 5
        })
        
        # 10 candles
        self.idx = pd.date_range("2025-01-01 08:00", periods=10, freq="5min", tz="UTC")
        
        # Primary: steadily going up
        p_open = np.linspace(1.0, 1.09, 10).copy()
        p_close = p_open + 0.005 # Bullish candles
        self.primary_df = pd.DataFrame({
            "open": p_open, "close": p_close, "high": p_close, "low": p_open, "volume": 100
        }, index=self.idx)
        
        # Secondary: initially going up, then flat
        s_open = np.linspace(1.2, 1.29, 10).copy()
        s_close = s_open.copy() # Flat candles except last one
        self.secondary_df = pd.DataFrame({
            "open": s_open, "close": s_close, "high": s_close + 0.01, "low": s_open, "volume": 100
        }, index=self.idx)
        
        # Make candle 8 a massive move in primary, 0 move in secondary
        self.primary_df.loc[self.idx[8], "open"] = 1.08
        self.primary_df.loc[self.idx[8], "close"] = 1.09
        
        self.secondary_df.loc[self.idx[8], "open"] = 1.28
        self.secondary_df.loc[self.idx[8], "close"] = 1.28
        
        # Candle 9: big move in both (not EUR driven)
        self.primary_df.loc[self.idx[9], "open"] = 1.09
        self.primary_df.loc[self.idx[9], "close"] = 1.10
        self.secondary_df.loc[self.idx[9], "open"] = 1.29
        self.secondary_df.loc[self.idx[9], "close"] = 1.30

    def test_identical_moves(self):
        # Force candle 7 to have identical % moves
        p_open_val = 1.07
        p_close_val = p_open_val * 1.005
        self.primary_df.loc[self.idx[7], "open"] = p_open_val
        self.primary_df.loc[self.idx[7], "close"] = p_close_val
        
        s_open_val = 1.27
        s_close_val = s_open_val * 1.005
        self.secondary_df.loc[self.idx[7], "open"] = s_open_val
        self.secondary_df.loc[self.idx[7], "close"] = s_close_val
        
        signals = self.idx[[7]]
        features = compute_cross_currency_features(self.primary_df, self.secondary_df, signals, self.mock_config)
        
        self.assertEqual(len(features), 1)
        self.assertLess(abs(features.iloc[0]["eur_move_score"]), 1e-9)
        self.assertFalse(features.iloc[0]["is_eur_driven"])

    def test_gbpusd_vol_classification(self):
        # Candle 8 is EUR driven (EUR moves, GBP stays flat)
        # BUT let's spike GBP volume to disqualify it
        self.secondary_df.loc[self.idx[8], "volume"] = 1000 # Massive spike
        
        signals = self.idx[[8]]
        features = compute_cross_currency_features(self.primary_df, self.secondary_df, signals, self.mock_config)
        
        self.assertGreater(features.iloc[0]["gbpusd_vol_zscore"], self.mock_config["cc_max_gbpusd_vol_z"])
        self.assertFalse(features.iloc[0]["is_eur_driven"]) # Fails condition

    def test_zero_scalar_on_non_eur_driven(self):
        df = pd.DataFrame({
            "is_eur_driven": [False, True, True],
            "corr_regime": ["HIGH_CORR", "LOW_CORR", "HIGH_CORR"],
            "eur_move_score": [0.005, 0.005, 0.005],
            "gbpusd_vol_zscore": [0.0, 0.0, 0.0]
        }, index=pd.date_range("2025-01-01", periods=3))
        
        sizes = compute_position_sizes(df, self.mock_config, base_size=1.0)
        
        self.assertEqual(sizes.iloc[0], 0.0)
        self.assertEqual(sizes.iloc[1], 0.0)
        self.assertGreater(sizes.iloc[2], 0.0)

    def test_backwards_compatibility_when_disabled(self):
        primary_df = pd.DataFrame()
        secondary_df = pd.DataFrame()
        features = compute_cross_currency_features(primary_df, secondary_df, pd.DatetimeIndex([]), CONFIG)
        self.assertTrue(features.empty)

if __name__ == "__main__":
    unittest.main()
