"""
Configuration constants and global parameters for the trading system.
"""

import os

# Trading constants
INITIAL_CAPITAL = 1000
MAKER_FEE = 0.0015
TAKER_FEE = 0.0025

# Best parameters found during optimization (used for live trading)
BEST_PARAMS = {
    "rsi_window": 14,
    "trail_pct": 0.10,
    "buy_quantile": 0.2,
    "sell_quantile": 0.8,
    "ml_thresh": 0.5,
    "granularity": "ONE_DAY",
}

# Timeframe mappings
GRANULARITY_TO_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
    "FOUR_HOUR": 14400,
    "ONE_DAY": 86400,
}

GRANULARITY_TO_FREQ = {
    "ONE_MINUTE": "1m",
    "FIVE_MINUTE": "5m",
    "FIFTEEN_MINUTE": "15m",
    "FOUR_HOUR": "4h",
    "ONE_HOUR": "1h",
    "ONE_DAY": "1d",
}

# Trading dates
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"

# File paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_STATE_FILE = os.path.join(PROJECT_ROOT, "test_portfolio_state.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CDP_KEY_FILE = os.path.join(PROJECT_ROOT, "cdp_api_key.json")
