"""
Configuration constants and global parameters for the trading system.
"""

import os

# Trading constants
INITIAL_CAPITAL = 1000
MAKER_FEE = 0.0015
TAKER_FEE = 0.0025

# Default trading asset (ETH selected for best risk-adjusted returns)
DEFAULT_ASSET = "ETH-USD"

# Best parameters found during optimization (used for live trading)
# Based on multi-asset comparison: ETH offers best balance
# Return: 74.99%, Sharpe: 1.75, Win Rate: 72.7%, Drawdown: 18.86%
BEST_PARAMS = {
    "rsi_window": 14,
    "trail_pct": 0.08,
    "buy_quantile": 0.20,
    "sell_quantile": 0.80,
    "ml_thresh": 0.50,
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
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"

# ML training parameters
ML_LOOKBACK_DAYS = 180  # Use only last N days for training

# File paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_STATE_FILE = os.path.join(PROJECT_ROOT, "test_portfolio_state.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CDP_KEY_FILE = os.path.join(PROJECT_ROOT, "cdp_api_key.json")

# Top 10 cryptocurrencies by trading volume (excluding stablecoins)
# Based on major exchange volumes
TOP_CRYPTOCURRENCIES = [
    "BTC-USD",  # Bitcoin - #1 by volume
    "ETH-USD",  # Ethereum - #2 by volume
    "BNB-USD",  # Binance Coin - #3 by volume
    "SOL-USD",  # Solana - #4 by volume
    "XRP-USD",  # Ripple - #5 by volume
    "ADA-USD",  # Cardano - #6 by volume
    "AVAX-USD",  # Avalanche - #7 by volume
    "DOGE-USD",  # Dogecoin - #8 by volume
    "DOT-USD",  # Polkadot - #9 by volume
    "LINK-USD",  # Chainlink - #10 by volume
]

# Trading pairs supported by major exchanges
SUPPORTED_PAIRS = {
    "coinbase": TOP_CRYPTOCURRENCIES,
    "binance": [
        pair.replace("USD", "USDT") for pair in TOP_CRYPTOCURRENCIES
    ],  # Binance uses USDT
    "kraken": TOP_CRYPTOCURRENCIES,
    "bybit": TOP_CRYPTOCURRENCIES,
}

# Symbol aliases for different exchanges
SYMBOL_ALIASES = {
    "binance": {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "BNB-USD": "BNBUSDT",
        "SOL-USD": "SOLUSDT",
        "XRP-USD": "XRPUSDT",
        "ADA-USD": "ADAUSDT",
        "AVAX-USD": "AVAXUSDT",
        "DOGE-USD": "DOGEUSDT",
        "DOT-USD": "DOTUSDT",
        "LINK-USD": "LINKUSDT",
    },
    "kraken": {
        "BTC-USD": "XBT/USD",
        "ETH-USD": "ETH/USD",
        "BNB-USD": "BNB/USD",
        "SOL-USD": "SOL/USD",
        "XRP-USD": "XRP/USD",
        "ADA-USD": "ADA/USD",
        "AVAX-USD": "AVAX/USD",
        "DOGE-USD": "DOGE/USD",
        "DOT-USD": "DOT/USD",
        "LINK-USD": "LINK/USD",
    },
}
