"""
Configuration constants and global parameters for the trading system.

This module now uses YAML-based configuration via ConfigLoader.
Legacy constants are still available for backward compatibility.
"""

from typing import Any, Dict, Optional

from src.config_loader import (
    ConfigLoader,
    get_config as _get_config,
    get_asset_param as _get_asset_param,
)

_config_loader: Optional[ConfigLoader] = None


def _get_loader() -> ConfigLoader:
    """Get or create the ConfigLoader singleton."""
    global _config_loader
    if _config_loader is None:
        _config_loader = _get_config()
    return _config_loader


class TradingConfig:
    """Trading configuration with YAML override support."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback to hardcoded defaults."""
        loader = _get_loader()
        value = loader.get(key)
        if value is not None:
            return value
        return default

    @property
    def INITIAL_CAPITAL(self) -> float:
        return self.get('global.initial_capital', 1000.0)

    @property
    def MAKER_FEE(self) -> float:
        return self.get('fees.maker_fee', 0.0015)

    @property
    def TAKER_FEE(self) -> float:
        return self.get('fees.taker_fee', 0.0025)

    @property
    def DEFAULT_ASSET(self) -> str:
        return self.get('trading.default_asset', 'ETH-USD')

    @property
    def START_DATE(self) -> str:
        return self.get('data.start_date', '2023-01-01')

    @property
    def END_DATE(self) -> str:
        return self.get('data.end_date', '2025-01-01')

    @property
    def ML_LOOKBACK_DAYS(self) -> int:
        return self.get('training.ml_lookback_days', 180)

    @property
    def PROJECT_ROOT(self) -> str:
        return _get_loader()._find_project_root()

    @property
    def CACHE_DIR(self) -> str:
        cache = self.get('paths.cache_dir', 'cache')
        return str(self.PROJECT_ROOT / cache)

    @property
    def TEST_STATE_FILE(self) -> str:
        return str(self.PROJECT_ROOT / "test_portfolio_state.json")

    @property
    def CDP_KEY_FILE(self) -> str:
        return str(self.PROJECT_ROOT / "cdp_api_key.json")

    @property
    def TOP_CRYPTOCURRENCIES(self) -> list:
        return self.get('assets.top_cryptocurrencies', [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
            "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD",
        ])

    @property
    def SUPPORTED_PAIRS(self) -> Dict[str, list]:
        return self.get('exchanges.supported_pairs', {
            "coinbase": self.TOP_CRYPTOCURRENCIES,
            "binance": [p.replace("USD", "USDT") for p in self.TOP_CRYPTOCURRENCIES],
            "kraken": self.TOP_CRYPTOCURRENCIES,
            "bybit": self.TOP_CRYPTOCURRENCIES,
        })

    @property
    def SYMBOL_ALIASES(self) -> Dict[str, Dict[str, str]]:
        return self.get('exchanges.symbol_aliases', {
            "binance": {
                "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "BNB-USD": "BNBUSDT",
                "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT", "ADA-USD": "ADAUSDT",
                "AVAX-USD": "AVAXUSDT", "DOGE-USD": "DOGEUSDT",
                "DOT-USD": "DOTUSDT", "LINK-USD": "LINKUSDT",
            },
            "kraken": {
                "BTC-USD": "XBT/USD", "ETH-USD": "ETH/USD", "BNB-USD": "BNB/USD",
                "SOL-USD": "SOL/USD", "XRP-USD": "XRP/USD", "ADA-USD": "ADA/USD",
                "AVAX-USD": "AVAX/USD", "DOGE-USD": "DOGE/USD",
                "DOT-USD": "DOT/USD", "LINK-USD": "LINK/USD",
            },
        })

    @property
    def GRANULARITY_TO_SECONDS(self) -> Dict[str, int]:
        return self.get('timeframe.granularity_to_seconds', {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600, "FOUR_HOUR": 14400, "ONE_DAY": 86400,
        })

    @property
    def GRANULARITY_TO_FREQ(self) -> Dict[str, str]:
        return self.get('timeframe.granularity_to_freq', {
            "ONE_MINUTE": "1m", "FIVE_MINUTE": "5m", "FIFTEEN_MINUTE": "15m",
            "FOUR_HOUR": "4h", "ONE_HOUR": "1h", "ONE_DAY": "1d",
        })


trading_config = TradingConfig()

INITIAL_CAPITAL = trading_config.INITIAL_CAPITAL
MAKER_FEE = trading_config.MAKER_FEE
TAKER_FEE = trading_config.TAKER_FEE
DEFAULT_ASSET = trading_config.DEFAULT_ASSET
START_DATE = trading_config.START_DATE
END_DATE = trading_config.END_DATE
ML_LOOKBACK_DAYS = trading_config.ML_LOOKBACK_DAYS
PROJECT_ROOT = trading_config.PROJECT_ROOT
TEST_STATE_FILE = trading_config.TEST_STATE_FILE
CACHE_DIR = trading_config.CACHE_DIR
CDP_KEY_FILE = trading_config.CDP_KEY_FILE
TOP_CRYPTOCURRENCIES = trading_config.TOP_CRYPTOCURRENCIES
SUPPORTED_PAIRS = trading_config.SUPPORTED_PAIRS
SYMBOL_ALIASES = trading_config.SYMBOL_ALIASES
GRANULARITY_TO_SECONDS = trading_config.GRANULARITY_TO_SECONDS
GRANULARITY_TO_FREQ = trading_config.GRANULARITY_TO_FREQ


def get_best_params(asset: str = None) -> Dict[str, Any]:
    """Get optimized parameters for an asset.

    Args:
        asset: Asset symbol. If None, returns default best params.

    Returns:
        Dictionary of optimized parameters
    """
    if asset:
        return {
            "rsi_window": _get_asset_param(asset, "strategy.rsi_window", 14),
            "trail_pct": _get_asset_param(asset, "strategy.trail_pct", 0.0224),
            "buy_quantile": _get_asset_param(asset, "strategy.buy_quantile", 0.28),
            "sell_quantile": _get_asset_param(asset, "strategy.sell_quantile", 0.67),
            "ml_thresh": _get_asset_param(asset, "strategy.ml_thresh", 0.41),
            "granularity": _get_asset_param(asset, "strategy.granularity", "ONE_DAY"),
            "atr_multiplier": _get_asset_param(asset, "strategy.atr_multiplier", 3.14),
            "max_drawdown_pct": _get_asset_param(asset, "risk.max_drawdown_pct", 0.22),
        }

    return {
        "rsi_window": 14,
        "trail_pct": 0.0224,
        "buy_quantile": 0.28,
        "sell_quantile": 0.67,
        "ml_thresh": 0.41,
        "granularity": "ONE_DAY",
        "atr_multiplier": 3.14,
        "max_drawdown_pct": 0.22,
    }


BEST_PARAMS = get_best_params()
