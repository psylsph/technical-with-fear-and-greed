"""
Configuration Loader for YAML-based configuration.

This module provides a ConfigLoader class that reads trading configurations
from YAML files and provides a unified interface for accessing configuration values.

Usage:
    from src.config_loader import get_config, get_asset_config
    
    # Get global trading config
    config = get_config()
    initial_capital = config.get('global.initial_capital', 1000.0)
    
    # Get asset-specific config
    asset_config = get_asset_config('BTC-USD')
    stop_loss = asset_config.get('trading.stop_loss_pct', 0.05)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """Load and manage trading configurations from YAML files."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    _asset_configs: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_configs()
        return cls._instance
    
    def _load_configs(self):
        """Load all configuration files."""
        project_root = self._find_project_root()
        config_dir = project_root / "config"
        
        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            return
        
        # Load global trading config
        trading_config = config_dir / "trading.yaml"
        if trading_config.exists():
            with open(trading_config, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        
        # Load asset configs
        assets_dir = config_dir / "assets"
        if assets_dir.exists():
            for config_file in assets_dir.glob("*.yaml"):
                asset_name = config_file.stem.replace("-", "_").upper()
                with open(config_file, 'r') as f:
                    asset_config = yaml.safe_load(f) or {}
                    symbol = asset_config.get('symbol', asset_name)
                    self._asset_configs[symbol] = asset_config
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent
        for _ in range(10):
            if (current / "config").exists() or (current / "AGENTS.md").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent.parent
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'trading.stop_loss_pct')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        
        return value
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """Get a global configuration value.
        
        Args:
            key: Configuration key (e.g., 'initial_capital')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def get_asset_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for a specific asset.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC-USD')
            
        Returns:
            Asset configuration dictionary or empty dict if not found
        """
        return self._asset_configs.get(symbol, {})
    
    def get_asset_param(self, symbol: str, param_path: str, default: Any = None) -> Any:
        """Get a parameter for a specific asset using dot notation.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC-USD')
            param_path: Dot-separated path (e.g., 'trading.stop_loss_pct')
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        asset_config = self.get_asset_config(symbol)
        if not asset_config:
            return default
        
        keys = param_path.split('.')
        value = asset_config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        
        return value
    
    def get_all_assets(self) -> list:
        """Get list of all configured assets."""
        return list(self._asset_configs.keys())
    
    def get_enabled_assets(self, live_trading: bool = False) -> list:
        """Get list of enabled assets.
        
        Args:
            live_trading: If True, only return assets enabled for live trading
            
        Returns:
            List of asset symbols
        """
        enabled = []
        for symbol, config in self._asset_configs.items():
            if live_trading:
                if config.get('live_trading_allowed', False):
                    enabled.append(symbol)
            else:
                if config.get('enabled', True):
                    enabled.append(symbol)
        return enabled
    
    def reload(self):
        """Reload all configuration files."""
        self._config = {}
        self._asset_configs = {}
        self._load_configs()
    
    @property
    def trading_config(self) -> Dict[str, Any]:
        """Get the full trading configuration."""
        return self._config
    
    @property
    def asset_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all asset configurations."""
        return self._asset_configs


# Singleton instance function
def get_config() -> ConfigLoader:
    """Get the global ConfigLoader instance."""
    return ConfigLoader()


def get_asset_config(symbol: str) -> Dict[str, Any]:
    """Get configuration for a specific asset.
    
    Args:
        symbol: Asset symbol (e.g., 'BTC-USD')
        
    Returns:
        Asset configuration dictionary
    """
    return get_config().get_asset_config(symbol)


def get_asset_param(symbol: str, param_path: str, default: Any = None) -> Any:
    """Get a parameter for a specific asset.
    
    Args:
        symbol: Asset symbol (e.g., 'BTC-USD')
        param_path: Dot-separated path (e.g., 'trading.stop_loss_pct')
        default: Default value if not found
        
    Returns:
        Parameter value or default
    """
    return get_config().get_asset_param(symbol, param_path, default)


def get_trading_mode() -> str:
    """Get the current trading mode.
    
    Returns:
        'paper', 'live', or 'backtest'
    """
    return get_config().get('global.mode', 'paper')


def get_initial_capital() -> float:
    """Get the initial capital for paper trading.
    
    Returns:
        Initial capital amount
    """
    return get_config().get('global.initial_capital', 1000.0)


def get_exchange_config(exchange_type: str) -> Dict[str, Any]:
    """Get configuration for a specific exchange.
    
    Args:
        exchange_type: Exchange type ('alpaca', 'coinbase', 'paper')
        
    Returns:
        Exchange configuration dictionary
    """
    project_root = get_config()._find_project_root()
    exchange_file = project_root / "config" / "exchanges" / f"{exchange_type}.yaml"
    
    if exchange_file.exists():
        with open(exchange_file, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


# Convenience functions for common parameters
def get_rsi_params() -> Dict[str, float]:
    """Get RSI parameters from config."""
    return {
        'window': get_config().get('strategy.rsi_window', 14),
        'oversold': get_config().get('strategy.rsi_oversold', 30),
        'overbought': get_config().get('strategy.rsi_overbought', 70),
    }


def get_fgi_thresholds() -> Dict[str, float]:
    """Get FGI thresholds from config."""
    return {
        'fear': get_config().get('strategy.fgi_fear_threshold', 30),
        'greed': get_config().get('strategy.fgi_greed_threshold', 70),
    }


def get_stop_params() -> Dict[str, float]:
    """Get stop loss and take profit parameters."""
    return {
        'stop_loss_pct': get_config().get('strategy.stop_loss_pct', 0.05),
        'take_profit_pct': get_config().get('strategy.take_profit_pct', 0.25),
        'trailing_stop_pct': get_config().get('strategy.trail_pct', 0.0224),
    }


def get_risk_params() -> Dict[str, float]:
    """Get risk management parameters."""
    return {
        'max_daily_loss_pct': get_config().get('risk.max_daily_loss_pct', 0.02),
        'max_drawdown_pct': get_config().get('risk.max_portfolio_drawdown_pct', 0.20),
        'kelly_fraction': get_config().get('risk.kelly_fraction', 0.5),
    }


__all__ = [
    'ConfigLoader',
    'get_config',
    'get_asset_config',
    'get_asset_param',
    'get_trading_mode',
    'get_initial_capital',
    'get_exchange_config',
    'get_rsi_params',
    'get_fgi_thresholds',
    'get_stop_params',
    'get_risk_params',
]