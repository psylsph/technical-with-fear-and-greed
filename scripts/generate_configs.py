#!/usr/bin/env python3
"""
Generate YAML configuration files from current hardcoded parameters.

This script creates the new YAML configuration structure based on
current hardcoded parameters in the codebase.

Usage:
    python scripts/generate_configs.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigGenerator:
    """Generate YAML configuration files from current codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.config_dir = self.project_root / "config"
        self.assets_dir = self.config_dir / "assets"
        self.exchanges_dir = self.config_dir / "exchanges"
        
    def create_directory_structure(self):
        """Create the configuration directory structure."""
        directories = [
            self.config_dir,
            self.assets_dir,
            self.exchanges_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory.relative_to(self.project_root)}")
        
        print()
    
    def generate_trading_config(self) -> Dict[str, Any]:
        """Generate global trading configuration."""
        config = {
            "global": {
                "mode": "paper",  # paper, live, backtest
                "initial_capital": 1000.0,
                "max_concurrent_trades": 3,
                "check_interval_seconds": 300,
            },
            "risk": {
                "max_daily_loss_pct": 0.02,
                "max_portfolio_drawdown_pct": 0.20,
                "kelly_fraction": 0.5,
            },
            "strategy": {
                "enable_short_selling": True,
                "enable_regime_filter": True,
                "use_atr_trailing_stop": True,
            },
            "logging": {
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
                "file": "trading.log",
            },
            "performance": {
                "track_performance": True,
                "daily_report": True,
                "weekly_report": True,
            },
        }
        
        return config
    
    def generate_asset_config(self, symbol: str) -> Dict[str, Any]:
        """Generate configuration for a specific asset.
        
        Args:
            symbol: Asset symbol (e.g., "ETH-USD")
            
        Returns:
            Asset configuration dictionary
        """
        # Default configuration template
        config = {
            "symbol": symbol,
            "name": self._get_asset_name(symbol),
            "category": self._get_asset_category(symbol),
            "enabled": True,
            "live_trading_allowed": True,
            "paper_trading_only": False,
            
            "training": {
                "rsi_window": 14,
                "lookback_days": 365,
                "ml_confidence_threshold": 0.6,
                "ml_retrain_frequency_days": 30,
            },
            
            "trading": {
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "fgi_fear_threshold": 30.0,
                "fgi_greed_threshold": 70.0,
                "take_profit_pct": 0.25,
                "stop_loss_pct": 0.08,
                "trailing_stop_pct": 0.03,
                "max_position_size_pct": 0.05,
                "max_drawdown_pct": 0.15,
                "max_daily_loss_pct": 0.02,
                "max_hold_days": 14,
            },
            
            "risk": {
                "high_volatility_threshold": 0.04,
                "low_volatility_threshold": 0.015,
                "max_slippage_pct": 0.005,
                "min_daily_volume_usd": 10000000,
            },
        }
        
        # Apply asset-specific overrides based on current multi_asset_config.py values
        overrides = self._get_asset_overrides(symbol)
        config = self._deep_update(config, overrides)
        
        return config
    
    def _get_asset_name(self, symbol: str) -> str:
        """Get asset name from symbol."""
        names = {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "XRP-USD": "Ripple",
            "SOL-USD": "Solana",
            "BNB-USD": "Binance Coin",
            "DOGE-USD": "Dogecoin",
            "UNI-USD": "Uniswap",
        }
        return names.get(symbol, symbol.replace("-USD", "").title())
    
    def _get_asset_category(self, symbol: str) -> str:
        """Get asset category."""
        categories = {
            "BTC-USD": "blue_chip",
            "ETH-USD": "blue_chip",
            "XRP-USD": "large_cap",
            "SOL-USD": "layer1",
            "BNB-USD": "large_cap",
            "DOGE-USD": "meme",
            "UNI-USD": "defi",
        }
        return categories.get(symbol, "large_cap")
    
    def _get_asset_overrides(self, symbol: str) -> Dict[str, Any]:
        """Get asset-specific overrides based on current multi_asset_config.py values."""
        overrides = {}
        
        # These values are from the current multi_asset_config.py file
        if symbol == "BTC-USD":
            overrides = {
                "training": {
                    "lookback_days": 730,
                    "ml_confidence_threshold": 0.65,
                },
                "trading": {
                    "max_position_size_pct": 0.08,
                    "take_profit_pct": 0.20,
                    "stop_loss_pct": 0.04,
                    "trailing_stop_pct": 0.025,
                    "short_take_profit_pct": 0.12,
                },
                "risk": {
                    "high_volatility_threshold": 0.035,
                    "max_slippage_pct": 0.001,
                },
            }
        elif symbol == "ETH-USD":
            overrides = {
                "training": {
                    "lookback_days": 730,
                    "ml_confidence_threshold": 0.65,
                },
                "trading": {
                    "max_position_size_pct": 0.07,
                    "take_profit_pct": 0.20,
                    "stop_loss_pct": 0.04,
                    "trailing_stop_pct": 0.03,
                    "short_take_profit_pct": 0.12,
                },
                "risk": {
                    "high_volatility_threshold": 0.04,
                    "max_slippage_pct": 0.002,
                },
            }
        elif symbol == "XRP-USD":
            overrides = {
                "trading": {
                    "max_position_size_pct": 0.06,
                    "take_profit_pct": 0.25,
                    "stop_loss_pct": 0.06,
                    "trailing_stop_pct": 0.04,
                },
                "risk": {
                    "high_volatility_threshold": 0.05,
                    "max_slippage_pct": 0.003,
                },
            }
        elif symbol == "SOL-USD":
            overrides = {
                "trading": {
                    "max_position_size_pct": 0.05,
                    "take_profit_pct": 0.30,
                    "stop_loss_pct": 0.06,
                    "trailing_stop_pct": 0.05,
                },
                "risk": {
                    "high_volatility_threshold": 0.06,
                    "max_slippage_pct": 0.004,
                },
            }
        elif symbol == "DOGE-USD":
            overrides = {
                "paper_trading_only": True,
                "trading": {
                    "max_position_size_pct": 0.03,
                    "take_profit_pct": 0.35,
                    "stop_loss_pct": 0.08,
                    "trailing_stop_pct": 0.06,
                },
                "risk": {
                    "high_volatility_threshold": 0.08,
                    "max_slippage_pct": 0.008,
                },
            }
        
        return overrides
    
    def _deep_update(self, original: Dict, update: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original
    
    def generate_exchange_config(self, exchange: str) -> Dict[str, Any]:
        """Generate configuration for an exchange."""
        configs = {
            "alpaca": {
                "name": "Alpaca",
                "type": "broker",
                "paper_trading_supported": True,
                "live_trading_supported": True,
                "rate_limit_requests_per_minute": 200,
                "supported_assets": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
                "symbol_format": "ETHUSD",  # No slash for Alpaca
            },
            "coinbase": {
                "name": "Coinbase",
                "type": "exchange",
                "paper_trading_supported": False,
                "live_trading_supported": True,
                "rate_limit_requests_per_minute": 10,
                "supported_assets": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD", "UNI-USD"],
                "symbol_format": "ETH-USD",  # With dash for Coinbase
            },
            "paper": {
                "name": "Paper Trading",
                "type": "simulation",
                "paper_trading_supported": True,
                "live_trading_supported": False,
                "rate_limit_requests_per_minute": 1000,
                "supported_assets": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD", "UNI-USD"],
                "symbol_format": "ETH-USD",
            },
        }
        
        return configs.get(exchange, {})
    
    def generate_notifications_config(self) -> Dict[str, Any]:
        """Generate notifications configuration."""
        config = {
            "telegram": {
                "enabled": False,
                "bot_token": "${TELEGRAM_BOT_TOKEN}",
                "chat_id": "${TELEGRAM_CHAT_ID}",
                "send_trade_notifications": True,
                "send_signal_notifications": True,
                "send_error_notifications": True,
                "send_daily_summary": True,
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your-email@gmail.com",
                "sender_password": "${EMAIL_PASSWORD}",
                "recipient_email": "your-email@gmail.com",
                "send_daily_reports": True,
                "send_error_alerts": True,
            },
        }
        
        return config
    
    def write_yaml_file(self, filepath: Path, data: Dict[str, Any]):
        """Write data to a YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Generated: {filepath.relative_to(self.project_root)}")
    
    def run_generation(self):
        """Run the configuration generation process."""
        print("=" * 60)
        print("CONFIGURATION GENERATOR")
        print("=" * 60)
        print()
        
        # Create directory structure
        self.create_directory_structure()
        
        # Generate global trading config
        trading_config = self.generate_trading_config()
        self.write_yaml_file(self.config_dir / "trading.yaml", trading_config)
        
        # Generate asset configs
        assets = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "UNI-USD"]
        for asset in assets:
            asset_config = self.generate_asset_config(asset)
            filename = asset.lower().replace("-", "_") + ".yaml"
            self.write_yaml_file(self.assets_dir / filename, asset_config)
        
        # Generate exchange configs
        exchanges = ["alpaca", "coinbase", "paper"]
        for exchange in exchanges:
            exchange_config = self.generate_exchange_config(exchange)
            self.write_yaml_file(self.exchanges_dir / f"{exchange}.yaml", exchange_config)
        
        # Generate notifications config
        notifications_config = self.generate_notifications_config()
        self.write_yaml_file(self.config_dir / "notifications.yaml", notifications_config)
        
        print()
        print("=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Generated {len(assets)} asset configurations")
        print(f"Generated {len(exchanges)} exchange configurations")
        print(f"Total files created: {len(assets) + len(exchanges) + 2}")
        print()
        print("Next steps:")
        print("1. Review the generated configuration files")
        print("2. Update with your specific API keys and settings")
        print("3. Run: python scripts/validate_configs.py")
        print("4. Test with: python scripts/test_config_loading.py")


def main():
    generator = ConfigGenerator()
    generator.run_generation()


if __name__ == "__main__":
    main()