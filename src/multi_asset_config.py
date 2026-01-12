"""
Multi-asset configuration system with per-asset tailored parameters.

This module provides asset-specific configurations for trading, training,
and risk management across multiple cryptocurrencies.

Configurations can now be loaded from YAML files via ConfigLoader,
with fallback to hardcoded defaults for backward compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.config_loader import get_asset_config as _get_yaml_asset_config


class AssetCategory(Enum):
    """Categories for different types of cryptocurrencies."""
    BLUE_CHIP = "blue_chip"  # BTC, ETH - high liquidity, lower volatility
    LARGE_CAP = "large_cap"  # Top 10 by market cap
    MID_CAP = "mid_cap"  # Top 11-50 by market cap
    SMALL_CAP = "small_cap"  # Top 51+ by market cap
    MEME = "meme"  # DOGE, SHIB - high volatility, sentiment-driven
    DEFI = "defi"  # UNI, AAVE, COMP - DeFi protocols
    L1 = "layer1"  # SOL, AVAX, DOT - Layer 1 blockchains
    L2 = "layer2"  # ARB, OP, MATIC - Layer 2 solutions


@dataclass
class AssetTrainingParams:
    """Tailored training parameters for each asset."""
    # Data parameters
    lookback_days: int = 365  # How much historical data to use for training
    min_trades_for_training: int = 20  # Minimum trades needed for reliable training
    validation_split: float = 0.2  # Train/validation split ratio
    
    # Feature engineering
    rsi_window: int = 14
    atr_window: int = 14
    sma_windows: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_windows: List[int] = field(default_factory=lambda: [12, 26])
    
    # ML training
    ml_lookback_days: int = 180  # Use only last N days for ML training
    ml_retrain_frequency_days: int = 30  # How often to retrain ML models
    ml_confidence_threshold: float = 0.6  # Minimum confidence for ML signals
    
    # Walk-forward optimization
    walk_forward_train_days: int = 180
    walk_forward_test_days: int = 30
    walk_forward_steps: int = 12


@dataclass
class AssetTradingParams:
    """Tailored trading parameters for each asset."""
    # Entry parameters
    rsi_oversold: float = 30.0  # RSI level for long entry
    rsi_overbought: float = 70.0  # RSI level for short entry
    fgi_fear_threshold: float = 30.0  # FGI level for long entry
    fgi_greed_threshold: float = 70.0  # FGI level for short entry
    
    # Exit parameters
    take_profit_pct: float = 0.25  # 25% profit target for longs
    short_take_profit_pct: float = 0.15  # 15% profit target for shorts
    stop_loss_pct: float = 0.08  # 8% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Position sizing
    max_position_size_pct: float = 0.05  # Max 5% of portfolio per asset
    kelly_fraction: float = 0.5  # Kelly fraction for position sizing
    min_position_value: float = 10.0  # Minimum position value in USD
    
    # Risk management
    max_daily_loss_pct: float = 0.02  # Max 2% daily loss per asset
    max_drawdown_pct: float = 0.15  # Max 15% drawdown
    volatility_multiplier: float = 2.0  # ATR multiplier for stops
    
    # Time-based exits
    max_hold_days: int = 14  # Close position after 14 days if no profit
    min_hold_bars: int = 3  # Minimum bars to hold position


@dataclass
class AssetRiskParams:
    """Tailored risk parameters for each asset."""
    # Volatility adjustments
    volatility_lookback_days: int = 30
    high_volatility_threshold: float = 0.04  # 4% daily volatility = high
    low_volatility_threshold: float = 0.015  # 1.5% daily volatility = low
    
    # Correlation settings
    correlation_lookback_days: int = 90
    high_correlation_threshold: float = 0.7
    low_correlation_threshold: float = 0.3
    
    # Liquidity settings
    min_daily_volume_usd: float = 10000000  # $10M minimum daily volume
    max_slippage_pct: float = 0.005  # Max 0.5% slippage
    spread_threshold_pct: float = 0.001  # Max 0.1% bid-ask spread


@dataclass
class AssetConfig:
    """Complete configuration for a single asset."""
    symbol: str
    name: str
    category: AssetCategory
    base_currency: str = "USD"
    exchange: str = "coinbase"  # Default exchange
    
    # Parameter groups
    training: AssetTrainingParams = field(default_factory=AssetTrainingParams)
    trading: AssetTradingParams = field(default_factory=AssetTradingParams)
    risk: AssetRiskParams = field(default_factory=AssetRiskParams)
    
    # Performance tracking
    historical_sharpe: Optional[float] = None
    historical_win_rate: Optional[float] = None
    historical_max_drawdown: Optional[float] = None
    last_optimization_date: Optional[str] = None
    
    # Active status
    enabled: bool = True
    paper_trading_only: bool = False  # Only trade in paper mode
    live_trading_allowed: bool = True
    
    def get_optimized_params(self) -> Dict:
        """Get optimized parameters for this asset."""
        return {
            "symbol": self.symbol,
            "rsi_window": self.training.rsi_window,
            "trail_pct": self.trading.trailing_stop_pct,
            "buy_quantile": self.trading.fgi_fear_threshold / 100.0,
            "sell_quantile": self.trading.fgi_greed_threshold / 100.0,
            "take_profit_pct": self.trading.take_profit_pct,
            "stop_loss_pct": self.trading.stop_loss_pct,
            "max_position_size_pct": self.trading.max_position_size_pct,
            "max_drawdown_pct": self.trading.max_drawdown_pct,
        }


class AssetRegistry:
    """Registry of all supported assets with their configurations."""
    
    def __init__(self):
        self.assets: Dict[str, AssetConfig] = {}
        self._initialize_default_assets()
    
    def _initialize_default_assets(self):
        """Initialize with default cryptocurrency configurations."""
        
        # Blue Chip Assets (BTC, ETH)
        self.add_asset(AssetConfig(
            symbol="BTC-USD",
            name="Bitcoin",
            category=AssetCategory.BLUE_CHIP,
            training=AssetTrainingParams(
                lookback_days=730,  # 2 years for stable assets
                ml_confidence_threshold=0.65,
                walk_forward_train_days=365,
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.08,  # Higher allocation for blue chips
                take_profit_pct=0.20,  # 4:1 risk-reward ratio (5% stop -> 20% target)
                stop_loss_pct=0.04,  # Tightened from 5% based on Monte Carlo analysis
                trailing_stop_pct=0.025,
                short_take_profit_pct=0.12,  # More conservative for shorts
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.035,
                max_slippage_pct=0.001,  # Very low slippage for high liquidity
            ),
            historical_sharpe=1.8,
            historical_win_rate=0.65,
            historical_max_drawdown=0.25,
        ))
        
        self.add_asset(AssetConfig(
            symbol="ETH-USD",
            name="Ethereum",
            category=AssetCategory.BLUE_CHIP,
            training=AssetTrainingParams(
                lookback_days=730,
                ml_confidence_threshold=0.65,
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.07,
                take_profit_pct=0.20,  # 5:1 risk-reward ratio (4% stop -> 20% target)
                stop_loss_pct=0.04,  # Tightened from 6% based on Monte Carlo analysis
                trailing_stop_pct=0.03,
                short_take_profit_pct=0.12,
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.04,
                max_slippage_pct=0.002,
            ),
            historical_sharpe=2.26,
            historical_win_rate=0.688,
            historical_max_drawdown=0.1842,
        ))
        
        # Large Cap Assets
        self.add_asset(AssetConfig(
            symbol="SOL-USD",
            name="Solana",
            category=AssetCategory.L1,
            training=AssetTrainingParams(
                lookback_days=365,
                ml_confidence_threshold=0.55,  # Lower confidence for more volatile
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.04,
                take_profit_pct=0.24,  # 4:1 risk-reward ratio (6% stop -> 24% target)
                stop_loss_pct=0.06,  # Tightened from 8% based on Monte Carlo analysis
                trailing_stop_pct=0.035,
                short_take_profit_pct=0.15,  # 2.5:1 risk-reward for shorts
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.06,
                max_slippage_pct=0.005,
                min_daily_volume_usd=50000000,
            ),
            paper_trading_only=True,  # Start with paper trading for volatile assets
        ))
        
        self.add_asset(AssetConfig(
            symbol="BNB-USD",
            name="Binance Coin",
            category=AssetCategory.LARGE_CAP,
            training=AssetTrainingParams(
                lookback_days=365,
                ml_retrain_frequency_days=15,  # Retrain more frequently
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.03,
                take_profit_pct=0.21,  # 3.5:1 risk-reward ratio (6% stop -> 21% target)
                stop_loss_pct=0.06,  # Tightened from 7% based on Monte Carlo analysis
                fgi_greed_threshold=75.0,  # Higher threshold for exchange tokens
                short_take_profit_pct=0.14,
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.045,
                max_slippage_pct=0.003,
            ),
        ))
        
        # Meme/High Volatility Assets
        self.add_asset(AssetConfig(
            symbol="DOGE-USD",
            name="Dogecoin",
            category=AssetCategory.MEME,
            training=AssetTrainingParams(
                lookback_days=180,  # Shorter lookback for meme coins
                ml_confidence_threshold=0.45,  # Much lower confidence
                min_trades_for_training=30,  # Need more data for reliability
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.02,  # Very small positions
                take_profit_pct=0.24,  # 3:1 risk-reward ratio (8% stop -> 24% target)
                stop_loss_pct=0.08,  # Tightened from 10% based on Monte Carlo analysis
                trailing_stop_pct=0.04,
                short_take_profit_pct=0.16,
                kelly_fraction=0.3,  # Very conservative position sizing
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.08,
                max_slippage_pct=0.01,  # High slippage possible
                min_daily_volume_usd=20000000,
            ),
            paper_trading_only=True,
            live_trading_allowed=False,  # Disable live trading initially
        ))
        
        # DeFi Assets
        self.add_asset(AssetConfig(
            symbol="UNI-USD",
            name="Uniswap",
            category=AssetCategory.DEFI,
            training=AssetTrainingParams(
                lookback_days=365,
                ml_confidence_threshold=0.6,
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.03,
                take_profit_pct=0.24,  # 4:1 risk-reward ratio (6% stop -> 24% target)
                stop_loss_pct=0.06,  # Tightened from 8% based on Monte Carlo analysis
                fgi_fear_threshold=25.0,  # More aggressive fear entry
                short_take_profit_pct=0.15,
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.05,
                max_slippage_pct=0.004,
            ),
        ))
        
        # XRP (Ripple) - Payment Protocol
        self.add_asset(AssetConfig(
            symbol="XRP-USD",
            name="Ripple",
            category=AssetCategory.LARGE_CAP,
            training=AssetTrainingParams(
                lookback_days=365,
                ml_confidence_threshold=0.55,
                ml_retrain_frequency_days=20,  # Retrain more frequently due to legal/news sensitivity
            ),
            trading=AssetTradingParams(
                max_position_size_pct=0.04,
                take_profit_pct=0.24,  # 4:1 risk-reward ratio (6% stop -> 24% target)
                stop_loss_pct=0.06,  # Tightened from 8% based on Monte Carlo analysis
                fgi_fear_threshold=28.0,
                fgi_greed_threshold=72.0,
                short_take_profit_pct=0.15,  # 2.5:1 risk-reward for shorts
                trailing_stop_pct=0.035,
                kelly_fraction=0.4,  # Conservative due to regulatory risks
            ),
            risk=AssetRiskParams(
                high_volatility_threshold=0.055,
                max_slippage_pct=0.003,
                min_daily_volume_usd=30000000,
            ),
            historical_sharpe=2.39,  # From backtest
            historical_win_rate=0.647,
            historical_max_drawdown=0.1447,
            paper_trading_only=False,  # Enabled for live trading based on exceptional performance
            live_trading_allowed=True,
        ))
    
    def add_asset(self, asset_config: AssetConfig):
        """Add an asset to the registry."""
        self.assets[asset_config.symbol] = asset_config
    
    def get_asset(self, symbol: str) -> Optional[AssetConfig]:
        """Get asset configuration by symbol."""
        return self.assets.get(symbol)
    
    def get_enabled_assets(self) -> List[AssetConfig]:
        """Get all enabled assets."""
        return [asset for asset in self.assets.values() if asset.enabled]
    
    def get_live_trading_assets(self) -> List[AssetConfig]:
        """Get assets enabled for live trading."""
        return [
            asset for asset in self.assets.values() 
            if asset.enabled and asset.live_trading_allowed
        ]
    
    def get_assets_by_category(self, category: AssetCategory) -> List[AssetConfig]:
        """Get assets by category."""
        return [asset for asset in self.assets.values() if asset.category == category]
    
    def update_asset_performance(
        self, 
        symbol: str, 
        sharpe: float, 
        win_rate: float, 
        max_drawdown: float
    ):
        """Update performance metrics for an asset."""
        asset = self.get_asset(symbol)
        if asset:
            asset.historical_sharpe = sharpe
            asset.historical_win_rate = win_rate
            asset.historical_max_drawdown = max_drawdown
            asset.last_optimization_date = datetime.now().strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict:
        """Convert registry to dictionary for serialization."""
        return {
            symbol: {
                "name": asset.name,
                "category": asset.category.value,
                "enabled": asset.enabled,
                "paper_trading_only": asset.paper_trading_only,
                "live_trading_allowed": asset.live_trading_allowed,
                "training": {
                    "lookback_days": asset.training.lookback_days,
                    "ml_confidence_threshold": asset.training.ml_confidence_threshold,
                },
                "trading": {
                    "max_position_size_pct": asset.trading.max_position_size_pct,
                    "take_profit_pct": asset.trading.take_profit_pct,
                    "stop_loss_pct": asset.trading.stop_loss_pct,
                },
                "performance": {
                    "sharpe": asset.historical_sharpe,
                    "win_rate": asset.historical_win_rate,
                    "max_drawdown": asset.historical_max_drawdown,
                }
            }
            for symbol, asset in self.assets.items()
        }


# Global registry instance
asset_registry = AssetRegistry()


def get_asset_config(symbol: str) -> AssetConfig:
    """Get asset configuration for a symbol.

    First checks YAML config files, then falls back to registry.

    Args:
        symbol: Asset symbol (e.g., 'BTC-USD')

    Returns:
        AssetConfig instance
    """
    yaml_config = _get_yaml_asset_config(symbol)

    if yaml_config and yaml_config.get('enabled', True):
        return _create_asset_config_from_yaml(symbol, yaml_config)

    config = asset_registry.get_asset(symbol)
    if not config:
        config = AssetConfig(
            symbol=symbol,
            name=symbol.split("-")[0],
            category=AssetCategory.MID_CAP,
            paper_trading_only=True,
            live_trading_allowed=False,
        )
        asset_registry.add_asset(config)
    return config


def _create_asset_config_from_yaml(symbol: str, yaml_config: Dict) -> AssetConfig:
    """Create AssetConfig from YAML configuration.

    Args:
        symbol: Asset symbol
        yaml_config: YAML configuration dictionary

    Returns:
        AssetConfig instance
    """
    trading = yaml_config.get('trading', {})
    risk = yaml_config.get('risk', {})

    trading_params = AssetTradingParams(
        rsi_oversold=trading.get('rsi_oversold', 30.0),
        rsi_overbought=trading.get('rsi_overbought', 70.0),
        fgi_fear_threshold=trading.get('fgi_fear_threshold', 30.0),
        fgi_greed_threshold=trading.get('fgi_greed_threshold', 70.0),
        take_profit_pct=trading.get('take_profit_pct', 0.25),
        short_take_profit_pct=trading.get('short_take_profit_pct', 0.15),
        stop_loss_pct=trading.get('stop_loss_pct', 0.08),
        trailing_stop_pct=trading.get('trailing_stop_pct', 0.03),
        max_position_size_pct=trading.get('max_position_size_pct', 0.05),
        kelly_fraction=trading.get('kelly_fraction', 0.5),
        min_position_value=trading.get('min_position_value', 10.0),
        max_daily_loss_pct=risk.get('max_daily_loss_pct', 0.02),
        max_drawdown_pct=risk.get('max_drawdown_pct', 0.15),
        volatility_multiplier=risk.get('volatility_multiplier', 2.0),
        max_hold_days=trading.get('max_hold_days', 14),
        min_hold_bars=trading.get('min_hold_bars', 3),
    )

    risk_params = AssetRiskParams(
        volatility_lookback_days=risk.get('volatility_lookback_days', 30),
        high_volatility_threshold=risk.get('high_volatility_threshold', 0.04),
        low_volatility_threshold=risk.get('low_volatility_threshold', 0.015),
        correlation_lookback_days=risk.get('correlation_lookback_days', 90),
        high_correlation_threshold=risk.get('high_correlation_threshold', 0.7),
        low_correlation_threshold=risk.get('low_correlation_threshold', 0.3),
        min_daily_volume_usd=risk.get('min_daily_volume_usd', 10000000),
        max_slippage_pct=risk.get('max_slippage_pct', 0.005),
        spread_threshold_pct=risk.get('spread_threshold_pct', 0.001),
    )

    category_str = yaml_config.get('category', 'mid_cap').lower()
    category_map = {
        'blue_chip': AssetCategory.BLUE_CHIP,
        'large_cap': AssetCategory.LARGE_CAP,
        'mid_cap': AssetCategory.MID_CAP,
        'small_cap': AssetCategory.SMALL_CAP,
        'meme': AssetCategory.MEME,
        'defi': AssetCategory.DEFI,
        'layer1': AssetCategory.L1,
        'l1': AssetCategory.L1,
        'layer2': AssetCategory.L2,
        'l2': AssetCategory.L2,
    }
    category = category_map.get(category_str, AssetCategory.MID_CAP)

    return AssetConfig(
        symbol=symbol,
        name=yaml_config.get('name', symbol.split("-")[0]),
        category=category,
        base_currency=yaml_config.get('base_currency', 'USD'),
        exchange=yaml_config.get('exchange', 'coinbase'),
        trading=trading_params,
        risk=risk_params,
        historical_sharpe=yaml_config.get('performance', {}).get('sharque'),
        historical_win_rate=yaml_config.get('performance', {}).get('win_rate'),
        historical_max_drawdown=yaml_config.get('performance', {}).get('max_drawdown'),
        enabled=yaml_config.get('enabled', True),
        paper_trading_only=yaml_config.get('paper_trading_only', False),
        live_trading_allowed=yaml_config.get('live_trading_allowed', True),
    )


def get_multi_asset_params(assets: List[str]) -> Dict[str, Dict]:
    """Get optimized parameters for multiple assets."""
    return {
        symbol: get_asset_config(symbol).get_optimized_params()
        for symbol in assets
    }


def validate_asset_portfolio(assets: List[str], total_allocation: float = 1.0) -> Tuple[bool, str]:
    """
    Validate a portfolio of assets for risk management.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not assets:
        return False, "No assets specified"
    
    total_position_pct = 0.0
    high_risk_count = 0
    
    for symbol in assets:
        config = get_asset_config(symbol)
        if not config.enabled:
            return False, f"Asset {symbol} is not enabled"
        
        total_position_pct += config.trading.max_position_size_pct
        
        # Count high-risk assets
        if config.category in [AssetCategory.MEME, AssetCategory.SMALL_CAP]:
            high_risk_count += 1
    
    # Check total allocation
    if total_position_pct > total_allocation:
        return False, f"Total allocation {total_position_pct:.1%} exceeds limit {total_allocation:.0%}"
    
    # Limit high-risk assets
    if high_risk_count > 2:
        return False, f"Too many high-risk assets: {high_risk_count} (max 2)"
    
    return True, "Portfolio validation passed"
