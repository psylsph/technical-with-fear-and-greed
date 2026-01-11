"""
Configuration Management System with environment-specific configs.

Features:
- Environment-specific configurations (dev/staging/prod)
- YAML-based configuration files
- Environment variable overrides
- Configuration validation
- Secrets management (API keys)
- Runtime configuration updates
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class Environment(Enum):
    """Application environments."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


@dataclass
class TradingConfig:
    """Trading strategy configuration."""

    initial_capital: float = 1000
    maker_fee: float = 0.0015
    taker_fee: float = 0.0025
    default_asset: str = "ETH-USD"

    # Best parameters from optimization
    rsi_window: int = 14
    trail_pct: float = 0.0224
    buy_quantile: float = 0.28
    sell_quantile: float = 0.67
    ml_thresh: float = 0.41
    granularity: str = "ONE_DAY"
    atr_multiplier: float = 3.14
    max_drawdown_pct: float = 0.22

    # Risk controls
    position_size_limit: float = 0.05  # 5% of portfolio
    max_drawdown_stop: float = 0.05  # 5% max drawdown
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    time_exit_days: int = 14  # Exit after 14 days if no profit

    # ML settings
    ml_lookback_days: int = 180
    use_ensemble: bool = True
    use_advanced_ml: bool = False
    ml_performance_threshold: float = 50.0


@dataclass
class APIConfig:
    """API configuration."""

    # Alpaca API
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_base_url: str = (
        "https://paper-api.alpaca.markets"  # Paper trading by default
    )

    # Coinbase API
    coinbase_api_key: Optional[str] = None
    coinbase_api_secret: Optional[str] = None

    # Fear & Greed API
    fgi_api_url: str = "https://api.alternative.me/fng/"

    # Rate limits
    max_requests_per_minute: int = 60
    max_requests_per_second: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration."""

    # SQLite paths
    market_data_db: str = "cache/market_data.db"
    portfolio_state_file: str = "test_portfolio_state.json"

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes default
    cache_max_size_mb: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""

    # Email notifications
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: list = field(default_factory=list)

    # Health checks
    health_check_interval_seconds: int = 60
    api_latency_threshold_ms: int = 1000

    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_threshold: float = 2.0  # Standard deviations


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = "INFO"
    log_file: str = "logs/trading.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # Structured logging
    json_logs: bool = False
    include_timestamp: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""

    environment: Environment = Environment.DEV
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigManager:
    """
    Configuration Manager with environment-specific configs.

    Features:
    - Load from YAML files
    - Environment variable overrides
    - Secrets management
    - Configuration validation
    """

    # Singleton instance
    _instance: Optional["ConfigManager"] = None
    _config: Optional[SystemConfig] = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._load_config()

    @property
    def config(self) -> SystemConfig:
        """Get the current configuration."""
        return self._config

    def _load_config(self) -> SystemConfig:
        """Load configuration from files and environment variables."""
        # Determine environment from environment variable
        env_str = os.getenv("TRADING_ENV", "dev").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            environment = Environment.DEV

        # Load base configuration
        config = SystemConfig(environment=environment)

        # Load from config files if they exist
        config_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "config"
        self._load_from_files(config, config_dir, environment)

        # Override with environment variables
        self._load_from_env(config)

        # Validate configuration
        self._validate_config(config)

        return config

    def _load_from_files(
        self, config: SystemConfig, config_dir: Path, environment: Environment
    ) -> None:
        """Load configuration from YAML files."""
        # Load base config
        base_config_path = config_dir / "base.yaml"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_config = yaml.safe_load(f)
                self._apply_dict_to_config(config, base_config)

        # Load environment-specific config
        env_config_path = config_dir / f"{environment.value}.yaml"
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_config = yaml.safe_load(f)
                self._apply_dict_to_config(config, env_config)

    def _load_from_env(self, config: SystemConfig) -> None:
        """Override configuration with environment variables."""
        # API keys
        config.api.alpaca_api_key = os.getenv(
            "ALPACA_API_KEY", config.api.alpaca_api_key
        )
        config.api.alpaca_api_secret = os.getenv(
            "ALPACA_API_SECRET", config.api.alpaca_api_secret
        )
        config.api.coinbase_api_key = os.getenv(
            "COINBASE_API_KEY", config.api.coinbase_api_key
        )
        config.api.coinbase_api_secret = os.getenv(
            "COINBASE_API_SECRET", config.api.coinbase_api_secret
        )

        # Email settings
        config.monitoring.smtp_username = os.getenv(
            "SMTP_USERNAME", config.monitoring.smtp_username
        )
        config.monitoring.smtp_password = os.getenv(
            "SMTP_PASSWORD", config.monitoring.smtp_password
        )
        config.monitoring.email_from = os.getenv(
            "EMAIL_FROM", config.monitoring.email_from
        )

        if email_to := os.getenv("EMAIL_TO"):
            config.monitoring.email_to = email_to.split(",")

        # API settings
        if api_base := os.getenv("ALPACA_BASE_URL"):
            config.api.alpaca_base_url = api_base

        # Trading settings
        if initial_capital := os.getenv("INITIAL_CAPITAL"):
            config.trading.initial_capital = float(initial_capital)

        if default_asset := os.getenv("DEFAULT_ASSET"):
            config.trading.default_asset = default_asset

        # Monitoring settings
        if email_enabled := os.getenv("EMAIL_ENABLED"):
            config.monitoring.email_enabled = email_enabled.lower() == "true"

    def _apply_dict_to_config(
        self, config: SystemConfig, config_dict: Dict[str, Any]
    ) -> None:
        """Apply a dictionary to the configuration object."""
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def _validate_config(self, config: SystemConfig) -> None:
        """Validate configuration values."""
        # Validate trading config
        assert config.trading.initial_capital > 0, "initial_capital must be positive"
        assert (
            0 <= config.trading.maker_fee <= 0.01
        ), "maker_fee must be between 0 and 1%"
        assert (
            0 <= config.trading.taker_fee <= 0.01
        ), "taker_fee must be between 0 and 1%"
        assert (
            0 < config.trading.position_size_limit <= 1
        ), "position_size_limit must be between 0 and 100%"
        assert (
            0 < config.trading.max_drawdown_stop <= 1
        ), "max_drawdown_stop must be between 0 and 100%"

        # Validate API config
        assert (
            config.api.max_requests_per_minute > 0
        ), "max_requests_per_minute must be positive"
        assert (
            config.api.max_requests_per_second > 0
        ), "max_requests_per_second must be positive"

        # Validate monitoring config
        if config.monitoring.email_enabled:
            assert (
                config.monitoring.smtp_username is not None
            ), "smtp_username required when email_enabled"
            assert (
                config.monitoring.smtp_password is not None
            ), "smtp_password required when email_enabled"
            assert (
                config.monitoring.email_from is not None
            ), "email_from required when email_enabled"
            assert (
                len(config.monitoring.email_to) > 0
            ), "email_to required when email_enabled"

    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self._config.trading

    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return self._config.api

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self._config.database

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self._config.monitoring

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self._config.logging

    def get_environment(self) -> Environment:
        """Get current environment."""
        return self._config.environment

    def is_prod(self) -> bool:
        """Check if running in production."""
        return self._config.environment == Environment.PROD

    def is_dev(self) -> bool:
        """Check if running in development."""
        return self._config.environment == Environment.DEV

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value at runtime."""
        section_obj = getattr(self._config, section)
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)
        else:
            raise ValueError(f"Invalid config key: {section}.{key}")

    def export_config(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding secrets)."""
        config_dict = {
            "environment": self._config.environment.value,
            "trading": {},
            "api": {},
            "database": {},
            "monitoring": {},
            "logging": {},
        }

        # Export each section (excluding secrets)
        for section_name, section_obj in [
            ("trading", self._config.trading),
            ("database", self._config.database),
            ("logging", self._config.logging),
        ]:
            for key, value in section_obj.__dict__.items():
                if not key.startswith("_") and value is not None:
                    config_dict[section_name][key] = value

        # API config (exclude secrets)
        for key, value in self._config.api.__dict__.items():
            if (
                not key.startswith("_")
                and value is not None
                and "key" not in key.lower()
                and "secret" not in key.lower()
            ):
                config_dict["api"][key] = value

        # Monitoring config (exclude secrets)
        for key, value in self._config.monitoring.__dict__.items():
            if (
                not key.startswith("_")
                and value is not None
                and "password" not in key.lower()
            ):
                config_dict["monitoring"][key] = value

        return config_dict

    def save_config_template(self, output_path: str = "config_template.yaml") -> None:
        """Save a configuration template file."""
        template = {
            "environment": "dev",
            "trading": {
                "initial_capital": 1000,
                "maker_fee": 0.0015,
                "taker_fee": 0.0025,
                "default_asset": "ETH-USD",
                "position_size_limit": 0.05,
                "max_drawdown_stop": 0.05,
                "use_advanced_ml": False,
            },
            "api": {
                "alpaca_base_url": "https://paper-api.alpaca.markets",
                "max_requests_per_minute": 60,
                "max_requests_per_second": 5,
            },
            "monitoring": {
                "email_enabled": False,
                "health_check_interval_seconds": 60,
                "anomaly_detection_enabled": True,
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "logs/trading.log",
            },
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(template, f, default_flow_style=False)

        print(f"Configuration template saved to {output_path}")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> SystemConfig:
    """Get the current system configuration."""
    return get_config_manager().config


def get_trading_config() -> TradingConfig:
    """Get trading configuration."""
    return get_config_manager().get_trading_config()


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config_manager().get_api_config()


def is_prod() -> bool:
    """Check if running in production."""
    return get_config_manager().is_prod()


def is_dev() -> bool:
    """Check if running in development."""
    return get_config_manager().is_dev()
