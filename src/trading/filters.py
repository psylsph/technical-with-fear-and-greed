"""
Trading Filters: Trend and Volume confirmation for signals.
"""

import pandas as pd
from typing import Tuple, Optional


class TrendFilter:
    """Filter signals based on trend (moving average)."""

    def __init__(self, trend_period: int = 50):
        """
        Args:
            trend_period: Period for moving average (default 50 days)
        """
        self.trend_period = trend_period

    def check(self, close: pd.Series, signal: str) -> Tuple[bool, Optional[str]]:
        """
        Check if signal should be allowed based on trend.

        Args:
            close: Price series
            signal: Trading signal ('buy', 'sell', etc.)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if len(close) < self.trend_period:
            # Not enough data - allow signal
            return True, None

        # Calculate moving average
        sma = close.rolling(window=self.trend_period).mean()
        current_sma = sma.iloc[-1]
        current_price = close.iloc[-1]

        # Trend filter: Only buy when price is above SMA (uptrend)
        if signal == "buy":
            if current_price < current_sma:
                diff_pct = ((current_price - current_sma) / current_sma) * 100
                return (
                    False,
                    f"Trend filter: Price ${current_price:.2f} below {self.trend_period}-day SMA ${current_sma:.2f} ({diff_pct:.2f}%)",
                )

        elif signal == "sell":
            # Allow sells regardless of trend (take profits)
            pass

        return True, None


class VolumeFilter:
    """Filter signals based on volume confirmation."""

    def __init__(self, volume_period: int = 20, volume_multiplier: float = 1.2):
        """
        Args:
            volume_period: Period for volume average (default 20 days)
            volume_multiplier: Required volume vs average (default 1.2x)
        """
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier

    def check(self, volume: pd.Series, signal: str) -> Tuple[bool, Optional[str]]:
        """
        Check if signal should be allowed based on volume.

        Args:
            volume: Volume series
            signal: Trading signal ('buy', 'sell', etc.)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if len(volume) < self.volume_period:
            # Not enough data - allow signal
            return True, None

        # Calculate average volume
        avg_volume = volume.rolling(window=self.volume_period).mean()
        current_volume = volume.iloc[-1]

        # Only check volume for entry signals (buy/short)
        if signal in ("buy", "short"):
            required_volume = avg_volume.iloc[-1] * self.volume_multiplier
            if current_volume < required_volume:
                vol_ratio = current_volume / avg_volume.iloc[-1]
                return (
                    False,
                    f"Volume filter: Current volume {current_volume:,.0f} below {self.volume_multiplier}x average ({vol_ratio:.2f}x)",
                )

        # Sells can happen regardless of volume (take profits)
        return True, None


class SignalFilters:
    """Combine all signal filters."""

    def __init__(self):
        self.trend_filter = TrendFilter(trend_period=50)
        self.volume_filter = VolumeFilter(volume_period=20, volume_multiplier=1.2)
        self.enabled = {
            "trend": True,
            "volume": True,
        }

    def check_all(
        self, close: pd.Series, volume: pd.Series, signal: str
    ) -> Tuple[bool, list]:
        """
        Check all enabled filters.

        Args:
            close: Price series
            volume: Volume series
            signal: Trading signal

        Returns:
            Tuple of (allowed: bool, list of blocked reasons)
        """
        blocked_reasons = []

        if self.enabled.get("trend", False):
            allowed, reason = self.trend_filter.check(close, signal)
            if not allowed:
                blocked_reasons.append(reason)

        if self.enabled.get("volume", False) and volume is not None and len(volume) > 0:
            allowed, reason = self.volume_filter.check(volume, signal)
            if not allowed:
                blocked_reasons.append(reason)

        return len(blocked_reasons) == 0, blocked_reasons

    def enable_filter(self, filter_name: str):
        """Enable a specific filter."""
        if filter_name in self.enabled:
            self.enabled[filter_name] = True

    def disable_filter(self, filter_name: str):
        """Disable a specific filter."""
        if filter_name in self.enabled:
            self.enabled[filter_name] = False
