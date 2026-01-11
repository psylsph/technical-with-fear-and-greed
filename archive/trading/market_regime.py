"""
Market Regime Detection: Detect bull/bear/sideways markets with adaptive parameters.
Adjusts trading strategy parameters based on market conditions.
"""

import json
import os
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple

import pandas as pd

from ..config import PROJECT_ROOT


class MarketRegime(Enum):
    """Market regime types."""

    STRONG_BULL = "strong_bull"  # Strong uptrend, high momentum
    BULL = "bull"  # Uptrend
    SIDEWAYS = "sideways"  # Range-bound
    BEAR = "bear"  # Downtrend
    STRONG_BEAR = "strong_bear"  # Strong downtrend, high volatility


class RegimeParameters:
    """Trading parameters adjusted for each market regime."""

    # Default parameters for each regime
    PARAMETERS = {
        MarketRegime.STRONG_BULL: {
            "position_size_multiplier": 1.5,  # Larger positions in strong uptrend
            "stop_loss_multiplier": 1.2,  # Wider stops (let profits run)
            "take_profit_multiplier": 1.5,  # Higher profit targets
            "trail_stop_multiplier": 0.8,  # Tighter trailing stops (lock in profits)
            "entry_threshold": 35,  # Lower fear threshold (easier entry)
            "exit_threshold": 75,  # Higher greed threshold (hold longer)
            "max_drawdown": 0.06,  # 6% max drawdown
            "leverage": 1.2,
        },
        MarketRegime.BULL: {
            "position_size_multiplier": 1.2,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.2,
            "trail_stop_multiplier": 1.0,
            "entry_threshold": 30,
            "exit_threshold": 70,
            "max_drawdown": 0.05,  # 5% max drawdown
            "leverage": 1.0,
        },
        MarketRegime.SIDEWAYS: {
            "position_size_multiplier": 0.8,  # Smaller positions in chop
            "stop_loss_multiplier": 0.8,  # Tighter stops
            "take_profit_multiplier": 0.8,  # Lower profit targets
            "trail_stop_multiplier": 1.2,  # Wider trailing stops
            "entry_threshold": 25,  # More selective entries
            "exit_threshold": 65,  # Earlier exits
            "max_drawdown": 0.04,  # 4% max drawdown
            "leverage": 1.0,
        },
        MarketRegime.BEAR: {
            "position_size_multiplier": 0.6,  # Smaller positions
            "stop_loss_multiplier": 0.7,  # Very tight stops
            "take_profit_multiplier": 0.6,  # Quick profits
            "trail_stop_multiplier": 0.7,  # Tight trailing stops
            "entry_threshold": 20,  # Very selective (extreme fear only)
            "exit_threshold": 60,  # Quick exits
            "max_drawdown": 0.03,  # 3% max drawdown (conservative)
            "leverage": 1.0,
        },
        MarketRegime.STRONG_BEAR: {
            "position_size_multiplier": 0.3,  # Minimal positions
            "stop_loss_multiplier": 0.5,  # Very tight stops
            "take_profit_multiplier": 0.5,  # Quick scalps only
            "trail_stop_multiplier": 0.5,
            "entry_threshold": 15,  # Only extreme panic
            "exit_threshold": 55,  # Very quick exits
            "max_drawdown": 0.02,  # 2% max drawdown (very conservative)
            "leverage": 1.0,
        },
    }


class MarketRegimeDetector:
    """Detect current market regime and provide adaptive parameters."""

    def __init__(
        self,
        lookback_period: int = 50,
        regime_lookback: int = 20,
        state_file: str = None,
    ):
        """
        Args:
            lookback_period: Period for trend analysis (default 50)
            regime_lookback: Period for regime confirmation (default 20)
            state_file: Optional state file for persistence
        """
        self.lookback_period = lookback_period
        self.regime_lookback = regime_lookback

        if state_file is None:
            state_file = os.path.join(PROJECT_ROOT, "market_regime_state.json")

        self.state_file = state_file
        self.state = self._load_state()

        self.current_regime = self.state.get("current_regime")
        self.regime_history = self.state.get("regime_history", [])

    def _load_state(self) -> dict:
        """Load regime state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self):
        """Save regime state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def detect_regime(
        self,
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        volume: pd.Series = None,
    ) -> Tuple[MarketRegime, Dict]:
        """
        Detect current market regime.

        Args:
            close: Close price series
            high: Optional high price series
            low: Optional low price series
            volume: Optional volume series

        Returns:
            Tuple of (regime: MarketRegime, details: dict)
        """
        if len(close) < self.lookback_period:
            return MarketRegime.SIDEWAYS, {
                "reason": "Insufficient data for regime detection",
                "confidence": 0.0,
            }

        # Calculate indicators
        current_price = close.iloc[-1]

        # Trend indicators
        sma_short = close.iloc[-20:].mean() if len(close) >= 20 else current_price
        sma_long = close.iloc[-self.lookback_period :].mean()

        # Momentum
        price_momentum = (
            current_price - close.iloc[-self.lookback_period]
        ) / close.iloc[-self.lookback_period]

        # Volatility (using ATR if high/low available, else using close differences)
        if high is not None and low is not None and len(high) >= 14:
            atr = self._calculate_atr(high, low, close, 14)
            volatility = atr / current_price if current_price > 0 else 0
        else:
            returns = close.pct_change().dropna()
            volatility = returns.iloc[-20:].std() if len(returns) >= 20 else 0

        # ADX for trend strength (if high/low available)
        if high is not None and low is not None:
            try:
                from ..indicators import calculate_adx

                adx = calculate_adx(high, low, close, 14)
                adx_val = adx.iloc[-1] if len(adx) > 0 else 20
            except Exception:
                adx_val = 20
        else:
            adx_val = 20

        # Determine regime
        regime = self._classify_regime(
            price_momentum, sma_short, sma_long, volatility, adx_val, current_price
        )

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            close, regime, volatility, adx_val
        )

        details = {
            "regime": regime.value,
            "price_momentum": float(price_momentum),
            "volatility": float(volatility),
            "adx": float(adx_val),
            "sma_short": float(sma_short),
            "sma_long": float(sma_long),
            "current_price": float(current_price),
            "confidence": float(confidence),
        }

        # Update state
        self.current_regime = regime.value
        self.regime_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "regime": regime.value,
                "confidence": float(confidence),
            }
        )

        # Keep only last 100 regime changes
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

        self.state["current_regime"] = self.current_regime
        self.state["regime_history"] = self.regime_history
        self._save_state()

        return regime, details

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> float:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]

    def _classify_regime(
        self,
        price_momentum: float,
        sma_short: float,
        sma_long: float,
        volatility: float,
        adx: float,
        current_price: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators."""

        # Check if price is above/below moving averages
        above_short = current_price > sma_short
        above_long = current_price > sma_long

        # Strong trend criteria
        strong_trend = adx > 25

        # Classification logic
        if price_momentum > 0.20 and above_long and strong_trend:
            return MarketRegime.STRONG_BULL
        elif price_momentum > 0.10 and above_short and above_long:
            return MarketRegime.BULL
        elif price_momentum < -0.20 and not above_short and strong_trend:
            return MarketRegime.STRONG_BEAR
        elif price_momentum < -0.10 and not above_short:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_regime_confidence(
        self, close: pd.Series, regime: MarketRegime, volatility: float, adx: float
    ) -> float:
        """Calculate confidence in regime detection (0-1)."""
        # Base confidence on ADX (trend strength)
        adx_confidence = min(adx / 50, 1.0)  # ADX > 50 means strong trend

        # Adjust for volatility (high volatility reduces confidence)
        volatility_penalty = min(volatility * 5, 0.3)

        # Adjust for data consistency
        recent_returns = close.pct_change().iloc[-20:].dropna()
        if len(recent_returns) > 0:
            consistency = 1.0 - min(recent_returns.std(), 0.5)
        else:
            consistency = 0.5

        confidence = (adx_confidence * 0.5 + consistency * 0.5) - volatility_penalty
        return max(0.0, min(confidence, 1.0))

    def get_regime_parameters(self, regime: MarketRegime = None) -> Dict:
        """
        Get trading parameters for a specific regime.

        Args:
            regime: Regime to get parameters for (uses current if None)

        Returns:
            Dict of regime-specific parameters
        """
        if regime is None:
            regime_value = self.current_regime
            if regime_value:
                regime = MarketRegime(regime_value)
            else:
                regime = MarketRegime.SIDEWAYS

        return RegimeParameters.PARAMETERS.get(
            regime, RegimeParameters.PARAMETERS[MarketRegime.SIDEWAYS]
        ).copy()

    def adjust_parameters_for_regime(
        self, base_params: Dict, regime: MarketRegime = None
    ) -> Dict:
        """
        Adjust base parameters based on current market regime.

        Args:
            base_params: Base strategy parameters
            regime: Regime to use (uses current if None)

        Returns:
            Adjusted parameters dict
        """
        regime_params = self.get_regime_parameters(regime)

        adjusted = base_params.copy()

        # Apply regime multipliers
        if "position_size" in base_params:
            adjusted["position_size"] = (
                base_params["position_size"] * regime_params["position_size_multiplier"]
            )

        if "stop_loss" in base_params:
            adjusted["stop_loss"] = (
                base_params["stop_loss"] * regime_params["stop_loss_multiplier"]
            )

        if "take_profit" in base_params:
            adjusted["take_profit"] = (
                base_params["take_profit"] * regime_params["take_profit_multiplier"]
            )

        if "trail_stop" in base_params:
            adjusted["trail_stop"] = (
                base_params["trail_stop"] * regime_params["trail_stop_multiplier"]
            )

        # Add regime-specific parameters
        adjusted["fear_entry_threshold"] = regime_params["entry_threshold"]
        adjusted["greed_exit_threshold"] = regime_params["exit_threshold"]
        adjusted["max_drawdown"] = regime_params["max_drawdown"]
        adjusted["regime_leverage"] = regime_params["leverage"]

        adjusted["regime"] = self.current_regime or "sideways"

        return adjusted

    def get_regime_history(self, limit: int = 30) -> list:
        """Get recent regime history."""
        return self.regime_history[-limit:] if self.regime_history else []

    def get_regime_summary(self) -> Dict:
        """Get summary of current and recent regimes."""
        recent_history = self.get_regime_history(30)

        # Count regime occurrences
        regime_counts = {}
        for entry in recent_history:
            regime = entry["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            "current_regime": self.current_regime or "unknown",
            "regime_history_days": len(recent_history),
            "regime_distribution": regime_counts,
            "current_parameters": self.get_regime_parameters(),
        }
