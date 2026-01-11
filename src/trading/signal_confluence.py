"""
Multi-signal Confluence: Require 2+ confirmations before entering trades.
Reduces false positives by requiring multiple signal alignment.
"""

from enum import Enum
from typing import Dict, List

import pandas as pd

from ..indicators import calculate_rsi


class SignalType(Enum):
    """Types of trading signals."""

    FEAR_GREED = "fear_greed"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    TREND_ALIGN = "trend_alignment"
    VOLUME_SPIKE = "volume_spike"
    SUPPORT_TOUCH = "support_touch"
    RESISTANCE_BREAK = "resistance_break"
    MOMENTUM = "momentum"


class SignalConfluence:
    """Require multiple signals to align before trading."""

    def __init__(
        self,
        min_confirmations: int = 2,
        max_signals: int = 5,
        signal_weights: Dict[SignalType, float] = None,
    ):
        """
        Args:
            min_confirmations: Minimum signals required for entry (default 2)
            max_signals: Maximum signals to consider (default 5)
            signal_weights: Optional weights for each signal type
        """
        self.min_confirmations = min_confirmations
        self.max_signals = max_signals

        # Default signal weights (higher = more important)
        self.signal_weights = signal_weights or {
            SignalType.FEAR_GREED: 1.0,  # Fear & Greed Index
            SignalType.RSI_OVERSOLD: 0.8,  # RSI oversold
            SignalType.RSI_OVERBOUGHT: 0.8,  # RSI overbought
            SignalType.TREND_ALIGN: 0.7,  # Price vs trend alignment
            SignalType.VOLUME_SPIKE: 0.5,  # Volume confirmation
            SignalType.SUPPORT_TOUCH: 0.9,  # Price at support
            SignalType.RESISTANCE_BREAK: 0.9,  # Resistance breakout
            SignalType.MOMENTUM: 0.6,  # Momentum signal
        }

        self.signal_history = []

    def check_buy_signals(
        self,
        fgi_value: int,
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        volume: pd.Series = None,
        support_levels: List[float] = None,
    ) -> Dict:
        """
        Check for buy signal confluence.

        Args:
            fgi_value: Fear & Greed Index value
            close: Close price series
            high: Optional high price series
            low: Optional low price series
            volume: Optional volume series
            support_levels: Optional support levels from analysis

        Returns:
            Dict with confluence analysis results
        """
        signals = []
        current_price = close.iloc[-1]

        # Signal 1: Fear & Greed Index (extreme fear)
        if fgi_value <= 25:
            signals.append(
                {
                    "type": SignalType.FEAR_GREED,
                    "name": "Extreme Fear",
                    "strength": 1.0 if fgi_value <= 20 else 0.8,
                    "value": fgi_value,
                }
            )
        elif fgi_value <= 35:
            signals.append(
                {
                    "type": SignalType.FEAR_GREED,
                    "name": "Fear",
                    "strength": 0.6,
                    "value": fgi_value,
                }
            )

        # Signal 2: RSI Oversold
        if len(close) >= 14:
            rsi = calculate_rsi(close, 14)
            current_rsi = rsi.iloc[-1]

            if current_rsi <= 30:
                strength = 1.0 if current_rsi <= 20 else 0.7
                signals.append(
                    {
                        "type": SignalType.RSI_OVERSOLD,
                        "name": f"RSI Oversold ({current_rsi:.1f})",
                        "strength": strength,
                        "value": current_rsi,
                    }
                )
            elif current_rsi <= 40:
                signals.append(
                    {
                        "type": SignalType.RSI_OVERSOLD,
                        "name": f"RSI Low ({current_rsi:.1f})",
                        "strength": 0.4,
                        "value": current_rsi,
                    }
                )

        # Signal 3: Trend Alignment (price below moving averages)
        if len(close) >= 50:
            sma_20 = close.iloc[-20:].mean()
            sma_50 = close.iloc[-50:].mean()

            if current_price < sma_20 and current_price < sma_50:
                # Price below both MAs (oversold territory)
                signals.append(
                    {
                        "type": SignalType.TREND_ALIGN,
                        "name": "Price Below MAs",
                        "strength": 0.8,
                        "value": current_price,
                    }
                )

        # Signal 4: Support Level Touch
        if support_levels and low is not None:
            current_low = low.iloc[-1]
            for level in support_levels:
                if current_low <= level * 1.02:  # Within 2% of support
                    signals.append(
                        {
                            "type": SignalType.SUPPORT_TOUCH,
                            "name": f"Support at ${level:.2f}",
                            "strength": 0.9,
                            "value": level,
                        }
                    )
                    break

        # Signal 5: Volume Spike (increased selling volume)
        if volume is not None and len(volume) >= 20:
            avg_volume = volume.iloc[-20:].mean()
            current_volume = volume.iloc[-1]

            if current_volume > avg_volume * 1.5 and current_price < close.iloc[-2]:
                # High volume on down bar (capitulation)
                signals.append(
                    {
                        "type": SignalType.VOLUME_SPIKE,
                        "name": "High Volume Sell",
                        "strength": 0.7,
                        "value": current_volume / avg_volume,
                    }
                )

        # Signal 6: Momentum (recent bounce)
        if len(close) >= 5:
            recent_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            if -0.05 < recent_change < 0:  # Slight decline but not crashing
                signals.append(
                    {
                        "type": SignalType.MOMENTUM,
                        "name": "Stabilizing",
                        "strength": 0.5,
                        "value": recent_change,
                    }
                )

        # Calculate confluence score
        return self._calculate_confluence_score(signals, "buy")

    def check_sell_signals(
        self,
        fgi_value: int,
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        volume: pd.Series = None,
        resistance_levels: List[float] = None,
    ) -> Dict:
        """
        Check for sell signal confluence.

        Args:
            fgi_value: Fear & Greed Index value
            close: Close price series
            high: Optional high price series
            low: Optional low price series
            volume: Optional volume series
            resistance_levels: Optional resistance levels from analysis

        Returns:
            Dict with confluence analysis results
        """
        signals = []
        current_price = close.iloc[-1]

        # Signal 1: Fear & Greed Index (extreme greed)
        if fgi_value >= 75:
            signals.append(
                {
                    "type": SignalType.FEAR_GREED,
                    "name": "Extreme Greed",
                    "strength": 1.0 if fgi_value >= 80 else 0.8,
                    "value": fgi_value,
                }
            )
        elif fgi_value >= 65:
            signals.append(
                {
                    "type": SignalType.FEAR_GREED,
                    "name": "Greed",
                    "strength": 0.6,
                    "value": fgi_value,
                }
            )

        # Signal 2: RSI Overbought
        if len(close) >= 14:
            rsi = calculate_rsi(close, 14)
            current_rsi = rsi.iloc[-1]

            if current_rsi >= 70:
                strength = 1.0 if current_rsi >= 80 else 0.7
                signals.append(
                    {
                        "type": SignalType.RSI_OVERBOUGHT,
                        "name": f"RSI Overbought ({current_rsi:.1f})",
                        "strength": strength,
                        "value": current_rsi,
                    }
                )
            elif current_rsi >= 60:
                signals.append(
                    {
                        "type": SignalType.RSI_OVERBOUGHT,
                        "name": f"RSI High ({current_rsi:.1f})",
                        "strength": 0.4,
                        "value": current_rsi,
                    }
                )

        # Signal 3: Trend Alignment (price above moving averages)
        if len(close) >= 50:
            sma_20 = close.iloc[-20:].mean()
            sma_50 = close.iloc[-50:].mean()

            if current_price > sma_20 and current_price > sma_50:
                signals.append(
                    {
                        "type": SignalType.TREND_ALIGN,
                        "name": "Price Above MAs",
                        "strength": 0.8,
                        "value": current_price,
                    }
                )

        # Signal 4: Resistance Level Touch
        if resistance_levels and high is not None:
            current_high = high.iloc[-1]
            for level in resistance_levels:
                if current_high >= level * 0.98:  # Within 2% of resistance
                    signals.append(
                        {
                            "type": SignalType.RESISTANCE_BREAK,
                            "name": f"Resistance at ${level:.2f}",
                            "strength": 0.9,
                            "value": level,
                        }
                    )
                    break

        # Calculate confluence score
        return self._calculate_confluence_score(signals, "sell")

    def _calculate_confluence_score(self, signals: List[Dict], direction: str) -> Dict:
        """
        Calculate confluence score from multiple signals.

        Args:
            signals: List of detected signals
            direction: "buy" or "sell"

        Returns:
            Dict with confluence analysis
        """
        # Sort signals by strength
        sorted_signals = sorted(signals, key=lambda x: x["strength"], reverse=True)

        # Take top N signals
        top_signals = sorted_signals[: self.max_signals]

        # Calculate weighted score
        weighted_score = 0.0
        for signal in top_signals:
            weight = self.signal_weights.get(signal["type"], 0.5)
            weighted_score += signal["strength"] * weight

        # Normalize score
        max_possible_score = sum(
            self.signal_weights.get(s["type"], 0.5) for s in top_signals
        )
        normalized_score = (
            weighted_score / max_possible_score if max_possible_score > 0 else 0
        )

        # Determine if entry is allowed
        has_min_confirmations = len(top_signals) >= self.min_confirmations
        is_strong_signal = normalized_score >= 0.6

        should_enter = has_min_confirmations and is_strong_signal

        return {
            "direction": direction,
            "signal_count": len(top_signals),
            "min_confirmations_met": has_min_confirmations,
            "weighted_score": weighted_score,
            "normalized_score": normalized_score,
            "should_enter": should_enter,
            "confidence": "high"
            if normalized_score >= 0.8
            else "medium"
            if normalized_score >= 0.6
            else "low",
            "signals": top_signals,
        }

    def get_signal_summary(self, confluence_result: Dict) -> str:
        """
        Get a human-readable summary of the confluence result.

        Args:
            confluence_result: Result from check_buy_signals or check_sell_signals

        Returns:
            Formatted summary string
        """
        direction = confluence_result["direction"].upper()
        signal_count = confluence_result["signal_count"]
        score = confluence_result["normalized_score"]
        confidence = confluence_result["confidence"].upper()
        should_enter = confluence_result["should_enter"]

        signals_list = confluence_result["signals"]
        signal_names = [s["name"] for s in signals_list]

        summary = f"{direction} Signal Confluence:\n"
        summary += f"  Signals: {signal_count}/{self.min_confirmations} required\n"
        summary += f"  Score: {score:.1%}\n"
        summary += f"  Confidence: {confidence}\n"
        summary += f"  Action: {'✅ ENTER' if should_enter else '❌ WAIT'}\n"
        summary += f"  Signals: {', '.join(signal_names)}"

        return summary
