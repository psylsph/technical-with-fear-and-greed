"""
Sentiment Analysis Module for cryptocurrency trading.

Features:
- Social media sentiment tracking (Twitter/X, Reddit)
- News sentiment analysis
- Fear & Greed Index integration
- Sentiment scoring and aggregation
- Trend detection
- Contrarian indicators
"""

import os
import re
import json
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib


class SentimentSource(Enum):
    """Sources of sentiment data."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FEAR_GREED = "fear_greed"
    AGGREGATED = "aggregated"


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentScore:
    """Individual sentiment score."""
    source: SentimentSource
    score: float  # -1 (extremely bearish) to +1 (extremely bullish)
    confidence: float  # 0 to 1
    timestamp: datetime
    label: SentimentLabel
    volume: int = 0  # Number of mentions/posts
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "score": self.score,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "label": self.label.value,
            "volume": self.volume,
            "metadata": self.metadata,
        }


@dataclass
class SentimentTrend:
    """Sentiment trend over time."""
    direction: str  # "rising", "falling", "stable"
    momentum: float  # Rate of change
    reversal_potential: float  # 0-1, likelihood of sentiment reversal
    divergence_detected: bool  # Price-sentiment divergence


@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis result."""
    overall_score: float  # -1 to +1
    overall_label: SentimentLabel
    confidence: float  # 0 to 1
    individual_scores: List[SentimentScore] = field(default_factory=list)
    trend: Optional[SentimentTrend] = None
    signals: List[str] = field(default_factory=list)
    contrarian_indicator: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.value,
            "confidence": self.confidence,
            "individual_scores": [s.to_dict() for s in self.individual_scores],
            "trend": {
                "direction": self.trend.direction,
                "momentum": self.trend.momentum,
                "reversal_potential": self.trend.reversal_potential,
                "divergence_detected": self.trend.divergence_detected,
            }
            if self.trend
            else None,
            "signals": self.signals,
            "contrarian_indicator": self.contrarian_indicator,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseSentimentAnalyzer(ABC):
    """Base class for sentiment analyzers."""

    def __init__(self, source: SentimentSource):
        self.source = source
        self._history: deque = deque(maxlen=1000)

    @abstractmethod
    async def fetch_sentiment(self, symbol: str, hours: int = 24) -> SentimentScore:
        """Fetch sentiment data."""
        pass

    @abstractmethod
    def _analyze_text(self, text: str) -> Tuple[float, float]:
        """Analyze text sentiment.

        Returns:
            (score, confidence) where score is -1 to +1
        """
        pass

    def add_to_history(self, score: SentimentScore) -> None:
        """Add score to history."""
        self._history.append(score)

    def get_history(self, limit: int = 100) -> List[SentimentScore]:
        """Get historical scores."""
        return list(self._history)[-limit:]


class FearGreedSentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer using Fear & Greed Index."""

    def __init__(self):
        super().__init__(SentimentSource.FEAR_GREED)
        self._api_url = "https://api.alternative.me/fng/"

    async def fetch_sentiment(self, symbol: str, hours: int = 24) -> SentimentScore:
        """Fetch Fear & Greed Index sentiment."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._api_url, params={"limit": 1}) as response:
                    data = await response.json()

                    if "data" not in data or not data["data"]:
                        # Return neutral on error
                        return SentimentScore(
                            source=self.source,
                            score=0.0,
                            confidence=0.0,
                            timestamp=datetime.now(),
                            label=SentimentLabel.NEUTRAL,
                            volume=0,
                        )

                    fgi_data = data["data"][0]
                    fgi_value = int(fgi_data["value"])
                    timestamp = datetime.fromtimestamp(int(fgi_data["timestamp"]))

                    # Convert FGI (0-100) to score (-1 to +1)
                    # 0-25: Extreme Fear (-1 to -0.5)
                    # 25-45: Fear (-0.5 to -0.2)
                    # 45-55: Neutral (-0.2 to 0.2)
                    # 55-75: Greed (0.2 to 0.5)
                    # 75-100: Extreme Greed (0.5 to 1)
                    if fgi_value <= 25:
                        score = -1.0 + (fgi_value / 25)
                        label = SentimentLabel.EXTREME_FEAR
                    elif fgi_value <= 45:
                        score = -0.5 + ((fgi_value - 25) / 20) * 0.3
                        label = SentimentLabel.FEAR
                    elif fgi_value <= 55:
                        score = -0.2 + ((fgi_value - 45) / 10) * 0.4
                        label = SentimentLabel.NEUTRAL
                    elif fgi_value <= 75:
                        score = 0.2 + ((fgi_value - 55) / 20) * 0.3
                        label = SentimentLabel.GREED
                    else:
                        score = 0.5 + ((fgi_value - 75) / 25) * 0.5
                        label = SentimentLabel.EXTREME_GREED

                    sentiment_score = SentimentScore(
                        source=self.source,
                        score=score,
                        confidence=0.8,  # FGI is a reliable aggregate indicator
                        timestamp=timestamp,
                        label=label,
                        volume=int(fgi_value),
                        metadata={"fgi_value": fgi_value, "classification": fgi_data["value_classification"]},
                    )

                    self.add_to_history(sentiment_score)
                    return sentiment_score

        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return SentimentScore(
                source=self.source,
                score=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                label=SentimentLabel.NEUTRAL,
                volume=0,
            )

    def _analyze_text(self, text: str) -> Tuple[float, float]:
        """Not applicable for FGI analyzer."""
        return 0.0, 0.0


class SocialMediaSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Social media sentiment analyzer using keyword analysis.

    Note: This is a simplified implementation. For production use,
    consider using APIs like:
    - Twitter API v2 (Academic Research access)
    - Reddit API
    - Sentiment analysis APIs (Google Cloud NLP, AWS Comprehend, etc.)
    """

    # Bullish keywords
    BULLISH_KEYWORDS = [
        "moon", "pump", "bull", "buy", "hold", "hodl", "accumulate",
        "rocket", "to the moon", "diamond hands", "bull run", "rally",
        "breakout", "surge", "soar", "upward", "gain", "profit", "win",
        "ath", "all time high", "bullish", "long", "calls", "uptrend",
    ]

    # Bearish keywords
    BEARISH_KEYWORDS = [
        "dump", "bear", "sell", "short", "crash", "collapse", "dumping",
        "bearish", "dip", "correction", "plummet", "fall", "drop", "decline",
        "loss", "put", "puts", "downtrend", "bear run", "panic", "fear",
        "scam", "ponzi", "bubble", "burst", "liquidation", "rekt",
    ]

    def __init__(self, source: SentimentSource):
        super().__init__(source)
        self._mock_data = True  # Use mock data by default (requires API keys for real data)

    async def fetch_sentiment(self, symbol: str, hours: int = 24) -> SentimentScore:
        """Fetch social media sentiment."""
        if self._mock_data:
            return self._generate_mock_sentiment(symbol)

        # Real implementation would call APIs here
        return await self._fetch_real_sentiment(symbol, hours)

    def _generate_mock_sentiment(self, symbol: str) -> SentimentScore:
        """Generate mock sentiment for testing."""
        import random

        # Simulate some randomness with slight bullish bias
        base_score = random.gauss(0.1, 0.3)  # Mean 0.1, std 0.3
        score = max(-1.0, min(1.0, base_score))

        # Determine label
        if score <= -0.6:
            label = SentimentLabel.EXTREME_FEAR
        elif score <= -0.2:
            label = SentimentLabel.FEAR
        elif score <= 0.2:
            label = SentimentLabel.NEUTRAL
        elif score <= 0.6:
            label = SentimentLabel.GREED
        else:
            label = SentimentLabel.EXTREME_GREED

        # Mock volume (number of mentions)
        volume = random.randint(100, 10000)

        sentiment_score = SentimentScore(
            source=self.source,
            score=score,
            confidence=0.5,  # Lower confidence for mock data
            timestamp=datetime.now(),
            label=label,
            volume=volume,
            metadata={"mock": True, "symbol": symbol},
        )

        self.add_to_history(sentiment_score)
        return sentiment_score

    async def _fetch_real_sentiment(self, symbol: str, hours: int) -> SentimentScore:
        """Fetch real sentiment from APIs (requires API keys)."""
        # This would integrate with:
        # - Twitter API v2
        # - Reddit API
        # - Scraping (with rate limiting)
        # For now, return mock data
        return self._generate_mock_sentiment(symbol)

    def _analyze_text(self, text: str) -> Tuple[float, float]:
        """Analyze text sentiment using keyword matching."""
        text_lower = text.lower()

        # Count bullish and bearish keywords
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.0  # Neutral, no confidence

        # Calculate score (-1 to +1)
        score = (bullish_count - bearish_count) / total

        # Confidence based on keyword density
        word_count = len(text.split())
        confidence = min(1.0, total / max(1, word_count / 20))

        return score, confidence


class NewsSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    News sentiment analyzer.

    Note: For production use, consider using:
    - NewsAPI (https://newsapi.org)
    - Google Cloud Natural Language API
    - AWS Comprehend
    - Azure Text Analytics
    """

    def __init__(self):
        super().__init__(SentimentSource.NEWS)
        self._mock_data = True

    async def fetch_sentiment(self, symbol: str, hours: int = 24) -> SentimentScore:
        """Fetch news sentiment."""
        if self._mock_data:
            return self._generate_mock_sentiment(symbol)

        return await self._fetch_real_news_sentiment(symbol, hours)

    def _generate_mock_sentiment(self, symbol: str) -> SentimentScore:
        """Generate mock news sentiment."""
        import random

        # News sentiment tends to be less extreme
        base_score = random.gauss(0.0, 0.2)
        score = max(-0.8, min(0.8, base_score))

        # Determine label
        if score <= -0.5:
            label = SentimentLabel.FEAR
        elif score <= 0.2:
            label = SentimentLabel.NEUTRAL
        elif score <= 0.5:
            label = SentimentLabel.GREED
        else:
            label = SentimentLabel.EXTREME_GREED

        # Fewer news articles than social posts
        volume = random.randint(5, 100)

        sentiment_score = SentimentScore(
            source=self.source,
            score=score,
            confidence=0.6,  # News is more reliable than social
            timestamp=datetime.now(),
            label=label,
            volume=volume,
            metadata={"mock": True, "symbol": symbol},
        )

        self.add_to_history(sentiment_score)
        return sentiment_score

    async def _fetch_real_news_sentiment(self, symbol: str, hours: int) -> SentimentScore:
        """Fetch real news sentiment from APIs."""
        # Would integrate with NewsAPI, etc.
        return self._generate_mock_sentiment(symbol)

    def _analyze_text(self, text: str) -> Tuple[float, float]:
        """Analyze news article sentiment."""
        # Simplified keyword-based analysis
        # Real implementation would use NLP models
        text_lower = text.lower()

        positive_words = [
            "surge", "rally", "gain", "rise", "growth", "expansion",
            "breakthrough", "adoption", "partnership", "launch", "upgrade",
        ]

        negative_words = [
            "fall", "drop", "decline", "crash", "loss", "concern",
            "regulation", "ban", "hack", "security", "lawsuit",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0, 0.0

        score = (positive_count - negative_count) / total
        confidence = min(1.0, total / 5)  # More keywords = higher confidence

        return score, confidence


class SentimentAggregator:
    """
    Aggregate sentiment from multiple sources.

    Features:
    - Weighted aggregation of sentiment scores
    - Trend detection
    - Contrarian indicator detection
    - Divergence detection (price vs sentiment)
    """

    def __init__(self):
        self.analyzers: List[BaseSentimentAnalyzer] = []
        self._score_history: deque = deque(maxlen=1000)
        self._setup_analyzers()

    def _setup_analyzers(self) -> None:
        """Setup sentiment analyzers."""
        self.analyzers = [
            FearGreedSentimentAnalyzer(),
            SocialMediaSentimentAnalyzer(SentimentSource.TWITTER),
            SocialMediaSentimentAnalyzer(SentimentSource.REDDIT),
            NewsSentimentAnalyzer(),
        ]

    async def analyze(self, symbol: str, hours: int = 24) -> SentimentAnalysis:
        """
        Perform complete sentiment analysis.

        Args:
            symbol: Trading symbol (e.g., "ETH-USD")
            hours: Number of hours to look back

        Returns:
            Complete sentiment analysis
        """
        # Fetch sentiment from all sources
        individual_scores = []
        for analyzer in self.analyzers:
            try:
                score = await analyzer.fetch_sentiment(symbol, hours)
                individual_scores.append(score)
            except Exception as e:
                print(f"Error fetching sentiment from {analyzer.source}: {e}")

        if not individual_scores:
            # Return neutral if no data
            return SentimentAnalysis(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                individual_scores=[],
            )

        # Calculate weighted aggregate score
        weights = {
            SentimentSource.FEAR_GREED: 0.35,  # FGI is most reliable
            SentimentSource.NEWS: 0.25,
            SentimentSource.TWITTER: 0.20,
            SentimentSource.REDDIT: 0.20,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        total_confidence = 0.0

        for score in individual_scores:
            weight = weights.get(score.source, 0.25)
            weighted_sum += score.score * weight * score.confidence
            total_weight += weight * score.confidence
            total_confidence += score.confidence

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        avg_confidence = total_confidence / len(individual_scores)

        # Determine label
        if overall_score <= -0.6:
            label = SentimentLabel.EXTREME_FEAR
        elif overall_score <= -0.2:
            label = SentimentLabel.FEAR
        elif overall_score <= 0.2:
            label = SentimentLabel.NEUTRAL
        elif overall_score <= 0.6:
            label = SentimentLabel.GREED
        else:
            label = SentimentLabel.EXTREME_GREED

        # Analyze trend
        trend = self._analyze_trend(overall_score)

        # Generate signals
        signals = self._generate_signals(overall_score, label, trend)

        # Check for contrarian indicator
        contrarian = self._is_contrarian_indicator(label, individual_scores)

        analysis = SentimentAnalysis(
            overall_score=overall_score,
            overall_label=label,
            confidence=avg_confidence,
            individual_scores=individual_scores,
            trend=trend,
            signals=signals,
            contrarian_indicator=contrarian,
        )

        # Store in history
        self._score_history.append(analysis)

        return analysis

    def _analyze_trend(self, current_score: float) -> SentimentTrend:
        """Analyze sentiment trend."""
        if len(self._score_history) < 3:
            return SentimentTrend(
                direction="stable",
                momentum=0.0,
                reversal_potential=0.0,
                divergence_detected=False,
            )

        recent_scores = list(self._score_history)[-10:]
        scores = [s.overall_score for s in recent_scores]

        # Calculate momentum
        if len(scores) >= 2:
            momentum = scores[-1] - scores[0]
        else:
            momentum = 0.0

        # Determine direction
        if abs(momentum) < 0.1:
            direction = "stable"
        elif momentum > 0:
            direction = "rising"
        else:
            direction = "falling"

        # Calculate reversal potential (high when extreme sentiment)
        reversal_potential = min(1.0, abs(current_score) * 1.5)

        # Check for divergence (would need price data)
        divergence_detected = False  # Placeholder

        return SentimentTrend(
            direction=direction,
            momentum=momentum,
            reversal_potential=reversal_potential,
            divergence_detected=divergence_detected,
        )

    def _generate_signals(
        self, score: float, label: SentimentLabel, trend: SentimentTrend
    ) -> List[str]:
        """Generate trading signals based on sentiment."""
        signals = []

        # Extreme sentiment signals
        if label == SentimentLabel.EXTREME_FEAR:
            signals.append("Extreme fear - potential buying opportunity")
        elif label == SentimentLabel.EXTREME_GREED:
            signals.append("Extreme greed - potential selling opportunity")

        # Trend signals
        if trend.direction == "rising" and trend.momentum > 0.2:
            signals.append("Sentiment rapidly improving - bullish bias")
        elif trend.direction == "falling" and trend.momentum < -0.2:
            signals.append("Sentiment rapidly deteriorating - bearish bias")

        # Reversal signals
        if trend.reversal_potential > 0.7:
            signals.append(f"High reversal potential ({trend.reversal_potential:.1%})")

        return signals

    def _is_contrarian_indicator(
        self, label: SentimentLabel, scores: List[SentimentScore]
    ) -> bool:
        """Check if sentiment suggests contrarian approach."""
        # Extreme greed -> bearish signal
        # Extreme fear -> bullish signal
        if label in [SentimentLabel.EXTREME_GREED, SentimentLabel.EXTREME_FEAR]:
            # Check if consensus is strong
            consensus_count = sum(
                1 for s in scores if s.label == label and s.confidence > 0.6
            )
            return consensus_count >= 2

        return False

    def get_history(self, limit: int = 100) -> List[SentimentAnalysis]:
        """Get historical sentiment analyses."""
        return list(self._score_history)[-limit:]


# Global sentiment aggregator instance
_sentiment_aggregator: Optional[SentimentAggregator] = None


def get_sentiment_aggregator() -> SentimentAggregator:
    """Get the global sentiment aggregator instance."""
    global _sentiment_aggregator
    if _sentiment_aggregator is None:
        _sentiment_aggregator = SentimentAggregator()
    return _sentiment_aggregator


async def analyze_sentiment(symbol: str, hours: int = 24) -> SentimentAnalysis:
    """
    Analyze sentiment for a trading symbol.

    Args:
        symbol: Trading symbol (e.g., "ETH-USD")
        hours: Number of hours to look back

    Returns:
        Complete sentiment analysis
    """
    aggregator = get_sentiment_aggregator()
    return await aggregator.analyze(symbol, hours)


def get_sentiment_score(symbol: str, hours: int = 24) -> float:
    """
    Get sentiment score (-1 to +1) for a symbol.

    Synchronous convenience function.

    Args:
        symbol: Trading symbol
        hours: Hours to look back

    Returns:
        Sentiment score from -1 (extremely bearish) to +1 (extremely bullish)
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        analysis = loop.run_until_complete(analyze_sentiment(symbol, hours))
        return analysis.overall_score
    except Exception:
        return 0.0  # Return neutral on error


def get_sentiment_label(symbol: str, hours: int = 24) -> SentimentLabel:
    """Get sentiment label for a symbol."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        analysis = loop.run_until_complete(analyze_sentiment(symbol, hours))
        return analysis.overall_label
    except Exception:
        return SentimentLabel.NEUTRAL
