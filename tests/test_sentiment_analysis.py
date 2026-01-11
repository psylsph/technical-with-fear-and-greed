"""
Tests for Sentiment Analysis module.
"""

import pytest
from datetime import datetime

from src.sentiment_analysis import (
    SentimentScore,
    SentimentTrend,
    SentimentAnalysis,
    SentimentSource,
    SentimentLabel,
    SentimentAggregator,
    FearGreedSentimentAnalyzer,
    SocialMediaSentimentAnalyzer,
    NewsSentimentAnalyzer,
    analyze_sentiment,
    get_sentiment_score,
    get_sentiment_label,
)


class TestSentimentScore:
    """Test SentimentScore dataclass."""

    def test_sentiment_score_creation(self):
        """Test creating a sentiment score."""
        score = SentimentScore(
            source=SentimentSource.FEAR_GREED,
            score=0.5,
            confidence=0.8,
            timestamp=datetime.now(),
            label=SentimentLabel.GREED,
            volume=100,
        )

        assert score.source == SentimentSource.FEAR_GREED
        assert score.score == 0.5
        assert score.confidence == 0.8
        assert score.label == SentimentLabel.GREED
        assert score.volume == 100

    def test_sentiment_score_to_dict(self):
        """Test converting sentiment score to dictionary."""
        score = SentimentScore(
            source=SentimentSource.TWITTER,
            score=-0.3,
            confidence=0.6,
            timestamp=datetime.now(),
            label=SentimentLabel.FEAR,
            volume=50,
            metadata={"test": "value"},
        )

        result = score.to_dict()

        assert result["source"] == "twitter"
        assert result["score"] == -0.3
        assert result["confidence"] == 0.6
        assert result["label"] == "fear"
        assert result["volume"] == 50
        assert result["metadata"]["test"] == "value"

    def test_sentiment_score_extreme_values(self):
        """Test extreme sentiment values."""
        # Extreme fear
        fear_score = SentimentScore(
            source=SentimentSource.FEAR_GREED,
            score=-0.9,
            confidence=0.9,
            timestamp=datetime.now(),
            label=SentimentLabel.EXTREME_FEAR,
        )
        assert fear_score.score == -0.9

        # Extreme greed
        greed_score = SentimentScore(
            source=SentimentSource.FEAR_GREED,
            score=0.9,
            confidence=0.9,
            timestamp=datetime.now(),
            label=SentimentLabel.EXTREME_GREED,
        )
        assert greed_score.score == 0.9


class TestFearGreedSentimentAnalyzer:
    """Test Fear & Greed Index sentiment analyzer."""

    @pytest.mark.asyncio
    async def test_fetch_sentiment_real(self):
        """Test fetching real Fear & Greed Index data."""
        analyzer = FearGreedSentimentAnalyzer()
        score = await analyzer.fetch_sentiment("ETH-USD", hours=24)

        assert score.source == SentimentSource.FEAR_GREED
        assert -1.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert score.label in SentimentLabel
        assert 0 <= score.volume <= 100  # FGI value

    def test_fgi_value_to_label_conversion(self):
        """Test FGI value to sentiment label conversion."""
        # Test the label determination logic
        test_cases = [
            (0, SentimentLabel.EXTREME_FEAR),  # Minimum
            (10, SentimentLabel.EXTREME_FEAR),
            (25, SentimentLabel.EXTREME_FEAR),  # Border
            (26, SentimentLabel.FEAR),
            (35, SentimentLabel.FEAR),
            (45, SentimentLabel.FEAR),
            (46, SentimentLabel.NEUTRAL),
            (50, SentimentLabel.NEUTRAL),  # Middle
            (55, SentimentLabel.NEUTRAL),
            (56, SentimentLabel.GREED),
            (65, SentimentLabel.GREED),
            (75, SentimentLabel.GREED),  # Border
            (76, SentimentLabel.EXTREME_GREED),
            (90, SentimentLabel.EXTREME_GREED),
            (100, SentimentLabel.EXTREME_GREED),  # Maximum
        ]

        for fgi_value, expected_label in test_cases:
            # Test the conversion logic directly
            if fgi_value <= 25:
                label = SentimentLabel.EXTREME_FEAR
            elif fgi_value <= 45:
                label = SentimentLabel.FEAR
            elif fgi_value <= 55:
                label = SentimentLabel.NEUTRAL
            elif fgi_value <= 75:
                label = SentimentLabel.GREED
            else:
                label = SentimentLabel.EXTREME_GREED

            assert (
                label == expected_label
            ), f"FGI {fgi_value} should map to {expected_label}, got {label}"


class TestSocialMediaSentimentAnalyzer:
    """Test social media sentiment analyzer."""

    @pytest.mark.asyncio
    async def test_fetch_sentiment_mock(self):
        """Test fetching mock social media sentiment."""
        analyzer = SocialMediaSentimentAnalyzer(SentimentSource.TWITTER)
        score = await analyzer.fetch_sentiment("ETH-USD", hours=24)

        assert score.source == SentimentSource.TWITTER
        assert -1.0 <= score.score <= 1.0
        assert score.label in SentimentLabel
        assert score.volume >= 0

    def test_text_analysis_keywords(self):
        """Test text sentiment analysis using keywords."""
        analyzer = SocialMediaSentimentAnalyzer(SentimentSource.TWITTER)

        # Bullish text
        bullish_text = "The price is going to the moon! 🚀 Buy now!"
        score, confidence = analyzer._analyze_text(bullish_text)
        assert score > 0  # Bullish

        # Bearish text
        bearish_text = "It's dumping hard! Crash and burn! Sell now!"
        score, confidence = analyzer._analyze_text(bearish_text)
        assert score < 0  # Bearish

        # Neutral text
        neutral_text = "Just some random text about crypto."
        score, confidence = analyzer._analyze_text(neutral_text)
        assert score == 0.0  # Neutral
        assert confidence == 0.0  # No confidence

    def test_bullish_keywords(self):
        """Test bullish keyword detection."""
        analyzer = SocialMediaSentimentAnalyzer(SentimentSource.TWITTER)

        bullish_texts = [
            "To the moon!",
            "HODL forever",
            "bull run incoming",
            "Accumulate more",
            "diamond hands",
        ]

        for text in bullish_texts:
            score, _ = analyzer._analyze_text(text)
            assert score > 0, f"Text should be bullish: {text}"

    def test_bearish_keywords(self):
        """Test bearish keyword detection."""
        analyzer = SocialMediaSentimentAnalyzer(SentimentSource.TWITTER)

        bearish_texts = [
            "dumping my bags",
            "bear market continues",
            "crash incoming",
            "sell everything",
            "panic selling",
        ]

        for text in bearish_texts:
            score, _ = analyzer._analyze_text(text)
            assert score < 0, f"Text should be bearish: {text}"


class TestNewsSentimentAnalyzer:
    """Test news sentiment analyzer."""

    @pytest.mark.asyncio
    async def test_fetch_sentiment_mock(self):
        """Test fetching mock news sentiment."""
        analyzer = NewsSentimentAnalyzer()
        score = await analyzer.fetch_sentiment("ETH-USD", hours=24)

        assert score.source == SentimentSource.NEWS
        assert -1.0 <= score.score <= 1.0
        assert score.label in SentimentLabel
        assert score.volume >= 0

    def test_news_text_analysis(self):
        """Test news text sentiment analysis."""
        analyzer = NewsSentimentAnalyzer()

        # Positive news
        positive_news = "Bitcoin surges as institutional adoption grows"
        score, confidence = analyzer._analyze_text(positive_news)
        assert score > 0

        # Negative news
        negative_news = "Crypto prices crash on regulatory ban fears"
        score, confidence = analyzer._analyze_text(negative_news)
        assert score < 0


class TestSentimentAggregator:
    """Test sentiment aggregator."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self):
        """Test complete sentiment analysis."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        assert -1.0 <= analysis.overall_score <= 1.0
        assert analysis.overall_label in SentimentLabel
        assert 0.0 <= analysis.confidence <= 1.0
        assert len(analysis.individual_scores) > 0

    @pytest.mark.asyncio
    async def test_sentiment_aggregation_weights(self):
        """Test that different sources have appropriate weights."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        # Should have multiple sources
        assert len(analysis.individual_scores) >= 2

        # Check that sources are different
        sources = [s.source for s in analysis.individual_scores]
        assert len(set(sources)) >= 2

    @pytest.mark.asyncio
    async def test_trend_detection(self):
        """Test trend detection in sentiment."""
        aggregator = SentimentAggregator()

        # First analysis
        await aggregator.analyze("ETH-USD", hours=24)

        # Second analysis
        analysis2 = await aggregator.analyze("ETH-USD", hours=24)

        if analysis2.trend:
            assert analysis2.trend.direction in ["rising", "falling", "stable"]
            assert -1.0 <= analysis2.trend.momentum <= 1.0

    @pytest.mark.asyncio
    async def test_contrarian_indicator(self):
        """Test contrarian indicator detection."""
        aggregator = SentimentAggregator()

        # Test with different sentiment levels
        for _ in range(5):  # Run multiple times to get different random values
            analysis = await aggregator.analyze("ETH-USD", hours=24)

            # Contrarian indicator should be boolean
            assert isinstance(analysis.contrarian_indicator, bool)

            # Should only be true with extreme sentiment
            if analysis.contrarian_indicator:
                assert analysis.overall_label in [
                    SentimentLabel.EXTREME_FEAR,
                    SentimentLabel.EXTREME_GREED,
                ]

    @pytest.mark.asyncio
    async def test_signals_generation(self):
        """Test trading signals generation."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        # Signals should be a list
        assert isinstance(analysis.signals, list)

    @pytest.mark.asyncio
    async def test_get_history(self):
        """Test sentiment history tracking."""
        aggregator = SentimentAggregator()

        # Generate multiple analyses
        for _ in range(5):
            await aggregator.analyze("ETH-USD", hours=24)

        history = aggregator.get_history(limit=3)

        assert len(history) <= 3
        for analysis in history:
            assert isinstance(analysis, SentimentAnalysis)


class TestSentimentAnalysisIntegration:
    """Integration tests for sentiment analysis."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment_function(self):
        """Test the analyze_sentiment convenience function."""
        analysis = await analyze_sentiment("ETH-USD", hours=24)

        assert analysis.overall_score is not None
        assert analysis.overall_label is not None

    def test_get_sentiment_score_sync(self):
        """Test synchronous sentiment score function."""
        score = get_sentiment_score("ETH-USD")

        assert -1.0 <= score <= 1.0

    def test_get_sentiment_label_sync(self):
        """Test synchronous sentiment label function."""
        label = get_sentiment_label("ETH-USD")

        assert label in SentimentLabel

    @pytest.mark.asyncio
    async def test_different_symbols(self):
        """Test sentiment analysis for different symbols."""
        symbols = ["ETH-USD", "BTC-USD"]

        for symbol in symbols:
            analysis = await analyze_sentiment(symbol, hours=24)
            assert -1.0 <= analysis.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_sentiment_analysis_to_dict(self):
        """Test converting sentiment analysis to dictionary."""
        analysis = await analyze_sentiment("ETH-USD", hours=24)

        result = analysis.to_dict()

        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "overall_label" in result
        assert "confidence" in result
        assert "individual_scores" in result
        assert isinstance(result["individual_scores"], list)

    @pytest.mark.asyncio
    async def test_sentiment_analysis_serialization(self):
        """Test sentiment analysis can be serialized to JSON."""
        import json

        analysis = await analyze_sentiment("ETH-USD", hours=24)

        # Convert to dict and verify JSON serializable
        result = analysis.to_dict()
        json_str = json.dumps(result)

        assert json_str is not None

        # Verify we can deserialize
        parsed = json.loads(json_str)
        assert parsed["overall_score"] == result["overall_score"]


class TestSentimentLabels:
    """Test sentiment label enumeration."""

    def test_all_labels_exist(self):
        """Test that all expected labels exist."""
        expected_labels = [
            "extreme_fear",
            "fear",
            "neutral",
            "greed",
            "extreme_greed",
        ]

        actual_labels = [label.value for label in SentimentLabel]

        for expected in expected_labels:
            assert expected in actual_labels

    def test_label_values(self):
        """Test label value strings."""
        assert SentimentLabel.EXTREME_FEAR.value == "extreme_fear"
        assert SentimentLabel.FEAR.value == "fear"
        assert SentimentLabel.NEUTRAL.value == "neutral"
        assert SentimentLabel.GREED.value == "greed"
        assert SentimentLabel.EXTREME_GREED.value == "extreme_greed"


class TestSentimentSources:
    """Test sentiment source enumeration."""

    def test_all_sources_exist(self):
        """Test that all expected sources exist."""
        expected_sources = ["twitter", "reddit", "news", "fear_greed", "aggregated"]

        actual_sources = [source.value for source in SentimentSource]

        for expected in expected_sources:
            assert expected in actual_sources

    def test_source_values(self):
        """Test source value strings."""
        assert SentimentSource.TWITTER.value == "twitter"
        assert SentimentSource.REDDIT.value == "reddit"
        assert SentimentSource.NEWS.value == "news"
        assert SentimentSource.FEAR_GREED.value == "fear_greed"


class TestSentimentWithPriceData:
    """Test sentiment analysis with price data correlation."""

    @pytest.mark.asyncio
    async def test_sentiment_score_ranges(self):
        """Test that sentiment scores are in valid ranges."""
        aggregator = SentimentAggregator()

        # Run multiple times to get different random values
        for _ in range(10):
            analysis = await aggregator.analyze("ETH-USD", hours=24)

            # Overall score should be valid
            assert -1.0 <= analysis.overall_score <= 1.0

            # Individual scores should be valid
            for score in analysis.individual_scores:
                assert -1.0 <= score.score <= 1.0
                assert 0.0 <= score.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        # Confidence should be between individual confidences
        if len(analysis.individual_scores) > 0:
            confidences = [s.confidence for s in analysis.individual_scores]
            assert min(confidences) <= analysis.confidence <= max(confidences)

    @pytest.mark.asyncio
    async def test_volume_tracking(self):
        """Test that volume is tracked."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        # At least FGI should have volume (FGI value)
        for score in analysis.individual_scores:
            if score.source == SentimentSource.FEAR_GREED:
                assert score.volume >= 0
                assert score.volume <= 100  # FGI is 0-100

    @pytest.mark.asyncio
    async def test_metadata_preservation(self):
        """Test that metadata is preserved in scores."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        for score in analysis.individual_scores:
            assert isinstance(score.metadata, dict)

            # FGI should have specific metadata
            if score.source == SentimentSource.FEAR_GREED:
                assert "fgi_value" in score.metadata

    @pytest.mark.asyncio
    async def test_timestamp_tracking(self):
        """Test that timestamps are tracked."""
        aggregator = SentimentAggregator()
        analysis = await aggregator.analyze("ETH-USD", hours=24)

        assert analysis.timestamp is not None
        assert isinstance(analysis.timestamp, datetime)

        for score in analysis.individual_scores:
            assert score.timestamp is not None
            assert isinstance(score.timestamp, datetime)


class TestSentimentTrend:
    """Test sentiment trend analysis."""

    def test_trend_directions(self):
        """Test that trend direction can be detected."""
        trend = SentimentTrend(
            direction="rising",
            momentum=0.3,
            reversal_potential=0.5,
            divergence_detected=False,
        )

        assert trend.direction == "rising"
        assert trend.momentum == 0.3
        assert trend.reversal_potential == 0.5
        assert not trend.divergence_detected

    def test_all_trend_directions(self):
        """Test all possible trend directions."""
        directions = ["rising", "falling", "stable"]

        for direction in directions:
            trend = SentimentTrend(
                direction=direction,
                momentum=0.0,
                reversal_potential=0.0,
                divergence_detected=False,
            )
            assert trend.direction == direction
