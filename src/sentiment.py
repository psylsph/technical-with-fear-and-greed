"""
Sentiment Analysis Module

Provides sentiment indicators for assets that don't have dedicated Fear & Greed Index.
Uses RSI and volatility as proxies for market sentiment.
"""

import pandas as pd

from .indicators import calculate_rsi


def calculate_rsi_sentiment(close: pd.Series, window: int = 14) -> pd.Series:
    """Convert RSI to FGI-equivalent sentiment score.

    RSI interpretation for FGI-equivalent:
    - RSI < 30: Fear (low FGI-like score)
    - RSI 30-45: Neutral Fear
    - RSI 45-55: Neutral
    - RSI 55-70: Neutral Greed
    - RSI > 70: Greed (high FGI-like score)

    This function directly uses RSI as sentiment (not inverted) so that:
    - Low RSI (oversold) = Fear = Low sentiment score (0-30) = BUY signal
    - High RSI (overbought) = Greed = High sentiment score (70-100) = SHORT signal

    Args:
        close: Price series
        window: RSI calculation window

    Returns:
        Series with sentiment score (0-100) matching FGI semantics
    """
    rsi = calculate_rsi(close, window=window)
    # Use RSI directly as sentiment (not inverted)
    # Low RSI = Fear = Low score = Buy signal
    # High RSI = Greed = High score = Short signal
    return rsi.clip(0, 100)


def calculate_volatility_sentiment(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volatility-based sentiment.

    High volatility often indicates fear/uncertainty.
    Low volatility indicates complacency/greed.

    Args:
        close: Price series
        window: Rolling window for volatility calculation

    Returns:
        Series with volatility sentiment (0-100, lower = more fearful)
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=window).std() * (252**0.5)  # Annualized

    # Normalize volatility to 0-100
    vol_min = volatility.rolling(252).min()
    vol_max = volatility.rolling(252).max()

    # Lower volatility = higher sentiment (greed/complacency)
    vol_sentiment = 100 - ((volatility - vol_min) / (vol_max - vol_min) * 100)
    return vol_sentiment.clip(0, 100)


def calculate_composite_sentiment(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate composite sentiment from RSI and volatility.

    Args:
        close: Price series
        window: RSI calculation window

    Returns:
        Series with composite sentiment score (0-100)
    """
    rsi_sentiment = calculate_rsi_sentiment(close, window)
    vol_sentiment = calculate_volatility_sentiment(close, window)

    # Equal weight composite
    composite = (rsi_sentiment + vol_sentiment) / 2
    return composite.clip(0, 100)


def get_sentiment_for_asset(
    close: pd.Series,
    asset_type: str = "btc",
    window: int = 14,
) -> pd.Series:
    """Get appropriate sentiment data for an asset.

    Args:
        close: Price series
        asset_type: 'btc', 'eth', or 'xrp'
        window: RSI calculation window

    Returns:
        Series with sentiment score (0-100)
    """
    if asset_type.lower() == "btc":
        # For BTC, we'd normally use FGI, but for consistency use RSI proxy
        return calculate_rsi_sentiment(close, window)
    else:
        # For ETH/XRP, use RSI sentiment proxy
        return calculate_rsi_sentiment(close, window)


def sentiment_to_fgi_equivalent(sentiment: pd.Series) -> pd.Series:
    """Convert sentiment score to FGI-equivalent (0-100 scale).

    Args:
        sentiment: Sentiment series (0-100)

    Returns:
        FGI-equivalent values
    """
    return sentiment.clip(0, 100)
