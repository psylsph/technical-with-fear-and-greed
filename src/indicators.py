"""
Technical analysis indicators.
"""

import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI manually."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD manually."""
    fast_ema = close.ewm(span=fast).mean()
    slow_ema = close.ewm(span=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Average Directional Index (ADX) for trend strength measurement.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for ADX calculation (default 14)

    Returns:
        ADX series
    """
    # True Range
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = pd.Series(0.0, index=high.index)
    dm_minus = pd.Series(0.0, index=high.index)

    # Calculate DM+ and DM-
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus_mask = (up_move > down_move) & (up_move > 0)
    dm_minus_mask = (down_move > up_move) & (down_move > 0)

    dm_plus[dm_plus_mask] = up_move[dm_plus_mask]
    dm_minus[dm_minus_mask] = down_move[dm_minus_mask]

    # Smooth TR, DM+ and DM-
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

    # DX and ADX
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
    adx = dx.rolling(window=period).mean()

    return adx
