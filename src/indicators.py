"""
Technical analysis indicators.
"""

import pandas as pd


def calculate_support_resistance(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20
) -> dict:
    """Calculate support and resistance levels based on recent price action.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        lookback: Number of periods to look back for levels (default 20)

    Returns:
        Dict with support and resistance levels
    """
    if len(close) < lookback:
        return {
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
        }

    current_price = close.iloc[-1]

    # Find recent swing lows (support) and highs (resistance)
    recent_highs = high.iloc[-lookback:].values
    recent_lows = low.iloc[-lookback:].values

    # Calculate support levels (recent swing lows)
    support_candidates = []
    for i in range(2, len(recent_lows) - 2):
        if (
            recent_lows[i] < recent_lows[i - 1]
            and recent_lows[i] < recent_lows[i - 2]
            and recent_lows[i] < recent_lows[i + 1]
            and recent_lows[i] < recent_lows[i + 2]
        ):
            support_candidates.append(recent_lows[i])

    # Calculate resistance levels (recent swing highs)
    resistance_candidates = []
    for i in range(2, len(recent_highs) - 2):
        if (
            recent_highs[i] > recent_highs[i - 1]
            and recent_highs[i] > recent_highs[i - 2]
            and recent_highs[i] > recent_highs[i + 1]
            and recent_highs[i] > recent_highs[i + 2]
        ):
            resistance_candidates.append(recent_highs[i])

    # Cluster nearby levels (within 1%)
    def cluster_levels(levels, current_price, tolerance=0.01):
        if not levels:
            return []

        levels_sorted = sorted(levels)
        clustered = []

        current_cluster = [levels_sorted[0]]
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] <= tolerance:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered

    support_levels = cluster_levels(support_candidates, current_price)
    resistance_levels = cluster_levels(resistance_candidates, current_price)

    # Find nearest levels (closest to current price)
    nearest_support = None
    nearest_resistance = None

    if support_levels:
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            nearest_support = max(supports_below)

    if resistance_levels:
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            nearest_resistance = min(resistances_above)

    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "current_price": current_price,
    }


def calculate_fibonacci_retracements(
    high: float, low: float, current_price: float
) -> dict:
    """Calculate Fibonacci retracement levels for a price move.

    Args:
        high: Swing high price
        low: Swing low price
        current_price: Current price

    Returns:
        Dict with Fibonacci levels and current position relative to them
    """
    diff = high - low

    # Key Fibonacci retracement levels
    fib_levels = {
        "0% (high)": high,
        "23.6%": high - (diff * 0.236),
        "38.2%": high - (diff * 0.382),
        "50%": high - (diff * 0.5),
        "61.8%": high - (diff * 0.618),
        "78.6%": high - (diff * 0.786),
        "100% (low)": low,
    }

    # Determine which level price is closest to
    closest_level = min(
        fib_levels.items(), key=lambda x: abs(x[1] - current_price)
    )

    # Calculate retracement percentage
    if diff > 0:
        retracement_pct = ((high - current_price) / diff) * 100
    else:
        retracement_pct = 0.0

    return {
        "levels": fib_levels,
        "closest_level": closest_level[0],
        "closest_level_price": closest_level[1],
        "retracement_pct": retracement_pct,
    }


def calculate_pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict:
    """Calculate pivot points (classic support/resistance).

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with pivot point and support/resistance levels
    """
    if len(close) < 1:
        return {}

    # Use previous day's data
    prev_high = high.iloc[-1]
    prev_low = low.iloc[-1]
    prev_close = close.iloc[-1]

    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3

    # Calculate support and resistance levels
    resistance_1 = (2 * pivot) - prev_low
    resistance_2 = pivot + (prev_high - prev_low)
    resistance_3 = prev_high + 2 * (pivot - prev_low)

    support_1 = (2 * pivot) - prev_high
    support_2 = pivot - (prev_high - prev_low)
    support_3 = prev_low - 2 * (prev_high - pivot)

    return {
        "pivot": pivot,
        "r1": resistance_1,
        "r2": resistance_2,
        "r3": resistance_3,
        "s1": support_1,
        "s2": support_2,
        "s3": support_3,
    }


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
