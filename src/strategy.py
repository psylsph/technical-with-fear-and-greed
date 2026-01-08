"""
Trading strategy implementation.
"""

import warnings

import pandas as pd
import vectorbt as vbt

from .config import INITIAL_CAPITAL, MAKER_FEE, TAKER_FEE
from .indicators import calculate_adx, calculate_rsi
from .data.data_fetchers import fetch_crypto_news_sentiment, fetch_options_flow

# Suppress vectorbt aggregation warning (internal to vectorbt, doesn't affect functionality)
warnings.filterwarnings(
    "ignore", message="Object has multiple columns.*Aggregating using.*mean"
)


def detect_market_regime(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Detect market regime using multiple indicators.

    Args:
        close: Price series
        lookback: Lookback period for regime detection

    Returns:
        Series with regime labels: 'strong_bull', 'bull', 'sideways', 'bear', 'strong_bear'
    """
    regime = pd.Series("unknown", index=close.index)

    for i in range(lookback, len(close)):
        price = close.iloc[i]
        lookback_prices = close.iloc[i - lookback : i]

        # Calculate returns and trends
        sma_short = close.iloc[i - 20 : i].mean() if i >= 20 else close.iloc[:i].mean()
        sma_long = lookback_prices.mean()

        # Trend direction
        price_momentum = (price - lookback_prices.iloc[0]) / lookback_prices.iloc[0]

        # ADX for trend strength
        try:
            high = close.iloc[: i + 1]
            low = close.iloc[: i + 1]
            adx = calculate_adx(high, low, close.iloc[: i + 1], period=14)
            adx_val = adx.iloc[-1] if len(adx) > 0 else 20
        except Exception:
            adx_val = 20

        # Classify regime
        if price_momentum > 0.30 and sma_short > sma_long and adx_val > 25:
            regime.iloc[i] = "strong_bull"
        elif price_momentum > 0.10 and sma_short > sma_long:
            regime.iloc[i] = "bull"
        elif price_momentum < -0.30 and sma_short < sma_long and adx_val > 25:
            regime.iloc[i] = "strong_bear"
        elif price_momentum < -0.10 and sma_short < sma_long:
            regime.iloc[i] = "bear"
        else:
            regime.iloc[i] = "sideways"

    return regime


def generate_signal(
    close: pd.Series,
    fgi_df: pd.DataFrame,
    rsi_window: int = 14,
    trail_pct: float = 0.10,
    buy_quantile: float = 0.2,
    sell_quantile: float = 0.8,
    ml_thresh: float = 0.5,
    pred_series: pd.Series = None,
    higher_tf_data: dict = None,
    enable_multi_tf: bool = True,
    # Risk-focused parameters for 2026
    fear_entry_threshold: int = 30,
    greed_exit_threshold: int = 70,
    max_drawdown_exit: float = 0.08,
    _volatility_stop_multiplier: float = 1.5,
    _volatility_lookback: int = 20,
    # New parameters for trend-following
    high: pd.Series = None,
    low: pd.Series = None,
    adx_threshold: float = 20.0,
    # Short selling parameters
    enable_short_selling: bool = False,
    # News sentiment enhancement
    enable_news_sentiment: bool = False,
    # Options flow enhancement
    enable_options_flow: bool = False,
) -> dict:
    """Generate trading signal using risk-focused strategy for 2026 market conditions.

    Returns dict with:
        - signal: 'buy', 'sell', 'short', 'cover', or 'hold'
        - in_position: bool, whether we should be in a position
        - position_type: 'long', 'short', or None
        - indicators: dict with fgi, rsi, pred, etc.
    """
    latest_close = close.iloc[-1]
    latest_dt = close.index[-1]

    dt_ts = pd.Timestamp(latest_dt)
    dt_date_only = dt_ts.normalize()

    # Use most recent available FGI data if today's data isn't available
    if dt_date_only not in fgi_df.index:
        if len(fgi_df) > 0:
            # Use the most recent FGI value
            dt_date_only = fgi_df.index[-1]
            fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]
        else:
            return {
                "signal": "hold",
                "in_position": False,
                "error": "FGI data not available",
            }
    else:
        fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]

    # Enhanced FGI with multiple data sources (if enabled)
    enhanced_fgi_val = fgi_val
    news_sentiment_data = None
    options_flow_data = None

    if enable_news_sentiment:
        try:
            news_sentiment_data = fetch_crypto_news_sentiment("BTC", hours_back=24)
            if news_sentiment_data and "sentiment_score" in news_sentiment_data:
                # Adjust FGI by sentiment (small adjustment to avoid over-influence)
                sentiment_adjustment = (
                    news_sentiment_data["sentiment_score"] * 5
                )  # Scale sentiment impact
                enhanced_fgi_val = min(100, max(0, fgi_val + sentiment_adjustment))
        except Exception as e:
            # Fallback to raw FGI if news sentiment fails
            print(f"News sentiment integration failed: {e}")

    if enable_options_flow:
        try:
            options_flow_data = fetch_options_flow("BTC", days_back=7)
            if options_flow_data and "fear_gauge" in options_flow_data:
                # Adjust FGI based on options fear gauge
                fear_adjustment = (
                    options_flow_data["fear_gauge"] - 0.5
                ) * 10  # Scale around neutral
                enhanced_fgi_val = min(100, max(0, enhanced_fgi_val + fear_adjustment))
        except Exception as e:
            # Fallback if options flow fails
            print(f"Options flow integration failed: {e}")

    # Use enhanced FGI for all calculations
    fgi_val = enhanced_fgi_val

    # Market regime detection (using FGI trends over last 30 days)
    fgi_30d_avg = (
        fgi_df["fgi_value"].rolling(30).mean().loc[dt_date_only]
        if len(fgi_df) >= 30
        else fgi_val
    )
    fgi_trend = (
        "bull"
        if fgi_val > fgi_30d_avg + 5
        else "bear"
        if fgi_val < fgi_30d_avg - 5
        else "sideways"
    )

    # Adjust thresholds based on market regime
    if fgi_trend == "bull":
        # In bull markets, be more conservative (wait for extreme fear)
        effective_fear_threshold = fear_entry_threshold - 5  # 25
        effective_greed_threshold = greed_exit_threshold + 5  # 75
    elif fgi_trend == "bear":
        # In bear markets, be more aggressive (capital preservation)
        effective_fear_threshold = fear_entry_threshold + 5  # 35
        effective_greed_threshold = greed_exit_threshold - 5  # 65
    else:
        # Sideways markets use default thresholds
        effective_fear_threshold = fear_entry_threshold
        effective_greed_threshold = greed_exit_threshold

    # Enhanced risk-focused strategy: Multiple entry conditions
    # Primary: Extreme fear (bear market capitulation)
    fear_signal = fgi_val <= effective_fear_threshold

    # Secondary: Momentum divergence (price weakness in uptrends)
    momentum_signal = False
    if len(close) >= 50:  # Need sufficient data for momentum
        # Check if price is breaking below recent lows while FGI is not extremely fearful
        recent_low = close.iloc[-50:].min()
        price_weakness = latest_close < recent_low * 1.02  # Within 2% of 50-day low
        fgi_not_extreme = fgi_val > effective_fear_threshold + 10  # FGI not too low
        momentum_signal = price_weakness and fgi_not_extreme and fgi_trend == "bull"

    # Tertiary: Trend-following (ADX strong trend + price momentum in bull markets)
    trend_following_signal = False
    if (
        high is not None
        and low is not None
        and len(close) >= 50
        and fgi_trend == "bull"
    ):
        try:
            # Calculate ADX for trend strength
            adx = calculate_adx(high, low, close, period=14)
            if len(adx) >= 14:
                current_adx = adx.iloc[-1]
                # Strong trend if ADX > threshold (typically 25-30)
                strong_trend = current_adx > adx_threshold

                # Price momentum: Recent uptrend
                recent_high = close.iloc[-20:].max()
                recent_low = close.iloc[-20:].min()
                price_momentum = (latest_close - recent_low) / (
                    recent_high - recent_low
                ) > 0.5  # In upper 50% of range

                trend_following_signal = strong_trend and price_momentum
        except Exception:
            # Fallback if ADX calculation fails
            trend_following_signal = False

    # Combined long entry: Extreme fear OR momentum divergence OR trend-following
    is_buy = fear_signal or momentum_signal or trend_following_signal

    # Enhanced long exit conditions
    # Primary: Extreme greed (bull market euphoria)
    greed_signal = fgi_val >= effective_greed_threshold

    # Secondary: Momentum reversal (price strength in downtrends)
    reversal_signal = False
    if len(close) >= 20:
        recent_high = close.iloc[-20:].max()
        price_strength = latest_close > recent_high * 0.98  # Within 2% of 20-day high
        reversal_signal = price_strength and fgi_trend == "bear"

    # Combined long exit: Greed OR momentum reversal OR drawdown
    is_sell = greed_signal or reversal_signal

    # Short selling signals (if enabled)
    is_short = False
    is_cover = False
    if enable_short_selling:
        # Short entry: Extreme greed (sell when euphoria peaks)
        short_signal = fgi_val >= effective_greed_threshold

        # Short exit (cover): Extreme fear (buy back when panic sets in)
        cover_signal = fgi_val <= effective_fear_threshold

        # Additional short signals for bull markets
        short_momentum_signal = False
        if len(close) >= 20 and fgi_trend == "bull":
            recent_high = close.iloc[-20:].max()
            price_weakness_short = (
                latest_close > recent_high * 0.98
            )  # Near recent highs
            short_momentum_signal = (
                price_weakness_short and fgi_val >= effective_greed_threshold - 10
            )

        is_short = short_signal or short_momentum_signal
        is_cover = cover_signal

    # Calculate _volatility for position sizing and stops
    if len(close) >= _volatility_lookback:
        recent_prices = close.iloc[-_volatility_lookback:]
        _volatility = recent_prices.std()
        _volatility_stop = latest_close - (_volatility * _volatility_stop_multiplier)
    else:
        _volatility = latest_close * 0.02  # Default 2% _volatility assumption
        _volatility_stop = latest_close * (1 - max_drawdown_exit)

    # Dynamic position sizing based on market regime and _volatility
    if fgi_trend == "bull":
        base_position_size = 0.05  # 5% of portfolio in bull markets
    elif fgi_trend == "bear":
        base_position_size = 0.03  # 3% of portfolio in bear markets (more conservative)
    else:
        base_position_size = 0.04  # 4% of portfolio in sideways markets

    # Adjust for _volatility (smaller positions in volatile markets)
    _volatility_adjustment = (
        min(1.0, 0.02 / (_volatility / latest_close)) if _volatility > 0 else 1.0
    )
    position_size = base_position_size * _volatility_adjustment

    # Calculate ADX for indicators if available
    adx_value = None
    if high is not None and low is not None and len(close) >= 50:
        try:
            adx_series = calculate_adx(high, low, close, period=14)
            if len(adx_series) >= 14:
                adx_value = adx_series.iloc[-1]
        except Exception:
            adx_value = None

    # Determine signal based on current position context and available signals
    signal = "hold"
    position_type = None

    # Priority for signals when no position
    if is_buy:
        signal = "buy"
        position_type = "long"
    elif is_short and enable_short_selling:
        signal = "short"
        position_type = "short"
    # For existing positions, the signal would be determined by position tracking logic
    # For live signals, we assume no position and suggest entry signals
    elif is_sell:
        signal = "sell"
        position_type = "long"  # Selling from long position
    elif is_cover and enable_short_selling:
        signal = "cover"
        position_type = "short"  # Covering short position

    return {
        "signal": signal,
        "in_position": False,  # Live trading doesn't track position state
        "position_type": position_type,
        "position_size": position_size,  # Dynamic position sizing
        "indicators": {
            "fgi": fgi_val,
            "fgi_trend": fgi_trend,
            "effective_fear_threshold": effective_fear_threshold,
            "effective_greed_threshold": effective_greed_threshold,
            "_volatility_stop": _volatility_stop,
            "max_drawdown_exit": max_drawdown_exit,
            "position_size_pct": position_size * 100,
            "price": latest_close,
            "is_extreme_fear": fgi_val <= effective_fear_threshold,
            "is_extreme_greed": fgi_val >= effective_greed_threshold,
            "adx": adx_value,
            "adx_threshold": adx_threshold,
            "trend_following_signal": trend_following_signal,
            "short_selling_enabled": enable_short_selling,
            "is_short_signal": is_short,
            "is_cover_signal": is_cover,
            "news_sentiment_enabled": enable_news_sentiment,
            "news_sentiment_data": news_sentiment_data,
            "options_flow_enabled": enable_options_flow,
            "options_flow_data": options_flow_data,
        },
    }


def run_strategy(
    close: pd.Series,
    freq: str,
    fgi_df: pd.DataFrame,
    granularity_name: str,
    rsi_window: int = 14,
    trail_pct: float = 0.10,
    buy_quantile: float = 0.2,
    sell_quantile: float = 0.8,
    ml_thresh: float = 0.5,
    pred_series: pd.Series = None,
    higher_tf_data: dict = None,
    enable_multi_tf: bool = False,
    use_atr_trail: bool = True,
    atr_multiplier: float = 2.0,
    max_drawdown_pct: float = 0.15,
    enable_regime_filter: bool = True,
    regime_filter_lookback: int = 50,
) -> dict:
    """Run fear & greed strategy with RSI filter, dynamic FGI thresholds, trailing stops, multi-TF analysis, and regime filtering, return performance metrics.

    Args:
        close: Price series at current timeframe
        freq: Frequency string (e.g., '1d', '1h', '15min')
        fgi_df: Fear & Greed Index data (daily)
        granularity_name: Granularity name for reporting
        rsi_window: RSI calculation window
        trail_pct: Trailing stop percentage (used if use_atr_trail=False)
        buy_quantile: FGI buy quantile threshold
        sell_quantile: FGI sell quantile threshold
        ml_thresh: ML prediction threshold
        pred_series: ML prediction series
        higher_tf_data: Dictionary with higher timeframe indicators
        enable_multi_tf: Enable multi-timeframe filtering
        use_atr_trail: Use ATR-based dynamic trailing stop
        atr_multiplier: ATR multiplier for trailing stop distance
        max_drawdown_pct: Maximum drawdown before forcing exit
        enable_regime_filter: Enable market regime filtering
        regime_filter_lookback: Lookback period for regime detection

    Returns:
        Dictionary with performance metrics
    """
    from .indicators import calculate_macd

    # Ensure close is a Series with a name for proper column handling in vectorbt
    if isinstance(close, pd.Series):
        if not close.name:
            close = close.copy()
            close.name = "close"
    else:
        # If DataFrame, extract the 'close' column or first column
        if isinstance(close, pd.DataFrame):
            if "close" in close.columns:
                close = close["close"]
            else:
                close = close.iloc[:, 0]

    # Calculate market regime for each point
    regime = detect_market_regime(close, lookback=regime_filter_lookback)

    # Create entries and exits as DataFrames with matching column name
    entries = pd.DataFrame(False, index=close.index, columns=[close.name])
    exits = pd.DataFrame(False, index=close.index, columns=[close.name])

    # Calculate indicators
    rsi = calculate_rsi(close, window=rsi_window)
    macd, signal = calculate_macd(close)

    # Dynamic FGI thresholds
    buy_thresh = fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(buy_quantile)
    sell_thresh = fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(sell_quantile)

    # Multi-TF: For sub-daily, check daily RSI (removed, using ML instead)
    in_position = False
    position_type = None  # 'long' or 'short'
    position_price = 0.0
    trailing_stop = 0.0
    take_profit_pct = 0.25

    # ATR for dynamic trailing stop
    atr = None
    if use_atr_trail:
        # Calculate ATR (Average True Range)
        high_low = pd.concat([close], axis=1)
        high = high_low.max(axis=1)
        low = high_low.min(axis=1)
        prev_close = close.shift(1)

        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(window=14).mean()

    # Max drawdown tracking
    max_drawdown_reached = False

    for i in range(len(close)):
        price = close.iloc[i]
        dt = close.index[i]

        dt_ts = pd.Timestamp(dt)
        dt_date_only = dt_ts.normalize()

        # Get current market regime
        current_regime = regime.iloc[i] if i < len(regime) else "unknown"

        # Apply regime filter - only trade in favorable regimes
        # In strong bull markets (uptrend), only buy on dips
        # In bear markets, only buy on extreme fear
        regime_allowed = True
        if enable_regime_filter:
            if current_regime == "strong_bull":
                # In strong bull, only buy on extreme fear (FGI <= 20)
                fgi_val = (
                    fgi_df.loc[dt_date_only, "fgi_value"]
                    if dt_date_only in fgi_df.index
                    else 50
                )
                if fgi_val > 25:
                    regime_allowed = False
            elif current_regime == "bear":
                # In bear market, more conservative - only buy on extreme fear
                fgi_val = (
                    fgi_df.loc[dt_date_only, "fgi_value"]
                    if dt_date_only in fgi_df.index
                    else 50
                )
                if fgi_val > 35:
                    regime_allowed = False
            elif current_regime == "sideways":
                # In sideways, require both FGI extreme and RSI extreme
                regime_allowed = True

        if dt_date_only not in fgi_df.index:
            continue

        fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]
        _rsi_val = rsi.iloc[i] if pd.notna(rsi.iloc[i]) else 50.0
        buy_thresh_val = buy_thresh.loc[dt_date_only]
        sell_thresh_val = sell_thresh.loc[dt_date_only]
        pred_val = (
            pred_series.loc[dt_date_only]
            if pred_series is not None and dt_date_only in pred_series.index
            else 0.5
        )
        # Buy condition: technical signals AND (ML if available, or always pass if ML disabled)
        technical_buy = fgi_val <= buy_thresh_val and _rsi_val < 30
        ml_buy = (
            pred_series is None or pred_val > ml_thresh
        )  # Pass if no ML or ML threshold met
        is_buy = technical_buy and ml_buy
        is_extreme_greed = fgi_val >= sell_thresh_val
        is_overbought = _rsi_val > 70

        # Multi-Timeframe Filtering (if enabled)
        higher_tf_buy_filter = True
        tf_conflict_logged = False

        if enable_multi_tf and higher_tf_data is not None:
            # Get higher timeframe indicators for this date
            higher_trend = higher_tf_data.get(
                "higher_trend", True
            )  # Default to bullish
            higher_rsi = higher_tf_data.get("higher_rsi", 50)  # Default to neutral

            # If Series, get value at current index (for backtesting with aligned data)
            if isinstance(higher_trend, pd.Series):
                higher_trend = higher_trend.iloc[i]
            if isinstance(higher_rsi, pd.Series):
                higher_rsi = higher_rsi.iloc[i]

            # Filter 1: Only buy if higher timeframe trend is bullish
            higher_tf_buy_filter = bool(higher_trend)

            # Filter 2: Don't buy if higher TF is overbought (RSI > 80, less strict than 70)
            if higher_rsi > 80:
                higher_tf_buy_filter = False

            # Log TF conflicts for analysis
            if not tf_conflict_logged:
                if is_buy and not higher_tf_buy_filter:
                    print(
                        f"  MTF CONFLICT at {dt_date_only}: Buy signal filtered by higher TF (trend={higher_trend}, rsi={higher_rsi:.1f})"
                    )
                    tf_conflict_logged = True

        # Apply multi-TF filters to buy/sell signals
        is_buy = is_buy and higher_tf_buy_filter

        # Apply regime filter to buy signals
        if enable_regime_filter and not regime_allowed:
            is_buy = False

        # Short condition: extreme greed + overbought RSI
        # Short entry: FGI >= 75 AND RSI > 70
        is_short = fgi_val >= 75 and _rsi_val > 70

        # Apply regime filter to short signals
        # In strong bull, avoid shorting until correction
        if enable_regime_filter and current_regime == "strong_bull":
            # Only short if FGI > 85 (extreme greed)
            if fgi_val < 85:
                is_short = False

        # Long entry
        if not in_position and is_buy:
            entries.iloc[i, 0] = True
            in_position = True
            position_type = "long"
            position_price = price
            if use_atr_trail:
                atr_val = (
                    atr.iloc[i]
                    if i < len(atr) and pd.notna(atr.iloc[i])
                    else price * 0.02
                )
                trailing_stop = price - (atr_val * atr_multiplier)
            else:
                trailing_stop = price * (1 - trail_pct)

        # Short entry
        elif not in_position and is_short:
            entries.iloc[i, 0] = True
            in_position = True
            position_type = "short"
            position_price = price
            if use_atr_trail:
                atr_val = (
                    atr.iloc[i]
                    if i < len(atr) and pd.notna(atr.iloc[i])
                    else price * 0.02
                )
                trailing_stop = price + (
                    atr_val * atr_multiplier
                )  # Short stop is ABOVE entry
            else:
                trailing_stop = price * (1 + trail_pct)

        if in_position and position_type == "long":
            # Update trailing stop for long
            if use_atr_trail:
                atr_val = (
                    atr.iloc[i]
                    if i < len(atr) and pd.notna(atr.iloc[i])
                    else price * 0.02
                )
                atr_stop = price - (atr_val * atr_multiplier)
                trailing_stop = max(trailing_stop, atr_stop)
            else:
                trailing_stop = max(trailing_stop, price * (1 - trail_pct))

            pnl_pct = (price - position_price) / position_price

            # Exit conditions for long
            long_exit = (
                is_extreme_greed
                or is_overbought
                or pnl_pct >= take_profit_pct
                or price <= trailing_stop
                or max_drawdown_reached
            )

            if long_exit:
                exits.iloc[i, 0] = True
                in_position = False
                position_type = None
                max_drawdown_reached = False

        if in_position and position_type == "short":
            # Update trailing stop for short (trailing stop moves DOWN)
            if use_atr_trail:
                atr_val = (
                    atr.iloc[i]
                    if i < len(atr) and pd.notna(atr.iloc[i])
                    else price * 0.02
                )
                atr_stop = price + (atr_val * atr_multiplier)
                trailing_stop = min(trailing_stop, atr_stop)  # Short stop moves DOWN
            else:
                trailing_stop = min(trailing_stop, price * (1 + trail_pct))

            # PnL for short (profit when price goes DOWN)
            pnl_pct = (position_price - price) / position_price

            # Max drawdown protection for short (when price rises)
            short_drawdown = (price - position_price) / position_price
            if max_drawdown_pct > 0 and short_drawdown >= max_drawdown_pct:
                max_drawdown_reached = True

            # Exit conditions for short
            short_exit = (
                (not is_extreme_greed and fgi_val <= 45)  # Fear returns
                or _rsi_val < 30  # Oversold
                or pnl_pct >= 0.15  # 15% profit (price dropped 15%)
                or price >= trailing_stop  # Stop hit
                or max_drawdown_reached
            )

            if short_exit:
                exits.iloc[i, 0] = True
                in_position = False
                position_type = None
                max_drawdown_reached = False

    if entries.sum().sum() == 0:
        return {
            "granularity": granularity_name,
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "outperformance": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "multi_tf_enabled": enable_multi_tf,
            "multi_tf_data": higher_tf_data is not None,
        }

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        freq=freq,
        init_cash=INITIAL_CAPITAL,
        fees=(MAKER_FEE, TAKER_FEE),
    )

    stats = pf.stats()
    benchmark_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    return {
        "granularity": granularity_name,
        "total_return": stats["Total Return [%]"],
        "benchmark_return": benchmark_return,
        "outperformance": stats["Total Return [%]"] - benchmark_return,
        "max_drawdown": stats["Max Drawdown [%]"],
        "total_trades": int(stats["Total Trades"]),
        "win_rate": stats["Win Rate [%]"],
        "sharpe_ratio": stats["Sharpe Ratio"],
        "multi_tf_enabled": enable_multi_tf,
        "multi_tf_data": higher_tf_data is not None,
    }
