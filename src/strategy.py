"""
Trading strategy implementation.
"""

import warnings

import pandas as pd
import vectorbt as vbt

from .config import INITIAL_CAPITAL, MAKER_FEE, TAKER_FEE

# Suppress vectorbt aggregation warning (internal to vectorbt, doesn't affect functionality)
warnings.filterwarnings(
    "ignore", message="Object has multiple columns.*Aggregating using.*mean"
)


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
) -> dict:
    """Generate trading signal using the exact same logic as backtesting.

    Returns dict with:
        - signal: 'buy', 'sell', or 'hold'
        - in_position: bool, whether we should be in a position
        - indicators: dict with fgi, rsi, pred, etc.
    """
    from .indicators import calculate_rsi

    rsi = calculate_rsi(close, window=rsi_window)

    latest_close = close.iloc[-1]
    latest_dt = close.index[-1]

    dt_ts = pd.Timestamp(latest_dt)
    dt_date_only = dt_ts.normalize()

    if dt_date_only not in fgi_df.index:
        return {
            "signal": "hold",
            "in_position": False,
            "error": "FGI data not available",
        }

    fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]

    latest_rsi = rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else 50.0

    buy_thresh = (
        fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(buy_quantile).iloc[-1]
    )
    sell_thresh = (
        fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(sell_quantile).iloc[-1]
    )

    pred_val = 0.5
    if pred_series is not None and dt_date_only in pred_series.index:
        pred_val = pred_series.loc[dt_date_only]

    # Buy condition: technical signals AND (ML if available, or always pass if ML disabled)
    technical_buy = fgi_val <= buy_thresh and latest_rsi < 30
    ml_buy = (
        pred_series is None or pred_val > ml_thresh
    )  # Pass if no ML or ML threshold met
    is_buy = technical_buy and ml_buy
    is_extreme_greed = fgi_val >= sell_thresh
    is_overbought = latest_rsi > 70

    # Multi-Timeframe Filtering (if enabled)
    higher_tf_buy_filter = True

    if enable_multi_tf and higher_tf_data is not None:
        # Get higher timeframe indicators for this date/bar
        higher_trend = higher_tf_data.get("higher_trend", True)
        higher_rsi = higher_tf_data.get("higher_rsi", 50)

        # If Series, get last value (for live trading with aligned data)
        if isinstance(higher_trend, pd.Series):
            higher_trend = higher_trend.iloc[-1]
        if isinstance(higher_rsi, pd.Series):
            higher_rsi = higher_rsi.iloc[-1]

        # Filter 1: Only buy if higher timeframe trend is bullish
        higher_tf_buy_filter = bool(higher_trend)

        # Filter 2: Don't buy if higher TF is overbought (RSI > 70)
        if higher_rsi > 70:
            higher_tf_buy_filter = False

        # Log TF conflicts for analysis
        if is_buy and not higher_tf_buy_filter:
            print(
                f"  MTF CONFLICT at {dt_date_only}: Buy signal filtered by higher TF (trend={higher_trend}, rsi={higher_rsi:.1f})"
            )

    # Apply multi-TF filters to buy/sell signals
    is_buy = is_buy and higher_tf_buy_filter

    indicators = {
        "fgi": fgi_val,
        "fgi_buy_thresh": buy_thresh,
        "fgi_sell_thresh": sell_thresh,
        "rsi": latest_rsi,
        "ml_pred": pred_val,
        "ml_thresh": ml_thresh,
        "price": latest_close,
        "is_extreme_greed": is_extreme_greed,
        "is_overbought": is_overbought,
        "multi_tf_enabled": enable_multi_tf,
    }

    if enable_multi_tf and higher_tf_data is not None:
        indicators["higher_trend"] = higher_tf_data.get("higher_trend", True)
        indicators["higher_rsi"] = higher_tf_data.get("higher_rsi", 50)

    return {
        "signal": "buy"
        if is_buy
        else "sell"
        if is_extreme_greed or is_overbought
        else "hold",
        "in_position": False,
        "indicators": indicators,
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
    enable_multi_tf: bool = True,
) -> dict:
    """Run fear & greed strategy with RSI filter, dynamic FGI thresholds, trailing stops, and multi-TF analysis, return performance metrics.

    Args:
        close: Price series at current timeframe
        freq: Frequency string (e.g., '1d', '1h', '15min')
        fgi_df: Fear & Greed Index data (daily)
        granularity_name: Granularity name for reporting
        rsi_window: RSI calculation window
        trail_pct: Trailing stop percentage
        buy_quantile: FGI buy quantile threshold
        sell_quantile: FGI sell quantile threshold
        ml_thresh: ML prediction threshold
        pred_series: ML prediction series
        higher_tf_data: Dictionary with higher timeframe indicators
        enable_multi_tf: Enable multi-timeframe filtering

    Returns:
        Dictionary with performance metrics
    """
    from .indicators import calculate_macd, calculate_rsi

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
    position_price = 0.0
    trailing_stop = 0.0
    take_profit_pct = 0.25

    for i in range(len(close)):
        price = close.iloc[i]
        dt = close.index[i]

        dt_ts = pd.Timestamp(dt)
        dt_date_only = dt_ts.normalize()

        if dt_date_only not in fgi_df.index:
            continue

        fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]
        rsi_val = rsi.iloc[i] if pd.notna(rsi.iloc[i]) else 50.0
        buy_thresh_val = buy_thresh.loc[dt_date_only]
        sell_thresh_val = sell_thresh.loc[dt_date_only]
        pred_val = (
            pred_series.loc[dt_date_only]
            if pred_series is not None and dt_date_only in pred_series.index
            else 0.5
        )
        # Buy condition: technical signals AND (ML if available, or always pass if ML disabled)
        technical_buy = fgi_val <= buy_thresh_val and rsi_val < 30
        ml_buy = (
            pred_series is None or pred_val > ml_thresh
        )  # Pass if no ML or ML threshold met
        is_buy = technical_buy and ml_buy
        is_extreme_greed = fgi_val >= sell_thresh_val
        is_overbought = rsi_val > 70

        # Multi-Timeframe Filtering (if enabled)
        higher_tf_buy_filter = True
        higher_tf_sell_filter = True
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

            # Filter 2: Don't buy if higher TF is overbought (RSI > 70)
            if higher_rsi > 70:
                higher_tf_buy_filter = False

            # Filter 3: Exit immediately if higher TF trend turns bearish
            if not bool(higher_trend):
                higher_tf_sell_filter = False

            # Log TF conflicts for analysis
            if not tf_conflict_logged:
                if is_buy and not higher_tf_buy_filter:
                    print(
                        f"  MTF CONFLICT at {dt_date_only}: Buy signal filtered by higher TF (trend={higher_trend}, rsi={higher_rsi:.1f})"
                    )
                    tf_conflict_logged = True
                elif in_position and not higher_tf_sell_filter:
                    print(
                        f"  MTF CONFLICT at {dt_date_only}: Hold position, but higher TF bearish (trend={higher_trend})"
                    )
                    tf_conflict_logged = True

        # Apply multi-TF filters to buy/sell signals
        is_buy = is_buy and higher_tf_buy_filter

        if not in_position and is_buy:
            entries.iloc[i, 0] = True
            in_position = True
            position_price = price
            trailing_stop = price * (1 - trail_pct)

        if in_position:
            # Update trailing stop
            trailing_stop = max(trailing_stop, price * (1 - trail_pct))
            pnl_pct = (price - position_price) / position_price

            if (
                is_extreme_greed
                or is_overbought
                or pnl_pct >= take_profit_pct
                or price <= trailing_stop
            ):
                exits.iloc[i, 0] = True
                in_position = False

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
