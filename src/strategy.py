"""
Trading strategy implementation.
"""

import warnings

import pandas as pd
import vectorbt as vbt

from .config import INITIAL_CAPITAL, MAKER_FEE, TAKER_FEE

# Suppress vectorbt aggregation warning (internal to vectorbt, doesn't affect functionality)
warnings.filterwarnings("ignore", message="Object has multiple columns.*Aggregating using.*mean")


def generate_signal(
    close: pd.Series,
    fgi_df: pd.DataFrame,
    rsi_window: int = 14,
    trail_pct: float = 0.10,
    buy_quantile: float = 0.2,
    sell_quantile: float = 0.8,
    ml_thresh: float = 0.5,
    pred_series: pd.Series = None,
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

    return {
        "signal": "buy"
        if is_buy
        else "sell"
        if is_extreme_greed or is_overbought
        else "hold",
        "in_position": False,
        "indicators": {
            "fgi": fgi_val,
            "fgi_buy_thresh": buy_thresh,
            "fgi_sell_thresh": sell_thresh,
            "rsi": latest_rsi,
            "ml_pred": pred_val,
            "ml_thresh": ml_thresh,
            "price": latest_close,
            "is_extreme_greed": is_extreme_greed,
            "is_overbought": is_overbought,
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
) -> dict:
    """Run fear & greed strategy with RSI filter, dynamic FGI thresholds, trailing stops, and multi-TF analysis, return performance metrics."""
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
    }
