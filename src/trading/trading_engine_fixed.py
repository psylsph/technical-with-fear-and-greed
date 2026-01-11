"""
Live and test trading engine.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from ..config import BEST_PARAMS, PROJECT_ROOT, TEST_STATE_FILE
from ..data.data_fetchers import get_current_price
from ..ml.ml_model import pred_series
from ..portfolio import load_test_state, save_test_state, calculate_portfolio_var
from ..strategy import generate_signal


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_return: float,
    avg_loss_return: float,
    max_kelly_fraction: float = 0.25,
) -> float:
    """Calculate Kelly criterion position size.

    Args:
        win_rate: Historical win rate (0-1)
        avg_win_return: Average return on winning trades
        avg_loss_return: Average return on losing trades (negative)
        max_kelly_fraction: Maximum allowed Kelly fraction

    Returns:
        Kelly fraction (0-1) representing position size as % of capital
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.05  # Conservative fallback

    if avg_loss_return >= 0:
        return 0.05  # Should be negative for losses

    # Kelly formula: K = (W * R - L) / R
    # Where W = win rate, L = loss rate, R = win/loss ratio
    loss_rate = 1 - win_rate
    win_loss_ratio = (
        abs(avg_win_return / avg_loss_return) if avg_loss_return != 0 else 1
    )

    kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

    # Apply bounds and conservatism
    kelly_fraction = max(0, min(kelly_fraction, max_kelly_fraction))

    # Half-Kelly for additional conservatism
    kelly_fraction *= 0.5

    return max(kelly_fraction, 0.01)  # Minimum 1% position size


def get_historical_performance() -> Dict:
    """Get historical trading performance for Kelly calculation."""
    try:
        log_file = os.path.join(PROJECT_ROOT, "trade_log.json")
        if not os.path.exists(log_file):
            # No historical data, use conservative defaults
            return {
                "win_rate": 0.5,
                "avg_win_return": 0.10,
                "avg_loss_return": -0.05,
                "total_trades": 0,
            }

        with open(log_file) as f:
            trades = json.load(f)

        if len(trades) < 5:
            # Insufficient data, use defaults
            return {
                "win_rate": 0.5,
                "avg_win_return": 0.10,
                "avg_loss_return": -0.05,
                "total_trades": len(trades),
            }

        # Calculate performance metrics
        winning_trades = []
        losing_trades = []

        # Group trades by entry/exit pairs
        trade_pairs = []
        current_trade = None

        for trade in trades:
            if trade["action"].lower() == "buy" and current_trade is None:
                current_trade = {"entry": trade}
            elif trade["action"].lower() == "sell" and current_trade is not None:
                current_trade["exit"] = trade
                trade_pairs.append(current_trade)
                current_trade = None

        for pair in trade_pairs:
            entry_price = pair["entry"]["price"]
            exit_price = pair["exit"]["price"]
            trade_return = (exit_price - entry_price) / entry_price

            if trade_return > 0:
                winning_trades.append(trade_return)
            else:
                losing_trades.append(trade_return)

        win_rate = len(winning_trades) / len(trade_pairs) if trade_pairs else 0.5
        avg_win_return = (
            sum(winning_trades) / len(winning_trades) if winning_trades else 0.10
        )
        avg_loss_return = (
            sum(losing_trades) / len(losing_trades) if losing_trades else -0.05
        )

        return {
            "win_rate": win_rate,
            "avg_win_return": avg_win_return,
            "avg_loss_return": avg_loss_return,
            "total_trades": len(trade_pairs),
        }

    except Exception as e:
        print(f"Error calculating historical performance: {e}")
        return {
            "win_rate": 0.5,
            "avg_win_return": 0.10,
            "avg_loss_return": -0.05,
            "total_trades": 0,
        }


# Alpaca trading imports (for live trading functionality)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Note: alpaca-py not installed. Live trading features disabled.")


# Quiet mode settings
QUIET_MODE = False
LAST_STATUS_TIME = None


def set_quiet_mode(enabled: bool = True):
    """Enable or disable quiet mode."""
    global QUIET_MODE
    QUIET_MODE = enabled


def quiet_log(message: str, force: bool = False):
    """Print message only if not in quiet mode or if forced."""
    global QUIET_MODE
    if not QUIET_MODE or force:
        print(message)


def quiet_status(
    asset: str, signal: str, price: float, position: float, pnl: float = 0
):
    """Print status update in quiet mode."""
    global LAST_STATUS_TIME
    now = datetime.now()

    # Print on trades
    if signal in ["buy", "sell", "short", "cover"]:
        quiet_log(f"🚨 TRADE: {asset} - {signal.upper()} @ ${price:,.2f}")

    # Print every minute
    if LAST_STATUS_TIME is None or (now - LAST_STATUS_TIME).seconds >= 60:
        if position > 0:
            quiet_log(
                f"📊 {asset}: ${price:,.2f} | Pos: {position:.4f} | PnL: ${pnl:,.2f}"
            )
        else:
            quiet_log(f"📊 {asset}: ${price:,.2f} | No position")
        LAST_STATUS_TIME = now


def execute_trade(
    symbol: str, side: str, qty: float, trading_client=None
) -> Optional[object]:
    """Execute trade via Alpaca."""
    if not ALPACA_AVAILABLE or trading_client is None:
        print(
            f"Trade execution disabled (Alpaca not configured): {side} {qty} {symbol}"
        )
        return None

    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    # Crypto orders use IOC (Immediate or Cancel), stocks use DAY
    if "/" in symbol:  # Crypto symbol (e.g., "ETH/USD")
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            type="market",
            time_in_force=TimeInForce.IOC,
        )
    else:  # Stock symbol
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            type="market",
            time_in_force=TimeInForce.DAY,
        )
    try:
        order = trading_client.submit_order(order_data)
        print(f"Order submitted: {order.id} - {side.upper()} {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None


def get_position(qsymbol: str, trading_client) -> float:
    """Get current position for a symbol.

    Args:
        qsymbol: Symbol to search for (e.g., "ETHUSD", "ETH-USD", "ETH/USD")
        trading_client: Alpaca trading client

    Returns:
        Position quantity (can be negative for short positions)
    """
    try:
        positions = trading_client.get_all_positions()

        # Try exact match first
        pos = next((p for p in positions if p.symbol == qsymbol), None)

        # If not found, try with hyphen removed (e.g., "ETH-USD" -> "ETHUSD")
        if pos is None and "-" in qsymbol:
            alt_symbol = qsymbol.replace("-", "")
            pos = next((p for p in positions if p.symbol == alt_symbol), None)

        # If still not found, try with slash removed (e.g., "ETH/USD" -> "ETHUSD")
        if pos is None and "/" in qsymbol:
            alt_symbol = qsymbol.replace("/", "")
            pos = next((p for p in positions if p.symbol == alt_symbol), None)

        # If still not found, try adding "USD" if missing (e.g., "ETH" -> "ETHUSD")
        if pos is None and not qsymbol.endswith("USD"):
            alt_symbol = qsymbol + "USD"
            pos = next((p for p in positions if p.symbol == alt_symbol), None)

        qty = float(pos.qty) if pos else 0.0
        return qty
    except Exception as e:
        print(f"Error getting position for {qsymbol}: {e}")
        return 0.0


def get_account_info(trading_client) -> Optional[Dict]:
    """Get account information."""
    try:
        account = trading_client.get_account()
        return {
            "cash": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
        }
    except Exception as e:
        print(f"Error getting account: {e}")
        return None


def analyze_live_signal(
    fgi_df: pd.DataFrame, symbol: str = "ETH-USD", trading_client=None
) -> Optional[Dict]:
    """Analyze current market using the optimized strategy from backtesting.

    Args:
        fgi_df: Fear & Greed Index data
        symbol: Trading symbol (default: ETH-USD)
        trading_client: Alpaca trading client for live pricing
    """
    try:
        close_series = None

        # For live trading, get price history from Alpaca
        if trading_client and ALPACA_AVAILABLE:
            try:
                alpaca_symbol = symbol  # e.g., "ETH-USD"
                # Get historical bars for RSI calculation (need at least 30 days)
                bars = trading_client.get_crypto_bars(alpaca_symbol, "1Day", limit=60)

                if bars and len(bars) >= 30:
                    # Convert Alpaca bars to pandas Series
                    closes = [bar.close for bar in bars]
                    timestamps = [pd.Timestamp(bar.timestamp) for bar in bars]
                    close_series = pd.Series(closes, index=timestamps).sort_index()
                    print(f"Using Alpaca live data: {len(close_series)} bars")
                else:
                    print(
                        f"Insufficient Alpaca data: {len(bars) if bars else 0} bars (need 30)"
                    )
            except Exception as e:
                print(f"Error getting Alpaca historical data: {e}")

        # Fallback to cached data if Alpaca fails
        if close_series is None or len(close_series) < 30:
            print("Falling back to cached data...")
            from ..data.data_fetchers import (
                fetch_unified_price_data,
                fetch_eth_price_data,
            )
            from datetime import datetime

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

            if symbol == "ETH-USD":
                ohlcv = fetch_eth_price_data(start_date, end_date, "1d")
            else:
                ohlcv = fetch_unified_price_data(symbol, start_date, end_date, "1d")

            if ohlcv is None or len(ohlcv) < 30:
                print(
                    f"Could not fetch sufficient {symbol} data (need at least 30 bars)"
                )
                return None

            close_series = ohlcv["close"]

        global pred_series
        original_pred_series = pred_series

        try:
            signal = generate_signal(
                close_series,
                fgi_df,
                rsi_window=BEST_PARAMS["rsi_window"],
                trail_pct=BEST_PARAMS["trail_pct"],
                buy_quantile=BEST_PARAMS["buy_quantile"],
                sell_quantile=BEST_PARAMS["sell_quantile"],
                ml_thresh=BEST_PARAMS["ml_thresh"],
                fear_entry_threshold=30,
                greed_exit_threshold=70,
                max_drawdown_exit=0.08,
                _volatility_stop_multiplier=1.5,
                pred_series=None,
                enable_multi_tf=False,
                enable_short_selling=True,
            )

            return signal
        finally:
            pred_series = original_pred_series

    except Exception as e:
        print(f"Error analyzing signal: {e}")
        return None


def analyze_test_signal(fgi_df: pd.DataFrame) -> Optional[Dict]:
    """Analyze current market using the optimized strategy from backtesting."""
    try:
        from ..config import DEFAULT_ASSET

        # Use cached data instead of live API to avoid rate limiting
        from ..data.data_fetchers import fetch_unified_price_data
        from datetime import datetime

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

        asset_data = fetch_unified_price_data(DEFAULT_ASSET, start_date, end_date, "1d")
        if asset_data is None or len(asset_data) < 30:
            print(f"Could not fetch {DEFAULT_ASSET} data (need at least 30 bars)")
            return None

        close_series = asset_data["close"]

        global pred_series
        original_pred_series = pred_series

        try:
            signal = generate_signal(
                close_series,
                fgi_df,
                rsi_window=BEST_PARAMS["rsi_window"],
                trail_pct=BEST_PARAMS["trail_pct"],
                buy_quantile=BEST_PARAMS["buy_quantile"],
                sell_quantile=BEST_PARAMS["sell_quantile"],
                ml_thresh=BEST_PARAMS["ml_thresh"],
                fear_entry_threshold=30,
                greed_exit_threshold=70,
                max_drawdown_exit=0.08,
                _volatility_stop_multiplier=1.5,
                pred_series=None,
                enable_multi_tf=False,
                enable_short_selling=True,
            )
            return signal
        finally:
            pred_series = original_pred_series

    except Exception as e:
        print(f"Error analyzing signal: {e}")
        return None


def should_trade(
    signal_info: dict,
    current_position: float,
    is_live: bool = False,
    account_info: Optional[Dict] = None,
) -> Tuple[str, float]:
    """Determine if a trade should be executed with Kelly criterion (supports short selling)."""
    signal = signal_info.get("signal", "hold")
    price = signal_info.get("indicators", {}).get("price", 0)

    # Long position entry
    if signal == "buy" and current_position == 0:
        # Use Kelly criterion for position sizing
        hist_perf = get_historical_performance()
        kelly_fraction = calculate_kelly_fraction(
            hist_perf["win_rate"],
            hist_perf["avg_win_return"],
            hist_perf["avg_loss_return"],
        )

        # Apply market regime adjustment
        fgi_trend = signal_info.get("indicators", {}).get("fgi_trend", "sideways")
        if fgi_trend == "bull":
            kelly_fraction *= 1.2  # Slightly more aggressive in bull markets
        elif fgi_trend == "bear":
            kelly_fraction *= 0.8  # More conservative in bear markets

        # Apply portfolio VaR adjustment for risk management
        if is_live and account_info:
            portfolio_value = account_info.get("equity", 1000)
            # Assume current position value for VaR calculation
            current_positions = (
                {price: current_position * price} if current_position != 0 else {}
            )
            var_metrics = calculate_portfolio_var(current_positions)

            # Reduce position size if approaching VaR limits (keep 20% buffer)
            available_for_risk = portfolio_value - var_metrics["daily_var"] * 1.2
            max_position_value = min(
                portfolio_value * kelly_fraction, available_for_risk
            )
        else:
            # Test mode - use portfolio cash with VaR consideration
            from ..config import DEFAULT_ASSET

            portfolio_state = load_test_state()
            portfolio_value = portfolio_state["cash"]
            current_positions = (
                {DEFAULT_ASSET: current_position * price}
                if current_position != 0
                else {}
            )
            var_metrics = calculate_portfolio_var(current_positions)

            # Apply VaR buffer
            available_for_risk = portfolio_value - var_metrics["daily_var"] * 1.2
            max_position_value = min(
                portfolio_value * kelly_fraction, available_for_risk
            )

        quantity = (max_position_value / price) * 0.95  # 5% buffer
        quantity = round(quantity, 6)
        return ("buy", max(quantity, 0.0001))

    # Short position entry
    elif signal == "short" and current_position == 0:
        # Use Kelly criterion for position sizing (more conservative for shorts)
        hist_perf = get_historical_performance()
        kelly_fraction = (
            calculate_kelly_fraction(
                hist_perf["win_rate"],
                hist_perf["avg_win_return"],
                hist_perf["avg_loss_return"],
            )
            * 0.7
        )  # More conservative for short selling

        # Apply market regime adjustment
        fgi_trend = signal_info.get("indicators", {}).get("fgi_trend", "sideways")
        if fgi_trend == "bull":
            kelly_fraction *= 1.1  # Slightly more aggressive in bull markets for shorts
        elif fgi_trend == "bear":
            kelly_fraction *= 0.6  # Very conservative in bear markets for shorts

        if is_live and account_info:
            max_position_value = account_info["equity"] * kelly_fraction
        else:
            # Test mode - use portfolio cash
            cash_amount = account_info.get("cash", 1000) if account_info else 1000.0
            max_position_value = cash_amount * kelly_fraction

        quantity = (max_position_value / price) * 0.95  # 5% buffer
        quantity = round(quantity, 6)
        return ("sell", max(quantity, 0.0001))  # Sell to short

    # Long position exit
    elif signal == "sell" and current_position > 0:
        return ("sell", current_position)

    # Short position exit (cover)
    elif signal == "cover" and current_position < 0:
        return ("buy", abs(current_position))  # Buy to cover short

    return ("hold", 0.0)


def should_trade_test(signal_info: dict, current_btc: float) -> Tuple[str, float]:
    """Determine if a trade should be executed in test mode with Kelly criterion (supports short selling)."""
    signal = signal_info.get("signal", "hold")
    price = signal_info.get("indicators", {}).get("price", 0)

    if signal == "buy" and current_btc == 0:
        # Load current portfolio state to get cash
        portfolio_state = load_test_state()

        # Use Kelly criterion for position sizing
        hist_perf = get_historical_performance()
        kelly_fraction = calculate_kelly_fraction(
            hist_perf["win_rate"],
            hist_perf["avg_win_return"],
            hist_perf["avg_loss_return"],
        )

        # Apply market regime adjustment
        fgi_trend = signal_info.get("indicators", {}).get("fgi_trend", "sideways")
        if fgi_trend == "bull":
            kelly_fraction *= 1.2  # Slightly more aggressive in bull markets
        elif fgi_trend == "bear":
            kelly_fraction *= 0.8  # More conservative in bear markets

        # Apply portfolio VaR adjustment for risk management
        from ..config import DEFAULT_ASSET

        portfolio_value = portfolio_state["cash"]
        current_positions = (
            {DEFAULT_ASSET: current_btc * price} if current_btc != 0 else {}
        )
        var_metrics = calculate_portfolio_var(current_positions)

        # Reduce position size if approaching VaR limits (keep 20% buffer)
        available_for_risk = portfolio_value - var_metrics["daily_var"] * 1.2
        max_position_value = min(portfolio_value * kelly_fraction, available_for_risk)

        quantity = (max_position_value / price) * 0.95
        quantity = round(quantity, 6)
        return ("buy", max(quantity, 0.0001))

    elif signal == "short" and current_btc == 0:
        # Load current portfolio state to get cash
        portfolio_state = load_test_state()

        # Use Kelly criterion for position sizing (more conservative for shorts)
        hist_perf = get_historical_performance()
        kelly_fraction = (
            calculate_kelly_fraction(
                hist_perf["win_rate"],
                hist_perf["avg_win_return"],
                hist_perf["avg_loss_return"],
            )
            * 0.7
        )  # More conservative for short selling

        # Apply market regime adjustment
        fgi_trend = signal_info.get("indicators", {}).get("fgi_trend", "sideways")
        if fgi_trend == "bull":
            kelly_fraction *= 1.1  # Slightly more aggressive in bull markets for shorts
        elif fgi_trend == "bear":
            kelly_fraction *= 0.6  # Very conservative in bear markets for shorts

        max_position_value = portfolio_state["cash"] * kelly_fraction
        quantity = (max_position_value / price) * 0.95
        quantity = round(quantity, 6)
        return ("sell", max(quantity, 0.0001))  # Sell to short

    elif signal == "sell" and current_btc > 0:
        return ("sell", current_btc)

    elif signal == "cover" and current_btc < 0:
        return ("buy", abs(current_btc))  # Buy to cover short

    return ("hold", 0.0)


def log_trade(
    signal_info: dict,
    action: str,
    quantity: float,
    symbol: str = "ETH/USD",
    order_id: Optional[str] = None,
) -> None:
    """Log trade to file."""
    try:
        indicators = signal_info.get("indicators", {})
        timestamp = datetime.now().isoformat()

        # Convert numpy types to JSON-serializable Python types
        def convert_value(val):
            if hasattr(val, "item"):  # numpy scalar
                return val.item()
            elif isinstance(val, (int, float)):
                return float(val)
            else:
                return str(val)

        log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "quantity": convert_value(quantity),
            "price": convert_value(indicators.get("price", 0)),
            "fgi": convert_value(indicators.get("fgi", 0)),
            "rsi": convert_value(indicators.get("rsi", 0)),
            "ml_pred": convert_value(indicators.get("ml_pred", 0)),
            "order_id": str(order_id) if order_id else None,
        }
        log_file = os.path.join(PROJECT_ROOT, "trade_log.json")
        logs = []
        if os.path.exists(log_file):
            with open(log_file) as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error logging trade: {e}")


def run_test_trading(fgi_df: pd.DataFrame):
    """Run test trading mode."""
    print("\n" + "=" * 60)
    print("TEST MODE (Simulated Live Trading)")
    print("=" * 60)

    from ..config import DEFAULT_ASSET

    SYMBOL = DEFAULT_ASSET.replace("-USD", "/USD")
    CHECK_INTERVAL = 300

    portfolio_state = load_test_state()

    if portfolio_state["initialized"]:
        print("Resuming test session from saved state:")
        print(f"  Cash: ${portfolio_state['cash']:.2f}")
        print(f"  {DEFAULT_ASSET} Held: {portfolio_state['btc_held']:.6f}")
        print(f"  Previous Trades: {len(portfolio_state['trades'])}")
    else:
        print("Starting new test session:")
        print(f"  Initial Cash: ${portfolio_state['cash']:.2f}")
        portfolio_state["initialized"] = True
        save_test_state(portfolio_state)

    from ..portfolio import get_test_portfolio_value

    print("\nUsing optimized strategy parameters:")
    print(f"  RSI Window: {BEST_PARAMS['rsi_window']}")
    print(f"  Trail %: {BEST_PARAMS['trail_pct']}")
    print(f"  Buy Quantile: {BEST_PARAMS['buy_quantile']}")
    print(f"  Sell Quantile: {BEST_PARAMS['sell_quantile']}")
    print(f"  ML Threshold: {BEST_PARAMS['ml_thresh']}")
    print(f"\nStarting test trading monitor for {SYMBOL}")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("State will be saved to: test_portfolio_state.json")
    print("-" * 60)

    try:
        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{now}] Checking signal...")

                current_price = get_current_price(DEFAULT_ASSET)
                if current_price is None:
                    print("  Could not fetch current price")
                    import time

                    time.sleep(60)
                    continue

                portfolio_value = get_test_portfolio_value(
                    portfolio_state, current_price
                )
                print(
                    f"  Portfolio: ${portfolio_value:.2f} ({portfolio_state['cash']:.2f} cash + {portfolio_state['btc_held']:.6f} {DEFAULT_ASSET} @ ${current_price:,.2f})"
                )

                signal_info = analyze_test_signal(fgi_df)
                if signal_info and "indicators" in signal_info:
                    ind = signal_info["indicators"]
                    print(f"  {DEFAULT_ASSET}: ${ind.get('price', 0):,.2f}")
                    print(
                        f"  FGI: {ind.get('fgi', 0)} (buy<= {ind.get('fgi_buy_thresh', 0):.0f}, sell>= {ind.get('fgi_sell_thresh', 0):.0f})"
                    )
                    print(f"  RSI: {ind.get('rsi', 0):.1f} (buy<30, sell>70)")
                    print(
                        f"  ML: {ind.get('ml_pred', 0):.2f} (>{ind.get('ml_thresh', 0):.2f})"
                    )
                    if ind.get("multi_tf_enabled"):
                        higher_trend = ind.get("higher_trend", True)
                        higher_rsi = ind.get("higher_rsi", 50)
                        trend_str = "BULLISH" if higher_trend else "BEARISH"
                        print(
                            f"  Higher TF (Daily): {trend_str}, RSI: {higher_rsi:.1f}"
                        )
                    print(f"  Signal: {signal_info['signal'].upper()}")

                    action, qty = should_trade_test(
                        signal_info, portfolio_state["btc_held"]
                    )

                    if action != "hold":
                        print(
                            f"\n  >>> SIMULATED TRADE: {action.upper()} {qty:.6f} {SYMBOL} <<<"
                        )
                        from ..portfolio import simulate_trade

                        portfolio_state = simulate_trade(
                            portfolio_state, SYMBOL, action, qty, current_price
                        )
                        save_test_state(portfolio_state)
                    else:
                        if portfolio_state["btc_held"] > 0:
                            print("  No trade: Holding long position")
                        else:
                            print("  No trade: Waiting for BUY signal")

                else:
                    print("  Could not analyze signal (data fetch error)")

            except KeyboardInterrupt:
                print("\n\nShutdown signal received...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback

                traceback.print_exc()
                import time

                print("Waiting 60 seconds before retry...")
                time.sleep(60)
                continue

            import time

            print(f"\nSleeping {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 60)
    print("Test Trading Session Ended")
    print("=" * 60)

    final_price = get_current_price(DEFAULT_ASSET)
    if final_price:
        final_value = get_test_portfolio_value(portfolio_state, final_price)
        from ..config import INITIAL_CAPITAL

        initial_value = INITIAL_CAPITAL
        return_pct = ((final_value - initial_value) / initial_value) * 100
        print(f"\nFinal Portfolio Value: ${final_value:.2f}")
        print(f"Initial Capital: ${initial_value:.2f}")
        print(f"Return: {return_pct:.2f}%")
        print(f"Total Trades: {len(portfolio_state['trades'])}")

        if portfolio_state["trades"]:
            print("\nLast 5 trades:")
            for t in portfolio_state["trades"][-5:]:
                print(
                    f"  {t['time']}: {t['side'].upper()} {t['quantity']:.6f} @ ${t['price']:,.2f}"
                )

    print(f"\nState saved to: {TEST_STATE_FILE}")
    print("Run again with --test to resume from this state.")


def run_live_trading(fgi_df: pd.DataFrame):
    """Run live trading mode."""
    print("\n" + "=" * 60)
    print("LIVE TRADING MODE")
    print("=" * 60)

    import os

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not ALPACA_AVAILABLE:
        print("\nLive Trading: alpaca-py not installed")
        print("Install with: pip install alpaca-py")
        return
    elif not api_key or not secret_key:
        print("\nLive Trading: ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
        print("Create a .env file with these variables")
        print(f"Debug: ALPACA_API_KEY present: {api_key is not None}")
        print(f"Debug: ALPACA_SECRET_KEY present: {secret_key is not None}")
        return

    print("\nInitializing Alpaca client...")
    try:
        # Use ETH-USD based on backtesting results (best risk-adjusted returns)
        from ..config import DEFAULT_ASSET

        SYMBOL = DEFAULT_ASSET  # "ETH-USD" for display and position checking
        ALPACA_SYMBOL = DEFAULT_ASSET.replace(
            "-", ""
        )  # "ETH-USD" -> "ETHUSD" for Alpaca API
        CHECK_INTERVAL = 300  # 5 minutes between checks
        TRADING_CLIENT = TradingClient(api_key, secret_key, paper=True)
        print("Alpaca client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Alpaca client: {e}")
        return

    print("\n" + "=" * 60)
    print("LIVE TRADING - ETH-USD (Best Risk-Adjusted Returns)")
    print("=" * 60)
    print("\nUsing optimized strategy parameters:")
    print(f"  RSI Window: {BEST_PARAMS['rsi_window']}")
    print(f"  Trail %: {BEST_PARAMS['trail_pct']}")
    print(f"  Buy Quantile: {BEST_PARAMS['buy_quantile']}")
    print(f"  Sell Quantile: {BEST_PARAMS['sell_quantile']}")
    print(f"  ML Threshold: {BEST_PARAMS['ml_thresh']}")
    print(f"\nStarting live trading monitor for {SYMBOL}")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    trade_log = []
    try:
        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if not QUIET_MODE:
                    print(f"\n[{now}] Checking signal...")

                account_info = get_account_info(TRADING_CLIENT)
                if account_info:
                    equity = float(account_info["equity"])
                    cash = float(account_info["cash"])
                    buying_power = float(account_info.get("buying_power", 0))

                    if QUIET_MODE:
                        quiet_log(
                            f"Account: ${equity:.2f} equity, ${cash:.2f} cash, ${buying_power:.2f} buying power"
                        )
                    else:
                        print(
                            f"  Account: ${equity:.2f} equity, ${cash:.2f} cash, ${buying_power:.2f} buying power"
                        )

                position = get_position(SYMBOL, TRADING_CLIENT)
                print(f"  Current position: {position:.6f} {SYMBOL}")

                signal_info = analyze_live_signal(fgi_df, SYMBOL, TRADING_CLIENT)
                if signal_info and "indicators" in signal_info:
                    ind = signal_info["indicators"]
                    price = ind.get("price", 0)
                    pnl = (
                        (price * position) - (position * 3000) if position > 0 else 0
                    )  # Simplified PnL

                    if QUIET_MODE:
                        quiet_status(
                            SYMBOL,
                            signal_info["signal"],
                            price,
                            position,
                            pnl,
                        )
                    else:
                        print(f"  ETH: ${price:,.2f}")
                        print(
                            f"  FGI: {ind.get('fgi', 0)} (buy<= {ind.get('fgi_buy_thresh', 0):.0f}, sell>= {ind.get('fgi_sell_thresh', 0):.0f})"
                        )
                        print(f"  RSI: {ind.get('rsi', 0):.1f} (buy<30, sell>70)")
                        print(f"  Signal: {signal_info['signal'].upper()}")

                    action, qty = should_trade(
                        signal_info,
                        position,
                        is_live=True,
                        account_info=account_info,
                    )

                    if action != "hold":
                        print(
                            f"\n  >>> Executing {action.upper()} {qty:.6f} {SYMBOL} <<<"
                        )
                        # Use ALPACA_SYMBOL format for execution (e.g., "ETHUSD")
                        order = execute_trade(
                            ALPACA_SYMBOL, action, qty, TRADING_CLIENT
                        )
                        order_id = (
                            str(order.id) if order and hasattr(order, "id") else None
                        )
                        log_trade(signal_info, action, qty, SYMBOL, order_id)
                        trade_log.append(
                            {
                                "time": now,
                                "action": action,
                                "qty": qty,
                                "price": ind.get("price", 0),
                            }
                        )
                    else:
                        if position > 0:
                            print("  No trade: Holding long position")
                        else:
                            print("  No trade: Waiting for BUY signal")

                else:
                    print("  Could not analyze signal (data fetch error)")

            except KeyboardInterrupt:
                print("\n\nShutdown signal received...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                import time

                print("Waiting 60 seconds before retry...")
                time.sleep(60)
                continue

            import time

            print(f"\nSleeping {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 60)
    print("Live Trading Stopped")
    print("=" * 60)
    if trade_log:
        print(f"Trade history ({len(trade_log)} trades):")
        for t in trade_log[-10:]:
            print(
                f"  {t['time']}: {t['action'].upper()} {t['qty']:.6f} @ ${t['price']:,.2f}"
            )
