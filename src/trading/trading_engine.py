"""
Live and test trading engine.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

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
    """Get current position for a symbol."""
    try:
        positions = trading_client.get_all_positions()
        pos = next((p for p in positions if p.symbol == qsymbol), None)
        return float(pos.qty) if pos else 0.0
    except Exception as e:
        print(f"Error getting position: {e}")
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


def analyze_live_signal(fgi_df: pd.DataFrame) -> Optional[Dict]:
    """Analyze current market using the optimized strategy from backtesting."""
    try:
        current_close = get_current_price("BTC-USD")

        if current_close is None:
            return None

        close_series = pd.Series([current_close], index=[pd.Timestamp.now(tz="UTC")])

        global pred_series
        original_pred_series = pred_series

        try:
            # Use risk-focused strategy for 2026 market conditions
            signal = generate_signal(
                close_series,
                fgi_df,
                # Risk-focused parameters (no ML, simple FGI thresholds)
                fear_entry_threshold=30,  # Enter when FGI <= 30 (extreme fear)
                greed_exit_threshold=70,  # Exit when FGI >= 70 (extreme greed)
                max_drawdown_exit=0.08,  # Exit if 8% loss
                volatility_stop_multiplier=1.5,  # Volatility-based stop
                pred_series=None,  # Disable ML for live trading
                enable_multi_tf=False,  # Disable multi-TF for simplicity
                enable_short_selling=True,  # Enable short selling
            )

            return signal
        finally:
            # Restore original pred_series (if needed)
            pred_series = original_pred_series

    except Exception as e:
        print(f"Error analyzing signal: {e}")
        return None


def analyze_test_signal(fgi_df: pd.DataFrame) -> Optional[Dict]:
    """Analyze current market using the optimized strategy from backtesting."""
    try:
        current_close = get_current_price("BTC-USD")
        if current_close is None:
            return None
        close_series = pd.Series([current_close], index=[pd.Timestamp.now(tz="UTC")])

        global pred_series
        original_pred_series = pred_series

        try:
            # Use risk-focused strategy for 2026 market conditions
            signal = generate_signal(
                close_series,
                fgi_df,
                # Risk-focused parameters (no ML, simple FGI thresholds)
                fear_entry_threshold=30,  # Enter when FGI <= 30 (extreme fear)
                greed_exit_threshold=70,  # Exit when FGI >= 70 (extreme greed)
                max_drawdown_exit=0.08,  # Exit if 8% loss
                volatility_stop_multiplier=1.5,  # Volatility-based stop
                pred_series=None,  # Disable ML for live trading
                enable_multi_tf=False,  # Disable multi-TF for simplicity
                enable_short_selling=True,  # Enable short selling
            )
            return signal
        finally:
            # Restore original pred_series (if needed)
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
            portfolio_state = load_test_state()
            portfolio_value = portfolio_state["cash"]
            current_positions = (
                {"BTC": current_position * price} if current_position != 0 else {}
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
        portfolio_value = portfolio_state["cash"]
        current_positions = {"BTC": current_btc * price} if current_btc != 0 else {}
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
    signal_info: dict, action: str, quantity: float, order_id: str = None
) -> None:
    """Log trade to file."""
    try:
        indicators = signal_info.get("indicators", {})
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "symbol": "BTC/USD",
            "action": action,
            "quantity": quantity,
            "price": indicators.get("price", 0),
            "fgi": indicators.get("fgi", 0),
            "rsi": indicators.get("rsi", 0),
            "ml_pred": indicators.get("ml_pred", 0),
            "order_id": order_id,
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

    SYMBOL = "BTC/USD"
    CHECK_INTERVAL = 300

    portfolio_state = load_test_state()

    if portfolio_state["initialized"]:
        print("Resuming test session from saved state:")
        print(f"  Cash: ${portfolio_state['cash']:.2f}")
        print(f"  BTC Held: {portfolio_state['btc_held']:.6f}")
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

                current_price = get_current_price("BTC-USD")
                if current_price is None:
                    print("  Could not fetch current price")
                    import time

                    time.sleep(60)
                    continue

                portfolio_value = get_test_portfolio_value(
                    portfolio_state, current_price
                )
                print(
                    f"  Portfolio: ${portfolio_value:.2f} ({portfolio_state['cash']:.2f} cash + {portfolio_state['btc_held']:.6f} BTC @ ${current_price:,.2f})"
                )

                signal_info = analyze_test_signal(fgi_df)
                if signal_info and "indicators" in signal_info:
                    ind = signal_info["indicators"]
                    print(f"  BTC: ${ind.get('price', 0):,.2f}")
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

    final_price = get_current_price("BTC-USD")
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
    elif not api_key or not secret_key:
        print("\nLive Trading: ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
        print("Create a .env file with these variables")
    else:
        SYMBOL = "BTC/USD"
        CHECK_INTERVAL = 300  # 5 minutes between checks
        TRADING_CLIENT = TradingClient(api_key, secret_key, paper=True)

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
                    print(f"\n[{now}] Checking signal...")

                    account_info = get_account_info(TRADING_CLIENT)
                    if account_info:
                        print(
                            f"  Account: ${account_info['equity']:.2f} equity, ${account_info['cash']:.2f} cash"
                        )

                    position = get_position(SYMBOL, TRADING_CLIENT)
                    print(f"  Position: {position:.6f} {SYMBOL}")

                    signal_info = analyze_live_signal(fgi_df)
                    if signal_info and "indicators" in signal_info:
                        ind = signal_info["indicators"]
                        print(f"  BTC: ${ind.get('price', 0):,.2f}")
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
                            order = execute_trade(SYMBOL, action, qty, TRADING_CLIENT)
                            order_id = order.id if order else None
                            log_trade(signal_info, action, qty, order_id)
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
