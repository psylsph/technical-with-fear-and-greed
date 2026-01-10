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
from .risk_controls import RiskControls


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
    symbol: str, side: str, qty: float, trading_client=None, account_info: Optional[Dict] = None
) -> Optional[object]:
    """Execute trade via Alpaca with proper checks."""
    if not ALPACA_AVAILABLE or trading_client is None:
        print(
            f"Trade execution disabled (Alpaca not configured): {side} {qty} {symbol}"
        )
        return None

    # Convert "ETH/USD" to "ETHUSD" format for Alpaca API
    alpaca_symbol = symbol.replace("/", "")

    # Pre-trade validation: check position and buying power
    position_info = get_position(symbol, trading_client)
    position_qty = position_info.get("qty", 0.0)

    if side.lower() == "buy":
        # Check if already holding position
        if position_qty > 0:
            print(f"  >>> BUY SKIP: Already holding {position_qty:.6f} {symbol}")
            return None
        # Check buying power
        if account_info:
            cash = account_info.get("cash", 0)
            current_price = get_current_price("ETH-USD")
            if current_price:
                required = qty * current_price * 1.01  # 1% buffer for fees/slippage
                if required > cash:
                    print(f"  >>> BUY SKIP: Insufficient cash. Required ${required:.2f}, Available ${cash:.2f}")
                    return None
    elif side.lower() == "sell":
        # Check if holding enough to sell
        if position_qty <= 0:
            print(f"  >>> SELL SKIP: No position to sell (holding {position_qty:.6f})")
            return None
        if qty > position_qty:
            print(f"  >>> SELL SKIP: Insufficient position. Trying to sell {qty:.6f}, holding {position_qty:.6f}")
            return None

    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    # For crypto orders, use IOC (Immediate or Cancel) instead of DAY
    # Crypto market orders must use IOC or GTC, not DAY
    order_data = MarketOrderRequest(
        symbol=alpaca_symbol,  # Use Alpaca format without slash
        qty=qty,
        side=side_enum,
        type="market",
        time_in_force=TimeInForce.IOC,  # Immediate or Cancel for crypto
    )
    try:
        order = trading_client.submit_order(order_data)
        print(f"Order submitted: {order.id} - {side.upper()} {qty} {alpaca_symbol}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_position(qsymbol: str, trading_client) -> dict:
    """Get current position details from Alpaca including entry price.

    Returns dict with:
        - qty: float (position quantity, 0 if no position)
        - entry_price: float (average entry price from Alpaca)
        - current_value: float (current market value)
        - unrealized_pl: float (unrealized P&L)
        - unrealized_plpc: float (unrealized P&L percent)
    """
    try:
        # Convert "ETH/USD" to "ETHUSD" format for Alpaca
        alpaca_symbol = qsymbol.replace("/", "")

        # Get all positions and find matching symbol
        positions = trading_client.get_all_positions()

        # Try exact match first, then case-insensitive
        pos = next((p for p in positions if p.symbol == alpaca_symbol), None)
        if not pos:
            pos = next((p for p in positions if p.symbol.upper() == alpaca_symbol.upper()), None)

        if pos:
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price) if pos.avg_entry_price else 0.0
            current_price = get_current_price("ETH-USD")

            # Calculate P&L manually (Alpaca's unrealized_plpc can be unreliable)
            if qty != 0 and entry_price > 0 and current_price > 0:
                # For long positions: profit when current > entry
                # For short positions: profit when current < entry
                if qty > 0:  # Long
                    unrealized_pl = (current_price - entry_price) * qty
                    unrealized_plpc = ((current_price - entry_price) / entry_price) * 100
                else:  # Short
                    unrealized_pl = (entry_price - current_price) * abs(qty)
                    unrealized_plpc = ((entry_price - current_price) / entry_price) * 100
            else:
                unrealized_pl = 0.0
                unrealized_plpc = 0.0

            position_info = {
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "current_value": abs(qty) * current_price if current_price else 0.0,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "side": "long" if qty > 0 else "short",
            }

            # Color code the P&L output
            pl_str = f"{unrealized_plpc:+.2f}%"
            if unrealized_plpc < -5:
                pl_str = f"🔴 {pl_str}"
            elif unrealized_plpc < 0:
                pl_str = f"🟠 {pl_str}"
            elif unrealized_plpc > 5:
                pl_str = f"🟢 {pl_str}"
            else:
                pl_str = f"⚪ {pl_str}"

            print(f"  Position: {qty:.6f} {alpaca_symbol} | Entry: ${entry_price:.2f} | P&L: {pl_str}")

            return position_info
        else:
            return {
                "qty": 0.0,
                "entry_price": 0.0,
                "current_price": 0.0,
                "current_value": 0.0,
                "unrealized_pl": 0.0,
                "unrealized_plpc": 0.0,
                "side": None,
            }
    except Exception as e:
        print(f"Error getting position: {e}")
        import traceback
        traceback.print_exc()
        return {
            "qty": 0.0,
            "entry_price": 0.0,
            "current_price": 0.0,
            "current_value": 0.0,
            "unrealized_pl": 0.0,
            "unrealized_plpc": 0.0,
            "side": None,
        }


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


def check_stop_loss(position_info: dict, signal_info: dict, max_drawdown: float = 0.05) -> Tuple[bool, str]:
    """Check if stop loss should be triggered based on position entry price.

    Args:
        position_info: Dict from get_position() with entry price, current price, P&L
        signal_info: Dict from analyze_live_signal() with volatility_stop
        max_drawdown: Maximum allowed drawdown (default 5%)

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    if position_info["qty"] == 0:
        return False, "No position"

    qty = position_info["qty"]
    entry_price = position_info["entry_price"]
    current_price = position_info["current_price"]
    unrealized_plpc = position_info["unrealized_plpc"]

    if entry_price <= 0 or current_price <= 0:
        return False, "Invalid price data"

    # Check 1: Max Drawdown Exit (5% from entry)
    # For long positions, negative PL means loss
    # For short positions, positive PL means loss (opposite)
    if qty > 0:  # Long position
        drawdown_pct = unrealized_plpc  # This is in percentage terms (e.g., -5.23 for -5.23%)
        # Convert max_drawdown from decimal to percentage (0.05 -> 5.0)
        max_drawdown_pct = max_drawdown * 100
        if drawdown_pct <= -max_drawdown_pct:
            actual_drawdown = abs(drawdown_pct)
            return True, f"MAX DRAWDOWN: Position down {actual_drawdown:.2f}% (entry ${entry_price:.2f}, current ${current_price:.2f})"
    else:  # Short position
        drawdown_pct = -unrealized_plpc  # Positive PL when losing on short
        max_drawdown_pct = max_drawdown * 100
        if drawdown_pct >= max_drawdown_pct:
            return True, f"MAX DRAWDOWN: Short position down {drawdown_pct:.2f}% (entry ${entry_price:.2f}, current ${current_price:.2f})"

    # Check 2: Volatility Stop Loss
    # DISABLED: The volatility stop calculation is too aggressive for live trading
    # It's based on single price point analysis, not entry price tracking
    # Only use the max drawdown stop for now
    # indicators = signal_info.get("indicators", {})
    # volatility_stop = indicators.get("volatility_stop", 0)
    # if volatility_stop > 0 and qty > 0:  # Long position
    #     if current_price < volatility_stop:
    #         return True, f"VOLATILITY STOP: Price ${current_price:.2f} below stop ${volatility_stop:.2f}"

    return False, "Hold"


def analyze_live_signal(fgi_df: pd.DataFrame) -> Optional[Dict]:
    """Analyze current market using the optimized strategy from backtesting."""
    try:
        current_close = get_current_price("ETH-USD")

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
                max_drawdown_exit=0.05,  # Exit if 5% loss
                _volatility_stop_multiplier=1.5,  # Volatility-based stop
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
        current_close = get_current_price("ETH-USD")
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
                max_drawdown_exit=0.05,  # Exit if 5% loss
                _volatility_stop_multiplier=1.5,  # Volatility-based stop
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
    position_info: dict,
    is_live: bool = False,
    account_info: Optional[Dict] = None,
) -> Tuple[str, float]:
    """Determine if a trade should be executed with Kelly criterion (supports short selling).

    Args:
        signal_info: Dict from analyze_live_signal() with signal and indicators
        position_info: Dict from get_position() with qty, entry_price, etc.
        is_live: Whether in live mode
        account_info: Account info dict from get_account_info()
    """
    signal = signal_info.get("signal", "hold")
    price = signal_info.get("indicators", {}).get("price", 0)
    current_position = position_info.get("qty", 0.0)

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
                {"ETH": current_position * price} if current_position != 0 else {}
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
                {"ETH": current_position * price} if current_position != 0 else {}
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
        return ("short", max(quantity, 0.0001))

    # Long position exit
    elif signal == "sell" and current_position > 0:
        return ("sell", current_position)

    # Short position exit (cover)
    elif signal == "cover" and current_position < 0:
        return ("buy", abs(current_position))  # Buy to cover short

    return ("hold", 0.0)


def should_trade_with_position_limit(
    signal_info: dict,
    position_info: dict,
    is_live: bool = False,
    account_info: Optional[Dict] = None,
    risk_controls: 'RiskControls' = None,
) -> Tuple[str, float]:
    """
    Wrapper around should_trade that enforces position size limits.

    This function calls should_trade to get the signal and quantity,
    then applies the 5% position size limit check.
    """
    # Get initial trade decision
    action, qty = should_trade(signal_info, position_info, is_live, account_info)

    # Only check position size limits for new positions (buy/short)
    if action in ("buy", "short") and risk_controls and is_live and account_info:
        price = signal_info.get("indicators", {}).get("price", 0)
        equity = account_info.get("equity", 0)

        if price > 0 and equity > 0:
            allowed, adjusted_qty, reason = risk_controls.check_position_size(
                qty, price, equity
            )

            if not allowed:
                print(f"  ⚠️  {reason}")
                # Use adjusted quantity
                return (action, adjusted_qty)

    return action, qty


def should_trade_test(signal_info: dict, current_eth: float) -> Tuple[str, float]:
    """Determine if a trade should be executed in test mode with Kelly criterion (supports short selling)."""
    signal = signal_info.get("signal", "hold")
    price = signal_info.get("indicators", {}).get("price", 0)

    if signal == "buy" and current_eth == 0:
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
        current_positions = {"ETH": current_eth * price} if current_eth != 0 else {}
        var_metrics = calculate_portfolio_var(current_positions)

        # Reduce position size if approaching VaR limits (keep 20% buffer)
        available_for_risk = portfolio_value - var_metrics["daily_var"] * 1.2
        max_position_value = min(portfolio_value * kelly_fraction, available_for_risk)

        quantity = (max_position_value / price) * 0.95
        quantity = round(quantity, 6)
        return ("buy", max(quantity, 0.0001))

    elif signal == "buy" and current_eth > 0:
        # Check if we can add to the existing position (only if profitable)
        portfolio_state = load_test_state()
        entry_price = portfolio_state.get("entry_price", 0)

        if entry_price > 0 and price > entry_price:
            # Position is profitable - allow adding to it (trail a winner)
            unrealized_pnl_pct = ((price - entry_price) / entry_price) * 100

            # Only add if position has at least 2% profit
            if unrealized_pnl_pct >= 2.0:
                portfolio_value = portfolio_state["cash"] + (current_eth * price)
                current_position_value = current_eth * price
                max_position_pct = 0.05  # Max 5% of portfolio in one position

                # Can only add up to max position size
                max_position_value = portfolio_value * max_position_pct
                remaining_allowance = max_position_value - current_position_value

                if remaining_allowance > 0:
                    # Use smaller Kelly fraction for adding to position (50% of normal)
                    hist_perf = get_historical_performance()
                    kelly_fraction = calculate_kelly_fraction(
                        hist_perf["win_rate"],
                        hist_perf["avg_win_return"],
                        hist_perf["avg_loss_return"],
                    ) * 0.5

                    add_amount = min(portfolio_state["cash"] * kelly_fraction, remaining_allowance)
                    quantity = (add_amount / price) * 0.95
                    quantity = round(quantity, 6)

                    if quantity > 0.0001:
                        return ("buy", quantity)

        # Hold - either not profitable or would exceed max position size
        return ("hold", 0.0)

    elif signal == "short" and current_eth == 0:
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

    elif signal == "sell" and current_eth > 0:
        return ("sell", current_eth)

    elif signal == "cover" and current_eth < 0:
        return ("buy", abs(current_eth))  # Buy to cover short

    return ("hold", 0.0)


def log_trade(
    signal_info: dict, action: str, quantity: float, order_id: str = None
) -> None:
    """Log trade to file."""
    try:
        indicators = signal_info.get("indicators", {})
        timestamp = datetime.now().isoformat()

        # Convert numpy types to native Python types for JSON serialization
        import numpy as np

        def convert_value(val):
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert_value(v) for v in val]
            return val

        log_entry = {
            "timestamp": timestamp,
            "symbol": "ETH/USD",
            "action": action,
            "quantity": float(quantity),
            "price": convert_value(indicators.get("price", 0)),
            "fgi": convert_value(indicators.get("fgi", 0)),
            "rsi": convert_value(indicators.get("rsi", 0)),
            "ml_pred": convert_value(indicators.get("ml_pred", 0)),
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
        import traceback
        traceback.print_exc()


def run_test_trading(fgi_df: pd.DataFrame):
    """Run test trading mode."""
    print("\n" + "=" * 60)
    print("TEST MODE (Simulated Live Trading)")
    print("=" * 60)

    SYMBOL = "ETH/USD"
    CHECK_INTERVAL = 300

    portfolio_state = load_test_state()

    if portfolio_state["initialized"]:
        print("Resuming test session from saved state:")
        print(f"  Cash: ${portfolio_state['cash']:.2f}")
        print(f"  ETH Held: {portfolio_state['eth_held']:.6f}")
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

                current_price = get_current_price("ETH-USD")
                if current_price is None:
                    print("  Could not fetch current price")
                    import time

                    time.sleep(60)
                    continue

                portfolio_value = get_test_portfolio_value(
                    portfolio_state, current_price
                )
                print(
                    f"  Portfolio: ${portfolio_value:.2f} ({portfolio_state['cash']:.2f} cash + {portfolio_state['eth_held']:.6f} ETH @ ${current_price:,.2f})"
                )

                signal_info = analyze_test_signal(fgi_df)
                if signal_info and "indicators" in signal_info:
                    ind = signal_info["indicators"]
                    print(f"  ETH: ${ind.get('price', 0):,.2f}")
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
                        signal_info, portfolio_state["eth_held"]
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
                        if portfolio_state["eth_held"] > 0:
                            entry_price = portfolio_state.get("entry_price", 0)
                            if entry_price > 0:
                                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                                if pnl_pct >= 2.0:
                                    print(f"  No trade: At max position size or insufficient cash (P&L: +{pnl_pct:.1f}%)")
                                elif current_price > entry_price:
                                    print(f"  No trade: Position profitable but below 2% threshold (P&L: +{pnl_pct:.1f}%)")
                                else:
                                    print(f"  No trade: Position not profitable (P&L: {pnl_pct:.1f}%)")
                            else:
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

    final_price = get_current_price("ETH-USD")
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
        SYMBOL = "ETH/USD"
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

        # Initialize risk controls and filters
        risk_controls = RiskControls()

        print("\n🛡️ RISK CONTROLS ACTIVE:")
        print("  • Daily Loss Limit: 2%")
        print("  • Time Exit: 14 days")
        print("  • Trailing Stop: 3%")
        print("  • Max Drawdown Stop: 5%")
        print("  • Position Size Limit: 5% max")
        print("\n📊 SIGNAL FILTERS ACTIVE:")
        print("  • Trend Filter: 50-day SMA")
        print("  • Volume Filter: 1.2x average")
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

                    position_info = get_position(SYMBOL, TRADING_CLIENT)
                    # Position details already printed in get_position()

                    # Check all risk controls (daily limit, trailing stop, time exit)
                    current_price = position_info.get("current_price", 0)
                    should_stop_risk, risk_reason = risk_controls.check_all_risks(
                        SYMBOL, position_info, current_price, account_info.get("equity", 0)
                    )
                    if should_stop_risk:
                        print(f"\n  🚨 RISK CONTROL: {risk_reason}")

                        # If it's a position-specific risk (trailing stop, time exit), close the position
                        position_qty = position_info.get("qty", 0.0)
                        if position_qty != 0 and any(x in risk_reason for x in ["Trailing stop", "Time exit"]):
                            action = "sell"
                            qty = abs(position_qty)
                            print(f"  >>> RISK EXIT: {action.upper()} {qty:.6f} {SYMBOL} <<<")
                            order = execute_trade(SYMBOL, action, qty, TRADING_CLIENT, account_info)
                            if order:
                                risk_controls.record_position_exit(SYMBOL)
                                trade_log.append({
                                    "time": now,
                                    "action": action,
                                    "qty": qty,
                                    "price": current_price,
                                    "reason": risk_reason,
                                })

                        # If daily limit hit, stop all trading for the day
                        if "Daily loss limit" in risk_reason:
                            print("\n  ⛔ DAILY LOSS LIMIT REACHED - TRADING HALTED")
                            import time
                            time.sleep(CHECK_INTERVAL)
                            continue

                    signal_info = analyze_live_signal(fgi_df)
                    if signal_info and "indicators" in signal_info:
                        ind = signal_info["indicators"]
                        print(f"  ETH: ${ind.get('price', 0):,.2f}")
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

                        # CRITICAL: Check stop losses FIRST if holding a position
                        position_qty = position_info.get("qty", 0.0)
                        if position_qty != 0:
                            should_exit, exit_reason = check_stop_loss(position_info, signal_info, max_drawdown=0.05)
                            if should_exit:
                                print(f"\n  🚨 STOP LOSS TRIGGERED: {exit_reason}")
                                # Force sell the entire position
                                action = "sell"
                                qty = abs(position_qty)
                                print(f"  >>> EMERGENCY EXIT: {action.upper()} {qty:.6f} {SYMBOL} <<<")
                                order = execute_trade(SYMBOL, action, qty, TRADING_CLIENT, account_info)
                                order_id = order.id if order else None
                                if order:
                                    log_trade(signal_info, action, qty, order_id)
                                    trade_log.append(
                                        {
                                            "time": now,
                                            "action": action,
                                            "qty": qty,
                                            "price": ind.get("price", 0),
                                            "reason": exit_reason,
                                        }
                                    )
                                # Skip normal trading logic after stop loss
                                import time
                                print(f"\nSleeping {CHECK_INTERVAL} seconds...")
                                time.sleep(CHECK_INTERVAL)
                                continue

                        # Normal trading logic
                        action, qty = should_trade_with_position_limit(
                            signal_info,
                            position_info,
                            is_live=True,
                            account_info=account_info,
                            risk_controls=risk_controls,
                        )

                        if action != "hold":
                            print(
                                f"\n  >>> Executing {action.upper()} {qty:.6f} {SYMBOL} <<<"
                            )
                            order = execute_trade(SYMBOL, action, qty, TRADING_CLIENT, account_info)
                            order_id = order.id if order else None
                            if order:  # Only log successful trades
                                log_trade(signal_info, action, qty, order_id)
                                # Record position entry/exit for risk controls
                                if action == "buy":
                                    risk_controls.record_position_entry(SYMBOL, current_price)
                                elif action == "sell":
                                    risk_controls.record_position_exit(SYMBOL)
                            trade_log.append(
                                {
                                    "time": now,
                                    "action": action,
                                    "qty": qty,
                                    "price": ind.get("price", 0),
                                }
                            )
                        else:
                            if position_qty > 0:
                                print("  No trade: Holding long position")
                            else:
                                print("  No trade: Waiting for BUY signal")

                        # Show risk control status
                        risk_status = risk_controls.get_status_summary()
                        if risk_status["daily_limits"]["num_trades"] > 0:
                            daily = risk_status["daily_limits"]
                            print(f"  Daily P&L: {daily['daily_pnl_pct']:.2%} (${daily['daily_pnl_$']:.2f}) | Trades: {daily['num_trades']} | Loss limit remaining: {daily['remaining_loss_limit']:.2%}")

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
