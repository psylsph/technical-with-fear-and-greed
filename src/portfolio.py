"""
Portfolio management and trading simulation.
"""

import json
import os
from datetime import datetime

from .config import INITIAL_CAPITAL, MAKER_FEE, TAKER_FEE, TEST_STATE_FILE


def calculate_portfolio_var(positions: dict, confidence_level: float = 0.95) -> dict:
    """Calculate Value at Risk (VaR) for the portfolio.

    Args:
        positions: Dict of position values by asset
        confidence_level: Confidence level for VaR (default 95%)

    Returns:
        Dict with VaR calculations for different time periods
    """
    import numpy as np
    from scipy.stats import norm

    if not positions:
        return {
            "daily_var": 0.0,
            "weekly_var": 0.0,
            "monthly_var": 0.0,
            "confidence_level": confidence_level,
            "portfolio_value": 0.0,
        }

    # Calculate total portfolio value
    total_portfolio_value = sum(abs(value) for value in positions.values())

    if total_portfolio_value == 0:
        return {
            "daily_var": 0.0,
            "weekly_var": 0.0,
            "monthly_var": 0.0,
            "confidence_level": confidence_level,
            "portfolio_value": 0.0,
        }

    # Assume 2% daily volatility for crypto (conservative estimate)
    # This could be enhanced with actual historical volatility
    daily_volatility = 0.02

    # Calculate VaR using normal distribution
    z_score = norm.ppf(1 - confidence_level)
    var_daily = total_portfolio_value * daily_volatility * abs(z_score)

    return {
        "daily_var": var_daily,
        "weekly_var": var_daily * np.sqrt(5),
        "monthly_var": var_daily * np.sqrt(21),
        "confidence_level": confidence_level,
        "portfolio_value": total_portfolio_value,
        "daily_volatility": daily_volatility,
    }


def load_test_state() -> dict:
    """Load test portfolio state from file."""
    if os.path.exists(TEST_STATE_FILE):
        try:
            with open(TEST_STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test state: {e}")
    return {
        "cash": INITIAL_CAPITAL,
        "eth_held": 0.0,
        "trades": [],
        "initialized": False,
    }


def save_test_state(state: dict) -> None:
    """Save test portfolio state to file."""
    try:
        with open(TEST_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving test state: {e}")


def simulate_trade(
    state: dict,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    fees: tuple = (MAKER_FEE, TAKER_FEE),
) -> dict:
    """Simulate a trade and update state. Returns updated state."""
    maker_fee, taker_fee = fees

    if side.lower() == "buy":
        cost = price * qty * (1 + taker_fee)
        if state["cash"] >= cost:
            state["cash"] -= cost
            state["eth_held"] += qty
            state["trades"].append(
                {
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": qty,
                    "price": price,
                    "fee": cost - price * qty,
                }
            )
            print(
                f"  SIMULATED BUY: {qty:.6f} @ ${price:,.2f} (fee: ${cost - price * qty:.4f})"
            )
        else:
            print(f"  SIMULATED BUY FAILED: Insufficient cash ${state['cash']:.2f}")

    elif side.lower() == "sell":
        if state["eth_held"] >= qty:
            proceeds = price * qty * (1 - maker_fee)
            state["cash"] += proceeds
            state["eth_held"] -= qty
            state["trades"].append(
                {
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": qty,
                    "price": price,
                    "fee": price * qty - proceeds,
                }
            )
            print(
                f"  SIMULATED SELL: {qty:.6f} @ ${price:,.2f} (fee: ${price * qty - proceeds:.4f})"
            )
        else:
            print(f"  SIMULATED SELL FAILED: Insufficient ETH {state['eth_held']:.6f}")

    return state


def get_test_portfolio_value(state: dict, current_price: float) -> float:
    """Calculate total portfolio value in test mode."""
    return state["cash"] + state["eth_held"] * current_price
