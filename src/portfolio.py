"""
Portfolio management and trading simulation.
"""

import json
import os
from datetime import datetime

from .config import INITIAL_CAPITAL, MAKER_FEE, TAKER_FEE, TEST_STATE_FILE


def calculate_portfolio_var(
    positions: dict,
    confidence_level: float = 0.95,
    daily_volatility: float = 0.02
) -> dict:
    """Calculate Value at Risk (VaR) for the portfolio.

    Args:
        positions: Dict of position values by asset
        confidence_level: Confidence level for VaR (default 95%)
        daily_volatility: Daily volatility assumption (default 2%)

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
            "var_pct": 0.0,
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
            "var_pct": 0.0,
        }

    # Calculate VaR using normal distribution
    z_score = norm.ppf(1 - confidence_level)
    var_daily = total_portfolio_value * daily_volatility * abs(z_score)
    var_pct = (var_daily / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0.0

    return {
        "daily_var": var_daily,
        "weekly_var": var_daily * np.sqrt(5),
        "monthly_var": var_daily * np.sqrt(21),
        "confidence_level": confidence_level,
        "confidence_level_pct": confidence_level * 100,
        "portfolio_value": total_portfolio_value,
        "daily_volatility": daily_volatility,
        "var_pct": var_pct,
        "z_score": z_score,
    }


def calculate_var_multiple_levels(
    positions: dict,
    daily_volatility: float = 0.02
) -> dict:
    """Calculate VaR at multiple confidence levels (95%, 99%, 99.9%).

    Args:
        positions: Dict of position values by asset
        daily_volatility: Daily volatility assumption (default 2%)

    Returns:
        Dict with VaR calculations at different confidence levels
    """
    confidence_levels = [0.95, 0.99, 0.999]
    var_results = {}

    for cl in confidence_levels:
        var_data = calculate_portfolio_var(positions, cl, daily_volatility)
        level_key = f"{int(cl * 100)}%"
        var_results[level_key] = {
            "daily_var": var_data["daily_var"],
            "weekly_var": var_data["weekly_var"],
            "monthly_var": var_data["monthly_var"],
            "var_pct": var_data["var_pct"],
            "z_score": var_data["z_score"],
        }

    # Add portfolio value
    if positions:
        var_results["portfolio_value"] = sum(abs(value) for value in positions.values())
    else:
        var_results["portfolio_value"] = 0.0

    var_results["daily_volatility"] = daily_volatility

    return var_results


def load_test_state() -> dict:
    """Load test portfolio state from file."""
    if os.path.exists(TEST_STATE_FILE):
        try:
            with open(TEST_STATE_FILE) as f:
                state = json.load(f)
                # Handle backward compatibility: convert old state files
                if "eth_held" not in state and "btc_held" in state:
                    state["eth_held"] = state["btc_held"]
                if "entry_price" not in state:
                    state["entry_price"] = 0.0
                if "initialized" not in state:
                    state["initialized"] = True
                if "trades" not in state:
                    state["trades"] = []
                return state
        except Exception as e:
            print(f"Error loading test state: {e}")
    return {
        "cash": INITIAL_CAPITAL,
        "eth_held": 0.0,
        "entry_price": 0.0,  # Track average entry price
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

    # Handle backward compatibility: old state files used "btc_held", new ones use "eth_held"
    # Also handle missing "entry_price" for very old state files
    if "eth_held" not in state:
        state["eth_held"] = state.pop("btc_held", 0.0)  # Convert old key to new
    if "entry_price" not in state:
        state["entry_price"] = 0.0
    if "initialized" not in state:
        state["initialized"] = True
    if "trades" not in state:
        state["trades"] = []

    if side.lower() == "buy":
        cost = price * qty * (1 + taker_fee)
        if state["cash"] >= cost:
            state["cash"] -= cost
            state["eth_held"] += qty
            # Update average entry price
            if state["eth_held"] > 0:
                total_cost = (state["entry_price"] * (state["eth_held"] - qty)) + cost
                state["entry_price"] = total_cost / state["eth_held"]
            else:
                state["entry_price"] = price
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
                f"  SIMULATED BUY: {qty:.6f} @ ${price:,.2f} (avg entry: ${state['entry_price']:.2f}, fee: ${cost - price * qty:.4f})"
            )
        else:
            print(f"  SIMULATED BUY FAILED: Insufficient cash ${state['cash']:.2f}")

    elif side.lower() == "sell":
        if state["eth_held"] >= qty:
            proceeds = price * qty * (1 - maker_fee)
            state["cash"] += proceeds
            state["eth_held"] -= qty
            # Reset entry price if position closed
            if state["eth_held"] <= 0.000001:  # Essentially zero
                state["entry_price"] = 0.0
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
    # Handle backward compatibility for old state files
    if "eth_held" not in state:
        state["eth_held"] = state.pop("btc_held", 0.0)
    return state["cash"] + state["eth_held"] * current_price
