"""
Risk Control Enhancements for Live Trading
Implements: Daily loss limits, time-based exits, trailing stops
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

from ..config import PROJECT_ROOT


class DailyLossLimit:
    """Track daily P&L and stop trading if loss exceeds limit."""

    def __init__(self, daily_loss_limit: float = 0.02):
        """
        Args:
            daily_loss_limit: Maximum daily loss as percentage (default 2%)
        """
        self.daily_loss_limit = daily_loss_limit
        self.state_file = os.path.join(PROJECT_ROOT, "daily_pnl_state.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load or initialize daily P&L state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    # Check if state is from today
                    if state.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        return state
            except Exception:
                pass

        # New day - reset state
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "daily_pnl": 0.0,
            "initial_equity": None,
            "trades_today": [],
            "trading_stopped": False,
            "stop_reason": None,
        }

    def _save_state(self):
        """Save current state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def check_daily_limit(self, current_equity: float) -> Tuple[bool, Optional[str]]:
        """
        Check if daily loss limit has been exceeded.

        Args:
            current_equity: Current portfolio equity

        Returns:
            Tuple of (should_stop_trading: bool, reason: str)
        """
        if self.state["trading_stopped"]:
            return True, self.state["stop_reason"]

        # Initialize equity on first call
        if self.state["initial_equity"] is None:
            self.state["initial_equity"] = current_equity
            self._save_state()
            return False, None

        # Calculate daily P&L percentage
        initial_equity = self.state["initial_equity"]
        daily_pnl_pct = (current_equity - initial_equity) / initial_equity
        self.state["daily_pnl"] = daily_pnl_pct

        # Check if loss limit exceeded
        if daily_pnl_pct <= -self.daily_loss_limit:
            self.state["trading_stopped"] = True
            self.state["stop_reason"] = f"Daily loss limit reached: {daily_pnl_pct:.2%} (limit: {self.daily_loss_limit:.2%})"
            self._save_state()
            return True, self.state["stop_reason"]

        self._save_state()
        return False, None

    def record_trade(self, pnl: float):
        """Record a trade in today's log."""
        self.state["trades_today"].append({
            "time": datetime.now().isoformat(),
            "pnl": pnl,
        })
        self._save_state()

    def get_daily_summary(self) -> dict:
        """Get summary of today's trading."""
        return {
            "date": self.state["date"],
            "daily_pnl_pct": self.state["daily_pnl"],
            "daily_pnl_$": (self.state["daily_pnl"] * self.state["initial_equity"])
                          if self.state["initial_equity"] else 0,
            "num_trades": len(self.state["trades_today"]),
            "trading_stopped": self.state["trading_stopped"],
            "remaining_loss_limit": self.daily_loss_limit + self.state["daily_pnl"]
                                    if self.state["daily_pnl"] < 0 else self.daily_loss_limit,
        }

    def reset_new_day(self):
        """Reset for a new trading day."""
        self.state = {
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "daily_pnl": 0.0,
            "initial_equity": None,
            "trades_today": [],
            "trading_stopped": False,
            "stop_reason": None,
        }
        self._save_state()


class TimeBasedExit:
    """Track position entry time and exit if held too long without profit."""

    def __init__(self, max_hold_days: int = 14):
        """
        Args:
            max_hold_days: Maximum days to hold without profit (default 14)
        """
        self.max_hold_days = max_hold_days
        self.state_file = os.path.join(PROJECT_ROOT, "position_tracking.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load position tracking state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"positions": {}}

    def _save_state(self):
        """Save position tracking state."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_entry(self, symbol: str, entry_price: float, entry_time: str = None):
        """Record a new position entry."""
        if entry_time is None:
            entry_time = datetime.now().isoformat()

        # Initialize position if not exists
        if symbol not in self.state["positions"]:
            self.state["positions"][symbol] = {}

        # Set entry information
        self.state["positions"][symbol]["entry_price"] = entry_price
        self.state["positions"][symbol]["entry_time"] = entry_time

        # Initialize highest price if not set
        if "highest_price" not in self.state["positions"][symbol]:
            self.state["positions"][symbol]["highest_price"] = entry_price
            self.state["positions"][symbol]["highest_price_time"] = entry_time

        self._save_state()

    def record_exit(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.state["positions"]:
            del self.state["positions"][symbol]
            self._save_state()

    def update_highest_price(self, symbol: str, current_price: float):
        """Update highest price for trailing stop calculation."""
        if symbol in self.state["positions"]:
            pos = self.state["positions"][symbol]
            if "highest_price" not in pos or current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
                pos["highest_price_time"] = datetime.now().isoformat()
                self._save_state()

    def check_time_exit(self, symbol: str, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited due to time limit.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        if symbol not in self.state["positions"]:
            return False, None

        pos = self.state["positions"][symbol]
        entry_time = datetime.fromisoformat(pos["entry_time"])
        days_held = (datetime.now() - entry_time).days

        # Check if in profit
        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price

        if days_held >= self.max_hold_days and pnl_pct < 0:
            return True, f"Time exit: Held {days_held} days at {pnl_pct:.2%} loss (limit: {self.max_hold_days} days)"

        return False, None


class TrailingStop:
    """Implement trailing stop to lock in profits."""

    def __init__(self, trailing_percent: float = 0.03):
        """
        Args:
            trailing_percent: Trailing stop percentage (default 3%)
        """
        self.trailing_percent = trailing_percent
        self.state_file = os.path.join(PROJECT_ROOT, "position_tracking.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load position tracking state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"positions": {}}

    def _save_state(self):
        """Save position tracking state."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update_highest_price(self, symbol: str, current_price: float):
        """Update highest price for trailing stop."""
        if symbol not in self.state["positions"]:
            self.state["positions"][symbol] = {
                "highest_price": current_price,
                "highest_price_time": datetime.now().isoformat(),
            }
        elif current_price > self.state["positions"][symbol].get("highest_price", 0):
            self.state["positions"][symbol]["highest_price"] = current_price
            self.state["positions"][symbol]["highest_price_time"] = datetime.now().isoformat()

        self._save_state()

    def check_trailing_stop(self, symbol: str, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if trailing stop is triggered.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        if symbol not in self.state["positions"]:
            return False, None

        highest_price = self.state["positions"][symbol]["highest_price"]
        trailing_stop_price = highest_price * (1 - self.trailing_percent)

        if current_price < trailing_stop_price:
            pnl_pct = (current_price - highest_price) / highest_price
            return True, f"Trailing stop: Price ${current_price:.2f} below stop ${trailing_stop_price:.2f} (highest was ${highest_price:.2f}, locked in {pnl_pct:.2%})"

        return False, None

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Get current trailing stop price for a position."""
        if symbol not in self.state["positions"]:
            return None
        highest = self.state["positions"][symbol]["highest_price"]
        return highest * (1 - self.trailing_percent)


class PositionSizeLimit:
    """Enforce maximum position size as percentage of portfolio."""

    def __init__(self, max_position_pct: float = 0.05):
        """
        Args:
            max_position_pct: Maximum position size as percentage of portfolio (default 5%)
        """
        self.max_position_pct = max_position_pct

    def check_position_size(self, proposed_qty: float, current_price: float, portfolio_equity: float) -> Tuple[bool, float, Optional[str]]:
        """
        Check if proposed position size exceeds limit.

        Args:
            proposed_qty: Proposed position quantity
            current_price: Current market price
            portfolio_equity: Total portfolio equity

        Returns:
            Tuple of (allowed: bool, adjusted_qty: float, reason: str)
        """
        # Calculate proposed position value
        proposed_value = proposed_qty * current_price
        proposed_pct = proposed_value / portfolio_equity if portfolio_equity > 0 else 0

        # If within limit, allow as-is
        if proposed_pct <= self.max_position_pct:
            return True, proposed_qty, None

        # Exceeds limit - calculate adjusted quantity
        max_allowed_value = portfolio_equity * self.max_position_pct
        adjusted_qty = (max_allowed_value / current_price)
        adjusted_qty = round(adjusted_qty, 6)

        return False, adjusted_qty, (
            f"Position size limit: Proposed {proposed_pct:.2%} exceeds {self.max_position_pct:.2%} maximum. "
            f"Adjusting from {proposed_qty:.6f} to {adjusted_qty:.6f} shares (${max_allowed_value:.2f} value)"
        )

    def get_max_quantity(self, portfolio_equity: float, current_price: float) -> float:
        """Get maximum allowed quantity for a given equity and price."""
        if current_price <= 0:
            return 0.0
        max_value = portfolio_equity * self.max_position_pct
        return round(max_value / current_price, 6)


class RiskControls:
    """Main risk controls coordinator."""

    def __init__(self):
        self.daily_limit = DailyLossLimit(daily_loss_limit=0.02)
        self.time_exit = TimeBasedExit(max_hold_days=14)
        self.trailing_stop = TrailingStop(trailing_percent=0.03)
        self.position_limit = PositionSizeLimit(max_position_pct=0.05)

    def check_all_risks(self, symbol: str, position_info: dict, current_price: float, current_equity: float) -> Tuple[bool, Optional[str]]:
        """
        Check all risk controls.

        Args:
            symbol: Trading symbol
            position_info: Position info dict
            current_price: Current market price
            current_equity: Current portfolio equity

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        # Check daily loss limit
        should_stop, reason = self.daily_limit.check_daily_limit(current_equity)
        if should_stop:
            return True, reason

        # Only check position-specific risks if holding a position
        if position_info.get("qty", 0) != 0:
            # Update trailing stop highest price
            self.trailing_stop.update_highest_price(symbol, current_price)

            # Check trailing stop
            should_exit, reason = self.trailing_stop.check_trailing_stop(symbol, current_price)
            if should_exit:
                return True, reason

            # Check time-based exit
            should_exit, reason = self.time_exit.check_time_exit(symbol, current_price)
            if should_exit:
                return True, reason

        return False, None

    def record_position_entry(self, symbol: str, entry_price: float):
        """Record a new position entry."""
        self.time_exit.record_entry(symbol, entry_price)
        self.trailing_stop.update_highest_price(symbol, entry_price)

    def record_position_exit(self, symbol: str):
        """Record a position exit."""
        self.time_exit.record_exit(symbol)
        # Trailing stop state will be cleaned up naturally

    def check_position_size(self, proposed_qty: float, current_price: float, portfolio_equity: float) -> Tuple[bool, float, Optional[str]]:
        """
        Check if proposed position size exceeds limit.

        Args:
            proposed_qty: Proposed position quantity
            current_price: Current market price
            portfolio_equity: Total portfolio equity

        Returns:
            Tuple of (allowed: bool, adjusted_qty: float, reason: str)
        """
        return self.position_limit.check_position_size(proposed_qty, current_price, portfolio_equity)

    def get_max_position_quantity(self, portfolio_equity: float, current_price: float) -> float:
        """Get maximum allowed position quantity."""
        return self.position_limit.get_max_quantity(portfolio_equity, current_price)

    def get_status_summary(self) -> dict:
        """Get summary of all risk control status."""
        daily_summary = self.daily_limit.get_daily_summary()

        return {
            "daily_limits": daily_summary,
            "tracked_positions": len(self.time_exit.state.get("positions", {})),
            "trailing_stops_active": len(self.trailing_stop.state.get("positions", {})),
            "max_position_pct": self.position_limit.max_position_pct,
        }
