"""
Advanced Execution Features: Entry Confirmation and Partial Exits.
Improves trade timing and profit taking.
"""

import json
import os
from datetime import datetime
from typing import Optional, Tuple

from ..config import PROJECT_ROOT


class EntryConfirmation:
    """Wait for confirmation after signal before entering trade."""

    def __init__(
        self,
        confirmation_bars: int = 2,
        confirmation_threshold: float = 0.01
    ):
        """
        Args:
            confirmation_bars: Number of bars to wait for confirmation (default 2)
            confirmation_threshold: Price movement threshold to confirm (default 1%)
        """
        self.confirmation_bars = confirmation_bars
        self.confirmation_threshold = confirmation_threshold

        self.pending_signals_file = os.path.join(PROJECT_ROOT, "pending_signals.json")
        self.pending_signals = self._load_state()

    def _load_state(self) -> dict:
        """Load pending signals from file."""
        if os.path.exists(self.pending_signals_file):
            try:
                with open(self.pending_signals_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self):
        """Save pending signals to file."""
        with open(self.pending_signals_file, "w") as f:
            json.dump(self.pending_signals, f, indent=2)

    def record_signal(
        self,
        signal: str,
        price: float,
        timestamp: str = None
    ):
        """
        Record a new trading signal.

        Args:
            signal: "buy" or "sell"
            price: Price at signal
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        signal_key = f"{signal}_{timestamp}"

        self.pending_signals[signal_key] = {
            "signal": signal,
            "entry_price": price,
            "timestamp": timestamp,
            "bars_waited": 0,
            "confirmed": False,
        }

        self._save_state()

    def check_confirmation(
        self,
        signal: str,
        entry_price: float,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if signal should be confirmed for entry.

        Args:
            signal: "buy" or "sell"
            entry_price: Original signal price
            current_price: Current price

        Returns:
            Tuple of (confirmed: bool, reason: str)
        """
        bars_waited = 0

        # Check if we have a matching pending signal
        for key, pending in list(self.pending_signals.items()):
            if pending["signal"] == signal and not pending["confirmed"]:
                bars_waited = pending["bars_waited"] + 1

                if bars_waited > self.confirmation_bars:
                    # Signal expired
                    del self.pending_signals[key]
                    self._save_state()
                    return False, f"Signal expired after {self.confirmation_bars} bars without confirmation"

                # Check confirmation based on signal type
                if signal == "buy":
                    # For buy: price should not fall significantly
                    price_change = (current_price - entry_price) / entry_price

                    if price_change >= 0:  # Price went up or stayed
                        pending["bars_waited"] = bars_waited
                        pending["confirmed"] = True
                        self._save_state()
                        return True, f"Buy confirmed after {bars_waited} bars (price +{price_change:.2%})"
                    elif price_change > -self.confirmation_threshold:
                        # Price down but within threshold - wait more
                        pending["bars_waited"] = bars_waited
                        self._save_state()
                        return False, f"Waiting for buy confirmation ({bars_waited}/{self.confirmation_bars} bars)"
                    else:
                        # Price down significantly - cancel signal
                        del self.pending_signals[key]
                        self._save_state()
                        return False, f"Buy signal cancelled (price fell {price_change:.2%})"

                elif signal == "sell":
                    # For sell: price should not rise significantly
                    price_change = (entry_price - current_price) / entry_price

                    if price_change >= 0:  # Price went down or stayed
                        pending["bars_waited"] = bars_waited
                        pending["confirmed"] = True
                        self._save_state()
                        return True, f"Sell confirmed after {bars_waited} bars (price {-price_change:.2%})"
                    elif price_change > -self.confirmation_threshold:
                        # Price up but within threshold - wait more
                        pending["bars_waited"] = bars_waited
                        self._save_state()
                        return False, f"Waiting for sell confirmation ({bars_waited}/{self.confirmation_bars} bars)"
                    else:
                        # Price up significantly - cancel signal
                        del self.pending_signals[key]
                        self._save_state()
                        return False, f"Sell signal cancelled (price rose {-price_change:.2%})"

        # No matching pending signal - this is a new signal
        self.record_signal(signal, entry_price)
        return False, f"Signal recorded, waiting {self.confirmation_bars} bars for confirmation"

    def clear_confirmed_signals(self):
        """Clear all confirmed signals (call after trade execution)."""
        keys_to_remove = [
            key for key, signal in self.pending_signals.items()
            if signal.get("confirmed", False)
        ]

        for key in keys_to_remove:
            del self.pending_signals[key]

        if keys_to_remove:
            self._save_state()

    def get_pending_count(self) -> int:
        """Get count of pending signals."""
        return len([
            s for s in self.pending_signals.values()
            if not s.get("confirmed", False)
        ])


class PartialExitManager:
    """Manage partial exits to lock in profits."""

    def __init__(
        self,
        profit_target_ratio: float = 2.0,
        partial_exit_pct: float = 0.5,
        trail_remaining: bool = True
    ):
        """
        Args:
            profit_target_ratio: Take partial profit at 2x risk (default 2.0)
            partial_exit_pct: Percentage to exit at target (default 50%)
            trail_remaining: Whether to trail stop on remaining position
        """
        self.profit_target_ratio = profit_target_ratio
        self.partial_exit_pct = partial_exit_pct
        self.trail_remaining = trail_remaining

        self.position_file = os.path.join(PROJECT_ROOT, "partial_exit_positions.json")
        self.positions = self._load_positions()

    def _load_positions(self) -> dict:
        """Load position tracking from file."""
        if os.path.exists(self.position_file):
            try:
                with open(self.position_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_positions(self):
        """Save position tracking to file."""
        with open(self.position_file, "w") as f:
            json.dump(self.positions, f, indent=2)

    def open_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_loss_price: float,
        side: str = "long"
    ):
        """
        Record a new position for partial exit management.

        Args:
            symbol: Trading symbol
            qty: Position quantity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            side: "long" or "short"
        """
        risk_per_share = abs(entry_price - stop_loss_price)
        profit_target_price = entry_price + (risk_per_share * self.profit_target_ratio)

        self.positions[symbol] = {
            "qty": qty,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "profit_target_price": profit_target_price,
            "partial_exit_qty": qty * self.partial_exit_pct,
            "remaining_qty": qty * (1 - self.partial_exit_pct),
            "partial_exit_done": False,
            "partial_exit_price": None,
            "highest_price": entry_price,
            "trail_stop_price": stop_loss_price,
            "side": side,
            "open_time": datetime.now().isoformat(),
        }

        self._save_positions()

    def check_partial_exit(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, float, str]:
        """
        Check if partial exit should be executed.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Tuple of (should_exit: bool, exit_qty: float, reason: str)
        """
        if symbol not in self.positions:
            return False, 0.0, "Position not tracked"

        position = self.positions[symbol]
        side = position["side"]

        # Update highest price for trailing stop
        if side == "long" and current_price > position["highest_price"]:
            position["highest_price"] = current_price
            # Update trailing stop for remaining position
            if position["partial_exit_done"]:
                risk = abs(position["entry_price"] - position["stop_loss_price"])
                position["trail_stop_price"] = current_price - (risk * 0.5)  # Trail at 50% of original risk

        # Check partial exit condition
        if not position["partial_exit_done"]:
            target_price = position["profit_target_price"]

            if side == "long" and current_price >= target_price:
                return True, position["partial_exit_qty"], (
                    f"Partial exit target reached: ${current_price:.2f} >= ${target_price:.2f} "
                    f"(exit {position['partial_exit_qty']:.6f} of {position['qty']:.6f})"
                )
            elif side == "short" and current_price <= target_price:
                return True, position["partial_exit_qty"], (
                    f"Partial exit target reached: ${current_price:.2f} <= ${target_price:.2f} "
                    f"(exit {position['partial_exit_qty']:.6f} of {position['qty']:.6f})"
                )

        # Check trailing stop for remaining position
        if position["partial_exit_done"] and self.trail_remaining:
            trail_stop = position["trail_stop_price"]

            if side == "long" and current_price <= trail_stop:
                return True, position["remaining_qty"], (
                    f"Trailing stop hit: ${current_price:.2f} <= ${trail_stop:.2f} "
                    f"(exit remaining {position['remaining_qty']:.6f})"
                )
            elif side == "short" and current_price >= trail_stop:
                return True, position["remaining_qty"], (
                    f"Trailing stop hit: ${current_price:.2f} >= ${trail_stop:.2f} "
                    f"(exit remaining {position['remaining_qty']:.6f})"
                )

        return False, 0.0, "No exit condition met"

    def execute_partial_exit(
        self,
        symbol: str,
        exit_price: float
    ):
        """
        Mark partial exit as executed.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
        """
        if symbol in self.positions:
            position = self.positions[symbol]

            if not position["partial_exit_done"]:
                position["partial_exit_done"] = True
                position["partial_exit_price"] = exit_price
                position["exit_time"] = datetime.now().isoformat()

                # Calculate realized profit on partial exit
                if position["side"] == "long":
                    profit = (exit_price - position["entry_price"]) * position["partial_exit_qty"]
                else:
                    profit = (position["entry_price"] - exit_price) * position["partial_exit_qty"]

                position["partial_exit_profit"] = profit

                self._save_positions()

                return profit

        return 0.0

    def close_position(self, symbol: str):
        """Remove position from tracking after full exit."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._save_positions()

    def get_position_status(self, symbol: str) -> Optional[dict]:
        """Get status of a tracked position."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> dict:
        """Get all tracked positions."""
        return self.positions.copy()
