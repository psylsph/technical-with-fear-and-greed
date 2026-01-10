"""
Automatic Recovery: Auto-restart after crashes with state preservation.
Implements crash detection, state backup, and recovery mechanisms.
"""

import json
import os
import signal
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..config import PROJECT_ROOT


class CrashRecoveryManager:
    """Manage crash recovery and state preservation for trading system."""

    def __init__(self, max_restarts: int = 3, restart_window_minutes: int = 10):
        """
        Args:
            max_restarts: Maximum number of restarts allowed within window
            restart_window_minutes: Time window for restart counting (default 10 min)
        """
        self.max_restarts = max_restarts
        self.restart_window_minutes = restart_window_minutes
        self.state_dir = os.path.join(PROJECT_ROOT, "recovery_state")
        os.makedirs(self.state_dir, exist_ok=True)

        self.recovery_log_file = os.path.join(self.state_dir, "recovery_log.json")
        self.checkpoint_file = os.path.join(self.state_dir, "last_checkpoint.json")
        self.crash_count_file = os.path.join(self.state_dir, "crash_count.json")

        self._setup_signal_handlers()

    def _load_json(self, filepath: str, default: Any = None) -> Any:
        """Load JSON file with default value."""
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception:
                pass
        return default if default is not None else {}

    def _save_json(self, filepath: str, data: dict):
        """Save data to JSON file."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        self.save_checkpoint("shutdown", {"signal": signum})
        sys.exit(0)

    def save_checkpoint(self, name: str, state_data: dict = None) -> bool:
        """
        Save a checkpoint with current system state.

        Args:
            name: Checkpoint name (e.g., "daily_init", "trade_complete")
            state_data: Optional state data to save

        Returns:
            True if checkpoint saved successfully
        """
        try:
            checkpoint = {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "state_data": state_data or {},
            }

            self._save_json(self.checkpoint_file, checkpoint)

            # Log to recovery log
            log = self._load_json(self.recovery_log_file, [])
            log.append(checkpoint)

            # Keep only last 100 checkpoints in log
            if len(log) > 100:
                log = log[-100:]

            self._save_json(self.recovery_log_file, log)

            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False

    def load_last_checkpoint(self) -> Optional[dict]:
        """
        Load the last saved checkpoint.

        Returns:
            Checkpoint dict or None if no checkpoint exists
        """
        return self._load_json(self.checkpoint_file)

    def get_recovery_log(self, limit: int = 10) -> list:
        """
        Get recent recovery log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent log entries
        """
        log = self._load_json(self.recovery_log_file, [])
        return log[-limit:] if log else []

    def increment_crash_count(self) -> dict:
        """
        Increment crash count and check if restart is allowed.

        Returns:
            Dict with crash count info and whether restart is allowed
        """
        crash_data = self._load_json(self.crash_count_file, {"crashes": []})

        now = datetime.now()
        window_start = now - timedelta(minutes=self.restart_window_minutes)

        # Filter crashes within the time window
        recent_crashes = [
            c for c in crash_data.get("crashes", [])
            if datetime.fromisoformat(c["timestamp"]) > window_start
        ]

        # Add current crash
        crash_info = {
            "timestamp": now.isoformat(),
            "traceback": traceback.format_exc(),
        }
        recent_crashes.append(crash_info)

        crash_data["crashes"] = recent_crashes
        self._save_json(self.crash_count_file, crash_data)

        can_restart = len(recent_crashes) <= self.max_restarts

        return {
            "crash_count": len(recent_crashes),
            "max_restarts": self.max_restarts,
            "can_restart": can_restart,
            "window_minutes": self.restart_window_minutes,
            "recent_crashes": recent_crashes,
        }

    def should_restart_after_crash(self) -> tuple[bool, str]:
        """
        Check if system should restart after a crash.

        Returns:
            Tuple of (should_restart: bool, reason: str)
        """
        crash_info = self.increment_crash_count()

        if not crash_info["can_restart"]:
            return False, (
                f"Too many crashes ({crash_info['crash_count']}) "
                f"within {crash_info['window_minutes']} minutes. "
                f"Manual intervention required."
            )

        return True, f"Crash {crash_info['crash_count']}/{crash_info['max_restarts']} - Restarting"

    def reset_crash_count(self):
        """Reset the crash count (call after successful startup)."""
        self._save_json(self.crash_count_file, {"crashes": []})

    def save_position_state(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        side: str = "long"
    ):
        """
        Save position state for recovery.

        Args:
            symbol: Trading symbol
            qty: Position quantity
            entry_price: Entry price
            side: Position side (long/short)
        """
        position_file = os.path.join(self.state_dir, "position_state.json")

        position_data = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "side": side,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_json(position_file, position_data)

    def load_position_state(self) -> Optional[dict]:
        """
        Load saved position state.

        Returns:
            Position data dict or None if no saved position
        """
        position_file = os.path.join(self.state_dir, "position_state.json")
        return self._load_json(position_file)

    def clear_position_state(self):
        """Clear saved position state (after position is closed)."""
        position_file = os.path.join(self.state_dir, "position_state.json")
        if os.path.exists(position_file):
            os.remove(position_file)

    def get_recovery_status(self) -> dict:
        """
        Get overall recovery status.

        Returns:
            Dict with recovery system status
        """
        crash_data = self._load_json(self.crash_count_file, {"crashes": []})
        checkpoint = self.load_last_checkpoint()

        now = datetime.now()
        window_start = now - timedelta(minutes=self.restart_window_minutes)

        recent_crashes = [
            c for c in crash_data.get("crashes", [])
            if datetime.fromisoformat(c["timestamp"]) > window_start
        ]

        return {
            "recent_crash_count": len(recent_crashes),
            "max_restarts": self.max_restarts,
            "can_restart": len(recent_crashes) <= self.max_restarts,
            "last_checkpoint": checkpoint,
            "last_checkpoint_time": checkpoint.get("timestamp") if checkpoint else None,
            "recovery_log_entries": len(self._load_json(self.recovery_log_file, [])),
        }


def run_with_recovery(
    main_func: Callable,
    checkpoint_name: str = "startup",
    state_data: dict = None,
    recovery_manager: CrashRecoveryManager = None,
):
    """
    Run a function with automatic crash recovery.

    Args:
        main_func: Main function to run
        checkpoint_name: Name for initial checkpoint
        state_data: Initial state data to save
        recovery_manager: Optional recovery manager instance

    Returns:
        Result from main function or None if too many crashes
    """
    if recovery_manager is None:
        recovery_manager = CrashRecoveryManager()

    # Save initial checkpoint
    recovery_manager.save_checkpoint(checkpoint_name, state_data)

    try:
        result = main_func()

        # If successful, reset crash count
        recovery_manager.reset_crash_count()
        recovery_manager.save_checkpoint("completion", {"success": True})

        return result

    except Exception as e:
        # Check if we should restart
        should_restart, reason = recovery_manager.should_restart_after_crash()

        recovery_manager.save_checkpoint("crash", {
            "error": str(e),
            "should_restart": should_restart,
            "reason": reason,
        })

        print(f"ERROR: {e}")
        print(f"Crash recovery: {reason}")

        if should_restart:
            print("Attempting restart...")
            # Recursive restart with same function
            return run_with_recovery(main_func, f"restart_{checkpoint_name}", state_data, recovery_manager)
        else:
            print("Manual intervention required. Exiting.")
            sys.exit(1)
