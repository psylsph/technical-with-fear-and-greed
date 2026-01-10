"""
API Circuit Breaker: Stop trading if API failure rate exceeds threshold.
Implements reliability protection for external API calls.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from ..config import PROJECT_ROOT


class APICircuitBreaker:
    """Track API call success/failure and stop trading if failure rate too high."""

    def __init__(self, failure_threshold: float = 0.20, window_minutes: int = 10):
        """
        Args:
            failure_threshold: Maximum allowed failure rate (default 20%)
            window_minutes: Time window to track failures (default 10 minutes)
        """
        self.failure_threshold = failure_threshold
        self.window_minutes = window_minutes
        self.state_file = os.path.join(PROJECT_ROOT, "api_circuit_breaker_state.json")
        self.state = self._load_state()

        self.is_tripped = self.state.get("is_tripped", False)
        self.trip_reason = self.state.get("trip_reason", None)
        self.trip_time = self.state.get("trip_time", None)

    def _load_state(self) -> dict:
        """Load circuit breaker state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "is_tripped": False,
            "trip_reason": None,
            "trip_time": None,
            "api_calls": [],
        }

    def _save_state(self):
        """Save circuit breaker state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_api_call(self, success: bool, api_name: str = "unknown", error: str = None):
        """
        Record an API call result.

        Args:
            success: Whether the API call was successful
            api_name: Name of the API being called
            error: Error message if failed
        """
        now = datetime.now()

        # Clean old calls outside the window
        cutoff_time = now - timedelta(minutes=self.window_minutes)
        self.state["api_calls"] = [
            call
            for call in self.state.get("api_calls", [])
            if datetime.fromisoformat(call["timestamp"]) > cutoff_time
        ]

        # Record this call
        self.state["api_calls"].append(
            {
                "timestamp": now.isoformat(),
                "success": success,
                "api_name": api_name,
                "error": error,
            }
        )

        # Check if we should trip the breaker
        self._check_failure_rate()
        self._save_state()

    def _check_failure_rate(self):
        """Check if failure rate exceeds threshold and trip if needed."""
        api_calls = self.state.get("api_calls", [])

        if len(api_calls) < 5:  # Need at least 5 calls to make a decision
            return

        total_calls = len(api_calls)
        failed_calls = sum(1 for call in api_calls if not call["success"])
        failure_rate = failed_calls / total_calls

        if failure_rate > self.failure_threshold:
            self.is_tripped = True
            self.state["is_tripped"] = True
            self.state["trip_reason"] = (
                f"API failure rate {failure_rate:.1%} exceeds threshold {self.failure_threshold:.1%} "
                f"({failed_calls}/{total_calls} calls failed in last {self.window_minutes} minutes)"
            )
            self.state["trip_time"] = datetime.now().isoformat()
            self.trip_reason = self.state["trip_reason"]
            self.trip_time = self.state["trip_time"]

    def is_allowed(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on circuit breaker state.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if self.is_tripped:
            return False, self.trip_reason or "Circuit breaker is tripped"

        # Auto-reset after 30 minutes if failure rate improved
        if self.state.get("trip_time"):
            trip_time = datetime.fromisoformat(self.state["trip_time"])
            if datetime.now() - trip_time > timedelta(minutes=30):
                # Check if recent failure rate is acceptable
                cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
                recent_calls = [
                    call
                    for call in self.state.get("api_calls", [])
                    if datetime.fromisoformat(call["timestamp"]) > cutoff_time
                ]

                if len(recent_calls) >= 5:
                    failed_calls = sum(1 for call in recent_calls if not call["success"])
                    failure_rate = failed_calls / len(recent_calls)

                    if failure_rate <= self.failure_threshold:
                        # Reset the breaker
                        self.is_tripped = False
                        self.state["is_tripped"] = False
                        self.state["trip_reason"] = None
                        self.state["trip_time"] = None
                        self.trip_reason = None
                        self.trip_time = None
                        self._save_state()
                        return True, None

        return True, None

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        api_calls = self.state.get("api_calls", [])

        # Calculate current failure rate
        total_calls = len(api_calls)
        failed_calls = sum(1 for call in api_calls if not call["success"])
        failure_rate = failed_calls / total_calls if total_calls > 0 else 0.0

        # Count failures by API
        api_failures = {}
        for call in api_calls:
            api_name = call["api_name"]
            if api_name not in api_failures:
                api_failures[api_name] = {"total": 0, "failed": 0}
            api_failures[api_name]["total"] += 1
            if not call["success"]:
                api_failures[api_name]["failed"] += 1

        return {
            "is_tripped": self.is_tripped,
            "trip_reason": self.trip_reason,
            "trip_time": self.trip_time,
            "failure_rate": failure_rate,
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "threshold": self.failure_threshold,
            "window_minutes": self.window_minutes,
            "api_failures": api_failures,
        }

    def reset(self):
        """Manually reset the circuit breaker."""
        self.is_tripped = False
        self.state["is_tripped"] = False
        self.state["trip_reason"] = None
        self.state["trip_time"] = None
        self.trip_reason = None
        self.trip_time = None
        self._save_state()
