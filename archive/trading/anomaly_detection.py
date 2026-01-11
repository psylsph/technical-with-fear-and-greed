"""
Anomaly Detection: Alert on unusual trading patterns.
Detects statistical anomalies in trading behavior and market data.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from enum import Enum
from collections import deque

import numpy as np

from ..config import PROJECT_ROOT


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    FREQUENT_TRADING = "frequent_trading"
    LARGE_LOSSES = "large_losses"
    UNUSUAL_PNL = "unusual_pnl"
    POSITION_DRIFT = "position_drift"
    GAP_ANOMALY = "gap_anomaly"
    VOLATILITY_SPIKE = "volatility_spike"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyDetector:
    """Detect statistical anomalies in trading patterns."""

    def __init__(
        self,
        price_spike_threshold: float = 3.0,  # 3 sigma
        volume_spike_threshold: float = 2.5,  # 2.5x average
        volatility_spike_threshold: float = 2.0,  # 2x average
        min_history: int = 20,  # Minimum data points for analysis
        alert_window_minutes: int = 60,  # Time window for alert grouping
    ):
        """
        Args:
            price_spike_threshold: Standard deviations for price anomaly
            volume_spike_threshold: Multiplier for volume anomaly
            volatility_spike_threshold: Multiplier for volatility anomaly
            min_history: Minimum historical data points required
            alert_window_minutes: Time window to group similar alerts
        """
        self.price_spike_threshold = price_spike_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.min_history = min_history
        self.alert_window_minutes = alert_window_minutes

        self.state_file = os.path.join(PROJECT_ROOT, "anomaly_detection_state.json")
        self.alert_history = self._load_state()

        # Price history for analysis
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)

        # Trade tracking
        self.trade_history = deque(maxlen=50)

    def _load_state(self) -> List[Dict]:
        """Load alert history from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    # Keep only last 100 alerts
                    return data[-100:] if data else []
            except Exception:
                pass
        return []

    def _save_state(self):
        """Save alert history to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.alert_history, f, indent=2)

    def _add_alert(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        message: str,
        details: dict = None,
    ) -> Dict:
        """Add an alert to the history."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": anomaly_type.value,
            "severity": severity.value,
            "message": message,
            "details": details or {},
        }

        # Check if similar alert exists in recent window
        recent_cutoff = datetime.now() - timedelta(minutes=self.alert_window_minutes)
        recent_alerts = [
            a
            for a in self.alert_history
            if datetime.fromisoformat(a["timestamp"]) > recent_cutoff
        ]

        similar_exists = any(
            a["type"] == alert["type"] and a["severity"] == alert["severity"]
            for a in recent_alerts
        )

        if not similar_exists:
            self.alert_history.append(alert)
            self._save_state()
            return alert

        return None  # Duplicate alert suppressed

    def check_price_anomaly(
        self, current_price: float, volume: float = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for price anomalies.

        Args:
            current_price: Current market price
            volume: Optional current volume

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []
        self.price_history.append(current_price)

        if volume is not None:
            self.volume_history.append(volume)

        if len(self.price_history) < self.min_history:
            return False, alerts

        # Calculate statistics
        prices = list(self.price_history)
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        if std_price == 0:
            return False, alerts

        # Z-score for current price
        z_score = abs((current_price - mean_price) / std_price)

        if z_score > self.price_spike_threshold:
            severity = AnomalySeverity.CRITICAL if z_score > 5 else AnomalySeverity.HIGH
            message = (
                f"Price spike detected: ${current_price:.2f} is "
                f"{z_score:.1f}σ from ${mean_price:.2f} mean"
            )
            alert = self._add_alert(
                AnomalyType.PRICE_SPIKE,
                severity,
                message,
                {
                    "current_price": current_price,
                    "mean_price": mean_price,
                    "z_score": z_score,
                },
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_volume_anomaly(
        self, current_volume: float, current_price: float = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for volume anomalies.

        Args:
            current_volume: Current trading volume
            current_price: Optional current price for gap analysis

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        if current_volume is None or current_volume == 0:
            return False, alerts

        self.volume_history.append(current_volume)

        if len(self.volume_history) < self.min_history:
            return False, alerts

        # Calculate volume statistics
        volumes = list(self.volume_history)
        median_volume = np.median(volumes)

        if median_volume == 0:
            return False, alerts

        # Volume spike
        volume_ratio = current_volume / median_volume
        if volume_ratio > self.volume_spike_threshold:
            severity = (
                AnomalySeverity.HIGH if volume_ratio > 5 else AnomalySeverity.MEDIUM
            )
            message = (
                f"Volume spike: {current_volume:.0f} is {volume_ratio:.1f}x "
                f"the median {median_volume:.0f}"
            )
            alert = self._add_alert(
                AnomalyType.VOLUME_ANOMALY,
                severity,
                message,
                {
                    "current_volume": current_volume,
                    "median_volume": median_volume,
                    "ratio": volume_ratio,
                },
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_volatility_anomaly(
        self, high: float, low: float, close: float
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for volatility anomalies.

        Args:
            high: Current period high
            low: Current period low
            close: Current period close

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        # Calculate true range
        if len(self.price_history) > 0:
            prev_close = self.price_history[-1]
            true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))

            # Normalize by close price
            normalized_range = true_range / close if close > 0 else 0

            self.volatility_history.append(normalized_range)

            if len(self.volatility_history) < self.min_history:
                return False, alerts

            # Calculate average volatility
            volatilities = list(self.volatility_history)
            avg_volatility = np.mean(volatilities)

            if avg_volatility > 0:
                volatility_ratio = normalized_range / avg_volatility

                if volatility_ratio > self.volatility_spike_threshold:
                    severity = (
                        AnomalySeverity.HIGH
                        if volatility_ratio > 3
                        else AnomalySeverity.MEDIUM
                    )
                    message = (
                        f"Volatility spike: {normalized_range:.2%} is "
                        f"{volatility_ratio:.1f}x the average {avg_volatility:.2%}"
                    )
                    alert = self._add_alert(
                        AnomalyType.VOLATILITY_SPIKE,
                        severity,
                        message,
                        {
                            "current_volatility": normalized_range,
                            "avg_volatility": avg_volatility,
                            "ratio": volatility_ratio,
                        },
                    )
                    if alert:
                        alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_gap_anomaly(
        self, previous_close: float, current_open: float
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for price gaps between periods.

        Args:
            previous_close: Previous period close price
            current_open: Current period open price

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        if previous_close <= 0 or current_open <= 0:
            return False, alerts

        gap_pct = (current_open - previous_close) / previous_close

        # Gap threshold: 2% for crypto
        gap_threshold = 0.02

        if abs(gap_pct) > gap_threshold:
            direction = "up" if gap_pct > 0 else "down"
            severity = (
                AnomalySeverity.HIGH if abs(gap_pct) > 0.05 else AnomalySeverity.MEDIUM
            )
            message = (
                f"Price gap {direction}: {gap_pct:+.2%} from "
                f"${previous_close:.2f} to ${current_open:.2f}"
            )
            alert = self._add_alert(
                AnomalyType.GAP_ANOMALY,
                severity,
                message,
                {
                    "previous_close": previous_close,
                    "current_open": current_open,
                    "gap_pct": gap_pct,
                },
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_frequent_trading(
        self, trade_timestamp: str = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for excessive trading frequency.

        Args:
            trade_timestamp: Optional trade timestamp (defaults to now)

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        if trade_timestamp is None:
            trade_timestamp = datetime.now().isoformat()

        self.trade_history.append(trade_timestamp)

        if len(self.trade_history) < 5:
            return False, alerts

        # Check trades in last 10 minutes
        now = datetime.now()
        recent_trades = [
            t
            for t in self.trade_history
            if now - datetime.fromisoformat(t) <= timedelta(minutes=10)
        ]

        # Alert if more than 5 trades in 10 minutes
        if len(recent_trades) > 5:
            severity = (
                AnomalySeverity.CRITICAL
                if len(recent_trades) > 10
                else AnomalySeverity.HIGH
            )
            message = f"Frequent trading: {len(recent_trades)} trades in 10 minutes"
            alert = self._add_alert(
                AnomalyType.FREQUENT_TRADING,
                severity,
                message,
                {"trade_count": len(recent_trades), "window_minutes": 10},
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_pnl_anomaly(
        self, current_pnl_pct: float, expected_daily_pnl_pct: float = 0.0
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for unusual P&L values.

        Args:
            current_pnl_pct: Current P&L as percentage
            expected_daily_pnl_pct: Expected daily P&L (default 0%)

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        # Large loss alert: > 3% daily loss
        if current_pnl_pct < -3.0:
            severity = (
                AnomalySeverity.CRITICAL
                if current_pnl_pct < -5.0
                else AnomalySeverity.HIGH
            )
            message = f"Large loss: {current_pnl_pct:.2%} daily P&L"
            alert = self._add_alert(
                AnomalyType.LARGE_LOSSES,
                severity,
                message,
                {"pnl_pct": current_pnl_pct, "threshold": -3.0},
            )
            if alert:
                alerts.append(alert)

        # Unusual gain: > 5% daily gain
        elif current_pnl_pct > 5.0:
            severity = AnomalySeverity.MEDIUM
            message = f"Unusual gain: {current_pnl_pct:.2%} daily P&L"
            alert = self._add_alert(
                AnomalyType.UNUSUAL_PNL,
                severity,
                message,
                {"pnl_pct": current_pnl_pct, "threshold": 5.0},
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def check_position_drift(
        self,
        target_position_size: float,
        actual_position_size: float,
        max_drift_pct: float = 0.10,
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for position drift from target.

        Args:
            target_position_size: Target position size
            actual_position_size: Actual position size
            max_drift_pct: Maximum allowed drift as percentage (default 10%)

        Returns:
            Tuple of (has_anomaly: bool, alerts: list)
        """
        alerts = []

        if target_position_size == 0:
            return False, alerts

        drift_pct = (
            abs(actual_position_size - target_position_size) / target_position_size
        )

        if drift_pct > max_drift_pct:
            severity = (
                AnomalySeverity.HIGH if drift_pct > 0.20 else AnomalySeverity.MEDIUM
            )
            message = (
                f"Position drift: Actual {actual_position_size:.6f} vs "
                f"Target {target_position_size:.6f} ({drift_pct:.1%} off)"
            )
            alert = self._add_alert(
                AnomalyType.POSITION_DRIFT,
                severity,
                message,
                {
                    "target": target_position_size,
                    "actual": actual_position_size,
                    "drift_pct": drift_pct,
                },
            )
            if alert:
                alerts.append(alert)

        return len(alerts) > 0, alerts

    def get_recent_alerts(
        self, hours: int = 24, severity: AnomalySeverity = None
    ) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            hours: Hours to look back (default 24)
            severity: Optional severity filter

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            a
            for a in self.alert_history
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]

        if severity:
            recent = [a for a in recent if a["severity"] == severity.value]

        return recent

    def get_anomaly_summary(self) -> str:
        """Get human-readable anomaly summary."""
        recent = self.get_recent_alerts(hours=24)

        if not recent:
            return "No anomalies detected in the last 24 hours.\n"

        # Count by severity
        severity_counts = {}
        for alert in recent:
            sev = alert["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Count by type
        type_counts = {}
        for alert in recent:
            atype = alert["type"]
            type_counts[atype] = type_counts.get(atype, 0) + 1

        summary = "Anomaly Summary (Last 24 Hours):\n"
        summary += f"  Total Alerts: {len(recent)}\n"

        if severity_counts:
            summary += "  By Severity:\n"
            for sev, count in sorted(severity_counts.items()):
                emoji = {
                    "critical": "🚨",
                    "high": "🔴",
                    "medium": "🟠",
                    "low": "🟡",
                }.get(sev, "⚪")
                summary += f"    {emoji} {sev.upper()}: {count}\n"

        if type_counts:
            summary += "  By Type:\n"
            for atype, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                summary += f"    {atype}: {count}\n"

        # Show most critical recent alerts
        critical = [a for a in recent if a["severity"] == "critical"][-3:]
        if critical:
            summary += "\n  Most Critical Alerts:\n"
            for alert in critical:
                summary += f"    - [{alert['type']}] {alert['message']}\n"

        return summary

    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Determine if trading should be paused due to anomalies.

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        # Check for critical anomalies in last hour
        recent_critical = self.get_recent_alerts(
            hours=1, severity=AnomalySeverity.CRITICAL
        )

        if recent_critical:
            # Count unique types
            types = set(a["type"] for a in recent_critical)
            return True, f"Critical anomalies detected: {', '.join(types)}"

        # Check for high severity anomalies
        recent_high = self.get_recent_alerts(hours=1, severity=AnomalySeverity.HIGH)
        if len(recent_high) >= 3:
            return (
                True,
                f"Multiple high severity anomalies: {len(recent_high)} in last hour",
            )

        return False, ""
