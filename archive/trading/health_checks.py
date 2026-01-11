"""
Health Checks: API latency monitoring and system health tracking.
Monitors API response times and alerts when performance degrades.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from enum import Enum

from ..config import PROJECT_ROOT


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class APILatencyMonitor:
    """Monitor API latency and detect performance issues."""

    def __init__(
        self,
        warning_threshold_ms: int = 1000,  # 1 second
        critical_threshold_ms: int = 3000,  # 3 seconds
        window_size: int = 50,  # Number of requests to track
        degradation_pct: float = 0.5,  # 50% slower than baseline
    ):
        """
        Args:
            warning_threshold_ms: Latency threshold for warnings (default 1000ms)
            critical_threshold_ms: Latency threshold for critical alerts (default 3000ms)
            window_size: Number of recent requests to analyze
            degradation_pct: Percentage increase over baseline to flag degradation
        """
        self.warning_threshold_ms = warning_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.window_size = window_size
        self.degradation_pct = degradation_pct

        self.state_file = os.path.join(PROJECT_ROOT, "api_latency_state.json")
        self.latency_history = self._load_state()

        self.baseline_latency_ms = self._calculate_baseline()

    def _load_state(self) -> List[Dict]:
        """Load latency history from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    # Clean old entries (older than 1 hour)
                    cutoff = datetime.now() - timedelta(hours=1)
                    return [
                        entry
                        for entry in data
                        if datetime.fromisoformat(entry["timestamp"]) > cutoff
                    ]
            except Exception:
                pass
        return []

    def _save_state(self):
        """Save latency history to file."""
        # Keep only last 500 entries
        data = self.latency_history[-500:] if self.latency_history else []
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _calculate_baseline(self) -> float:
        """Calculate baseline latency from recent history."""
        if not self.latency_history:
            return 500.0  # Default baseline 500ms

        # Use median of recent requests as baseline
        recent = self.latency_history[-min(50, len(self.latency_history)) :]
        latencies = [entry["latency_ms"] for entry in recent]
        latencies.sort()
        return latencies[len(latencies) // 2] if latencies else 500.0

    def record_api_call(
        self, endpoint: str, latency_ms: float, success: bool, status_code: int = None
    ) -> Dict:
        """
        Record an API call and check for issues.

        Args:
            endpoint: API endpoint called
            latency_ms: Request latency in milliseconds
            success: Whether the call succeeded
            status_code: Optional HTTP status code

        Returns:
            Dict with health check results
        """
        timestamp = datetime.now().isoformat()

        entry = {
            "timestamp": timestamp,
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "success": success,
            "status_code": status_code,
        }

        self.latency_history.append(entry)
        self._save_state()

        # Analyze health
        return self._analyze_health(endpoint, latency_ms, success)

    def _analyze_health(self, endpoint: str, latency_ms: float, success: bool) -> Dict:
        """Analyze the health status of an API call."""
        status = HealthStatus.HEALTHY
        alerts = []

        # Check if call failed
        if not success:
            status = HealthStatus.CRITICAL
            alerts.append(f"API call failed for {endpoint}")
        # Check critical threshold
        elif latency_ms > self.critical_threshold_ms:
            status = HealthStatus.CRITICAL
            alerts.append(
                f"Critical latency: {latency_ms:.0f}ms > {self.critical_threshold_ms}ms "
                f"for {endpoint}"
            )
        # Check warning threshold
        elif latency_ms > self.warning_threshold_ms:
            status = HealthStatus.UNHEALTHY
            alerts.append(
                f"High latency: {latency_ms:.0f}ms > {self.warning_threshold_ms}ms "
                f"for {endpoint}"
            )
        # Check for degradation compared to baseline
        elif self._is_degraded(latency_ms):
            status = HealthStatus.DEGRADED
            alerts.append(
                f"Latency degradation: {latency_ms:.0f}ms is "
                f"{((latency_ms / self.baseline_latency_ms) - 1) * 100:.0f}% above baseline"
            )

        return {
            "status": status.value,
            "latency_ms": latency_ms,
            "endpoint": endpoint,
            "baseline_ms": self.baseline_latency_ms,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }

    def _is_degraded(self, latency_ms: float) -> bool:
        """Check if latency is degraded compared to baseline."""
        threshold = self.baseline_latency_ms * (1 + self.degradation_pct)
        return latency_ms > threshold and latency_ms < self.warning_threshold_ms

    def get_current_status(self) -> Dict:
        """Get current overall health status."""
        if not self.latency_history:
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "No data yet",
                "total_requests": 0,
            }

        # Get recent requests
        recent = self.latency_history[
            -min(self.window_size, len(self.latency_history)) :
        ]
        total_requests = len(recent)

        # Calculate stats
        latencies = [r["latency_ms"] for r in recent]
        success_rate = sum(1 for r in recent if r["success"]) / total_requests

        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        max_latency = max(latencies)

        # Determine overall status
        status = HealthStatus.HEALTHY
        issues = []

        if success_rate < 0.95:
            status = HealthStatus.CRITICAL
            issues.append(f"Low success rate: {success_rate:.1%}")
        elif success_rate < 0.99:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Reduced success rate: {success_rate:.1%}")

        if p99_latency > self.critical_threshold_ms:
            status = HealthStatus.CRITICAL
            issues.append(f"P99 latency critical: {p99_latency:.0f}ms")
        elif p95_latency > self.warning_threshold_ms:
            if status != HealthStatus.CRITICAL:
                status = HealthStatus.UNHEALTHY
            issues.append(f"P95 latency high: {p95_latency:.0f}ms")
        elif avg_latency > self.baseline_latency_ms * (1 + self.degradation_pct):
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Avg latency degraded: {avg_latency:.0f}ms")

        return {
            "status": status.value,
            "total_requests": total_requests,
            "success_rate": success_rate,
            "latency_stats": {
                "avg_ms": avg_latency,
                "p50_ms": p50_latency,
                "p95_ms": p95_latency,
                "p99_ms": p99_latency,
                "max_ms": max_latency,
            },
            "baseline_ms": self.baseline_latency_ms,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
        }

    def get_endpoint_stats(self, endpoint: str) -> Dict:
        """Get statistics for a specific endpoint."""
        endpoint_calls = [r for r in self.latency_history if r["endpoint"] == endpoint]

        if not endpoint_calls:
            return {
                "endpoint": endpoint,
                "total_calls": 0,
                "message": "No data for this endpoint",
            }

        recent = endpoint_calls[-min(self.window_size, len(endpoint_calls)) :]
        latencies = [r["latency_ms"] for r in recent]
        success_rate = sum(1 for r in recent if r["success"]) / len(recent)

        return {
            "endpoint": endpoint,
            "total_calls": len(recent),
            "success_rate": success_rate,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    def reset_baseline(self) -> float:
        """Recalculate baseline from current data."""
        self.baseline_latency_ms = self._calculate_baseline()
        return self.baseline_latency_ms

    def get_health_summary(self) -> str:
        """Get a human-readable health summary."""
        status = self.get_current_status()

        emoji = {
            HealthStatus.HEALTHY.value: "🟢",
            HealthStatus.DEGRADED.value: "🟠",
            HealthStatus.UNHEALTHY.value: "🔴",
            HealthStatus.CRITICAL.value: "🚨",
        }.get(status["status"], "⚪")

        summary = f"{emoji} API Health Status: {status['status'].upper()}\n"
        summary += f"  Total Requests: {status['total_requests']}\n"
        summary += f"  Success Rate: {status['success_rate']:.1%}\n"

        stats = status.get("latency_stats", {})
        if stats:
            summary += "  Latency (avg/P50/P95/P99): "
            summary += f"{stats['avg_ms']:.0f}/{stats['p50_ms']:.0f}/"
            summary += f"{stats['p95_ms']:.0f}/{stats['p99_ms']:.0f}ms\n"
            summary += f"  Baseline: {status['baseline_ms']:.0f}ms"

        if status.get("issues"):
            summary += f"\n  Issues: {', '.join(status['issues'])}"

        return summary


class SystemHealthChecker:
    """Comprehensive system health monitoring."""

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether health checking is enabled
        """
        self.enabled = enabled
        self.latency_monitor = APILatencyMonitor()

        self.health_state_file = os.path.join(PROJECT_ROOT, "system_health_state.json")
        self.health_history = self._load_health_history()

    def _load_health_history(self) -> List[Dict]:
        """Load health history from file."""
        if os.path.exists(self.health_state_file):
            try:
                with open(self.health_state_file) as f:
                    data = json.load(f)
                    # Keep only last 100 entries
                    return data[-100:] if data else []
            except Exception:
                pass
        return []

    def _save_health_history(self):
        """Save health history to file."""
        with open(self.health_state_file, "w") as f:
            json.dump(self.health_history, f, indent=2)

    def check_api_health(
        self, endpoint: str, latency_ms: float, success: bool, status_code: int = None
    ) -> Tuple[bool, List[str]]:
        """
        Check API health after a call.

        Args:
            endpoint: API endpoint called
            latency_ms: Request latency in milliseconds
            success: Whether the call succeeded
            status_code: Optional HTTP status code

        Returns:
            Tuple of (is_healthy: bool, alerts: list)
        """
        if not self.enabled:
            return True, []

        result = self.latency_monitor.record_api_call(
            endpoint, latency_ms, success, status_code
        )

        # Record to health history
        self.health_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "status": result["status"],
                "endpoint": endpoint,
                "latency_ms": latency_ms,
            }
        )
        self._save_health_history()

        is_healthy = result["status"] in [
            HealthStatus.HEALTHY.value,
            HealthStatus.DEGRADED.value,
        ]

        return is_healthy, result["alerts"]

    def get_overall_health(self) -> Dict:
        """Get overall system health status."""
        if not self.enabled:
            return {
                "status": "unknown",
                "message": "Health checking disabled",
            }

        api_health = self.latency_monitor.get_current_status()

        return {
            "timestamp": datetime.now().isoformat(),
            "api_health": api_health,
            "overall_status": self._determine_overall_status(api_health),
        }

    def _determine_overall_status(self, api_health: Dict) -> str:
        """Determine overall system status from components."""
        api_status = api_health.get("status")

        if api_status == HealthStatus.CRITICAL.value:
            return "critical"
        elif api_status == HealthStatus.UNHEALTHY.value:
            return "unhealthy"
        elif api_status == HealthStatus.DEGRADED.value:
            return "degraded"
        else:
            return "healthy"

    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Determine if trading should be paused due to health issues.

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        if not self.enabled:
            return False, ""

        health = self.get_overall_health()
        api_health = health["api_health"]

        # Pause if critical status
        if api_health["status"] == HealthStatus.CRITICAL.value:
            return (
                True,
                f"Critical API health: {', '.join(api_health.get('issues', []))}",
            )

        # Pause if success rate too low
        if api_health["success_rate"] < 0.90:
            return True, f"Low success rate: {api_health['success_rate']:.1%}"

        # Pause if very high latency
        stats = api_health.get("latency_stats", {})
        if stats.get("p99_ms", 0) > 5000:  # 5 seconds
            return True, f"Extreme P99 latency: {stats['p99_ms']:.0f}ms"

        return False, ""

    def get_health_report(self) -> str:
        """Get comprehensive health report."""
        if not self.enabled:
            return "Health checking is disabled.\n"

        health = self.get_overall_health()

        report = self.latency_monitor.get_health_summary()
        report += f"\n\nOverall Status: {health['overall_status'].upper()}"

        should_pause, reason = self.should_pause_trading()
        if should_pause:
            report += f"\n⚠️ RECOMMEND PAUSE: {reason}"

        return report
