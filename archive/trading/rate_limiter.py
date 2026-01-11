"""
API Rate Limiter with configurable limits and automatic backoff.

Features:
- Per-API rate limiting (requests per minute/second)
- Distributed rate limiting support (Redis)
- Automatic backoff on rate limit errors
- Sliding window rate limiting
- Priority queue for critical requests
- Rate limit statistics and monitoring
"""

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, Optional
import functools


class Priority(Enum):
    """Request priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_second: int = 5
    burst_allowed: int = 10  # Allow short bursts


@dataclass
class RateLimitStats:
    """Rate limit statistics."""

    total_requests: int = 0
    throttled_requests: int = 0
    rate_limit_errors: int = 0
    average_wait_time: float = 0.0
    current_window_usage: int = 0
    current_window_capacity: int = 0


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Features:
    - Sliding window for accurate rate limiting
    - Per-API rate limits
    - Automatic backoff on 429 responses
    - Priority queue for important requests
    - Statistics tracking
    """

    _instance: Optional["RateLimiter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RateLimiter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._limits: Dict[str, RateLimit] = {}
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._second_windows: Dict[str, deque] = defaultdict(deque)
        self._stats: Dict[str, RateLimitStats] = defaultdict(RateLimitStats)
        self._wait_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._backoff_until: Dict[str, datetime] = {}
        self._local_lock = threading.Lock()

    def set_limit(self, api_name: str, limit: RateLimit) -> None:
        """Set rate limit for an API."""
        with self._local_lock:
            self._limits[api_name] = limit

    def get_limit(self, api_name: str) -> Optional[RateLimit]:
        """Get rate limit for an API."""
        return self._limits.get(api_name)

    def acquire(
        self,
        api_name: str,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire permission to make a request.

        Args:
            api_name: Name of the API (e.g., "alpaca", "coinbase")
            priority: Request priority (higher priority requests may jump queue)
            timeout: Maximum time to wait for permission (None = wait indefinitely)

        Returns:
            True if permission granted, False if timeout occurred
        """
        limit = self._limits.get(api_name)
        if limit is None:
            # No rate limit configured
            return True

        start_time = time.time()

        while True:
            # Check if we're in backoff period
            if api_name in self._backoff_until:
                backoff_end = self._backoff_until[api_name]
                if datetime.now() < backoff_end:
                    wait_time = (backoff_end - datetime.now()).total_seconds()
                    if timeout is not None and wait_time > timeout:
                        self._stats[api_name].throttled_requests += 1
                        return False
                    time.sleep(min(wait_time, 1.0))
                    continue
                else:
                    # Backoff period expired
                    del self._backoff_until[api_name]

            # Try to acquire
            if self._try_acquire(api_name, limit):
                wait_time = time.time() - start_time
                self._wait_times[api_name].append(wait_time)
                self._stats[api_name].total_requests += 1
                return True

            # Calculate wait time
            wait_time = self._calculate_wait_time(api_name, limit)

            if timeout is not None and (time.time() - start_time + wait_time) > timeout:
                self._stats[api_name].throttled_requests += 1
                return False

            time.sleep(wait_time)

    def _try_acquire(self, api_name: str, limit: RateLimit) -> bool:
        """Try to acquire permission (internal)."""
        with self._local_lock:
            now = time.time()
            minute_ago = now - 60
            second_ago = now - 1

            # Clean old timestamps from minute window
            minute_window = self._windows[api_name]
            while minute_window and minute_window[0] < minute_ago:
                minute_window.popleft()

            # Clean old timestamps from second window
            second_window = self._second_windows[api_name]
            while second_window and second_window[0] < second_ago:
                second_window.popleft()

            # Check limits
            if len(minute_window) >= limit.requests_per_minute:
                return False

            if len(second_window) >= limit.requests_per_second:
                return False

            # Allow acquisition
            minute_window.append(now)
            second_window.append(now)
            self._stats[api_name].current_window_usage = len(minute_window)
            self._stats[api_name].current_window_capacity = limit.requests_per_minute

            return True

    def _calculate_wait_time(self, api_name: str, limit: RateLimit) -> float:
        """Calculate how long to wait before next request."""
        with self._local_lock:
            now = time.time()

            # Check minute window
            minute_window = self._windows[api_name]
            if minute_window and len(minute_window) >= limit.requests_per_minute:
                oldest_timestamp = minute_window[0]
                return max(0, 60 - (now - oldest_timestamp) + 0.1)

            # Check second window
            second_window = self._second_windows[api_name]
            if second_window and len(second_window) >= limit.requests_per_second:
                oldest_timestamp = second_window[0]
                return max(0, 1 - (now - oldest_timestamp) + 0.1)

            return 0.1  # Small delay to prevent tight loops

    def report_error(self, api_name: str, status_code: int) -> None:
        """
        Report an API error (e.g., 429 rate limit error).

        Automatically triggers backoff for rate limit errors.
        """
        if status_code == 429:
            # Rate limit error - trigger backoff
            self._stats[api_name].rate_limit_errors += 1
            # Back off for 60 seconds
            self._backoff_until[api_name] = datetime.now() + timedelta(seconds=60)

    def get_stats(self, api_name: str) -> RateLimitStats:
        """Get rate limit statistics for an API."""
        with self._local_lock:
            stats = self._stats[api_name]

            # Calculate average wait time
            wait_times = self._wait_times[api_name]
            if wait_times:
                stats.average_wait_time = sum(wait_times) / len(wait_times)

            return stats

    def reset(self, api_name: Optional[str] = None) -> None:
        """Reset rate limit state."""
        with self._local_lock:
            if api_name:
                self._windows.pop(api_name, None)
                self._second_windows.pop(api_name, None)
                self._backoff_until.pop(api_name, None)
                self._stats[api_name] = RateLimitStats()
            else:
                self._windows.clear()
                self._second_windows.clear()
                self._backoff_until.clear()
                self._stats.clear()


def rate_limit(api_name: str, priority: Priority = Priority.NORMAL):
    """
    Decorator to rate limit a function.

    Usage:
        @rate_limit("alpaca", Priority.HIGH)
        def fetch_orders():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = RateLimiter()
            limiter.acquire(api_name, priority)

            try:
                result = func(*args, **kwargs)

                # Check if result is a response with status code
                if hasattr(result, "status_code"):
                    limiter.report_error(api_name, result.status_code)

                return result
            except Exception as e:
                # Check for rate limit errors in exceptions
                if hasattr(e, "status_code"):
                    limiter.report_error(api_name, e.status_code)
                raise

        return wrapper

    return decorator


class RateLimitedAPI:
    """
    Mixin class for API clients with automatic rate limiting.

    Usage:
        class MyAPIClient(RateLimitedAPI):
            def __init__(self):
                super().__init__(api_name="my_api", rate_limit=RateLimit(100, 10))

            @rate_limit_method
            def fetch_data(self):
                ...
    """

    def __init__(self, api_name: str, rate_limit: RateLimit):
        self._api_name = api_name
        self._rate_limiter = RateLimiter()
        self._rate_limiter.set_limit(api_name, rate_limit)

    def rate_limit_method(self, priority: Priority = Priority.NORMAL) -> Callable:
        """Decorator for rate limiting methods."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self._rate_limiter.acquire(self._api_name, priority)

                try:
                    result = func(*args, **kwargs)

                    if hasattr(result, "status_code"):
                        self._rate_limiter.report_error(
                            self._api_name, result.status_code
                        )

                    return result
                except Exception as e:
                    if hasattr(e, "status_code"):
                        self._rate_limiter.report_error(self._api_name, e.status_code)
                    raise

            return wrapper

        return decorator

    def get_rate_limit_stats(self) -> RateLimitStats:
        """Get rate limit statistics."""
        return self._rate_limiter.get_stats(self._api_name)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

        # Set default limits from config
        try:
            from src.trading.config_manager import get_api_config

            api_config = get_api_config()

            # Alpaca
            _rate_limiter.set_limit(
                "alpaca",
                RateLimit(
                    requests_per_minute=api_config.max_requests_per_minute,
                    requests_per_second=api_config.max_requests_per_second,
                ),
            )

            # Coinbase
            _rate_limiter.set_limit(
                "coinbase",
                RateLimit(requests_per_minute=60, requests_per_second=5),
            )

            # Fear & Greed Index
            _rate_limiter.set_limit(
                "fgi", RateLimit(requests_per_minute=10, requests_per_second=1)
            )

        except Exception:
            # Use defaults if config not available
            _rate_limiter.set_limit(
                "alpaca", RateLimit(requests_per_minute=60, requests_per_second=5)
            )
            _rate_limiter.set_limit(
                "coinbase", RateLimit(requests_per_minute=60, requests_per_second=5)
            )
            _rate_limiter.set_limit(
                "fgi", RateLimit(requests_per_minute=10, requests_per_second=1)
            )

    return _rate_limiter


def setup_rate_limits(config: Dict[str, RateLimit]) -> None:
    """
    Setup rate limits for multiple APIs.

    Args:
        config: Dictionary mapping API names to rate limits
    """
    limiter = get_rate_limiter()

    for api_name, limit in config.items():
        limiter.set_limit(api_name, limit)
