"""
Intelligent Caching: TTL-based caching with adaptive refresh.
Manages market data caching to reduce API calls and improve performance.
"""

import os
import time
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading

import pandas as pd
import numpy as np

from ..config import PROJECT_ROOT


class CacheStrategy(Enum):
    """Cache refresh strategies."""
    TTL = "ttl"  # Time-based expiration
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns
    LAZY = "lazy"  # Refresh on access if expired
    EAGER = "eager"  # Pre-emptive refresh before expiration


class CacheHitRate(Enum):
    """Cache performance categories."""
    EXCELLENT = "excellent"  # > 90% hit rate
    GOOD = "good"  # > 70% hit rate
    FAIR = "fair"  # > 50% hit rate
    POOR = "poor"  # <= 50% hit rate


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    access_count: int
    last_access: float
    size_bytes: int
    metadata: Dict

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > self.ttl

    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp

    def access_age_seconds(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_access


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    evictions: int
    total_size_bytes: int
    hit_rate: float
    avg_access_time_ms: float
    last_updated: str

    @property
    def hit_rate_category(self) -> str:
        """Get hit rate performance category."""
        if self.hit_rate >= 0.9:
            return CacheHitRate.EXCELLENT.value
        elif self.hit_rate >= 0.7:
            return CacheHitRate.GOOD.value
        elif self.hit_rate >= 0.5:
            return CacheHitRate.FAIR.value
        else:
            return CacheHitRate.POOR.value


class TTLCache:
    """
    Time-based cache with configurable TTL.

    Features:
    - Per-key TTL configuration
    - Automatic expiration
    - Size-based eviction
    - Thread-safe operations
    """

    def __init__(
        self,
        default_ttl: int = 300,  # 5 minutes
        max_size_mb: float = 100,
        max_entries: int = 10000,
        cleanup_interval: int = 60,
    ):
        """
        Args:
            default_ttl: Default time-to-live in seconds
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of entries
            cleanup_interval: Seconds between cleanup runs
        """
        self.default_ttl = default_ttl
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Statistics
        self._stats = CacheStats(
            total_requests=0,
            cache_hits=0,
            cache_misses=0,
            evictions=0,
            total_size_bytes=0,
            hit_rate=0.0,
            avg_access_time_ms=0.0,
            last_updated=datetime.now().isoformat(),
        )
        self._access_times: List[float] = []

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_parts = [str(a) for a in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, (pd.Series, np.ndarray)):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode())

    def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return 0

        expired_keys = []
        total_size = 0

        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
                else:
                    total_size += entry.size_bytes

            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

            self._stats.total_size_bytes = total_size
            self._last_cleanup = now

        return len(expired_keys)

    def _evict_lru(self, required_space: int) -> int:
        """Evict least recently used entries to free space."""
        evicted = 0
        freed_space = 0

        with self._lock:
            # Sort by last access time
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_access
            )

            for key, entry in sorted_entries:
                if freed_space >= required_space:
                    break
                del self._cache[key]
                freed_space += entry.size_bytes
                self._stats.evictions += 1
                evicted += 1

            self._stats.total_size_bytes -= freed_space

        return evicted

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()

        with self._lock:
            self._stats.total_requests += 1

            if key not in self._cache:
                self._stats.cache_misses += 1
                self._update_hit_rate()
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.cache_misses += 1
                self._stats.evictions += 1
                self._stats.total_size_bytes -= entry.size_bytes
                self._update_hit_rate()
                return None

            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()

            self._stats.cache_hits += 1

        # Record access time
        access_time_ms = (time.time() - start_time) * 1000
        self._access_times.append(access_time_ms)
        if len(self._access_times) > 1000:
            self._access_times.pop(0)
        self._stats.avg_access_time_ms = np.mean(self._access_times)

        self._update_hit_rate()
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        metadata: Dict = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            metadata: Optional metadata dict

        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        now = time.time()

        # Calculate size
        size_bytes = self._calculate_size(value)

        # Check if we need to evict
        with self._lock:
            # Remove existing entry if updating
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size_bytes -= old_entry.size_bytes

            # Check size limits
            would_exceed = (
                self._stats.total_size_bytes + size_bytes > self.max_size_bytes or
                len(self._cache) >= self.max_entries
            )

            if would_exceed:
                self._evict_lru(size_bytes + 1024 * 1024)  # Extra 1MB buffer

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=now,
                ttl=ttl,
                access_count=0,
                last_access=now,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

            self._cache[key] = entry
            self._stats.total_size_bytes += size_bytes

        # Periodic cleanup
        self._cleanup_expired()

        return True

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.total_size_bytes -= entry.size_bytes
                return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.total_size_bytes = 0

    def _update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        with self._lock:
            if self._stats.total_requests > 0:
                self._stats.hit_rate = (
                    self._stats.cache_hits / self._stats.total_requests
                )
            self._stats.last_updated = datetime.now().isoformat()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._cleanup_expired()
        self._update_hit_rate()
        return self._stats

    def get_entries_info(self) -> List[Dict]:
        """Get information about all cache entries."""
        with self._lock:
            return [
                {
                    "key": entry.key[:16] + "...",
                    "age_seconds": round(entry.age_seconds(), 1),
                    "ttl": entry.ttl,
                    "access_count": entry.access_count,
                    "size_kb": round(entry.size_bytes / 1024, 2),
                    "expired": entry.is_expired(),
                }
                for entry in self._cache.values()
            ]


class AdaptiveCache(TTLCache):
    """
    Adaptive cache that adjusts TTL based on usage patterns.

    Features:
    - Automatic TTL adjustment based on access frequency
    - Predictive pre-fetching
    - Usage pattern analysis
    """

    def __init__(
        self,
        default_ttl: int = 300,
        max_size_mb: float = 100,
        max_entries: int = 10000,
        min_ttl: int = 60,
        max_ttl: int = 3600,
        learning_window: int = 10,
    ):
        """
        Args:
            default_ttl: Default TTL in seconds
            max_size_mb: Maximum cache size
            max_entries: Maximum entries
            min_ttl: Minimum TTL for adaptive adjustment
            max_ttl: Maximum TTL for adaptive adjustment
            learning_window: Number of accesses before adjusting TTL
        """
        super().__init__(default_ttl, max_size_mb, max_entries)
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.learning_window = learning_window

        # Track access patterns
        self._access_patterns: Dict[str, List[float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value and track access pattern."""
        value = super().get(key)

        if value is not None and key in self._cache:
            # Track access time
            now = time.time()
            if key not in self._access_patterns:
                self._access_patterns[key] = []

            self._access_patterns[key].append(now)

            # Keep only recent accesses
            if len(self._access_patterns[key]) > self.learning_window:
                self._access_patterns[key].pop(0)

            # Adjust TTL based on access pattern
            if len(self._access_patterns[key]) >= self.learning_window:
                self._adjust_ttl(key)

        return value

    def _adjust_ttl(self, key: str) -> None:
        """Adjust TTL based on access frequency."""
        if key not in self._cache or key not in self._access_patterns:
            return

        accesses = self._access_patterns[key]
        if len(accesses) < 2:
            return

        # Calculate average time between accesses
        intervals = [accesses[i] - accesses[i - 1] for i in range(1, len(accesses))]
        avg_interval = np.mean(intervals)

        # Set TTL to slightly more than average interval
        new_ttl = int(min(max(avg_interval * 1.5, self.min_ttl), self.max_ttl))

        entry = self._cache[key]
        entry.ttl = new_ttl


class IntelligentCacheManager:
    """
    Intelligent cache manager for market data.

    Features:
    - Multiple cache strategies
    - Adaptive TTL adjustment
    - Performance monitoring
    - Automatic cleanup
    """

    def __init__(
        self,
        cache_dir: str = None,
        default_ttl: int = 300,  # 5 minutes
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    ):
        """
        Args:
            cache_dir: Directory for persistent cache storage
            default_ttl: Default TTL in seconds
            strategy: Cache refresh strategy
        """
        self.cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "cache", "intelligent")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.default_ttl = default_ttl
        self.strategy = strategy

        # Create cache based on strategy
        if strategy == CacheStrategy.ADAPTIVE:
            self.cache = AdaptiveCache(default_ttl=default_ttl)
        else:
            self.cache = TTLCache(default_ttl=default_ttl)

        # Persistent cache for disk storage
        self.persistent_file = os.path.join(self.cache_dir, "persistent_cache.pkl")

        # Load persistent cache
        self._load_persistent()

    def _load_persistent(self) -> None:
        """Load persistent cache from disk."""
        if os.path.exists(self.persistent_file):
            try:
                with open(self.persistent_file, "rb") as f:
                    data = pickle.load(f)
                    for key, entry_data in data.items():
                        # Restore entries that aren't expired
                        entry = entry_data
                        if not entry.is_expired():
                            self.cache._cache[key] = entry
                            self.cache._stats.total_size_bytes += entry.size_bytes
            except Exception as e:
                print(f"Error loading persistent cache: {e}")

    def _save_persistent(self) -> None:
        """Save persistent cache to disk."""
        try:
            # Filter entries worth persisting (frequently accessed, long TTL)
            persistent_entries = {}
            for key, entry in self.cache._cache.items():
                # Persist if accessed multiple times or TTL > 10 minutes
                if entry.access_count >= 3 or entry.ttl >= 600:
                    persistent_entries[key] = entry

            with open(self.persistent_file, "wb") as f:
                pickle.dump(persistent_entries, f)
        except Exception as e:
            print(f"Error saving persistent cache: {e}")

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        fetch_func: Callable = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get market data with caching.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "5m", "1h")
            limit: Number of bars to retrieve
            fetch_func: Function to call on cache miss

        Returns:
            DataFrame with market data or None
        """
        cache_key = f"market_data:{symbol}:{timeframe}:{limit}"

        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Cache miss - fetch if function provided
        if fetch_func is None:
            return None

        try:
            data = fetch_func(symbol, timeframe, limit)

            if data is not None and not data.empty:
                # Cache with adaptive TTL based on timeframe
                ttl = self._get_timeframe_ttl(timeframe)
                self.cache.set(cache_key, data, ttl=ttl, metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                })

            return data

        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def _get_timeframe_ttl(self, timeframe: str) -> int:
        """Get appropriate TTL for timeframe."""
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return timeframe_map.get(timeframe, self.default_ttl)

    def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cache entries for a symbol.

        Args:
            symbol: Symbol to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0
        for key in list(self.cache._cache.keys()):
            if key.startswith(f"market_data:{symbol}:"):
                self.cache.delete(key)
                count += 1
        return count

    def get_performance_report(self) -> str:
        """Generate cache performance report."""
        stats = self.cache.get_stats()

        report = "Cache Performance Report\n"
        report += f"{'=' * 40}\n\n"
        report += f"Strategy: {self.strategy.value}\n"
        report += f"Total Requests: {stats.total_requests}\n"
        report += f"Cache Hits: {stats.cache_hits}\n"
        report += f"Cache Misses: {stats.cache_misses}\n"
        report += f"Hit Rate: {stats.hit_rate:.1%} ({stats.hit_rate_category})\n"
        report += f"Evictions: {stats.evictions}\n"
        report += f"Total Size: {stats.total_size_bytes / 1024 / 1024:.2f} MB\n"
        report += f"Entries: {len(self.cache._cache)}\n"
        report += f"Avg Access Time: {stats.avg_access_time_ms:.2f} ms\n\n"

        # Show top entries by access count
        entries_info = self.cache.get_entries_info()
        if entries_info:
            top_entries = sorted(
                entries_info,
                key=lambda x: x["access_count"],
                reverse=True
            )[:5]

            report += "Top 5 Entries by Access:\n"
            for entry in top_entries:
                report += (
                    f"  {entry['key']}: "
                    f"{entry['access_count']} accesses, "
                    f"{entry['size_kb']} KB\n"
                )

        return report

    def save_state(self) -> None:
        """Save cache state to disk."""
        self._save_persistent()

    def cleanup(self) -> Dict:
        """
        Perform cache cleanup.

        Returns:
            Dict with cleanup results
        """
        expired_count = self.cache._cleanup_expired()

        return {
            "expired_removed": expired_count,
            "remaining_entries": len(self.cache._cache),
            "total_size_mb": self.cache._stats.total_size_bytes / 1024 / 1024,
        }


def cached(ttl: int = 300, key_func: Callable = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_func: Function to generate cache key

    Returns:
        Decorated function
    """
    cache = TTLCache(default_ttl=ttl)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()

            # Try cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            if result is not None:
                cache.set(cache_key, result, ttl=ttl)

            return result

        wrapper.cache = cache  # Expose cache for inspection
        return wrapper

    return decorator


def get_cache_manager() -> IntelligentCacheManager:
    """
    Get singleton cache manager instance.

    Returns:
        IntelligentCacheManager instance
    """
    if not hasattr(get_cache_manager, "_instance"):
        get_cache_manager._instance = IntelligentCacheManager()
    return get_cache_manager._instance
