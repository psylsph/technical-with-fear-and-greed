"""
Centralized State Management System with persistence.

Features:
- Thread-safe state management
- Automatic persistence to disk
- State versioning and rollback
- State change events
- Memory-efficient caching
- Concurrent access support
"""

import os
import json
import threading
import fcntl
import tempfile
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from enum import Enum
import hashlib


class StateEventType(Enum):
    """Types of state events."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    LOADED = "loaded"
    SAVED = "saved"
    ROLLED_BACK = "rolled_back"


@dataclass
class StateEvent:
    """State change event."""
    event_type: StateEventType
    key: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


StateCallback = Callable[[StateEvent], None]


T = TypeVar("T")


class StateStore(ABC, Generic[T]):
    """Abstract base class for state stores."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: T) -> None:
        """Set a value by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value by key."""
        pass

    @abstractmethod
    def list_keys(self) -> List[str]:
        """List all keys."""
        pass


@dataclass
class StateSnapshot:
    """Snapshot of state at a point in time."""
    timestamp: datetime
    data: Dict[str, Any]
    version: int
    checksum: str


class InMemoryStateStore(StateStore[Any]):
    """In-memory state store with change tracking."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._callbacks: List[StateCallback] = []

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a value by key."""
        with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value

            # Emit event
            self._emit_event(
                StateEvent(
                    event_type=StateEventType.UPDATED if old_value is not None else StateEventType.CREATED,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                )
            )

    def delete(self, key: str) -> None:
        """Delete a value by key."""
        with self._lock:
            if key in self._data:
                old_value = self._data.pop(key)
                self._emit_event(
                    StateEvent(
                        event_type=StateEventType.DELETED,
                        key=key,
                        old_value=old_value,
                    )
                )

    def list_keys(self) -> List[str]:
        """List all keys."""
        with self._lock:
            return list(self._data.keys())

    def subscribe(self, callback: StateCallback) -> None:
        """Subscribe to state change events."""
        with self._lock:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: StateCallback) -> None:
        """Unsubscribe from state change events."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _emit_event(self, event: StateEvent) -> None:
        """Emit state change event to subscribers."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let one bad callback break others

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._data.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary."""
        with self._lock:
            return dict(self._data)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import state from dictionary."""
        with self._lock:
            self._data = dict(data)
            self._emit_event(
                StateEvent(event_type=StateEventType.LOADED, key="*")
            )


class PersistentStateStore(InMemoryStateStore):
    """
    Persistent state store with automatic file-based persistence.

    Features:
    - Atomic writes (write to temp file, then rename)
    - File locking for concurrent access
    - Automatic snapshot creation
    - Checksum validation
    """

    def __init__(
        self,
        state_file: str,
        auto_save: bool = True,
        snapshot_interval: int = 100,
        max_snapshots: int = 10,
    ):
        """
        Initialize persistent state store.

        Args:
            state_file: Path to state file
            auto_save: Automatically save on changes
            snapshot_interval: Create snapshot every N changes
            max_snapshots: Maximum number of snapshots to keep
        """
        super().__init__()

        self.state_file = Path(state_file)
        self.auto_save = auto_save
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self._change_count = 0
        self._version = 0
        self._snapshots: List[StateSnapshot] = []

        # Create directory if needed
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state
        self._load_from_disk()

    def set(self, key: str, value: Any) -> None:
        """Set a value and optionally save."""
        super().set(key, value)

        self._change_count += 1

        if self.auto_save and self._change_count % self.snapshot_interval == 0:
            self._create_snapshot()

        if self.auto_save:
            self._save_to_disk()

    def delete(self, key: str) -> None:
        """Delete a value and optionally save."""
        super().delete(key)

        if self.auto_save:
            self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Save state to disk atomically."""
        data = {
            "version": self._version + 1,
            "timestamp": datetime.now().isoformat(),
            "data": self.to_dict(),
        }

        # Calculate checksum
        data_str = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        data["checksum"] = checksum

        # Write to temp file first (atomic write)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.state_file.parent,
            prefix=f".{self.state_file.name}.",
            delete=False,
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2, default=str)
            tmp_path = tmp_file.name

        # Atomic rename
        try:
            # Use file locking to prevent concurrent writes
            with open(self.state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                shutil.move(tmp_path, self.state_file.name)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self._version += 1

            # Emit save event
            self._emit_event(
                StateEvent(
                    event_type=StateEventType.SAVED,
                    key="*",
                    new_value=f"v{self._version}",
                )
            )

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def _load_from_disk(self) -> None:
        """Load state from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r") as f:
                # Use file locking to prevent concurrent reads during writes
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Validate checksum
            if "checksum" in data:
                stored_checksum = data["checksum"]
                data_copy = data.copy()
                data_copy.pop("checksum")
                data_str = json.dumps(data_copy, sort_keys=True, default=str)
                calculated_checksum = hashlib.sha256(data_str.encode()).hexdigest()

                if stored_checksum != calculated_checksum:
                    raise ValueError("State file checksum mismatch - data may be corrupted")

            self._version = data.get("version", 0)
            self.from_dict(data.get("data", {}))

            # Emit load event
            self._emit_event(
                StateEvent(
                    event_type=StateEventType.LOADED,
                    key="*",
                    new_value=f"v{self._version}",
                )
            )

        except Exception as e:
            print(f"Error loading state file: {e}")
            # Start with empty state on error
            self.clear()

    def _create_snapshot(self) -> None:
        """Create a snapshot of current state."""
        data = self.to_dict()
        data_str = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()

        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            data=data.copy(),
            version=self._version,
            checksum=checksum,
        )

        self._snapshots.append(snapshot)

        # Keep only recent snapshots
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def rollback_to_snapshot(self, snapshot_index: int = -1) -> None:
        """
        Rollback to a previous snapshot.

        Args:
            snapshot_index: Index of snapshot to rollback to (-1 = most recent)
        """
        if not self._snapshots:
            raise ValueError("No snapshots available")

        snapshot = self._snapshots[snapshot_index]

        # Validate checksum
        data_str = json.dumps(snapshot.data, sort_keys=True, default=str)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()

        if checksum != snapshot.checksum:
            raise ValueError("Snapshot checksum mismatch")

        self.from_dict(snapshot.data)

        # Emit rollback event
        self._emit_event(
            StateEvent(
                event_type=StateEventType.ROLLED_BACK,
                key="*",
                new_value=f"v{snapshot.version}",
            )
        )

    def get_snapshots(self) -> List[StateSnapshot]:
        """Get all snapshots."""
        return list(self._snapshots)

    def force_save(self) -> None:
        """Force immediate save to disk."""
        self._save_to_disk()


class StateManager:
    """
    Centralized state management system.

    Provides:
    - Multiple named state stores
    - Thread-safe access
    - Automatic persistence
    - State change notifications
    - Snapshot and rollback
    """

    _instance: Optional["StateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StateManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._stores: Dict[str, StateStore] = {}
        self._local_lock = threading.RLock()

        # Register default stores
        self._register_default_stores()

    def _register_default_stores(self) -> None:
        """Register default state stores."""
        try:
            from src.trading.config_manager import get_database_config

            db_config = get_database_config()

            # Portfolio state store
            self.register_store(
                "portfolio",
                PersistentStateStore(
                    state_file=db_config.portfolio_state_file,
                    auto_save=True,
                ),
            )

        except Exception:
            # Fallback to default path
            self.register_store(
                "portfolio",
                PersistentStateStore(
                    state_file="cache/portfolio_state.json",
                    auto_save=True,
                ),
            )

        # In-memory stores for ephemeral state
        self.register_store("cache", InMemoryStateStore())
        self.register_store("session", InMemoryStateStore())
        self.register_store("alerts", InMemoryStateStore())

    def register_store(self, name: str, store: StateStore) -> None:
        """Register a state store."""
        with self._local_lock:
            self._stores[name] = store

    def get_store(self, name: str) -> StateStore:
        """Get a state store by name."""
        with self._local_lock:
            if name not in self._stores:
                # Create default in-memory store
                self._stores[name] = InMemoryStateStore()

            return self._stores[name]

    def get(self, store_name: str, key: str, default: Any = None) -> Any:
        """Get a value from a store."""
        store = self.get_store(store_name)
        result = store.get(key)
        return result if result is not None else default

    def set(self, store_name: str, key: str, value: Any) -> None:
        """Set a value in a store."""
        store = self.get_store(store_name)
        store.set(key, value)

    def delete(self, store_name: str, key: str) -> None:
        """Delete a value from a store."""
        store = self.get_store(store_name)
        store.delete(key)

    def subscribe(self, store_name: str, callback: StateCallback) -> None:
        """Subscribe to state change events for a store."""
        store = self.get_store(store_name)
        if isinstance(store, InMemoryStateStore):
            store.subscribe(callback)

    def force_save(self, store_name: str) -> None:
        """Force save a store to disk."""
        store = self.get_store(store_name)
        if isinstance(store, PersistentStateStore):
            store.force_save()

    def clear_store(self, store_name: str) -> None:
        """Clear all data in a store."""
        store = self.get_store(store_name)
        if isinstance(store, InMemoryStateStore):
            store.clear()

    def export_state(self, store_name: str) -> Dict[str, Any]:
        """Export state from a store."""
        store = self.get_store(store_name)
        if isinstance(store, InMemoryStateStore):
            return store.to_dict()
        return {}

    def import_state(self, store_name: str, data: Dict[str, Any]) -> None:
        """Import state to a store."""
        store = self.get_store(store_name)
        if isinstance(store, InMemoryStateStore):
            store.from_dict(data)


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# Convenience functions for common state operations
def get_state(key: str, store: str = "portfolio", default: Any = None) -> Any:
    """Get a state value."""
    return get_state_manager().get(store, key, default)


def set_state(key: str, value: Any, store: str = "portfolio") -> None:
    """Set a state value."""
    get_state_manager().set(store, key, value)


def delete_state(key: str, store: str = "portfolio") -> None:
    """Delete a state value."""
    get_state_manager().delete(store, key)


def subscribe_to_state(callback: StateCallback, store: str = "portfolio") -> None:
    """Subscribe to state changes."""
    get_state_manager().subscribe(store, callback)


def force_save_state(store: str = "portfolio") -> None:
    """Force save state to disk."""
    get_state_manager().force_save(store)
