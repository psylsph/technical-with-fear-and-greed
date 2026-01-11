"""
Change Data Capture: Efficiently process only new or changed data.
Reduces processing overhead by tracking and processing only data changes.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

import pandas as pd

from ..config import PROJECT_ROOT


class ChangeType(Enum):
    """Types of data changes."""

    INSERT = "insert"  # New data
    UPDATE = "update"  # Modified data
    DELETE = "delete"  # Removed data
    NONE = "none"  # No changes


@dataclass
class DataChange:
    """Represents a single data change."""

    change_type: ChangeType
    row_id: str
    timestamp: str
    data: Dict
    previous_data: Dict = None

    def __post_init__(self):
        if self.previous_data is None:
            self.previous_data = {}


@dataclass
class ChangeSet:
    """A set of changes for a processing run."""

    source: str
    capture_time: str
    inserts: List[DataChange]
    updates: List[DataChange]
    deletes: List[DataChange]
    total_changes: int
    processing_time_ms: float

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.total_changes > 0


@dataclass
class DataSnapshot:
    """Snapshot of data at a point in time."""

    snapshot_id: str
    source: str
    timestamp: str
    row_count: int
    data_hash: str
    column_schemas: Dict[str, str]
    latest_timestamp: str


class ChangeDetector:
    """
    Detect changes between data snapshots.

    Features:
    - Row-level change detection
    - Column-level change tracking
    - Hash-based comparison
    - Key-based identification
    """

    def __init__(
        self,
        key_columns: List[str] = None,
        hash_columns: List[str] = None,
        ignore_columns: List[str] = None,
    ):
        """
        Args:
            key_columns: Columns that uniquely identify rows
            hash_columns: Columns to include in hash calculation
            ignore_columns: Columns to ignore when comparing
        """
        self.key_columns = key_columns or ["timestamp"]
        self.hash_columns = hash_columns
        self.ignore_columns = ignore_columns or []

    def _get_row_key(self, row: pd.Series) -> str:
        """Generate unique key for a row."""
        key_parts = [str(row[col]) for col in self.key_columns if col in row.index]
        return "|".join(key_parts)

    def _calculate_row_hash(self, row: pd.Series) -> str:
        """Calculate hash for a row."""
        # Exclude ignored columns
        cols_to_hash = [col for col in row.index if col not in self.ignore_columns]

        if self.hash_columns:
            cols_to_hash = [col for col in cols_to_hash if col in self.hash_columns]

        # Create deterministic string
        row_str = "|".join(str(row[col]) for col in sorted(cols_to_hash))
        return hashlib.md5(row_str.encode()).hexdigest()

    def detect_changes(
        self,
        previous_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> ChangeSet:
        """
        Detect changes between two data snapshots.

        Args:
            previous_data: Previous data snapshot
            current_data: Current data snapshot

        Returns:
            ChangeSet with all detected changes
        """
        start_time = datetime.now()

        inserts = []
        updates = []
        deletes = []

        if previous_data is None or previous_data.empty:
            # All data is new
            for _, row in current_data.iterrows():
                change = DataChange(
                    change_type=ChangeType.INSERT,
                    row_id=self._get_row_key(row),
                    timestamp=datetime.now().isoformat(),
                    data=row.to_dict(),
                )
                inserts.append(change)
        elif current_data is None or current_data.empty:
            # All data was deleted
            for _, row in previous_data.iterrows():
                change = DataChange(
                    change_type=ChangeType.DELETE,
                    row_id=self._get_row_key(row),
                    timestamp=datetime.now().isoformat(),
                    data={},
                    previous_data=row.to_dict(),
                )
                deletes.append(change)
        else:
            # Compare row by row
            prev_keys = set(
                self._get_row_key(row) for _, row in previous_data.iterrows()
            )
            curr_keys = set(
                self._get_row_key(row) for _, row in current_data.iterrows()
            )

            # Detect inserts (new keys)
            inserted_keys = curr_keys - prev_keys
            for key in inserted_keys:
                # Find the row with this key
                matching_rows = current_data[
                    current_data.apply(self._get_row_key, axis=1) == key
                ]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    change = DataChange(
                        change_type=ChangeType.INSERT,
                        row_id=key,
                        timestamp=datetime.now().isoformat(),
                        data=row.to_dict(),
                    )
                    inserts.append(change)

            # Detect deletes (removed keys)
            deleted_keys = prev_keys - curr_keys
            for key in deleted_keys:
                matching_rows = previous_data[
                    previous_data.apply(self._get_row_key, axis=1) == key
                ]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    change = DataChange(
                        change_type=ChangeType.DELETE,
                        row_id=key,
                        timestamp=datetime.now().isoformat(),
                        data={},
                        previous_data=row.to_dict(),
                    )
                    deletes.append(change)

            # Detect updates (same keys, different data)
            common_keys = prev_keys & curr_keys
            for key in common_keys:
                prev_row = previous_data[
                    previous_data.apply(self._get_row_key, axis=1) == key
                ].iloc[0]

                curr_row = current_data[
                    current_data.apply(self._get_row_key, axis=1) == key
                ].iloc[0]

                # Compare hashes
                prev_hash = self._calculate_row_hash(prev_row)
                curr_hash = self._calculate_row_hash(curr_row)

                if prev_hash != curr_hash:
                    change = DataChange(
                        change_type=ChangeType.UPDATE,
                        row_id=key,
                        timestamp=datetime.now().isoformat(),
                        data=curr_row.to_dict(),
                        previous_data=prev_row.to_dict(),
                    )
                    updates.append(change)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ChangeSet(
            source="unknown",
            capture_time=datetime.now().isoformat(),
            inserts=inserts,
            updates=updates,
            deletes=deletes,
            total_changes=len(inserts) + len(updates) + len(deletes),
            processing_time_ms=processing_time,
        )


class ChangeDataCapture:
    """
    Track and process only new or changed data.

    Features:
    - Maintain data snapshots
    - Detect changes since last capture
    - Efficient incremental processing
    - Change log persistence
    """

    def __init__(
        self,
        source_name: str,
        state_file: str = None,
        key_columns: List[str] = None,
        enable_snapshots: bool = True,
    ):
        """
        Args:
            source_name: Name of the data source
            state_file: Path to state file
            key_columns: Columns that identify unique rows
            enable_snapshots: Whether to maintain snapshots
        """
        self.source_name = source_name
        self.enable_snapshots = enable_snapshots

        # State file for persistence
        self.state_file = state_file or os.path.join(
            PROJECT_ROOT, "cache", "cdc_state", f"{source_name}.json"
        )
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # Change detector
        self.detector = ChangeDetector(key_columns=key_columns)

        # State
        self.last_snapshot: Optional[pd.DataFrame] = None
        self.last_snapshot_time: Optional[str] = None
        self.change_history: List[ChangeSet] = []

        # Load state
        self._load_state()

    def _load_state(self) -> None:
        """Load CDC state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)

                self.last_snapshot_time = data.get("last_snapshot_time")

                # Load change history
                for change_data in data.get("change_history", []):
                    # Reconstruct DataChange objects
                    inserts = [
                        DataChange(
                            change_type=ChangeType(c["change_type"]),
                            row_id=c["row_id"],
                            timestamp=c["timestamp"],
                            data=c["data"],
                            previous_data=c.get("previous_data"),
                        )
                        for c in change_data.get("inserts", [])
                    ]

                    updates = [
                        DataChange(
                            change_type=ChangeType(c["change_type"]),
                            row_id=c["row_id"],
                            timestamp=c["timestamp"],
                            data=c["data"],
                            previous_data=c.get("previous_data"),
                        )
                        for c in change_data.get("updates", [])
                    ]

                    deletes = [
                        DataChange(
                            change_type=ChangeType(c["change_type"]),
                            row_id=c["row_id"],
                            timestamp=c["timestamp"],
                            data=c["data"],
                            previous_data=c.get("previous_data"),
                        )
                        for c in change_data.get("deletes", [])
                    ]

                    change_set = ChangeSet(
                        source=change_data["source"],
                        capture_time=change_data["capture_time"],
                        inserts=inserts,
                        updates=updates,
                        deletes=deletes,
                        total_changes=change_data["total_changes"],
                        processing_time_ms=change_data["processing_time_ms"],
                    )

                    self.change_history.append(change_set)

            except Exception as e:
                print(f"Error loading CDC state: {e}")

    def _save_state(self) -> None:
        """Save CDC state to file."""
        data = {
            "source": self.source_name,
            "last_snapshot_time": self.last_snapshot_time,
            "last_updated": datetime.now().isoformat(),
            "change_history": [
                {
                    "source": cs.source,
                    "capture_time": cs.capture_time,
                    "inserts": [
                        {
                            "change_type": c.change_type.value,
                            "row_id": c.row_id,
                            "timestamp": c.timestamp,
                            "data": c.data,
                            "previous_data": c.previous_data,
                        }
                        for c in cs.inserts
                    ],
                    "updates": [
                        {
                            "change_type": c.change_type.value,
                            "row_id": c.row_id,
                            "timestamp": c.timestamp,
                            "data": c.data,
                            "previous_data": c.previous_data,
                        }
                        for c in cs.updates
                    ],
                    "deletes": [
                        {
                            "change_type": c.change_type.value,
                            "row_id": c.row_id,
                            "timestamp": c.timestamp,
                            "data": c.data,
                            "previous_data": c.previous_data,
                        }
                        for c in cs.deletes
                    ],
                    "total_changes": cs.total_changes,
                    "processing_time_ms": cs.processing_time_ms,
                }
                for cs in self.change_history[-100:]  # Keep last 100
            ],
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of entire DataFrame."""
        # Sort by key columns for deterministic hash
        if self.detector.key_columns:
            sorted_data = data.sort_values(self.detector.key_columns)
        else:
            sorted_data = data.sort_index()

        data_str = sorted_data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def capture_changes(self, current_data: pd.DataFrame) -> ChangeSet:
        """
        Capture changes from previous snapshot.

        Args:
            current_data: Current data snapshot

        Returns:
            ChangeSet with detected changes
        """
        # Detect changes
        change_set = self.detector.detect_changes(
            self.last_snapshot,
            current_data,
        )
        change_set.source = self.source_name

        # Update snapshot
        if self.enable_snapshots:
            self.last_snapshot = current_data.copy()
            self.last_snapshot_time = datetime.now().isoformat()

        # Track changes
        if change_set.has_changes:
            self.change_history.append(change_set)

        # Save state
        self._save_state()

        return change_set

    def get_new_data(
        self,
        current_data: pd.DataFrame,
        process_func: Callable = None,
    ) -> Tuple[pd.DataFrame, ChangeSet]:
        """
        Get only new or changed data.

        Args:
            current_data: Current data snapshot
            process_func: Optional function to process changes

        Returns:
            Tuple of (new/changed_data, change_set)
        """
        change_set = self.capture_changes(current_data)

        if not change_set.has_changes:
            return pd.DataFrame(), change_set

        # Combine inserts and updates
        new_rows = [c.data for c in change_set.inserts]
        updated_rows = [c.data for c in change_set.updates]

        all_changes = new_rows + updated_rows

        if not all_changes:
            return pd.DataFrame(), change_set

        changed_data = pd.DataFrame(all_changes)

        # Process if function provided
        if process_func is not None:
            try:
                process_func(changed_data, change_set)
            except Exception as e:
                print(f"Error in process_func: {e}")

        return changed_data, change_set

    def needs_processing(self, current_data: pd.DataFrame) -> bool:
        """
        Check if current data has changes.

        Args:
            current_data: Current data snapshot

        Returns:
            True if there are changes
        """
        if self.last_snapshot is None:
            return not current_data.empty

        # Quick hash check
        current_hash = self._calculate_data_hash(current_data)
        last_hash = self._calculate_data_hash(self.last_snapshot)

        return current_hash != last_hash

    def get_incremental_data(
        self,
        fetch_func: Callable,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get incremental data using fetch function.

        Args:
            fetch_func: Function that fetches current data
            force_refresh: Force full data refresh

        Returns:
            DataFrame with new/changed data
        """
        current_data = fetch_func()

        if current_data is None or current_data.empty:
            return pd.DataFrame()

        if force_refresh:
            # Reset and capture all as new
            self.last_snapshot = None
            return current_data

        # Get only changes
        new_data, _ = self.get_new_data(current_data)

        return new_data

    def get_change_statistics(self, hours: int = 24) -> Dict:
        """
        Get statistics about recent changes.

        Args:
            hours: Number of hours to analyze

        Returns:
            Statistics dict
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_changes = [
            cs
            for cs in self.change_history
            if datetime.fromisoformat(cs.capture_time) > cutoff
        ]

        if not recent_changes:
            return {
                "period_hours": hours,
                "total_changesets": 0,
                "total_changes": 0,
            }

        total_inserts = sum(len(cs.inserts) for cs in recent_changes)
        total_updates = sum(len(cs.updates) for cs in recent_changes)
        total_deletes = sum(len(cs.deletes) for cs in recent_changes)
        total_changes = sum(cs.total_changes for cs in recent_changes)

        # Calculate change rate
        avg_changes_per_set = (
            total_changes / len(recent_changes) if recent_changes else 0
        )

        return {
            "period_hours": hours,
            "total_changesets": len(recent_changes),
            "total_changes": total_changes,
            "inserts": total_inserts,
            "updates": total_updates,
            "deletes": total_deletes,
            "avg_changes_per_set": round(avg_changes_per_set, 2),
            "last_capture": recent_changes[-1].capture_time if recent_changes else None,
        }

    def reset(self) -> None:
        """Reset CDC state."""
        self.last_snapshot = None
        self.last_snapshot_time = None
        self.change_history = []
        self._save_state()

    def get_full_data(self) -> Optional[pd.DataFrame]:
        """Get the full current snapshot."""
        return self.last_snapshot


class CDCManager:
    """
    Manages multiple Change Data Capture instances.

    Features:
    - Multiple source tracking
    - Aggregated statistics
    - Bulk operations
    """

    def __init__(self, state_dir: str = None):
        """
        Args:
            state_dir: Directory for CDC state files
        """
        self.state_dir = state_dir or os.path.join(PROJECT_ROOT, "cache", "cdc_state")
        os.makedirs(self.state_dir, exist_ok=True)

        self.instances: Dict[str, ChangeDataCapture] = {}

    def get_cdc(
        self,
        source: str,
        key_columns: List[str] = None,
    ) -> ChangeDataCapture:
        """
        Get or create CDC instance for a source.

        Args:
            source: Source name
            key_columns: Key columns for change detection

        Returns:
            ChangeDataCapture instance
        """
        if source not in self.instances:
            state_file = os.path.join(self.state_dir, f"{source}.json")
            self.instances[source] = ChangeDataCapture(
                source_name=source,
                state_file=state_file,
                key_columns=key_columns,
            )

        return self.instances[source]

    def get_all_statistics(self, hours: int = 24) -> Dict:
        """Get aggregated statistics for all sources."""
        stats = {
            "period_hours": hours,
            "sources": {},
            "total_changes": 0,
        }

        for source, cdc in self.instances.items():
            source_stats = cdc.get_change_statistics(hours)
            stats["sources"][source] = source_stats
            stats["total_changes"] += source_stats.get("total_changes", 0)

        return stats

    def reset_all(self) -> None:
        """Reset all CDC instances."""
        for cdc in self.instances.values():
            cdc.reset()


def get_cdc_manager() -> CDCManager:
    """Get singleton CDC manager instance."""
    if not hasattr(get_cdc_manager, "_instance"):
        get_cdc_manager._instance = CDCManager()
    return get_cdc_manager._instance


def track_market_data_changes(
    symbol: str,
    fetch_func: Callable,
    process_func: Callable = None,
) -> pd.DataFrame:
    """
    Convenience function to track market data changes.

    Args:
        symbol: Trading symbol
        fetch_func: Function to fetch current data
        process_func: Optional function to process changes

    Returns:
        DataFrame with new/changed data
    """
    manager = get_cdc_manager()
    cdc = manager.get_cdc(
        source=f"market_data_{symbol}",
        key_columns=["timestamp"],
    )

    return cdc.get_incremental_data(fetch_func)
