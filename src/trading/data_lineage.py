"""
Data Lineage: Track data provenance and transformation history.
Provides complete audit trail for all data flowing through the system.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib

import pandas as pd

from ..config import PROJECT_ROOT


class DataSourceType(Enum):
    """Types of data sources."""
    API = "api"
    FILE = "file"
    DATABASE = "database"
    CACHE = "cache"
    CALCULATED = "calculated"
    AGGREGATED = "aggregated"
    EXTERNAL = "external"


class TransformationType(Enum):
    """Types of data transformations."""
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    MERGE = "merge"
    CALCULATE = "calculate"
    RESAMPLE = "resample"
    CLEAN = "clean"
    VALIDATE = "validate"


@dataclass
class DataOrigin:
    """Origin information for a dataset."""
    source_type: DataSourceType
    source_name: str
    source_id: str
    timestamp: str
    url: str = None
    api_endpoint: str = None
    file_path: str = None
    query: str = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Transformation:
    """A transformation applied to data."""
    transformation_id: str
    transformation_type: TransformationType
    name: str
    description: str
    parameters: Dict
    input_data_ids: List[str]
    output_data_id: str
    timestamp: str
    performer: str = "system"
    code_reference: str = None

    def __post_init__(self):
        if not self.transformation_id:
            self.transformation_id = str(uuid.uuid4())


@dataclass
class DataLineageRecord:
    """Complete lineage record for a dataset."""
    data_id: str
    name: str
    description: str
    origin: DataOrigin
    transformations: List[Transformation]
    data_hash: str
    row_count: int
    column_count: int
    size_bytes: int
    schema: Dict[str, str]
    tags: List[str]
    parent_ids: List[str]
    child_ids: List[str]
    created_at: str
    updated_at: str
    metadata: Dict

    def __post_init__(self):
        if not self.data_id:
            self.data_id = str(uuid.uuid4())
        if self.tags is None:
            self.tags = []
        if self.parent_ids is None:
            self.parent_ids = []
        if self.child_ids is None:
            self.child_ids = []
        if self.metadata is None:
            self.metadata = {}

    def get_lineage_chain(self) -> List[str]:
        """Get full chain of data IDs from origin to current."""
        chain = [self.data_id]
        current = self

        # Add all parent IDs recursively
        to_process = current.parent_ids.copy()
        while to_process:
            chain.extend(to_process)
            # This would need access to the full lineage store
            # For now, just return direct parents
            to_process = []

        return list(reversed(chain))

    def add_transformation(self, transformation: Transformation) -> None:
        """Add a transformation to the lineage."""
        self.transformations.append(transformation)
        self.updated_at = datetime.now().isoformat()


class DataLineageTracker:
    """
    Track complete data lineage and provenance.

    Features:
    - Track data origin and source
    - Record all transformations
    - Maintain data hashes for integrity
    - Build lineage chains
    - Export lineage reports
    """

    def __init__(
        self,
        lineage_file: str = None,
        enable_hashing: bool = True,
    ):
        """
        Args:
            lineage_file: Path to lineage JSON file
            enable_hashing: Whether to calculate data hashes
        """
        self.lineage_file = lineage_file or os.path.join(
            PROJECT_ROOT, "data_lineage.json"
        )
        self.enable_hashing = enable_hashing

        self.records: Dict[str, DataLineageRecord] = {}
        self._load_lineage()

    def _load_lineage(self) -> None:
        """Load lineage records from file."""
        if os.path.exists(self.lineage_file):
            try:
                with open(self.lineage_file) as f:
                    data = json.load(f)

                for record_data in data.get("records", []):
                    # Reconstruct DataOrigin
                    origin_data = record_data["origin"]
                    origin = DataOrigin(
                        source_type=DataSourceType(origin_data["source_type"]),
                        source_name=origin_data["source_name"],
                        source_id=origin_data["source_id"],
                        timestamp=origin_data["timestamp"],
                        url=origin_data.get("url"),
                        api_endpoint=origin_data.get("api_endpoint"),
                        file_path=origin_data.get("file_path"),
                        query=origin_data.get("query"),
                        metadata=origin_data.get("metadata", {}),
                    )

                    # Reconstruct Transformations
                    transformations = []
                    for trans_data in record_data["transformations"]:
                        trans = Transformation(
                            transformation_id=trans_data["transformation_id"],
                            transformation_type=TransformationType(trans_data["transformation_type"]),
                            name=trans_data["name"],
                            description=trans_data["description"],
                            parameters=trans_data["parameters"],
                            input_data_ids=trans_data["input_data_ids"],
                            output_data_id=trans_data["output_data_id"],
                            timestamp=trans_data["timestamp"],
                            performer=trans_data.get("performer", "system"),
                            code_reference=trans_data.get("code_reference"),
                        )
                        transformations.append(trans)

                    # Reconstruct record
                    record = DataLineageRecord(
                        data_id=record_data["data_id"],
                        name=record_data["name"],
                        description=record_data["description"],
                        origin=origin,
                        transformations=transformations,
                        data_hash=record_data["data_hash"],
                        row_count=record_data["row_count"],
                        column_count=record_data["column_count"],
                        size_bytes=record_data["size_bytes"],
                        schema=record_data["schema"],
                        tags=record_data.get("tags", []),
                        parent_ids=record_data.get("parent_ids", []),
                        child_ids=record_data.get("child_ids", []),
                        created_at=record_data["created_at"],
                        updated_at=record_data["updated_at"],
                        metadata=record_data.get("metadata", {}),
                    )

                    self.records[record.data_id] = record

            except Exception as e:
                print(f"Error loading lineage: {e}")

    def _save_lineage(self) -> None:
        """Save lineage records to file."""
        # Convert to serializable format
        records_data = []
        for record in self.records.values():
            record_dict = {
                "data_id": record.data_id,
                "name": record.name,
                "description": record.description,
                "origin": {
                    "source_type": record.origin.source_type.value,
                    "source_name": record.origin.source_name,
                    "source_id": record.origin.source_id,
                    "timestamp": record.origin.timestamp,
                    "url": record.origin.url,
                    "api_endpoint": record.origin.api_endpoint,
                    "file_path": record.origin.file_path,
                    "query": record.origin.query,
                    "metadata": record.origin.metadata,
                },
                "transformations": [
                    {
                        "transformation_id": t.transformation_id,
                        "transformation_type": t.transformation_type.value,
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "input_data_ids": t.input_data_ids,
                        "output_data_id": t.output_data_id,
                        "timestamp": t.timestamp,
                        "performer": t.performer,
                        "code_reference": t.code_reference,
                    }
                    for t in record.transformations
                ],
                "data_hash": record.data_hash,
                "row_count": record.row_count,
                "column_count": record.column_count,
                "size_bytes": record.size_bytes,
                "schema": record.schema,
                "tags": record.tags,
                "parent_ids": record.parent_ids,
                "child_ids": record.child_ids,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "metadata": record.metadata,
            }
            records_data.append(record_dict)

        data = {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(self.records),
            "records": records_data,
        }

        with open(self.lineage_file, "w") as f:
            json.dump(data, f, indent=2)

    def _calculate_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for integrity checking."""
        if not self.enable_hashing:
            return ""

        # Create deterministic string representation
        sorted_cols = sorted(data.columns)
        data_str = data[sorted_cols].to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _estimate_size(self, data: pd.DataFrame) -> int:
        """Estimate size of DataFrame in bytes."""
        return data.memory_usage(deep=True).sum()

    def track_data_source(
        self,
        data: pd.DataFrame,
        name: str,
        source_type: DataSourceType,
        source_name: str,
        description: str = "",
        tags: List[str] = None,
        **source_kwargs,
    ) -> str:
        """
        Track a new data source.

        Args:
            data: The DataFrame being tracked
            name: Name for this dataset
            source_type: Type of data source
            source_name: Name of the source (e.g., "Alpaca API")
            description: Description of the data
            tags: Tags for categorization
            **source_kwargs: Additional source-specific params

        Returns:
            data_id of the tracked dataset
        """
        data_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Create origin
        origin = DataOrigin(
            source_type=source_type,
            source_name=source_name,
            source_id=data_id,
            timestamp=now,
            url=source_kwargs.get("url"),
            api_endpoint=source_kwargs.get("api_endpoint"),
            file_path=source_kwargs.get("file_path"),
            query=source_kwargs.get("query"),
            metadata=source_kwargs.get("metadata", {}),
        )

        # Create record
        record = DataLineageRecord(
            data_id=data_id,
            name=name,
            description=description,
            origin=origin,
            transformations=[],
            data_hash=self._calculate_hash(data),
            row_count=len(data),
            column_count=len(data.columns),
            size_bytes=self._estimate_size(data),
            schema={col: str(data[col].dtype) for col in data.columns},
            tags=tags or [],
            parent_ids=[],
            child_ids=[],
            created_at=now,
            updated_at=now,
            metadata={"index_type": str(type(data.index).__name__)},
        )

        self.records[data_id] = record
        self._save_lineage()

        return data_id

    def track_transformation(
        self,
        input_data_ids: List[str],
        output_data: pd.DataFrame,
        transformation_name: str,
        transformation_type: TransformationType,
        description: str = "",
        parameters: Dict = None,
        performer: str = "system",
        output_name: str = None,
    ) -> str:
        """
        Track a data transformation.

        Args:
            input_data_ids: List of input data IDs
            output_data: The transformed output DataFrame
            transformation_name: Name of the transformation
            transformation_type: Type of transformation
            description: Description of what was done
            parameters: Parameters used in transformation
            performer: Who/what performed the transformation
            output_name: Optional name for output dataset

        Returns:
            data_id of the output dataset
        """
        output_data_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Get input records for lineage
        parent_records = [
            self.records[pid] for pid in input_data_ids if pid in self.records
        ]

        # Determine output name
        if output_name is None:
            if parent_records:
                output_name = f"{parent_records[0].name}_{transformation_name}"
            else:
                output_name = f"transformed_{transformation_name}"

        # Create transformation record
        transformation = Transformation(
            transformation_id=str(uuid.uuid4()),
            transformation_type=transformation_type,
            name=transformation_name,
            description=description,
            parameters=parameters or {},
            input_data_ids=input_data_ids,
            output_data_id=output_data_id,
            timestamp=now,
            performer=performer,
        )

        # Use origin from first parent (or create synthetic origin)
        if parent_records:
            origin = parent_records[0].origin
        else:
            origin = DataOrigin(
                source_type=DataSourceType.CALCULATED,
                source_name="system",
                source_id=output_data_id,
                timestamp=now,
            )

        # Create output record
        record = DataLineageRecord(
            data_id=output_data_id,
            name=output_name,
            description=f"{description} (from {len(input_data_ids)} sources)",
            origin=origin,
            transformations=[transformation],
            data_hash=self._calculate_hash(output_data),
            row_count=len(output_data),
            column_count=len(output_data.columns),
            size_bytes=self._estimate_size(output_data),
            schema={col: str(output_data[col].dtype) for col in output_data.columns},
            tags=[],
            parent_ids=input_data_ids,
            child_ids=[],
            created_at=now,
            updated_at=now,
            metadata={"transformation_count": 1},
        )

        # Update parent records to include this as child
        for pid in input_data_ids:
            if pid in self.records:
                if output_data_id not in self.records[pid].child_ids:
                    self.records[pid].child_ids.append(output_data_id)

        self.records[output_data_id] = record
        self._save_lineage()

        return output_data_id

    def get_lineage(self, data_id: str) -> Optional[DataLineageRecord]:
        """
        Get lineage record for a data ID.

        Args:
            data_id: Data ID to look up

        Returns:
            DataLineageRecord or None if not found
        """
        return self.records.get(data_id)

    def get_lineage_chain(self, data_id: str) -> List[Dict]:
        """
        Get full lineage chain for a data ID.

        Args:
            data_id: Data ID to trace

        Returns:
            List of lineage records from origin to target
        """
        chain = []
        visited = set()
        to_visit = [data_id]

        while to_visit and visited != set(to_visit):
            current_id = to_visit.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id in self.records:
                record = self.records[current_id]
                chain.append({
                    "data_id": record.data_id,
                    "name": record.name,
                    "origin": record.origin.source_name,
                    "timestamp": record.created_at,
                    "transformations": len(record.transformations),
                })

                # Add parents
                for pid in record.parent_ids:
                    if pid not in visited:
                        to_visit.append(pid)

        return list(reversed(chain))

    def verify_integrity(self, data_id: str, data: pd.DataFrame) -> bool:
        """
        Verify data integrity against stored hash.

        Args:
            data_id: Data ID to verify
            data: DataFrame to compare

        Returns:
            True if hash matches
        """
        record = self.records.get(data_id)
        if not record or not record.data_hash:
            return False

        current_hash = self._calculate_hash(data)
        return current_hash == record.data_hash

    def find_by_tag(self, tag: str) -> List[DataLineageRecord]:
        """Find all records with a specific tag."""
        return [
            record for record in self.records.values()
            if tag in record.tags
        ]

    def find_by_source(self, source_name: str) -> List[DataLineageRecord]:
        """Find all records from a specific source."""
        return [
            record for record in self.records.values()
            if record.origin.source_name == source_name
        ]

    def generate_lineage_report(self, data_id: str) -> str:
        """
        Generate human-readable lineage report.

        Args:
            data_id: Data ID to generate report for

        Returns:
            Formatted report string
        """
        record = self.records.get(data_id)
        if not record:
            return f"No lineage record found for {data_id}"

        chain = self.get_lineage_chain(data_id)

        report = f"Data Lineage Report: {record.name}\n"
        report += f"{'=' * 60}\n\n"
        report += f"Data ID: {record.data_id}\n"
        report += f"Description: {record.description}\n"
        report += f"Created: {record.created_at}\n"
        report += f"Updated: {record.updated_at}\n\n"

        report += "Data Statistics:\n"
        report += f"  Rows: {record.row_count:,}\n"
        report += f"  Columns: {record.column_count}\n"
        report += f"  Size: {record.size_bytes / 1024:.2f} KB\n"
        report += f"  Hash: {record.data_hash}\n\n"

        report += "Origin:\n"
        report += f"  Type: {record.origin.source_type.value}\n"
        report += f"  Source: {record.origin.source_name}\n"
        report += f"  Timestamp: {record.origin.timestamp}\n"
        if record.origin.api_endpoint:
            report += f"  API: {record.origin.api_endpoint}\n"
        if record.origin.file_path:
            report += f"  File: {record.origin.file_path}\n\n"

        if record.transformations:
            report += f"Transformations ({len(record.transformations)}):\n"
            for i, trans in enumerate(record.transformations, 1):
                report += f"  {i}. {trans.name} ({trans.transformation_type.value})\n"
                report += f"     {trans.description}\n"
                if trans.parameters:
                    report += f"     Parameters: {trans.parameters}\n"
                report += f"     Performed: {trans.timestamp}\n\n"

        if len(chain) > 1:
            report += f"Lineage Chain ({len(chain)} steps):\n"
            for i, step in enumerate(chain, 1):
                report += f"  {i}. {step['name']} ({step['origin']})\n"
                report += f"     ID: {step['data_id']}\n"
                report += f"     Time: {step['timestamp']}\n"

        if record.tags:
            report += f"\nTags: {', '.join(record.tags)}\n"

        return report

    def get_statistics(self) -> Dict:
        """Get lineage statistics."""
        if not self.records:
            return {"total_records": 0}

        source_counts = {}
        type_counts = {}

        for record in self.records.values():
            source_name = record.origin.source_name
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

            source_type = record.origin.source_type.value
            type_counts[source_type] = type_counts.get(source_type, 0) + 1

        return {
            "total_records": len(self.records),
            "source_breakdown": source_counts,
            "type_breakdown": type_counts,
            "total_transformations": sum(len(r.transformations) for r in self.records.values()),
            "total_size_mb": sum(r.size_bytes for r in self.records.values()) / 1024 / 1024,
        }


def get_lineage_tracker() -> DataLineageTracker:
    """Get singleton lineage tracker instance."""
    if not hasattr(get_lineage_tracker, "_instance"):
        get_lineage_tracker._instance = DataLineageTracker()
    return get_lineage_tracker._instance


def track_market_data_fetch(
    data: pd.DataFrame,
    symbol: str,
    source: str = "alpaca",
) -> str:
    """
    Convenience function to track market data fetch.

    Args:
        data: Fetched market data DataFrame
        symbol: Trading symbol
        source: Data source name

    Returns:
        data_id of the tracked dataset
    """
    tracker = get_lineage_tracker()
    return tracker.track_data_source(
        data=data,
        name=f"{symbol}_market_data",
        source_type=DataSourceType.API,
        source_name=source,
        description=f"Market data for {symbol} from {source}",
        tags=[symbol, "market_data", source],
    )
