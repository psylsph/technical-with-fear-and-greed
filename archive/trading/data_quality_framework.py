"""
Data Quality Framework: Comprehensive data validation, cleaning, and scoring.
Provides enhanced data quality management beyond basic validation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from ..config import PROJECT_ROOT


class QualityIssue(Enum):
    """Types of quality issues."""

    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE = "duplicate"
    GAP = "gap"


class QualityAction(Enum):
    """Actions to take on quality issues."""

    KEEP = "keep"
    REMOVE = "remove"
    CORRECT = "correct"
    FLAG = "flag"
    INTERPOLATE = "interpolate"


@dataclass
class QualityReport:
    """Comprehensive data quality report."""

    source: str
    timestamp: str
    record_count: int
    overall_score: float  # 0-100
    freshness_score: float
    completeness_score: float
    consistency_score: float
    validity_score: float
    issues: List[Dict]
    actions_taken: List[Dict]
    recommendations: List[str]


@dataclass
class DataPoint:
    """A single data point with metadata."""

    value: Any
    timestamp: datetime
    source: str
    is_valid: bool = True
    is_cleaned: bool = False
    quality_flags: List[str] = None
    data_point_id: str = None

    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []
        if self.data_point_id is None:
            self.data_point_id = f"{self.source}_{self.timestamp.isoformat()}"


class DataCleaner:
    """
    Clean and correct market data issues.

    Features:
    - Remove/correct outliers
    - Fill missing values
    - Smooth noisy data
    - Correct OHLC inconsistencies
    """

    def __init__(
        self,
        outlier_method: str = "zscore",
        outlier_threshold: float = 3.0,
        fill_method: str = "forward",
        smoothing_window: int = 3,
    ):
        """
        Args:
            outlier_method: Method for outlier detection ('zscore', 'iqr', 'isolation')
            outlier_threshold: Threshold for outlier detection
            fill_method: Method for filling missing data ('forward', 'backward', 'interpolate', 'mean')
            smoothing_window: Window size for moving average smoothing
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.fill_method = fill_method
        self.smoothing_window = smoothing_window

    def remove_outliers(
        self, data: pd.Series, method: str = None, threshold: float = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Remove or correct outliers in data.

        Args:
            data: Series of data values
            method: Override default outlier method
            threshold: Override default threshold

        Returns:
            Tuple of (cleaned_series, info_dict)
        """
        method = method or self.outlier_method
        threshold = threshold or self.outlier_threshold

        if len(data) < 10:
            return data, {"removed_count": 0, "method": method}

        cleaned = data.copy()
        outlier_mask = pd.Series([False] * len(data), index=data.index)

        if method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_mask = z_scores > threshold

        elif method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)

        elif method == "isolation":
            try:
                from sklearn.ensemble import IsolationForest

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(data.values.reshape(-1, 1))
                outlier_mask = pd.Series(outliers == -1, index=data.index)
            except ImportError:
                # Fallback to zscore if sklearn not available
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_mask = z_scores > threshold

        # Replace outliers with NaN
        cleaned[outlier_mask] = np.nan

        return cleaned, {
            "removed_count": outlier_mask.sum(),
            "method": method,
            "threshold": threshold,
            "outlier_indices": outlier_mask[outlier_mask].index.tolist(),
        }

    def fill_missing_values(
        self, data: pd.Series, method: str = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Fill missing values in data.

        Args:
            data: Series with potential missing values
            method: Override default fill method

        Returns:
            Tuple of (filled_series, info_dict)
        """
        method = method or self.fill_method
        filled = data.copy()
        missing_count = filled.isna().sum()

        if missing_count == 0:
            return filled, {"filled_count": 0, "method": method}

        original_missing = missing_count

        if method == "forward":
            filled = filled.fillna(method="ffill")
        elif method == "backward":
            filled = filled.fillna(method="bfill")
        elif method == "interpolate":
            filled = filled.interpolate(method="time")
        elif method == "mean":
            filled = filled.fillna(filled.mean())
        else:
            filled = filled.fillna(method="ffill")

        # Fill any remaining NaN with backward fill
        filled = filled.fillna(method="bfill")

        return filled, {
            "filled_count": original_missing,
            "method": method,
            "remaining_na": filled.isna().sum(),
        }

    def smooth_data(
        self, data: pd.Series, window: int = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Apply moving average smoothing to reduce noise.

        Args:
            data: Series of data values
            window: Smoothing window size

        Returns:
            Tuple of (smoothed_series, info_dict)
        """
        window = window or self.smoothing_window

        if len(data) < window:
            return data, {"window": window, "smoothed": False}

        smoothed = data.rolling(window=window, center=True, min_periods=1).mean()

        return smoothed, {
            "window": window,
            "smoothed": True,
            "original_std": float(data.std()),
            "smoothed_std": float(smoothed.std()),
        }

    def correct_ohlc_inconsistencies(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Correct inconsistencies in OHLC data.

        Args:
            df: DataFrame with open, high, low, close columns

        Returns:
            Tuple of (corrected_df, info_dict)
        """
        required_cols = ["open", "high", "low", "close"]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            return df, {"corrected": False, "missing_columns": missing}

        corrected = df.copy()
        corrections = []

        # Ensure high >= max(open, close) and low <= min(open, close)
        expected_high = df[["open", "close"]].max(axis=1)
        expected_low = df[["open", "close"]].min(axis=1)

        # Fix high values
        high_fix_mask = corrected["high"] < expected_high
        if high_fix_mask.any():
            corrections.append(f"Fixed {high_fix_mask.sum()} low high values")
            corrected.loc[high_fix_mask, "high"] = expected_high[high_fix_mask]

        # Fix low values
        low_fix_mask = corrected["low"] > expected_low
        if low_fix_mask.any():
            corrections.append(f"Fixed {low_fix_mask.sum()} high low values")
            corrected.loc[low_fix_mask, "low"] = expected_low[low_fix_mask]

        return corrected, {
            "corrections": corrections,
            "corrected": len(corrections) > 0,
        }


class FreshnessChecker:
    """
    Check data freshness and timeliness.

    Features:
    - Calculate data age
    - Detect stale data
    - Track update frequency
    """

    def __init__(
        self,
        max_age_minutes: int = 10,
        warning_age_minutes: int = 5,
    ):
        """
        Args:
            max_age_minutes: Maximum acceptable data age in minutes
            warning_age_minutes: Age threshold for warnings
        """
        self.max_age_minutes = max_age_minutes
        self.warning_age_minutes = warning_age_minutes

    def check_freshness(
        self, timestamp: datetime, reference_time: datetime = None
    ) -> Dict:
        """
        Check if data is fresh.

        Args:
            timestamp: Data timestamp
            reference_time: Reference time (default: now)

        Returns:
            Dict with freshness information
        """
        reference = reference_time or datetime.now(timestamp.tzinfo)
        age = reference - timestamp

        age_minutes = age.total_seconds() / 60
        age_seconds = age.total_seconds()

        is_fresh = age_minutes <= self.max_age_minutes
        is_warning = age_minutes <= self.warning_age_minutes and age_minutes > 0

        freshness_pct = max(0, 100 * (1 - age_minutes / self.max_age_minutes))

        return {
            "is_fresh": is_fresh,
            "is_warning": is_warning,
            "age_minutes": age_minutes,
            "age_seconds": age_seconds,
            "max_age_minutes": self.max_age_minutes,
            "freshness_pct": freshness_pct,
            "timestamp": timestamp.isoformat(),
            "reference_time": reference.isoformat(),
        }

    def calculate_freshness_score(self, data: pd.DataFrame) -> float:
        """
        Calculate freshness score for a dataset.

        Args:
            data: DataFrame with DatetimeIndex

        Returns:
            Freshness score from 0 to 100
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return 0.0

        if len(data) == 0:
            return 0.0

        # Get most recent data point
        latest_timestamp = data.index.max()
        freshness_info = self.check_freshness(latest_timestamp.to_pydatetime())

        return freshness_info["freshness_pct"]

    def detect_stale_periods(
        self, data: pd.DataFrame, expected_interval_minutes: int = 5
    ) -> List[Dict]:
        """
        Detect periods where data is stale (not updating).

        Args:
            data: DataFrame with DatetimeIndex
            expected_interval_minutes: Expected data update interval

        Returns:
            List of stale periods
        """
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return []

        expected_interval = timedelta(minutes=expected_interval_minutes)
        time_diffs = data.index.to_series().diff()

        # Find gaps > 2x expected interval
        stale_threshold = expected_interval * 2
        stale_mask = time_diffs > stale_threshold

        stale_periods = []
        for idx in data.index[stale_mask][1:]:
            gap_size = time_diffs.loc[idx]
            prev_idx = data.index.get_loc(idx) - 1

            stale_periods.append(
                {
                    "start": str(data.index[prev_idx]),
                    "end": str(idx),
                    "gap_size": str(gap_size),
                    "gap_minutes": gap_size.total_seconds() / 60,
                }
            )

        return stale_periods


class CompletenessChecker:
    """
    Check data completeness.

    Features:
    - Check for missing required fields
    - Calculate completeness percentage
    - Detect missing time periods
    """

    def __init__(self, required_columns: List[str] = None):
        """
        Args:
            required_columns: List of required column names
        """
        self.required_columns = required_columns or [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def check_completeness(self, data: pd.DataFrame) -> Dict:
        """
        Check data completeness.

        Args:
            data: DataFrame to check

        Returns:
            Dict with completeness information
        """
        total_records = len(data)
        if total_records == 0:
            return {
                "completeness_pct": 0.0,
                "missing_columns": [],
                "complete_records": 0,
            }

        # Check for required columns
        present_columns = set(data.columns)
        required = set(self.required_columns)
        missing_columns = list(required - present_columns)

        # Check for missing values in each column
        column_completeness = {}
        for col in present_columns:
            na_count = data[col].isna().sum()
            complete_pct = 100 * (1 - na_count / total_records)
            column_completeness[col] = {
                "complete_pct": complete_pct,
                "missing_count": int(na_count),
            }

        # Calculate overall completeness
        total_cells = total_records * len(self.required_columns)
        missing_cells = sum(
            data[col].isna().sum() if col in data.columns else total_records
            for col in self.required_columns
        )
        completeness_pct = (
            100 * (1 - missing_cells / total_cells) if total_cells > 0 else 0
        )

        return {
            "completeness_pct": completeness_pct,
            "missing_columns": missing_columns,
            "column_completeness": column_completeness,
            "total_records": total_records,
            "complete_records": int(total_records - data.isna().any(axis=1).sum()),
        }

    def detect_missing_time_periods(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "5T",
    ) -> List[datetime]:
        """
        Detect missing time periods in expected range.

        Args:
            data: DataFrame with DatetimeIndex
            start_date: Expected start date
            end_date: Expected end date
            frequency: Expected frequency (pandas freq string)

        Returns:
            List of missing timestamps
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return []

        # Generate expected index
        expected_index = pd.date_range(start=start_date, end=end_date, freq=frequency)

        # Find missing timestamps
        missing_timestamps = expected_index.difference(data.index)

        return missing_timestamps.tolist()


class ConsistencyChecker:
    """
    Check data consistency across sources and within itself.

    Features:
    - Cross-source validation
    - Internal consistency checks
    - Temporal consistency
    """

    def __init__(self):
        """Initialize consistency checker."""
        pass

    def check_ohlc_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check OHLC relationships are consistent.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            Dict with consistency issues
        """
        required_cols = ["open", "high", "low", "close"]
        issues = []

        if not all(col in df.columns for col in required_cols):
            return {"consistent": False, "missing_columns": True}

        # Check high >= low
        high_low_issue = df["high"] < df["low"]
        if high_low_issue.any():
            issues.append(
                {
                    "type": "high_below_low",
                    "count": int(high_low_issue.sum()),
                    "indices": df[high_low_issue].index.tolist(),
                }
            )

        # Check high >= open and high >= close
        high_open_issue = df["high"] < df["open"]
        if high_open_issue.any():
            issues.append(
                {
                    "type": "high_below_open",
                    "count": int(high_open_issue.sum()),
                }
            )

        high_close_issue = df["high"] < df["close"]
        if high_close_issue.any():
            issues.append(
                {
                    "type": "high_below_close",
                    "count": int(high_close_issue.sum()),
                }
            )

        # Check low <= open and low <= close
        low_open_issue = df["low"] > df["open"]
        if low_open_issue.any():
            issues.append(
                {
                    "type": "low_above_open",
                    "count": int(low_open_issue.sum()),
                }
            )

        low_close_issue = df["low"] > df["close"]
        if low_close_issue.any():
            issues.append(
                {
                    "type": "low_above_close",
                    "count": int(low_close_issue.sum()),
                }
            )

        # Check for zero or negative prices
        zero_price = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        )
        if zero_price.any():
            issues.append(
                {
                    "type": "zero_or_negative_prices",
                    "count": int(zero_price.sum()),
                }
            )

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "issue_count": len(issues),
        }

    def check_cross_source_consistency(
        self, source1: pd.DataFrame, source2: pd.DataFrame, tolerance_pct: float = 0.01
    ) -> Dict:
        """
        Check consistency between two data sources.

        Args:
            source1: First data source
            source2: Second data source
            tolerance_pct: Acceptable difference percentage

        Returns:
            Dict with consistency information
        """
        # Align on common index
        common_index = source1.index.intersection(source2.index)

        if len(common_index) == 0:
            return {"consistent": False, "reason": "no_common_timestamps"}

        # Compare close prices
        s1_close = (
            source1.loc[common_index, "close"] if "close" in source1.columns else None
        )
        s2_close = (
            source2.loc[common_index, "close"] if "close" in source2.columns else None
        )

        if s1_close is None or s2_close is None:
            return {"consistent": False, "reason": "missing_close_column"}

        # Calculate differences
        diff_pct = abs((s1_close - s2_close) / s2_close * 100)
        inconsistent = diff_pct > (tolerance_pct * 100)

        return {
            "consistent": not inconsistent.any(),
            "tolerance_pct": tolerance_pct,
            "max_diff_pct": float(diff_pct.max()),
            "mean_diff_pct": float(diff_pct.mean()),
            "inconsistent_count": int(inconsistent.sum()),
            "total_compared": len(common_index),
        }

    def check_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check temporal consistency (no duplicates, chronological order).

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            Dict with temporal consistency information
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"temporal_consistency": False, "reason": "not_datetime_index"}

        issues = []

        # Check for duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            issues.append(
                {
                    "type": "duplicate_timestamps",
                    "count": int(duplicates.sum()),
                }
            )

        # Check if sorted
        is_sorted = df.index.is_monotonic_increasing
        if not is_sorted:
            issues.append(
                {
                    "type": "not_chronological",
                }
            )

        # Check for future timestamps
        now = pd.Timestamp.now()
        future = df.index > now
        if future.any():
            issues.append(
                {
                    "type": "future_timestamps",
                    "count": int(future.sum()),
                }
            )

        return {
            "temporal_consistency": len(issues) == 0,
            "is_sorted": is_sorted,
            "has_duplicates": duplicates.any(),
            "issues": issues,
        }


class DataQualityFramework:
    """
    Comprehensive data quality framework.

    Combines cleaning, validation, and scoring capabilities.
    """

    def __init__(
        self,
        quality_log_file: str = None,
        max_age_minutes: int = 10,
        outlier_threshold: float = 3.0,
    ):
        """
        Args:
            quality_log_file: Path to quality log JSON file
            max_age_minutes: Maximum acceptable data age
            outlier_threshold: Standard deviations for outlier detection
        """
        self.quality_log_file = quality_log_file or os.path.join(
            PROJECT_ROOT, "data_quality_log.json"
        )

        self.cleaner = DataCleaner(outlier_threshold=outlier_threshold)
        self.freshness_checker = FreshnessChecker(max_age_minutes=max_age_minutes)
        self.completeness_checker = CompletenessChecker()
        self.consistency_checker = ConsistencyChecker()

        self.quality_history: List[QualityReport] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load quality history from file."""
        if os.path.exists(self.quality_log_file):
            try:
                with open(self.quality_log_file) as f:
                    data = json.load(f)
                    for report_data in data.get("reports", []):
                        report = QualityReport(**report_data)
                        self.quality_history.append(report)
            except Exception as e:
                print(f"Error loading quality history: {e}")

    def _save_history(self) -> None:
        """Save quality history to file."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_reports": len(self.quality_history),
            "reports": [asdict(r) for r in self.quality_history],
        }

        with open(self.quality_log_file, "w") as f:
            json.dump(data, f, indent=2)

    def assess_quality(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        clean_data: bool = True,
    ) -> Tuple[Optional[pd.DataFrame], QualityReport]:
        """
        Comprehensive quality assessment and optional cleaning.

        Args:
            data: DataFrame to assess
            source: Data source identifier
            clean_data: Whether to clean the data

        Returns:
            Tuple of (cleaned_data or None, quality_report)
        """
        issues = []
        actions_taken = []
        recommendations = []

        working_data = data.copy()

        # 1. Freshness check
        freshness_score = self.freshness_checker.calculate_freshness_score(working_data)
        if freshness_score < 50:
            issues.append(
                {
                    "type": QualityIssue.STALE_DATA.value,
                    "severity": "high",
                    "message": f"Data freshness score: {freshness_score:.1f}%",
                }
            )
            recommendations.append("Check data feed - data may be stale")

        # 2. Completeness check
        completeness_result = self.completeness_checker.check_completeness(working_data)
        completeness_score = completeness_result["completeness_pct"]

        if completeness_result["missing_columns"]:
            issues.append(
                {
                    "type": QualityIssue.MISSING_DATA.value,
                    "severity": "critical",
                    "message": f"Missing columns: {completeness_result['missing_columns']}",
                }
            )

        if completeness_score < 90:
            issues.append(
                {
                    "type": QualityIssue.MISSING_DATA.value,
                    "severity": "medium",
                    "message": f"Completeness: {completeness_score:.1f}%",
                }
            )

        # 3. Consistency check
        consistency_result = self.consistency_checker.check_ohlc_consistency(
            working_data
        )
        temporal_result = self.consistency_checker.check_temporal_consistency(
            working_data
        )

        consistency_score = 100.0
        if not consistency_result["consistent"]:
            consistency_score -= 20 * consistency_result["issue_count"]
            for issue in consistency_result["issues"]:
                issues.append(
                    {
                        "type": QualityIssue.INCONSISTENT.value,
                        "severity": "medium",
                        "message": f"OHLC issue: {issue['type']}",
                    }
                )

        if not temporal_result["temporal_consistency"]:
            consistency_score -= 10
            for issue in temporal_result["issues"]:
                issues.append(
                    {
                        "type": QualityIssue.INCONSISTENT.value,
                        "severity": "low",
                        "message": f"Temporal issue: {issue['type']}",
                    }
                )

        # 4. Validity check (OHLC relationships valid)
        validity_score = 100.0 if consistency_result["consistent"] else 50.0

        # 5. Clean data if requested
        if clean_data:
            # Correct OHLC inconsistencies
            corrected, corr_info = self.cleaner.correct_ohlc_inconsistencies(
                working_data
            )
            if corr_info["corrected"]:
                actions_taken.append(
                    {
                        "action": "corrected_ohlc",
                        "details": corr_info["corrections"],
                    }
                )
                working_data = corrected

            # Handle outliers in close prices
            if "close" in working_data.columns:
                cleaned, clean_info = self.cleaner.remove_outliers(
                    working_data["close"]
                )
                if clean_info["removed_count"] > 0:
                    actions_taken.append(
                        {
                            "action": "removed_outliers",
                            "details": clean_info,
                        }
                    )
                    working_data["close"] = cleaned

            # Fill missing values
            for col in working_data.columns:
                if working_data[col].isna().any():
                    filled, fill_info = self.cleaner.fill_missing_values(
                        working_data[col]
                    )
                    actions_taken.append(
                        {
                            "action": "filled_missing",
                            "column": col,
                            "details": fill_info,
                        }
                    )
                    working_data[col] = filled

        # 6. Calculate overall score
        weights = {
            "freshness": 0.25,
            "completeness": 0.30,
            "consistency": 0.25,
            "validity": 0.20,
        }

        overall_score = (
            freshness_score * weights["freshness"]
            + completeness_score * weights["completeness"]
            + consistency_score * weights["consistency"]
            + validity_score * weights["validity"]
        )

        # Create report
        report = QualityReport(
            source=source,
            timestamp=datetime.now().isoformat(),
            record_count=len(data),
            overall_score=round(overall_score, 2),
            freshness_score=round(freshness_score, 2),
            completeness_score=round(completeness_score, 2),
            consistency_score=round(max(0, consistency_score), 2),
            validity_score=round(validity_score, 2),
            issues=issues,
            actions_taken=actions_taken,
            recommendations=recommendations,
        )

        # Save to history
        self.quality_history.append(report)
        self._save_history()

        return working_data if clean_data else None, report

    def get_quality_summary(self, last_n: int = 10) -> Dict:
        """
        Get summary of recent quality reports.

        Args:
            last_n: Number of recent reports to summarize

        Returns:
            Summary dict
        """
        recent = self.quality_history[-last_n:] if self.quality_history else []

        if not recent:
            return {"message": "No quality reports available"}

        avg_scores = {
            "overall": np.mean([r.overall_score for r in recent]),
            "freshness": np.mean([r.freshness_score for r in recent]),
            "completeness": np.mean([r.completeness_score for r in recent]),
            "consistency": np.mean([r.consistency_score for r in recent]),
            "validity": np.mean([r.validity_score for r in recent]),
        }

        total_issues = sum(len(r.issues) for r in recent)

        return {
            "period": f"Last {len(recent)} reports",
            "average_scores": {k: round(v, 2) for k, v in avg_scores.items()},
            "total_issues": total_issues,
            "total_actions": sum(len(r.actions_taken) for r in recent),
            "sources": list(set(r.source for r in recent)),
        }

    def get_quality_trend(self, days: int = 7) -> Dict:
        """
        Get quality score trend over time.

        Args:
            days: Number of days to analyze

        Returns:
            Trend dict with scores over time
        """
        cutoff = datetime.now() - timedelta(days=days)

        filtered = [
            r
            for r in self.quality_history
            if datetime.fromisoformat(r.timestamp) > cutoff
        ]

        if not filtered:
            return {"message": f"No reports in last {days} days"}

        return {
            "period_days": days,
            "report_count": len(filtered),
            "scores_over_time": [
                {
                    "timestamp": r.timestamp,
                    "overall_score": r.overall_score,
                    "source": r.source,
                }
                for r in filtered
            ],
            "trend": "improving"
            if filtered[-1].overall_score > filtered[0].overall_score
            else "declining",
        }


def assess_data_quality(
    data: pd.DataFrame,
    source: str = "unknown",
    clean: bool = False,
) -> Tuple[float, str]:
    """
    Convenience function to assess data quality.

    Args:
        data: DataFrame to assess
        source: Data source identifier
        clean: Whether to clean the data

    Returns:
        Tuple of (quality_score, summary_text)
    """
    framework = DataQualityFramework()
    _, report = framework.assess_quality(data, source=source, clean_data=clean)

    summary = f"Quality Score: {report.overall_score:.1f}/100\n"
    summary += f"  Freshness: {report.freshness_score:.1f}%\n"
    summary += f"  Completeness: {report.completeness_score:.1f}%\n"
    summary += f"  Consistency: {report.consistency_score:.1f}%\n"
    summary += f"  Validity: {report.validity_score:.1f}%\n"

    if report.issues:
        summary += f"\nIssues Found: {len(report.issues)}\n"
        for issue in report.issues[:5]:  # Show first 5
            summary += f"  - [{issue['severity'].upper()}] {issue['message']}\n"

    return report.overall_score, summary
