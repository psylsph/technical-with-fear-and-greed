"""
Data Validation: Outlier detection and gap analysis for market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class DataValidator:
    """Validate market data for outliers and gaps."""

    def __init__(
        self,
        outlier_std_threshold: float = 3.0,
        max_gap_pct: float = 0.10,
        max_price_change_pct: float = 0.20,
    ):
        """
        Args:
            outlier_std_threshold: Standard deviations for outlier detection (default 3σ)
            max_gap_pct: Maximum allowed gap as percentage of price (default 10%)
            max_price_change_pct: Maximum allowed single-period price change (default 20%)
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.max_gap_pct = max_gap_pct
        self.max_price_change_pct = max_price_change_pct

    def detect_outliers(self, price_series: pd.Series) -> Dict:
        """
        Detect price outliers using statistical methods.

        Args:
            price_series: Series of prices

        Returns:
            Dict with outlier information
        """
        if len(price_series) < 10:
            return {"outliers": [], "outlier_indices": [], "outlier_count": 0}

        # Calculate returns
        returns = price_series.pct_change().dropna()

        # Calculate z-scores
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return {"outliers": [], "outlier_indices": [], "outlier_count": 0}

        z_scores = (returns - mean_return) / std_return

        # Find outliers (beyond threshold standard deviations)
        outlier_mask = np.abs(z_scores) > self.outlier_std_threshold
        outlier_indices = returns[outlier_mask].index.tolist()
        outlier_values = price_series[outlier_mask].tolist()
        outlier_returns = returns[outlier_mask].tolist()

        outliers = [
            {
                "timestamp": str(idx),
                "price": float(price),
                "return": float(ret),
                "z_score": float(z_scores[idx]),
            }
            for idx, price, ret in zip(outlier_indices, outlier_values, outlier_returns)
        ]

        return {
            "outliers": outliers,
            "outlier_indices": [str(i) for i in outlier_indices],
            "outlier_count": len(outliers),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "threshold": self.outlier_std_threshold,
        }

    def detect_gaps(
        self, price_series: pd.Series, volume_series: pd.Series = None
    ) -> Dict:
        """
        Detect data gaps in time series.

        Args:
            price_series: Series of prices with DatetimeIndex
            volume_series: Optional series of volumes

        Returns:
            Dict with gap information
        """
        if not isinstance(price_series.index, pd.DatetimeIndex):
            return {"gaps": [], "gap_count": 0, "message": "Not a DatetimeIndex"}

        # Detect time gaps
        if len(price_series) < 2:
            return {"gaps": [], "gap_count": 0}

        # Calculate time differences
        time_diffs = price_series.index.to_series().diff()

        # Estimate expected frequency (most common difference)
        mode_diff = (
            time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        )

        # Find gaps (differences > 2x expected frequency)
        gap_mask = time_diffs > (mode_diff * 2)
        gap_indices = time_diffs[gap_mask].index.tolist()

        gaps = []
        for idx in gap_indices:
            prev_idx = price_series.index.get_loc(idx) - 1
            if prev_idx >= 0:
                gap_size = time_diffs.loc[idx]
                expected_size = mode_diff
                gap_ratio = gap_size / expected_size if expected_size > 0 else 0

                gaps.append(
                    {
                        "start": str(price_series.index[prev_idx]),
                        "end": str(idx),
                        "gap_size": str(gap_size),
                        "gap_ratio": float(gap_ratio),
                        "missing_periods": int(gap_ratio / 1.0),  # Approximate
                    }
                )

        # Detect price gaps (abnormal price jumps)
        price_changes = price_series.pct_change().abs()
        price_gap_mask = price_changes > self.max_gap_pct
        price_gap_indices = price_changes[price_gap_mask].index.tolist()

        price_gaps = []
        for idx in price_gap_indices:
            loc = price_series.index.get_loc(idx)
            if loc > 0:
                prev_price = price_series.iloc[loc - 1]
                curr_price = price_series.iloc[loc]
                gap_pct = float((curr_price - prev_price) / prev_price * 100)

                price_gaps.append(
                    {
                        "timestamp": str(idx),
                        "previous_price": float(prev_price),
                        "current_price": float(curr_price),
                        "gap_pct": gap_pct,
                    }
                )

        return {
            "time_gaps": gaps,
            "time_gap_count": len(gaps),
            "price_gaps": price_gaps,
            "price_gap_count": len(price_gaps),
            "total_gap_count": len(gaps) + len(price_gaps),
            "expected_frequency": str(mode_diff),
        }

    def validate_price_data(
        self, ohlcv_data: pd.DataFrame
    ) -> Tuple[bool, List[str], Dict]:
        """
        Validate OHLCV data for quality issues.

        Args:
            ohlcv_data: DataFrame with OHLCV columns

        Returns:
            Tuple of (is_valid: bool, warnings: list, details: dict)
        """
        warnings = []
        details = {}

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [
            col for col in required_columns if col not in ohlcv_data.columns
        ]

        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")
            return False, warnings, {}

        # Check for OHLC consistency
        invalid_ohlc = (
            (ohlcv_data["high"] < ohlcv_data["low"])
            | (ohlcv_data["high"] < ohlcv_data["open"])
            | (ohlcv_data["high"] < ohlcv_data["close"])
            | (ohlcv_data["low"] > ohlcv_data["open"])
            | (ohlcv_data["low"] > ohlcv_data["close"])
        )

        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            warnings.append(f"{invalid_count} bars with invalid OHLC relationships")

        # Check for zero or negative prices
        zero_prices = (
            (ohlcv_data["open"] <= 0)
            | (ohlcv_data["high"] <= 0)
            | (ohlcv_data["low"] <= 0)
            | (ohlcv_data["close"] <= 0)
        ).sum()

        if zero_prices > 0:
            warnings.append(f"{zero_prices} bars with zero or negative prices")

        # Detect outliers
        if "close" in ohlcv_data.columns:
            outlier_result = self.detect_outliers(ohlcv_data["close"])
            details["outliers"] = outlier_result

            if outlier_result["outlier_count"] > 0:
                warnings.append(
                    f"{outlier_result['outlier_count']} price outliers detected "
                    f"(>{self.outlier_std_threshold}σ from mean)"
                )

        # Detect gaps
        gap_result = self.detect_gaps(ohlcv_data["close"])
        details["gaps"] = gap_result

        if gap_result["total_gap_count"] > 0:
            warnings.append(
                f"{gap_result['time_gap_count']} time gaps, "
                f"{gap_result['price_gap_count']} price gaps detected"
            )

        # Check for extreme price changes
        if "close" in ohlcv_data.columns:
            extreme_changes = (
                ohlcv_data["close"].pct_change().abs() > self.max_price_change_pct
            )
            extreme_count = extreme_changes.sum()

            if extreme_count > 0:
                warnings.append(
                    f"{extreme_count} extreme price changes "
                    f"(>{self.max_price_change_pct * 100:.0f}%)"
                )

        # Check volume if available
        if "volume" in ohlcv_data.columns:
            zero_volume = (ohlcv_data["volume"] <= 0).sum()
            if zero_volume > 0:
                warnings.append(f"{zero_volume} bars with zero or negative volume")

        is_valid = len(warnings) == 0
        return is_valid, warnings, details

    def get_data_quality_score(self, ohlcv_data: pd.DataFrame) -> float:
        """
        Calculate a data quality score (0-100).

        Args:
            ohlcv_data: DataFrame with OHLCV columns

        Returns:
            Quality score from 0 (poor) to 100 (excellent)
        """
        is_valid, warnings, details = self.validate_price_data(ohlcv_data)

        base_score = 100.0

        # Deduct points for each warning
        deductions = {
            "missing": 30,
            "invalid_ohlc": 20,
            "zero_prices": 25,
            "outliers": 5,
            "gaps": 10,
            "extreme_changes": 10,
            "zero_volume": 5,
        }

        for warning in warnings:
            if "Missing" in warning:
                base_score -= deductions["missing"]
            elif "invalid OHLC" in warning:
                base_score -= min(deductions["invalid_ohlc"], base_score)
            elif "zero or negative prices" in warning:
                base_score -= min(deductions["zero_prices"], base_score)
            elif "outliers" in warning:
                base_score -= min(
                    deductions["outliers"]
                    * details.get("outliers", {}).get("outlier_count", 1),
                    base_score,
                )
            elif "gaps" in warning:
                base_score -= min(deductions["gaps"], base_score)
            elif "extreme price changes" in warning:
                base_score -= min(deductions["extreme_changes"], base_score)
            elif "zero or negative volume" in warning:
                base_score -= min(deductions["zero_volume"], base_score)

        return max(0.0, base_score)
