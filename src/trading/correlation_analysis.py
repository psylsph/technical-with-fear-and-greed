"""
Correlation Analysis: Track correlation between ETH and BTC for diversification insights.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT


class CorrelationAnalyzer:
    """Analyze correlation between assets for portfolio diversification."""

    def __init__(self, lookback_days: int = 30):
        """
        Args:
            lookback_days: Days to look back for correlation calculation (default 30)
        """
        self.lookback_days = lookback_days
        self.state_file = os.path.join(PROJECT_ROOT, "correlation_state.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load correlation state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "correlations": [],
            "last_update": None,
        }

    def _save_state(self):
        """Save correlation state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def calculate_rolling_correlation(
        self,
        price_series_1: pd.Series,
        price_series_2: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling correlation between two price series.

        Args:
            price_series_1: First price series (e.g., ETH)
            price_series_2: Second price series (e.g., BTC)
            window: Rolling window size (default 20)

        Returns:
            Series of rolling correlations
        """
        # Calculate returns
        returns_1 = price_series_1.pct_change().dropna()
        returns_2 = price_series_2.pct_change().dropna()

        # Align series
        aligned_returns = pd.DataFrame({
            "returns_1": returns_1,
            "returns_2": returns_2
        }).dropna()

        if len(aligned_returns) < window:
            return pd.Series([], dtype=float)

        # Calculate rolling correlation
        rolling_corr = aligned_returns["returns_1"].rolling(
            window=window
        ).corr(aligned_returns["returns_2"])

        return rolling_corr

    def calculate_correlation_metrics(
        self,
        price_series_1: pd.Series,
        price_series_2: pd.Series,
        asset_1_name: str = "Asset1",
        asset_2_name: str = "Asset2"
    ) -> Dict:
        """
        Calculate comprehensive correlation metrics between two assets.

        Args:
            price_series_1: First price series
            price_series_2: Second price series
            asset_1_name: Name of first asset
            asset_2_name: Name of second asset

        Returns:
            Dict with correlation metrics
        """
        # Calculate returns
        returns_1 = price_series_1.pct_change().dropna()
        returns_2 = price_series_2.pct_change().dropna()

        # Align series
        aligned_returns = pd.DataFrame({
            f"{asset_1_name}_returns": returns_1,
            f"{asset_2_name}_returns": returns_2
        }).dropna()

        if len(aligned_returns) < 2:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "sample_size": 0,
                "interpretation": "Insufficient data"
            }

        # Calculate Pearson correlation
        correlation = aligned_returns.iloc[:, 0].corr(aligned_returns.iloc[:, 1])

        # Calculate p-value (simplified - assumes normal distribution)
        n = len(aligned_returns)
        if n > 2:
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0

        # Interpret correlation strength
        if abs(correlation) >= 0.8:
            interpretation = "Strong"
        elif abs(correlation) >= 0.5:
            interpretation = "Moderate"
        elif abs(correlation) >= 0.3:
            interpretation = "Weak"
        else:
            interpretation = "Very Weak"

        direction = "Positive" if correlation > 0 else "Negative"

        # Calculate beta (asset 1 sensitivity to asset 2)
        covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
        variance_2 = aligned_returns.iloc[:, 1].var()

        beta = covariance / variance_2 if variance_2 > 0 else 0.0

        # Calculate rolling correlations for different windows
        rolling_correlations = {}
        for window in [7, 14, 30]:
            if len(aligned_returns) >= window:
                rolling_corr = aligned_returns.iloc[:, 0].rolling(
                    window=window
                ).corr(aligned_returns.iloc[:, 1])
                rolling_correlations[f"{window}_day"] = float(rolling_corr.iloc[-1]) if not pd.isna(rolling_corr.iloc[-1]) else 0.0

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "sample_size": n,
            "interpretation": f"{interpretation} {direction}",
            "beta": float(beta),
            "rolling_correlations": rolling_correlations,
            "is_significant": p_value < 0.05,
            "diversification_benefit": abs(correlation) < 0.7,
        }

    def analyze_eth_btc_correlation(
        self,
        eth_prices: pd.Series,
        btc_prices: pd.Series,
        save_to_state: bool = True
    ) -> Dict:
        """
        Analyze correlation between ETH and BTC.

        Args:
            eth_prices: ETH price series
            btc_prices: BTC price series
            save_to_state: Whether to save results to state file

        Returns:
            Dict with correlation analysis results
        """
        result = self.calculate_correlation_metrics(
            eth_prices,
            btc_prices,
            "ETH",
            "BTC"
        )

        result["analysis_time"] = datetime.now().isoformat()
        result["current_eth_price"] = float(eth_prices.iloc[-1]) if len(eth_prices) > 0 else None
        result["current_btc_price"] = float(btc_prices.iloc[-1]) if len(btc_prices) > 0 else None

        # Calculate 30-day correlation trend
        if len(eth_prices) >= 30 and len(btc_prices) >= 30:
            eth_recent = eth_prices.iloc[-30:]
            btc_recent = btc_prices.iloc[-30:]

            rolling_correlations = []
            for i in range(10, 30):
                eth_window = eth_recent.iloc[i-10:i]
                btc_window = btc_recent.iloc[i-10:i]

                corr = eth_window.pct_change().corr(btc_window.pct_change())
                if not pd.isna(corr):
                    rolling_correlations.append(corr)

            if rolling_correlations:
                corr_trend = "increasing" if rolling_correlations[-1] > rolling_correlations[0] else "decreasing"
                result["correlation_trend"] = corr_trend
                result["recent_correlations"] = [float(c) for c in rolling_correlations[-5:]]

        # Save to state if requested
        if save_to_state:
            self.state["correlations"].append({
                "timestamp": result["analysis_time"],
                "correlation": result["correlation"],
                "interpretation": result["interpretation"],
                "is_significant": result["is_significant"],
                "diversification_benefit": result["diversification_benefit"],
            })

            # Keep only last 100 correlations
            if len(self.state["correlations"]) > 100:
                self.state["correlations"] = self.state["correlations"][-100:]

            self.state["last_update"] = datetime.now().isoformat()
            self._save_state()

        return result

    def get_correlation_history(self, limit: int = 30) -> List[Dict]:
        """
        Get historical correlation data.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of historical correlation records
        """
        return self.state.get("correlations", [])[-limit:]

    def get_average_correlation(self, days: int = 30) -> Optional[float]:
        """
        Get average correlation over recent days.

        Args:
            days: Number of days to average

        Returns:
            Average correlation or None if insufficient data
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        recent_correlations = [
            c for c in self.state.get("correlations", [])
            if datetime.fromisoformat(c["timestamp"]) > cutoff_time
        ]

        if not recent_correlations:
            return None

        return sum(c["correlation"] for c in recent_correlations) / len(recent_correlations)

    def assess_diversification_benefit(
        self,
        correlation: float,
        threshold: float = 0.7
    ) -> Dict:
        """
        Assess whether adding an asset provides diversification benefit.

        Args:
            correlation: Correlation coefficient
            threshold: Correlation threshold (default 0.7)

        Returns:
            Dict with diversification assessment
        """
        abs_corr = abs(correlation)

        if abs_corr < 0.3:
            level = "Excellent"
            benefit = "Strong diversification benefit"
        elif abs_corr < 0.5:
            level = "Good"
            benefit = "Moderate diversification benefit"
        elif abs_corr < threshold:
            level = "Fair"
            benefit = "Limited diversification benefit"
        else:
            level = "Poor"
            benefit = "Minimal diversification benefit (assets move together)"

        return {
            "level": level,
            "benefit": benefit,
            "recommended": abs_corr < threshold,
            "correlation_abs": abs_corr,
        }
