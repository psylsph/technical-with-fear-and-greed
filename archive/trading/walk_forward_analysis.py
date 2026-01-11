"""
Walk-Forward Analysis: Rolling optimization windows for strategy validation.
Tests strategy robustness across different time periods with rolling train/test splits.
"""

from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


class WalkForwardResult(Enum):
    """Result categories for walk-forward analysis."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float
    train_max_drawdown: float
    test_max_drawdown: float
    train_win_rate: float
    test_win_rate: float


class WalkForwardAnalyzer:
    """
    Perform walk-forward analysis to validate strategy robustness.

    Walk-forward analysis splits data into rolling windows:
    1. Train the strategy on the training window
    2. Test on the out-of-sample test window
    3. Roll forward and repeat

    This prevents look-ahead bias and tests strategy adaptability.
    """

    def __init__(
        self,
        train_period_months: int = 6,  # 6 months training data
        test_period_months: int = 1,  # 1 month test data
        step_period_months: int = 1,  # Roll forward by 1 month
        min_return_threshold: float = 0.0,  # Minimum acceptable return
        min_sharpe_threshold: float = 0.5,  # Minimum Sharpe ratio
        max_drawdown_threshold: float = 0.15,  # Maximum acceptable drawdown
    ):
        """
        Args:
            train_period_months: Training window size in months
            test_period_months: Test window size in months
            step_period_months: Step size for rolling window in months
            min_return_threshold: Minimum acceptable return (default 0%)
            min_sharpe_threshold: Minimum Sharpe ratio (default 0.5)
            max_drawdown_threshold: Maximum acceptable drawdown (default 15%)
        """
        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.step_period_months = step_period_months
        self.min_return_threshold = min_return_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
        self.max_drawdown_threshold = max_drawdown_threshold

    def generate_walk_forward_windows(
        self, data: pd.DataFrame, date_column: str = "timestamp"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate rolling train/test windows from data.

        Args:
            data: Historical price data with dates
            date_column: Name of date column

        Returns:
            List of (train_data, test_data) tuples
        """
        # Ensure data is sorted by date
        data = data.sort_values(date_column).copy()
        data[date_column] = pd.to_datetime(data[date_column])

        start_date = data[date_column].iloc[0]
        end_date = data[date_column].iloc[-1]

        windows = []
        current_start = start_date

        while True:
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_period_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_period_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Split data
            train_data = data[
                (data[date_column] >= train_start) & (data[date_column] < train_end)
            ].copy()

            test_data = data[
                (data[date_column] >= test_start) & (data[date_column] < test_end)
            ].copy()

            # Ensure both windows have sufficient data
            if len(train_data) > 100 and len(test_data) > 20:
                windows.append((train_data, test_data))

            # Roll forward
            current_start += pd.DateOffset(months=self.step_period_months)

        return windows

    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate performance metrics for a return series.

        Args:
            returns: Series of returns

        Returns:
            Dict with total_return, sharpe_ratio, max_drawdown, win_rate
        """
        if len(returns) == 0 or returns.isna().all():
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        # Total return
        total_return = returns.sum()

        # Sharpe ratio (annualized, assuming daily returns)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    def simulate_strategy(
        self, data: pd.DataFrame, signals: pd.Series, initial_capital: float = 10000.0
    ) -> pd.Series:
        """
        Simulate a simple strategy based on signals.

        Args:
            data: Price data with 'close' column
            signals: Series of signals (1=buy, -1=sell, 0=hold)
            initial_capital: Starting capital

        Returns:
            Series of returns
        """
        if len(signals) == 0 or len(data) == 0:
            return pd.Series([], dtype=float)

        # Align signals with data
        signals = signals.reindex(data.index).fillna(0)

        # Calculate returns
        returns = data["close"].pct_change().fillna(0)

        # Apply signals (long-only for simplicity)
        strategy_returns = returns * signals.shift(1).fillna(0)

        return strategy_returns

    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.Series] = None,
        initial_capital: float = 10000.0,
    ) -> Tuple[WalkForwardResult, List[WalkForwardWindow], Dict]:
        """
        Run complete walk-forward analysis.

        Args:
            data: Historical price data
            signal_generator: Function that generates trading signals from data
                              If None, uses a simple trend-following strategy
            initial_capital: Starting capital for simulation

        Returns:
            Tuple of (result: WalkForwardResult, windows: list, summary: dict)
        """
        windows = []

        # Generate walk-forward windows
        wf_windows = self.generate_walk_forward_windows(data)

        if not wf_windows:
            return (
                WalkForwardResult.INCONCLUSIVE,
                windows,
                {
                    "error": "Insufficient data for walk-forward analysis",
                    "total_windows": 0,
                },
            )

        # Default signal generator if none provided
        if signal_generator is None:

            def default_signal_generator(df):
                # Simple trend-following: buy when price > SMA(20)
                if len(df) < 20:
                    return pd.Series([0] * len(df), index=df.index)
                sma = df["close"].rolling(20).mean()
                signals = (df["close"] > sma).astype(int).replace(0, -1)
                return signals

            signal_generator = default_signal_generator

        # Test each window
        for i, (train_data, test_data) in enumerate(wf_windows):
            # Generate signals on training data
            train_signals = signal_generator(train_data)

            # Generate signals on test data
            test_signals = signal_generator(test_data)

            # Simulate strategy
            train_returns = self.simulate_strategy(
                train_data, train_signals, initial_capital
            )
            test_returns = self.simulate_strategy(
                test_data, test_signals, initial_capital
            )

            # Calculate metrics
            train_metrics = self.calculate_performance_metrics(train_returns)
            test_metrics = self.calculate_performance_metrics(test_returns)

            window = WalkForwardWindow(
                train_start=train_data["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                train_end=train_data["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
                test_start=test_data["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                test_end=test_data["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
                train_return=train_metrics["total_return"],
                test_return=test_metrics["total_return"],
                train_sharpe=train_metrics["sharpe_ratio"],
                test_sharpe=test_metrics["sharpe_ratio"],
                train_max_drawdown=train_metrics["max_drawdown"],
                test_max_drawdown=test_metrics["max_drawdown"],
                train_win_rate=train_metrics["win_rate"],
                test_win_rate=test_metrics["win_rate"],
            )
            windows.append(window)

        # Analyze overall results
        summary = self._analyze_results(windows)

        # Determine overall result
        result = self._determine_result(windows, summary)

        return result, windows, summary

    def _analyze_results(self, windows: List[WalkForwardWindow]) -> Dict:
        """Analyze walk-forward results across all windows."""
        if not windows:
            return {
                "total_windows": 0,
                "passing_windows": 0,
                "pass_rate": 0.0,
            }

        passing_windows = 0
        train_returns = []
        test_returns = []
        train_sharpe = []
        test_sharpe = []
        test_drawdowns = []

        for window in windows:
            train_returns.append(window.train_return)
            test_returns.append(window.test_return)
            train_sharpe.append(window.train_sharpe)
            test_sharpe.append(window.test_sharpe)
            test_drawdowns.append(window.test_max_drawdown)

            # Check if window passes all thresholds
            if (
                window.test_return >= self.min_return_threshold
                and window.test_sharpe >= self.min_sharpe_threshold
                and window.test_max_drawdown <= self.max_drawdown_threshold
            ):
                passing_windows += 1

        # Calculate consistency (correlation between train and test returns)
        if len(train_returns) > 1 and len(test_returns) > 1:
            consistency = np.corrcoef(train_returns, test_returns)[0, 1]
        else:
            consistency = 0.0

        return {
            "total_windows": len(windows),
            "passing_windows": passing_windows,
            "pass_rate": passing_windows / len(windows) if windows else 0.0,
            "avg_train_return": np.mean(train_returns) if train_returns else 0.0,
            "avg_test_return": np.mean(test_returns) if test_returns else 0.0,
            "std_test_return": np.std(test_returns) if test_returns else 0.0,
            "avg_train_sharpe": np.mean(train_sharpe) if train_sharpe else 0.0,
            "avg_test_sharpe": np.mean(test_sharpe) if test_sharpe else 0.0,
            "avg_test_drawdown": np.mean(test_drawdowns) if test_drawdowns else 0.0,
            "consistency": consistency,
            "min_test_return": min(test_returns) if test_returns else 0.0,
            "max_test_return": max(test_returns) if test_returns else 0.0,
        }

    def _determine_result(
        self, windows: List[WalkForwardWindow], summary: Dict
    ) -> WalkForwardResult:
        """Determine overall walk-forward result."""
        if not windows:
            return WalkForwardResult.INCONCLUSIVE

        # Need at least 3 windows
        if len(windows) < 3:
            return WalkForwardResult.INCONCLUSIVE

        # Pass rate > 60% and consistency > 0.3
        if summary["pass_rate"] >= 0.6 and summary["consistency"] >= 0.3:
            return WalkForwardResult.PASS

        # Pass rate < 40% or consistency < 0
        if summary["pass_rate"] < 0.4 or summary["consistency"] < 0:
            return WalkForwardResult.FAIL

        return WalkForwardResult.INCONCLUSIVE

    def get_walk_forward_report(
        self, result: WalkForwardResult, windows: List[WalkForwardWindow], summary: Dict
    ) -> str:
        """
        Generate human-readable walk-forward report.

        Args:
            result: Overall walk-forward result
            windows: List of walk-forward windows
            summary: Summary statistics

        Returns:
            Formatted report string
        """
        emoji = {
            WalkForwardResult.PASS: "✅",
            WalkForwardResult.FAIL: "❌",
            WalkForwardResult.INCONCLUSIVE: "⚠️",
        }.get(result, "❓")

        report = f"{emoji} Walk-Forward Analysis Results\n"
        report += f"{'='*50}\n\n"

        report += f"Overall Result: {result.value.upper()}\n"
        report += f"Total Windows: {summary['total_windows']}\n"
        report += f"Passing Windows: {summary['passing_windows']} ({summary['pass_rate']:.1%})\n\n"

        report += "Performance Metrics:\n"
        report += f"  Avg Train Return: {summary['avg_train_return']:.2%}\n"
        report += f"  Avg Test Return: {summary['avg_test_return']:.2%} ± {summary['std_test_return']:.2%}\n"
        report += f"  Test Return Range: [{summary['min_test_return']:.2%}, {summary['max_test_return']:.2%}]\n"
        report += f"  Avg Train Sharpe: {summary['avg_train_sharpe']:.2f}\n"
        report += f"  Avg Test Sharpe: {summary['avg_test_sharpe']:.2f}\n"
        report += f"  Avg Test Drawdown: {summary['avg_test_drawdown']:.2%}\n"
        report += f"  Train/Test Consistency: {summary['consistency']:.2f}\n\n"

        if windows:
            report += "Window Details:\n"
            for i, window in enumerate(windows, 1):
                train_emoji = (
                    "✅" if window.train_return >= self.min_return_threshold else "❌"
                )
                test_emoji = (
                    "✅" if window.test_return >= self.min_return_threshold else "❌"
                )

                report += (
                    f"  Window {i}: {window.train_start} to {window.test_end}\n"
                    f"    Train: {window.train_return:.2%} (Sharpe: {window.train_sharpe:.2f}) {train_emoji}\n"
                    f"    Test:  {window.test_return:.2%} (Sharpe: {window.test_sharpe:.2f}) {test_emoji}\n"
                )

        return report


def run_walk_forward_on_strategy(
    data: pd.DataFrame,
    train_period_months: int = 6,
    test_period_months: int = 1,
    step_period_months: int = 1,
) -> Tuple[WalkForwardResult, str]:
    """
    Convenience function to run walk-forward analysis on current strategy.

    Args:
        data: Historical price data with 'timestamp' and 'close' columns
        train_period_months: Training window size
        test_period_months: Test window size
        step_period_months: Step size for rolling

    Returns:
        Tuple of (result: WalkForwardResult, report: str)
    """
    analyzer = WalkForwardAnalyzer(
        train_period_months=train_period_months,
        test_period_months=test_period_months,
        step_period_months=step_period_months,
    )

    result, windows, summary = analyzer.run_walk_forward_analysis(data)
    report = analyzer.get_walk_forward_report(result, windows, summary)

    return result, report
