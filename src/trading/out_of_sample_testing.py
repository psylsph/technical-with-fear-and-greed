"""
Out-of-Sample Testing: Train on historical data, test on recent data.
Validates strategy performance on unseen data to detect overfitting.
"""

from typing import Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


class TestResult(Enum):
    """Result categories for out-of-sample testing."""
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    total_trades: int


class OutOfSampleTester:
    """
    Out-of-sample testing for strategy validation.

    Splits data into training and testing periods:
    1. Train/optimize strategy on training data
    2. Test on unseen out-of-sample data
    3. Compare performance to detect overfitting
    """

    def __init__(
        self,
        min_train_periods: int = 252,  # ~1 year of daily data
        min_test_periods: int = 63,     # ~3 months of daily data
        performance_decay_threshold: float = 0.5,  # Test perf should be >= 50% of train
        sharpe_threshold: float = 0.5,   # Minimum Sharpe ratio
    ):
        """
        Args:
            min_train_periods: Minimum periods in training data
            min_test_periods: Minimum periods in test data
            performance_decay_threshold: Max acceptable performance decay (train to test)
            sharpe_threshold: Minimum acceptable Sharpe ratio
        """
        self.min_train_periods = min_train_periods
        self.min_test_periods = min_test_periods
        self.performance_decay_threshold = performance_decay_threshold
        self.sharpe_threshold = sharpe_threshold

    def split_data(
        self,
        data: pd.DataFrame,
        train_end_date: str,
        date_column: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets by date.

        Args:
            data: Historical data with dates
            train_end_date: Last date to include in training data
            date_column: Name of date column

        Returns:
            Tuple of (train_data, test_data)
        """
        data = data.sort_values(date_column).copy()
        data[date_column] = pd.to_datetime(data[date_column])
        train_end = pd.to_datetime(train_end_date)

        train_data = data[data[date_column] <= train_end].copy()
        test_data = data[data[date_column] > train_end].copy()

        return train_data, test_data

    def split_by_ratio(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        date_column: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets by ratio (chronological).

        Args:
            data: Historical data with dates
            train_ratio: Fraction of data to use for training (default 0.7)
            date_column: Name of date column

        Returns:
            Tuple of (train_data, test_data)
        """
        data = data.sort_values(date_column).copy()
        split_idx = int(len(data) * train_ratio)

        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        return train_data, test_data

    def calculate_metrics(
        self,
        returns: pd.Series
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Series of returns

        Returns:
            PerformanceMetrics object
        """
        if len(returns) == 0 or returns.isna().all():
            return PerformanceMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                total_trades=0,
            )

        # Total return
        total_return = returns.sum()

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        total_trades = (returns != 0).sum()
        win_rate = (returns > 0).sum() / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average win/loss
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.0

        # Best/worst trades
        best_trade = returns.max()
        worst_trade = returns.min()

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_trades=int(total_trades),
        )

    def simulate_strategy(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.Series],
    ) -> pd.Series:
        """
        Simulate strategy returns based on signals.

        Args:
            data: Price data with 'close' column
            signal_generator: Function that generates trading signals

        Returns:
            Series of returns
        """
        if len(data) == 0:
            return pd.Series([], dtype=float)

        # Generate signals
        signals = signal_generator(data)

        # Calculate returns
        returns = data["close"].pct_change().fillna(0)

        # Apply signals (long-only for simplicity)
        strategy_returns = returns * signals.shift(1).fillna(0)

        return strategy_returns

    def run_out_of_sample_test(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.Series],
        train_end_date: str = None,
        train_ratio: float = None,
    ) -> Tuple[TestResult, PerformanceMetrics, PerformanceMetrics, Dict]:
        """
        Run complete out-of-sample test.

        Args:
            data: Historical price data
            signal_generator: Function that generates trading signals
            train_end_date: Optional specific date to split data
            train_ratio: Optional ratio to split data (default 0.7)

        Returns:
            Tuple of (result, train_metrics, test_metrics, comparison_dict)
        """
        # Split data
        if train_end_date:
            train_data, test_data = self.split_data(data, train_end_date)
        elif train_ratio:
            train_data, test_data = self.split_by_ratio(data, train_ratio or 0.7)
        else:
            train_data, test_data = self.split_by_ratio(data, 0.7)

        # Check minimum data requirements
        if len(train_data) < self.min_train_periods:
            return (
                TestResult.INCONCLUSIVE,
                PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                {"error": f"Insufficient training data: {len(train_data)} < {self.min_train_periods}"}
            )

        if len(test_data) < self.min_test_periods:
            return (
                TestResult.INCONCLUSIVE,
                PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                {"error": f"Insufficient test data: {len(test_data)} < {self.min_test_periods}"}
            )

        # Simulate strategy on both periods
        train_returns = self.simulate_strategy(train_data, signal_generator)
        test_returns = self.simulate_strategy(test_data, signal_generator)

        # Calculate metrics
        train_metrics = self.calculate_metrics(train_returns)
        test_metrics = self.calculate_metrics(test_returns)

        # Compare performance
        comparison = self._compare_performance(train_metrics, test_metrics)

        # Determine result
        result = self._determine_result(train_metrics, test_metrics, comparison)

        return result, train_metrics, test_metrics, comparison

    def _compare_performance(
        self,
        train_metrics: PerformanceMetrics,
        test_metrics: PerformanceMetrics
    ) -> Dict:
        """Compare train vs test performance."""
        # Calculate performance ratios (test / train)
        def safe_ratio(test_val: float, train_val: float) -> float:
            if train_val == 0:
                return 0.0 if test_val == 0 else (1.0 if test_val > 0 else -1.0)
            return test_val / train_val

        return {
            "return_ratio": safe_ratio(test_metrics.total_return, train_metrics.total_return),
            "sharpe_ratio": safe_ratio(test_metrics.sharpe_ratio, train_metrics.sharpe_ratio),
            "drawdown_ratio": safe_ratio(test_metrics.max_drawdown, train_metrics.max_drawdown),
            "win_rate_ratio": safe_ratio(test_metrics.win_rate, train_metrics.win_rate),
            "return_difference": test_metrics.total_return - train_metrics.total_return,
            "sharpe_difference": test_metrics.sharpe_ratio - train_metrics.sharpe_ratio,
        }

    def _determine_result(
        self,
        train_metrics: PerformanceMetrics,
        test_metrics: PerformanceMetrics,
        comparison: Dict
    ) -> TestResult:
        """Determine overall test result."""
        # Check minimum Sharpe threshold
        if test_metrics.sharpe_ratio < self.sharpe_threshold:
            return TestResult.FAIL

        # Check if test performance is acceptable compared to train
        if comparison["return_ratio"] < self.performance_decay_threshold:
            return TestResult.FAIL

        # Pass if both train and test have positive returns
        if train_metrics.total_return > 0 and test_metrics.total_return > 0:
            return TestResult.PASS

        return TestResult.INCONCLUSIVE

    def get_test_report(
        self,
        result: TestResult,
        train_metrics: PerformanceMetrics,
        test_metrics: PerformanceMetrics,
        comparison: Dict
    ) -> str:
        """Generate human-readable out-of-sample test report."""
        emoji = {
            TestResult.PASS: "✅",
            TestResult.FAIL: "❌",
            TestResult.INCONCLUSIVE: "⚠️",
        }.get(result, "❓")

        report = f"{emoji} Out-of-Sample Test Results\n"
        report += f"{'='*50}\n\n"

        report += f"Overall Result: {result.value.upper()}\n\n"

        report += "Training Performance:\n"
        report += f"  Total Return: {train_metrics.total_return:.2%}\n"
        report += f"  Sharpe Ratio: {train_metrics.sharpe_ratio:.2f}\n"
        report += f"  Sortino Ratio: {train_metrics.sortino_ratio:.2f}\n"
        report += f"  Max Drawdown: {train_metrics.max_drawdown:.2%}\n"
        report += f"  Win Rate: {train_metrics.win_rate:.1%}\n"
        report += f"  Profit Factor: {train_metrics.profit_factor:.2f}\n"
        report += f"  Total Trades: {train_metrics.total_trades}\n\n"

        report += "Test Performance:\n"
        report += f"  Total Return: {test_metrics.total_return:.2%}\n"
        report += f"  Sharpe Ratio: {test_metrics.sharpe_ratio:.2f}\n"
        report += f"  Sortino Ratio: {test_metrics.sortino_ratio:.2f}\n"
        report += f"  Max Drawdown: {test_metrics.max_drawdown:.2%}\n"
        report += f"  Win Rate: {test_metrics.win_rate:.1%}\n"
        report += f"  Profit Factor: {test_metrics.profit_factor:.2f}\n"
        report += f"  Total Trades: {test_metrics.total_trades}\n\n"

        report += "Performance Comparison:\n"
        report += f"  Return Ratio (Test/Train): {comparison['return_ratio']:.2%}\n"
        report += f"  Sharpe Ratio (Test/Train): {comparison['sharpe_ratio']:.2%}\n"
        report += f"  Return Difference: {comparison['return_difference']:+.2%}\n"
        report += f"  Sharpe Difference: {comparison['sharpe_difference']:+.2f}\n"

        # Interpretation
        report += "\nInterpretation:\n"
        if comparison["return_ratio"] >= 1.0:
            report += "  • Test outperformed training - very good!\n"
        elif comparison["return_ratio"] >= 0.8:
            report += "  • Test performance close to training - good generalization.\n"
        elif comparison["return_ratio"] >= 0.5:
            report += "  • Some performance decay - acceptable but monitor.\n"
        else:
            report += "  • Significant performance decay - possible overfitting!\n"

        if comparison["sharpe_difference"] > 0:
            report += "  • Test Sharpe higher than train - positive sign.\n"
        elif comparison["sharpe_difference"] > -0.5:
            report += "  • Test Sharpe comparable to train - acceptable.\n"
        else:
            report += "  • Test Sharpe much lower than train - concerning.\n"

        return report


def run_oos_test(
    data: pd.DataFrame,
    signal_generator: Callable[[pd.DataFrame], pd.Series],
    train_ratio: float = 0.7
) -> Tuple[TestResult, str]:
    """
    Convenience function to run out-of-sample test.

    Args:
        data: Historical price data with 'timestamp' and 'close' columns
        signal_generator: Function that generates trading signals
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of (result: TestResult, report: str)
    """
    tester = OutOfSampleTester()

    result, train_metrics, test_metrics, comparison = tester.run_out_of_sample_test(
        data,
        signal_generator,
        train_ratio=train_ratio
    )
    report = tester.get_test_report(result, train_metrics, test_metrics, comparison)

    return result, report
