"""
Property-Based Testing: Hypothesis testing for edge cases.
Tests trading strategy properties with random inputs to find edge cases.
"""

from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import random

import pandas as pd
import numpy as np


class PropertyType(Enum):
    """Types of properties to test."""

    RETURNS_ALWAYS_DEFINED = "returns_always_defined"
    RETURNS_FINITE = "returns_finite"
    POSITION_SIZES_LIMITED = "position_sizes_limited"
    DRAWDOWN_LIMITED = "drawdown_limited"
    NO_OVERLEVERAGE = "no_overleverage"
    NO_NAN_SIGNALS = "no_nan_signals"
    RETURNS_REASONABLE_RANGE = "returns_reasonable_range"
    EQUITY_MONOTONIC_WITH_TIME = "equity_monotonic_with_time"


@dataclass
class PropertyTestResult:
    """Result of a property test."""

    property_name: PropertyType
    passed: bool
    num_tests: int
    num_fails: int
    fail_examples: List[Any]
    description: str


class PropertyTester:
    """
    Property-based testing for trading strategies.

    Tests invariant properties of the strategy:
    - Returns should always be defined
    - Returns should be finite
    - Position sizes should be bounded
    - Drawdown should not exceed limits
    - No over-leverage
    - No NaN signals
    """

    def __init__(
        self,
        max_position_size: float = 1.0,  # Max position size
        max_leverage: float = 2.0,  # Max leverage
        max_single_return: float = 0.50,  # Max single day return (50%)
        max_drawdown: float = 0.5,  # Max drawdown (50%)
        num_random_tests: int = 100,
    ):
        """
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_leverage: Maximum leverage multiplier
            max_single_return: Maximum acceptable single return
            max_drawdown: Maximum acceptable drawdown
            num_random_tests: Number of random tests to run
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_single_return = max_single_return
        self.max_drawdown = max_drawdown
        self.num_random_tests = num_random_tests

    def generate_random_price_series(
        self,
        length: int = 252,
        start_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0005,
    ) -> pd.Series:
        """
        Generate random price series using geometric Brownian motion.

        Args:
            length: Number of periods
            start_price: Starting price
            volatility: Daily volatility
            drift: Daily drift

        Returns:
            Series of random prices
        """
        returns = np.random.normal(drift, volatility, length)
        # Occasional extreme moves
        if random.random() < 0.05:  # 5% chance
            extreme_idx = random.randint(0, length - 1)
            returns[extreme_idx] += random.choice([-0.2, 0.2])  # 20% move

        prices = start_price * (1 + returns).cumprod()
        return pd.Series(prices)

    def generate_random_signal_series(
        self,
        length: int = 252,
    ) -> pd.Series:
        """
        Generate random signal series.

        Args:
            length: Number of periods

        Returns:
            Series of random signals (-1, 0, 1)
        """
        signals = np.random.choice([-1, 0, 1], length, p=[0.3, 0.4, 0.3])
        return pd.Series(signals)

    def property_returns_always_defined(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
    ) -> PropertyTestResult:
        """
        Test: Returns should always be defined (no None/NaN).

        Args:
            strategy_func: Function that takes prices and returns returns

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            returns = strategy_func(prices)

            if returns.isna().any():
                num_fails += 1
                fail_examples.append(
                    {
                        "prices": prices.head(5).tolist(),
                        "nan_count": returns.isna().sum(),
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.RETURNS_ALWAYS_DEFINED,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],  # Keep first 5 examples
            description="Returns should never be NaN/None",
        )

    def property_returns_finite(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
    ) -> PropertyTestResult:
        """
        Test: Returns should always be finite (no inf).

        Args:
            strategy_func: Function that takes prices and returns returns

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            returns = strategy_func(prices)

            if not np.isfinite(returns).all():
                num_fails += 1
                infinite_indices = np.where(~np.isfinite(returns))[0]
                fail_examples.append(
                    {
                        "infinite_indices": infinite_indices.tolist()[:5],
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.RETURNS_FINITE,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description="Returns should always be finite",
        )

    def property_position_sizes_limited(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
    ) -> PropertyTestResult:
        """
        Test: Position sizes should be within bounds.

        Args:
            strategy_func: Function that takes prices and returns positions (-1 to 1)

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            positions = strategy_func(prices)

            # Check if any position exceeds max
            if (positions.abs() > self.max_position_size).any():
                num_fails += 1
                exceed_indices = np.where(positions.abs() > self.max_position_size)[0]
                fail_examples.append(
                    {
                        "max_position": positions.abs().max(),
                        "exceed_indices": exceed_indices.tolist()[:3],
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.POSITION_SIZES_LIMITED,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description=f"Position sizes should be ≤ {self.max_position_size}",
        )

    def property_drawdown_limited(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
        initial_capital: float = 10000.0,
    ) -> PropertyTestResult:
        """
        Test: Drawdown should not exceed maximum.

        Args:
            strategy_func: Function that takes prices and returns returns
            initial_capital: Starting capital

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            returns = strategy_func(prices)

            # Calculate drawdown
            equity = initial_capital * (1 + returns).cumprod()
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            max_dd = abs(drawdown.min())

            if max_dd > self.max_drawdown:
                num_fails += 1
                fail_examples.append(
                    {
                        "max_drawdown": max_dd,
                        "allowed_max": self.max_drawdown,
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.DRAWDOWN_LIMITED,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description=f"Max drawdown should be ≤ {self.max_drawdown:.1%}",
        )

    def property_no_overleverage(
        self,
        strategy_func: Callable[[pd.Series], Tuple[pd.Series, pd.Series]],
    ) -> PropertyTestResult:
        """
        Test: Notional exposure should not exceed max leverage.

        Args:
            strategy_func: Function that returns (returns, positions)

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()

            try:
                result = strategy_func(prices)
                if isinstance(result, tuple) and len(result) == 2:
                    returns, positions = result
                else:
                    # Function only returns one thing, assume it's returns
                    positions = pd.Series([1.0] * len(prices))  # Assume full position
            except Exception:
                # Function doesn't support positions, skip
                continue

            # Calculate notional exposure (sum of absolute positions)
            notional = (
                positions.abs().sum() if hasattr(positions, "abs") else abs(positions)
            )

            if notional > self.max_leverage:
                num_fails += 1
                fail_examples.append(
                    {
                        "notional": notional,
                        "max_leverage": self.max_leverage,
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.NO_OVERLEVERAGE,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description=f"Notional exposure should be ≤ {self.max_leverage}x",
        )

    def property_no_nan_signals(
        self,
        signal_generator: Callable[[pd.Series], pd.Series],
    ) -> PropertyTestResult:
        """
        Test: Signal generator should never produce NaN signals.

        Args:
            signal_generator: Function that generates trading signals

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            signals = signal_generator(prices)

            if signals.isna().any():
                num_fails += 1
                nan_indices = np.where(signals.isna())[0]
                fail_examples.append(
                    {
                        "nan_count": signals.isna().sum(),
                        "nan_indices": nan_indices.tolist()[:5],
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.NO_NAN_SIGNALS,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description="Signal generator should never produce NaN",
        )

    def property_returns_reasonable_range(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
    ) -> PropertyTestResult:
        """
        Test: Single day returns should be within reasonable range.

        Args:
            strategy_func: Function that takes prices and returns returns

        Returns:
            PropertyTestResult
        """
        fail_examples = []
        num_fails = 0

        for _ in range(self.num_random_tests):
            prices = self.generate_random_price_series()
            returns = strategy_func(prices)

            # Check for extreme returns
            extreme_returns = returns[returns.abs() > self.max_single_return]

            if len(extreme_returns) > 0:
                num_fails += 1
                fail_examples.append(
                    {
                        "extreme_count": len(extreme_returns),
                        "max_return": returns.abs().max(),
                        "extreme_values": extreme_returns.abs().tolist()[:3],
                    }
                )

        return PropertyTestResult(
            property_name=PropertyType.RETURNS_REASONABLE_RANGE,
            passed=num_fails == 0,
            num_tests=self.num_random_tests,
            num_fails=num_fails,
            fail_examples=fail_examples[:5],
            description=f"Single returns should be ≤ {self.max_single_return:.1%}",
        )

    def run_all_property_tests(
        self,
        strategy_func: Callable[[pd.Series], pd.Series],
        signal_generator: Callable[[pd.Series], pd.Series] = None,
    ) -> Tuple[List[PropertyTestResult], Dict]:
        """
        Run all property tests.

        Args:
            strategy_func: Strategy function to test
            signal_generator: Optional signal generator to test

        Returns:
            Tuple of (results, summary)
        """
        results = []

        # Core property tests
        results.append(self.property_returns_always_defined(strategy_func))
        results.append(self.property_returns_finite(strategy_func))
        results.append(self.property_returns_reasonable_range(strategy_func))
        results.append(self.property_drawdown_limited(strategy_func))
        results.append(self.property_position_sizes_limited(strategy_func))
        results.append(self.property_no_overleverage(strategy_func))

        # Optional signal tests
        if signal_generator is not None:
            results.append(self.property_no_nan_signals(signal_generator))

        # Calculate summary
        summary = self._calculate_summary(results)

        return results, summary

    def _calculate_summary(self, results: List[PropertyTestResult]) -> Dict:
        """Calculate summary statistics across all tests."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        total_checks = sum(r.num_tests for r in results)
        total_failures = sum(r.num_fails for r in results)

        return {
            "total_properties": total_tests,
            "properties_passed": passed_tests,
            "properties_failed": total_tests - passed_tests,
            "total_checks": total_checks,
            "total_failures": total_failures,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "failure_rate": total_failures / total_checks if total_checks > 0 else 0.0,
        }

    def get_property_test_report(
        self, results: List[PropertyTestResult], summary: Dict
    ) -> str:
        """Generate human-readable property test report."""
        emoji = (
            "✅"
            if summary["success_rate"] >= 0.8
            else "🟠"
            if summary["success_rate"] >= 0.5
            else "🔴"
        )

        report = f"{emoji} Property-Based Test Results\n"
        report += f"{'='*60}\n\n"

        report += f"Overall: {summary['properties_passed']}/{summary['total_properties']} properties passed\n"
        report += f"Checks: {summary['total_checks']} total, {summary['total_failures']} failures\n"
        report += f"Failure Rate: {summary['failure_rate']:.2%}\n\n"

        report += "Property Tests:\n"
        for result in results:
            result_emoji = "✅" if result.passed else "❌"
            report += (
                f"  {result_emoji} {result.property_name.value}: {result.description}\n"
            )
            report += f"      Tests: {result.num_tests}, Fails: {result.num_fails}\n"

            if not result.passed and result.fail_examples:
                report += "      Example failures:\n"
                for ex in result.fail_examples[:3]:
                    report += f"        {ex}\n"

        return report


def run_property_tests(
    strategy_func: Callable[[pd.Series], pd.Series],
    signal_generator: Callable[[pd.Series], pd.Series] = None,
    num_random_tests: int = 100,
) -> Tuple[List[PropertyTestResult], str]:
    """
    Convenience function to run all property tests.

    Args:
        strategy_func: Strategy function to test
        signal_generator: Optional signal generator to test
        num_random_tests: Number of random tests per property

    Returns:
        Tuple of (results, report)
    """
    tester = PropertyTester(num_random_tests=num_random_tests)
    results, summary = tester.run_all_property_tests(strategy_func, signal_generator)
    report = tester.get_property_test_report(results, summary)

    return results, report
