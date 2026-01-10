"""
Stress Testing: Black swan and flash crash scenario testing.
Tests strategy resilience under extreme market conditions.
"""

from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import random

import pandas as pd
import numpy as np


class StressScenario(Enum):
    """Types of stress scenarios."""
    FLASH_CRASH = "flash_crash"
    BLACK_SWAN = "black_swan"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    GAP_DOWN = "gap_down"
    GAP_UP = "gap_up"
    EXTREME_TREND = "extreme_trend"
    WHIPSAW = "whipsaw"


@dataclass
class StressTestResult:
    """Result of a stress test."""
    scenario: StressScenario
    final_equity: float
    total_return: float
    max_drawdown: float
    final_equity_pct_of_initial: float
    survived: bool
    details: str


class StressTester:
    """
    Stress testing for trading strategies.

    Tests strategy performance under extreme market conditions:
    - Flash crashes (sudden large drops)
    - Black swan events (unexpected extreme moves)
    - Volatility spikes
    - Liquidity crises
    - Gap moves
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        survival_threshold: float = 0.5,  # 50% of initial capital
        max_drawdown_limit: float = 0.4,     # 40% max drawdown
    ):
        """
        Args:
            initial_capital: Starting capital
            survival_threshold: Min % of capital to survive (default 50%)
            max_drawdown_limit: Max acceptable drawdown (default 40%)
        """
        self.initial_capital = initial_capital
        self.survival_threshold = survival_threshold
        self.max_drawdown_limit = max_drawdown_limit

    def create_flash_crash_scenario(
        self,
        base_returns: pd.Series,
        crash_day: int = None,
        crash_magnitude: float = -0.30,  # 30% drop
    ) -> pd.Series:
        """
        Create a flash crash scenario.

        Args:
            base_returns: Base return series
            crash_day: Day to insert crash (default random)
            crash_magnitude: Magnitude of crash (default -30%)

        Returns:
            Modified return series with flash crash
        """
        returns = base_returns.copy()

        if crash_day is None:
            crash_day = random.randint(len(returns) // 2, len(returns) - 1)

        returns.iloc[crash_day] = crash_magnitude

        return returns

    def create_black_swan_scenario(
        self,
        base_returns: pd.Series,
        event_day: int = None,
        event_magnitude: float = -0.50,  # 50% drop
        volatility_multiplier: float = 3.0,
    ) -> pd.Series:
        """
        Create a black swan scenario.

        Args:
            base_returns: Base return series
            event_day: Day of black swan (default random)
            event_magnitude: Magnitude of event (default -50%)
            volatility_multiplier: Multiply volatility by this after event

        Returns:
            Modified return series with black swan
        """
        returns = base_returns.copy()

        if event_day is None:
            event_day = random.randint(len(returns) // 3, len(returns) - 1)

        # Insert black swan event
        returns.iloc[event_day] = event_magnitude

        # Increase volatility after event
        post_event_returns = returns.iloc[event_day + 1:]
        returns.iloc[event_day + 1:] = post_event_returns * volatility_multiplier

        return returns

    def create_volatility_spike_scenario(
        self,
        base_returns: pd.Series,
        spike_day: int = None,
        spike_duration: int = 10,
        volatility_multiplier: float = 5.0,
    ) -> pd.Series:
        """
        Create a volatility spike scenario.

        Args:
            base_returns: Base return series
            spike_day: Day to start spike (default random)
            spike_duration: Days of elevated volatility
            volatility_multiplier: Multiply volatility by this

        Returns:
            Modified return series with volatility spike
        """
        returns = base_returns.copy()

        if spike_day is None:
            spike_day = random.randint(len(returns) // 4, len(returns) - spike_duration - 1)

        end_day = min(spike_day + spike_duration, len(returns))
        returns.iloc[spike_day:end_day] = returns.iloc[spike_day:end_day] * volatility_multiplier

        return returns

    def create_liquidity_crisis_scenario(
        self,
        base_returns: pd.Series,
        crisis_day: int = None,
        spread_increase: float = 0.05,  # 5% spread
        duration: int = 20,
    ) -> pd.Series:
        """
        Create a liquidity crisis scenario (increased transaction costs).

        Args:
            base_returns: Base return series
            crisis_day: Day crisis starts (default random)
            spread_increase: Spread/transaction cost increase
            duration: Days of crisis

        Returns:
            Modified return series with liquidity crisis impact
        """
        returns = base_returns.copy()

        if crisis_day is None:
            crisis_day = random.randint(len(returns) // 4, len(returns) - duration - 1)

        end_day = min(crisis_day + duration, len(returns))

        # Apply spread/cost to returns during crisis
        for i in range(crisis_day, end_day):
            returns.iloc[i] -= spread_increase

        return returns

    def create_gap_scenario(
        self,
        base_returns: pd.Series,
        gap_day: int = None,
        gap_magnitude: float = -0.15,  # 15% gap
    ) -> pd.Series:
        """
        Create a gap scenario (overnight gap).

        Args:
            base_returns: Base return series
            gap_day: Day of gap (default random)
            gap_magnitude: Magnitude of gap

        Returns:
            Modified return series with gap
        """
        returns = base_returns.copy()

        if gap_day is None:
            gap_day = random.randint(len(returns) // 4, len(returns) - 1)

        returns.iloc[gap_day] = gap_magnitude

        return returns

    def create_extreme_trend_scenario(
        self,
        base_returns: pd.Series,
        trend_day: int = None,
        trend_duration: int = 30,
        daily_trend: float = -0.02,  # 2% daily downtrend
    ) -> pd.Series:
        """
        Create an extreme trend scenario.

        Args:
            base_returns: Base return series
            trend_day: Day trend starts (default random)
            trend_duration: Days of extreme trend
            daily_trend: Daily trend return

        Returns:
            Modified return series with extreme trend
        """
        returns = base_returns.copy()

        if trend_day is None:
            trend_day = random.randint(len(returns) // 4, len(returns) - trend_duration - 1)

        end_day = min(trend_day + trend_duration, len(returns))
        returns.iloc[trend_day:end_day] = daily_trend

        return returns

    def create_whipsaw_scenario(
        self,
        base_returns: pd.Series,
        start_day: int = None,
        duration: int = 20,
        volatility: float = 0.10,  # 10% daily swings
    ) -> pd.Series:
        """
        Create a whipsaw scenario (extreme choppiness).

        Args:
            base_returns: Base return series
            start_day: Day whipsaw starts (default random)
            duration: Days of whipsaw
            volatility: Daily volatility amplitude

        Returns:
            Modified return series with whipsaw
        """
        returns = base_returns.copy()

        if start_day is None:
            start_day = random.randint(len(returns) // 4, len(returns) - duration - 1)

        end_day = min(start_day + duration, len(returns))

        # Alternate between large gains and losses
        for i in range(start_day, end_day):
            returns.iloc[i] = volatility if i % 2 == 0 else -volatility

        return returns

    def run_stress_test(
        self,
        base_returns: pd.Series,
        scenario: StressScenario,
        signal_generator: Callable[[pd.Series], pd.Series] = None,
        **scenario_params
    ) -> StressTestResult:
        """
        Run a single stress test.

        Args:
            base_returns: Base return series
            scenario: Stress scenario to test
            signal_generator: Optional signal generator for strategy
            **scenario_params: Additional parameters for scenario

        Returns:
            StressTestResult
        """
        # Create scenario returns
        if scenario == StressScenario.FLASH_CRASH:
            test_returns = self.create_flash_crash_scenario(base_returns, **scenario_params)
            details = f"Flash crash: {scenario_params.get('crash_magnitude', -0.30):.1%} drop"
        elif scenario == StressScenario.BLACK_SWAN:
            test_returns = self.create_black_swan_scenario(base_returns, **scenario_params)
            details = f"Black swan: {scenario_params.get('event_magnitude', -0.50):.1%} drop"
        elif scenario == StressScenario.VOLATILITY_SPIKE:
            test_returns = self.create_volatility_spike_scenario(base_returns, **scenario_params)
            details = f"Volatility spike: {scenario_params.get('volatility_multiplier', 5.0)}x normal"
        elif scenario == StressScenario.LIQUIDITY_CRISIS:
            test_returns = self.create_liquidity_crisis_scenario(base_returns, **scenario_params)
            details = f"Liquidity crisis: {scenario_params.get('spread_increase', 0.05):.1%} spread increase"
        elif scenario == StressScenario.GAP_DOWN:
            test_returns = self.create_gap_scenario(base_returns, **scenario_params)
            details = f"Gap down: {scenario_params.get('gap_magnitude', -0.15):.1%}"
        elif scenario == StressScenario.GAP_UP:
            test_returns = self.create_gap_scenario(base_returns, gap_magnitude=0.15, **scenario_params)
            details = "Gap up: +15%"
        elif scenario == StressScenario.EXTREME_TREND:
            test_returns = self.create_extreme_trend_scenario(base_returns, **scenario_params)
            details = f"Extreme trend: {scenario_params.get('daily_trend', -0.02):.1%} daily"
        elif scenario == StressScenario.WHIPSAW:
            test_returns = self.create_whipsaw_scenario(base_returns, **scenario_params)
            details = f"Whipsaw: {scenario_params.get('volatility', 0.10):.1%} daily swings"
        else:
            test_returns = base_returns
            details = "Unknown scenario"

        # Apply strategy if signal generator provided
        if signal_generator is not None:
            # Note: This is simplified - in practice you'd need price data too
            test_returns = test_returns * signal_generator(test_returns).shift(1).fillna(0)

        # Calculate performance
        equity_curve = self.initial_capital * (1 + test_returns).cumprod()
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Check survival
        final_equity_pct = final_equity / self.initial_capital
        survived = (
            final_equity_pct >= self.survival_threshold and
            max_drawdown <= self.max_drawdown_limit
        )

        return StressTestResult(
            scenario=scenario,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_drawdown,
            final_equity_pct_of_initial=final_equity_pct,
            survived=survived,
            details=details,
        )

    def run_all_stress_tests(
        self,
        base_returns: pd.Series,
        signal_generator: Callable[[pd.Series], pd.Series] = None,
    ) -> Tuple[List[StressTestResult], Dict]:
        """
        Run all stress test scenarios.

        Args:
            base_returns: Base return series
            signal_generator: Optional signal generator

        Returns:
            Tuple of (results, summary)
        """
        results = []

        # Run all scenarios
        scenarios = [
            (StressScenario.FLASH_CRASH, {"crash_magnitude": -0.30}),
            (StressScenario.FLASH_CRASH, {"crash_magnitude": -0.50}),
            (StressScenario.BLACK_SWAN, {"event_magnitude": -0.50}),
            (StressScenario.BLACK_SWAN, {"event_magnitude": -0.70}),
            (StressScenario.VOLATILITY_SPIKE, {"volatility_multiplier": 5.0}),
            (StressScenario.VOLATILITY_SPIKE, {"volatility_multiplier": 10.0}),
            (StressScenario.LIQUIDITY_CRISIS, {"spread_increase": 0.05}),
            (StressScenario.LIQUIDITY_CRISIS, {"spread_increase": 0.10}),
            (StressScenario.GAP_DOWN, {"gap_magnitude": -0.15}),
            (StressScenario.GAP_DOWN, {"gap_magnitude": -0.25}),
            (StressScenario.GAP_UP, {"gap_magnitude": 0.15}),
            (StressScenario.EXTREME_TREND, {"daily_trend": -0.02}),
            (StressScenario.EXTREME_TREND, {"daily_trend": -0.05}),
            (StressScenario.WHIPSAW, {"volatility": 0.10}),
            (StressScenario.WHIPSAW, {"volatility": 0.20}),
        ]

        for scenario, params in scenarios:
            result = self.run_stress_test(base_returns, scenario, signal_generator, **params)
            results.append(result)

        # Calculate summary
        summary = self._calculate_summary(results)

        return results, summary

    def _calculate_summary(self, results: List[StressTestResult]) -> Dict:
        """Calculate summary statistics across all stress tests."""
        total_tests = len(results)
        survived_tests = sum(1 for r in results if r.survived)

        final_equities = [r.final_equity for r in results]
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]

        # Survival by scenario type
        scenario_survival = {}
        for scenario in StressScenario:
            scenario_results = [r for r in results if r.scenario == scenario]
            if scenario_results:
                survival_rate = sum(1 for r in scenario_results if r.survived) / len(scenario_results)
                scenario_survival[scenario.value] = survival_rate

        return {
            "total_tests": total_tests,
            "survived_tests": survived_tests,
            "survival_rate": survived_tests / total_tests if total_tests > 0 else 0.0,
            "avg_final_equity": np.mean(final_equities) if final_equities else 0.0,
            "min_final_equity": min(final_equities) if final_equities else 0.0,
            "avg_total_return": np.mean(total_returns) if total_returns else 0.0,
            "worst_return": min(total_returns) if total_returns else 0.0,
            "best_return": max(total_returns) if total_returns else 0.0,
            "avg_max_drawdown": np.mean(max_drawdowns) if max_drawdowns else 0.0,
            "worst_drawdown": max(max_drawdowns) if max_drawdowns else 0.0,
            "scenario_survival": scenario_survival,
        }

    def get_stress_test_report(
        self,
        results: List[StressTestResult],
        summary: Dict
    ) -> str:
        """Generate human-readable stress test report."""
        emoji = "✅" if summary["survival_rate"] >= 0.8 else "🟠" if summary["survival_rate"] >= 0.5 else "🔴"

        report = f"{emoji} Stress Test Results\n"
        report += f"{'='*60}\n\n"

        report += f"Overall Survival Rate: {summary['survival_rate']:.1%} "
        report += f"({summary['survived_tests']}/{summary['total_tests']} tests passed)\n\n"

        report += "Average Performance:\n"
        report += f"  Avg Final Equity: ${summary['avg_final_equity']:,.2f}\n"
        report += f"  Min Final Equity: ${summary['min_final_equity']:,.2f}\n"
        report += f"  Avg Return: {summary['avg_total_return']:.2%}\n"
        report += f"  Worst Return: {summary['worst_return']:.2%}\n"
        report += f"  Best Return: {summary['best_return']:.2%}\n"
        report += f"  Avg Max Drawdown: {summary['avg_max_drawdown']:.2%}\n"
        report += f"  Worst Drawdown: {summary['worst_drawdown']:.2%}\n\n"

        report += "Survival by Scenario:\n"
        for scenario, rate in summary["scenario_survival"].items():
            scenario_emoji = "✅" if rate >= 0.8 else "🟠" if rate >= 0.5 else "🔴"
            report += f"  {scenario_emoji} {scenario}: {rate:.1%}\n"

        report += "\nDetailed Results:\n"
        for i, result in enumerate(results, 1):
            result_emoji = "✅" if result.survived else "❌"
            report += (
                f"  {i}. {result_emoji} {result.scenario.value}: "
                f"${result.final_equity:,.2f} ({result.total_return:+.1%}), "
                f"DD: {result.max_drawdown:.1%} - {result.details}\n"
            )

        return report


def run_stress_tests(
    returns: pd.Series,
    initial_capital: float = 10000.0
) -> Tuple[List[StressTestResult], str]:
    """
    Convenience function to run all stress tests.

    Args:
        returns: Historical return series
        initial_capital: Starting capital

    Returns:
        Tuple of (results, report)
    """
    tester = StressTester(initial_capital=initial_capital)
    results, summary = tester.run_all_stress_tests(returns)
    report = tester.get_stress_test_report(results, summary)

    return results, report
