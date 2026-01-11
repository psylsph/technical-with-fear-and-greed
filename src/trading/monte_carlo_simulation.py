"""
Monte Carlo Simulation: Test strategy with 1000+ random permutations.
Simulates thousands of possible scenarios to estimate probability distributions of outcomes.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import random

import pandas as pd
import numpy as np


class ConfidenceLevel(Enum):
    """Confidence levels for risk analysis."""

    P90 = 0.90
    P95 = 0.95
    P99 = 0.99


@dataclass
class SimulationResult:
    """Result of a single Monte Carlo simulation run."""

    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy validation.

    Simulates thousands of random scenarios to:
    - Estimate probability distributions of returns
    - Calculate confidence intervals for performance
    - Identify worst-case scenarios
    - Test strategy robustness under different conditions
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        initial_capital: float = 10000.0,
        random_seed: int = None,
        confidence_levels: List[float] = None,
    ):
        """
        Args:
            num_simulations: Number of simulations to run (default 1000)
            initial_capital: Starting capital for each simulation
            random_seed: Optional random seed for reproducibility
            confidence_levels: Confidence levels to calculate (default [0.90, 0.95, 0.99])
        """
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital
        self.random_seed = random_seed
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def bootstrap_returns(
        self, returns: pd.Series, sample_size: int = None
    ) -> pd.Series:
        """
        Bootstrap a returns series by sampling with replacement.

        Args:
            returns: Historical return series
            sample_size: Size of bootstrap sample (default: same as input)

        Returns:
            Bootstrapped return series
        """
        if sample_size is None:
            sample_size = len(returns)

        return returns.sample(n=sample_size, replace=True).reset_index(drop=True)

    def permute_returns(self, returns: pd.Series) -> pd.Series:
        """
        Randomly permute the order of returns.

        Args:
            returns: Historical return series

        Returns:
            Permuted return series
        """
        return returns.sample(frac=1).reset_index(drop=True)

    def simulate_single_path(
        self, returns: pd.Series, method: str = "bootstrap"
    ) -> SimulationResult:
        """
        Simulate a single path using specified method.

        Args:
            returns: Historical return series
            method: "bootstrap" or "permute"

        Returns:
            SimulationResult with performance metrics
        """
        # Generate simulated returns
        if method == "bootstrap":
            sim_returns = self.bootstrap_returns(returns)
        elif method == "permute":
            sim_returns = self.permute_returns(returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate cumulative returns
        equity_curve = self.initial_capital * (1 + sim_returns).cumprod()

        # Calculate metrics
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (annualized)
        if sim_returns.std() > 0:
            sharpe_ratio = (sim_returns.mean() / sim_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Win rate
        win_rate = (
            (sim_returns > 0).sum() / len(sim_returns) if len(sim_returns) > 0 else 0.0
        )

        # Number of trades (roughly number of non-zero returns)
        total_trades = (sim_returns != 0).sum()

        return SimulationResult(
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
        )

    def run_simulation(
        self, returns: pd.Series, method: str = "bootstrap"
    ) -> Tuple[List[SimulationResult], Dict]:
        """
        Run full Monte Carlo simulation.

        Args:
            returns: Historical return series
            method: "bootstrap" or "permute"

        Returns:
            Tuple of (results: list, statistics: dict)
        """
        results = []

        for _ in range(self.num_simulations):
            result = self.simulate_single_path(returns, method)
            results.append(result)

        # Calculate statistics across all simulations
        statistics = self._calculate_statistics(results)

        return results, statistics

    def _calculate_statistics(self, results: List[SimulationResult]) -> Dict:
        """Calculate statistics across all simulation results."""
        final_equities = [r.final_equity for r in results]
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        win_rates = [r.win_rate for r in results]

        # Calculate percentiles for each metric
        def calc_percentiles(values: List[float]) -> Dict:
            percentiles = {}
            for cl in self.confidence_levels:
                pct_value = np.percentile(values, cl * 100)
                percentiles[f"p{int(cl * 100)}"] = pct_value
            return percentiles

        return {
            "final_equity": {
                "mean": np.mean(final_equities),
                "std": np.std(final_equities),
                "median": np.median(final_equities),
                "min": min(final_equities),
                "max": max(final_equities),
                "percentiles": calc_percentiles(final_equities),
            },
            "total_return": {
                "mean": np.mean(total_returns),
                "std": np.std(total_returns),
                "median": np.median(total_returns),
                "min": min(total_returns),
                "max": max(total_returns),
                "percentiles": calc_percentiles(total_returns),
            },
            "max_drawdown": {
                "mean": np.mean(max_drawdowns),
                "std": np.std(max_drawdowns),
                "median": np.median(max_drawdowns),
                "min": min(max_drawdowns),
                "max": max(max_drawdowns),
                "percentiles": calc_percentiles(max_drawdowns),
            },
            "sharpe_ratio": {
                "mean": np.mean(sharpe_ratios),
                "std": np.std(sharpe_ratios),
                "median": np.median(sharpe_ratios),
                "min": min(sharpe_ratios),
                "max": max(sharpe_ratios),
                "percentiles": calc_percentiles(sharpe_ratios),
            },
            "win_rate": {
                "mean": np.mean(win_rates),
                "std": np.std(win_rates),
                "median": np.median(win_rates),
                "min": min(win_rates),
                "max": max(win_rates),
                "percentiles": calc_percentiles(win_rates),
            },
            "num_simulations": len(results),
        }

    def calculate_probability_of_profit(self, results: List[SimulationResult]) -> Dict:
        """
        Calculate probability of achieving different profit levels.

        Args:
            results: List of simulation results

        Returns:
            Dict with probabilities at different thresholds
        """
        returns = [r.total_return for r in results]

        return {
            "prob_profit": sum(1 for r in returns if r > 0) / len(returns)
            if returns
            else 0.0,
            "prob_5pct_gain": sum(1 for r in returns if r > 0.05) / len(returns)
            if returns
            else 0.0,
            "prob_10pct_gain": sum(1 for r in returns if r > 0.10) / len(returns)
            if returns
            else 0.0,
            "prob_20pct_gain": sum(1 for r in returns if r > 0.20) / len(returns)
            if returns
            else 0.0,
            "prob_5pct_loss": sum(1 for r in returns if r < -0.05) / len(returns)
            if returns
            else 0.0,
            "prob_10pct_loss": sum(1 for r in returns if r < -0.10) / len(returns)
            if returns
            else 0.0,
            "prob_20pct_loss": sum(1 for r in returns if r < -0.20) / len(returns)
            if returns
            else 0.0,
        }

    def get_worst_case_scenarios(
        self, results: List[SimulationResult], n: int = 10
    ) -> List[SimulationResult]:
        """
        Get the n worst performing scenarios.

        Args:
            results: List of simulation results
            n: Number of worst scenarios to return

        Returns:
            List of worst simulation results
        """
        return sorted(results, key=lambda r: r.total_return)[:n]

    def get_best_case_scenarios(
        self, results: List[SimulationResult], n: int = 10
    ) -> List[SimulationResult]:
        """
        Get the n best performing scenarios.

        Args:
            results: List of simulation results
            n: Number of best scenarios to return

        Returns:
            List of best simulation results
        """
        return sorted(results, key=lambda r: r.total_return, reverse=True)[:n]

    def get_simulation_report(
        self,
        results: List[SimulationResult],
        statistics: Dict,
        method: str = "bootstrap",
    ) -> str:
        """
        Generate human-readable Monte Carlo simulation report.

        Args:
            results: List of simulation results
            statistics: Statistics dictionary
            method: Method used ("bootstrap" or "permute")

        Returns:
            Formatted report string
        """
        prob_profit = self.calculate_probability_of_profit(results)
        worst_cases = self.get_worst_case_scenarios(results, 5)
        best_cases = self.get_best_case_scenarios(results, 5)

        report = f"Monte Carlo Simulation Results ({method.capitalize()})\n"
        report += f"{'='*60}\n\n"

        report += f"Number of Simulations: {statistics['num_simulations']}\n"
        report += f"Initial Capital: ${self.initial_capital:,.2f}\n\n"

        # Total Return Statistics
        tr = statistics["total_return"]
        report += "Total Return Statistics:\n"
        report += f"  Mean: {tr['mean']:.2%}\n"
        report += f"  Median: {tr['median']:.2%}\n"
        report += f"  Std Dev: {tr['std']:.2%}\n"
        report += f"  Range: [{tr['min']:.2%}, {tr['max']:.2%}]\n"
        report += "  Confidence Intervals:\n"
        for pct_name, pct_value in tr["percentiles"].items():
            report += f"    {pct_name}: {pct_value:.2%}\n"
        report += "\n"

        # Final Equity Statistics
        fe = statistics["final_equity"]
        report += "Final Equity Statistics:\n"
        report += f"  Mean: ${fe['mean']:,.2f}\n"
        report += f"  Median: ${fe['median']:,.2f}\n"
        report += f"  Range: [${fe['min']:,.2f}, ${fe['max']:,.2f}]\n"
        report += "  Confidence Intervals:\n"
        for pct_name, pct_value in fe["percentiles"].items():
            report += f"    {pct_name}: ${pct_value:,.2f}\n"
        report += "\n"

        # Max Drawdown Statistics
        dd = statistics["max_drawdown"]
        report += "Max Drawdown Statistics:\n"
        report += f"  Mean: {dd['mean']:.2%}\n"
        report += f"  Median: {dd['median']:.2%}\n"
        report += f"  Range: [{dd['min']:.2%}, {dd['max']:.2%}]\n"
        report += "  Confidence Intervals:\n"
        for pct_name, pct_value in dd["percentiles"].items():
            report += f"    {pct_name}: {pct_value:.2%}\n"
        report += "\n"

        # Sharpe Ratio Statistics
        sr = statistics["sharpe_ratio"]
        report += "Sharpe Ratio Statistics:\n"
        report += f"  Mean: {sr['mean']:.2f}\n"
        report += f"  Median: {sr['median']:.2f}\n"
        report += "  Confidence Intervals:\n"
        for pct_name, pct_value in sr["percentiles"].items():
            report += f"    {pct_name}: {pct_value:.2f}\n"
        report += "\n"

        # Probability of Profit
        report += "Probability Analysis:\n"
        report += f"  Prob(Profit): {prob_profit['prob_profit']:.1%}\n"
        report += f"  Prob(5%+ Gain): {prob_profit['prob_5pct_gain']:.1%}\n"
        report += f"  Prob(10%+ Gain): {prob_profit['prob_10pct_gain']:.1%}\n"
        report += f"  Prob(20%+ Gain): {prob_profit['prob_20pct_gain']:.1%}\n"
        report += f"  Prob(5%+ Loss): {prob_profit['prob_5pct_loss']:.1%}\n"
        report += f"  Prob(10%+ Loss): {prob_profit['prob_10pct_loss']:.1%}\n"
        report += f"  Prob(20%+ Loss): {prob_profit['prob_20pct_loss']:.1%}\n"
        report += "\n"

        # Best and Worst Cases
        report += "Best 5 Scenarios:\n"
        for i, result in enumerate(best_cases, 1):
            report += (
                f"  {i}. Return: {result.total_return:.2%}, "
                f"Final Equity: ${result.final_equity:,.2f}, "
                f"Max DD: {result.max_drawdown:.2%}\n"
            )

        report += "\nWorst 5 Scenarios:\n"
        for i, result in enumerate(worst_cases, 1):
            report += (
                f"  {i}. Return: {result.total_return:.2%}, "
                f"Final Equity: ${result.final_equity:,.2f}, "
                f"Max DD: {result.max_drawdown:.2%}\n"
            )

        return report


def run_monte_carlo(
    returns: pd.Series,
    num_simulations: int = 1000,
    initial_capital: float = 10000.0,
    method: str = "bootstrap",
    random_seed: int = 42,
) -> Tuple[List[SimulationResult], str]:
    """
    Convenience function to run Monte Carlo simulation.

    Args:
        returns: Historical return series
        num_simulations: Number of simulations
        initial_capital: Starting capital
        method: "bootstrap" or "permute"
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (results: list, report: str)
    """
    simulator = MonteCarloSimulator(
        num_simulations=num_simulations,
        initial_capital=initial_capital,
        random_seed=random_seed,
    )

    results, statistics = simulator.run_simulation(returns, method)
    report = simulator.get_simulation_report(results, statistics, method)

    return results, report
