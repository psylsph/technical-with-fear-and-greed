"""
Parameter optimization module for trading strategy.

Provides grid search, random search, and walk-forward optimization
for finding optimal strategy parameters.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import BEST_PARAMS, PROJECT_ROOT
from .strategy import run_strategy

logger = logging.getLogger(__name__)

OPTIMIZATION_CACHE_DIR = Path(PROJECT_ROOT) / "cache" / "optimization"
OPTIMIZATION_RESULTS_FILE = Path(PROJECT_ROOT) / "optimization_results.json"


@dataclass
class OptimizationResult:
    """Result of a single parameter optimization run."""

    params: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        return cls(
            params=data["params"],
            metrics=data["metrics"],
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class GridSearchConfig:
    """Configuration for grid search optimization."""

    param_grid: Dict[str, List[Any]]
    objective: str = "total_return"
    maximize: bool = True
    parallel_jobs: int = 1
    verbose: bool = True


@dataclass
class RandomSearchConfig:
    """Configuration for random search optimization."""

    param_distributions: Dict[str, Union[List, Tuple, np.ndarray]]
    n_iterations: int = 100
    objective: str = "total_return"
    maximize: bool = True
    random_seed: int = 42
    verbose: bool = True


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    param_grid: Dict[str, List[Any]]
    train_window: int = 180
    test_window: int = 30
    step_size: int = 30
    objective: str = "total_return"
    maximize: bool = True
    verbose: bool = True


def _ensure_cache_dir() -> Path:
    """Ensure optimization cache directory exists."""
    OPTIMIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return OPTIMIZATION_CACHE_DIR


class ParameterOptimizer:
    """Parameter optimizer for trading strategy.

    Supports grid search, random search, and walk-forward optimization.
    """

    def __init__(
        self,
        close: pd.Series,
        fgi_df: pd.DataFrame,
        granularity: str = "ONE_DAY",
        pred_series: Optional[pd.Series] = None,
        higher_tf_data: Optional[Dict[str, pd.Series]] = None,
    ):
        """Initialize optimizer with data.

        Args:
            close: Price series
            fgi_df: Fear & Greed Index data
            granularity: Timeframe granularity name
            pred_series: ML prediction series (optional)
            higher_tf_data: Higher timeframe indicators (optional)
        """
        self.close = close
        self.fgi_df = fgi_df
        self.granularity = granularity
        self.pred_series = pred_series
        self.higher_tf_data = higher_tf_data
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    def _run_strategy_with_params(
        self,
        params: Dict[str, Any],
        freq: str,
    ) -> Dict[str, float]:
        """Run strategy with given parameters and return metrics."""
        try:
            result = run_strategy(
                close=self.close,
                freq=freq,
                fgi_df=self.fgi_df,
                granularity_name=self.granularity,
                rsi_window=params.get("rsi_window", 14),
                trail_pct=params.get("trail_pct", 0.10),
                buy_quantile=params.get("buy_quantile", 0.2),
                sell_quantile=params.get("sell_quantile", 0.8),
                ml_thresh=params.get("ml_thresh", 0.5),
                pred_series=self.pred_series,
                higher_tf_data=self.higher_tf_data,
                enable_multi_tf=params.get("enable_multi_tf", False),
                use_atr_trail=params.get("use_atr_trail", True),
                atr_multiplier=params.get("atr_multiplier", 2.0),
                max_drawdown_pct=params.get("max_drawdown_pct", 0.15),
            )
            return {
                "total_return": result["total_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "win_rate": result["win_rate"],
                "max_drawdown": result["max_drawdown"],
                "total_trades": result["total_trades"],
                "outperformance": result["outperformance"],
            }
        except Exception as e:
            logger.error(f"Strategy run failed with params {params}: {e}")
            return {
                "total_return": -float("inf"),
                "sharpe_ratio": -float("inf"),
                "win_rate": 0.0,
                "max_drawdown": 100.0,
                "total_trades": 0,
                "outperformance": -float("inf"),
            }

    def grid_search(
        self,
        config: GridSearchConfig,
        freq: str = "1d",
    ) -> OptimizationResult:
        """Perform grid search optimization.

        Args:
            config: Grid search configuration
            freq: Frequency string for vectorbt

        Returns:
            Best optimization result
        """
        from itertools import product

        param_names = list(config.param_grid.keys())
        param_values = list(config.param_grid.values())
        total_combinations = np.prod([len(v) for v in param_values])

        if config.verbose:
            print(f"\n{'=' * 60}")
            print("GRID SEARCH OPTIMIZATION")
            print(f"{'=' * 60}")
            print(f"Total combinations: {total_combinations}")
            print(f"Objective: {config.objective}")
            print(f"{'-' * 60}")

        self.results = []
        best_score = -float("inf") if config.maximize else float("inf")
        best_params = None

        for idx, combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, combination))
            metrics = self._run_strategy_with_params(params, freq)
            score = metrics.get(config.objective, 0.0)

            if config.maximize:
                is_better = score > best_score
            else:
                is_better = score < best_score

            if is_better:
                best_score = score
                best_params = params

            result = OptimizationResult(params=params, metrics=metrics)
            self.results.append(result)

            if config.verbose and idx % 10 == 0:
                print(
                    f"  Progress: {idx}/{total_combinations} ({100 * idx / total_combinations:.1f}%)"
                )

        self.best_result = OptimizationResult(
            params=best_params,
            metrics=self._run_strategy_with_params(best_params, freq),
        )

        if config.verbose:
            print(f"\n{'=' * 60}")
            print("GRID SEARCH RESULTS")
            print(f"{'=' * 60}")
            print(f"Best {config.objective}: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            print(f"{'-' * 60}")

            top_results = sorted(
                self.results, key=lambda x: x.metrics[config.objective], reverse=True
            )[:5]
            print("\nTop 5 parameter combinations:")
            for i, res in enumerate(top_results, 1):
                print(
                    f"  {i}. {config.objective}={res.metrics[config.objective]:.2f}, "
                    f"Sharpe={res.metrics['sharpe_ratio']:.2f}, "
                    f"WinRate={res.metrics['win_rate']:.1f}%, "
                    f"Params: {res.params}"
                )

        return self.best_result

    def random_search(
        self,
        config: RandomSearchConfig,
        freq: str = "1d",
    ) -> OptimizationResult:
        """Perform random search optimization.

        Args:
            config: Random search configuration
            freq: Frequency string for vectorbt

        Returns:
            Best optimization result
        """
        np.random.seed(config.random_seed)

        if config.verbose:
            print(f"\n{'=' * 60}")
            print("RANDOM SEARCH OPTIMIZATION")
            print(f"{'=' * 60}")
            print(f"Iterations: {config.n_iterations}")
            print(f"Objective: {config.objective}")
            print(f"{'-' * 60}")

        self.results = []
        best_score = -float("inf") if config.maximize else float("inf")
        best_params = None

        for i in range(config.n_iterations):
            params = {}
            for param_name, distribution in config.param_distributions.items():
                if isinstance(distribution, (list, tuple)):
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, np.ndarray):
                    if distribution.dtype.kind in "ifu":  # integer, float, unsigned
                        if np.issubdtype(distribution.dtype, np.integer):
                            params[param_name] = int(
                                np.random.uniform(distribution[0], distribution[-1])
                            )
                        else:
                            params[param_name] = np.random.uniform(
                                distribution[0], distribution[-1]
                            )
                    else:
                        params[param_name] = np.random.choice(distribution)

            metrics = self._run_strategy_with_params(params, freq)
            score = metrics.get(config.objective, 0.0)

            if config.maximize:
                is_better = score > best_score
            else:
                is_better = score < best_score

            if is_better:
                best_score = score
                best_params = params

            result = OptimizationResult(params=params, metrics=metrics)
            self.results.append(result)

            if config.verbose and (i + 1) % 20 == 0:
                print(
                    f"  Progress: {i + 1}/{config.n_iterations} ({(i + 1) / config.n_iterations * 100:.1f}%)"
                )

        self.best_result = OptimizationResult(
            params=best_params,
            metrics=self._run_strategy_with_params(best_params, freq),
        )

        if config.verbose:
            print(f"\n{'=' * 60}")
            print("RANDOM SEARCH RESULTS")
            print(f"{'=' * 60}")
            print(f"Best {config.objective}: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            print(f"{'-' * 60}")

            top_results = sorted(
                self.results, key=lambda x: x.metrics[config.objective], reverse=True
            )[:5]
            print("\nTop 5 parameter combinations:")
            for i, res in enumerate(top_results, 1):
                print(
                    f"  {i}. {config.objective}={res.metrics[config.objective]:.2f}, "
                    f"Sharpe={res.metrics['sharpe_ratio']:.2f}, "
                    f"WinRate={res.metrics['win_rate']:.1f}%, "
                    f"Params: {res.params}"
                )

        return self.best_result

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get optimization results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = []
        for res in self.results:
            row = res.params.copy()
            row.update(res.metrics)
            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """Save optimization results to JSON file."""
        if filepath is None:
            filepath = (
                _ensure_cache_dir()
                / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "granularity": self.granularity,
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "all_results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved optimization results to {filepath}")
        return filepath

    def load_results(self, filepath: Path) -> None:
        """Load optimization results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.results = [
            OptimizationResult.from_dict(r) for r in data.get("all_results", [])
        ]
        if data.get("best_result"):
            self.best_result = OptimizationResult.from_dict(data["best_result"])

        logger.info(f"Loaded {len(self.results)} results from {filepath}")


def run_walk_forward_optimization(
    close: pd.Series,
    fgi_df: pd.DataFrame,
    config: WalkForwardConfig,
    granularity: str = "ONE_DAY",
    freq: str = "1d",
) -> Dict[str, Any]:
    """Run walk-forward optimization with rolling train/test windows.

    Args:
        close: Price series
        fgi_df: Fear & Greed Index data
        config: Walk-forward configuration
        granularity: Granularity name for reporting
        freq: Frequency string for vectorbt

    Returns:
        Dictionary with optimization results and best parameters
    """
    if config.verbose:
        print(f"\n{'=' * 60}")
        print("WALK-FORWARD OPTIMIZATION")
        print(f"{'=' * 60}")
        print(f"Train window: {config.train_window} days")
        print(f"Test window: {config.test_window} days")
        print(f"Step size: {config.step_size} days")
        print(f"Objective: {config.objective}")
        print(f"{'-' * 60}")

    fgi_aligned = fgi_df.reindex(close.index, method="ffill")

    window_results = []
    best_overall_params = None
    best_overall_score = -float("inf") if config.maximize else float("inf")

    total_windows = (len(close) - config.train_window) // config.step_size

    for window_idx in range(total_windows):
        train_start = window_idx * config.step_size
        train_end = train_start + config.train_window
        test_start = train_end
        test_end = min(train_end + config.test_window, len(close))

        if test_end - test_start < 10:
            break

        if config.verbose:
            print(
                f"\nWindow {window_idx + 1}/{total_windows}: "
                f"{close.index[train_start].date()} to {close.index[test_end - 1].date()}"
            )

        train_close = close.iloc[train_start:train_end]
        train_fgi = fgi_aligned.iloc[train_start:train_end]
        test_close = close.iloc[test_start:test_end]
        test_fgi = fgi_aligned.iloc[test_start:test_end]

        optimizer = ParameterOptimizer(
            close=train_close,
            fgi_df=train_fgi,
            granularity=granularity,
        )

        grid_config = GridSearchConfig(
            param_grid=config.param_grid,
            objective=config.objective,
            maximize=config.maximize,
            verbose=False,
        )

        best_result = optimizer.grid_search(grid_config, freq)
        best_params = best_result.params

        if config.verbose:
            print(f"  Best train params: {best_params}")
            print(
                f"  Train {config.objective}: {best_result.metrics[config.objective]:.2f}"
            )

        test_optimizer = ParameterOptimizer(
            close=test_close,
            fgi_df=test_fgi,
            granularity=granularity,
        )
        test_metrics = test_optimizer._run_strategy_with_params(best_params, freq)

        if config.verbose:
            print(f"  Test {config.objective}: {test_metrics[config.objective]:.2f}")
            print(f"  Test Sharpe: {test_metrics['sharpe_ratio']:.2f}")
            print(f"  Test Win Rate: {test_metrics['win_rate']:.1f}%")

        window_results.append(
            {
                "window_idx": window_idx,
                "train_start": str(close.index[train_start].date()),
                "train_end": str(close.index[train_end - 1].date()),
                "test_start": str(close.index[test_start].date()),
                "test_end": str(close.index[test_end - 1].date()),
                "best_params": best_params,
                "train_metrics": best_result.metrics,
                "test_metrics": test_metrics,
            }
        )

        test_score = test_metrics.get(config.objective, 0.0)
        if config.maximize:
            if test_score > best_overall_score:
                best_overall_score = test_score
                best_overall_params = best_params
        else:
            if test_score < best_overall_score:
                best_overall_score = test_score
                best_overall_params = best_params

    aggregated_results = aggregate_walk_forward_results(window_results)

    if config.verbose:
        print(f"\n{'=' * 60}")
        print("WALK-FORWARD AGGREGATED RESULTS")
        print(f"{'=' * 60}")
        print(f"Total windows: {len(window_results)}")
        print(
            f"Average test {config.objective}: {aggregated_results['avg_test_objective']:.2f}"
        )
        print(
            f"Std test {config.objective}: {aggregated_results['std_test_objective']:.2f}"
        )
        print(f"Average test Sharpe: {aggregated_results['avg_test_sharpe']:.2f}")
        print(f"Average test win rate: {aggregated_results['avg_test_win_rate']:.1f}%")
        print("\nMost common best parameters:")
        for param_name, param_value in best_overall_params.items():
            print(f"  {param_name}: {param_value}")

    return {
        "window_results": window_results,
        "best_overall_params": best_overall_params,
        "aggregated_results": aggregated_results,
        "total_windows": len(window_results),
    }


def aggregate_walk_forward_results(
    window_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate walk-forward optimization results.

    Args:
        window_results: List of window optimization results

    Returns:
        Dictionary with aggregated metrics
    """
    if not window_results:
        return {
            "avg_test_objective": 0.0,
            "std_test_objective": 0.0,
            "avg_test_sharpe": 0.0,
            "avg_test_win_rate": 0.0,
            "avg_test_drawdown": 0.0,
        }

    test_objectives = [w["test_metrics"]["total_return"] for w in window_results]
    test_sharpes = [w["test_metrics"]["sharpe_ratio"] for w in window_results]
    test_win_rates = [w["test_metrics"]["win_rate"] for w in window_results]
    test_drawdowns = [w["test_metrics"]["max_drawdown"] for w in window_results]

    return {
        "avg_test_objective": np.mean(test_objectives),
        "std_test_objective": np.std(test_objectives),
        "avg_test_sharpe": np.mean(test_sharpes),
        "avg_test_win_rate": np.mean(test_win_rates),
        "avg_test_drawdown": np.mean(test_drawdowns),
        "min_test_objective": np.min(test_objectives),
        "max_test_objective": np.max(test_objectives),
    }


def optimize_strategy_parameters(
    close: pd.Series,
    fgi_df: pd.DataFrame,
    granularity: str = "ONE_DAY",
    pred_series: Optional[pd.Series] = None,
    optimization_type: str = "grid",
    freq: str = "1d",
    save_results: bool = True,
) -> Dict[str, Any]:
    """Run complete strategy parameter optimization.

    Args:
        close: Price series
        fgi_df: Fear & Greed Index data
        granularity: Granularity name for reporting
        pred_series: ML prediction series (optional)
        optimization_type: 'grid', 'random', or 'walk_forward'
        freq: Frequency string for vectorbt
        save_results: Whether to save results to file

    Returns:
        Dictionary with optimization results and best parameters
    """
    print(f"\n{'=' * 70}")
    print(f"STRATEGY PARAMETER OPTIMIZATION ({optimization_type.upper()})")
    print(f"{'=' * 70}")

    if optimization_type == "grid":
        param_grid = {
            "rsi_window": [7, 14, 21],
            "trail_pct": [0.05, 0.10, 0.15],
            "buy_quantile": [0.15, 0.20, 0.25],
            "sell_quantile": [0.75, 0.80, 0.85],
            "ml_thresh": [0.4, 0.5, 0.6],
            "atr_multiplier": [1.5, 2.0, 2.5],
            "max_drawdown_pct": [0.10, 0.15, 0.20],
        }

        config = GridSearchConfig(
            param_grid=param_grid,
            objective="total_return",
            maximize=True,
            verbose=True,
        )

        optimizer = ParameterOptimizer(
            close=close,
            fgi_df=fgi_df,
            granularity=granularity,
            pred_series=pred_series,
        )

        best_result = optimizer.grid_search(config, freq)

    elif optimization_type == "random":
        param_distributions = {
            "rsi_window": np.arange(5, 30),
            "trail_pct": np.arange(0.02, 0.25, 0.01),
            "buy_quantile": np.arange(0.10, 0.35, 0.05),
            "sell_quantile": np.arange(0.65, 0.90, 0.05),
            "ml_thresh": np.arange(0.3, 0.7, 0.05),
            "atr_multiplier": np.arange(1.0, 3.5, 0.25),
            "max_drawdown_pct": np.arange(0.05, 0.25, 0.02),
        }

        config = RandomSearchConfig(
            param_distributions=param_distributions,
            n_iterations=100,
            objective="total_return",
            maximize=True,
            random_seed=42,
            verbose=True,
        )

        optimizer = ParameterOptimizer(
            close=close,
            fgi_df=fgi_df,
            granularity=granularity,
            pred_series=pred_series,
        )

        best_result = optimizer.random_search(config, freq)

    elif optimization_type == "walk_forward":
        param_grid = {
            "rsi_window": [7, 14, 21],
            "trail_pct": [0.05, 0.10, 0.15],
            "buy_quantile": [0.15, 0.20, 0.25],
            "sell_quantile": [0.75, 0.80, 0.85],
            "ml_thresh": [0.4, 0.5, 0.6],
        }

        config = WalkForwardConfig(
            train_window=180,
            test_window=30,
            step_size=30,
            param_grid=param_grid,
            objective="total_return",
            maximize=True,
            verbose=True,
        )

        # Create optimizer for walk-forward
        optimizer = ParameterOptimizer(
            close=close,
            fgi_df=fgi_df,
            granularity=granularity,
            pred_series=pred_series,
        )

        results = run_walk_forward_optimization(
            close=close,
            fgi_df=fgi_df,
            config=config,
            granularity=granularity,
            freq=freq,
        )
        best_result = OptimizationResult(
            params=results["best_overall_params"],
            metrics=results["aggregated_results"],
        )

    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")

    if save_results:
        filepath = (
            _ensure_cache_dir()
            / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if hasattr(optimizer, "save_results"):
            optimizer.save_results(filepath)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print("Best parameters found:")
    for param_name, param_value in best_result.params.items():
        print(f"  {param_name}: {param_value}")
    print("\nBest metrics:")
    for metric_name, metric_value in best_result.metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"{'=' * 70}\n")

    return {
        "best_params": best_result.params,
        "best_metrics": best_result.metrics,
        "optimization_type": optimization_type,
    }


def update_best_params(best_params: Dict[str, Any]) -> None:
    """Update BEST_PARAMS in config with optimized values.

    Args:
        best_params: Dictionary of best parameters
    """
    param_mapping = {
        "rsi_window": "rsi_window",
        "trail_pct": "trail_pct",
        "buy_quantile": "buy_quantile",
        "sell_quantile": "sell_quantile",
        "ml_thresh": "ml_thresh",
    }

    for old_key, new_key in param_mapping.items():
        if old_key in best_params:
            BEST_PARAMS[new_key] = best_params[old_key]

    logger.info(f"Updated BEST_PARAMS: {BEST_PARAMS}")


def load_optimization_results(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Load previous optimization results.

    Args:
        filepath: Path to optimization results file (uses latest if None)

    Returns:
        Dictionary with optimization results
    """
    if filepath is None:
        cache_files = list(OPTIMIZATION_CACHE_DIR.glob("optimization_*.json"))
        if not cache_files:
            return {}
        filepath = max(cache_files, key=lambda x: x.stat().st_mtime)

    with open(filepath, "r") as f:
        return json.load(f)


def compare_parameter_sensitivity(
    close: pd.Series,
    fgi_df: pd.DataFrame,
    param_name: str,
    param_values: List[Any],
    other_params: Optional[Dict[str, Any]] = None,
    granularity: str = "ONE_DAY",
    freq: str = "1d",
) -> pd.DataFrame:
    """Analyze sensitivity of strategy to a single parameter.

    Args:
        close: Price series
        fgi_df: Fear & Greed Index data
        param_name: Name of parameter to analyze
        param_values: List of values to test
        other_params: Fixed parameters (uses defaults if None)
        granularity: Granularity name for reporting
        freq: Frequency string for vectorbt

    Returns:
        DataFrame with results for each parameter value
    """
    default_params = {
        "rsi_window": 14,
        "trail_pct": 0.10,
        "buy_quantile": 0.2,
        "sell_quantile": 0.8,
        "ml_thresh": 0.5,
        "enable_multi_tf": False,
        "use_atr_trail": True,
        "atr_multiplier": 2.0,
        "max_drawdown_pct": 0.15,
    }

    if other_params:
        default_params.update(other_params)

    results = []
    for value in param_values:
        params = default_params.copy()
        params[param_name] = value

        optimizer = ParameterOptimizer(
            close=close,
            fgi_df=fgi_df,
            granularity=granularity,
        )

        metrics = optimizer._run_strategy_with_params(params, freq)
        row = {param_name: value}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    return df.sort_values(by="total_return", ascending=False)
