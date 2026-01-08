#!/usr/bin/env python3
"""
Trading Strategy Backtest Suite

Comprehensive backtesting and analysis tool for the Fear & Greed trading strategy
with short selling support across multiple assets.

Usage:
    python backtest_suite.py --help
    python backtest_suite.py --asset ETH-USD          # Single asset backtest
    python backtest_suite.py --compare                 # Multi-asset comparison
    python backtest_suite.py --walk-forward            # Walk-forward analysis
    python backtest_suite.py --validate                 # Train/validation split
"""

import argparse
import sys
from typing import Dict

import pandas as pd

from src.data.data_fetchers import (
    fetch_eth_price_data,
    fetch_fear_greed_index,
    fetch_unified_price_data,
    fetch_xrp_price_data,
)
from src.sentiment import calculate_rsi_sentiment
from src.strategy import run_strategy


ASSETS = {
    "BTC-USD": {
        "fetcher": lambda s, e, f: fetch_unified_price_data("BTC-USD", s, e, f),
        "name": "Bitcoin",
    },
    "ETH-USD": {
        "fetcher": lambda s, e, f: fetch_eth_price_data(s, e, f),
        "name": "Ethereum",
    },
    "XRP-USD": {
        "fetcher": lambda s, e, f: fetch_xrp_price_data(s, e, f),
        "name": "Ripple",
    },
}

DEFAULT_PARAMS = {
    "rsi_window": 14,
    "trail_pct": 0.08,
    "buy_quantile": 0.20,
    "sell_quantile": 0.80,
    "ml_thresh": 0.50,
    "enable_multi_tf": False,
    "use_atr_trail": True,
    "atr_multiplier": 2.5,
    "max_drawdown_pct": 0.15,
    "enable_regime_filter": True,
}


def create_sentiment_proxy(close: pd.Series) -> pd.DataFrame:
    """Create FGI-equivalent DataFrame using RSI sentiment."""
    sentiment = calculate_rsi_sentiment(close, window=14)

    def classify(val):
        if val <= 25:
            return "Extreme Fear"
        elif val <= 35:
            return "Fear"
        elif val <= 45:
            return "Neutral Fear"
        elif val <= 55:
            return "Neutral"
        elif val <= 65:
            return "Neutral Greed"
        elif val <= 75:
            return "Greed"
        else:
            return "Extreme Greed"

    return pd.DataFrame(
        {
            "fgi_value": sentiment.values,
            "fgi_classification": sentiment.apply(classify),
        },
        index=sentiment.index,
    )


def run_single_asset(
    asset_symbol: str, start_date: str, end_date: str, params: Dict
) -> Dict:
    """Run backtest on a single asset."""
    asset_info = ASSETS[asset_symbol]
    print(f"\n{'=' * 60}")
    print(f"Testing {asset_info['name']} ({asset_symbol})")
    print(f"{'=' * 60}")

    ohlcv = ASSETS[asset_symbol]["fetcher"](start_date, end_date, "1d")
    if ohlcv is None or len(ohlcv) < 50:
        return {"error": "Insufficient data", "asset": asset_info["name"]}

    close = ohlcv["close"]
    print(
        f"Data: {len(close)} bars | {close.index[0].date()} to {close.index[-1].date()}"
    )

    if asset_symbol == "BTC-USD":
        fgi_df = fetch_fear_greed_index()
    else:
        fgi_df = create_sentiment_proxy(close)

    fgi_aligned = fgi_df.reindex(close.index, method="ffill").ffill().bfill().fillna(50)

    try:
        result = run_strategy(
            close=close,
            freq="1d",
            fgi_df=fgi_aligned,
            granularity_name=asset_symbol,
            **params,
            pred_series=None,
            higher_tf_data=None,
        )
        return {
            "asset": asset_info["name"],
            "symbol": asset_symbol,
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "win_rate": result["win_rate"],
            "max_drawdown": result["max_drawdown"],
            "total_trades": result["total_trades"],
            "benchmark_return": result["benchmark_return"],
            "outperformance": result["outperformance"],
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "asset": asset_info["name"], "symbol": asset_symbol}


def compare_assets(start_date: str, end_date: str, params: Dict):
    """Compare all assets."""
    print("\n" + "=" * 70)
    print("MULTI-ASSET STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\nPeriod: {start_date} to {end_date}")

    results = {}
    for asset_symbol in ASSETS.keys():
        result = run_single_asset(asset_symbol, start_date, end_date, params)
        results[asset_symbol] = result

    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print("\nAsset       Return %   Sharpe    Win Rate   Drawdown   Trades   Benchmark")
    print("-" * 90)

    valid = {k: v for k, v in results.items() if not v.get("error")}
    for asset_symbol, result in results.items():
        if result.get("error"):
            print(f"{result['asset']:<12} ERROR")
        else:
            print(
                "{:12} {:>10.2f}% {:>9.2f} {:>10.1f}% {:>10.2f}% {:>7} {:>10.2f}%".format(
                    result["asset"],
                    result["total_return"],
                    result["sharpe_ratio"],
                    result["win_rate"],
                    result["max_drawdown"],
                    int(result["total_trades"]),
                    result["benchmark_return"],
                )
            )

    if valid:
        ranked = sorted(valid.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True)
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        winner = ranked[0][1]
        print(
            f"\nBest risk-adjusted: {winner['asset']} (Sharpe: {winner['sharpe_ratio']:.2f})"
        )
        print(
            f"  Return: {winner['total_return']:.2f}% | Drawdown: {winner['max_drawdown']:.2f}%"
        )

    return results


def walk_forward_analysis(start_date: str, end_date: str, params: Dict):
    """Run walk-forward validation."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 70)

    train_end = "2024-07-01"
    val_start = "2024-07-01"
    val_end = end_date

    print(f"\nTraining: {start_date} to {train_end}")
    print(f"Validation: {val_start} to {val_end}")

    print("\n--- Training Period ---")
    train_result = run_single_asset("ETH-USD", start_date, train_end, params)
    if not train_result.get("error"):
        print(
            f"Return: {train_result['total_return']:.2f}% | Sharpe: {train_result['sharpe_ratio']:.2f}"
        )

    print("\n--- Validation Period ---")
    val_result = run_single_asset("ETH-USD", val_start, val_end, params)
    if not val_result.get("error"):
        print(
            f"Return: {val_result['total_return']:.2f}% | Sharpe: {val_result['sharpe_ratio']:.2f}"
        )
        print(
            f"Drawdown: {val_result['max_drawdown']:.2f}% | Trades: {int(val_result['total_trades'])}"
        )

    return {"train": train_result, "validation": val_result}


def validate_params(start_date: str, end_date: str, params: Dict):
    """Train/validation split for parameter validation."""
    print("\n" + "=" * 70)
    print("PARAMETER VALIDATION (Train/Validation Split)")
    print("=" * 70)

    split_date = "2024-10-01"

    print(f"\nTraining: {start_date} to {split_date}")
    print(f"Validation: {split_date} to {end_date}")

    train_result = run_single_asset("ETH-USD", start_date, split_date, params)
    val_result = run_single_asset("ETH-USD", split_date, end_date, params)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not train_result.get("error"):
        print(
            f"\nTraining: {train_result['total_return']:.2f}% return, Sharpe {train_result['sharpe_ratio']:.2f}"
        )
    if not val_result.get("error"):
        print(
            f"Validation: {val_result['total_return']:.2f}% return, Sharpe {val_result['sharpe_ratio']:.2f}"
        )

    return {"train": train_result, "validation": val_result}


def main():
    parser = argparse.ArgumentParser(description="Trading Strategy Backtest Suite")
    parser.add_argument(
        "--asset",
        default="ETH-USD",
        choices=list(ASSETS.keys()),
        help="Asset to trade (default: ETH-USD)",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--compare", action="store_true", help="Compare all assets")
    parser.add_argument(
        "--walk-forward", action="store_true", help="Walk-forward analysis"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Train/validation split"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Override params as JSON (e.g., '{\"trail_pct\":0.10}')",
    )

    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.params:
        import json

        override = json.loads(args.params)
        params.update(override)
        print(f"Using custom params: {override}")

    if args.compare:
        compare_assets(args.start, args.end, params)
    elif args.walk_forward:
        walk_forward_analysis(args.start, args.end, params)
    elif args.validate:
        validate_params(args.start, args.end, params)
    else:
        result = run_single_asset(args.asset, args.start, args.end, params)

        if result.get("error"):
            print(f"Error: {result['error']}")
            return 1

        print("\n" + "=" * 70)
        print(f"BACKTEST RESULTS: {result['asset']}")
        print("=" * 70)
        print(f"\n  Total Return:    {result['total_return']:.2f}%")
        print(f"  Benchmark:       {result['benchmark_return']:.2f}%")
        print(f"  Outperformance:  {result['outperformance']:.2f}%")
        print(f"  Sharpe Ratio:    {result['sharpe_ratio']:.2f}")
        print(f"  Win Rate:        {result['win_rate']:.1f}%")
        print(f"  Max Drawdown:    {result['max_drawdown']:.2f}%")
        print(f"  Total Trades:    {int(result['total_trades'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
