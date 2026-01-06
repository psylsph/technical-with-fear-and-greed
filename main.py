#!/usr/bin/env python3
"""
Main entry point for the Fear & Greed Index Trading Strategy.
"""

import argparse
import json
import os

import pandas as pd

from src.config import (
    BEST_PARAMS,
    END_DATE,
    GRANULARITY_TO_FREQ,
    START_DATE,
)
from src.data.data_fetchers import (
    calculate_higher_tf_indicators,
    fetch_coinbase_historical,
    fetch_fear_greed_index,
    fetch_yahoo_data,
)
from src.ml.ml_model import train_ml_model
from src.strategy import run_strategy


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Fear & Greed Trading Strategy")
    parser.add_argument(
        "--live", action="store_true", help="Run live trading with real money"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run simulated live trading (paper mode with local state persistence)",
    )
    parser.add_argument(
        "--multi-tf",
        action="store_true",
        help="Run multi-timeframe backtesting with filtering comparisons",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BTC Fear & Greed Index Strategy - Multiple Timeframes")
    print("=" * 60)

    # Load FGI data
    fgi_df = fetch_fear_greed_index()
    fgi_values_full = fgi_df["fgi_value"]
    print(f"\nFGI Range: {fgi_values_full.min()} - {fgi_values_full.max()}")
    print(f"Average FGI: {fgi_values_full.mean():.1f}")
    print(f"Fear days (FGI<=35): {(fgi_values_full <= 35).sum()}")
    print(f"Extreme Greed days (FGI>80): {(fgi_values_full > 80).sum()}")

    # Run backtesting
    if args.multi_tf:
        run_multi_tf_backtesting(fgi_df)
    elif not args.live and not args.test:
        run_backtesting(fgi_df)

    elif args.test:
        from src.trading.trading_engine import run_test_trading

        run_test_trading(fgi_df)

    elif args.live:
        from src.trading.trading_engine import run_live_trading

        run_live_trading(fgi_df)


def run_backtesting(fgi_df: pd.DataFrame):
    """Run backtesting analysis."""
    print("\nRunning initial backtests (no ML)...")

    GRANULARITIES_TO_TEST = ["ONE_DAY", "ONE_HOUR"]
    initial_results = []

    for granularity in GRANULARITIES_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"Testing {granularity} ({GRANULARITY_TO_FREQ[granularity]})")
        print(f"{'=' * 60}")

        close = None
        freq = GRANULARITY_TO_FREQ[granularity]
        data_source = "Unknown"

        try:
            # Try Coinbase first
            try:
                from src.config import CDP_KEY_FILE

                if os.path.exists(CDP_KEY_FILE):
                    with open(CDP_KEY_FILE, "r") as f:
                        cdp_keys = json.load(f)
                    # Set environment variables for Coinbase
                    os.environ["COINBASE_API_KEY"] = cdp_keys.get("name", "")
                    os.environ["COINBASE_SECRET_KEY"] = cdp_keys.get("privateKey", "")
                    print("Using Coinbase...")
                    close = fetch_coinbase_historical(
                        "BTC-USD",
                        START_DATE + "T00:00:00Z",
                        END_DATE + "T00:00:00Z",
                        granularity.upper(),
                    )
                    if isinstance(close, pd.DataFrame):
                        close = close["close"]
                    data_source = "Coinbase"
                else:
                    raise FileNotFoundError("CDP key file not found")
            except Exception as e:
                print(f"Coinbase setup error: {e}, using Yahoo Finance...")
                close = fetch_yahoo_data(
                    "BTC-USD", START_DATE, END_DATE, GRANULARITY_TO_FREQ[granularity]
                )
                data_source = "Yahoo Finance"
        except Exception as e:
            print(f"Data fetch error for {granularity}: {e}")
            continue

        if close is None or len(close) < 10:
            print(f"Insufficient data for {granularity}")
            continue

        print(f"Data source: {data_source}")
        print(f"Total bars: {len(close)}")

        # Initial backtest without ML (pred_series is None, so ML is disabled)
        result = run_strategy(close, freq, fgi_df, granularity)
        initial_results.append(result)

    print(f"\n{'=' * 80}")
    print("INITIAL BACKTEST RESULTS (No ML)")
    print(f"{'=' * 80}")
    print(
        f"{'Granularity':<15} {'Return %':<12} {'Benchmark %':<14} {'Outper %':<12} {'Win Rate %':<12} {'Trades':<8}"
    )
    print(f"{'-' * 80}")

    for result in initial_results:
        print(
            f"{result['granularity']:<15} "
            f"{result['total_return']:>10.2f}% "
            f"{result['benchmark_return']:>12.2f}% "
            f"{result['outperformance']:>10.2f}% "
            f"{result['win_rate']:>10.1f}% "
            f"{result['total_trades']:>6}"
        )

    # Train ML model with automated lookback optimization
    print("\nTraining ML Model (testing 90, 180, 365 day lookback periods)...")
    daily_close = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, "1d")

    lookback_periods = [90, 180, 365]
    best_lookback = None
    best_return = -float("inf")
    best_pred_series = None

    for lookback in lookback_periods:
        print(f"\n  Testing {lookback}-day lookback...")
        model, pred_series, metrics = train_ml_model(
            daily_close, fgi_df, lookback_days=lookback
        )

        # Quick backtest with default params to evaluate this lookback period
        result = run_strategy(
            daily_close,
            "1d",
            fgi_df,
            "ONE_DAY",
            rsi_window=14,
            trail_pct=0.10,
            buy_quantile=0.2,
            sell_quantile=0.8,
            ml_thresh=0.5,
            pred_series=pred_series,
            higher_tf_data=None,
            enable_multi_tf=False,
        )

        total_return = result["total_return"]
        print(f"    Return: {total_return:.2f}%, Trades: {result['total_trades']}")

        if total_return > best_return:
            best_return = total_return
            best_lookback = lookback
            best_pred_series = pred_series

    print(
        f"\n  Best lookback period: {best_lookback} days (return: {best_return:.2f}%)"
    )
    pred_series = best_pred_series
    print("ML Model trained with optimal lookback period.")

    # Parameter optimization
    print("\nOptimizing parameters for ONE_DAY...")
    close_oned = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, "1d")
    if close_oned is not None and len(close_oned) > 10:
        combos = [
            (14, 0.05, 0.2, 0.8, 0.4),  # tighter trail, lower ML
            (14, 0.05, 0.2, 0.8, 0.6),  # tighter trail, higher ML
            (14, 0.15, 0.2, 0.8, 0.4),  # looser trail, lower ML
            (14, 0.15, 0.2, 0.8, 0.6),  # looser trail, higher ML
        ]
        best_ret = -float("inf")
        best_combo = None
        for rsi, trail, buy_q, sell_q, ml_t in combos:
            result = run_strategy(
                close_oned,
                "1d",
                fgi_df,
                "ONE_DAY",
                rsi,
                trail,
                buy_q,
                sell_q,
                ml_t,
                pred_series,
            )
            ret = result["total_return"]
            print(
                f"Combo RSI{rsi} Trail{trail} BuyQ{buy_q} SellQ{sell_q} ML{ml_t}: Return {ret:.2f}%, Win {result['win_rate']:.1f}%, Trades {result['total_trades']}"
            )
            if ret > best_ret:
                best_ret = ret
                best_combo = (rsi, trail, buy_q, sell_q, ml_t)

        if best_combo:
            BEST_PARAMS["rsi_window"] = best_combo[0]
            BEST_PARAMS["trail_pct"] = best_combo[1]
            BEST_PARAMS["buy_quantile"] = best_combo[2]
            BEST_PARAMS["sell_quantile"] = best_combo[3]
            BEST_PARAMS["ml_thresh"] = best_combo[4]
            print(
                f"\nBest params: RSI={best_combo[0]}, Trail={best_combo[1]}, BuyQ={best_combo[2]}, SellQ={best_combo[3]}, ML={best_combo[4]}"
            )
            print(f"Best return: {best_ret:.2f}%")

            # Final backtest with optimized parameters
            print("\nFinal backtest with optimized parameters...")
            final_results = []
            for granularity in GRANULARITIES_TO_TEST:
                close_final = None
                freq = GRANULARITY_TO_FREQ[granularity]

                try:
                    # Try Coinbase first
                    try:
                        from src.config import CDP_KEY_FILE

                        if os.path.exists(CDP_KEY_FILE):
                            with open(CDP_KEY_FILE, "r") as f:
                                cdp_keys = json.load(f)
                            os.environ["COINBASE_API_KEY"] = cdp_keys.get("name", "")
                            os.environ["COINBASE_SECRET_KEY"] = cdp_keys.get(
                                "privateKey", ""
                            )
                            close_final = fetch_coinbase_historical(
                                "BTC-USD",
                                START_DATE + "T00:00:00Z",
                                END_DATE + "T00:00:00Z",
                                granularity.upper(),
                            )
                            if isinstance(close_final, pd.DataFrame):
                                close_final = close_final["close"]
                        else:
                            raise FileNotFoundError("CDP key file not found")
                    except Exception as e:
                        print(
                            f"Coinbase setup error for final backtest: {e}, using Yahoo Finance..."
                        )
                        close_final = fetch_yahoo_data(
                            "BTC-USD",
                            START_DATE,
                            END_DATE,
                            GRANULARITY_TO_FREQ[granularity],
                        )
                except Exception as e:
                    print(f"Final backtest data fetch error for {granularity}: {e}")
                    continue

                if close_final is not None and len(close_final) >= 10:
                    result = run_strategy(
                        close_final,
                        freq,
                        fgi_df,
                        granularity,
                        rsi_window=BEST_PARAMS["rsi_window"],
                        trail_pct=BEST_PARAMS["trail_pct"],
                        buy_quantile=BEST_PARAMS["buy_quantile"],
                        sell_quantile=BEST_PARAMS["sell_quantile"],
                        ml_thresh=BEST_PARAMS["ml_thresh"],
                        pred_series=pred_series,
                    )
                    final_results.append(result)

            if final_results:
                print(f"\n{'=' * 80}")
                print("FINAL BACKTEST RESULTS (Optimized Parameters + ML)")
                print(f"{'=' * 80}")
                print(
                    f"{'Granularity':<15} {'Return %':<12} {'Benchmark %':<14} {'Outper %':<12} {'Win Rate %':<12} {'Trades':<8}"
                )
                print(f"{'-' * 80}")

                for result in final_results:
                    print(
                        f"{result['granularity']:<15} "
                        f"{result['total_return']:>10.2f}% "
                        f"{result['benchmark_return']:>12.2f}% "
                        f"{result['outperformance']:>10.2f}% "
                        f"{result['win_rate']:>10.1f}% "
                        f"{result['total_trades']:>6}"
                    )

                best_return = max(final_results, key=lambda x: x["total_return"])
                best_sharpe = max(final_results, key=lambda x: x["sharpe_ratio"])

                print(f"{'-' * 80}")
                print(
                    f"Best Return: {best_return['granularity']} with {best_return['total_return']:.2f}%"
                )
                print(
                    f"Best Sharpe: {best_sharpe['granularity']} with {best_sharpe['sharpe_ratio']:.2f}"
                )
                print(f"{'=' * 80}")
    else:
        print("ONE_DAY data insufficient for optimization")

    print(
        "\nBacktest complete. Run with --live for live trading or --test for simulated live trading."
    )


def run_multi_tf_backtesting(fgi_df: pd.DataFrame):
    """Run multi-timeframe backtesting with filtering comparisons."""
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME BACKTESTING")
    print("=" * 80)

    # Define timeframes: higher TF (daily) and lower TFs (4H, 1H)
    higher_tf_granularity = "ONE_DAY"
    lower_tf_granularities = ["FOUR_HOUR", "ONE_HOUR"]

    # Fetch higher timeframe data (daily) for trend filtering
    print(f"\nFetching higher timeframe data: {higher_tf_granularity}...")
    higher_freq = GRANULARITY_TO_FREQ[higher_tf_granularity]
    higher_tf_close = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, higher_freq)

    if higher_tf_close is None or len(higher_tf_close) < 50:
        print("Insufficient higher timeframe data")
        return

    # Calculate higher timeframe indicators
    print("Calculating higher timeframe indicators...")
    higher_tf_indicators = calculate_higher_tf_indicators(
        higher_tf_close, higher_tf_granularity
    )

    print(f"  Higher TF Trend: bullish={higher_tf_indicators['trend'].iloc[-1]}")
    print(f"  Higher TF RSI: {higher_tf_indicators['rsi'].iloc[-1]:.1f}")

    # Run comparison tests for each lower timeframe
    comparison_results = []

    for lower_tf in lower_tf_granularities:
        print(f"\n{'=' * 80}")
        print(f"Testing {lower_tf} vs Higher TF ({higher_tf_granularity}) Filter")
        print(f"{'=' * 80}")

        lower_freq = GRANULARITY_TO_FREQ[lower_tf]
        lower_tf_close = None

        try:
            lower_tf_close = fetch_yahoo_data(
                "BTC-USD", START_DATE, END_DATE, lower_freq
            )
            data_source = "Yahoo Finance"
        except Exception as e:
            print(f"Error fetching {lower_tf}: {e}")
            continue

        if lower_tf_close is None or len(lower_tf_close) < 10:
            print(f"Insufficient data for {lower_tf}")
            continue

        print(f"Data source: {data_source}")
        print(f"Total bars: {len(lower_tf_close)}")

        # Align higher TF data with lower TF
        print("Aligning higher timeframe indicators...")
        aligned_data = {}
        for key, value in higher_tf_indicators.items():
            if isinstance(value, pd.Series):
                # Resample to lower timeframe and forward fill
                resampled = value.resample(lower_freq).ffill()
                # Reindex to match lower TF exactly
                aligned = resampled.reindex(lower_tf_close.index, method="ffill")
                aligned_data[f"higher_{key}"] = aligned
            else:
                aligned_data[f"higher_{key}"] = value

        # Test 1: Unfiltered strategy (baseline)
        print("\n--- Test 1: Unfiltered Strategy (Baseline) ---")
        result_unfiltered = run_strategy(
            lower_tf_close,
            lower_freq,
            fgi_df,
            lower_tf,
            rsi_window=BEST_PARAMS["rsi_window"],
            trail_pct=BEST_PARAMS["trail_pct"],
            buy_quantile=BEST_PARAMS["buy_quantile"],
            sell_quantile=BEST_PARAMS["sell_quantile"],
            ml_thresh=BEST_PARAMS["ml_thresh"],
            pred_series=None,
            higher_tf_data=None,
            enable_multi_tf=False,
        )

        print(
            f"  Return: {result_unfiltered['total_return']:.2f}%, "
            f"Win Rate: {result_unfiltered['win_rate']:.1f}%, "
            f"Trades: {result_unfiltered['total_trades']}"
        )

        # Test 2: Multi-TF filtered strategy
        print("\n--- Test 2: Multi-TF Filtered Strategy ---")
        result_filtered = run_strategy(
            lower_tf_close,
            lower_freq,
            fgi_df,
            lower_tf,
            rsi_window=BEST_PARAMS["rsi_window"],
            trail_pct=BEST_PARAMS["trail_pct"],
            buy_quantile=BEST_PARAMS["buy_quantile"],
            sell_quantile=BEST_PARAMS["sell_quantile"],
            ml_thresh=BEST_PARAMS["ml_thresh"],
            pred_series=None,
            higher_tf_data=aligned_data,
            enable_multi_tf=True,
        )

        print(
            f"  Return: {result_filtered['total_return']:.2f}%, "
            f"Win Rate: {result_filtered['win_rate']:.1f}%, "
            f"Trades: {result_filtered['total_trades']}"
        )

        # Calculate improvements
        return_improvement = (
            result_filtered["total_return"] - result_unfiltered["total_return"]
        )
        win_rate_improvement = (
            result_filtered["win_rate"] - result_unfiltered["win_rate"]
        )
        drawdown_improvement = (
            result_unfiltered["max_drawdown"] - result_filtered["max_drawdown"]
        )
        trade_count_change = (
            result_filtered["total_trades"] - result_unfiltered["total_trades"]
        )

        print("\n--- Comparison Summary ---")
        print(f"  Return Improvement: {return_improvement:+.2f}%")
        print(f"  Win Rate Improvement: {win_rate_improvement:+.2f}%")
        print(f"  Drawdown Reduction: {drawdown_improvement:+.2f}%")
        print(f"  Trade Count Change: {trade_count_change:+}")

        comparison_results.append(
            {
                "granularity": lower_tf,
                "unfiltered": result_unfiltered,
                "filtered": result_filtered,
                "return_improvement": return_improvement,
                "win_rate_improvement": win_rate_improvement,
                "drawdown_improvement": drawdown_improvement,
                "trade_count_change": trade_count_change,
            }
        )

    # Print overall summary
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME FILTERING SUMMARY")
    print("=" * 80)
    print(
        f"{'Granularity':<15} {'Return Imp %':<15} {'Win Rate Imp %':<15} {'Drawdown Red %':<15} {'Trades Δ':<10}"
    )
    print("-" * 80)

    for result in comparison_results:
        print(
            f"{result['granularity']:<15} "
            f"{result['return_improvement']:>13.2f}% "
            f"{result['win_rate_improvement']:>13.2f}% "
            f"{result['drawdown_improvement']:>13.2f}% "
            f"{result['trade_count_change']:>10}"
        )

    # Calculate average improvements across all timeframes
    avg_return_imp = sum(r["return_improvement"] for r in comparison_results) / len(
        comparison_results
    )
    avg_win_rate_imp = sum(r["win_rate_improvement"] for r in comparison_results) / len(
        comparison_results
    )
    avg_drawdown_imp = sum(r["drawdown_improvement"] for r in comparison_results) / len(
        comparison_results
    )

    print("-" * 80)
    print("AVERAGE IMPROVEMENTS:")
    print(f"  Return: {avg_return_imp:+.2f}%")
    print(f"  Win Rate: {avg_win_rate_imp:+.2f}%")
    print(f"  Drawdown Reduction: {avg_drawdown_imp:+.2f}%")

    # Recommendation based on results
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    if avg_return_imp > 0 and avg_win_rate_imp > 0:
        print("✓ Multi-TF filtering IMPROVES performance")
        print("  → Enable multi-TF filtering for production trading")
    elif avg_return_imp < 0:
        print("✗ Multi-TF filtering REDUCES performance")
        print("  → Consider disabling or adjusting filter thresholds")
    else:
        print("◦ Multi-TF filtering has mixed results")
        print("  → Analyze individual timeframe performance")

    if avg_drawdown_imp > 0:
        print(f"✓ Drawdown reduced by {avg_drawdown_imp:.2f}% on average")
        print("  → Better risk management with multi-TF")


if __name__ == "__main__":
    main()
