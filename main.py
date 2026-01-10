#!/usr/bin/env python3
"""
Main entry point for Fear & Greed Index Trading Strategy.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from src.config import (
    BEST_PARAMS,
    END_DATE,
    GRANULARITY_TO_FREQ,
    INITIAL_CAPITAL,
    MAKER_FEE,
    START_DATE,
    TAKER_FEE,
    TOP_CRYPTOCURRENCIES,
)
from src.data.data_fetchers import (
    calculate_higher_tf_indicators,
    fetch_fear_greed_index,
    fetch_unified_price_data,
    get_current_price,
)
from src.indicators import calculate_adx
from src.ml.ml_model import train_ml_model
from src.strategy import run_strategy, generate_signal


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
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward analysis to validate strategy",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run real-time trading system with WebSocket feeds and event-driven architecture",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run parameter optimization (grid, random, or walk_forward)",
    )
    parser.add_argument(
        "--optimization-type",
        type=str,
        default="grid",
        choices=["grid", "random", "walk_forward"],
        help="Type of optimization to run (default: grid)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode - minimal output, status every minute or on trades",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ETH Fear & Greed Index Strategy - Multiple Timeframes")
    print("=" * 60)

    # Load FGI data
    print("Loading FGI data...")
    fgi_df = fetch_fear_greed_index()
    print(f"FGI data loaded: {len(fgi_df)} rows")
    fgi_values_full = fgi_df["fgi_value"]
    print(f"\nFGI Range: {fgi_values_full.min()} - {fgi_values_full.max()}")
    print(f"Average FGI: {fgi_values_full.mean():.1f}")
    print(f"Fear days (FGI<=35): {(fgi_values_full <= 35).sum()}")
    print(f"Extreme Greed days (FGI>80): {(fgi_values_full > 80).sum()}")
    print("FGI data processing complete")

    # Run backtesting
    if args.realtime:
        import asyncio

        from src.real_time_system import run_real_time_system

        print("\n" + "=" * 70)
        print("REAL-TIME TRADING MODE")
        print("=" * 70)
        print(f"\nTracking {len(TOP_CRYPTOCURRENCIES)} top cryptocurrencies by volume:")
        for i, symbol in enumerate(TOP_CRYPTOCURRENCIES, 1):
            print(f"  {i}. {symbol}")
        print("\nConnecting to exchanges: Coinbase, Binance, Kraken, Bybit")
        print("\nPress Ctrl+C to stop\n")

        asyncio.run(
            run_real_time_system(
                symbols=TOP_CRYPTOCURRENCIES,
                fgi_data=fgi_df,
                config={
                    "strategy": {
                        "fear_threshold": 30,
                        "greed_threshold": 70,
                    },
                    "enable_trading": False,
                },
                duration=3600,
            )
        )
    elif args.walk_forward:
        run_walk_forward_analysis(fgi_df)
    elif args.optimize:
        run_parameter_optimization(fgi_df, args.optimization_type)
    elif args.multi_tf:
        run_multi_tf_backtesting(fgi_df)
    elif not args.live and not args.test:
        run_backtesting(fgi_df)

    elif args.test:
        # Run a single test analysis instead of continuous trading
        print("\n" + "=" * 60)
        print("TEST MODE (Single Signal Analysis)")
        print("=" * 60)

        from src.trading.trading_engine import analyze_test_signal
        from src.config import DEFAULT_ASSET

        signal_info = analyze_test_signal(fgi_df)
        if signal_info:
            ind = signal_info.get("indicators", {})
            print(f"Current {DEFAULT_ASSET} Price: ${ind.get('price', 0):,.2f}")
            print(
                f"FGI: {ind.get('fgi', 0)} (Effective: Fear≤{ind.get('effective_fear_threshold', 30)}, Greed≥{ind.get('effective_greed_threshold', 70)})"
            )
            print(f"Market Regime: {ind.get('fgi_trend', 'unknown').upper()}")
            print(f"Signal: {signal_info['signal'].upper()}")
            print(f"Position Size: {ind.get('position_size_pct', 0):.1f}% of portfolio")
            print(f"Extreme Fear: {ind.get('is_extreme_fear', False)}")
            print(f"Extreme Greed: {ind.get('is_extreme_greed', False)}")
            print(f"Volatility Stop: ${ind.get('volatility_stop', 0):,.2f}")
            print("\nEnhanced Risk-Focused Strategy for 2026:")
            print("- Market regime detection (Bull/Bear/Sideways)")
            print("- Dynamic position sizing based on volatility")
            print("- Adaptive thresholds per market regime")
            print("- Enter on fear, exit on greed or drawdown")
            print("- Capital preservation in adverse conditions")

            # Add performance insights
            fgi_val = ind.get("fgi", 50)
            if fgi_val <= 25:
                print("\n🚨 EXTREME FEAR DETECTED - STRONG BUY SIGNAL")
                print("   Market sentiment indicates panic selling")
            elif fgi_val >= 75:
                print("\n⚠️  EXTREME GREED DETECTED - TAKE PROFITS")
                print("   Market sentiment indicates euphoria")
            elif fgi_val <= 40:
                print("\n📈 FEAR PRESENT - WATCH FOR ENTRIES")
                print("   Market showing signs of capitulation")
            elif fgi_val >= 60:
                print("\n📉 GREED BUILDING - CONSIDER REDUCING RISK")
                print("   Market showing signs of optimism")

            # Analyze multi-asset signals for diversification
            analyze_multi_asset_signals(fgi_df)

        else:
            print("Could not analyze current market signal")

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

        freq = GRANULARITY_TO_FREQ[granularity]

        # Use unified fetch (Coinbase primary, Yahoo fallback)
        ohlcv_data = fetch_unified_price_data("ETH-USD", START_DATE, END_DATE, freq)

        if ohlcv_data is None or len(ohlcv_data) < 10:
            print(f"Insufficient data for {granularity}")
            continue

        close = ohlcv_data["close"]
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
    daily_ohlcv = fetch_unified_price_data("ETH-USD", START_DATE, END_DATE, "1d")

    if daily_ohlcv is None or len(daily_ohlcv) < 30:
        print("Failed to fetch data for ML training")
        return

    daily_close = daily_ohlcv["close"]

    lookback_periods = [90, 180, 365]
    best_lookback = None
    best_return = -float("inf")
    best_pred_series = None

    for lookback in lookback_periods:
        print(f"\n  Testing {lookback}-day lookback...")
        model, pred_series, metrics = train_ml_model(
            daily_ohlcv, fgi_df, lookback_days=lookback, use_ensemble=True
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
    if daily_close is not None and len(daily_close) > 10:
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
                daily_close,
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
                freq = GRANULARITY_TO_FREQ[granularity]

                # Use unified fetch (Coinbase primary, Yahoo fallback)
                ohlcv_final = fetch_unified_price_data(
                    "ETH-USD", START_DATE, END_DATE, freq
                )

                if ohlcv_final is not None and len(ohlcv_final) >= 10:
                    # Test without MTA first (technical strategy only)
                    result_no_mta = run_strategy(
                        ohlcv_final["close"],
                        freq,
                        fgi_df,
                        granularity,
                        rsi_window=BEST_PARAMS["rsi_window"],
                        trail_pct=BEST_PARAMS["trail_pct"],
                        buy_quantile=BEST_PARAMS["buy_quantile"],
                        sell_quantile=BEST_PARAMS["sell_quantile"],
                        ml_thresh=BEST_PARAMS["ml_thresh"],
                        pred_series=pred_series,
                        higher_tf_data=None,
                        enable_multi_tf=False,
                    )

                    # Test with MTA (multi-timeframe analysis)
                    higher_freq = GRANULARITY_TO_FREQ["ONE_DAY"]
                    higher_ohlcv = fetch_unified_price_data(
                        "ETH-USD", START_DATE, END_DATE, higher_freq
                    )

                    higher_tf_indicators = calculate_higher_tf_indicators(
                        higher_ohlcv["close"], "ONE_DAY"
                    )

                    # Align higher TF indicators with current timeframe
                    aligned_data = {}
                    for key, value in higher_tf_indicators.items():
                        if isinstance(value, pd.Series):
                            resampled = value.resample(freq).ffill()
                            aligned = resampled.reindex(
                                ohlcv_final["close"].index, method="ffill"
                            )
                            aligned_data[f"higher_{key}"] = aligned
                        else:
                            aligned_data[f"higher_{key}"] = value

                    result_with_mta = run_strategy(
                        ohlcv_final["close"],
                        freq,
                        fgi_df,
                        granularity,
                        rsi_window=BEST_PARAMS["rsi_window"],
                        trail_pct=BEST_PARAMS["trail_pct"],
                        buy_quantile=BEST_PARAMS["buy_quantile"],
                        sell_quantile=BEST_PARAMS["sell_quantile"],
                        ml_thresh=BEST_PARAMS["ml_thresh"],
                        pred_series=pred_series,
                        higher_tf_data=aligned_data,
                        enable_multi_tf=True,
                    )

                    print(f"\n  MTA Comparison for {granularity}:")
                    print(
                        f"    Without MTA: Return {result_no_mta['total_return']:.2f}%, Trades {result_no_mta['total_trades']}"
                    )
                    print(
                        f"    With MTA:    Return {result_with_mta['total_return']:.2f}%, Trades {result_with_mta['total_trades']}"
                    )

                    # Use better performing one
                    if result_no_mta["total_return"] > result_with_mta["total_return"]:
                        print("    → Using WITHOUT MTA (better performance)")
                        final_results.append(result_no_mta)
                    else:
                        print("    → Using WITH MTA (better performance)")
                        final_results.append(result_with_mta)

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
    higher_tf_ohlcv = fetch_unified_price_data(
        "ETH-USD", START_DATE, END_DATE, higher_freq
    )
    if higher_tf_ohlcv is None or len(higher_tf_ohlcv) < 50:
        print("Insufficient higher timeframe data")
        return

    higher_tf_close = higher_tf_ohlcv["close"]

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

        # Use unified fetch (Coinbase primary, Yahoo fallback)
        lower_tf_ohlcv = fetch_unified_price_data(
            "ETH-USD", START_DATE, END_DATE, lower_freq
        )

        if lower_tf_ohlcv is None or len(lower_tf_ohlcv) < 10:
            print(f"Insufficient data for {lower_tf}")
            continue

        lower_tf_close = lower_tf_ohlcv["close"]
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


def analyze_multi_asset_signals(fgi_df: pd.DataFrame):
    """Analyze current ETH-USD signal for trading."""
    assets = ["ETH-USD"]

    for asset in assets:
        print(f"\n{'=' * 60}")
        print(f"{asset.replace('-USD', '')}/USD SIGNAL ANALYSIS")
        print("=" * 60)

        try:
            # Get current asset price
            asset_price = get_current_price(asset)
            if asset_price is None:
                print(f"Could not fetch {asset} price")
                continue

            # Create price series
            price_series = pd.Series([asset_price], index=[pd.Timestamp.now(tz="UTC")])

            # Generate signal
            signal = generate_signal(
                price_series,
                fgi_df,
                fear_entry_threshold=30,
                greed_exit_threshold=70,
                max_drawdown_exit=0.08,
                _volatility_stop_multiplier=1.5,
                pred_series=None,
                enable_multi_tf=False,
                enable_short_selling=True,  # Enable short selling
                enable_news_sentiment=False,  # Disable for now (needs API key)
            )

            indicators = signal.get("indicators", {})
            print(f"Current {asset.replace('-USD', '')} Price: ${asset_price:,.2f}")
            print(
                f"FGI: {indicators.get('fgi', 0)} (Effective: Fear≤{indicators.get('effective_fear_threshold', 30)}, Greed≥{indicators.get('effective_greed_threshold', 70)})"
            )
            print(f"Market Regime: {indicators.get('fgi_trend', 'unknown').upper()}")
            print(f"Signal: {signal['signal'].upper()}")
            print(
                f"Position Size: {indicators.get('position_size_pct', 0):.1f}% of portfolio"
            )
            print(f"Extreme Fear: {indicators.get('is_extreme_fear', False)}")
            print(f"Extreme Greed: {indicators.get('is_extreme_greed', False)}")

            # Add performance insights
            fgi_val = indicators.get("fgi", 50)
            if fgi_val <= 25:
                print("\n🚨 EXTREME FEAR DETECTED - STRONG BUY SIGNAL")
            elif fgi_val >= 75:
                print("\n⚠️  EXTREME GREED DETECTED - TAKE PROFITS")
            elif fgi_val <= 40:
                print("\n📈 FEAR PRESENT - WATCH FOR ENTRIES")
            elif fgi_val >= 60:
                print("\n📉 GREED BUILDING - CONSIDER REDUCING RISK")

        except Exception as e:
            print(f"Error analyzing {asset} signals: {e}")


def run_walk_forward_analysis(fgi_df: pd.DataFrame):
    """Run walk-forward analysis with rolling train/test windows."""
    print("\nRunning walk-forward analysis...")
    print("=" * 80)

    # Walk-forward parameters
    train_days = 180
    test_days = 30
    step_days = 30

    # Fetch daily data
    daily_ohlcv = fetch_unified_price_data("ETH-USD", START_DATE, END_DATE, "1d")

    if daily_ohlcv is None or len(daily_ohlcv) < train_days + test_days:
        print("Insufficient data for walk-forward analysis")
        return

    close = daily_ohlcv["close"]
    high = daily_ohlcv["high"]
    low = daily_ohlcv["low"]

    # Align FGI data with price data
    fgi_aligned = fgi_df.reindex(close.index, method="ffill")

    results = []
    window_num = 0

    # Walk-forward windows
    for i in range(train_days, len(close) - test_days, step_days):
        train_start = i - train_days
        train_end = i
        test_start = i
        test_end = min(i + test_days, len(close))

        window_num += 1
        print(
            f"\n--- Window {window_num}: {close.index[train_start].date()} to {close.index[test_end - 1].date()} ---"
        )

        # Train data
        train_close = close.iloc[train_start:train_end]
        train_fgi = fgi_aligned.iloc[train_start:train_end]

        # Test data
        test_close = close.iloc[test_start:test_end]
        test_fgi = fgi_aligned.iloc[test_start:test_end]
        test_high = high.iloc[test_start:test_end]
        test_low = low.iloc[test_start:test_end]

        # Train ML model on training data
        try:
            train_ml_model(
                pd.DataFrame({"close": train_close}),
                train_fgi,
                lookback_days=90,
                use_ensemble=True,
            )
        except Exception as e:
            print(f"  ML training failed: {e}, using no-ML approach")

        # Run strategy on test data with various parameter combinations
        combos = [
            (30, 70, 20, 1.5),
            (25, 75, 20, 1.5),
            (35, 65, 20, 1.5),
        ]

        window_results = []
        for fear_thresh, greed_thresh, adx_thresh, vol_mult in combos:
            try:
                result = run_simple_strategy(
                    pd.DataFrame(
                        {"close": test_close, "high": test_high, "low": test_low}
                    ),
                    test_fgi,
                    "Walk-Forward",
                    volatility_lookback=20,
                    max_drawdown_exit=0.08,
                    trend_filter_days=50,
                    fear_entry_threshold=fear_thresh,
                    greed_exit_threshold=greed_thresh,
                    _volatility_stop_multiplier=vol_mult,
                    adx_threshold=adx_thresh,
                )
                window_results.append(
                    (result, (fear_thresh, greed_thresh, adx_thresh, vol_mult))
                )
            except Exception as e:
                print(f"  Strategy failed for params {fear_thresh}/{greed_thresh}: {e}")
                continue

        # Best result for this window
        if window_results:
            best_result, best_params = max(
                window_results, key=lambda x: x[0]["total_return"]
            )
            results.append(best_result)
            print(
                f"  Best params: Fear≤{best_params[0]}, Greed≥{best_params[1]}, ADX>{best_params[2]}"
            )
            print(
                f"  Return: {best_result['total_return']:.2f}%, Trades: {best_result['total_trades']}, Win Rate: {best_result['win_rate']:.1f}%"
            )

    if not results:
        print("\nNo valid walk-forward windows")
        return

    # Aggregate results
    avg_return = sum(r["total_return"] for r in results) / len(results)
    avg_trades = sum(r["total_trades"] for r in results) / len(results)
    avg_win_rate = sum(r["win_rate"] for r in results) / len(results)
    avg_drawdown = sum(r["max_drawdown"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total windows tested: {len(results)}")
    print(f"Average return per window: {avg_return:.2f}%")
    print(f"Average trades per window: {avg_trades:.1f}")
    print(f"Average win rate: {avg_win_rate:.1f}%")
    print(f"Average max drawdown: {avg_drawdown:.2f}%")


def run_simple_strategy(
    ohlcv_data: pd.DataFrame,
    fgi_df: pd.DataFrame,
    granularity_name: str,
    volatility_lookback: int = 20,
    max_drawdown_exit: float = 0.08,
    trend_filter_days: int = 50,
    fear_entry_threshold: int = 30,
    greed_exit_threshold: int = 70,
    _volatility_stop_multiplier: float = 1.5,
    adx_threshold: float = 20.0,
) -> dict:
    """Run enhanced risk-focused strategy for 2026 market conditions with trend-following."""
    import vectorbt as vbt

    # Extract OHLC data
    close = ohlcv_data["close"]
    high = ohlcv_data["high"]
    low = ohlcv_data["low"]

    # Ensure close is a Series with a name
    if isinstance(close, pd.Series):
        if not close.name:
            close = close.copy()
            close.name = "close"

    # Create entries and exits (separate for long and short)
    entries = pd.DataFrame(False, index=close.index, columns=[close.name])
    exits = pd.DataFrame(False, index=close.index, columns=[close.name])
    short_entries = pd.DataFrame(False, index=close.index, columns=[close.name])
    short_exits = pd.DataFrame(False, index=close.index, columns=[close.name])

    # Strategy logic: Risk management focused (supports short selling)
    position_type = None  # 'long', 'short', or None
    position_price = 0.0

    for i in range(len(close)):
        if i < trend_filter_days:
            continue  # Need sufficient data for trend analysis

        price = close.iloc[i]
        dt = close.index[i]
        dt_date_only = pd.Timestamp(dt).normalize()

        if dt_date_only not in fgi_df.index:
            continue

        fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]

        # Calculate volatility-based stop
        if i >= volatility_lookback:
            recent_prices = close.iloc[i - volatility_lookback : i]
            volatility = recent_prices.std()
            volatility_stop = price - (volatility * _volatility_stop_multiplier)
        else:
            volatility_stop = price * (1 - max_drawdown_exit)  # Default stop

        # Market regime detection (using FGI trends over last 30 days)
        fgi_30d_avg = (
            fgi_df["fgi_value"].rolling(30).mean().loc[dt_date_only]
            if len(fgi_df) >= 30
            else fgi_val
        )
        fgi_trend = (
            "bull"
            if fgi_val > fgi_30d_avg + 5
            else "bear"
            if fgi_val < fgi_30d_avg - 5
            else "sideways"
        )

        # Adjust thresholds based on market regime
        if fgi_trend == "bull":
            effective_fear_threshold = fear_entry_threshold - 5  # 25
            effective_greed_threshold = greed_exit_threshold + 5  # 75
        elif fgi_trend == "bear":
            effective_fear_threshold = fear_entry_threshold + 5  # 35
            effective_greed_threshold = greed_exit_threshold - 5  # 65
        else:
            effective_fear_threshold = fear_entry_threshold
            effective_greed_threshold = greed_exit_threshold

        # Enhanced entry conditions
        # Primary: Extreme fear (bear market capitulation)
        fear_signal = fgi_val <= effective_fear_threshold

        # Secondary: Momentum divergence (price weakness in uptrends)
        momentum_signal = False
        if i >= 50:  # Need sufficient data for momentum
            recent_low = close.iloc[i - 50 : i].min()
            price_weakness = price < recent_low * 1.02  # Within 2% of 50-day low
            fgi_not_extreme = fgi_val > effective_fear_threshold + 10  # FGI not too low
            momentum_signal = price_weakness and fgi_not_extreme and fgi_trend == "bull"

        # Tertiary: Trend-following (ADX strong trend + price momentum in bull markets)
        trend_following_signal = False
        if i >= 50 and fgi_trend == "bull":
            try:
                # Calculate ADX for trend strength
                adx = calculate_adx(
                    high.iloc[: i + 1],
                    low.iloc[: i + 1],
                    close.iloc[: i + 1],
                    period=14,
                )
                if len(adx) >= 14:
                    current_adx = adx.iloc[-1]
                    # Strong trend if ADX > threshold
                    strong_trend = current_adx > adx_threshold

                    # Price momentum: Recent uptrend
                    recent_high = close.iloc[i - 20 : i].max()
                    recent_low = close.iloc[i - 20 : i].min()
                    if recent_high > recent_low:
                        price_momentum = (price - recent_low) / (
                            recent_high - recent_low
                        ) > 0.5  # Lower threshold for more signals

                        trend_following_signal = strong_trend and price_momentum
            except Exception:
                trend_following_signal = False

        # Combined long entry: Extreme fear OR momentum divergence OR trend-following
        is_buy = fear_signal or momentum_signal or trend_following_signal

        # Short selling signals
        is_short = False
        is_cover = False

        # Short entry: Extreme greed (sell when euphoria peaks)
        short_signal = fgi_val >= effective_greed_threshold

        # Short exit (cover): Extreme fear (buy back when panic sets in)
        cover_signal = fgi_val <= effective_fear_threshold

        # Additional short signals for bull markets
        short_momentum_signal = False
        if i >= 20 and fgi_trend == "bull":
            recent_high = close.iloc[i - 20 : i].max()
            price_weakness_short = price > recent_high * 0.98  # Near recent highs
            short_momentum_signal = (
                price_weakness_short and fgi_val >= effective_greed_threshold - 10
            )

        is_short = short_signal or short_momentum_signal
        is_cover = cover_signal

        # Enhanced exit conditions
        is_exit = False
        is_cover = False  # For short positions

        if position_type == "long":
            # Exit long on FGI extreme greed
            if fgi_val >= effective_greed_threshold:
                is_exit = True

            # Exit on maximum drawdown (capital preservation)
            drawdown_pct = (position_price - price) / position_price
            if drawdown_pct >= max_drawdown_exit:
                is_exit = True

            # Exit on volatility-based stop
            if price <= volatility_stop:
                is_exit = True

        elif position_type == "short":
            # Cover short on FGI extreme fear
            if fgi_val <= effective_fear_threshold:
                is_cover = True

            # For shorts: profit when price goes down, loss when price goes up
            # Take profit if price drops enough (short profit target)
            profit_pct = (position_price - price) / position_price
            if (
                profit_pct >= max_drawdown_exit
            ):  # Price dropped 8%, take profits on short
                is_cover = True
            elif profit_pct <= -max_drawdown_exit:  # Price rose 8%, cut losses on short
                is_cover = True

            # Also cover on volatility-based stop (if price rises too much)
            if price >= position_price * (1 + max_drawdown_exit):
                is_cover = True

        # Execute signals (supports short selling)
        if position_type is None and is_buy:
            # Long entry
            entries.iloc[i, 0] = True
            position_type = "long"
            position_price = price

        elif position_type is None and is_short:
            # Short entry
            short_entries.iloc[i, 0] = True
            position_type = "short"
            position_price = price

        elif position_type == "long" and is_exit:
            # Long exit
            exits.iloc[i, 0] = True
            position_type = None

        elif position_type == "short" and is_cover:
            # Short exit (cover)
            short_exits.iloc[i, 0] = True
            position_type = None

    # Force close any remaining positions at the end
    if i == len(close) - 1 and position_type is not None:
        if position_type == "long":
            exits.iloc[i, 0] = True
        elif position_type == "short":
            short_exits.iloc[i, 0] = True

    if entries.sum().sum() == 0 and short_entries.sum().sum() == 0:
        return {
            "granularity": granularity_name,
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "outperformance": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
        }

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        freq="1d",
        init_cash=INITIAL_CAPITAL,
        fees=(MAKER_FEE, TAKER_FEE),
    )

    stats = pf.stats()
    benchmark_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    return {
        "granularity": granularity_name,
        "total_return": stats.get("Total Return [%]", 0.0),
        "benchmark_return": benchmark_return,
        "outperformance": stats.get("Total Return [%]", 0.0) - benchmark_return,
        "max_drawdown": stats.get("Max Drawdown [%]", 0.0),
        "total_trades": int(stats.get("Total Trades", 0)),
        "win_rate": stats.get("Win Rate [%]", 0.0),
        "sharpe_ratio": stats.get("Sharpe Ratio", 0.0),
    }


def run_parameter_optimization(fgi_df: pd.DataFrame, optimization_type: str = "grid"):
    """Run parameter optimization for the trading strategy.

    Args:
        fgi_df: Fear & Greed Index data
        optimization_type: Type of optimization (grid, random, walk_forward)
    """
    from src.parameter_optimizer import (
        optimize_strategy_parameters,
        update_best_params,
    )

    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Optimization type: {optimization_type.upper()}")

    daily_ohlcv = fetch_unified_price_data("ETH-USD", START_DATE, END_DATE, "1d")

    if daily_ohlcv is None or len(daily_ohlcv) < 100:
        print("Insufficient data for optimization")
        return

    close = daily_ohlcv["close"]
    print(f"Data points: {len(close)}")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    results = optimize_strategy_parameters(
        close=close,
        fgi_df=fgi_df,
        granularity="ONE_DAY",
        pred_series=None,
        optimization_type=optimization_type,
        freq="1d",
        save_results=True,
    )

    print("\nUpdating BEST_PARAMS with optimized values...")
    update_best_params(results["best_params"])
    print(f"New BEST_PARAMS: {BEST_PARAMS}")


if __name__ == "__main__":
    main()
