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
    if not args.live and not args.test:
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

    # Train ML model
    print("\nTraining ML Model...")
    daily_close = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, "1d")
    ml_model, pred_series = train_ml_model(daily_close, fgi_df)
    print("ML Model trained.")

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


if __name__ == "__main__":
    main()
