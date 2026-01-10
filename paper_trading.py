#!/usr/bin/env python3
"""
Paper Trading Simulation for Fear & Greed Strategy
Simulates live trading without real money
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.config import BEST_PARAMS, DEFAULT_ASSET, INITIAL_CAPITAL, PROJECT_ROOT
from src.data.data_fetchers import (
    fetch_fear_greed_index,
    fetch_unified_price_data,
)
from src.sentiment import calculate_rsi_sentiment
from src.strategy import generate_signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates live trading with paper money."""

    def __init__(self, initial_capital: float = 10000.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position = 0.0  # Amount of asset held
        self.position_value = 0.0
        self.trades = []
        self.equity_curve = []
        self.state_file = os.path.join(PROJECT_ROOT, "paper_portfolio_state.json")

    def load_state(self):
        """Load previous state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.capital = state.get("capital", self.capital)
                self.position = state.get("position", 0.0)
                self.trades = state.get("trades", [])
            logger.info(
                f"Loaded state: capital=${self.capital:.2f}, position={self.position:.6f}"
            )

    def save_state(self):
        """Save current state to file."""
        state = {
            "capital": self.capital,
            "position": self.position,
            "trades": self.trades,
            "last_update": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_equity(self, current_price: float) -> float:
        """Calculate current equity."""
        return self.capital + (self.position * current_price)

    def execute_buy(self, price: float, amount: float, fee: float = 0.0):
        """Execute a buy order."""
        cost = amount * price + fee
        if cost > self.capital:
            logger.warning(
                f"Insufficient capital for buy: ${cost:.2f} > ${self.capital:.2f}"
            )
            return False

        self.capital -= cost
        self.position += amount
        self.trades.append(
            {
                "type": "BUY",
                "price": price,
                "amount": amount,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.info(f"BUY: {amount:.6f} @ ${price:.2f} (total: ${cost:.2f})")
        return True

    def execute_sell(
        self, price: float, amount: Optional[float] = None, fee: float = 0.0
    ):
        """Execute a sell order."""
        if amount is None:
            amount = self.position

        if amount > self.position:
            logger.warning(
                f"Insufficient position for sell: {amount:.6f} > {self.position:.6f}"
            )
            return False

        proceeds = amount * price - fee
        self.capital += proceeds
        self.position -= amount
        self.trades.append(
            {
                "type": "SELL",
                "price": price,
                "amount": amount,
                "proceeds": proceeds,
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.info(f"SELL: {amount:.6f} @ ${price:.2f} (proceeds: ${proceeds:.2f})")
        return True

    def get_performance(self) -> Dict:
        """Calculate performance metrics."""
        equity = (
            self.get_equity(self.trades[-1]["price"]) if self.trades else self.capital
        )
        total_return = (equity - self.initial_capital) / self.initial_capital * 100

        winning_trades = [
            t
            for t in self.trades
            if t["type"] == "SELL" and t.get("proceeds", 0) > t.get("cost", 0)
        ]
        win_rate = (
            len(winning_trades)
            / max(1, len([t for t in self.trades if t["type"] == "SELL"]))
            * 100
        )

        return {
            "initial_capital": self.initial_capital,
            "current_equity": equity,
            "total_return": total_return,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "position": self.position,
        }


def create_sentiment_proxy(close: pd.Series) -> pd.DataFrame:
    """Create FGI-equivalent DataFrame using RSI sentiment."""
    sentiment = calculate_rsi_sentiment(close, window=BEST_PARAMS["rsi_window"])

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


def run_paper_trading(asset: str = DEFAULT_ASSET, days: int = 30):
    """Run paper trading simulation."""
    logger.info("=" * 60)
    logger.info("PAPER TRADING SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Asset: {asset}")
    logger.info(f"Period: Last {days} days")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")

    # Fetch price data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.Timedelta(days=days + 30)).strftime("%Y-%m-%d")

    # Fetch price data (ETH-USD only)
    ohlcv = fetch_unified_price_data(asset, start_date, end_date, "1d")

    if ohlcv is None or len(ohlcv) < 10:
        logger.error("Insufficient data for paper trading")
        return

    # Get FGI data
    fgi_df = fetch_fear_greed_index()

    # Align FGI with price data
    close = ohlcv["close"]
    fgi_aligned = fgi_df.reindex(close.index, method="ffill").ffill().bfill().fillna(50)

    # Initialize paper trader
    trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
    trader.load_state()

    # Run simulation
    params = BEST_PARAMS.copy()
    fee = INITIAL_CAPITAL * 0.001  # 0.1% fee estimate

    logger.info(f"\nStarting simulation with ${trader.capital:.2f} capital")

    for i, (date, price) in enumerate(close.items()):
        if i < 14:  # Skip first days for RSI calculation
            continue

        # Get signal
        current_close = close.loc[:date]
        current_fgi_df = fgi_aligned.loc[:date]

        signal_result = generate_signal(
            close=current_close,
            fgi_df=current_fgi_df,
            rsi_window=params["rsi_window"],
            trail_pct=params["trail_pct"],
            buy_quantile=params["buy_quantile"],
            sell_quantile=params["sell_quantile"],
            ml_thresh=params["ml_thresh"],
        )

        signal = signal_result["signal"]
        position_size = (
            signal_result.get("indicators", {}).get("position_size_pct", 5.0) / 100
        )

        # Execute trades
        if signal == "buy" and trader.position == 0:
            buy_amount = (trader.capital * position_size) / price
            trader.execute_buy(price, buy_amount, fee)
            fee = INITIAL_CAPITAL * 0.001  # Reset fee

        elif signal == "sell" and trader.position > 0:
            trader.execute_sell(price, fee=fee)
            fee = INITIAL_CAPITAL * 0.001  # Reset fee

        # Check trailing stop
        if trader.position > 0:
            entry_price = trader.trades[-1]["price"]
            stop_price = entry_price * (1 - params["trail_pct"])
            if price <= stop_price:
                logger.info(f"TRAILING STOP triggered @ ${price:.2f}")
                trader.execute_sell(price, fee=fee)
                fee = INITIAL_CAPITAL * 0.001

        # Save state periodically
        if i % 10 == 0:
            trader.save_state()

    # Final results
    trader.save_state()
    performance = trader.get_performance()

    logger.info("\n" + "=" * 60)
    logger.info("PAPER TRADING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Final Equity: ${performance['current_equity']:.2f}")
    logger.info(f"Total Return: {performance['total_return']:.2f}%")
    logger.info(f"Total Trades: {performance['total_trades']}")
    logger.info(f"Win Rate: {performance['win_rate']:.1f}%")
    logger.info(f"Final Position: {performance['position']:.6f} {asset}")
    logger.info("=" * 60)

    return performance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Simulation")
    parser.add_argument("--asset", default="ETH-USD", help="Trading asset")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to simulate"
    )
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")

    args = parser.parse_args()

    # Override initial capital
    import src.config

    src.config.INITIAL_CAPITAL = args.capital

    run_paper_trading(args.asset, args.days)
