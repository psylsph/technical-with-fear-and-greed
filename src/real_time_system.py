"""
Integration module for real-time trading system.
Combines WebSocket feeds, event-driven architecture, and smart order routing.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .websocket_feed import (
    WebSocketPriceFeed,
    PriceUpdate,
    ArbitrageDetector,
)
from .event_driven import (
    EventBus,
    EventType,
    Event,
    TradingSystem,
)
from .order_routing import (
    OrderManager,
    SmartOrderRouter,
    OrderSide,
    OrderType,
)
import pandas as pd


class RealTimeTradingSystem:
    """Complete real-time trading system with all components."""

    def __init__(
        self,
        symbols: List[str],
        fgi_data: pd.DataFrame,
        exchanges: List[str] = None,
        config: Dict = None,
    ):
        """Initialize real-time trading system.

        Args:
            symbols: List of symbols to trade
            fgi_data: FGI DataFrame for analysis
            exchanges: List of exchanges to connect to
            config: System configuration
        """
        self.symbols = symbols
        self.exchanges = exchanges or ["coinbase", "binance"]
        self.config = config or {}
        self.fgi_data = fgi_data
        self.logger = logging.getLogger(__name__)

        # Initialize event bus
        self.event_bus = EventBus()

        # Initialize WebSocket price feed
        self.price_feed = WebSocketPriceFeed(
            symbols=symbols,
            exchanges=self.exchanges,
        )

        # Initialize arbitrage detector
        self.arb_detector = ArbitrageDetector(self.price_feed)

        # Initialize trading system
        self.trading_system = TradingSystem(
            fgi_data=fgi_data,
            config=self.config,
        )

        # Connect price feed to trading system
        self._setup_price_feed_callback()

        # Connect trading system to order manager if configured
        self.order_manager: Optional[OrderManager] = None
        if self.config.get("enable_trading", False):
            self._setup_order_manager()

        # Running state
        self.running = False

    def _setup_price_feed_callback(self):
        """Setup callback to process price updates through trading system."""

        async def price_callback(update: PriceUpdate):
            # Publish price update to event bus
            price_event = Event(
                event_type=EventType.PRICE_UPDATE,
                timestamp=update.timestamp,
                data={
                    "symbol": update.symbol,
                    "price": update.price,
                    "exchange": update.exchange,
                },
            )

            await self.event_bus.publish(price_event)

        self.price_feed.on_price_update = price_callback

    def _setup_order_manager(self):
        """Setup order manager for trade execution."""
        if "order_manager" not in self.config:
            return

        router_config = self.config["order_manager"].get("router", {})
        router = SmartOrderRouter(
            exchanges=[],
            max_slippage=router_config.get("max_slippage", 0.001),
            min_liquidity=router_config.get("min_liquidity", 1000.0),
        )

        self.order_manager = OrderManager(router)

        # Setup trade execution callback
        async def execute_trade(
            symbol: str, side: str, price: float, quantity: float
        ) -> Dict:
            order = asyncio.create_task(
                self.order_manager.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                )
            )

            result = await order

            return {
                "quantity": result.filled_quantity,
                "price": result.avg_fill_price,
                "order_id": result.order_id,
            }

        # Inject callback into trading system
        self.trading_system.trading_system.trade_executor.execute_trade_callback = (
            execute_trade
        )

    async def start(self):
        """Start the real-time trading system."""
        self.running = True
        self.logger.info("Starting real-time trading system")

        # Start trading system
        await self.trading_system.start()

        # Start WebSocket price feed
        self.price_feed.running = True

        # Create tasks for each exchange
        tasks = []

        for exchange in self.exchanges:
            for symbol in self.symbols:
                if exchange == "coinbase":
                    tasks.append(self.price_feed._connect_coinbase(symbol))
                elif exchange == "binance":
                    tasks.append(self.price_feed._connect_binance(symbol))
                elif exchange == "kraken":
                    tasks.append(self.price_feed._connect_kraken(symbol))
                elif exchange == "bybit":
                    tasks.append(self.price_feed._connect_bybit(symbol))

        # Run all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop the real-time trading system."""
        self.running = False
        self.logger.info("Stopping real-time trading system")

        await self.price_feed.stop()

    def get_latest_prices(self) -> Dict[str, Dict[str, PriceUpdate]]:
        """Get latest prices from all exchanges.

        Returns:
            Dictionary mapping symbols to exchange prices
        """
        return self.price_feed.latest_prices

    def get_arbitrage_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get recent arbitrage opportunities.

        Args:
            limit: Maximum number of opportunities

        Returns:
            List of arbitrage opportunities
        """
        return self.arb_detector.get_recent_opportunities(limit)

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status.

        Returns:
            Portfolio status dictionary
        """
        status = {
            "positions": self.trading_system.trade_executor.positions,
            "portfolio_value": self.trading_system.risk_monitor.portfolio_value,
            "peak_value": self.trading_system.risk_monitor.peak_value,
            "drawdown": (
                self.trading_system.risk_monitor.peak_value
                - self.trading_system.risk_monitor.portfolio_value
            )
            / self.trading_system.risk_monitor.peak_value
            if self.trading_system.risk_monitor.peak_value > 0
            else 0.0,
        }

        if self.order_manager:
            status["orders"] = self.order_manager.get_all_orders()

        return status

    def get_system_metrics(self) -> Dict:
        """Get system performance metrics.

        Returns:
            System metrics dictionary
        """
        metrics = {
            "symbols_tracked": self.symbols,
            "exchanges_connected": self.exchanges,
            "running": self.running,
            "uptime": datetime.now(),
        }

        # Add order manager metrics if available
        if self.order_manager:
            metrics["execution_stats"] = self.order_manager.router.get_execution_stats()

        # Add arbitrage metrics
        metrics["arbitrage_opportunities"] = len(self.arb_detector.opportunities)

        return metrics


class MonitoringDashboard:
    """Dashboard for monitoring real-time trading system."""

    def __init__(self, trading_system: RealTimeTradingSystem):
        """Initialize monitoring dashboard.

        Args:
            trading_system: Real-time trading system instance
        """
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)

        # Subscribe to all events for monitoring
        self.trading_system.event_bus.subscribe(
            EventType.PRICE_UPDATE, self._on_price_update
        )
        self.trading_system.event_bus.subscribe(
            EventType.SIGNAL_GENERATED, self._on_signal
        )
        self.trading_system.event_bus.subscribe(
            EventType.TRADE_EXECUTED, self._on_trade
        )
        self.trading_system.event_bus.subscribe(
            EventType.RISK_ALERT, self._on_risk_alert
        )
        self.trading_system.event_bus.subscribe(
            EventType.ARBITRAGE_OPPORTUNITY, self._on_arbitrage
        )

        # Metrics storage
        self.price_updates: List[Dict] = []
        self.signals: List[Dict] = []
        self.trades: List[Dict] = []
        self.risk_alerts: List[Dict] = []
        self.arbitrage_opportunities: List[Dict] = []

    async def _on_price_update(self, event: Event):
        """Handle price update events.

        Args:
            event: Price update event
        """
        self.price_updates.append(event.data)
        if len(self.price_updates) > 1000:
            self.price_updates = self.price_updates[-1000:]

        # Print real-time analysis for price updates
        symbol = event.data.get("symbol")
        price = event.data.get("price")
        exchange = event.data.get("exchange")
        timestamp = event.timestamp.strftime("%H:%M:%S")

        print(f"\n{'=' * 70}")
        print(f"[{timestamp}] PRICE UPDATE - {exchange.upper()} - {symbol}")
        print(f"{'=' * 70}")
        print(f"Current Price: ${price:,.2f}")
        print(f"Exchange: {exchange.upper()}")

        # Get current market conditions
        fgi_data = self.trading_system.fgi_data
        current_fgi = fgi_data["fgi_value"].iloc[-1]

        # Get market regime
        regime = self.trading_system.trading_system.signal_generator.market_regime

        # Get effective thresholds
        fear_threshold = 30
        greed_threshold = 70
        if regime == "bull":
            fear_threshold -= 5
            greed_threshold += 5
        elif regime == "bear":
            fear_threshold += 5
            greed_threshold -= 5

        print("\n--- MARKET ANALYSIS ---")
        print(f"FGI: {current_fgi:.0f}")
        print(f"Market Regime: {regime.upper()}")
        print(f"Effective Thresholds: Fear≤{fear_threshold}, Greed≥{greed_threshold}")

        # Determine current signal
        if current_fgi <= fear_threshold:
            signal = "BUY"
            confidence = (fear_threshold - current_fgi) / fear_threshold
            color = "🟢"
        elif current_fgi >= greed_threshold:
            signal = "SELL"
            confidence = (current_fgi - greed_threshold) / (100 - greed_threshold)
            color = "🔴"
        else:
            signal = "HOLD"
            confidence = 0.5
            color = "🟡"

        print("\n--- SIGNAL ANALYSIS ---")
        print(f"{color} Signal: {signal}")
        print(f"Confidence: {confidence:.2%}")

        # Check if we have a position
        position = self.trading_system.trading_system.trade_executor.positions.get(
            symbol, 0
        )
        if position > 0:
            print(f"Current Position: {position:.6f} {symbol}")

        # Action to be taken
        print("\n--- ACTION ---")
        if signal == "BUY" and position == 0:
            if (
                confidence >= 0.7
                and self.trading_system.trading_system.trade_executor.execute_trade_callback
            ):
                print("⚡ Action: EXECUTE BUY ORDER (high confidence)")
            else:
                print(f"👀 Action: MONITOR (confidence {confidence:.0%} < 70%)")
        elif signal == "SELL" and position > 0:
            if (
                confidence >= 0.7
                and self.trading_system.trading_system.trade_executor.execute_trade_callback
            ):
                print("⚡ Action: EXECUTE SELL ORDER (high confidence)")
            else:
                print(f"👀 Action: MONITOR (confidence {confidence:.0%} < 70%)")
        else:
            print("👀 Action: MONITOR (no action required)")

        # Check recent signals
        recent_signals = [s for s in self.signals if s.get("symbol") == symbol][-3:]
        if recent_signals:
            print(f"\n--- RECENT SIGNALS ({symbol}) ---")
            for i, sig in enumerate(reversed(recent_signals), 1):
                sig_time = sig.get("timestamp", "")
                if sig_time:
                    sig_time = sig_time.strftime("%H:%M:%S")
                print(
                    f"{i}. [{sig_time}] {sig.get('signal').upper()} at ${sig.get('price', 0):,.2f} (conf: {sig.get('confidence', 0):.2%})"
                )

    async def _on_signal(self, event: Event):
        """Handle signal generation events.

        Args:
            event: Signal event
        """
        self.signals.append(event.data)

        # Print detailed signal information
        symbol = event.data.get("symbol")
        signal = event.data.get("signal").upper()
        price = event.data.get("price", 0)
        confidence = event.data.get("confidence", 0)
        fgi = event.data.get("fgi", 0)
        regime = event.data.get("market_regime", "unknown").upper()
        timestamp = event.timestamp.strftime("%H:%M:%S")

        print(f"\n{'🔔' * 35}")
        print(f"[{timestamp}] SIGNAL GENERATED - {symbol}")
        print(f"{'🔔' * 35}")
        print(f"Signal: {signal}")
        print(f"Price: ${price:,.2f}")
        print(f"FGI: {fgi:.0f}")
        print(f"Market Regime: {regime}")
        print(f"Confidence: {confidence:.2%}")
        print(
            f"Effective Thresholds: Fear≤{event.data.get('fear_threshold', 0)}, Greed≥{event.data.get('greed_threshold', 0)}"
        )

        # Position check
        position = self.trading_system.trading_system.trade_executor.positions.get(
            symbol, 0
        )
        if position > 0:
            print(f"Current Position: {position:.6f} {symbol}")
        else:
            print("Current Position: None")

        # Action determined
        if signal == "BUY" and position == 0:
            if confidence >= 0.7:
                if self.trading_system.trading_system.trade_executor.execute_trade_callback:
                    print("\n⚡⚡⚡ ACTION: EXECUTING BUY ORDER ⚡⚡⚡")
                else:
                    print("\n⚠️  ACTION: Would execute BUY (trading disabled)")
            else:
                print(f"\n👀 ACTION: Monitoring (confidence {confidence:.0%} < 70%)")
        elif signal == "SELL" and position > 0:
            if confidence >= 0.7:
                if self.trading_system.trading_system.trade_executor.execute_trade_callback:
                    print("\n⚡⚡⚡ ACTION: EXECUTING SELL ORDER ⚡⚡⚡")
                else:
                    print("\n⚠️  ACTION: Would execute SELL (trading disabled)")
            else:
                print(f"\n👀 ACTION: Monitoring (confidence {confidence:.0%} < 70%)")
        else:
            print("\n👀 ACTION: No action required")

    async def _on_trade(self, event: Event):
        """Handle trade execution events.

        Args:
            event: Trade event
        """
        self.trades.append(event.data)

        # Print detailed trade execution information
        side = event.data.get("side").upper()
        quantity = event.data.get("quantity")
        symbol = event.data.get("symbol")
        price = event.data.get("price", 0)
        timestamp = event.timestamp.strftime("%H:%M:%S")

        print(f"\n{'💰' * 35}")
        print(f"[{timestamp}] TRADE EXECUTED - {symbol}")
        print(f"{'💰' * 35}")
        print(f"Side: {side}")
        print(f"Quantity: {quantity:.6f} {symbol}")
        print(f"Price: ${price:,.2f}")
        print(f"Total Value: ${quantity * price:,.2f}")

        signal_data = event.data.get("signal_data", {})
        if signal_data:
            print("\nSignal Details:")
            print(f"  Confidence: {signal_data.get('confidence', 0):.2%}")
            print(f"  FGI: {signal_data.get('fgi', 0):.0f}")
            print(
                f"  Market Regime: {signal_data.get('market_regime', 'unknown').upper()}"
            )

    async def _on_risk_alert(self, event: Event):
        """Handle risk alert events.

        Args:
            event: Risk alert event
        """
        self.risk_alerts.append(event.data)
        self.logger.warning(f"Risk alert: {event.data.get('type')} - {event.data}")

    async def _on_arbitrage(self, event: Event):
        """Handle arbitrage opportunity events.

        Args:
            event: Arbitrage event
        """
        self.arbitrage_opportunities.append(event.data)

        # Print detailed arbitrage opportunity
        symbol = event.data.get("symbol")
        min_exchange = event.data.get("min_exchange")
        max_exchange = event.data.get("max_exchange")
        min_price = event.data.get("min_price")
        max_price = event.data.get("max_price")
        spread_pct = event.data.get("spread_pct", 0)
        timestamp = event.timestamp.strftime("%H:%M:%S")

        print(f"\n{'💎' * 35}")
        print(f"[{timestamp}] ARBITRAGE OPPORTUNITY - {symbol}")
        print(f"{'💎' * 35}")
        print(f"Buy on: {min_exchange.upper()} at ${min_price:,.2f}")
        print(f"Sell on: {max_exchange.upper()} at ${max_price:,.2f}")
        print(f"Spread: {spread_pct:.2f}%")
        print(f"Profit Potential: ${(max_price - min_price):,.2f} per unit")

        if spread_pct > 1.0:
            print(f"\n⚠️  HIGH OPPORTUNITY: {spread_pct:.2f}% spread!")
        elif spread_pct > 0.5:
            print(f"\n✅ GOOD OPPORTUNITY: {spread_pct:.2f}% spread")

    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data.

        Returns:
            Dashboard data dictionary
        """
        return {
            "price_updates": self.price_updates[-100:],  # Last 100 updates
            "signals": self.signals[-20:],  # Last 20 signals
            "trades": self.trades[-50:],  # Last 50 trades
            "risk_alerts": self.risk_alerts[-10:],  # Last 10 alerts
            "arbitrage_opportunities": self.arbitrage_opportunities[-10:],  # Last 10
            "portfolio_status": self.trading_system.get_portfolio_status(),
            "system_metrics": self.trading_system.get_system_metrics(),
        }


def create_real_time_system(
    symbols: List[str],
    fgi_data: pd.DataFrame,
    config: Dict = None,
) -> RealTimeTradingSystem:
    """Create and configure a real-time trading system.

    Args:
        symbols: List of symbols to trade
        fgi_data: FGI DataFrame for analysis
        config: System configuration

    Returns:
        Configured real-time trading system
    """
    default_config = {
        "strategy": {
            "fear_threshold": 30,
            "greed_threshold": 70,
            "portfolio_value": 10000,
        },
        "max_drawdown": 0.10,
        "max_position_pct": 0.30,
        "enable_trading": False,
    }

    if config:
        default_config.update(config)

    return RealTimeTradingSystem(
        symbols=symbols,
        fgi_data=fgi_data,
        exchanges=default_config.get("exchanges", ["coinbase", "binance"]),
        config=default_config,
    )


async def run_real_time_system(
    symbols: List[str],
    fgi_data: pd.DataFrame,
    config: Dict = None,
    duration: int = 3600,
):
    """Run real-time trading system for a specified duration.

    Args:
        symbols: List of symbols to trade
        fgi_data: FGI DataFrame for analysis
        config: System configuration
        duration: Run duration in seconds (default 1 hour)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create system
    system = create_real_time_system(symbols, fgi_data, config)

    # Create dashboard
    dashboard = MonitoringDashboard(system)

    try:
        # Start system
        await system.start()

        # Run for specified duration
        await asyncio.sleep(duration)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await system.stop()

        # Print summary
        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)

        status = system.get_portfolio_status()
        metrics = system.get_system_metrics()

        print(f"Portfolio Value: ${status.get('portfolio_value', 0):,.2f}")
        print(f"Peak Value: ${status.get('peak_value', 0):,.2f}")
        print(f"Drawdown: {status.get('drawdown', 0):.2%}")
        print(f"Total Trades: {len(dashboard.trades)}")
        print(f"Total Signals: {len(dashboard.signals)}")
        print(f"Risk Alerts: {len(dashboard.risk_alerts)}")
        print(f"Arbitrage Opportunities: {len(dashboard.arbitrage_opportunities)}")

        if "execution_stats" in metrics:
            stats = metrics["execution_stats"]
            if stats:
                print("\nExecution Stats:")
                print(f"  Total Orders: {stats.get('total_orders', 0)}")
                print(f"  Avg Slippage: {stats.get('avg_slippage', 0):.4f}")
                print(f"  Max Slippage: {stats.get('max_slippage', 0):.4f}")
                print(f"  Total Fees: ${stats.get('total_fees', 0):,.2f}")


if __name__ == "__main__":
    # Example usage
    from ..data.data_fetchers import fetch_fear_greed_index

    async def main():
        # Fetch FGI data
        fgi_df = fetch_fear_greed_index()

        # Create system
        system = create_real_time_system(
            symbols=["BTC-USD", "ETH-USD"],
            fgi_data=fgi_df,
            config={
                "strategy": {"fear_threshold": 30, "greed_threshold": 70},
                "enable_trading": False,
            },
        )

        # Start system
        await system.start()

    asyncio.run(main())
