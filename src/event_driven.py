"""
Event-driven architecture for real-time trading signal generation.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import pandas as pd


class EventType(Enum):
    """Event types for the trading system."""

    PRICE_UPDATE = "price_update"
    FGI_UPDATE = "fgi_update"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    RISK_ALERT = "risk_alert"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    MARKET_REGIME_CHANGE = "market_regime_change"
    POSITION_UPDATED = "position_updated"
    ERROR = "error"


@dataclass
class Event:
    """Base event class."""

    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, {self.timestamp}, {list(self.data.keys())})"


class EventBus:
    """Central event bus for publishing and subscribing to events."""

    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                self.logger.debug(f"Unsubscribed from {event_type.value}")

    async def publish(self, event: Event):
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        self.logger.debug(f"Publishing {event.event_type.value} event")

        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    self.logger.error(
                        f"Error in subscriber callback for {event.event_type.value}: {e}"
                    )

                    # Publish error event
                    error_event = Event(
                        event_type=EventType.ERROR,
                        timestamp=datetime.now(),
                        data={
                            "original_event": event,
                            "error": str(e),
                        },
                    )
                    await self.publish(error_event)

    def clear_subscribers(self, event_type: Optional[EventType] = None):
        """Clear subscribers for an event type or all events.

        Args:
            event_type: Specific event type to clear, or None for all
        """
        if event_type:
            if event_type in self.subscribers:
                del self.subscribers[event_type]
                self.logger.debug(f"Cleared subscribers for {event_type.value}")
        else:
            self.subscribers.clear()
            self.logger.debug("Cleared all subscribers")


class SignalGenerator:
    """Generate trading signals from events."""

    def __init__(
        self,
        event_bus: EventBus,
        fgi_data: pd.DataFrame,
        strategy_config: Dict[str, Any] = None,
    ):
        """Initialize signal generator.

        Args:
            event_bus: Event bus for publishing signals
            fgi_data: FGI DataFrame for analysis
            strategy_config: Strategy configuration parameters
        """
        self.event_bus = event_bus
        self.fgi_data = fgi_data
        self.config = strategy_config or {}
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.current_prices: Dict[str, float] = {}
        self.current_signals: Dict[str, str] = {}
        self.market_regime: str = "unknown"

        # Subscribe to price updates
        self.event_bus.subscribe(EventType.PRICE_UPDATE, self._on_price_update)

        # Subscribe to FGI updates
        self.event_bus.subscribe(EventType.FGI_UPDATE, self._on_fgi_update)

    async def _on_price_update(self, event: Event):
        """Handle price update events.

        Args:
            event: Price update event
        """
        symbol = event.data.get("symbol")
        price = event.data.get("price")

        if symbol and price:
            self.current_prices[symbol] = price

            # Generate signal for this symbol
            await self._generate_signal(symbol)

    async def _on_fgi_update(self, event: Event):
        """Handle FGI update events.

        Args:
            event: FGI update event
        """
        fgi_value = event.data.get("fgi_value")

        if fgi_value:
            # Update market regime
            self._update_market_regime(fgi_value)

            # Regenerate signals for all tracked symbols
            for symbol in self.current_prices.keys():
                await self._generate_signal(symbol)

    def _update_market_regime(self, fgi_value: float):
        """Update market regime based on FGI.

        Args:
            fgi_value: Current FGI value
        """
        if len(self.fgi_data) >= 30:
            fgi_30d_avg = self.fgi_data["fgi_value"].rolling(30).mean().iloc[-1]

            if fgi_value > fgi_30d_avg + 5:
                new_regime = "bull"
            elif fgi_value < fgi_30d_avg - 5:
                new_regime = "bear"
            else:
                new_regime = "sideways"

            if new_regime != self.market_regime:
                self.market_regime = new_regime

                # Publish regime change event
                asyncio.create_task(
                    self.event_bus.publish(
                        Event(
                            event_type=EventType.MARKET_REGIME_CHANGE,
                            timestamp=datetime.now(),
                            data={
                                "old_regime": self.market_regime,
                                "new_regime": new_regime,
                                "fgi_value": fgi_value,
                            },
                        )
                    )
                )

    async def _generate_signal(self, symbol: str):
        """Generate trading signal for a symbol.

        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.current_prices:
            return

        price = self.current_prices[symbol]
        current_fgi = self.fgi_data["fgi_value"].iloc[-1]

        # Get effective thresholds based on market regime
        fear_threshold = self.config.get("fear_threshold", 30)
        greed_threshold = self.config.get("greed_threshold", 70)

        if self.market_regime == "bull":
            fear_threshold -= 5
            greed_threshold += 5
        elif self.market_regime == "bear":
            fear_threshold += 5
            greed_threshold -= 5

        # Generate signal
        if current_fgi <= fear_threshold:
            signal = "buy"
            confidence = (fear_threshold - current_fgi) / fear_threshold
        elif current_fgi >= greed_threshold:
            signal = "sell"
            confidence = (current_fgi - greed_threshold) / (100 - greed_threshold)
        else:
            signal = "hold"
            confidence = 0.5

        # Check if signal changed
        previous_signal = self.current_signals.get(symbol)
        if signal != previous_signal:
            self.current_signals[symbol] = signal

            # Publish signal event
            await self.event_bus.publish(
                Event(
                    event_type=EventType.SIGNAL_GENERATED,
                    timestamp=datetime.now(),
                    data={
                        "symbol": symbol,
                        "signal": signal,
                        "price": price,
                        "fgi": current_fgi,
                        "market_regime": self.market_regime,
                        "confidence": confidence,
                        "fear_threshold": fear_threshold,
                        "greed_threshold": greed_threshold,
                    },
                )
            )


class TradeExecutor:
    """Execute trades based on signals."""

    def __init__(
        self,
        event_bus: EventBus,
        execute_trade_callback: Callable = None,
        config: Dict[str, Any] = None,
    ):
        """Initialize trade executor.

        Args:
            event_bus: Event bus for listening to signals
            execute_trade_callback: Function to execute actual trades
            config: Configuration dict for position sizing
        """
        self.event_bus = event_bus
        self.execute_trade_callback = execute_trade_callback
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Position tracking
        self.positions: Dict[str, float] = {}

        # Subscribe to signal events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)

    async def _on_signal(self, event: Event):
        """Handle signal generation events.

        Args:
            event: Signal event
        """
        symbol = event.data.get("symbol")
        signal = event.data.get("signal")
        price = event.data.get("price")
        confidence = event.data.get("confidence", 0)

        if not symbol or not signal or not price:
            return

        # Only execute trades with high confidence
        if confidence < 0.7:
            return

        # Check if we should execute the trade
        current_position = self.positions.get(symbol, 0)

        if signal == "buy" and current_position == 0:
            # Execute buy order
            await self._execute_buy(symbol, price, event.data)
        elif signal == "sell" and current_position > 0:
            # Execute sell order
            await self._execute_sell(symbol, price, current_position, event.data)

    async def _execute_buy(
        self, symbol: str, price: float, signal_data: Dict[str, Any]
    ):
        """Execute buy order.

        Args:
            symbol: Trading pair symbol
            price: Current price
            signal_data: Additional signal data
        """
        self.logger.info(f"Executing BUY for {symbol} at ${price:.2f}")

        # Call callback if provided
        if self.execute_trade_callback:
            try:
                result = self.execute_trade_callback(
                    symbol=symbol,
                    side="buy",
                    price=price,
                    quantity=self._calculate_position_size(price),
                )

                # Update position
                self.positions[symbol] = self.positions.get(symbol, 0) + result.get(
                    "quantity", 0
                )

                # Publish trade executed event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.TRADE_EXECUTED,
                        timestamp=datetime.now(),
                        data={
                            "symbol": symbol,
                            "side": "buy",
                            "price": price,
                            "quantity": result.get("quantity", 0),
                            "signal_data": signal_data,
                        },
                    )
                )

            except Exception as e:
                self.logger.error(f"Error executing buy order: {e}")

    async def _execute_sell(
        self,
        symbol: str,
        price: float,
        quantity: float,
        signal_data: Dict[str, Any],
    ):
        """Execute sell order.

        Args:
            symbol: Trading pair symbol
            price: Current price
            quantity: Quantity to sell
            signal_data: Additional signal data
        """
        self.logger.info(f"Executing SELL for {symbol} at ${price:.2f}")

        # Call callback if provided
        if self.execute_trade_callback:
            try:
                self.execute_trade_callback(
                    symbol=symbol,
                    side="sell",
                    price=price,
                    quantity=quantity,
                )

                # Update position
                self.positions[symbol] = max(
                    0, self.positions.get(symbol, 0) - quantity
                )

                # Publish trade executed event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.TRADE_EXECUTED,
                        timestamp=datetime.now(),
                        data={
                            "symbol": symbol,
                            "side": "sell",
                            "price": price,
                            "quantity": quantity,
                            "signal_data": signal_data,
                        },
                    )
                )

            except Exception as e:
                self.logger.error(f"Error executing sell order: {e}")

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management.

        Args:
            price: Current price

        Returns:
            Position size (quantity)
        """
        # Simple position sizing: 10% of portfolio value / price
        portfolio_value = self.config.get("portfolio_value", 10000)
        position_value = portfolio_value * 0.10

        return position_value / price


class RiskMonitor:
    """Monitor risk and generate alerts."""

    def __init__(
        self,
        event_bus: EventBus,
        max_drawdown: float = 0.10,
        max_position_pct: float = 0.30,
    ):
        """Initialize risk monitor.

        Args:
            event_bus: Event bus for publishing alerts
            max_drawdown: Maximum allowed drawdown (10%)
            max_position_pct: Maximum position as percentage of portfolio (30%)
        """
        self.event_bus = event_bus
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.portfolio_value: float = 10000
        self.peak_value: float = 10000
        self.positions: Dict[str, Dict] = {}

        # Subscribe to trade events
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade)

    async def _on_trade(self, event: Event):
        """Handle trade execution events.

        Args:
            event: Trade event
        """
        symbol = event.data.get("symbol")
        side = event.data.get("side")
        price = event.data.get("price")
        quantity = event.data.get("quantity")

        if not all([symbol, side, price, quantity]):
            return

        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "avg_price": 0}

        if side == "buy":
            position = self.positions[symbol]
            total_quantity = position["quantity"] + quantity
            position["quantity"] = total_quantity
            position["avg_price"] = (
                position["avg_price"] * position["quantity"] + price * quantity
            ) / total_quantity
        else:
            self.positions[symbol]["quantity"] -= quantity

        # Check risk limits
        await self._check_position_sizes()
        await self._check_drawdown()

    async def _check_position_sizes(self):
        """Check if any position exceeds maximum size."""
        for symbol, position in self.positions.items():
            position_value = position["quantity"] * position["avg_price"]
            position_pct = position_value / self.portfolio_value

            if position_pct > self.max_position_pct:
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.RISK_ALERT,
                        timestamp=datetime.now(),
                        data={
                            "type": "position_size",
                            "symbol": symbol,
                            "position_value": position_value,
                            "position_pct": position_pct,
                            "max_pct": self.max_position_pct,
                        },
                    )
                )

    async def _check_drawdown(self):
        """Check if portfolio drawdown exceeds maximum."""
        current_value = self._calculate_portfolio_value()

        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value

            if drawdown > self.max_drawdown:
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.RISK_ALERT,
                        timestamp=datetime.now(),
                        data={
                            "type": "drawdown",
                            "peak_value": self.peak_value,
                            "current_value": current_value,
                            "drawdown": drawdown,
                            "max_drawdown": self.max_drawdown,
                        },
                    )
                )

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value.

        Returns:
            Portfolio value
        """
        cash = self.portfolio_value

        for position in self.positions.values():
            cash += position["quantity"] * position["avg_price"]

        return cash


class TradingSystem:
    """Complete event-driven trading system."""

    def __init__(self, fgi_data: pd.DataFrame, config: Dict[str, Any] = None):
        """Initialize trading system.

        Args:
            fgi_data: FGI DataFrame for analysis
            config: System configuration
        """
        self.config = config or {}
        self.event_bus = EventBus()
        self.fgi_data = fgi_data

        # Initialize components
        self.signal_generator = SignalGenerator(
            event_bus=self.event_bus,
            fgi_data=fgi_data,
            strategy_config=self.config.get("strategy", {}),
        )

        self.trade_executor = TradeExecutor(
            event_bus=self.event_bus,
            execute_trade_callback=self.config.get("execute_trade_callback"),
            config=self.config,
        )

        self.risk_monitor = RiskMonitor(
            event_bus=self.event_bus,
            max_drawdown=self.config.get("max_drawdown", 0.10),
            max_position_pct=self.config.get("max_position_pct", 0.30),
        )

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the trading system."""
        self.logger.info("Starting event-driven trading system")

        # Initial FGI update
        current_fgi = self.fgi_data["fgi_value"].iloc[-1]
        fgi_event = Event(
            event_type=EventType.FGI_UPDATE,
            timestamp=datetime.now(),
            data={"fgi_value": current_fgi},
        )
        await self.event_bus.publish(fgi_event)

    async def process_price_update(self, symbol: str, price: float):
        """Process a price update.

        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        price_event = Event(
            event_type=EventType.PRICE_UPDATE,
            timestamp=datetime.now(),
            data={"symbol": symbol, "price": price},
        )
        await self.event_bus.publish(price_event)

    def get_event_bus(self) -> EventBus:
        """Get the event bus instance.

        Returns:
            EventBus instance
        """
        return self.event_bus


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create sample FGI data
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    fgi_values = pd.Series([50 + i * 0.1 for i in range(100)], index=dates)
    fgi_df = pd.DataFrame({"fgi_value": fgi_values})

    # Create trading system
    system = TradingSystem(fgi_data=fgi_df)

    async def run_system():
        await system.start()

        # Simulate price updates
        for i in range(10):
            await system.process_price_update("BTC-USD", 45000 + i * 100)
            await asyncio.sleep(1)

    asyncio.run(run_system())
