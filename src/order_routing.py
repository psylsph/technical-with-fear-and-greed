"""
Smart order routing and automated order execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import pandas as pd


class OrderSide(Enum):
    """Order side types."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status types."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data class."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    timestamp: datetime = None
    exchange: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ExchangeConnector:
    """Base class for exchange connectors."""

    def __init__(self, exchange_name: str, api_key: str = None, api_secret: str = None):
        """Initialize exchange connector.

        Args:
            exchange_name: Name of the exchange
            api_key: API key for authentication
            api_secret: API secret for authentication
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.logger = logging.getLogger(f"{__name__}.{exchange_name}")

    async def get_order_book(
        self, symbol: str, depth: int = 10
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Get current order book.

        Args:
            symbol: Trading pair symbol
            depth: Depth of order book

        Returns:
            Order book with bids and asks
        """
        raise NotImplementedError("Subclasses must implement get_order_book")

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance.

        Returns:
            Dictionary of asset balances
        """
        raise NotImplementedError("Subclasses must implement get_balance")

    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the exchange.

        Args:
            order: Order to submit

        Returns:
            Updated order with exchange order ID
        """
        raise NotImplementedError("Subclasses must implement submit_order")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement cancel_order")

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order status
        """
        raise NotImplementedError("Subclasses must implement get_order_status")


class SmartOrderRouter:
    """Smart order routing for best execution."""

    def __init__(
        self,
        exchanges: List[ExchangeConnector],
        max_slippage: float = 0.001,
        min_liquidity: float = 1000.0,
    ):
        """Initialize smart order router.

        Args:
            exchanges: List of exchange connectors
            max_slippage: Maximum allowed slippage (0.1%)
            min_liquidity: Minimum liquidity required ($1000)
        """
        self.exchanges = exchanges
        self.max_slippage = max_slippage
        self.min_liquidity = min_liquidity
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.execution_history: List[Dict] = []

    async def route_order(self, order: Order) -> Tuple[Order, str]:
        """Route order to best exchange.

        Args:
            order: Order to route

        Returns:
            Tuple of (executed order, exchange name)
        """
        self.logger.info(
            f"Routing order: {order.side.value} {order.quantity} {order.symbol}"
        )

        # Get order books from all exchanges
        order_books = await self._get_all_order_books(order.symbol)

        if not order_books:
            self.logger.error("No order books available")
            order.status = OrderStatus.REJECTED
            return order, ""

        # Select best exchange
        best_exchange, expected_slippage = self._select_best_exchange(
            order, order_books
        )

        if not best_exchange:
            self.logger.error("No suitable exchange found")
            order.status = OrderStatus.REJECTED
            return order, ""

        # Submit order to selected exchange
        order.exchange = best_exchange.exchange_name
        executed_order = await best_exchange.submit_order(order)

        # Track execution
        self._track_execution(executed_order, expected_slippage)

        return executed_order, best_exchange.exchange_name

    async def route_order_split(
        self, order: Order, min_split_size: float = 0.1
    ) -> Tuple[List[Order], List[str]]:
        """Route order by splitting across multiple exchanges.

        Args:
            order: Order to route
            min_split_size: Minimum split size percentage (10%)

        Returns:
            Tuple of (executed orders, exchange names)
        """
        self.logger.info(
            f"Routing split order: {order.side.value} {order.quantity} {order.symbol}"
        )

        # Get order books from all exchanges
        order_books = await self._get_all_order_books(order.symbol)

        if not order_books:
            self.logger.error("No order books available")
            order.status = OrderStatus.REJECTED
            return [order], []

        # Rank exchanges by execution quality
        ranked_exchanges = self._rank_exchanges(order, order_books)

        # Calculate optimal split
        splits = self._calculate_split(order, ranked_exchanges, min_split_size)

        # Execute splits
        executed_orders = []
        exchange_names = []

        for i, (exchange, split_size) in enumerate(splits):
            if split_size > 0:
                split_order = Order(
                    order_id=f"{order.order_id}_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=split_size,
                    price=order.price,
                    stop_price=order.stop_price,
                )

                executed_order = await exchange.submit_order(split_order)
                executed_orders.append(executed_order)
                exchange_names.append(exchange.exchange_name)

                self._track_execution(executed_order, 0.0)

        return executed_orders, exchange_names

    async def _get_all_order_books(self, symbol: str) -> Dict[ExchangeConnector, Dict]:
        """Get order books from all exchanges.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary mapping exchanges to order books
        """
        order_books = {}

        tasks = []
        for exchange in self.exchanges:
            tasks.append(exchange.get_order_book(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for exchange, result in zip(self.exchanges, results):
            if not isinstance(result, Exception):
                order_books[exchange] = result

        return order_books

    def _select_best_exchange(
        self, order: Order, order_books: Dict[ExchangeConnector, Dict]
    ) -> Tuple[Optional[ExchangeConnector], float]:
        """Select best exchange for order execution.

        Args:
            order: Order to execute
            order_books: Order books from all exchanges

        Returns:
            Tuple of (best exchange, expected slippage)
        """
        best_exchange = None
        best_score = float("-inf")
        best_slippage = 0.0

        for exchange, order_book in order_books.items():
            score, slippage = self._calculate_execution_score(order, order_book)

            if score > best_score:
                best_score = score
                best_exchange = exchange
                best_slippage = slippage

        # Check if slippage is acceptable
        if best_slippage > self.max_slippage:
            self.logger.warning(
                f"Best exchange {best_exchange.exchange_name} has slippage {best_slippage:.3f} "
                f"exceeding max {self.max_slippage:.3f}"
            )

        return best_exchange, best_slippage

    def _calculate_execution_score(
        self, order: Order, order_book: Dict[str, List[Tuple[float, float]]]
    ) -> Tuple[float, float]:
        """Calculate execution quality score.

        Args:
            order: Order to execute
            order_book: Order book

        Returns:
            Tuple of (score, expected_slippage)
        """
        if order.side == OrderSide.BUY:
            entries = order_book.get("asks", [])
        else:
            entries = order_book.get("bids", [])

        if not entries:
            return float("-inf"), 1.0

        # Calculate VWAP and slippage
        total_quantity = 0.0
        total_value = 0.0
        remaining_quantity = order.quantity

        for price, volume in entries:
            if remaining_quantity <= 0:
                break

            fill_quantity = min(volume, remaining_quantity)
            total_quantity += fill_quantity
            total_value += fill_quantity * price
            remaining_quantity -= fill_quantity

        if total_quantity < order.quantity * (1 - self.max_slippage):
            # Insufficient liquidity
            return float("-inf"), 1.0

        if total_quantity > 0:
            vwap = total_value / total_quantity
            midpoint = (entries[0][0] + entries[0][0]) / 2  # Rough midpoint
            slippage = abs(vwap - midpoint) / midpoint
        else:
            return float("-inf"), 1.0

        # Score based on slippage and fees
        # Lower slippage = higher score
        score = 1.0 - slippage

        return score, slippage

    def _rank_exchanges(
        self, order: Order, order_books: Dict[ExchangeConnector, Dict]
    ) -> List[Tuple[ExchangeConnector, float]]:
        """Rank exchanges by execution quality.

        Args:
            order: Order to execute
            order_books: Order books from all exchanges

        Returns:
            List of (exchange, score) tuples sorted by score
        """
        ranked = []

        for exchange, order_book in order_books.items():
            score, _ = self._calculate_execution_score(order, order_book)
            ranked.append((exchange, score))

        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def _calculate_split(
        self,
        order: Order,
        ranked_exchanges: List[Tuple[ExchangeConnector, float]],
        min_split_size: float,
    ) -> List[Tuple[ExchangeConnector, float]]:
        """Calculate optimal order split across exchanges.

        Args:
            order: Order to split
            ranked_exchanges: Ranked list of exchanges
            min_split_size: Minimum split size percentage

        Returns:
            List of (exchange, quantity) tuples
        """
        splits = []
        remaining_quantity = order.quantity
        min_quantity = order.quantity * min_split_size

        # Distribute order across top exchanges
        for exchange, score in ranked_exchanges:
            if remaining_quantity <= 0:
                break

            if score <= 0:
                continue

            # Calculate split size based on score
            if len(splits) < 2:
                # Top 2 exchanges get larger shares
                split_size = min(remaining_quantity, order.quantity * 0.6)
            else:
                # Remaining exchanges get smaller shares
                split_size = min(remaining_quantity, order.quantity * 0.2)

            # Ensure minimum split size
            if split_size >= min_quantity:
                splits.append((exchange, split_size))
                remaining_quantity -= split_size

        return splits

    def _track_execution(self, order: Order, expected_slippage: float):
        """Track order execution for analytics.

        Args:
            order: Executed order
            expected_slippage: Expected slippage from order book
        """
        if order.status == OrderStatus.FILLED:
            actual_slippage = 0.0

            if order.filled_quantity > 0 and order.price:
                # Calculate actual slippage (simplified)
                # In reality, compare to mid price at order time
                actual_slippage = abs(order.avg_fill_price - order.price) / order.price

            execution_record = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": order.price,
                "avg_fill_price": order.avg_fill_price,
                "fees": order.fees,
                "expected_slippage": expected_slippage,
                "actual_slippage": actual_slippage,
                "exchange": order.exchange,
                "timestamp": order.timestamp,
            }

            self.execution_history.append(execution_record)

            self.logger.info(
                f"Order executed: {order.side.value} {order.quantity} {order.symbol} "
                f"at ${order.avg_fill_price:.2f} (slippage: {actual_slippage:.4f})"
            )

    def get_execution_stats(self) -> Dict[str, float]:
        """Get execution statistics.

        Returns:
            Dictionary of execution statistics
        """
        if not self.execution_history:
            return {}

        df = pd.DataFrame(self.execution_history)

        return {
            "total_orders": len(df),
            "avg_slippage": df["actual_slippage"].mean(),
            "max_slippage": df["actual_slippage"].max(),
            "total_fees": df["fees"].sum(),
            "avg_fill_price_deviation": (
                (df["avg_fill_price"] - df["price"]).abs() / df["price"]
            ).mean(),
        }


class OrderManager:
    """Manage order lifecycle and state."""

    def __init__(self, router: SmartOrderRouter):
        """Initialize order manager.

        Args:
            router: Smart order router instance
        """
        self.router = router
        self.orders: Dict[str, Order] = {}
        self.logger = logging.getLogger(__name__)

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        split_order: bool = False,
    ) -> Order:
        """Submit a new order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            split_order: Whether to split order across exchanges

        Returns:
            Executed order
        """
        # Create order
        order = Order(
            order_id=f"order_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )

        self.logger.info(f"Submitting order: {order.order_id}")

        # Store order
        self.orders[order.order_id] = order

        # Execute order
        try:
            if split_order:
                executed_orders, exchanges = await self.router.route_order_split(order)
                if executed_orders:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = sum(
                        o.filled_quantity for o in executed_orders
                    )
                    order.avg_fill_price = (
                        sum(
                            o.avg_fill_price * o.filled_quantity
                            for o in executed_orders
                        )
                        / order.filled_quantity
                    )
                    order.fees = sum(o.fees for o in executed_orders)
            else:
                executed_order, exchange = await self.router.route_order(order)
                order.status = executed_order.status
                order.filled_quantity = executed_order.filled_quantity
                order.avg_fill_price = executed_order.avg_fill_price
                order.fees = executed_order.fees

        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if order.status not in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]:
            self.logger.warning(
                f"Order {order_id} cannot be cancelled (status: {order.status.value})"
            )
            return False

        # Find exchange connector and cancel
        for exchange in self.router.exchanges:
            if exchange.exchange_name == order.exchange:
                try:
                    success = await exchange.cancel_order(order_id)
                    if success:
                        order.status = OrderStatus.CANCELLED
                        self.logger.info(f"Order {order_id} cancelled")
                    return success
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {e}")
                    return False

        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)

    def get_all_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders, optionally filtered by status.

        Args:
            status: Filter by status (optional)

        Returns:
            List of orders
        """
        orders = list(self.orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        return orders


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create mock exchange connectors (would be real implementations)
    # exchanges = [ExchangeConnector("coinbase"), ExchangeConnector("binance")]
    # router = SmartOrderRouter(exchanges)
    # order_manager = OrderManager(router)

    # # Submit a buy order
    # order = asyncio.run(
    #     order_manager.submit_order(
    #         symbol="BTC-USD",
    #         side=OrderSide.BUY,
    #         order_type=OrderType.MARKET,
    #         quantity=0.5,
    #         split_order=True,
    #     )
    # )

    # print(f"Order executed: {order}")
