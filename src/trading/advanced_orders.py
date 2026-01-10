"""
Advanced Order Execution: Limit orders and order splitting.
Reduces slippage and market impact for larger trades.
"""

from datetime import datetime
from typing import Dict, List, Tuple
from enum import Enum


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class LimitOrderManager:
    """Manage limit orders to reduce slippage."""

    def __init__(
        self,
        default_slippage_tolerance: float = 0.001,  # 0.1%
        max_wait_seconds: int = 300,
        price_improvement: float = 0.0005  # Try to get better price by 0.05%
    ):
        """
        Args:
            default_slippage_tolerance: Default slippage tolerance (default 0.1%)
            max_wait_seconds: Maximum seconds to wait for fill (default 300)
            price_improvement: Try to beat mid by this amount (default 0.05%)
        """
        self.default_slippage_tolerance = default_slippage_tolerance
        self.max_wait_seconds = max_wait_seconds
        self.price_improvement = price_improvement

        self.pending_orders = {}
        self.order_history = []

    def calculate_limit_price(
        self,
        side: OrderSide,
        current_price: float,
        slippage_tolerance: float = None,
        reference_price: float = None
    ) -> float:
        """
        Calculate optimal limit price.

        Args:
            side: Order side (buy/sell)
            current_price: Current market price
            slippage_tolerance: Slippage tolerance (optional)
            reference_price: Reference price (e.g., VWAP, previous close)

        Returns:
            Limit price to use
        """
        slippage = slippage_tolerance or self.default_slippage_tolerance

        if reference_price:
            # Use reference price for better execution
            if side == OrderSide.BUY:
                # Buy: don't pay more than reference + slippage
                limit_price = min(current_price, reference_price) * (1 + slippage)
            else:
                # Sell: don't accept less than reference - slippage
                limit_price = max(current_price, reference_price) * (1 - slippage)
        else:
            # Use current price
            if side == OrderSide.BUY:
                # Buy: limit at current + slippage (don't overpay)
                limit_price = current_price * (1 + slippage - self.price_improvement)
            else:
                # Sell: limit at current - slippage + improvement (don't undersell)
                limit_price = current_price * (1 - slippage + self.price_improvement)

        return limit_price

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        slippage_tolerance: float = None
    ) -> Dict:
        """
        Create a limit order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            current_price: Current market price
            slippage_tolerance: Slippage tolerance

        Returns:
            Order dict with price and details
        """
        limit_price = self.calculate_limit_price(
            side,
            current_price,
            slippage_tolerance
        )

        order = {
            "symbol": symbol,
            "side": side.value,
            "type": OrderType.LIMIT.value,
            "quantity": quantity,
            "limit_price": limit_price,
            "current_price": current_price,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "order_id": f"{symbol}_{side.value}_{datetime.now().timestamp()}",
        }

        self.pending_orders[order["order_id"]] = order

        return order

    def should_use_limit_order(
        self,
        quantity: float,
        current_price: float,
        portfolio_value: float,
        urgency: str = "normal"
    ) -> bool:
        """
        Determine if limit order should be used vs market order.

        Args:
            quantity: Order quantity
            current_price: Current price
            portfolio_value: Total portfolio value
            urgency: Trade urgency (low/normal/high)

        Returns:
            True if limit order recommended
        """
        # Use market order for high urgency or small orders
        if urgency == "high":
            return False

        # Calculate order size as percentage of portfolio
        order_value = quantity * current_price
        order_pct = order_value / portfolio_value if portfolio_value > 0 else 0

        # Use limit order for larger orders (>1% of portfolio)
        if order_pct > 0.01:
            return True

        # Use limit order for normal urgency on medium orders
        if urgency == "normal" and order_pct > 0.005:
            return True

        return False

    def check_order_fill(
        self,
        order_id: str,
        current_price: float,
        side: OrderSide
    ) -> Tuple[bool, str]:
        """
        Check if limit order would be filled.

        Args:
            order_id: Order ID
            current_price: Current market price
            side: Order side

        Returns:
            Tuple of (filled: bool, reason: str)
        """
        if order_id not in self.pending_orders:
            return False, "Order not found"

        order = self.pending_orders[order_id]
        limit_price = order["limit_price"]
        created_at = datetime.fromisoformat(order["created_at"])
        elapsed = (datetime.now() - created_at).total_seconds()

        # Check if order expired
        if elapsed > self.max_wait_seconds:
            # Cancel order and suggest market order
            del self.pending_orders[order_id]
            return False, f"Order expired after {elapsed}s, consider market order"

        # Check if price crossed limit
        if side == OrderSide.BUY:
            # Buy order fills if current price <= limit price
            if current_price <= limit_price:
                order["filled_at"] = datetime.now().isoformat()
                order["fill_price"] = current_price
                order["status"] = "filled"
                del self.pending_orders[order_id]
                self.order_history.append(order)
                return True, f"Filled at ${current_price:.2f} (limit was ${limit_price:.2f})"
        else:
            # Sell order fills if current price >= limit price
            if current_price >= limit_price:
                order["filled_at"] = datetime.now().isoformat()
                order["fill_price"] = current_price
                order["status"] = "filled"
                del self.pending_orders[order_id]
                self.order_history.append(order)
                return True, f"Filled at ${current_price:.2f} (limit was ${limit_price:.2f})"

        return False, f"Waiting (current: ${current_price:.2f}, limit: ${limit_price:.2f})"

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order["status"] = "cancelled"
            order["cancelled_at"] = datetime.now().isoformat()
            del self.pending_orders[order_id]
            self.order_history.append(order)
            return True
        return False

    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        return self.order_history[-limit:]


class OrderSplitter:
    """Split large orders into smaller chunks to reduce market impact."""

    def __init__(
        self,
        max_order_size_pct: float = 0.02,  # 2% of portfolio per order
        min_order_size: float = 0.001,     # Minimum order size
        chunk_count: int = 3,                # Default number of chunks
        delay_between_chunks: int = 30       # Seconds between orders
    ):
        """
        Args:
            max_order_size_pct: Maximum order size as % of portfolio
            min_order_size: Minimum order size
            chunk_count: Default number of chunks to split into
            delay_between_chunks: Seconds to wait between chunks
        """
        self.max_order_size_pct = max_order_size_pct
        self.min_order_size = min_order_size
        self.chunk_count = chunk_count
        self.delay_between_chunks = delay_between_chunks

        self.split_history = {}

    def should_split_order(
        self,
        quantity: float,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[bool, int]:
        """
        Check if order should be split.

        Args:
            quantity: Order quantity
            current_price: Current price
            portfolio_value: Portfolio value

        Returns:
            Tuple of (should_split: bool, num_chunks: int)
        """
        order_value = quantity * current_price
        order_pct = order_value / portfolio_value if portfolio_value > 0 else 0

        if order_pct > self.max_order_size_pct:
            # Calculate number of chunks needed
            num_chunks = int(order_pct / self.max_order_size_pct) + 1
            return True, min(num_chunks, self.chunk_count)

        return False, 1

    def split_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        current_price: float,
        portfolio_value: float
    ) -> List[Dict]:
        """
        Split an order into smaller chunks.

        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to trade
            current_price: Current market price
            portfolio_value: Portfolio value

        Returns:
            List of chunked orders with delays
        """
        should_split, num_chunks = self.should_split_order(
            total_quantity,
            current_price,
            portfolio_value
        )

        if not should_split:
            # Return single order
            return [{
                "symbol": symbol,
                "side": side.value,
                "quantity": total_quantity,
                "price": current_price,
                "chunk": 1,
                "total_chunks": 1,
                "delay_seconds": 0,
            }]

        # Calculate chunk sizes
        chunk_qty = total_quantity / num_chunks
        chunk_qty = round(chunk_qty, 6)

        orders = []
        for i in range(num_chunks):
            # Last chunk gets remainder
            if i == num_chunks - 1:
                qty = total_quantity - (chunk_qty * (num_chunks - 1))
            else:
                qty = chunk_qty

            # Skip if too small
            if qty < self.min_order_size:
                continue

            orders.append({
                "symbol": symbol,
                "side": side.value,
                "quantity": qty,
                "price": current_price,
                "chunk": i + 1,
                "total_chunks": num_chunks,
                "delay_seconds": i * self.delay_between_chunks,
            })

        # Record split
        self.split_history[f"{symbol}_{side.value}_{datetime.now().timestamp()}"] = {
            "original_quantity": total_quantity,
            "num_chunks": len(orders),
            "timestamp": datetime.now().isoformat(),
        }

        return orders

    def get_split_stats(self) -> Dict:
        """Get statistics about order splitting."""
        if not self.split_history:
            return {
                "total_splits": 0,
                "avg_chunks": 0,
            }

        total_splits = len(self.split_history)
        total_chunks = sum(s["num_chunks"] for s in self.split_history.values())
        avg_chunks = total_chunks / total_splits if total_splits > 0 else 0

        return {
            "total_splits": total_splits,
            "avg_chunks": avg_chunks,
            "max_chunks_seen": max(s["num_chunks"] for s in self.split_history.values()),
        }
