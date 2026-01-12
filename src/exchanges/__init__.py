"""
Exchange abstraction layer for multi-asset trading.

This module provides a unified interface for different exchanges (Alpaca, Coinbase, Paper)
allowing the trading engine to work with any exchange through a consistent API.

Usage:
    from src.exchanges import get_exchange, ExchangeInterface
    
    exchange = get_exchange("alpaca", api_key="...", secret_key="...", paper=True)
    position = exchange.get_position("BTC-USD")
    order = exchange.submit_order("BTC-USD", "buy", 0.001, "market")
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class OrderRequest:
    """Order request data class."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    time_in_force: TimeInForce = TimeInForce.DAY
    stop_price: Optional[float] = None  # Required for stop orders
    

@dataclass
class Order:
    """Order data class."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float
    price: float
    status: str
    created_at: datetime
    updated_at: datetime
    filled_at: Optional[datetime] = None
    
    def is_filled(self) -> bool:
        """Check if order is filled."""
        return self.status == "filled"
    
    def is_open(self) -> bool:
        """Check if order is open (pending)."""
        return self.status in ["pending", "open", "partially_filled"]


@dataclass
class Position:
    """Position data class."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: PositionSide
    avg_entry_price: float = 0.0
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT
    
    def is_empty(self) -> bool:
        """Check if position is empty (no position)."""
        return abs(self.quantity) < 0.00000001


@dataclass
class Account:
    """Account data class."""
    id: str
    cash: float
    portfolio_value: float
    buying_power: float
    day_trades_remaining: Optional[int] = None
    pattern_day_trader: bool = False
    
    def equity(self) -> float:
        """Calculate total equity (cash + positions value)."""
        return self.cash + self.portfolio_value


class ExchangeError(Exception):
    """Base exception for exchange errors."""
    pass


class OrderError(ExchangeError):
    """Exception for order-related errors."""
    pass


class ConnectionError(ExchangeError):
    """Exception for connection-related errors."""
    pass


class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations.
    
    All exchange implementations must inherit from this class and
    implement all abstract methods.
    """
    
    # Exchange capabilities (to be overridden by subclasses)
    SUPPORTED_ORDER_TYPES: List[OrderType] = []
    SUPPORTED_TIME_IN_FORCE: List[TimeInForce] = []
    SUPPORTS_SHORT_SELLING: bool = False
    SUPPORTS_MARGIN: bool = False
    PAPER_TRADING_AVAILABLE: bool = False
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = False,
        **kwargs
    ):
        """Initialize exchange interface.
        
        Args:
            api_key: API key for authentication
            secret_key: Secret key for authentication
            paper: Use paper trading mode if available
            **kwargs: Additional exchange-specific parameters
        """
        self.api_key = api_key or os.getenv(f"{self.name().upper()}_API_KEY")
        self.secret_key = secret_key or os.getenv(f"{self.name().upper()}_SECRET_KEY")
        self.paper = paper
        self.kwargs = kwargs
        self._connected = False
        
    @abstractmethod
    def name(self) -> str:
        """Return exchange name."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        pass
    
    @abstractmethod
    def get_account(self) -> Account:
        """Get account information.
        
        Returns:
            Account object with balance and buying power
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            
        Returns:
            Position object with current position details
        """
        pass
    
    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        """Get all open positions.
        
        Returns:
            List of Position objects for all open positions
        """
        pass
    
    @abstractmethod
    def submit_order(self, order: OrderRequest) -> Order:
        """Submit a new order.
        
        Args:
            order: OrderRequest object with order details
            
        Returns:
            Order object with order details and status
            
        Raises:
            OrderError: If order submission fails
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Get order details.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order object with order details
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open orders.
        
        Returns:
            List of Order objects for all open orders
        """
        pass
    
    @abstractmethod
    def get_closed_orders(self, limit: int = 10) -> List[Order]:
        """Get recently closed orders.
        
        Args:
            limit: Maximum number of orders to return
            
        Returns:
            List of Order objects for recently closed orders
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            
        Returns:
            Current price or None if not available
        """
        pass
    
    @abstractmethod
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to exchange-specific format.
        
        Args:
            symbol: Standard symbol (e.g., "BTC-USD")
            
        Returns:
            Exchange-specific symbol format
        """
        pass
    
    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        """Denormalize exchange-specific symbol to standard format.
        
        Args:
            symbol: Exchange-specific symbol
            
        Returns:
            Standard symbol format (e.g., "BTC-USD")
        """
        pass
    
    def validate_order(self, order: OrderRequest) -> bool:
        """Validate an order before submission.
        
        Args:
            order: OrderRequest to validate
            
        Returns:
            True if order is valid
            
        Raises:
            OrderError: If order is invalid
        """
        # Check order type is supported
        if order.order_type not in self.SUPPORTED_ORDER_TYPES:
            raise OrderError(f"Order type {order.order_type} not supported by {self.name()}")
        
        # Check time in force is supported
        if order.time_in_force not in self.SUPPORTED_TIME_IN_FORCE:
            raise OrderError(f"Time in force {order.time_in_force} not supported by {self.name()}")
        
        # Check quantity is positive
        if order.quantity <= 0:
            raise OrderError("Order quantity must be positive")
        
        # For short selling, check support
        if order.side == OrderSide.SELL and not self.SUPPORTS_SHORT_SELLING:
            raise OrderError(f"{self.name()} does not support short selling")
        
        # For limit orders, check price is provided
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise OrderError("Limit orders require a price")
        
        # For stop orders, check stop price is provided
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                raise OrderError("Stop orders require a stop price")
        
        return True
    
    def calculate_order_value(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None
    ) -> float:
        """Calculate order value in quote currency.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (uses current price if not provided)
            
        Returns:
            Order value in quote currency (e.g., USD)
        """
        if price is None:
            price = self.get_current_price(symbol)
            if price is None:
                raise OrderError(f"Cannot get price for {symbol} to calculate order value")
        
        return quantity * price
    
    def __repr__(self) -> str:
        return f"{self.name()}(paper={self.paper}, connected={self._connected})"
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


def get_exchange(
    exchange_type: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: bool = False,
    **kwargs
) -> ExchangeInterface:
    """Factory function to get an exchange instance.
    
    Args:
        exchange_type: Exchange type ('alpaca', 'coinbase', 'paper')
        api_key: API key for authentication
        secret_key: Secret key for authentication
        paper: Use paper trading mode if available
        **kwargs: Additional exchange-specific parameters
        
    Returns:
        ExchangeInterface implementation
        
    Raises:
        ValueError: If exchange type is not recognized
    """
    exchange_type = exchange_type.lower()
    
    if exchange_type == "alpaca":
        from .alpaca_exchange import AlpacaExchange
        return AlpacaExchange(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            **kwargs
        )
    elif exchange_type == "coinbase":
        from .coinbase_exchange import CoinbaseExchange
        return CoinbaseExchange(
            api_key=api_key,
            secret_key=secret_key,
            paper=False,  # Coinbase doesn't have paper trading
            **kwargs
        )
    elif exchange_type == "paper":
        from .paper_exchange import PaperExchange
        return PaperExchange(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")


# Re-export for convenience
__all__ = [
    "ExchangeInterface",
    "OrderRequest",
    "Order",
    "Position",
    "Account",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "PositionSide",
    "ExchangeError",
    "OrderError",
    "ConnectionError",
    "get_exchange",
]