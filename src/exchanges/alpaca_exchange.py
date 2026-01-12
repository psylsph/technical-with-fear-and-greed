"""
Alpaca Exchange implementation for multi-asset trading.

This module provides the Alpaca-specific implementation of the ExchangeInterface,
supporting both live trading and paper trading modes.

Prerequisites:
    pip install alpaca-py

Usage:
    from src.exchanges import get_exchange, OrderSide, OrderType
    
    exchange = get_exchange(
        "alpaca",
        api_key="your_api_key",
        secret_key="your_secret_key",
        paper=True  # Use paper trading
    )
    
    with exchange:
        account = exchange.get_account()
        position = exchange.get_position("BTC-USD")
        order = exchange.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
"""

from typing import Dict, List, Optional
import logging

from . import (
    ExchangeInterface,
    OrderRequest,
    Order,
    Position,
    Account,
    OrderSide,
    OrderType,
    TimeInForce,
    PositionSide,
    OrderError,
    ConnectionError,
)

logger = logging.getLogger(__name__)

# Check for Alpaca availability
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
    from alpaca.trading.enums import OrderStatus
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
    from alpaca.trading.requests import StopLimitOrderRequest
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not available. Install with: pip install alpaca-py")
    # Define dummy placeholders to prevent NameError in type hints
    OrderStatus = "OrderStatus"


class AlpacaExchange(ExchangeInterface):
    """Alpaca exchange implementation.
    
    Alpaca is a commission-free stock and crypto brokerage API.
    This implementation supports:
    - Both live trading and paper trading
    - Market, limit, stop, and stop-limit orders
    - Crypto and stock trading
    - Short selling (crypto only)
    """
    
    # Exchange capabilities
    SUPPORTED_ORDER_TYPES = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]
    SUPPORTED_TIME_IN_FORCE = [TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK]
    SUPPORTS_SHORT_SELLING = True  # Crypto supports short selling
    SUPPORTS_MARGIN = True
    PAPER_TRADING_AVAILABLE = True
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = False,
        **kwargs
    ):
        """Initialize Alpaca exchange.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default: False)
            **kwargs: Additional parameters
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not installed. Run: pip install alpaca-py")
        
        super().__init__(api_key, secret_key, paper, **kwargs)
        
        self._client: Optional[TradingClient] = None
        self._data_client: Optional[CryptoHistoricalDataClient] = None
        self._account_id: Optional[str] = None
        
        # Determine base URL based on paper mode
        self._base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        
    def name(self) -> str:
        """Return exchange name."""
        return "Alpaca"
    
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            if not self.api_key or not self.secret_key:
                raise ConnectionError(
                    "Alpaca API key or secret key not provided. "
                    "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
                    "or pass api_key and secret_key parameters."
                )
            
            self._client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
            
            # Initialize data client (doesn't need auth for free tier usually, but better to be safe if pro)
            self._data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection by getting account
            account = self._client.get_account()
            self._account_id = account.id
            self._connected = True
            
            logger.info(f"Connected to Alpaca (paper={self.paper}): account={str(self._account_id)[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")
    
    def disconnect(self):
        """Disconnect from Alpaca API."""
        self._client = None
        self._account_id = None
        self._connected = False
        logger.info("Disconnected from Alpaca")
    
    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        if not self._client:
            return False
        
        try:
            # Try to get account to verify connection
            self._client.get_account()
            return True
        except Exception:
            self._connected = False
            return False
    
    def get_account(self) -> Account:
        """Get Alpaca account information."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            alpaca_account = self._client.get_account()
            
            account = Account(
                id=alpaca_account.id,
                cash=float(alpaca_account.cash),
                portfolio_value=float(alpaca_account.portfolio_value),
                buying_power=float(alpaca_account.buying_power),
                day_trades_remaining=int(getattr(alpaca_account, 'daytrades_remaining', 0) or 0) if getattr(alpaca_account, 'daytrades_remaining', None) else None,
                pattern_day_trader=alpaca_account.pattern_day_trader,
            )
            
            return account
            
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise ConnectionError(f"Failed to get account: {e}")
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a specific symbol."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            # Get all positions and find matching symbol
            positions = self._client.get_all_positions()
            
            # Try exact match first, then case-insensitive
            pos = next((p for p in positions if p.symbol == alpaca_symbol), None)
            if not pos:
                pos = next(
                    (p for p in positions if p.symbol.upper() == alpaca_symbol.upper()),
                    None,
                )
            
            current_price = self.get_current_price(symbol)
            
            if pos:
                qty = float(pos.qty)
                entry_price = float(pos.avg_entry_price) if pos.avg_entry_price else 0.0
                
                # Determine side
                side = PositionSide.LONG if qty > 0 else PositionSide.SHORT
                
                # Calculate P&L
                if qty != 0 and entry_price > 0 and current_price:
                    if side == PositionSide.LONG:
                        unrealized_pnl = (current_price - entry_price) * abs(qty)
                        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        unrealized_pnl = (entry_price - current_price) * abs(qty)
                        unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                else:
                    unrealized_pnl = 0.0
                    unrealized_pnl_pct = 0.0
                
                market_value = abs(qty) * current_price if current_price else 0.0
                
                position = Position(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=entry_price,
                    current_price=current_price or 0.0,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    side=side,
                    avg_entry_price=entry_price,
                )
                
                logger.debug(f"Position for {symbol}: {position}")
                return position
            else:
                # No position
                return Position(
                    symbol=symbol,
                    quantity=0.0,
                    entry_price=0.0,
                    current_price=current_price or 0.0,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    side=PositionSide.LONG,  # Default to long for empty positions
                )
                
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise ConnectionError(f"Failed to get position: {e}")
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            positions = []
            alpaca_positions = self._client.get_all_positions()
            
            for alpaca_pos in alpaca_positions:
                symbol = self.denormalize_symbol(alpaca_pos.symbol)
                position = self.get_position(symbol)
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get all positions: {e}")
            raise ConnectionError(f"Failed to get positions: {e}")
    
    def submit_order(self, order: OrderRequest) -> Order:
        """Submit an order to Alpaca."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        # Validate order
        self.validate_order(order)
        
        try:
            alpaca_symbol = self.normalize_symbol(order.symbol)
            
            # Convert order side
            side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Convert time in force
            tif_mapping = {
                TimeInForce.DAY: AlpacaTimeInForce.DAY,
                TimeInForce.GTC: AlpacaTimeInForce.GTC,
                TimeInForce.IOC: AlpacaTimeInForce.IOC,
                TimeInForce.FOK: AlpacaTimeInForce.FOK,
            }
            tif = tif_mapping.get(order.time_in_force, AlpacaTimeInForce.DAY)
            
            # Create order request based on type
            if order.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=str(order.quantity),
                    side=side,
                    time_in_force=tif,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise OrderError("Limit orders require a price")
                order_request = LimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=str(order.quantity),
                    side=side,
                    time_in_force=tif,
                    limit_price=str(order.price),
                )
            elif order.order_type == OrderType.STOP:
                if order.stop_price is None:
                    raise OrderError("Stop orders require a stop price")
                order_request = StopOrderRequest(
                    symbol=alpaca_symbol,
                    qty=str(order.quantity),
                    side=side,
                    time_in_force=tif,
                    stop_price=str(order.stop_price),
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.price is None or order.stop_price is None:
                    raise OrderError("Stop-limit orders require both price and stop price")
                order_request = StopLimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=str(order.quantity),
                    side=side,
                    time_in_force=tif,
                    limit_price=str(order.price),
                    stop_price=str(order.stop_price),
                )
            else:
                raise OrderError(f"Unsupported order type: {order.order_type}")
            
            # Submit order
            alpaca_order = self._client.submit_order(order_request)
            
            # Create Order object
            order_obj = Order(
                id=alpaca_order.id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else 0.0,
                status=self._map_order_status(alpaca_order.status),
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at,
                filled_at=alpaca_order.filled_at or None,
            )
            
            logger.info(f"Order submitted: {order_obj}")
            return order_obj
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise OrderError(f"Failed to submit order: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            self._client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Order:
        """Get order details."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            alpaca_order = self._client.get_order_by_id(order_id)
            
            # Map Alpaca symbol back to standard format
            symbol = self.denormalize_symbol(alpaca_order.symbol)
            
            # Determine side from Alpaca order
            if alpaca_order.side == AlpacaOrderSide.BUY:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Determine order type
            if alpaca_order.order_type.value == "market":
                order_type = OrderType.MARKET
            elif alpaca_order.order_type.value == "limit":
                order_type = OrderType.LIMIT
            elif alpaca_order.order_type.value == "stop":
                order_type = OrderType.STOP
            else:
                order_type = OrderType.MARKET
            
            order = Order(
                id=alpaca_order.id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else 0.0,
                status=self._map_order_status(alpaca_order.status),
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at,
                filled_at=alpaca_order.filled_at or None,
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise OrderError(f"Failed to get order: {e}")
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            orders = []
            alpaca_orders = self._client.get_orders(status="open")
            
            for alpaca_order in alpaca_orders:
                orders.append(self._convert_alpaca_order(alpaca_order))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise ConnectionError(f"Failed to get open orders: {e}")

    def get_closed_orders(self, limit: int = 10) -> List[Order]:
        """Get recently closed orders."""
        if not self._client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=limit,
                nested=True  # useful for finding legs of multi-leg orders if needed
            )
            
            alpaca_orders = self._client.get_orders(filter=request)
            orders = []
            
            for alpaca_order in alpaca_orders:
                try:
                    orders.append(self._convert_alpaca_order(alpaca_order))
                except Exception as e:
                    logger.warning(f"Skipping order conversion error: {e}")
                    continue
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get closed orders: {e}")
            raise ConnectionError(f"Failed to get closed orders: {e}")

    def _convert_alpaca_order(self, alpaca_order) -> Order:
        """Helper to convert Alpaca order to internal Order."""
        symbol = self.denormalize_symbol(alpaca_order.symbol)
        
        # Determine side
        if alpaca_order.side == AlpacaOrderSide.BUY:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Determine order type
        order_type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT
        }
        order_type = order_type_map.get(str(alpaca_order.order_type), OrderType.MARKET)
        
        price = 0.0
        if alpaca_order.filled_avg_price:
             price = float(alpaca_order.filled_avg_price)
        elif alpaca_order.limit_price:
             price = float(alpaca_order.limit_price)

        return Order(
            id=str(alpaca_order.id),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(alpaca_order.qty) if alpaca_order.qty else 0.0,
            filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
            price=price,
            status=self._map_order_status(alpaca_order.status),
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at,
            filled_at=alpaca_order.filled_at or None,
        )
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            alpaca_symbol = self.normalize_symbol(symbol, for_data=True)
            
            if not self._data_client:
                # Lazy initialization if not connected
                self._data_client = CryptoHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
            
            # Use the data client to get latest quote
            request_params = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
            quotes = self._data_client.get_crypto_latest_quote(request_params)
            quote = quotes.get(alpaca_symbol)
            
            if quote:
                # Use ask price for buying, bid price for selling
                # Return mid price as current price
                if quote.ask_price and quote.bid_price:
                    return (float(quote.ask_price) + float(quote.bid_price)) / 2
                elif quote.ask_price:
                    return float(quote.ask_price)
                elif quote.bid_price:
                    return float(quote.bid_price)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        prices = {}
        
        try:
            alpaca_symbols = [self.normalize_symbol(s, for_data=True) for s in symbols]
            
            if not self._data_client:
                self._data_client = CryptoHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                
            request_params = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbols)
            quotes = self._data_client.get_crypto_latest_quote(request_params)
            
            for symbol in symbols:
                alpaca_symbol = self.normalize_symbol(symbol, for_data=True)
                quote = quotes.get(alpaca_symbol)
                
                if quote:
                    price = None
                    if quote.ask_price and quote.bid_price:
                        price = (float(quote.ask_price) + float(quote.bid_price)) / 2
                    elif quote.ask_price:
                        price = float(quote.ask_price)
                    elif quote.bid_price:
                        price = float(quote.bid_price)
                        
                    if price is not None:
                        prices[symbol] = price
                        
        except Exception as e:
            logger.warning(f"Failed to get prices for {symbols}: {e}")
            
        return prices
    
    def normalize_symbol(self, symbol: str, for_data: bool = False) -> str:
        """Convert standard symbol to Alpaca format.
        
        Alpaca uses format: BTCUSD (no slash, no dash) for Trading API
        Alpaca uses format: BTC/USD (with slash) for Data API
        Standard format: BTC-USD (with dash)
        """
        if for_data:
            return symbol.replace("-", "/")
        return symbol.replace("-", "").replace("/", "")
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert Alpaca symbol to standard format.
        
        Alpaca uses format: BTCUSD
        Standard format: BTC-USD
        """
        # Crypto symbols are 4+ characters without special characters
        if len(symbol) >= 6 and symbol.isalpha():
            # Insert dash before last 3 characters (USD)
            if symbol.endswith("USD"):
                return f"{symbol[:-3]}-USD"
            elif symbol.endswith("USDT"):
                return f"{symbol[:-4]}-USDT"
        return symbol
    
    def _map_order_status(self, alpaca_status: OrderStatus) -> str:
        """Map Alpaca order status to standard status."""
        status_mapping = {
            OrderStatus.PENDING_NEW: "pending",
            OrderStatus.ACCEPTED: "pending",
            OrderStatus.NEW: "open",
            OrderStatus.PARTIALLY_FILLED: "partially_filled",
            OrderStatus.FILLED: "filled",
            OrderStatus.DONE_FOR_DAY: "done_for_day",
            OrderStatus.CANCELED: "cancelled",
            OrderStatus.EXPIRED: "expired",
            OrderStatus.REJECTED: "rejected",
        }
        return status_mapping.get(alpaca_status, "unknown")
    
    def get_trading_client(self) -> Optional[TradingClient]:
        """Get the underlying Alpaca trading client for advanced operations."""
        return self._client