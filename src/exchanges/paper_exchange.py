"""
Paper Exchange implementation for simulated trading.
"""

import uuid
import random
from typing import Dict, List, Optional
from datetime import datetime

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
)

DEFAULT_PRICES = {
    "BTC-USD": 90792.96,
    "ETH-USD": 3121.04,
    "XRP-USD": 2.09,
    "SOL-USD": 182.50,
    "BNB-USD": 625.30,
    "DOGE-USD": 0.38,
    "UNI-USD": 12.45,
}


class PaperExchange(ExchangeInterface):
    SUPPORTED_ORDER_TYPES = [OrderType.MARKET, OrderType.LIMIT]
    SUPPORTED_TIME_IN_FORCE = [TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC]
    SUPPORTS_SHORT_SELLING = True
    SUPPORTS_MARGIN = False
    PAPER_TRADING_AVAILABLE = True
    
    def __init__(
        self,
        api_key=None,
        secret_key=None,
        paper=True,
        initial_balance=10000.0,
        **kwargs
    ):
        super().__init__(api_key, secret_key, paper, **kwargs)
        
        self._account = Account(
            id="paper-" + str(uuid.uuid4())[:8],
            cash=initial_balance,
            portfolio_value=0.0,
            buying_power=initial_balance,
        )
        
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._trade_history: List[Dict] = []
        self._price_data = DEFAULT_PRICES.copy()
        self._price_volatility = {symbol: price * 0.0001 for symbol, price in DEFAULT_PRICES.items()}
    
    def name(self) -> str:
        return "PaperTrading"
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self):
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_account(self) -> Account:
        portfolio_value = sum(pos.market_value for pos in self._positions.values())
        self._account.portfolio_value = portfolio_value
        return self._account
    
    def get_position(self, symbol: str) -> Position:
        current_price = self.get_current_price(symbol) or 0.0
        
        if symbol in self._positions:
            return self._positions[symbol]
        
        return Position(
            symbol=symbol,
            quantity=0.0,
            entry_price=0.0,
            current_price=current_price,
            market_value=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            side=PositionSide.LONG,
        )
    
    def get_all_positions(self) -> List[Position]:
        return list(self._positions.values())
    
    def submit_order(self, order: OrderRequest) -> Order:
        self.validate_order(order)
        
        current_price = self.get_current_price(order.symbol)
        if current_price is None:
            raise OrderError(f"Unknown symbol: {order.symbol}")
        
        if order.order_type == OrderType.MARKET:
            slippage = random.uniform(-0.0001, 0.0001)
            execution_price = current_price * (1 + slippage)
        else:
            execution_price = order.price or current_price
        
        return self._execute_order(order, execution_price, current_price)
    
    def submit_order_simple(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = None,
        order_type: str = "market",
    ) -> Order:
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        ot = OrderType.MARKET if order_type == "market" else OrderType.LIMIT
        
        if price is None:
            price = self.get_current_price(symbol) or 0.0
        
        order = OrderRequest(
            symbol=symbol,
            side=side_enum,
            order_type=ot,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.IOC,
        )
        
        return self.submit_order(order)
    
    def _execute_order(self, order: OrderRequest, execution_price: float, current_price: float) -> Order:
        order_value = order.quantity * execution_price
        
        if order.side == OrderSide.BUY:
            if order_value > self._account.cash:
                raise OrderError(f"Insufficient funds. Required: ${order_value:,.2f}")
            self._account.cash -= order_value
        else:
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                if order.quantity > abs(pos.quantity):
                    raise OrderError(f"Insufficient position. Required: {order.quantity}")
                self._account.cash += order_value
        
        now = datetime.now()
        order_obj = Order(
            id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            price=execution_price,
            status="filled",
            created_at=now,
            updated_at=now,
            filled_at=now,
        )
        
        self._update_position(order.symbol, order.side, order.quantity, current_price)
        
        self._trade_history.append({
            "order_id": order_obj.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": execution_price,
            "value": order_value,
            "timestamp": now.isoformat(),
        })
        
        return order_obj
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: float, current_price: float):
        current_price = current_price or 0.0
        
        if symbol not in self._positions:
            pos_qty = quantity if side == OrderSide.BUY else -quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=pos_qty,
                entry_price=current_price,
                current_price=current_price,
                market_value=abs(pos_qty) * current_price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                side=PositionSide.LONG if pos_qty > 0 else PositionSide.SHORT,
                avg_entry_price=current_price,
            )
        else:
            pos = self._positions[symbol]
            
            if side == OrderSide.BUY:
                new_qty = pos.quantity + quantity
                if pos.quantity == 0:
                    new_avg_price = current_price
                else:
                    new_avg_price = (pos.quantity * pos.entry_price + quantity * current_price) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_avg_price
            else:
                new_qty = pos.quantity - quantity
                pos.quantity = new_qty
            
            pos.side = PositionSide.LONG if pos.quantity > 0 else PositionSide.SHORT
            
            market_value = abs(pos.quantity) * current_price
            if pos.quantity != 0 and pos.entry_price > 0:
                if pos.quantity > 0:
                    unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
                    unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                else:
                    unrealized_pnl = (pos.entry_price - current_price) * abs(pos.quantity)
                    unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
            else:
                unrealized_pnl = 0.0
                unrealized_pnl_pct = 0.0
            
            pos.current_price = current_price
            pos.market_value = market_value
            pos.unrealized_pnl = unrealized_pnl
            pos.unrealized_pnl_pct = unrealized_pnl_pct
        
        if symbol in self._positions and self._positions[symbol].is_empty():
            del self._positions[symbol]
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False
    
    def get_order(self, order_id: str) -> Order:
        if order_id in self._orders:
            return self._orders[order_id]
        raise OrderError(f"Order not found: {order_id}")
    
    def get_open_orders(self) -> List[Order]:
        return list(self._orders.values())
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        if symbol in self._price_data:
            volatility = self._price_volatility.get(symbol, 0.0001)
            current_price = self._price_data[symbol]
            original_price = DEFAULT_PRICES.get(symbol, current_price)
            mean_reversion = 0.0001
            change = random.gauss(0, volatility) - (current_price - original_price) * mean_reversion
            new_price = current_price * (1 + change)
            min_price = original_price * 0.5
            max_price = original_price * 1.5
            new_price = max(min_price, min(max_price, new_price))
            self._price_data[symbol] = new_price
            return new_price
        return None
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices
    
    def normalize_symbol(self, symbol: str) -> str:
        return symbol
    
    def denormalize_symbol(self, symbol: str) -> str:
        return symbol
    
    def reset_account(self, initial_balance: float = 10000.0):
        self._account = Account(
            id="paper-" + str(uuid.uuid4())[:8],
            cash=initial_balance,
            portfolio_value=0.0,
            buying_power=initial_balance,
        )
        self._positions = {}
        self._orders = {}
        self._trade_history = []
        self._price_data = DEFAULT_PRICES.copy()
    
    def get_trade_history(self) -> List[Dict]:
        return self._trade_history