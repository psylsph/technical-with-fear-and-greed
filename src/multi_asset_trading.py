"""
Multi-asset strategy execution engine.

This module extends the trading engine to support multiple assets
with parallel processing and asset-specific parameter optimization.
"""

import asyncio
import concurrent.futures
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import INITIAL_CAPITAL, TEST_STATE_FILE
from src.multi_asset_config import get_asset_config, validate_asset_portfolio
from src.multi_asset_data import data_manager
from src.strategy import generate_signal, run_strategy
from src.trading.trading_engine import (
    should_trade,
    should_trade_test,
)

# Exchange integration
try:
    from src.exchanges import (
        get_exchange,
        ExchangeInterface,
        OrderSide,
        OrderType,
        TimeInForce,
    )
    EXCHANGES_AVAILABLE = True
except ImportError:
    EXCHANGES_AVAILABLE = False
    get_exchange = None
    ExchangeInterface = None

# Telegram integration
try:
    from src.telegram_bot import (
        send_trade_notification,
        send_portfolio_notification,
        send_multi_asset_trade_notification,
        get_telegram_bot,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    send_trade_notification = None
    send_portfolio_notification = None
    send_multi_asset_trade_notification = None
    get_telegram_bot = None


class MultiAssetPortfolio:
    """Portfolio manager for multiple assets."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.entry_prices: Dict[str, float] = {}  # symbol -> entry price
        self.position_sides: Dict[str, str] = {}  # symbol -> "long" or "short"
        self.trade_history: List[Dict] = []
        
        # Load from saved state if exists
        self.load_state()
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str = "long",
    ):
        """Add a position to the portfolio."""
        asset_config = get_asset_config(symbol)
        position_value = abs(quantity) * price
        
        # Check position size limit
        max_position_value = self.get_total_value() * asset_config.trading.max_position_size_pct
        if position_value > max_position_value:
            # Scale down to max size
            quantity = (max_position_value / price) * (1 if quantity > 0 else -1)
            position_value = max_position_value
            print(f"Scaled down {symbol} position to {quantity:.6f} (max ${max_position_value:.2f})")
        
        # Update cash
        if side == "long":
            self.cash -= position_value
        else:  # short
            self.cash += position_value  # Receive cash when shorting
        
        # Update positions
        if symbol in self.positions:
            # Average entry price for existing position
            old_quantity = self.positions[symbol]
            old_value = old_quantity * self.entry_prices[symbol]
            new_value = quantity * price
            
            total_quantity = old_quantity + quantity
            if total_quantity != 0:
                self.entry_prices[symbol] = (old_value + new_value) / total_quantity
            self.positions[symbol] = total_quantity
        else:
            self.positions[symbol] = quantity
            self.entry_prices[symbol] = price
            self.position_sides[symbol] = side
        
        # Record trade
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "buy" if side == "long" else "short",
            "quantity": quantity,
            "price": price,
            "value": position_value,
            "cash_after": self.cash,
        })
        
        self.save_state()
    
    def remove_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ):
        """Remove a position from the portfolio."""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        
        current_quantity = self.positions[symbol]
        side = self.position_sides.get(symbol, "long")
        
        # Calculate P&L
        entry_price = self.entry_prices[symbol]
        if side == "long":
            pnl = (price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - price) * quantity
        
        # Update cash
        position_value = abs(quantity) * price
        if side == "long":
            self.cash += position_value  # Receive cash from sale
        else:  # short
            self.cash -= position_value  # Pay cash to cover short
        
        # Update position
        new_quantity = current_quantity - quantity
        if abs(new_quantity) < 0.000001:  # Essentially zero
            del self.positions[symbol]
            del self.entry_prices[symbol]
            del self.position_sides[symbol]
        else:
            self.positions[symbol] = new_quantity
            # Keep entry price for remaining position
        
        # Record trade
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "sell" if side == "long" else "cover",
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "cash_after": self.cash,
        })
        
        self.save_state()
        return pnl
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position."""
        if symbol not in self.positions:
            return 0.0
        
        quantity = self.positions[symbol]
        return abs(quantity) * current_price
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Get total portfolio value."""
        total = self.cash
        
        if current_prices:
            for symbol, quantity in self.positions.items():
                if symbol in current_prices:
                    price = current_prices[symbol]
                    total += quantity * price if quantity > 0 else -quantity * price
                else:
                    # Use entry price as fallback
                    price = self.entry_prices.get(symbol, 0)
                    total += quantity * price if quantity > 0 else -quantity * price
        
        return total
    
    def get_asset_allocation(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get current allocation percentage for each asset."""
        total_value = self.get_total_value(current_prices)
        if total_value == 0:
            return {}
        
        allocations = {}
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                position_value = abs(quantity) * current_prices[symbol]
                allocations[symbol] = position_value / total_value
        
        return allocations
    
    def save_state(self):
        """Save portfolio state to file."""
        state = {
            "cash": self.cash,
            "positions": self.positions,
            "entry_prices": self.entry_prices,
            "position_sides": self.position_sides,
            "last_updated": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(TEST_STATE_FILE), exist_ok=True)
        with open(TEST_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load portfolio state from file."""
        if os.path.exists(TEST_STATE_FILE):
            try:
                with open(TEST_STATE_FILE, "r") as f:
                    state = json.load(f)
                
                self.cash = state.get("cash", self.initial_capital)
                self.positions = state.get("positions", {})
                self.entry_prices = state.get("entry_prices", {})
                self.position_sides = state.get("position_sides", {})
            except Exception as e:
                print(f"Error loading portfolio state: {e}")
                # Start fresh
                self.cash = self.initial_capital
                self.positions = {}
                self.entry_prices = {}
                self.position_sides = {}


class MultiAssetTradingEngine:
    """Trading engine for multiple assets."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.portfolio = MultiAssetPortfolio()
        self.fgi_data = None
    
    def update_fgi_data(self):
        """Update Fear & Greed Index data."""
        self.fgi_data = data_manager.fetch_fgi_data()
    
    def analyze_asset_signal(
        self,
        symbol: str,
        current_price: Optional[float],
    ) -> Optional[Dict]:
        """Analyze trading signal for a single asset."""
        if self.fgi_data is None:
            self.update_fgi_data()
        
        if current_price is None or self.fgi_data is None:
            return None
        
        # Create price series for signal generation
        price_series = pd.Series([current_price], index=[pd.Timestamp.now(tz="UTC")])
        
        # Get asset-specific parameters
        asset_config = get_asset_config(symbol)
        
        # Generate signal with asset-specific parameters
        signal = generate_signal(
            close=price_series,
            fgi_df=self.fgi_data,
            rsi_window=asset_config.training.rsi_window,
            trail_pct=asset_config.trading.trailing_stop_pct,
            buy_quantile=asset_config.trading.fgi_fear_threshold / 100.0,
            sell_quantile=asset_config.trading.fgi_greed_threshold / 100.0,
            enable_short_selling=True,
            fear_entry_threshold=int(asset_config.trading.fgi_fear_threshold),
            greed_exit_threshold=int(asset_config.trading.fgi_greed_threshold),
            max_drawdown_exit=asset_config.trading.max_drawdown_pct,
        )
        
        if signal:
            signal["symbol"] = symbol
            signal["current_price"] = current_price
        
        return signal
    
    def analyze_multiple_assets(
        self,
        symbols: List[str],
    ) -> Dict[str, Optional[Dict]]:
        """Analyze signals for multiple assets in parallel."""
        # Get current prices
        current_prices = data_manager.get_current_prices(symbols)
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.analyze_asset_signal,
                    symbol,
                    current_prices.get(symbol, 0),
                ): symbol
                for symbol in symbols
                if symbol in current_prices and current_prices[symbol] is not None
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result()
                    results[symbol] = signal
                except Exception as e:
                    print(f"Error analyzing signal for {symbol}: {e}")
                    results[symbol] = None
        
        return results
    
    def should_trade_asset(
        self,
        symbol: str,
        signal_info: Dict,
        is_live: bool = False,
    ) -> Tuple[bool, str, float]:
        """Determine if we should trade an asset."""
        current_price = signal_info.get("current_price", 0)
        
        if is_live:
            # Live trading logic
            current_position = self.portfolio.positions.get(symbol, 0)
            trade_decision = should_trade(
                signal_info=signal_info,
                position_info=current_position,  # Can be float for backward compatibility
                is_live=True,
                account_info={"equity": self.portfolio.get_total_value()},
            )
        else:
            # Test trading logic
            current_position = self.portfolio.positions.get(symbol, 0)
            trade_decision = should_trade_test(
                signal_info=signal_info,
                current_eth=current_position,
            )
        
        if trade_decision:
            action, quantity = trade_decision
            return True, action, quantity
        
        return False, "hold", 0.0
    
    def execute_trades(
        self,
        symbols: List[str],
        is_live: bool = False,
        max_concurrent_trades: int = 3,
    ) -> Dict[str, Dict]:
        """Execute trades for multiple assets."""
        # Validate portfolio
        is_valid, message = validate_asset_portfolio(symbols)
        if not is_valid:
            print(f"Portfolio validation failed: {message}")
            return {}
        
        # Analyze signals for all assets
        signals = self.analyze_multiple_assets(symbols)
        
        # Get current prices for portfolio valuation
        current_prices = data_manager.get_current_prices(symbols)
        
        trades_executed = {}
        
        for symbol, signal in signals.items():
            if signal is None:
                continue
            
            should_trade_flag, action, quantity = self.should_trade_asset(
                symbol=symbol,
                signal_info=signal,
                is_live=is_live,
            )
            
            if not should_trade_flag or quantity == 0:
                continue
            
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
            
            # Execute trade
            try:
                if action in ["buy", "short"]:
                    # Entry trade
                    side = "long" if action == "buy" else "short"
                    self.portfolio.add_position(
                        symbol=symbol,
                        quantity=quantity if action == "buy" else -quantity,
                        price=current_price,
                        side=side,
                    )
                    
                    trades_executed[symbol] = {
                        "action": action,
                        "quantity": quantity,
                        "price": current_price,
                        "side": side,
                    }
                    
                    print(f"Executed {action} for {symbol}: {quantity:.6f} @ ${current_price:.2f}")
                
                elif action in ["sell", "cover"]:
                    # Exit trade
                    pnl = self.portfolio.remove_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=current_price,
                    )
                    
                    trades_executed[symbol] = {
                        "action": action,
                        "quantity": quantity,
                        "price": current_price,
                        "pnl": pnl,
                    }
                    
                    print(f"Executed {action} for {symbol}: {quantity:.6f} @ ${current_price:.2f}, P&L: ${pnl:.2f}")
            
            except Exception as e:
                print(f"Error executing trade for {symbol}: {e}")
        
        return trades_executed
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        granularity: str = "ONE_DAY",
    ) -> Dict[str, Dict]:
        """Run backtest for multiple assets."""
        # Fetch historical data
        all_data = data_manager.fetch_multiple_assets(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )
        
        if self.fgi_data is None:
            self.update_fgi_data()
        
        results = {}
        
        for symbol, data in all_data.items():
            if data is None or data.empty:
                print(f"No data for {symbol}")
                continue
            
            # Get asset-specific parameters
            asset_config = get_asset_config(symbol)
            
            # Run strategy
            try:
                result = run_strategy(
                    close=data['close'],
                    freq=granularity,
                    fgi_df=self.fgi_data,
                    granularity_name=granularity,
                    rsi_window=asset_config.training.rsi_window,
                    trail_pct=asset_config.trading.trailing_stop_pct,
                    buy_quantile=asset_config.trading.fgi_fear_threshold / 100.0,
                    sell_quantile=asset_config.trading.fgi_greed_threshold / 100.0,
                    max_drawdown_pct=asset_config.trading.max_drawdown_pct,
                )
                
                results[symbol] = result
                
                # Update asset performance metrics
                if "sharpe_ratio" in result:
                    asset_config.historical_sharpe = result["sharpe_ratio"]
                if "win_rate" in result:
                    asset_config.historical_win_rate = result["win_rate"]
                if "max_drawdown" in result:
                    asset_config.historical_max_drawdown = result["max_drawdown"]
            
            except Exception as e:
                print(f"Error running backtest for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def optimize_asset_parameters(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        optimization_type: str = "grid",
    ) -> Dict:
        """Optimize parameters for a single asset."""
        # This would integrate with the existing parameter optimizer
        # For now, return current parameters
        asset_config = get_asset_config(symbol)
        return asset_config.get_optimized_params()
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of current portfolio."""
        current_prices = data_manager.get_current_prices(list(self.portfolio.positions.keys()))
        total_value = self.portfolio.get_total_value(current_prices)
        
        # Calculate P&L for each position
        position_details = []
        total_pnl = 0.0
        
        for symbol, quantity in self.portfolio.positions.items():
            if symbol in current_prices and current_prices[symbol]:
                current_price = current_prices[symbol]
                entry_price = self.portfolio.entry_prices.get(symbol, 0)
                side = self.portfolio.position_sides.get(symbol, "long")
                
                if side == "long":
                    pnl = (current_price - entry_price) * quantity
                else:  # short
                    pnl = (entry_price - current_price) * abs(quantity)
                
                position_value = abs(quantity) * current_price
                pnl_pct = (pnl / (entry_price * abs(quantity))) * 100 if entry_price > 0 else 0
                
                position_details.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "side": side,
                    "value": position_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                
                total_pnl += pnl
        
        return {
            "cash": self.portfolio.cash,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "pnl_pct": (total_pnl / self.portfolio.initial_capital) * 100,
            "positions": position_details,
            "num_positions": len(self.portfolio.positions),
            "trade_count": len(self.portfolio.trade_history),
        }


# Global trading engine instance
trading_engine = MultiAssetTradingEngine()


def run_multi_asset_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    granularity: str = "ONE_DAY",
) -> Dict:
    """Convenience function for multi-asset backtesting."""
    return trading_engine.run_backtest(symbols, start_date, end_date, granularity)


def execute_multi_asset_trades(
    symbols: List[str],
    is_live: bool = False,
) -> Dict[str, Dict]:
    """Convenience function for multi-asset trading."""
    return trading_engine.execute_trades(symbols, is_live=is_live)


def get_portfolio_performance() -> Dict:
    """Get current portfolio performance."""
    return trading_engine.get_portfolio_summary()


async def monitor_assets_async(
    symbols: List[str],
    interval_seconds: int = 300,
    max_iterations: Optional[int] = None,
    is_live: bool = False,
):
    """Async monitoring loop for multiple assets with trade execution."""
    iteration = 0
    
    while max_iterations is None or iteration < max_iterations:
        print(f"\n--- Monitoring iteration {iteration + 1} ---")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current prices
        current_prices = data_manager.get_current_prices(symbols)
        print(f"Current prices: { {s: f'${p:.2f}' for s, p in current_prices.items() if p} }")
        
        # Execute trades based on signals
        trades = trading_engine.execute_trades(
            symbols=symbols,
            is_live=is_live,
            max_concurrent_trades=3,
        )
        
        # Report executed trades
        if trades:
            print(f"\n✅ Executed {len(trades)} trades:")
            for symbol, trade in trades.items():
                print(f"  {symbol}: {trade['action']} {trade['quantity']:.6f} @ ${trade['price']:.2f}")
        else:
            print("\n⏸️  No trades executed (all signals were HOLD)")
        
        # Get portfolio summary
        summary = trading_engine.get_portfolio_summary()
        print(f"Portfolio value: ${summary['total_value']:.2f}")
        print(f"Total P&L: ${summary['total_pnl']:.2f} ({summary['pnl_pct']:.1f}%)")
        print(f"Positions: {summary['num_positions']}")
        print(f"Total trades: {summary['trade_count']}")
        
        iteration += 1
        await asyncio.sleep(interval_seconds)


class ExchangeMultiAssetTradingEngine:
    """Multi-asset trading engine with exchange abstraction.
    
    This class extends the basic MultiAssetTradingEngine with support for
    real exchange connections (Alpaca, Coinbase, etc.) through the
    exchange abstraction layer.
    
    Usage:
        # For paper trading
        engine = ExchangeMultiAssetTradingEngine(
            exchange_type="paper",
            initial_capital=10000.0,
        )
        
        # For live trading with Alpaca
        engine = ExchangeMultiAssetTradingEngine(
            exchange_type="alpaca",
            api_key="your_api_key",
            secret_key="your_secret_key",
            paper=True,  # or False for live trading
        )
    """
    
    def __init__(
        self,
        exchange_type: str = "paper",
        initial_capital: float = INITIAL_CAPITAL,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        **kwargs
    ):
        """Initialize the exchange-based multi-asset trading engine.
        
        Args:
            exchange_type: Exchange type ('alpaca', 'coinbase', 'paper')
            initial_capital: Starting capital for paper trading
            api_key: API key for exchange authentication
            secret_key: Secret key for exchange authentication
            paper: Use paper trading mode (default: True)
            **kwargs: Additional exchange-specific parameters
        """
        self.exchange_type = exchange_type
        self.initial_capital = initial_capital
        self.paper = paper
        
        if not EXCHANGES_AVAILABLE:
            raise ImportError(
                "Exchange modules not available. "
                "Ensure src/exchanges/__init__.py exists."
            )
        
        # Initialize exchange
        self.exchange: ExchangeInterface = get_exchange(
            exchange_type=exchange_type,
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            initial_balance=initial_capital,
            **kwargs
        )
        
        # Portfolio state
        self._trade_history: List[Dict] = []
        
        # Initialize Telegram bot
        self._telegram_bot = None
        if TELEGRAM_AVAILABLE and get_telegram_bot:
            try:
                self._telegram_bot = get_telegram_bot()
                if self._telegram_bot.is_enabled():
                    def get_multi_asset_status():
                        """Return current multi-asset trading status for Telegram queries."""
                        summary = self.get_account_summary()
                        return {
                            "account": {
                                "equity": summary.get("total_value", 0),
                                "cash": summary.get("cash", 0),
                                "pnl": summary.get("total_pnl", 0),
                                "day_pnl": summary.get("day_pnl", 0),
                            },
                            "positions": [
                                {
                                    "symbol": pos["symbol"],
                                    "qty": pos["quantity"],
                                    "avg_entry": pos["entry_price"],
                                    "current_price": pos["current_price"],
                                    "unrealized_pnl": pos["pnl"],
                                    "unrealized_pnl_pct": pos["pnl_pct"],
                                }
                                for pos in summary.get("positions", [])
                            ],
                            "recent_trades": self._trade_history[-10:],
                        }
                    
                    self._telegram_bot.set_status_callback(get_multi_asset_status)
                    self._telegram_bot.start()
                    print("Telegram bot started for multi-asset trading")
                else:
                    print("Telegram bot not enabled (missing API keys)")
            except Exception as e:
                print(f"Failed to start Telegram bot: {e}")
        
    def connect(self) -> bool:
        """Connect to the exchange."""
        return self.exchange.connect()
    
    def disconnect(self):
        """Disconnect from the exchange."""
        self.exchange.disconnect()
        # Stop Telegram bot
        if self._telegram_bot:
            self._telegram_bot.stop()
    
    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        return self.exchange.is_connected()
    
    def get_account_summary(self) -> Dict:
        """Get account and portfolio summary."""
        account = self.exchange.get_account()
        positions = self.exchange.get_all_positions()
        
        # Build positions list
        positions_list = []
        for pos in positions:
            positions_list.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "side": pos.side.value,
                "value": pos.market_value,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
            })
        
        # Calculate totals
        total_position_value = sum(pos.market_value for pos in positions)
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        initial_capital = self.initial_capital
        
        return {
            "account": {
                "id": account.id,
                "cash": account.cash,
                "equity": account.portfolio_value,
                "buying_power": account.buying_power,
            },
            "total_value": account.portfolio_value,
            "total_pnl": total_pnl,
            "pnl_pct": (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0,
            "positions": positions_list,
            "num_positions": len([p for p in positions if not p.is_empty()]),
            "trade_count": len(self._trade_history),
        }
    
    def get_position(self, symbol: str) -> Dict:
        """Get position for a specific symbol."""
        position = self.exchange.get_position(symbol)
        
        return {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "side": position.side.value,
            "value": position.market_value,
            "pnl": position.unrealized_pnl,
            "pnl_pct": position.unrealized_pnl_pct,
            "is_empty": position.is_empty(),
            "is_long": position.is_long(),
            "is_short": position.is_short(),
        }
    
    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_type: str = "market",
    ) -> Dict:
        """Submit an order to the exchange.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            action: "buy", "sell", "short", "cover"
            quantity: Order quantity
            price: Price (for limit orders)
            order_type: "market" or "limit"
            
        Returns:
            Order result dictionary
        """
        # Convert action to order side
        if action in ["buy", "short"]:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Convert order type
        if order_type == "market":
            ot = OrderType.MARKET
        else:
            ot = OrderType.LIMIT
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=ot,
            quantity=quantity,
            price=price if order_type == "limit" else None,
            time_in_force=TimeInForce.IOC,
        )
        
        # Submit order
        order = self.exchange.submit_order(order_request)
        
        # Add to trade history
        self._trade_history.append({
            "order_id": order.id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": order.price,
            "status": order.status,
            "timestamp": order.created_at.isoformat(),
        })

        # Send Telegram notification
        if TELEGRAM_AVAILABLE and send_multi_asset_trade_notification:
            try:
                portfolio_value = self.get_account_summary().get("total_value", 0)
                send_multi_asset_trade_notification(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=order.price,
                    portfolio_value=portfolio_value,
                    reason=f"Signal-based {action}",
                )
            except Exception as e:
                print(f"Failed to send Telegram notification: {e}")

        return {
            "order_id": order.id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": order.price,
            "status": order.status,
            "filled": order.is_filled(),
        }
    
    def analyze_multiple_assets(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Analyze signals for multiple assets."""
        # Get current prices from exchange
        current_prices = self.exchange.get_current_prices(symbols)
        
        results = {}
        for symbol in symbols:
            if symbol not in current_prices:
                results[symbol] = None
                continue
            
            # Get asset-specific parameters
            asset_config = get_asset_config(symbol)
            
            # Generate signal
            signal = self._generate_asset_signal(
                symbol=symbol,
                current_price=current_prices[symbol],
                asset_config=asset_config,
            )
            
            results[symbol] = signal
        
        return results
    
    def _generate_asset_signal(
        self,
        symbol: str,
        current_price: float,
        asset_config,
    ) -> Optional[Dict]:
        """Generate trading signal for an asset."""
        from src.strategy import generate_signal
        import pandas as pd
        
        # Create price series
        price_series = pd.Series(
            [current_price],
            index=[pd.Timestamp.now(tz="UTC")]
        )
        
        # Get FGI data
        fgi_df = data_manager.get_fgi_data()
        if fgi_df is None or fgi_df.empty:
            return None
        
        # Generate signal
        signal = generate_signal(
            close=price_series,
            fgi_df=fgi_df,
            rsi_window=asset_config.training.rsi_window,
            trail_pct=asset_config.trading.trailing_stop_pct,
            buy_quantile=asset_config.trading.fgi_fear_threshold / 100.0,
            sell_quantile=asset_config.trading.fgi_greed_threshold / 100.0,
            enable_short_selling=True,
            fear_entry_threshold=int(asset_config.trading.fgi_fear_threshold),
            greed_exit_threshold=int(asset_config.trading.fgi_greed_threshold),
            max_drawdown_exit=asset_config.trading.max_drawdown_pct,
        )
        
        if signal:
            signal["symbol"] = symbol
            signal["current_price"] = current_price
        
        return signal
    
    async def monitor_assets_async(
        self,
        symbols: List[str],
        interval_seconds: int = 300,
        max_iterations: Optional[int] = None,
        is_live: bool = True,
    ) -> None:
        """Monitor and trade multiple assets continuously.
        
        Args:
            symbols: List of trading symbols
            interval_seconds: Seconds between iterations
            max_iterations: Maximum iterations (None for infinite)
            is_live: Execute real trades if True
        """
        iteration = 0
        
        # Connect to exchange
        if not self.exchange.is_connected():
            self.exchange.connect()
        
        print(f"\nStarting multi-asset monitoring for {symbols}")
        print(f"Exchange: {self.exchange.name()} (paper={self.paper})")
        print(f"Interval: {interval_seconds} seconds")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            while max_iterations is None or iteration < max_iterations:
                print(f"\n--- Monitoring iteration {iteration + 1} ---")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get current prices from exchange
                current_prices = self.exchange.get_current_prices(symbols)
                print(f"Current prices: { {s: f'${p:.2f}' for s, p in current_prices.items() if p} }")
                
                # Analyze signals
                signals = self.analyze_multiple_assets(symbols)
                
                # Execute trades
                trades_executed = {}
                for symbol, signal in signals.items():
                    if signal is None:
                        continue
                    
                    action = signal.get("action")
                    if action == "hold":
                        continue
                    
                    # Get asset config for position sizing
                    asset_config = get_asset_config(symbol)
                    position_size_pct = asset_config.trading.max_position_size_pct
                    
                    # Calculate position size
                    account = self.exchange.get_account()
                    position_value = account.portfolio_value * position_size_pct
                    current_price = current_prices.get(symbol, 0)
                    quantity = position_value / current_price if current_price > 0 else 0
                    
                    if quantity <= 0:
                        continue
                    
                    # Execute trade
                    try:
                        if is_live:
                            result = self.submit_order(
                                symbol=symbol,
                                action=action,
                                quantity=quantity,
                                price=current_price,
                            )
                            trades_executed[symbol] = {
                                "action": action,
                                "quantity": quantity,
                                "price": current_price,
                                "order_id": result.get("order_id"),
                            }
                            print(f"  ✅ Executed {action} for {symbol}: {quantity:.6f} @ ${current_price:.2f}")
                        else:
                            # Paper trading mode - just record the signal
                            trades_executed[symbol] = {
                                "action": action,
                                "quantity": quantity,
                                "price": current_price,
                            }
                            print(f"  📋 Paper {action.upper()} signal for {symbol}: {quantity:.6f} @ ${current_price:.2f}")
                    except Exception as e:
                        print(f"  ❌ Failed to execute {action} for {symbol}: {e}")
                
                if not trades_executed:
                    print("\n⏸️  No trades executed (all signals were HOLD)")
                else:
                    print(f"\n✅ Executed {len(trades_executed)} trade(s)")
                
                # Get portfolio summary
                summary = self.get_account_summary()
                print(f"\nPortfolio value: ${summary['total_value']:.2f}")
                print(f"Total P&L: ${summary['total_pnl']:.2f} ({summary['pnl_pct']:.1f}%)")
                print(f"Positions: {summary['num_positions']}")
                print(f"Total trades: {summary['trade_count']}")
                
                # Show positions
                if summary["positions"]:
                    print("\nCurrent Positions:")
                    for pos in summary["positions"]:
                        pnl_emoji = "🟢" if pos["pnl"] >= 0 else "🔴"
                        print(f"  {pnl_emoji} {pos['symbol']}: {pos['quantity']:+.6f} @ ${pos['entry_price']:.2f}")
                        print(f"      Current: ${pos['current_price']:.2f} | P&L: ${pos['pnl']:+.2f} ({pos['pnl_pct']:+.1f}%)")

                # Send Telegram portfolio notification
                if TELEGRAM_AVAILABLE and send_portfolio_notification:
                    try:
                        positions_for_telegram = [
                            {
                                "symbol": pos["symbol"],
                                "qty": pos["quantity"],
                                "value": pos["quantity"] * pos["current_price"],
                            }
                            for pos in summary.get("positions", [])
                        ]
                        send_portfolio_notification(
                            portfolio_summary={
                                "total_value": summary["total_value"],
                                "cash": summary.get("cash", summary["total_value"]),
                                "total_pnl": summary["total_pnl"],
                                "total_pnl_pct": summary["pnl_pct"],
                                "day_pnl": summary.get("day_pnl", 0),
                            },
                            positions=positions_for_telegram,
                        )
                    except Exception as e:
                        print(f"Failed to send portfolio Telegram notification: {e}")
                
                iteration += 1
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.exchange.disconnect()
            print("Exchange connection closed")


async def run_exchange_live_trading(
    symbols: List[str],
    exchange_type: str = "alpaca",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: bool = True,
    interval_seconds: int = 300,
    max_iterations: Optional[int] = None,
):
    """Run live trading with exchange integration.
    
    Args:
        symbols: List of trading symbols
        exchange_type: Exchange type ('alpaca', 'coinbase', 'paper')
        api_key: API key for exchange authentication
        secret_key: Secret key for exchange authentication
        paper: Use paper trading mode
        interval_seconds: Seconds between iterations
        max_iterations: Maximum iterations (None for infinite)
    """
    print("\n" + "=" * 60)
    print(f"LIVE TRADING MODE - {exchange_type.upper()}")
    print("=" * 60)
    
    try:
        engine = ExchangeMultiAssetTradingEngine(
            exchange_type=exchange_type,
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        
        await engine.monitor_assets_async(
            symbols=symbols,
            interval_seconds=interval_seconds,
            max_iterations=max_iterations,
            is_live=not paper,
        )
        
    except ImportError as e:
        print(f"Exchange modules not available: {e}")
        print("Ensure src/exchanges/ directory exists with exchange implementations.")
    except Exception as e:
        print(f"Error in live trading: {e}")
        raise
