"""
WebSocket integration for real-time price feeds.
"""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

try:
    import websockets

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websockets library not available. Install with: pip install websockets")

from datetime import datetime


@dataclass
class PriceUpdate:
    """Price update data class."""

    symbol: str
    price: float
    timestamp: datetime
    exchange: str
    volume: Optional[float] = None


class WebSocketPriceFeed:
    """WebSocket client for real-time price feeds from multiple exchanges."""

    def __init__(
        self,
        symbols: List[str],
        exchanges: List[str] = None,
        on_price_update: Callable[[PriceUpdate], None] = None,
    ):
        """Initialize WebSocket price feed.

        Args:
            symbols: List of symbols to track (e.g., ["BTC-USD", "ETH-USD"])
            exchanges: List of exchanges to connect to (default: ["coinbase", "binance"])
            on_price_update: Callback function for price updates
        """
        self.symbols = symbols
        self.exchanges = exchanges or ["coinbase", "binance"]
        self.on_price_update = on_price_update
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.running = False
        self.logger = logging.getLogger(__name__)

        # Exchange WebSocket URLs
        self.exchange_urls = {
            "coinbase": "wss://ws-feed.exchange.coinbase.com",
            "binance": "wss://stream.binance.com:9443/ws",
            "kraken": "wss://ws.kraken.com",
            "bybit": "wss://stream.bybit.com/v5/public/spot",
        }

        # Store latest prices
        self.latest_prices: Dict[str, Dict[str, PriceUpdate]] = {}

    async def _connect_coinbase(
        self, symbol: str
    ) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to Coinbase WebSocket.

        Args:
            symbol: Trading pair symbol

        Returns:
            WebSocket connection or None if failed
        """
        try:
            ws_url = self.exchange_urls["coinbase"]
            async with websockets.connect(
                ws_url, ping_interval=20, ping_timeout=10, close_timeout=10
            ) as websocket:
                # Subscribe to ticker channel
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": [symbol],
                    "channels": ["ticker"],
                }
                await websocket.send(json.dumps(subscribe_msg))

                self.logger.info(f"Connected to Coinbase for {symbol}")

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)

                        if data.get("type") == "ticker":
                            price = float(data.get("price", 0))
                            update = PriceUpdate(
                                symbol=symbol,
                                price=price,
                                timestamp=datetime.now(),
                                exchange="coinbase",
                                volume=float(data.get("volume_24h", 0)),
                            )

                            await self._handle_price_update(update)

                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except Exception as e:
                        self.logger.error(f"Error receiving Coinbase data: {e}")
                        break

        except Exception as e:
            self.logger.error(f"Failed to connect to Coinbase: {e}")
            return None

    async def _connect_binance(
        self, symbol: str
    ) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to Binance WebSocket.

        Args:
            symbol: Trading pair symbol

        Returns:
            WebSocket connection or None if failed
        """
        try:
            # Convert symbol to Binance format using aliases
            binance_aliases = {
                "BTC-USD": "BTCUSDT",
                "ETH-USD": "ETHUSDT",
                "BNB-USD": "BNBUSDT",
                "SOL-USD": "SOLUSDT",
                "XRP-USD": "XRPUSDT",
                "ADA-USD": "ADAUSDT",
                "AVAX-USD": "AVAXUSDT",
                "DOGE-USD": "DOGEUSDT",
                "DOT-USD": "DOTUSDT",
                "LINK-USD": "LINKUSDT",
            }
            binance_symbol = (
                binance_aliases.get(symbol, symbol.replace("-", "").lower()) + "t"
            )
            stream_name = f"{binance_symbol.lower()}@ticker"
            ws_url = f"{self.exchange_urls['binance']}/{stream_name}"

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                self.logger.info(f"Connected to Binance for {symbol}")

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)

                        if "p" in data:  # Current price
                            price = float(data["p"])
                            update = PriceUpdate(
                                symbol=symbol,
                                price=price,
                                timestamp=datetime.now(),
                                exchange="binance",
                                volume=float(data.get("v", 0)),
                            )

                            await self._handle_price_update(update)

                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Binance timeout for {symbol}, reconnecting..."
                        )
                        break
                    except Exception as e:
                        self.logger.error(f"Error receiving Binance data: {e}")
                        break

        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return None

    async def _connect_kraken(
        self, symbol: str
    ) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to Kraken WebSocket.

        Args:
            symbol: Trading pair symbol

        Returns:
            WebSocket connection or None if failed
        """
        try:
            # Convert symbol to Kraken format
            kraken_symbol = self._convert_to_kraken_symbol(symbol)
            ws_url = self.exchange_urls["kraken"]

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                # Subscribe to ticker
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": [kraken_symbol],
                    "subscription": {"name": "ticker"},
                }
                await websocket.send(json.dumps(subscribe_msg))

                self.logger.info(f"Connected to Kraken for {symbol}")

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)

                        if isinstance(data, list) and len(data) > 0:
                            # Kraken ticker format
                            ticker_data = data[0]
                            if "c" in ticker_data:  # Close price
                                price = float(ticker_data["c"][0])
                                update = PriceUpdate(
                                    symbol=symbol,
                                    price=price,
                                    timestamp=datetime.now(),
                                    exchange="kraken",
                                )

                                await self._handle_price_update(update)

                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Kraken timeout for {symbol}, reconnecting..."
                        )
                        break
                    except Exception as e:
                        self.logger.error(f"Error receiving Kraken data: {e}")
                        break

        except Exception as e:
            self.logger.error(f"Failed to connect to Kraken: {e}")
            return None

    async def _connect_bybit(
        self, symbol: str
    ) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to Bybit WebSocket.

        Args:
            symbol: Trading pair symbol

        Returns:
            WebSocket connection or None if failed
        """
        try:
            # Convert symbol to Bybit format
            bybit_symbol = symbol.replace("-", "").upper()
            ws_url = self.exchange_urls["bybit"]

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                # Subscribe to tickers
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"tickers.{bybit_symbol}"],
                }
                await websocket.send(json.dumps(subscribe_msg))

                self.logger.info(f"Connected to Bybit for {symbol}")

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)

                        if data.get("topic") == f"tickers.{bybit_symbol}":
                            ticker = data.get("data", {})
                            if "lastPrice" in ticker:
                                price = float(ticker["lastPrice"])
                                update = PriceUpdate(
                                    symbol=symbol,
                                    price=price,
                                    timestamp=datetime.now(),
                                    exchange="bybit",
                                    volume=float(ticker.get("turnover24h", 0)),
                                )

                                await self._handle_price_update(update)

                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Bybit timeout for {symbol}, reconnecting..."
                        )
                        break
                    except Exception as e:
                        self.logger.error(f"Error receiving Bybit data: {e}")
                        break

        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            return None

    def _convert_to_kraken_symbol(self, symbol: str) -> str:
        """Convert symbol to Kraken format.

        Args:
            symbol: Standard symbol (e.g., "BTC-USD")

        Returns:
            Kraken symbol (e.g., "XBT/USD")
        """
        kraken_aliases = {
            "BTC-USD": "XBT/USD",
            "ETH-USD": "ETH/USD",
            "BNB-USD": "BNB/USD",
            "SOL-USD": "SOL/USD",
            "XRP-USD": "XRP/USD",
            "ADA-USD": "ADA/USD",
            "AVAX-USD": "AVAX/USD",
            "DOGE-USD": "DOGE/USD",
            "DOT-USD": "DOT/USD",
            "LINK-USD": "LINK/USD",
        }
        return kraken_aliases.get(symbol, symbol.replace("-", "/"))

    async def _handle_price_update(self, update: PriceUpdate):
        """Handle incoming price update.

        Args:
            update: Price update object
        """
        # Store latest price
        if update.symbol not in self.latest_prices:
            self.latest_prices[update.symbol] = {}

        self.latest_prices[update.symbol][update.exchange] = update

        # Call callback if provided
        if self.on_price_update:
            try:
                if asyncio.iscoroutinefunction(self.on_price_update):
                    await self.on_price_update(update)
                else:
                    self.on_price_update(update)
            except Exception as e:
                self.logger.error(
                    f"Error in price update callback for {update.symbol}: {e}"
                )
                # Don't let callback errors break the connection
                return

    async def start(self):
        """Start WebSocket connections to all configured exchanges."""
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websockets library not available")

        self.running = True
        self.logger.info(f"Starting WebSocket price feed for {self.symbols}")

        # Create tasks for each exchange-symbol combination
        tasks = []
        for exchange in self.exchanges:
            for symbol in self.symbols:
                if exchange == "coinbase":
                    tasks.append(self._connect_coinbase(symbol))
                elif exchange == "binance":
                    tasks.append(self._connect_binance(symbol))
                elif exchange == "kraken":
                    tasks.append(self._connect_kraken(symbol))
                elif exchange == "bybit":
                    tasks.append(self._connect_bybit(symbol))

        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop WebSocket connections."""
        self.running = False
        self.logger.info("Stopping WebSocket price feed")

    def get_latest_price(
        self, symbol: str, exchange: Optional[str] = None
    ) -> Optional[PriceUpdate]:
        """Get latest price for a symbol.

        Args:
            symbol: Trading pair symbol
            exchange: Specific exchange (optional)

        Returns:
            PriceUpdate object or None if not available
        """
        if symbol not in self.latest_prices:
            return None

        if exchange:
            return self.latest_prices[symbol].get(exchange)

        # Return average price across all exchanges
        prices = [update.price for update in self.latest_prices[symbol].values()]
        if not prices:
            return None

        avg_price = sum(prices) / len(prices)
        return PriceUpdate(
            symbol=symbol,
            price=avg_price,
            timestamp=datetime.now(),
            exchange="average",
        )

    def get_price_history(
        self, symbol: str, max_points: int = 100
    ) -> List[PriceUpdate]:
        """Get price history for analysis.

        Args:
            symbol: Trading pair symbol
            max_points: Maximum number of data points to return

        Returns:
            List of PriceUpdate objects
        """
        history = []

        # Collect prices from all exchanges
        if symbol in self.latest_prices:
            for exchange, update in self.latest_prices[symbol].items():
                history.append(update)

        # Sort by timestamp and limit
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history[:max_points]


class ArbitrageDetector:
    """Detect arbitrage opportunities across exchanges."""

    def __init__(self, price_feed: WebSocketPriceFeed, min_spread: float = 0.005):
        """Initialize arbitrage detector.

        Args:
            price_feed: WebSocket price feed instance
            min_spread: Minimum spread percentage to trigger alert (default 0.5%)
        """
        self.price_feed = price_feed
        self.min_spread = min_spread
        self.logger = logging.getLogger(__name__)
        self.opportunities: List[Dict] = []

    def check_arbitrage(self, update: PriceUpdate):
        """Check for arbitrage opportunities when price updates.

        Args:
            update: Latest price update
        """
        symbol = update.symbol
        latest_prices = self.price_feed.get_latest_price(symbol)

        if not latest_prices or symbol not in self.price_feed.latest_prices:
            return

        prices = self.price_feed.latest_prices[symbol]

        # Find min and max prices across exchanges
        if len(prices) < 2:
            return

        price_list = [(exchange, update.price) for exchange, update in prices.items()]
        price_list.sort(key=lambda x: x[1])

        min_exchange, min_price = price_list[0]
        max_exchange, max_price = price_list[-1]

        # Calculate spread percentage
        spread_pct = (max_price - min_price) / min_price * 100

        if spread_pct >= self.min_spread * 100:
            opportunity = {
                "symbol": symbol,
                "min_exchange": min_exchange,
                "max_exchange": max_exchange,
                "min_price": min_price,
                "max_price": max_price,
                "spread_pct": spread_pct,
                "timestamp": datetime.now(),
            }

            self.opportunities.append(opportunity)
            self.logger.info(
                f"Arbitrage opportunity: {symbol} buy on {min_exchange} at ${min_price:.2f}, "
                f"sell on {max_exchange} at ${max_price:.2f} (spread: {spread_pct:.2f}%)"
            )

    def get_recent_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get recent arbitrage opportunities.

        Args:
            limit: Maximum number of opportunities to return

        Returns:
            List of opportunity dictionaries
        """
        return self.opportunities[-limit:]


async def run_websocket_feed(
    symbols: List[str],
    exchanges: List[str] = None,
    on_price_update: Callable[[PriceUpdate], None] = None,
    enable_arbitrage: bool = True,
):
    """Run WebSocket feed with optional arbitrage detection.

    Args:
        symbols: List of symbols to track
        exchanges: List of exchanges to connect to
        on_price_update: Callback for price updates
        enable_arbitrage: Enable arbitrage detection
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create price feed
    price_feed = WebSocketPriceFeed(
        symbols=symbols,
        exchanges=exchanges,
        on_price_update=on_price_update,
    )

    # Setup arbitrage detector if enabled
    arb_detector = None
    if enable_arbitrage and on_price_update:
        arb_detector = ArbitrageDetector(price_feed)

        # Wrap original callback
        def wrapped_callback(update: PriceUpdate):
            on_price_update(update)
            arb_detector.check_arbitrage(update)

        price_feed.on_price_update = wrapped_callback

    try:
        # Start feed
        await price_feed.start()
    except KeyboardInterrupt:
        await price_feed.stop()
        print("\nWebSocket feed stopped")


if __name__ == "__main__":
    # Example usage
    async def price_callback(update: PriceUpdate):
        print(f"[{update.exchange}] {update.symbol}: ${update.price:.2f}")

    # Run for BTC-USD and ETH-USD
    asyncio.run(
        run_websocket_feed(
            symbols=["BTC-USD", "ETH-USD"],
            exchanges=["coinbase", "binance"],
            on_price_update=price_callback,
            enable_arbitrage=True,
        )
    )
