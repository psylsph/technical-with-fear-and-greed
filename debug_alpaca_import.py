
import sys
import traceback

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    print("Attempting to import alpaca.trading.client...")
    from alpaca.trading.client import TradingClient
    print("  SUCCESS")
except ImportError:
    print("  FAILURE")
    traceback.print_exc()
except Exception:
    print("  FAILURE (Unknown error)")
    traceback.print_exc()

try:
    print("Attempting to import alpaca.trading.enums...")
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
    print("  SUCCESS")
except ImportError:
    print("  FAILURE")
    traceback.print_exc()

try:
    print("Attempting to import alpaca.trading.requests...")
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
    from alpaca.trading.requests import StopLimitOrderRequest
    print("  SUCCESS")
except ImportError:
    print("  FAILURE")
    traceback.print_exc()

try:
    print("Attempting to import OrderStatus from alpaca.trading.enums...")
    from alpaca.trading.enums import OrderStatus
    print("  SUCCESS: Found in alpaca.trading.enums")
except ImportError:
    print("  FAILURE: Not in alpaca.trading.enums")

try:
    print("Attempting to import OrderStatus from alpaca.common.enums...")
    from alpaca.common.enums import OrderStatus
    print("  SUCCESS: Found in alpaca.common.enums")
except ImportError:
    print("  FAILURE: Not in alpaca.common.enums")
