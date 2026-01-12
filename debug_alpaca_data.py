
import sys
import traceback

try:
    print("Attempting to import CryptoHistoricalDataClient...")
    from alpaca.data.historical import CryptoHistoricalDataClient
    print("  SUCCESS: Found CryptoHistoricalDataClient")
except ImportError:
    print("  FAILURE: CryptoHistoricalDataClient not found")
    traceback.print_exc()

try:
    print("Attempting to import CryptoLatestQuoteRequest...")
    from alpaca.data.requests import CryptoLatestQuoteRequest
    print("  SUCCESS: Found CryptoLatestQuoteRequest")
except ImportError:
    print("  FAILURE: CryptoLatestQuoteRequest not found")
    traceback.print_exc()
