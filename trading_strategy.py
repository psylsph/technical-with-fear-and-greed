import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt

from dotenv import load_dotenv

import pandas as pd
import vectorbt as vbt
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

INITIAL_CAPITAL = 1000

MAKER_FEE = 0.0015
TAKER_FEE = 0.0025


def load_cdp_credentials() -> tuple[str, str] | None:
    """Load CDP API credentials from JSON file."""
    cdp_path = os.path.join(os.path.dirname(__file__), "cdp_api_key.json")
    if not os.path.exists(cdp_path):
        print("CDP credentials file not found")
        return None

    try:
        with open(cdp_path) as f:
            creds = json.load(f)

        private_key = creds.get("privateKey", "")
        private_key = private_key.replace("\\n", "\n")

        name = creds.get("name", "")

        api_key_id = name.split("/")[-1] if "/" in name else ""

        return api_key_id, private_key
    except Exception as e:
        print(f"Error loading CDP credentials: {e}")
        return None


def generate_jwt(api_key_id: str, private_key: str, uri: str | None = None) -> str:
    """Generate JWT token for Coinbase API authentication."""
    private_key_obj = serialization.load_pem_private_key(
        private_key.encode(),
        password=None,
        backend=default_backend()
    )

    payload = {
        "sub": api_key_id,
        "iss": "cdp",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
    }

    if uri:
        payload["uri"] = uri

    nonce = os.urandom(8).hex()
    
    token = jwt.encode(
        payload,
        private_key_obj,
        algorithm="ES256",
        headers={"kid": api_key_id, "nonce": nonce}
    )
    
    return token


def fetch_coinbase_historical(
    product_id: str = "BTC-USD",
    start: str = "2025-01-01T00:00:00Z",
    end: str = "2026-01-04T00:00:00Z",
    granularity: str = "FIFTEEN_MINUTE"
) -> pd.DataFrame | None:
    """Fetch historical candles using Coinbase Advanced Trade API."""
    creds = load_cdp_credentials()
    if not creds:
        return None

    api_key_id, private_key = creds

    try:
        from coinbase.rest import RESTClient

        client = RESTClient(
            api_key=api_key_id,
            api_secret=private_key
        )

        start_ts = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp())
        end_ts = int(datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp())

        all_candles = []
        max_candles = 349
        chunk_count = 0

        current_start = end_ts - (max_candles * 900)

        while current_start >= start_ts:
            chunk_count += 1

            interval_end = min(current_start + (max_candles * 900), end_ts)

            candles = client.get_candles(
                product_id=product_id,
                start=str(current_start),
                end=str(interval_end),
                granularity=granularity
            )

            if not candles or not candles.candles:
                print(f"No candles in response for chunk {chunk_count}")
                break

            chunk_size = len(candles.candles)

            for candle in candles.candles:
                all_candles.append({
                    "time": int(candle.start),
                    "low": float(candle.low),
                    "high": float(candle.high),
                    "open": float(candle.open),
                    "close": float(candle.close),
                    "volume": float(candle.volume)
                })

            print(f"Chunk {chunk_count}: fetched {chunk_size} candles (total: {len(all_candles)})")

            if chunk_size < max_candles:
                break

            current_start = current_start - 900

            if chunk_count > 400:
                print("Reached maximum chunk limit")
                break

            time.sleep(0.3)

        if not all_candles:
            print("No candles fetched from Coinbase API")
            return None

        df = pd.DataFrame(all_candles)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df = df.sort_index()

        print(f"Total: {len(df)} candles from Coinbase in {chunk_count} chunks")
        return df

    except Exception as e:
        print(f"Coinbase API error: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_fear_greed_index(limit: int = 730) -> pd.DataFrame:
    """Fetch historical Fear and Greed Index data from alternative.me API."""
    url = f"https://api.alternative.me/fng/?limit={limit}&date_format=us"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    records = []
    for item in data["data"]:
        date = pd.to_datetime(item["timestamp"], format="%m-%d-%Y")
        records.append({
            "date": date,
            "fgi_value": int(item["value"]),
            "fgi_classification": item["value_classification"]
        })

    fgi_df = pd.DataFrame(records)
    fgi_df.set_index("date", inplace=True)
    return fgi_df.sort_index()


START_DATE = "2025-01-01"
END_DATE = "2026-01-04"

print("=== BTC Fear & Greed Index Strategy ===\n")

print("Fetching 15-min data from Coinbase Advanced Trade API...")
coinbase_data = fetch_coinbase_historical(
    product_id="BTC-USD",
    start=f"{START_DATE}T00:00:00Z",
    end=f"{END_DATE}T00:00:00Z",
    granularity="FIFTEEN_MINUTE"
)

close = None
freq = "1h"
data_source = "Unknown"

if coinbase_data is not None and len(coinbase_data) > 5000:
    close = coinbase_data["close"]
    freq = "15m"
    data_source = "Coinbase (15-min)"
else:
    if coinbase_data is not None:
        print(f"Coinbase data insufficient ({len(coinbase_data) if coinbase_data else 0} bars)")
    else:
        print("No Coinbase API access")

    print("Using Yahoo Finance hourly data...")
    data_hourly = vbt.YFData.download("BTC", start=START_DATE, end=END_DATE, interval="1h")
    close_data_hourly = data_hourly.get("Close")
    if isinstance(close_data_hourly, tuple):
        close_hourly = close_data_hourly[0]
    else:
        close_hourly = close_data_hourly
    if hasattr(close_hourly, 'Close'):
        close_hourly = close_hourly.Close
    close = close_hourly
    freq = "1h"
    data_source = "Yahoo Finance (Hourly)"

print(f"\nData source: {data_source}")
print(f"Total bars: {len(close)}")

fgi_df = fetch_fear_greed_index()

fgi_values_full = fgi_df["fgi_value"]
fgi_extreme_fear = fgi_values_full <= 20
fgi_fear = (fgi_values_full > 20) & (fgi_values_full <= 35)
fgi_extreme_greed = fgi_values_full > 80

fgi_buy = fgi_extreme_fear | fgi_fear

print(f"\nFGI Range: {fgi_values_full.min()} - {fgi_values_full.max()}")
print(f"Average FGI: {fgi_values_full.mean():.1f}")
print(f"Fear days (FGI<=35): {(fgi_values_full <= 35).sum()}")
print(f"Extreme Greed days (FGI>80): {(fgi_values_full > 80).sum()}")

entries = pd.DataFrame.vbt.signals.empty_like(close).to_frame()
exits = pd.DataFrame.vbt.signals.empty_like(close).to_frame()

in_position = False
position_price = 0.0
stop_loss_pct = 0.15
take_profit_pct = 0.25

print("\nRunning strategy...")

for i in range(len(close)):
    price = close.iloc[i]
    dt = close.index[i]

    dt_ts = pd.Timestamp(dt)
    dt_date_only = dt_ts.normalize()

    if dt_date_only not in fgi_df.index:
        continue

    fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]
    is_buy = fgi_val <= 35
    is_extreme_greed = fgi_val > 80

    if not in_position and is_buy:
        entries.iloc[i, 0] = True
        in_position = True
        position_price = price

    if in_position:
        pnl_pct = (price - position_price) / position_price

        if is_extreme_greed or pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
            exits.iloc[i, 0] = True
            in_position = False

print(f"Entries: {entries.sum().sum()}, Exits: {exits.sum().sum()}")

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    freq=freq,
    init_cash=INITIAL_CAPITAL,
    fees=(MAKER_FEE, TAKER_FEE),
)

stats = pf.stats()
benchmark_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

print("\n" + "=" * 50)
print("Portfolio Performance")
print("=" * 50)
print(f"Data Source: {data_source}")
print(f"Period: {stats['Start']} to {stats['End']}")
print(f"Total Bars: {len(close)}")
print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
print(f"Final Value: ${INITIAL_CAPITAL * (1 + stats['Total Return [%]']/100):.2f}")
print(f"Total Return: {stats['Total Return [%]']:.2f}%")
print(f"Benchmark (Buy&Hold): {benchmark_return:.2f}%")
print(f"Outperformance: {stats['Total Return [%]'] - benchmark_return:+.2f}%")
print(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
print(f"Total Trades: {int(stats['Total Trades'])}")
print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
print(f"Total Fees Paid: ${stats['Total Fees Paid']:.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
print("=" * 50)
