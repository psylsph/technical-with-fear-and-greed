"""
Data fetching utilities with SQLite caching for market data and external APIs.
"""

import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import vectorbt as vbt
import yfinance as yf

from ..config import CACHE_DIR, CDP_KEY_FILE, GRANULARITY_TO_SECONDS


def init_database():
    """Initialize SQLite database for data caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    db_path = os.path.join(CACHE_DIR, "market_data.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            symbol TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            granularity TEXT,
            PRIMARY KEY (symbol, timestamp, granularity)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fgi_data (
            date TEXT PRIMARY KEY,
            fgi_value INTEGER,
            fgi_classification TEXT
        )
    """
    )

    conn.commit()
    conn.close()
    return db_path


def get_cached_data(
    symbol: str, start_date: str, end_date: str, granularity: str, source: str
) -> Optional[pd.DataFrame]:
    """Get cached data from SQLite database."""
    db_path = init_database()
    conn = sqlite3.connect(db_path)

    try:
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? AND granularity = ? AND source = ?
            ORDER BY timestamp
        """

        df = pd.read_sql_query(
            query, conn, params=[symbol, start_ts, end_ts, granularity, source]
        )

        if len(df) == 0:
            return None

        # Convert timestamp back to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.rename(columns={"close": "close"})

        return df

    except Exception as e:
        print(f"Error reading cached data: {e}")
        return None
    finally:
        conn.close()


def save_cached_data(symbol: str, df: pd.DataFrame, granularity: str, source: str):
    """Save data to SQLite cache."""
    if df is None or len(df) == 0:
        return

    db_path = init_database()
    conn = sqlite3.connect(db_path)

    try:
        # Prepare data for insertion
        data_to_insert = []
        for timestamp, row in df.iterrows():
            ts = int(timestamp.timestamp())
            data_to_insert.append(
                (
                    symbol,
                    ts,
                    float(row["open"]) if "open" in row else float(row["close"]),
                    float(row["high"]) if "high" in row else float(row["close"]),
                    float(row["low"]) if "low" in row else float(row["close"]),
                    float(row["close"]),
                    float(row["volume"]) if "volume" in row else 0.0,
                    source,
                    granularity,
                )
            )

        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_data
            (symbol, timestamp, open, high, low, close, volume, source, granularity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data_to_insert,
        )

        conn.commit()
        print(
            f"Cached {len(data_to_insert)} records for {symbol} ({granularity}) from {source}"
        )

    except Exception as e:
        print(f"Error saving cached data: {e}")
    finally:
        conn.close()


def get_cached_fgi(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Get cached FGI data from SQLite database."""
    db_path = init_database()
    conn = sqlite3.connect(db_path)

    try:
        query = """
            SELECT date, fgi_value, fgi_classification
            FROM fgi_data
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=[start_date, end_date])

        if len(df) == 0:
            return None

        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        return df

    except Exception as e:
        print(f"Error reading cached FGI data: {e}")
        return None
    finally:
        conn.close()


def save_cached_fgi(fgi_df: pd.DataFrame):
    """Save FGI data to SQLite cache."""
    if fgi_df is None or len(fgi_df) == 0:
        return

    db_path = init_database()
    conn = sqlite3.connect(db_path)

    try:
        data_to_insert = []
        for date, row in fgi_df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            data_to_insert.append(
                (date_str, int(row["fgi_value"]), row["fgi_classification"])
            )

        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO fgi_data
            (date, fgi_value, fgi_classification)
            VALUES (?, ?, ?)
        """,
            data_to_insert,
        )

        conn.commit()
        print(f"Cached {len(data_to_insert)} FGI records")

    except Exception as e:
        print(f"Error saving cached FGI data: {e}")
    finally:
        conn.close()


def load_cdp_credentials() -> tuple[str, str] | None:
    """Load CDP API credentials from JSON file."""
    if not os.path.exists(CDP_KEY_FILE):
        return None

    try:
        with open(CDP_KEY_FILE) as f:
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
    import jwt
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    private_key_obj = serialization.load_pem_private_key(
        private_key.encode(), password=None, backend=default_backend()
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
        headers={"kid": api_key_id, "nonce": nonce},
    )

    return token


def fetch_coinbase_historical(
    product_id: str = "BTC-USD",
    start: str = "2025-01-01T00:00:00Z",
    end: str = "2026-01-04T00:00:00Z",
    granularity: str = "FIFTEEN_MINUTE",
    use_cache: bool = True,
    cache_hours: int = 24,
) -> pd.DataFrame | None:
    """Fetch historical candles using Coinbase Advanced Trade API with SQLite caching."""
    # Check cache first
    if use_cache:
        cached_data = get_cached_data(product_id, start, end, granularity, "coinbase")
        if cached_data is not None:
            print(f"Using cached Coinbase data ({len(cached_data)} bars)")
            return cached_data

    creds = load_cdp_credentials()
    if not creds:
        return None

    api_key_id, private_key = creds

    try:
        # Import Coinbase client with error handling
        try:
            from coinbase.rest import RESTClient
        except ImportError as ie:
            print(f"Coinbase import error: {ie}")
            return None

        client = RESTClient(api_key=api_key_id, api_secret=private_key)

        start_ts = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp())
        end_ts = int(datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp())

        all_candles = []
        max_candles = 349
        chunk_count = 0

        granularity_seconds = GRANULARITY_TO_SECONDS[granularity]

        current_start = end_ts - (max_candles * granularity_seconds)

        while current_start >= start_ts:
            chunk_count += 1

            interval_end = min(
                current_start + (max_candles * granularity_seconds), end_ts
            )

            candles = client.get_candles(
                product_id=product_id,
                start=str(current_start),
                end=str(interval_end),
                granularity=granularity,
            )

            if not candles or not candles.candles:
                print(f"No candles in response for chunk {chunk_count}")
                break

            chunk_size = len(candles.candles)

            for candle in candles.candles:
                all_candles.append(
                    {
                        "time": int(candle.start),
                        "low": float(candle.low),
                        "high": float(candle.high),
                        "open": float(candle.open),
                        "close": float(candle.close),
                        "volume": float(candle.volume),
                    }
                )

            print(
                f"Chunk {chunk_count}: fetched {chunk_size} candles (total: {len(all_candles)})"
            )

            if chunk_size < max_candles:
                break

            current_start = current_start - granularity_seconds

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

        # Cache the data
        save_cached_data(product_id, df, granularity, "coinbase")

        print(f"Total: {len(df)} candles from Coinbase in {chunk_count} chunks")
        return df

    except Exception as e:
        print(f"Coinbase API error: {e}")
        import traceback

        traceback.print_exc()
        return None


def fetch_fear_greed_index(limit: int = 730) -> pd.DataFrame:
    """Fetch historical Fear and Greed Index data from alternative.me API with caching."""
    # Check cache first
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=limit)).strftime("%Y-%m-%d")

    cached_data = get_cached_fgi(start_date, end_date)
    if (
        cached_data is not None and len(cached_data) >= limit * 0.9
    ):  # 90% of requested data
        print(f"Using cached FGI data ({len(cached_data)} days)")
        return cached_data

    url = f"https://api.alternative.me/fng/?limit={limit}&date_format=us"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    records = []
    for item in data["data"]:
        date = pd.to_datetime(item["timestamp"], format="%m-%d-%Y")
        records.append(
            {
                "date": date,
                "fgi_value": int(item["value"]),
                "fgi_classification": item["value_classification"],
            }
        )

    fgi_df = pd.DataFrame(records)
    fgi_df.set_index("date", inplace=True)
    fgi_df.index = pd.to_datetime(fgi_df.index).tz_localize("UTC")

    # Cache the data
    save_cached_fgi(fgi_df)

    return fgi_df.sort_index()


def get_current_fgi() -> int:
    """Fetch current Fear & Greed Index."""
    url = "https://api.alternative.me/fng/?limit=1"
    response = requests.get(url)
    data = response.json()
    return int(data["data"][0]["value"])


def get_current_price(symbol: str) -> Optional[float]:
    """Fetch current price using yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    return hist["Close"].iloc[-1] if not hist.empty else None


def fetch_unified_price_data(
    symbol: str, start: str, end: str, freq: str = "1d"
) -> pd.DataFrame:
    """Fetch price data using Coinbase as primary source, Yahoo as fallback.

    Priority order:
    1. Coinbase (cached or fresh fetch)
    2. Yahoo (only if Coinbase unavailable for requested dates)

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        freq: Frequency string (e.g., "1d", "1h", "4h")

    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    import numpy as np

    # Map freq to granularity for Coinbase
    freq_to_granularity = {
        "1d": "ONE_DAY",
        "1h": "ONE_HOUR",
        "4h": "FOUR_HOUR",
        "15m": "FIFTEEN_MINUTE",
        "5m": "FIVE_MINUTE",
        "1m": "ONE_MINUTE",
    }

    granularity = freq_to_granularity.get(freq, "ONE_DAY")
    product_id = symbol.replace("-", "-")  # BTC-USD format for Coinbase

    # Try Coinbase first
    print(f"Fetching data: {symbol} {start} to {end} ({freq})")
    print("  Trying Coinbase (primary source)...")

    coinbase_data = None

    # Check cache first
    cached_coinbase = get_cached_data(product_id, start, end, granularity, "coinbase")

    if cached_coinbase is not None:
        print(f"  Using cached Coinbase data ({len(cached_coinbase)} bars)")
        coinbase_data = cached_coinbase
    else:
        # Try fresh Coinbase fetch
        coinbase_data = fetch_coinbase_historical(
            product_id=product_id,
            start=start + "T00:00:00Z",
            end=end + "T00:00:00Z",
            granularity=granularity,
            use_cache=True,
        )

    if coinbase_data is not None and len(coinbase_data) > 0:
        # Check if Coinbase data covers the requested range
        date_range_start = pd.Timestamp(start).tz_localize("UTC")
        date_range_end = pd.Timestamp(end).tz_localize("UTC")

        data_start = coinbase_data.index[0]
        data_end = coinbase_data.index[-1]

        # Allow small gap (within 5% of requested range)
        requested_days = (date_range_end - date_range_start).days
        actual_days = (data_end - data_start).days

        coverage_ratio = actual_days / requested_days if requested_days > 0 else 0

        if coverage_ratio >= 0.95:
            print(f"  Coinbase covers {coverage_ratio:.1%} of requested range")
            return coinbase_data[["open", "high", "low", "close", "volume"]]
        else:
            print(f"  Coinbase covers only {coverage_ratio:.1%} of requested range")
            print(f"  Data range: {data_start.date()} to {data_end.date()}")
            print("  Falling back to Yahoo to supplement...")

    # Try Yahoo as fallback/supplement
    print("  Trying Yahoo (supplementary source)...")

    yahoo_data = fetch_yahoo_data(symbol, start, end, freq)

    if yahoo_data is None:
        print("  Yahoo also unavailable")
        # Return whatever Coinbase data we have
        if coinbase_data is not None and len(coinbase_data) > 0:
            return coinbase_data[["open", "high", "low", "close", "volume"]]
        return None

    # Yahoo returns Series, convert to DataFrame
    if isinstance(yahoo_data, pd.Series):
        yahoo_df = pd.DataFrame(
            {
                "open": yahoo_data,
                "high": yahoo_data,
                "low": yahoo_data,
                "close": yahoo_data,
                "volume": np.nan,
            },
            index=yahoo_data.index,
        )
    else:
        yahoo_df = yahoo_data

    # Merge Coinbase and Yahoo if both have data
    if coinbase_data is not None and len(coinbase_data) > 0 and len(yahoo_df) > 0:
        print("  Merging Coinbase and Yahoo data...")

        # Prioritize Coinbase, fill gaps with Yahoo
        merged_data = coinbase_data.copy()

        # Yahoo data not in Coinbase
        yahoo_only = yahoo_df[~yahoo_df.index.isin(merged_data.index)]

        if len(yahoo_only) > 0:
            print(f"    Adding {len(yahoo_only)} bars from Yahoo (not in Coinbase)")
            merged_data = pd.concat([merged_data, yahoo_only]).sort_index()

        # Fill missing volume from Coinbase with Yahoo (or NaN)
        if merged_data["volume"].isna().any():
            merged_data["volume"] = merged_data["volume"].fillna(yahoo_df["volume"])

        print(
            f"  Merged result: {len(merged_data)} bars ({len(coinbase_data)} from Coinbase + {len(yahoo_only)} from Yahoo)"
        )

        # Cache merged data as Coinbase (since it's the primary)
        save_cached_data(product_id, merged_data, granularity, "coinbase")

        return merged_data[["open", "high", "low", "close", "volume"]]

    # Only Yahoo available
    print(f"  Using Yahoo data only ({len(yahoo_df)} bars)")
    # Cache as Coinbase since it's filling in
    save_cached_data(product_id, yahoo_df, granularity, "coinbase")
    return yahoo_df[["open", "high", "low", "close", "volume"]]


def fetch_yahoo_data(symbol: str, start: str, end: str, interval: str) -> pd.Series:
    """Fetch data from Yahoo Finance with SQLite caching and date range handling."""
    # Check cache first
    cached_data = get_cached_data(symbol, start, end, interval, "yahoo")
    if cached_data is not None:
        print(f"Using cached Yahoo data ({len(cached_data)} bars)")
        return cached_data["close"]

    # Handle Yahoo's date limitations
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    now = pd.Timestamp.now()

    # Yahoo limits:
    # - 1h data: max 730 days back
    # - Other intervals: vary

    if interval == "1h":
        max_days_back = 730
    elif interval in ["1d", "1wk", "1mo"]:
        max_days_back = 365 * 10  # Roughly 10 years
    else:
        max_days_back = 365 * 2  # 2 years for other intervals

    days_requested = (end_date - start_date).days
    if days_requested > max_days_back:
        print(f"Yahoo {interval} data limited to {max_days_back} days back")
        adjusted_start = max(start_date, now - pd.Timedelta(days=max_days_back))
        print(
            f"Adjusting start date from {start_date.date()} to {adjusted_start.date()}"
        )
        start = adjusted_start.strftime("%Y-%m-%d")

    try:
        data = vbt.YFData.download(symbol, start=start, end=end, interval=interval)
        close_data = data.get("Close")
        if isinstance(close_data, tuple):
            close_series = close_data[0]
        else:
            close_series = close_data
        if hasattr(close_series, "Close"):
            close_series = close_series.Close

        if close_series is None or len(close_series) == 0:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame format for caching
        df_cache = pd.DataFrame(index=close_series.index)
        df_cache["close"] = close_series
        df_cache["open"] = close_series  # Dummy values
        df_cache["high"] = close_series
        df_cache["low"] = close_series
        df_cache["volume"] = 0.0

        # Cache the data
        save_cached_data(symbol, df_cache, interval, "yahoo")

        return close_series

    except Exception as e:
        print(f"Yahoo Finance error for {symbol}: {e}")
        raise


def resample_higher_tf(
    higher_tf_data: pd.DataFrame | pd.Series, target_freq: str, method: str = "ffill"
) -> pd.DataFrame | pd.Series:
    """Resample higher timeframe data to target frequency.

    Args:
        higher_tf_data: Data at higher timeframe (e.g., daily)
        target_freq: Target frequency string (e.g., '1H', '15min', '30min')
        method: Resampling method ('ffill', 'bfill', 'nearest')

    Returns:
        Resampled data aligned to target frequency
    """
    if isinstance(higher_tf_data, pd.Series):
        # Resample Series
        resampled = higher_tf_data.resample(target_freq).ffill()
    else:
        # Resample DataFrame
        resampled = higher_tf_data.resample(target_freq).ffill()

    return resampled


def calculate_higher_tf_indicators(
    close: pd.Series | pd.DataFrame, granularity: str = "ONE_DAY"
) -> dict:
    """Calculate higher timeframe indicators for trend filtering.

    Args:
        close: Price data at higher timeframe
        granularity: Granularity name from config

    Returns:
        Dictionary with higher timeframe indicators
    """
    from ..indicators import calculate_rsi

    # Convert to Series if DataFrame
    if isinstance(close, pd.DataFrame):
        if "close" in close.columns:
            close = close["close"]
        else:
            close = close.iloc[:, 0]

    # Calculate EMA trend (fast vs slow)
    ema_fast = close.ewm(span=20).mean()
    ema_slow = close.ewm(span=50).mean()
    trend = ema_fast > ema_slow

    # Calculate RSI
    rsi = calculate_rsi(close, window=14)

    return {"trend": trend, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi}


def align_multi_tf_data(
    lower_tf_close: pd.Series,
    higher_tf_indicators: dict,
    lower_tf_freq: str,
) -> pd.DataFrame:
    """Align higher timeframe indicators with lower timeframe data.

    Args:
        lower_tf_close: Price data at lower timeframe
        higher_tf_indicators: Dictionary with higher timeframe indicators
        lower_tf_freq: Lower timeframe frequency string

    Returns:
        Lower TF DataFrame with aligned higher TF indicators
    """
    result = pd.DataFrame(index=lower_tf_close.index, data={"close": lower_tf_close})

    # Resample each higher TF indicator to lower TF frequency
    for key, value in higher_tf_indicators.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            # Resample and forward-fill to cover all lower TF bars
            resampled = value.resample(lower_tf_freq).ffill()

            # Align indices
            resampled = resampled.reindex(lower_tf_close.index, method="ffill")
            result[f"higher_{key}"] = resampled
        else:
            # Constant value, broadcast to all bars
            result[f"higher_{key}"] = value

    return result


def fetch_multi_timeframe_data(
    symbol: str = "BTC-USD",
    start: str = "2024-01-01",
    end: str = "2025-01-01",
    granularities: list = None,
) -> dict:
    """Fetch price data for multiple timeframes simultaneously.

    Args:
        symbol: Trading symbol
        start: Start date
        end: End date
        granularities: List of granularity names (default: ONE_DAY, FOUR_HOUR, ONE_HOUR)

    Returns:
        Dictionary with granularity as key and price data as value
    """
    if granularities is None:
        granularities = ["ONE_DAY", "FOUR_HOUR", "ONE_HOUR"]

    from .config import GRANULARITY_TO_FREQ

    all_data = {}
    for granularity in granularities:
        freq = GRANULARITY_TO_FREQ[granularity]

        try:
            close = fetch_yahoo_data(symbol, start, end, freq)
            all_data[granularity] = close
            print(f"Fetched {granularity} data: {len(close)} bars")
        except Exception as e:
            print(f"Error fetching {granularity}: {e}")
            all_data[granularity] = None

    return all_data


def fetch_live_higher_tf_data(
    symbol: str = "BTC-USD", days: int = 100, higher_tf: str = "ONE_DAY"
) -> dict:
    """Fetch higher timeframe data for live trading signal filtering.

    Args:
        symbol: Trading symbol
        days: Number of days of historical data to fetch
        higher_tf: Higher timeframe granularity (default: ONE_DAY)

    Returns:
        Dictionary with higher_trend (bool) and higher_rsi (float)
    """
    from datetime import datetime, timedelta

    from .config import GRANULARITY_TO_FREQ

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    freq = GRANULARITY_TO_FREQ.get(higher_tf, "1d")

    try:
        higher_tf_close = fetch_yahoo_data(
            symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), freq
        )

        if higher_tf_close is None or len(higher_tf_close) < 50:
            print(f"Insufficient higher timeframe data for {higher_tf}")
            return {"higher_trend": True, "higher_rsi": 50}

        higher_tf_indicators = calculate_higher_tf_indicators(
            higher_tf_close, higher_tf
        )

        higher_trend = bool(higher_tf_indicators["trend"].iloc[-1])
        higher_rsi = higher_tf_indicators["rsi"].iloc[-1]

        return {"higher_trend": higher_trend, "higher_rsi": higher_rsi}

    except Exception as e:
        print(f"Error fetching higher timeframe data: {e}")
        return {"higher_trend": True, "higher_rsi": 50}
