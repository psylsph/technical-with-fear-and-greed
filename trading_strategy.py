import json
import os
import time
import urllib.request
from datetime import datetime
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt

from dotenv import load_dotenv

import pandas as pd
import vectorbt as vbt
import warnings
import sklearn.ensemble as skl
import os
import requests
import yfinance as yf
import argparse
# Alpaca trading imports (for live trading functionality)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Note: alpaca-py not installed. Live trading features disabled.")

warnings.filterwarnings("ignore")

load_dotenv()

parser = argparse.ArgumentParser(description='Fear & Greed Trading Strategy')
parser.add_argument('--live', action='store_true', help='Run live trading instead of backtesting')
args = parser.parse_args()

INITIAL_CAPITAL = 1000

MAKER_FEE = 0.0015
TAKER_FEE = 0.0025

pred_series = None  # Global ML predictions

GRANULARITY_TO_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
    "FOUR_HOUR": 14400,
    "ONE_DAY": 86400,
}

GRANULARITY_TO_FREQ = {
    "ONE_MINUTE": "1m",
    "FIVE_MINUTE": "5m",
    "FIFTEEN_MINUTE": "15m",
    "FOUR_HOUR": "4h",
    "ONE_HOUR": "1h",
    "ONE_DAY": "1d",
}


def load_cdp_credentials() -> tuple[str, str] | None:
    """Load CDP API credentials from JSON file."""
    cdp_path = os.path.join(os.path.dirname(__file__), "cdp_api_key.json")
    if not os.path.exists(cdp_path):
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
    """Fetch historical candles using Coinbase Advanced Trade API with caching."""
    import hashlib

    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = hashlib.md5(
        f"{product_id}_{start}_{end}_{granularity}".encode()
    ).hexdigest()
    cache_file = os.path.join(cache_dir, f"coinbase_{cache_key}.csv")

    if use_cache and os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        file_age_hours = file_age / 3600

        if file_age_hours < cache_hours:
            print(f"Loading cached data ({file_age_hours:.1f} hours old)")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        else:
            print(f"Cache expired ({file_age_hours:.1f} hours old)")

    creds = load_cdp_credentials()
    if not creds:
        return None

    api_key_id, private_key = creds

    try:
        from coinbase.rest import RESTClient

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

        df.to_csv(cache_file)
        print(
            f"Total: {len(df)} candles from Coinbase in {chunk_count} chunks (cached to {cache_file})"
        )
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
        records.append(
            {
                "date": date,
                "fgi_value": int(item["value"]),
                "fgi_classification": item["value_classification"],
            }
        )

    fgi_df = pd.DataFrame(records)
    fgi_df.set_index("date", inplace=True)
    fgi_df.index = pd.to_datetime(fgi_df.index).tz_localize('UTC')
    return fgi_df.sort_index()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI manually."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD manually."""
    fast_ema = close.ewm(span=fast).mean()
    slow_ema = close.ewm(span=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


def prepare_ml_data(close: pd.Series, fgi_df: pd.DataFrame, rsi: pd.Series, volume: pd.Series = None):
    """Prepare dataset for ML training."""
    if volume is None:
        volume = close * 0.01  # dummy
    df = pd.DataFrame({
        'fgi': fgi_df['fgi_value'],
        'close': close,
        'rsi': rsi,
        'volume': volume,
        'fgi_lag1': fgi_df['fgi_value'].shift(1)
    }).dropna()
    df['target'] = (df['fgi'].shift(-1) > df['fgi']).astype(int)  # 1 if next FGI up
    return df


def get_current_fgi():
    """Fetch current Fear & Greed Index."""
    url = "https://api.alternative.me/fng/?limit=1"
    response = requests.get(url)
    data = response.json()
    return int(data['data'][0]['value'])


def get_current_price(symbol):
    """Fetch current price using yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1d')
    return hist['Close'].iloc[-1] if not hist.empty else None


def execute_trade(symbol, side, qty, trading_client=None):
    """Execute trade via Alpaca."""
    if not ALPACA_AVAILABLE or trading_client is None:
        print(f"Trade execution disabled (Alpaca not configured): {side} {qty} {symbol}")
        return None
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force=TimeInForce.DAY
    )
    try:
        order = trading_client.submit_order(order_data)
        print(f"Order submitted: {order.id}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None


def run_strategy(
    close: pd.Series, freq: str, fgi_df: pd.DataFrame, granularity_name: str,
    rsi_window: int = 14, trail_pct: float = 0.10, buy_quantile: float = 0.2,
    sell_quantile: float = 0.8, ml_thresh: float = 0.5
) -> dict:
    """Run fear & greed strategy with RSI filter, dynamic FGI thresholds, trailing stops, and multi-TF analysis, return performance metrics."""
    entries = pd.DataFrame.vbt.signals.empty_like(close).to_frame()
    exits = pd.DataFrame.vbt.signals.empty_like(close).to_frame()

    # Calculate indicators
    rsi = calculate_rsi(close, window=rsi_window)
    macd, signal = calculate_macd(close)

    # Dynamic FGI thresholds
    buy_thresh = fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(buy_quantile)
    sell_thresh = fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(sell_quantile)

    # Multi-TF: For sub-daily, check daily RSI (removed, using ML instead)



    in_position = False
    position_price = 0.0
    trailing_stop = 0.0
    take_profit_pct = 0.25

    for i in range(len(close)):
        price = close.iloc[i]
        dt = close.index[i]

        dt_ts = pd.Timestamp(dt)
        dt_date_only = dt_ts.normalize()

        if dt_date_only not in fgi_df.index:
            continue

        fgi_val = fgi_df.loc[dt_date_only, "fgi_value"]
        rsi_val = rsi.iloc[i] if pd.notna(rsi.iloc[i]) else 50.0
        buy_thresh_val = buy_thresh.loc[dt_date_only]
        sell_thresh_val = sell_thresh.loc[dt_date_only]
        pred_val = pred_series.loc[dt_date_only] if dt_date_only in pred_series.index else 0.5
        is_buy = (fgi_val <= buy_thresh_val and rsi_val < 30) and (pred_val > ml_thresh)
        is_extreme_greed = fgi_val >= sell_thresh_val
        is_overbought = rsi_val > 70

        if not in_position and is_buy:
            entries.iloc[i, 0] = True
            in_position = True
            position_price = price
            trailing_stop = price * (1 - trail_pct)

        if in_position:
            # Update trailing stop
            trailing_stop = max(trailing_stop, price * (1 - trail_pct))
            pnl_pct = (price - position_price) / position_price

            if (
                is_extreme_greed
                or is_overbought
                or pnl_pct >= take_profit_pct
                or price <= trailing_stop
            ):
                exits.iloc[i, 0] = True
                in_position = False

    if entries.sum().sum() == 0:
        return {
            "granularity": granularity_name,
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "outperformance": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
        }

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

    return {
        "granularity": granularity_name,
        "total_return": stats["Total Return [%]"],
        "benchmark_return": benchmark_return,
        "outperformance": stats["Total Return [%]"] - benchmark_return,
        "max_drawdown": stats["Max Drawdown [%]"],
        "total_trades": int(stats["Total Trades"]),
        "win_rate": stats["Win Rate [%]"],
        "sharpe_ratio": stats["Sharpe Ratio"],
    }


def fetch_yahoo_data(symbol: str, start: str, end: str, interval: str) -> pd.Series:
    """Fetch data from Yahoo Finance."""
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

    return close_series


START_DATE = "2024-01-01"
END_DATE = "2025-01-01"

print("=" * 60)
print("BTC Fear & Greed Index Strategy - Multiple Timeframes")
print("=" * 60)

fgi_df = fetch_fear_greed_index()

fgi_values_full = fgi_df["fgi_value"]
print(f"\nFGI Range: {fgi_values_full.min()} - {fgi_values_full.max()}")
print(f"Average FGI: {fgi_values_full.mean():.1f}")
print(f"Fear days (FGI<=35): {(fgi_values_full <= 35).sum()}")
print(f"Extreme Greed days (FGI>80): {(fgi_values_full > 80).sum()}")

# ML Training on daily data
print("\nTraining ML Model...")
daily_close = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, "1d")
daily_rsi = calculate_rsi(daily_close)
volume = daily_close * 0.01  # dummy volume
ml_df = prepare_ml_data(daily_close, fgi_df, daily_rsi, volume)
features = ml_df[['fgi', 'close', 'rsi', 'volume', 'fgi_lag1']]
target = ml_df['target']
model = skl.RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, target)
pred_proba = model.predict_proba(features)[:, 1]
pred_series = pd.Series(pred_proba, index=ml_df.index)
print("ML Model trained.")

GRANULARITIES_TO_TEST = [
    "ONE_DAY",
    "ONE_HOUR",
]

if not args.live:
    results = []

    for granularity in GRANULARITIES_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"Testing {granularity} ({GRANULARITY_TO_FREQ[granularity]})")
        print(f"{'=' * 60}")

        close = None
        freq = GRANULARITY_TO_FREQ[granularity]
        data_source = "Unknown"

        yf_interval = GRANULARITY_TO_FREQ[granularity]
        try:
            try:
                with open('cdp_api_key.json', 'r') as f:
                    cdp_keys = json.load(f)
                coinbase_api_key = cdp_keys['name']
                coinbase_secret = cdp_keys['privateKey']
                os.environ['COINBASE_API_KEY'] = coinbase_api_key
                os.environ['COINBASE_SECRET_KEY'] = coinbase_secret
                print("Using Coinbase...")
                close = fetch_coinbase_historical("BTC-USD", START_DATE + "T00:00:00Z", END_DATE + "T00:00:00Z", granularity.upper())
                if isinstance(close, pd.DataFrame):
                    close = close['close']
                data_source = "Coinbase"
            except Exception as e:
                print(f"Coinbase setup error: {e}, using Yahoo Finance...")
                close = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, yf_interval)
                data_source = "Yahoo Finance"
        except Exception as e:
            print(f"Data fetch error: {e}")
            continue

        if close is None or len(close) < 10:
            print(f"Insufficient data for {granularity}")
            continue

        print(f"Data source: {data_source}")
        print(f"Total bars: {len(close)}")

        result = run_strategy(close, freq, fgi_df, granularity)
        results.append(result)

print(f"\n{'=' * 80}")
print("SUMMARY: Performance by Timeframe")
print(f"{'=' * 80}")
print(
    f"{'Granularity':<15} {'Return %':<12} {'Benchmark %':<14} {'Outper %':<12} {'Win Rate %':<12} {'Trades':<8}"
)
print(f"{'-' * 80}")

for result in results:
    print(
        f"{result['granularity']:<15} "
        f"{result['total_return']:>10.2f}% "
        f"{result['benchmark_return']:>12.2f}% "
        f"{result['outperformance']:>10.2f}% "
        f"{result['win_rate']:>10.1f}% "
        f"{result['total_trades']:>6}"
    )

best_return = max(results, key=lambda x: x["total_return"])
best_sharpe = max(results, key=lambda x: x["sharpe_ratio"])

print(f"{'-' * 80}")
print(
    f"Best Return: {best_return['granularity']} with {best_return['total_return']:.2f}%"
)
    print(
        f"Best Sharpe: {best_sharpe['granularity']} with {best_sharpe['sharpe_ratio']:.2f}"
    )
    print(f"{'=' * 80}")

    # Parameter optimization for ONE_DAY (long data available)
    print("\nOptimizing parameters for ONE_DAY...")
    close_oned = fetch_yahoo_data("BTC-USD", START_DATE, END_DATE, '1d')
    if close_oned is not None and len(close_oned) > 10:
        combos = [
            (14, 0.05, 0.2, 0.8, 0.4),  # tighter trail, lower ML
            (14, 0.05, 0.2, 0.8, 0.6),  # tighter trail, higher ML
            (14, 0.15, 0.2, 0.8, 0.4),  # looser trail, lower ML
            (14, 0.15, 0.2, 0.8, 0.6),  # looser trail, higher ML
        ]
        best_ret = -float('inf')
        best_combo = None
        for rsi, trail, buy_q, sell_q, ml_t in combos:
            result = run_strategy(close_oned, '1d', fgi_df, 'ONE_DAY', rsi, trail, buy_q, sell_q, ml_t)
            ret = result['total_return']
            print(f"Combo RSI{rsi} Trail{trail} BuyQ{buy_q} SellQ{sell_q} ML{ml_t}: Return {ret:.2f}%, Win {result['win_rate']:.1f}%, Trades {result['total_trades']}")
            if ret > best_ret:
                best_ret = ret
                best_combo = (rsi, trail, buy_q, sell_q, ml_t)
        print(f"Best combo: {best_combo} with {best_ret:.2f}% return")
    else:
        print("ONE_DAY data insufficient for optimization")

if args.live:
    # Live Trading Setup (Paper Mode)
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    if api_key and secret_key and ALPACA_AVAILABLE:
        trading_client = TradingClient(api_key, secret_key, paper=True)
        account = trading_client.get_account()
        print("\nPaper Trading Setup: Connected to Alpaca")
        print(f"Account Cash: ${account.cash}")
        print(f"Account Equity: ${account.equity}")

        # Get current position
        positions = trading_client.get_all_positions()
        btc_position = next((p for p in positions if p.symbol == 'BTC/USD'), None)
        print(f"Current BTC position: {btc_position.qty if btc_position else 0}")

        # Get current signal
        current_fgi = get_current_fgi()
        current_close = get_current_price('BTC-USD')
        if current_close is None:
            print("Could not fetch current price")
        else:
            current_rsi = calculate_rsi(pd.Series([current_close]), 14).iloc[-1]
            pred_val = pred_series.loc[pd.Timestamp.now().normalize()] if pred_series is not None and pd.Timestamp.now().normalize() in pred_series.index else 0.5
            is_buy = current_fgi <= 20 and current_rsi < 30 and pred_val > 0.6
            is_sell = current_fgi >= 80 or current_rsi > 70

            print(f"Current FGI: {current_fgi}, RSI: {current_rsi:.2f}, Pred: {pred_val:.2f}, Signal: {'BUY' if is_buy else 'SELL' if is_sell else 'HOLD'}")

            if is_buy and (btc_position is None or float(btc_position.qty) == 0):
                execute_trade('BTC/USD', OrderSide.BUY, 0.001)
            elif is_sell and btc_position and float(btc_position.qty) > 0:
                execute_trade('BTC/USD', OrderSide.SELL, float(btc_position.qty))
    elif not ALPACA_AVAILABLE:
        print("\nPaper Trading: alpaca-py not installed")
    else:
        print("\nPaper Trading: ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
else:
    print("\nBacktest complete. Use --live for live trading.")
#print("\nLive Trading: Code implemented for Alpaca paper trading. Requires env vars and correct imports.")
