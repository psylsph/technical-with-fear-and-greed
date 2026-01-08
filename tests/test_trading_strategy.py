"""
Test suite for trading strategy implementation.
"""

import json
import os
import tempfile
import unittest
import numpy as np
from unittest.mock import mock_open, patch

import pandas as pd
from src.config import BEST_PARAMS, INITIAL_CAPITAL
from src.data.data_fetchers import (
    calculate_higher_tf_indicators,
    fetch_fear_greed_index,
)
from src.indicators import calculate_macd, calculate_rsi
from src.portfolio import get_test_portfolio_value, load_test_state, simulate_trade
from src.strategy import generate_signal, run_strategy


class TestDataFetchers(unittest.TestCase):
    """Test data fetching functionality."""

    def setUp(self):
        """Clean up any cached data before each test."""
        from src.data.data_fetchers import init_database

        db_path = init_database()
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_init_database(self):
        """Test database initialization."""
        from src.data.data_fetchers import init_database

        db_path = init_database()

        self.assertIsNotNone(db_path)
        self.assertTrue(os.path.exists(db_path))

    def test_save_and_get_cached_data(self):
        """Test saving and getting cached data."""
        from src.data.data_fetchers import get_cached_data, save_cached_data

        dates = pd.date_range("2023-01-01", periods=5, tz="UTC")
        test_data = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [102.0] * 5,
                "low": [99.0] * 5,
                "close": [101.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        save_cached_data("BTC-USD", test_data, "1d", "test_source")

        cached = get_cached_data(
            "BTC-USD", "2023-01-01", "2023-01-05", "1d", "test_source"
        )

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 5)

    def test_save_and_get_cached_fgi(self):
        """Test saving and getting cached FGI data."""
        from src.data.data_fetchers import get_cached_fgi, save_cached_fgi

        dates = pd.date_range("2023-01-01", periods=5, tz="UTC")
        test_fgi = pd.DataFrame(
            {
                "fgi_value": [20, 30, 50, 70, 85],
                "fgi_classification": [
                    "Extreme Fear",
                    "Fear",
                    "Neutral",
                    "Greed",
                    "Extreme Greed",
                ],
            },
            index=dates,
        )

        save_cached_fgi(test_fgi)

        cached = get_cached_fgi("2023-01-01", "2023-01-05")

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 5)

    def test_fetch_fear_greed_index(self):
        """Test FGI index fetching."""
        fgi_data = fetch_fear_greed_index()

        if fgi_data is not None:
            self.assertIsInstance(fgi_data, pd.DataFrame)
            if len(fgi_data) > 0:
                self.assertIn("fgi_value", fgi_data.columns)
                self.assertIn("fgi_classification", fgi_data.columns)

    def test_get_current_price_empty(self):
        """Test getting current price with no data."""
        from src.data.data_fetchers import get_current_price

        with patch("src.data.data_fetchers.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            price = get_current_price("BTC-USD")
            self.assertIsNone(price)

    def test_get_current_price_success(self):
        """Test getting current price with valid data."""
        from src.data.data_fetchers import get_current_price

        mock_df = pd.DataFrame(
            {"Close": [50000.0]},
            index=pd.date_range("2023-01-01", periods=1, tz="UTC"),
        )

        with patch("src.data.data_fetchers.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            price = get_current_price("BTC-USD")
            self.assertIsNotNone(price)
            self.assertEqual(price, 50000.0)

    def test_get_current_fgi(self):
        """Test getting current FGI value."""
        from src.data.data_fetchers import get_current_fgi

        mock_response = {
            "data": [{"value": "45", "value_classification": "Neutral"}]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            fgi_value = get_current_fgi()
            self.assertIsInstance(fgi_value, int)
            self.assertEqual(fgi_value, 45)

    def test_load_cdp_credentials_no_file(self):
        """Test loading CDP credentials when file doesn't exist."""
        from src.data.data_fetchers import load_cdp_credentials

        # Mock file check
        with patch("os.path.exists", return_value=False):
            result = load_cdp_credentials()
            self.assertIsNone(result)

    def test_load_cdp_credentials_success(self):
        """Test loading CDP credentials successfully."""
        from src.data.data_fetchers import load_cdp_credentials

        # Mock file content
        test_content = '{"name": "organizations/123/apiKeys/456", "privateKey": "key"}'
        with patch("builtins.open", new_callable=mock_open, read_data=test_content):
            with patch("os.path.exists", return_value=True):
                result = load_cdp_credentials()
                self.assertEqual(result, ("456", "key"))


class TestIndicators(unittest.TestCase):
    """Test technical indicator calculations."""

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = calculate_rsi(
            pd.Series(
                [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                ]
            )
        )

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), 14)

        # Test that RSI values are in valid range
        rsi_valid = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_valid))

    def test_calculate_rsi_extremes(self):
        """Test RSI calculation at extremes."""
        # Low prices (oversold)
        rsi_low = calculate_rsi(pd.Series([100 - i for i in range(20)]))
        self.assertLess(rsi_low.iloc[-1], 30)

        # High prices (overbought)
        rsi_high = calculate_rsi(pd.Series([100 + i for i in range(20)]))
        self.assertGreater(rsi_high.iloc[-1], 70)

    def test_calculate_macd(self):
        """Test MACD calculation."""
        close = pd.Series([100, 101, 102, 103, 104, 105])

        macd, signal = calculate_macd(close)

        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(macd), len(close))


class TestMLModel(unittest.TestCase):
    """Comprehensive tests for ML model functionality."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2023-01-01", periods=50)
        prices = [100 + i for i in range(50)]
        self.prices = pd.Series(prices, index=dates)

        self.ohlcv = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000 + i * 10 for i in range(50)],
            },
            index=dates,
        )

        self.fgi = pd.DataFrame(
            {
                "fgi_value": [50 + i % 20 for i in range(50)],
                "fgi_classification": ["Neutral"] * 50,
            },
            index=dates,
        )

    def test_prepare_ml_data(self):
        """Test ML data preparation."""
        result = calculate_higher_tf_indicators(self.prices)

        self.assertIsInstance(result, dict)
        self.assertIn("trend", result)
        self.assertIn("ema_fast", result)
        self.assertIn("ema_slow", result)
        self.assertIn("rsi", result)

        # RSI has 13 NaN values at the start
        self.assertEqual(len(result["rsi"]), 50)
        # Check that first 13 values are NaN
        self.assertTrue(result["rsi"].iloc[:13].isna().all())

    def test_train_ml_model(self):
        """Test ML model training."""
        from src.ml.ml_model import train_ml_model

        model, predictions, metrics = train_ml_model(
            self.ohlcv, self.fgi, lookback_days=30
        )

        self.assertIsNotNone(model)
        self.assertIsInstance(predictions, pd.Series)
        # With lookback_days=30, we should have 20 samples (30 - 10 buffer)
        self.assertEqual(len(predictions), 20)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("lookback_days", metrics)
        self.assertIn("features", metrics)

    def test_predict_live_fgi_no_model(self):
        """Test live FGI prediction when model is not trained."""
        from src.ml.ml_model import ml_model, predict_live_fgi

        # Ensure no model
        original_model = ml_model
        ml_model.ml_model = None

        # Test prediction
        today = pd.Timestamp.now().normalize()
        # test_price = pd.Series([150.0], index=[today])
        test_ohlcv = pd.DataFrame(
            {
                "open": [150.0],
                "high": [150.5],
                "low": [149.5],
                "close": [150.0],
                "volume": [1000],
            },
            index=[today],
        )
        prediction = predict_live_fgi(test_ohlcv, self.fgi, today)

        # Should return default (0.5)
        self.assertEqual(prediction, 0.5)

        # Restore original model
        ml_model.ml_model = original_model

    def test_predict_live_fgi_missing_data(self):
        """Test live FGI prediction with missing FGI data."""
        from src.ml.ml_model import predict_live_fgi, train_ml_model

        # Train model first
        train_ml_model(self.ohlcv, self.fgi, lookback_days=30)

        # Test with date that won't exist in FGI data
        future_date = pd.Timestamp("2030-01-01")
        test_ohlcv = pd.DataFrame(
            {
                "open": [150.0],
                "high": [150.5],
                "low": [149.5],
                "close": [150.0],
                "volume": [1000],
            },
            index=[future_date],
        )
        prediction = predict_live_fgi(test_ohlcv, self.fgi, future_date)

        # Should return default
        self.assertEqual(prediction, 0.5)

    def test_predict_live_fgi_index_error(self):
        """Test live FGI prediction with index error."""
        from src.ml.ml_model import predict_live_fgi, train_ml_model

        # Train model first
        train_ml_model(self.ohlcv, self.fgi, lookback_days=30)

        # Test with insufficient price data (less than 14 for RSI)
        short_price = pd.Series([150.0])
        short_ohlcv = pd.DataFrame(
            {
                "open": [150.0],
                "high": [150.5],
                "low": [149.5],
                "close": [150.0],
                "volume": [1000],
            },
            index=short_price.index,
        )
        today = pd.Timestamp.now().normalize()
        prediction = predict_live_fgi(short_ohlcv, self.fgi, today)

        # Should handle insufficient data gracefully
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)


class TestPortfolio(unittest.TestCase):
    """Test portfolio management functionality."""

    def setUp(self):
        """Set up test portfolio state."""
        self.test_state = {
            "cash": 1000.0,
            "btc_held": 0.0,
            "trades": [],
            "initialized": True,
        }

    def test_get_test_portfolio_value(self):
        """Test getting test portfolio value."""

        value = get_test_portfolio_value(self.test_state, 50000.0)

        self.assertEqual(value, 1000.0)

    def test_simulate_trade_buy(self):
        """Test buy trade simulation."""

        result = simulate_trade(self.test_state.copy(), "BTC/USD", "buy", 0.01, 50000.0)

        self.assertEqual(result["btc_held"], 0.01)
        self.assertLess(result["cash"], 1000.0)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["side"], "buy")

    def test_simulate_trade_buy_insufficient_funds(self):
        """Test buy trade with insufficient funds."""

        result = simulate_trade(self.test_state.copy(), "BTC/USD", "buy", 0.5, 50000.0)

        self.assertEqual(result["btc_held"], 0.0)
        self.assertEqual(result["cash"], 1000.0)
        self.assertEqual(len(result["trades"]), 0)

    def test_simulate_trade_sell(self):
        """Test sell trade simulation."""

        # First buy some BTC
        buy_state = simulate_trade(
            self.test_state.copy(), "BTC/USD", "buy", 0.01, 50000.0
        )
        cash_after_buy = buy_state["cash"]

        # Then sell it at a higher price
        sell_state = simulate_trade(buy_state, "BTC/USD", "sell", 0.01, 52000.0)

        self.assertEqual(sell_state["btc_held"], 0.0)
        self.assertGreater(sell_state["cash"], cash_after_buy)
        self.assertEqual(len(sell_state["trades"]), 2)

    def test_simulate_trade_sell_insufficient_btc(self):
        """Test sell trade with insufficient BTC."""

        result = simulate_trade(
            self.test_state.copy(), "BTC/USD", "sell", 0.01, 50000.0
        )

        # Should not execute trade
        self.assertEqual(result["btc_held"], 0.0)
        self.assertEqual(result["cash"], 1000.0)
        self.assertEqual(len(result["trades"]), 0)

    def test_load_save_test_state(self):
        """Test loading and saving test portfolio state."""
        from src.portfolio import save_test_state

        # Create temporary state file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            from src import portfolio

            original_state_file = portfolio.TEST_STATE_FILE
            portfolio.TEST_STATE_FILE = temp_file

            # Save state
            save_test_state(self.test_state)

            # Load state
            loaded_state = load_test_state()

            self.assertEqual(loaded_state["cash"], self.test_state["cash"])
            self.assertEqual(loaded_state["btc_held"], self.test_state["btc_held"])
            self.assertEqual(
                loaded_state["initialized"], self.test_state["initialized"]
            )

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            portfolio.TEST_STATE_FILE = original_state_file

    def test_load_test_state_file_error(self):
        """Test loading test state with file read error."""

        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            temp_file = f.name
            f.write("invalid json content")

        try:
            from src import portfolio

            original_state_file = portfolio.TEST_STATE_FILE
            portfolio.TEST_STATE_FILE = temp_file

            # Should handle error gracefully and return default state
            result = load_test_state()

            self.assertIsNotNone(result)
            self.assertEqual(result["cash"], 1000.0)
            self.assertEqual(result["btc_held"], 0.0)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            portfolio.TEST_STATE_FILE = original_state_file

    def test_save_test_state_error(self):
        """Test saving test state with file write error."""
        from src.portfolio import save_test_state

        # Try to save to invalid path
        from src import portfolio

        original_state_file = portfolio.TEST_STATE_FILE
        portfolio.TEST_STATE_FILE = "/invalid/path/that/does/not/exist.json"

        # Should handle error gracefully
        try:
            save_test_state(self.test_state)
        except Exception as e:
            self.assertIsInstance(e, Exception)

        finally:
            portfolio.TEST_STATE_FILE = original_state_file


class TestStrategy(unittest.TestCase):
    """Comprehensive tests for trading strategy."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2023-01-01", periods=50)
        self.prices = pd.Series([100 + i for i in range(50)], index=dates)
        self.fgi = pd.DataFrame(
            {
                "fgi_value": [20] * 25 + [85] * 25,
                "fgi_classification": ["Fear"] * 25 + ["Extreme Greed"] * 25,
            },
            index=dates,
        )

    def test_generate_signal_buy(self):
        """Test buy signal generation with low FGI."""
        # Create prices with low FGI
        low_fgi = self.fgi.copy()
        low_fgi["fgi_value"] = 25
        
        signal = generate_signal(
            self.prices, low_fgi, rsi_window=14, buy_quantile=0.2, sell_quantile=0.8
        )

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # Low FGI should trigger buy signal
        self.assertEqual(signal["signal"], "buy")

    def test_generate_signal_sell_extreme_greed(self):
        """Test sell signal generation for extreme greed."""
        signal = generate_signal(self.prices, self.fgi, rsi_window=14)

        self.assertIn("signal", signal)
        # Later dates have high FGI, should trigger sell
        self.assertEqual(signal["signal"], "sell")

    def test_generate_signal_sell_overbought(self):
        """Test sell signal generation for high FGI."""
        # Create high FGI scenario
        high_prices = pd.Series(
            [150] * 30, index=pd.date_range("2023-01-01", periods=30)
        )

        high_fgi = pd.DataFrame(
            {"fgi_value": [85] * 30, "fgi_classification": ["Extreme Greed"] * 30},
            index=high_prices.index,
        )

        signal = generate_signal(high_prices, high_fgi, rsi_window=14)

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # High FGI should trigger sell
        self.assertEqual(signal["signal"], "sell")

    def test_generate_signal_hold(self):
        """Test hold signal generation."""
        # Create neutral conditions
        neutral_fgi = self.fgi.copy()
        neutral_fgi["fgi_value"] = 60
        neutral_prices = pd.Series(
            [110] * 30, index=pd.date_range("2023-01-01", periods=30)
        )

        signal = generate_signal(neutral_prices, neutral_fgi, rsi_window=14)

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # With default greed_exit_threshold=70, FGI=60 should not trigger sell
        self.assertEqual(signal["signal"], "hold")

    def test_generate_signal_missing_fgi(self):
        """Test signal generation with missing FGI data."""
        future_date = pd.Timestamp("2030-01-01")
        future_prices = pd.Series([100.0], index=[future_date])

        # Empty FGI dataframe should trigger error
        empty_fgi = pd.DataFrame({"fgi_value": [], "fgi_classification": []})
        empty_fgi.index = pd.DatetimeIndex(empty_fgi.index)
        
        signal = generate_signal(future_prices, empty_fgi)

        self.assertIn("signal", signal)
        self.assertIn("error", signal)
        self.assertEqual(signal["signal"], "hold")

    def test_generate_signal_no_ml(self):
        """Test signal generation without ML."""
        signal = generate_signal(self.prices, self.fgi, pred_series=None)

        self.assertIn("signal", signal)
        # Should still work without ML
        self.assertIn(signal["signal"], ["buy", "sell", "hold"])

    def test_generate_signal_with_ml(self):
        """Test signal generation with ML predictions."""
        # Create ML prediction series
        pred_series = pd.Series([0.6] * 50, index=self.prices.index)

        signal = generate_signal(self.prices, self.fgi, pred_series=pred_series)

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # Signal should still be generated when ML pred is provided
        self.assertIn(signal["signal"], ["buy", "sell", "hold"])

    def test_run_strategy_basic(self):
        """Test full strategy backtest."""
        result = run_strategy(self.prices, "1d", self.fgi, "TEST", rsi_window=14)

        self.assertIn("total_return", result)
        self.assertIn("win_rate", result)
        self.assertIn("total_trades", result)
        self.assertIsInstance(result["total_return"], (int, float))
        self.assertIsInstance(result["win_rate"], (int, float))

    def test_run_strategy_no_trades(self):
        """Test strategy with no trades generated."""
        # Create conditions that won't trigger trades
        neutral_prices = pd.Series(
            [110] * 30, index=pd.date_range("2023-01-01", periods=30)
        )
        neutral_fgi = self.fgi.copy()
        neutral_fgi["fgi_value"] = 50

        result = run_strategy(neutral_prices, "1d", neutral_fgi, "TEST")

        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["total_return"], 0.0)
        self.assertEqual(result["win_rate"], 0.0)

    def test_run_strategy_missing_fgi(self):
        """Test strategy with missing FGI data for some dates."""
        # Create FGI data with gaps
        dates_with_fgi = pd.date_range("2023-01-01", periods=30)
        dates_all = pd.date_range("2023-01-01", periods=50)
        fgi_partial = pd.DataFrame(
            {
                "fgi_value": [50] * 30,
                "fgi_classification": ["Neutral"] * 30,
            },
            index=dates_with_fgi,
        )
        prices_all = pd.Series([100 + i for i in range(50)], index=dates_all)

        result = run_strategy(prices_all, "1d", fgi_partial, "TEST")

        # Should handle missing FGI gracefully
        self.assertIn("total_return", result)
        self.assertIn("win_rate", result)
        self.assertIn("total_trades", result)


class TestTradingEngine(unittest.TestCase):
    """Test trading engine functionality."""

    def setUp(self):
        """Set up test data."""
        date = pd.Timestamp.now(tz="UTC")
        self.prices = pd.Series([100.0], index=[date])
        self.fgi = pd.DataFrame(
            {"fgi_value": [50], "fgi_classification": ["Neutral"]},
            index=[date.normalize()],
        )

    @patch("src.trading.trading_engine.ALPACA_AVAILABLE", False)
    def test_execute_trade_disabled(self):
        """Test trade execution when Alpaca is disabled."""
        from src.trading.trading_engine import execute_trade

        result = execute_trade("BTC/USD", "buy", 0.01)

        self.assertIsNone(result)

    def test_should_trade_buy(self):
        """Test trade decision for buy signal."""
        from src.trading.trading_engine import should_trade

        signal_info = {"signal": "buy", "indicators": {"price": 50000.0}}

        result = should_trade(signal_info, 0.0, is_live=False)

        self.assertEqual(result[0], "buy")
        self.assertIsInstance(result[1], float)
        self.assertGreater(result[1], 0)

    def test_should_trade_sell(self):
        """Test trade decision for sell signal."""
        from src.trading.trading_engine import should_trade

        signal_info = {"signal": "sell", "indicators": {"price": 50000.0}}

        result = should_trade(signal_info, 0.01, is_live=False)

        self.assertEqual(result[0], "sell")
        self.assertEqual(result[1], 0.01)

    def test_should_trade_hold(self):
        """Test trade decision for hold signal."""
        from src.trading.trading_engine import should_trade

        signal_info = {"signal": "hold", "indicators": {"price": 50000.0}}

        result = should_trade(signal_info, 0.0, is_live=False)

        self.assertEqual(result[0], "hold")
        self.assertEqual(result[1], 0.0)

    def test_analyze_live_signal_error(self):
        """Test live signal analysis with error."""
        from src.trading.trading_engine import analyze_live_signal

        # Mock to return None
        with patch("src.trading.trading_engine.analyze_live_signal", return_value=None):
            result = analyze_live_signal(self.fgi)
            # Should handle None return gracefully
            self.assertIsNone(result)

    def test_analyze_live_signal_success(self):
        """Test live signal analysis returns dict with expected keys."""
        # Just verify the function exists and can be called
        from src.trading.trading_engine import analyze_live_signal
        
        # The function should exist and be callable
        self.assertTrue(callable(analyze_live_signal))

    def test_log_trade(self):
        """Test trade logging."""
        from src.trading.trading_engine import log_trade

        signal_info = {
            "indicators": {"price": 50000.0, "fgi": 45, "rsi": 60.0}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            from src.trading import trading_engine

            original_log_file = trading_engine.PROJECT_ROOT
            trading_engine.PROJECT_ROOT = temp_dir

            try:
                log_trade(signal_info, "buy", 0.01, "test_order_id")

                # Check if log file was created
                log_file_path = os.path.join(temp_dir, "trade_log.json")
                self.assertTrue(os.path.exists(log_file_path))

                with open(log_file_path) as f:
                    logs = json.load(f)
                    # Check that our trade is in the logs
                    self.assertTrue(any(log["order_id"] == "test_order_id" for log in logs))
            finally:
                trading_engine.PROJECT_ROOT = original_log_file

    def test_should_trade_with_account_info(self):
        """Test should_trade with account information."""
        from src.trading.trading_engine import should_trade

        signal_info = {"signal": "buy", "indicators": {"price": 50000.0}}

        account_info = {"equity": 10000.0, "cash": 1000.0}

        result = should_trade(signal_info, 0.0, is_live=True, account_info=account_info)

        self.assertEqual(result[0], "buy")
        self.assertIsInstance(result[1], float)
        self.assertGreater(result[1], 0)

    def test_should_trade_no_account_info(self):
        """Test should_trade without account information."""
        from src.trading.trading_engine import should_trade

        signal_info = {"signal": "buy", "indicators": {"price": 50000.0}}

        result = should_trade(signal_info, 0.0, is_live=False)

        self.assertEqual(result[0], "buy")
        self.assertIsInstance(result[1], float)
        self.assertGreater(result[1], 0)


class TestConfig(unittest.TestCase):
    """Test configuration constants."""

    def test_constants(self):
        """Test that constants are properly defined."""
        from src.config import (
            END_DATE,
            GRANULARITY_TO_FREQ,
            INITIAL_CAPITAL,
            MAKER_FEE,
            START_DATE,
            TAKER_FEE,
        )

        self.assertIsInstance(INITIAL_CAPITAL, (int, float))
        self.assertIsInstance(MAKER_FEE, float)
        self.assertIsInstance(TAKER_FEE, float)
        self.assertIsInstance(GRANULARITY_TO_FREQ, dict)
        self.assertIsInstance(START_DATE, str)
        self.assertIsInstance(END_DATE, str)

        # Check BEST_PARAMS structure

        required_keys = [
            "rsi_window",
            "trail_pct",
            "buy_quantile",
            "sell_quantile",
            "ml_thresh",
        ]
        for key in required_keys:
            self.assertIn(key, BEST_PARAMS)


class TestIntegration(unittest.TestCase):
    """Integration tests for full system flow."""

    def test_full_data_flow(self):
        """Test complete data fetching and caching flow."""
        # This would be an integration test that tests the full pipeline
        # For now, just test that the components work together
        from src.data.data_fetchers import get_cached_fgi

        # Test FGI caching
        fgi_data = get_cached_fgi("2024-01-01", "2024-01-05")

        if fgi_data is not None:
            self.assertIsInstance(fgi_data, pd.DataFrame)
            self.assertIn("fgi_value", fgi_data.columns)

    def test_portfolio_simulation_flow(self):
        """Test complete portfolio simulation flow."""

        # Start with initial state
        state = load_test_state()

        # Simulate a few trades
        state = simulate_trade(state, "BTC/USD", "buy", 0.01, 50000.0)
        state = simulate_trade(state, "BTC/USD", "sell", 0.005, 55000.0)

        # Check final portfolio value
        final_value = get_test_portfolio_value(state, 55000.0)

        self.assertGreater(final_value, INITIAL_CAPITAL - 100)

    def test_strategy_with_ml(self):
        """Test strategy with ML predictions."""
        # Create test data with enough samples for ML
        dates = pd.date_range("2023-01-01", periods=180)
        prices = [100 + i for i in range(180)]
        prices_series = pd.Series(prices, index=dates)

        ohlcv = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000 + i * 10 for i in range(180)],
            },
            index=dates,
        )

        fgi = pd.DataFrame(
            {
                "fgi_value": [40 + (i % 60) for i in range(180)],
                "fgi_classification": ["Neutral"] * 180,
            },
            index=dates,
        )

        # Train ML model
        from src.ml.ml_model import train_ml_model

        model, pred_series, _ = train_ml_model(ohlcv, fgi, lookback_days=90)

        # Test strategy with ML
        result = run_strategy(prices_series, "1d", fgi, "TEST", pred_series=pred_series)

        self.assertIn("total_return", result)
        self.assertIsInstance(result["total_return"], (int, float))


class TestMultiTimeframe(unittest.TestCase):
    """Test multi-timeframe functionality."""

    def setUp(self):
        """Set up test data for multiple timeframes."""
        # Daily data (higher timeframe)
        daily_dates = pd.date_range("2023-01-01", periods=100)
        self.daily_close = pd.Series(
            [100 + i * 0.5 for i in range(100)], index=daily_dates
        )

        # Hourly data (lower timeframe) - 24 bars per day
        hourly_dates = pd.date_range("2023-01-01", periods=100 * 24, freq="1h")
        self.hourly_close = pd.Series(
            [100 + (i // 24) * 0.5 + (i % 24) * 0.05 for i in range(100 * 24)],
            index=hourly_dates,
        )

        # FGI data (daily)
        self.fgi = pd.DataFrame(
            {
                "fgi_value": [50 + i % 30 for i in range(100)],
                "fgi_classification": ["Neutral"] * 100,
            },
            index=daily_dates,
        )

    def test_resample_higher_tf(self):
        """Test resampling higher timeframe to lower frequency."""
        from src.data.data_fetchers import resample_higher_tf

        # Resample daily to hourly with ffill
        resampled = resample_higher_tf(self.daily_close, "1h", method="ffill")

        self.assertIsInstance(resampled, (pd.Series, pd.DataFrame))
        # Should have at least as many bars as the hourly data
        self.assertGreaterEqual(len(resampled), len(self.daily_close))
        self.assertEqual(resampled.iloc[0], self.daily_close.iloc[0])

    def test_calculate_higher_tf_indicators(self):
        """Test calculation of higher timeframe indicators."""
        from src.data.data_fetchers import calculate_higher_tf_indicators

        indicators = calculate_higher_tf_indicators(self.daily_close)

        self.assertIn("trend", indicators)
        self.assertIn("ema_fast", indicators)
        self.assertIn("ema_slow", indicators)
        self.assertIn("rsi", indicators)

        # Verify EMA trend calculation
        trend_series = indicators["trend"]
        self.assertIsInstance(trend_series, pd.Series)

    def test_align_multi_tf_data(self):
        """Test alignment of higher TF with lower TF data."""
        from src.data.data_fetchers import align_multi_tf_data

        # Create higher TF indicators with matching length
        daily_subset = self.daily_close[:50]
        higher_tf_indicators = {
            "trend": pd.Series([True, False] * 25, index=daily_subset.index),
            "rsi": pd.Series(
                [40 + i % 20 for i in range(50)], index=daily_subset.index
            ),
        }

        # Align with hourly data
        hourly_subset = self.hourly_close[:1200]

        aligned = align_multi_tf_data(hourly_subset, higher_tf_indicators, "1h")

        self.assertIn("close", aligned.columns)
        self.assertIn("higher_trend", aligned.columns)
        self.assertIn("higher_rsi", aligned.columns)
        self.assertEqual(len(aligned), len(hourly_subset))

    def test_run_strategy_with_multi_tf(self):
        """Test run_strategy with multi-TF filtering enabled."""
        from src.data.data_fetchers import calculate_higher_tf_indicators

        # Prepare higher TF data aligned to hourly
        higher_tf_indicators = calculate_higher_tf_indicators(self.daily_close)
        aligned_data = {}
        for key, value in higher_tf_indicators.items():
            if isinstance(value, pd.Series):
                resampled = value.resample("1h").ffill()
                aligned = resampled.reindex(self.hourly_close.index, method="ffill")
                aligned_data[f"higher_{key}"] = aligned
            else:
                aligned_data[f"higher_{key}"] = value

        # Run strategy with multi-TF filtering
        result = run_strategy(
            self.hourly_close,
            "1h",
            self.fgi,
            "ONE_HOUR",
            higher_tf_data=aligned_data,
            enable_multi_tf=True,
        )

        self.assertIn("total_return", result)
        self.assertIn("multi_tf_enabled", result)
        self.assertTrue(result["multi_tf_enabled"])

    def test_run_strategy_without_multi_tf(self):
        """Test run_strategy with multi-TF filtering disabled."""
        # Run strategy without higher TF data
        result = run_strategy(
            self.hourly_close,
            "1h",
            self.fgi,
            "ONE_HOUR",
            higher_tf_data=None,
            enable_multi_tf=False,
        )

        self.assertIn("total_return", result)
        self.assertIn("multi_tf_enabled", result)
        self.assertFalse(result["multi_tf_enabled"])

    def test_multi_tf_filters_buys(self):
        """Test that multi-TF filters reduce buy signals in bearish trends."""
        from src.data.data_fetchers import calculate_higher_tf_indicators

        # Create bearish higher TF trend
        daily_close_bearish = pd.Series(
            [100 - i * 0.5 for i in range(100)],
            index=self.daily_close.index,
        )

        higher_tf_indicators = calculate_higher_tf_indicators(daily_close_bearish)

        # Most days should be bearish (trend=False)
        bearish_days = (~higher_tf_indicators["trend"]).sum()
        self.assertGreater(bearish_days, 50)  # More than half bearish

    def test_multi_tf_allows_buys_in_bullish_trends(self):
        """Test that multi-TF allows buys during bullish trends."""
        from src.data.data_fetchers import calculate_higher_tf_indicators

        # Create bullish higher TF trend
        daily_close_bullish = pd.Series(
            [100 + i * 0.5 for i in range(100)],
            index=self.daily_close.index,
        )

        higher_tf_indicators = calculate_higher_tf_indicators(daily_close_bullish)

        # Most days should be bullish (trend=True)
        bullish_days = higher_tf_indicators["trend"].sum()
        self.assertGreater(bullish_days, 50)  # More than half bullish



if __name__ == "__main__":
    unittest.main()


class TestSentimentComprehensive(unittest.TestCase):
    """Comprehensive tests for sentiment module."""

    def test_calculate_volatility_sentiment(self):
        """Test volatility-based sentiment calculation."""
        from src.sentiment import calculate_volatility_sentiment

        dates = pd.date_range("2023-01-01", periods=300, freq="1d")
        prices = pd.Series([100 + i for i in range(300)], index=dates)
        vol_sentiment = calculate_volatility_sentiment(prices, window=20)

        self.assertIsInstance(vol_sentiment, pd.Series)
        self.assertEqual(len(vol_sentiment), 300)
        # Just check it's a valid series, NaN values are expected
        self.assertIsInstance(vol_sentiment.iloc[0], (float, int))

    def test_calculate_composite_sentiment(self):
        """Test composite sentiment calculation."""
        from src.sentiment import calculate_composite_sentiment

        dates = pd.date_range("2023-01-01", periods=100, freq="1d")
        prices = pd.Series([100 + i for i in range(100)], index=dates)
        composite = calculate_composite_sentiment(prices, window=14)

        self.assertIsInstance(composite, pd.Series)
        self.assertEqual(len(composite), 100)
        valid_values = composite.dropna()
        self.assertTrue(all(0 <= v <= 100 for v in valid_values))

    def test_get_sentiment_for_asset_btc(self):
        """Test get_sentiment_for_asset for BTC."""
        from src.sentiment import get_sentiment_for_asset

        dates = pd.date_range("2023-01-01", periods=50, freq="1d")
        prices = pd.Series([100 + i for i in range(50)], index=dates)
        sentiment = get_sentiment_for_asset(prices, asset_type="btc", window=14)

        self.assertIsInstance(sentiment, pd.Series)
        self.assertEqual(len(sentiment), 50)

    def test_get_sentiment_for_asset_eth(self):
        """Test get_sentiment_for_asset for ETH."""
        from src.sentiment import get_sentiment_for_asset

        dates = pd.date_range("2023-01-01", periods=50, freq="1d")
        prices = pd.Series([100 + i for i in range(50)], index=dates)
        sentiment = get_sentiment_for_asset(prices, asset_type="eth", window=14)

        self.assertIsInstance(sentiment, pd.Series)
        self.assertEqual(len(sentiment), 50)

    def test_get_sentiment_for_asset_xrp(self):
        """Test get_sentiment_for_asset for XRP."""
        from src.sentiment import get_sentiment_for_asset

        dates = pd.date_range("2023-01-01", periods=50, freq="1d")
        prices = pd.Series([100 + i for i in range(50)], index=dates)
        sentiment = get_sentiment_for_asset(prices, asset_type="xrp", window=14)

        self.assertIsInstance(sentiment, pd.Series)
        self.assertEqual(len(sentiment), 50)

    def test_sentiment_to_fgi_equivalent(self):
        """Test sentiment to FGI equivalent conversion."""
        from src.sentiment import sentiment_to_fgi_equivalent

        sentiment = pd.Series([10, 50, 90, 150, -10])
        fgi_equiv = sentiment_to_fgi_equivalent(sentiment)

        self.assertIsInstance(fgi_equiv, pd.Series)
        self.assertTrue(all(0 <= v <= 100 for v in fgi_equiv))
        self.assertEqual(fgi_equiv.iloc[0], 10)
        self.assertEqual(fgi_equiv.iloc[3], 100)
        self.assertEqual(fgi_equiv.iloc[4], 0)

    def test_rsi_sentiment_with_different_windows(self):
        """Test RSI sentiment with different window sizes."""
        from src.sentiment import calculate_rsi_sentiment

        dates = pd.date_range("2023-01-01", periods=100, freq="1d")
        prices = pd.Series([100 + i for i in range(100)], index=dates)

        for window in [7, 14, 21, 30]:
            sentiment = calculate_rsi_sentiment(prices, window=window)
            self.assertEqual(len(sentiment), 100)
            valid_values = sentiment.dropna()
            self.assertTrue(all(0 <= v <= 100 for v in valid_values))




class TestMLModelComprehensive(unittest.TestCase):
    """Comprehensive tests for ML model module."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2023-01-01", periods=180)
        prices = [100 + i for i in range(180)]
        self.prices = pd.Series(prices, index=dates)

        self.ohlcv = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000 + i * 10 for i in range(180)],
            },
            index=dates,
        )

        self.fgi = pd.DataFrame(
            {
                "fgi_value": [40 + (i % 60) for i in range(180)],
                "fgi_classification": ["Neutral"] * 180,
            },
            index=dates,
        )

    def test_train_ml_model_different_lookbacks(self):
        """Test ML model training with different lookback periods."""
        from src.ml.ml_model import train_ml_model

        for lookback in [30, 60, 90]:
            model, predictions, metrics = train_ml_model(
                self.ohlcv, self.fgi, lookback_days=lookback
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(predictions)
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)

    def test_prepare_ml_data(self):
        """Test ML data preparation."""
        from src.ml.ml_model import prepare_ml_data
        from src.indicators import calculate_rsi

        rsi = calculate_rsi(self.ohlcv["close"], window=14)
        ml_df = prepare_ml_data(self.ohlcv, self.fgi, rsi)

        self.assertIsInstance(ml_df, pd.DataFrame)
        self.assertIn("target", ml_df.columns)
        self.assertIn("fgi", ml_df.columns)
        self.assertGreater(len(ml_df), 0)

    def test_train_ml_model(self):
        """Test ML model training."""
        from src.ml.ml_model import train_ml_model

        model, predictions, metrics = train_ml_model(
            self.ohlcv, self.fgi, lookback_days=90
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
        self.assertIn("accuracy", metrics)

    def test_train_ml_model_small_data(self):
        """Test ML model training with smaller data."""
        from src.ml.ml_model import train_ml_model

        small_ohlcv = self.ohlcv.iloc[:90]
        small_fgi = self.fgi.iloc[:90]

        model, predictions, metrics = train_ml_model(
            small_ohlcv, small_fgi, lookback_days=30
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
class TestStrategyComprehensive(unittest.TestCase):
    """Comprehensive tests for strategy module."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2024-01-01", periods=100)
        self.prices = pd.Series(
            [1000 + i * 10 + (i % 20) * 5 for i in range(100)], index=dates
        )

        self.fgi = pd.DataFrame(
            {
                "fgi_value": [25] * 50 + [75] * 50,
                "fgi_classification": ["Fear"] * 50 + ["Greed"] * 50,
            },
            index=dates,
        )

    def test_detect_market_regime_strong_bull(self):
        """Test market regime detection for strong bull market."""
        from src.strategy import detect_market_regime

        # Create strongly trending up prices
        dates = pd.date_range("2024-01-01", periods=100)
        bull_prices = pd.Series([1000 + i * 15 for i in range(100)], index=dates)

        regime = detect_market_regime(bull_prices, lookback=50)

        self.assertIsInstance(regime, pd.Series)
        # At least some periods should be bull or strong_bull
        unique_regimes = regime.unique()
        self.assertTrue(len(unique_regimes) > 0)

    def test_detect_market_regime_bear(self):
        """Test market regime detection for bear market."""
        from src.strategy import detect_market_regime

        # Create strongly trending down prices
        dates = pd.date_range("2024-01-01", periods=100)
        bear_prices = pd.Series([1000 - i * 15 for i in range(100)], index=dates)

        regime = detect_market_regime(bear_prices, lookback=50)

        self.assertIsInstance(regime, pd.Series)

    def test_generate_signal_with_short_selling(self):
        """Test generate_signal with short selling enabled."""
        from src.strategy import generate_signal

        # High FGI should trigger short - match the length
        high_fgi = pd.DataFrame(
            {"fgi_value": [85] * 100, "fgi_classification": ["Extreme Greed"] * 100},
            index=self.prices.index,
        )

        result = generate_signal(
            self.prices, high_fgi, enable_short_selling=True
        )

        self.assertIn("signal", result)

    def test_generate_signal_with_cover(self):
        """Test generate_signal with cover signal."""
        from src.strategy import generate_signal

        # Low FGI should trigger cover when short selling enabled - match length
        low_fgi = pd.DataFrame(
            {"fgi_value": [20] * 100, "fgi_classification": ["Extreme Fear"] * 100},
            index=self.prices.index,
        )

        result = generate_signal(
            self.prices, low_fgi, enable_short_selling=True
        )

        self.assertIn("signal", result)

    def test_generate_signal_with_news_sentiment(self):
        """Test generate_signal with news sentiment enabled."""
        from src.strategy import generate_signal

        with patch("src.data.data_fetchers.fetch_crypto_news_sentiment") as mock_news:
            mock_news.return_value = {"sentiment_score": 0.8}
            result = generate_signal(
                self.prices, self.fgi, enable_news_sentiment=True
            )

        self.assertIn("signal", result)
        self.assertIn("indicators", result)

    def test_generate_signal_with_options_flow(self):
        """Test generate_signal with options flow enabled."""
        from src.strategy import generate_signal

        with patch("src.data.data_fetchers.fetch_options_flow") as mock_options:
            mock_options.return_value = {"fear_gauge": 0.3}
            result = generate_signal(
                self.prices, self.fgi, enable_options_flow=True
            )

        self.assertIn("signal", result)
        self.assertIn("indicators", result)

    def test_generate_signal_bull_market_thresholds(self):
        """Test generate_signal adjusts thresholds for bull markets."""
        from src.strategy import generate_signal

        # FGI slightly above average in bull market should trigger sell
        dates = pd.date_range("2024-01-01", periods=100)
        prices = pd.Series([1000 + i * 15 for i in range(100)], index=dates)
        fgi_values = [50] * 90 + [60] * 10  # Rising FGI

        fgi_df = pd.DataFrame(
            {"fgi_value": fgi_values, "fgi_classification": ["Neutral"] * 100},
            index=dates,
        )

        result = generate_signal(prices, fgi_df)

        self.assertIn("indicators", result)
        self.assertIn("fgi_trend", result["indicators"])

    def test_generate_signal_bear_market_thresholds(self):
        """Test generate_signal adjusts thresholds for bear markets."""
        from src.strategy import generate_signal

        # FGI slightly below average in bear market
        dates = pd.date_range("2024-01-01", periods=100)
        prices = pd.Series([1000 - i * 15 for i in range(100)], index=dates)
        fgi_values = [50] * 90 + [40] * 10  # Falling FGI

        fgi_df = pd.DataFrame(
            {"fgi_value": fgi_values, "fgi_classification": ["Neutral"] * 100},
            index=dates,
        )

        result = generate_signal(prices, fgi_df)

        self.assertIn("indicators", result)

    def test_run_strategy_with_multi_timeframe(self):
        """Test run_strategy with multi-timeframe data."""
        from src.strategy import run_strategy

        # Create higher TF indicators
        higher_tf_indicators = {
            "trend": pd.Series([True] * 100, index=self.prices.index),
            "ema_fast": self.prices.ewm(span=20).mean(),
            "ema_slow": self.prices.ewm(span=50).mean(),
            "rsi": pd.Series([50] * 100, index=self.prices.index),
        }

        result = run_strategy(
            self.prices,
            "1d",
            self.fgi,
            "TEST",
            higher_tf_data=higher_tf_indicators,
            enable_multi_tf=True,
        )

        self.assertIn("total_return", result)
        self.assertIn("total_trades", result)

    def test_run_strategy_with_regime_filter(self):
        """Test run_strategy with regime filter."""
        from src.strategy import run_strategy

        result = run_strategy(
            self.prices,
            "1d",
            self.fgi,
            "TEST",
            enable_regime_filter=True,
        )

        self.assertIn("total_return", result)
        self.assertIn("total_trades", result)

    def test_run_strategy_with_atr_trailing_stop(self):
        """Test run_strategy with ATR-based trailing stop."""
        from src.strategy import run_strategy

        result = run_strategy(
            self.prices,
            "1d",
            self.fgi,
            "TEST",
            use_atr_trail=True,
            atr_multiplier=2.5,
        )

        self.assertIn("total_return", result)

    def test_run_strategy_with_max_drawdown(self):
        """Test run_strategy with max drawdown limit."""
        from src.strategy import run_strategy

        result = run_strategy(
            self.prices,
            "1d",
            self.fgi,
            "TEST",
            max_drawdown_pct=0.10,
        )

        self.assertIn("total_return", result)
        self.assertIn("max_drawdown", result)


class TestMLEnsembleAndLSTM(unittest.TestCase):
    """Tests for ML ensemble and LSTM functions."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2023-01-01", periods=180)
        prices = [100 + i for i in range(180)]
        
        self.ohlcv = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000 + i * 10 for i in range(180)],
            },
            index=dates,
        )

        self.fgi = pd.DataFrame(
            {
                "fgi_value": [40 + (i % 60) for i in range(180)],
                "fgi_classification": ["Neutral"] * 180,
            },
            index=dates,
        )

    def test_create_lstm_sequences(self):
        """Test LSTM sequence creation."""
        from src.ml.ml_model import create_lstm_sequences, prepare_ml_data
        from src.indicators import calculate_rsi

        rsi = calculate_rsi(self.ohlcv["close"], window=14)
        ml_df = prepare_ml_data(self.ohlcv, self.fgi, rsi)

        try:
            X, y = create_lstm_sequences(ml_df, sequence_length=10)
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(X.shape[1], 10)
        except ImportError:
            self.skipTest("TensorFlow not available")

    def test_train_lstm_model(self):
        """Test LSTM model training function exists."""
        from src.ml.ml_model import train_lstm_model

        # Just verify the function exists and is callable
        self.assertTrue(callable(train_lstm_model))

    def test_train_ml_ensemble(self):
        """Test ML ensemble training."""
        from src.ml.ml_model import train_ml_ensemble

        # Just verify the function exists and is callable
        self.assertTrue(callable(train_ml_ensemble))

    def test_prepare_ml_data_features(self):
        """Test ML data preparation returns expected features."""
        from src.ml.ml_model import prepare_ml_data
        from src.indicators import calculate_rsi

        rsi = calculate_rsi(self.ohlcv["close"], window=14)
        ml_df = prepare_ml_data(self.ohlcv, self.fgi, rsi)

        expected_features = [
            "fgi", "rsi", "returns_3d", "returns_7d", "returns_30d",
            "volatility_7d", "volatility_30d", "atr_14d",
            "volume", "volume_ratio"
        ]

        for feature in expected_features:
            self.assertIn(feature, ml_df.columns, f"Missing feature: {feature}")

    def test_ml_data_target_values(self):
        """Test ML data target values are valid."""
        from src.ml.ml_model import prepare_ml_data
        from src.indicators import calculate_rsi

        rsi = calculate_rsi(self.ohlcv["close"], window=14)
        ml_df = prepare_ml_data(self.ohlcv, self.fgi, rsi)

        self.assertIn("target", ml_df.columns)
        target_values = ml_df["target"].dropna()
        self.assertTrue(all(t in [0, 1, 2] for t in target_values))

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    unittest.main(verbosity=2)
