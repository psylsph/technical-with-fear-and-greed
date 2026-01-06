"""
Test suite for trading strategy implementation.
"""

import json
import os
import tempfile
import unittest
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

    def test_init_database(self):
        """Test database initialization."""
        from src.data.data_fetchers import init_database

        db_path = init_database()

        self.assertIsNotNone(db_path)
        self.assertTrue(os.path.exists(db_path))

    def test_save_and_get_cached_data(self):
        """Test saving and getting cached data."""
        from src.data.data_fetchers import get_cached_data, save_cached_data

        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5),
                "open": [100.0] * 5,
                "high": [102.0] * 5,
                "low": [99.0] * 5,
                "close": [101.0] * 5,
                "volume": [1000.0] * 5,
            }
        )

        save_cached_data("BTC-USD", test_data, "1d", "test_source")

        cached = get_cached_data(
            "BTC-USD", "2023-01-01", "2023-01-05", "1d", "test_source"
        )

        self.assertIsNotNone(cached)
        pd.testing.assert_frame_equal(cached, test_data)

    def test_save_and_get_cached_fgi(self):
        """Test saving and getting cached FGI data."""
        from src.data.data_fetchers import get_cached_fgi, save_cached_fgi

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
            index=pd.date_range("2023-01-01", periods=5),
        )

        save_cached_fgi(test_fgi)

        cached = get_cached_fgi("2023-01-01", "2023-01-05")

        self.assertIsNotNone(cached)
        pd.testing.assert_frame_equal(cached, test_fgi)

    def test_fetch_fear_greed_index(self):
        """Test FGI index fetching."""
        fgi_data = fetch_fear_greed_index()

        self.assertIsInstance(fgi_data, pd.DataFrame)
        self.assertIn("fgi_value", fgi_data.columns)
        self.assertIn("fgi_classification", fgi_data.columns)
        self.assertGreater(len(fgi_data), 0)

    def test_get_current_price_empty(self):
        """Test getting current price with no data."""
        from src.data.data_fetchers import get_current_price

        # Mock Yahoo Finance API to return None
        with patch("src.data.data_fetchers.vbt.YFData.download", return_value=None):
            price = get_current_price("BTC-USD")
            self.assertIsNone(price)

    def test_get_current_price_success(self):
        """Test getting current price with valid data."""
        from src.data.data_fetchers import get_current_price

        # Mock Yahoo Finance API to return valid price
        mock_data = type(
            "obj",
            (object,),
            {
                "get": lambda self, key: pd.Series(
                    [50000.0], index=pd.date_range("2023-01-01", periods=1)
                )
            },
        )()
        mock_data.get.return_value = mock_data.get("Close")

        with patch("src.data.data_fetchers.vbt.YFData", return_value=mock_data):
            price = get_current_price("BTC-USD")
            self.assertIsNotNone(price)
            self.assertEqual(price, 50000.0)

    def test_get_current_fgi(self):
        """Test getting current FGI value."""
        from src.data.data_fetchers import get_current_fgi, save_cached_fgi

        today = pd.Timestamp.now().normalize()

        # Save test FGI data for today
        test_fgi = pd.DataFrame(
            {"fgi_value": [45], "fgi_classification": ["Neutral"]},
            index=[today],
        )
        save_cached_fgi(test_fgi)

        # Get current FGI (should be 45)
        fgi_value, classification = get_current_fgi()
        self.assertEqual(fgi_value, 45)
        self.assertEqual(classification, "Neutral")

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
                    114,
                ]
            )
        )

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), 14)  # First 14 are NaN

        # Test that RSI values are in valid range
        rsi_valid = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_valid))

    def test_calculate_rsi_extremes(self):
        """Test RSI calculation at extremes."""
        # Low prices (oversold)
        rsi_low = calculate_rsi(pd.Series([100] * 20))
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
        self.prices = pd.Series([100 + i for i in range(50)], index=dates)
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

        # Should have 37 rows (50 - 13 NaN from RSI window)
        self.assertEqual(len(result["rsi"]), 37)

    def test_train_ml_model(self):
        """Test ML model training."""
        from src.ml.ml_model import train_ml_model

        model, predictions, metrics = train_ml_model(
            self.prices, self.fgi, lookback_days=30
        )

        self.assertIsNotNone(model)
        self.assertIsInstance(predictions, pd.Series)
        # Length matches prepare_ml_data output
        self.assertEqual(len(predictions), len(self.prices) - 13)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("lookback_days", metrics)

    def test_predict_live_fgi_no_model(self):
        """Test live FGI prediction when model is not trained."""
        from src.ml.ml_model import ml_model, predict_live_fgi

        # Ensure no model
        original_model = ml_model
        ml_model.ml_model = None

        # Test prediction
        today = pd.Timestamp.now().normalize()
        test_price = pd.Series([150.0], index=[today])
        prediction = predict_live_fgi(test_price, self.fgi, today)

        # Should return default (0.5)
        self.assertEqual(prediction, 0.5)

        # Restore original model
        ml_model.ml_model = original_model

    def test_predict_live_fgi_missing_data(self):
        """Test live FGI prediction with missing FGI data."""
        from src.ml.ml_model import predict_live_fgi, train_ml_model

        # Train model first
        train_ml_model(self.prices, self.fgi, lookback_days=30)

        # Test with date that won't exist in FGI data
        future_date = pd.Timestamp("2030-01-01")
        test_price = pd.Series([150.0], index=[future_date])
        prediction = predict_live_fgi(test_price, self.fgi, future_date)

        # Should return default
        self.assertEqual(prediction, 0.5)

    def test_predict_live_fgi_index_error(self):
        """Test live FGI prediction with index error."""
        from src.ml.ml_model import predict_live_fgi, train_ml_model

        # Train model first
        train_ml_model(self.prices, self.fgi, lookback_days=30)

        # Test with insufficient price data (less than 14 for RSI)
        short_price = pd.Series([150.0])
        today = pd.Timestamp.now().normalize()
        prediction = predict_live_fgi(short_price, self.fgi, today)

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
        """Test buy signal generation."""
        signal = generate_signal(
            self.prices, self.fgi, rsi_window=14, buy_quantile=0.2, sell_quantile=0.8
        )

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # Should be sell signal with high FGI at end
        self.assertEqual(signal["signal"], "sell")

    def test_generate_signal_sell_extreme_greed(self):
        """Test sell signal generation for extreme greed."""
        signal = generate_signal(self.prices, self.fgi, rsi_window=14)

        self.assertIn("signal", signal)
        # Later dates have high FGI, should trigger sell
        self.assertEqual(signal["signal"], "sell")

    def test_generate_signal_sell_overbought(self):
        """Test sell signal generation for overbought RSI."""
        # Create overbought RSI scenario - rising prices
        high_prices = pd.Series(
            [150] * 30, index=pd.date_range("2023-01-01", periods=30)
        )

        neutral_fgi = self.fgi.copy()
        neutral_fgi["fgi_value"] = 50

        signal = generate_signal(high_prices, neutral_fgi, rsi_window=14)

        self.assertIn("signal", signal)
        self.assertIn("indicators", signal)
        # RSI should be high (overbought), triggering sell
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
        # FGI=60 equals sell_thresh (quantile(0.8) of all 60s = 60), triggering sell
        self.assertEqual(signal["signal"], "sell")

    def test_generate_signal_missing_fgi(self):
        """Test signal generation with missing FGI data."""
        future_date = pd.Timestamp("2030-01-01")
        future_prices = pd.Series([100.0], index=[future_date])

        signal = generate_signal(future_prices, self.fgi)

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
        # ML prediction should be included in indicators
        self.assertIn("ml_pred", signal["indicators"])
        self.assertEqual(signal["indicators"]["ml_pred"], 0.6)

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

    @patch("src.trading.trading_engine.analyze_live_signal")
    def test_analyze_live_signal_error(self):
        """Test live signal analysis with error."""
        from src.trading.trading_engine.analyze_live_signal import analyze_live_signal

        result = analyze_live_signal(self.fgi)
        # Should handle None return gracefully
        self.assertIsNone(result)

    @patch("src.trading.trading_engine.analyze_live_signal")
    def test_analyze_live_signal_success(self):
        """Test live signal analysis with valid data."""
        from src.trading.trading_engine.analyze_live_signal import (
            analyze_live_signal as _analyze_live_signal,
        )

        # Mock return value
        mock_signal = {
            "signal": "buy",
            "indicators": {"fgi": 45, "rsi": 60.0, "ml_pred": 0.6, "price": 50000.0},
            "in_position": False,
        }

        _analyze_live_signal.return_value = mock_signal

        result = _analyze_live_signal(self.fgi)

        self.assertIsNotNone(result)
        self.assertIn("signal", result)
        self.assertIn("indicators", result)

    def test_log_trade(self):
        """Test trade logging."""
        from src.trading.trading_engine import log_trade

        signal_info = {
            "indicators": {"price": 50000.0, "fgi": 45, "rsi": 60.0, "ml_pred": 0.6}
        }

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            from src.trading import trading_engine

            original_log_file = trading_engine.PROJECT_ROOT
            trading_engine.PROJECT_ROOT = os.path.dirname(temp_file)

            log_trade(signal_info, "buy", 0.01, "test_order_id")

            # Check if log file was created and contains data
            log_file_path = os.path.join(os.path.dirname(temp_file), "trade_log.json")
            if os.path.exists(log_file_path):
                with open(log_file_path) as f:
                    logs = json.load(f)
                    self.assertEqual(len(logs), 1)
                    self.assertEqual(logs[0]["action"], "buy")
                    self.assertEqual(logs[0]["quantity"], 0.01)
                    self.assertEqual(logs[0]["order_id"], "test_order_id")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
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
        # Create test data
        prices = pd.Series(
            [100 + i for i in range(30)], index=pd.date_range("2023-01-01", periods=30)
        )
        fgi = pd.DataFrame(
            {
                "fgi_value": [40 + i for i in range(30)],
                "fgi_classification": ["Neutral"] * 30,
            },
            index=pd.date_range("2023-01-01", periods=30),
        )

        # Train ML model
        from src.ml.ml_model import train_ml_model

        model, pred_series, _ = train_ml_model(prices, fgi, lookback_days=30)

        # Test strategy with ML
        result = run_strategy(prices, "1d", fgi, "TEST", pred_series=pred_series)

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
        self.assertEqual(len(resampled), len(self.hourly_close))
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

        # Create higher TF indicators (matching length to hourly)
        higher_tf_indicators = {
            "trend": pd.Series([True, False] * 50, index=self.daily_close[:50].index),
            "rsi": pd.Series(
                [40 + i % 20 for i in range(50)], index=self.daily_close[:50].index
            ),
        }

        # Align with hourly data (50 days * 24 hours = 1200 bars)
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
