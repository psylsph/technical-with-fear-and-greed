"""
Unit tests for Telegram bot functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.telegram_bot import TelegramBot, get_telegram_bot


class TestTelegramBot:
    """Test cases for TelegramBot class."""

    def setup_method(self):
        """Setup before each test."""
        # Clear any existing bot instance
        import src.telegram_bot
        src.telegram_bot._bot_instance = None

        # Mock environment variables
        self.mock_env = {
            'TELEGRAM_BOT_TOKEN': 'test_token_123',
            'TELEGRAM_CHAT_ID': '123456789'
        }

    def teardown_method(self):
        """Cleanup after each test."""
        import src.telegram_bot
        src.telegram_bot._bot_instance = None

    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_bot_initialization_enabled(self):
        """Test bot initializes correctly when credentials are set."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', True):
            bot = TelegramBot()
            assert bot.enabled is True
            assert bot.token == 'test_token'
            assert bot.chat_id == '123'

    @patch.dict(os.environ, {}, clear=True)
    def test_bot_initialization_disabled_no_credentials(self):
        """Test bot initializes as disabled when no credentials."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', True):
            bot = TelegramBot()
            assert bot.enabled is False

    def test_bot_initialization_disabled_no_library(self):
        """Test bot initializes as disabled when library not available."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', False):
            bot = TelegramBot()
            assert bot.enabled is False

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_send_notification_enabled(self):
        """Test sending notification when bot is enabled."""
        bot = TelegramBot()

        with patch.object(bot, '_message_queue') as mock_queue:
            result = bot.send_notification("Test message")
            assert result is True
            mock_queue.put.assert_called_once()

    def test_send_notification_disabled(self):
        """Test sending notification when bot is disabled."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', False):
            bot = TelegramBot()
            result = bot.send_notification("Test message")
            assert result is False

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_send_trade_notification(self):
        """Test sending trade notification."""
        bot = TelegramBot()

        with patch.object(bot, 'send_notification') as mock_send:
            result = bot.send_trade_notification(
                symbol="ETH-USD",
                action="buy",
                quantity=1.5,
                price=3000.0,
                reason="Test trade"
            )
            assert result is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert "TRADE EXECUTED" in call_args
            assert "ETH-USD" in call_args
            assert "BUY" in call_args
            assert "1.500000" in call_args

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_send_signal_notification(self):
        """Test sending signal notification."""
        bot = TelegramBot()

        with patch.object(bot, 'send_notification') as mock_send:
            indicators = {"fgi": 25, "fgi_trend": "bearish"}
            result = bot.send_signal_notification(
                symbol="ETH-USD",
                signal="buy",
                price=3000.0,
                indicators=indicators
            )
            assert result is True
            mock_send.assert_called_once()

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_send_error_notification(self):
        """Test sending error notification."""
        bot = TelegramBot()

        with patch.object(bot, 'send_notification') as mock_send:
            result = bot.send_error_notification("Test error", "Test context")
            assert result is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert "ERROR" in call_args
            assert "Test error" in call_args
            assert "Test context" in call_args

    def test_get_telegram_bot_singleton(self):
        """Test that get_telegram_bot returns singleton instance."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', False):
            bot1 = get_telegram_bot()
            bot2 = get_telegram_bot()
            assert bot1 is bot2

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_start_bot_disabled_library(self):
        """Test starting bot when library not available."""
        with patch('src.telegram_bot.TELEGRAM_AVAILABLE', False):
            bot = TelegramBot()
            result = bot.start()
            assert result is False

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_start_bot_already_running(self):
        """Test starting bot when already running."""
        bot = TelegramBot()
        bot._running = True
        result = bot.start()
        assert result is True

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    @patch('threading.Thread')
    def test_start_bot_success(self, mock_thread):
        """Test successful bot start."""
        bot = TelegramBot()
        result = bot.start()
        assert result is True
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


class TestTelegramBotIntegration:
    """Integration tests for Telegram bot."""

    def setup_method(self):
        """Setup before each test."""
        # Clear any existing bot instance
        import src.telegram_bot
        src.telegram_bot._bot_instance = None

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_command_handler_registration(self):
        """Test that command handlers are properly registered."""
        with patch('src.telegram_bot.Application') as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.builder.return_value.token.return_value.build.return_value = mock_app

            bot = TelegramBot()
            # Simulate the handler registration from _run_bot
            bot.application = mock_app

            # Manually call the handler setup (simulating _run_bot)
            from src.telegram_bot import CommandHandler
            with patch('src.telegram_bot.CommandHandler') as mock_handler_class:
                # This would normally happen in _run_bot
                handlers = [
                    ('start', bot._cmd_start),
                    ('help', bot._cmd_help),
                    ('status', bot._cmd_status),
                    ('account', bot._cmd_account),
                    ('positions', bot._cmd_positions),
                    ('trades', bot._cmd_trades),
                ]

                for cmd, handler in handlers:
                    mock_app.add_handler.assert_any_call(mock_handler_class.return_value)
                    mock_handler_class.assert_any_call(cmd, handler)

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_status_callback_integration(self):
        """Test that status callback is properly integrated."""
        bot = TelegramBot()

        # Set a mock status callback
        mock_callback = Mock(return_value={'account': {'equity': 10000}})
        bot.set_status_callback(mock_callback)

        assert bot._status_callback is mock_callback

    @patch('src.telegram_bot.TELEGRAM_AVAILABLE', True)
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': '123'})
    def test_message_queue_threading(self):
        """Test message queue handling in threaded environment."""
        import queue
        import threading

        bot = TelegramBot()

        # Test queue operations
        assert isinstance(bot._message_queue, queue.Queue)

        # Test putting messages in queue
        bot._message_queue.put(("test message", "Markdown"))
        assert not bot._message_queue.empty()

        # Test getting messages from queue
        message, parse_mode = bot._message_queue.get()
        assert message == "test message"
        assert parse_mode == "Markdown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])</content>
<parameter name="filePath">/home/stuart/repos/technical-with-fear-and-greed/test_telegram_bot.py