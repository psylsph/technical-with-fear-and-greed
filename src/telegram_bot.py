"""
Telegram bot for read-only trading notifications and status queries.

This bot provides:
- Notifications for buy/sell signals
- Status queries (portfolio, positions, recent trades)
- Read-only interaction (no trading commands from Telegram)

Setup:
1. Create a bot via @BotFather on Telegram
2. Get the bot token
3. Set TELEGRAM_BOT_TOKEN in .env file
4. Start a chat with your bot and get your chat_id
5. Set TELEGRAM_CHAT_ID in .env file
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import queue
import threading

# Lazy import to avoid issues if library not installed
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    ContextTypes = None


class TelegramBot:
    """Telegram bot for trading notifications and status queries."""

    def __init__(self):
        """Initialize the Telegram bot."""
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(TELEGRAM_AVAILABLE and self.token and self.chat_id)
        self.application: Optional["Application"] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._startup_event = threading.Event()

        # Callback for getting trading status
        self._status_callback: Optional[callable] = None

        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                print("Telegram: python-telegram-bot not installed")
                print("  Install with: pip install python-telegram-bot")
            else:
                print(
                    "Telegram: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env"
                )

    def set_status_callback(self, callback: callable):
        """Set the callback function for getting trading status."""
        self._status_callback = callback

    def _is_authorized_chat(self, chat_id) -> bool:
        """Check if the chat is authorized to send commands."""
        return str(chat_id) == str(self.chat_id)

    def is_enabled(self) -> bool:
        """Check if Telegram bot is properly configured."""
        return self.enabled

    def send_notification(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a notification message to the configured chat.

        Args:
            message: The message to send
            parse_mode: Parse mode (Markdown, HTML, or None)

        Returns:
            True if message was sent successfully
        """
        if not self.enabled:
            return False

        # Always queue messages - the queue processor will handle them
        # This avoids timeout issues with run_coroutine_threadsafe
        try:
            self._message_queue.put((message, parse_mode), block=False)
            return True
        except queue.Full:
            print("Telegram: Message queue full, dropping message")
            return False

    async def _send_message_async(self, message: str, parse_mode: str):
        """Send a message asynchronously."""
        await self.application.bot.send_message(
            chat_id=self.chat_id, text=message, parse_mode=parse_mode
        )

    def send_portfolio_notification(
        self,
        portfolio_summary: Dict[str, Any],
        positions: List[Dict[str, Any]],
    ) -> bool:
        """Send a portfolio summary notification for multi-asset trading.

        Args:
            portfolio_summary: Portfolio-level summary dict
            positions: List of position dicts for all assets

        Returns:
            True if notification was sent
        """
        if not self.enabled:
            return False

        total_value = portfolio_summary.get("total_value", 0)
        total_pnl = portfolio_summary.get("total_pnl", 0)
        total_pnl_pct = portfolio_summary.get("total_pnl_pct", 0)
        cash = portfolio_summary.get("cash", 0)
        day_pnl = portfolio_summary.get("day_pnl", 0)

        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

        message = "📊 *PORTFOLIO SUMMARY*\n\n"
        message += f"*Total Value:* ${total_value:,.2f}\n"
        message += f"*Cash:* ${cash:,.2f}\n"
        message += f"{pnl_emoji} *Total P&L:* ${total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)\n"
        message += f"*Day P&L:* ${day_pnl:+,.2f}\n"

        if positions:
            long_value = sum(
                pos.get("value", 0) for pos in positions if pos.get("qty", 0) > 0
            )
            short_value = sum(
                abs(pos.get("value", 0)) for pos in positions if pos.get("qty", 0) < 0
            )
            message += f"\n*Long Positions:* ${long_value:,.2f}\n"
            message += f"*Short Positions:* ${short_value:,.2f}\n"
            message += f"*Total Positions:* {len(positions)}\n"

        return self.send_notification(message)

    def send_multi_asset_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        reason: str = "",
    ) -> bool:
        """Send a trade notification with portfolio context.

        Args:
            symbol: Trading symbol (e.g., "ETH-USD")
            action: "buy" or "sell"
            quantity: Quantity traded
            price: Execution price
            portfolio_value: Current portfolio value
            reason: Reason for the trade

        Returns:
            True if notification was sent
        """
        if not self.enabled:
            return False

        emoji = "🟢" if action.lower() == "buy" else "🔴"
        action_upper = action.upper()
        trade_value = abs(quantity * price)
        allocation = (trade_value / portfolio_value * 100) if portfolio_value > 0 else 0

        message = f"{emoji} *{action_upper} {symbol}*\n\n"
        message += f"*Quantity:* {abs(quantity):.6f}\n"
        message += f"*Price:* ${price:,.2f}\n"
        message += f"*Value:* ${trade_value:,.2f}\n"
        message += f"*Portfolio:* ${portfolio_value:,.2f}\n"
        message += f"*Allocation:* {allocation:.2f}%\n"

        if reason:
            message += f"*Reason:* {reason}\n"

        message += f"\n*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_notification(message)

    def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        reason: str = "",
        indicators: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a trade execution notification.

        Args:
            symbol: Trading symbol (e.g., "ETH-USD")
            action: "buy" or "sell"
            quantity: Quantity traded
            price: Execution price
            reason: Reason for the trade
            indicators: Optional dict of indicator values

        Returns:
            True if notification was sent
        """
        if not self.enabled:
            return False

        emoji = "🟢" if action.lower() == "buy" else "🔴"
        action_upper = action.upper()

        message = f"{emoji} *TRADE EXECUTED*\n\n"
        message += f"*Symbol:* {symbol}\n"
        message += f"*Action:* {action_upper}\n"
        message += f"*Quantity:* {quantity:.6f}\n"
        message += f"*Price:* ${price:,.2f}\n"
        message += f"*Value:* ${quantity * price:,.2f}\n"

        if reason:
            message += f"*Reason:* {reason}\n"

        if indicators:
            message += "\n*Indicators:*\n"
            fgi = indicators.get("fgi", "N/A")
            message += f"  FGI: {fgi}\n"

            signal = indicators.get("signal", "N/A")
            message += f"  Signal: {signal}\n"

            position_size = indicators.get("position_size_pct", "N/A")
            if position_size != "N/A":
                message += f"  Position Size: {position_size:.1f}%\n"

        message += f"\n*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_notification(message)

    def send_signal_notification(
        self, symbol: str, signal: str, price: float, indicators: Dict[str, Any]
    ) -> bool:
        """Send a new signal notification.

        Args:
            symbol: Trading symbol
            signal: Signal type (buy, sell, hold)
            price: Current price
            indicators: Dict of indicator values

        Returns:
            True if notification was sent
        """
        if not self.enabled:
            return False

        emoji_map = {
            "buy": "📈",
            "sell": "📉",
            "hold": "⏸️",
        }
        emoji = emoji_map.get(signal.lower(), "❓")

        message = f"{emoji} *NEW SIGNAL: {signal.upper()}*\n\n"
        message += f"*Symbol:* {symbol}\n"
        message += f"*Price:* ${price:,.2f}\n"

        if indicators:
            fgi = indicators.get("fgi", "N/A")
            message += f"*FGI:* {fgi}\n"

            fgi_trend = indicators.get("fgi_trend", "N/A")
            if fgi_trend != "N/A":
                message += f"*Trend:* {fgi_trend.upper()}\n"

            regime = indicators.get("market_regime", "N/A")
            if regime != "N/A":
                message += f"*Regime:* {regime.upper()}\n"

            vol_stop = indicators.get("volatility_stop")
            if vol_stop:
                message += f"*Vol Stop:* ${vol_stop:,.2f}\n"

        return self.send_notification(message)

    def send_error_notification(self, error: str, context: str = "") -> bool:
        """Send an error notification.

        Args:
            error: Error message
            context: Additional context

        Returns:
            True if notification was sent
        """
        if not self.enabled:
            return False

        message = "🚨 *ERROR*\n\n"
        if context:
            message += f"*Context:* {context}\n"
        message += f"*Message:* {error}\n"
        message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_notification(message)

    async def _cmd_status(self, update, context) -> None:
        """Handle /status command - show current trading status."""
        print("Telegram: === RECEIVED /status COMMAND ===")
        print(f"Telegram: From chat: {update.effective_chat.id}")
        print(
            f"Telegram: From user: {update.effective_user.username if update.effective_user else 'unknown'}"
        )
        print(f"Telegram: Configured chat: {self.chat_id}")
        print(f"Telegram: Has status callback: {self._status_callback is not None}")

        # Only respond to commands from the configured chat
        if str(update.effective_chat.id) != str(self.chat_id):
            print(
                f"Telegram: ❌ BLOCKED - Wrong chat ({update.effective_chat.id} != {self.chat_id})"
            )
            await update.message.reply_text(
                "❌ Unauthorized chat. Commands only work from the configured chat."
            )
            return

        print("Telegram: ✅ Authorized chat - processing command")

        if not self._status_callback:
            await update.message.reply_text(
                "Status callback not configured. Please check bot setup."
            )
            return

        try:
            status = self._status_callback()
            message = "*📊 TRADING STATUS*\n\n"

            # Account info
            account = status.get("account", {})
            if account:
                message += "*Account:*\n"
                message += f"  Equity: ${account.get('equity', 0):,.2f}\n"
                message += f"  Cash: ${account.get('cash', 0):,.2f}\n"
                message += f"  P&L: ${account.get('pnl', 0):+,.2f}\n\n"

            # Positions
            positions = status.get("positions", [])
            if positions:
                message += f"*Positions ({len(positions)}):*\n"
                for pos in positions:
                    symbol = pos.get("symbol", "N/A")
                    qty = pos.get("qty", 0)
                    entry = pos.get("avg_entry", 0)
                    current = pos.get("current_price", 0)
                    pnl = pos.get("unrealized_pnl", 0)
                    pnl_pct = pos.get("unrealized_pnl_pct", 0)

                    message += f"  {symbol}: {qty:+.6f} @ ${entry:,.2f}\n"
                    message += f"    Current: ${current:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
            else:
                message += "*Positions:* None\n\n"

            # Recent trades
            trades = status.get("recent_trades", [])
            if trades:
                message += f"*Recent Trades ({len(trades)}):*\n"
                for trade in trades[:5]:  # Show last 5
                    action = trade.get("side", "N/A").upper()
                    qty = trade.get("qty", 0)
                    price = trade.get("price", 0)
                    time_str = trade.get("time", "")
                    message += f"  {action} {qty:.6f} @ ${price:,.2f} - {time_str}\n"

            message += f"\n*Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(f"Error getting status: {e}")

    async def _cmd_help(self, update, context) -> None:
        """Handle /help command - show available commands."""
        print("Telegram: === RECEIVED /help COMMAND ===")
        print(f"Telegram: From chat: {update.effective_chat.id}")
        print(
            f"Telegram: From user: {update.effective_user.username if update.effective_user else 'unknown'}"
        )
        print(f"Telegram: Configured chat: {self.chat_id}")

        # Only respond to commands from the configured chat
        if str(update.effective_chat.id) != str(self.chat_id):
            print(
                f"Telegram: ❌ BLOCKED - Wrong chat ({update.effective_chat.id} != {self.chat_id})"
            )
            await update.message.reply_text(
                "❌ Unauthorized chat. Commands only work from the configured chat."
            )
            return

        print("Telegram: ✅ Authorized chat - processing command")

        message = """*📖 Trading Bot Commands*

*Status Commands:*
/status - Show current trading status
/account - Show account information
/positions - Show current positions
/trades - Show recent trades

*Bot Commands:*
/help - Show this help message
/start - Start the bot

*Note:* This is a read-only bot. All trading is handled by the main system.
"""
        await update.message.reply_text(message, parse_mode="Markdown")

    async def _cmd_start(self, update, context) -> None:
        """Handle /start command."""
        message = """*👋 Welcome to the Trading Bot!*

I'll send you notifications for:
• Buy/Sell signals
• Trade executions
• Errors and alerts

Use /help to see available commands.
"""
        await update.message.reply_text(message, parse_mode="Markdown")

    async def _cmd_account(self, update, context) -> None:
        """Handle /account command."""
        if not self._status_callback:
            await update.message.reply_text("Status callback not configured.")
            return

        status = self._status_callback()
        account = status.get("account", {})

        if not account:
            await update.message.reply_text("No account information available.")
            return

        message = f"""*💰 ACCOUNT*

*Equity:* ${account.get('equity', 0):,.2f}
*Cash:* ${account.get('cash', 0):,.2f}
*Buying Power:* ${account.get('buying_power', 0):,.2f}

*Portfolio Value:* ${account.get('portfolio_value', 0):,.2f}

*Today's P&L:* ${account.get('day_pnl', 0):+,.2f}
*Total P&L:* ${account.get('pnl', 0):+,.2f}

*Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await update.message.reply_text(message, parse_mode="Markdown")

    async def _cmd_positions(self, update, context) -> None:
        """Handle /positions command."""
        if not self._status_callback:
            await update.message.reply_text("Status callback not configured.")
            return

        status = self._status_callback()
        positions = status.get("positions", [])

        if not positions:
            await update.message.reply_text("No open positions.")
            return

        message = f"*📦 POSITIONS ({len(positions)})*\n\n"
        for pos in positions:
            symbol = pos.get("symbol", "N/A")
            qty = pos.get("qty", 0)
            entry = pos.get("avg_entry", 0)
            current = pos.get("current_price", 0)
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)

            side = "LONG" if qty > 0 else "SHORT"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"

            message += f"{pnl_emoji} *{symbol}* ({side})\n"
            message += f"  Quantity: {abs(qty):.6f}\n"
            message += f"  Entry: ${entry:,.2f}\n"
            message += f"  Current: ${current:,.2f}\n"
            message += f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n\n"

        await update.message.reply_text(message, parse_mode="Markdown")

    async def _cmd_trades(self, update, context) -> None:
        """Handle /trades command."""
        if not self._status_callback:
            await update.message.reply_text("Status callback not configured.")
            return

        status = self._status_callback()
        trades = status.get("recent_trades", [])

        if not trades:
            await update.message.reply_text("No recent trades.")
            return

        message = f"*📜 RECENT TRADES ({len(trades)})*\n\n"
        for trade in trades[:10]:  # Show last 10
            action = trade.get("side", "N/A").upper()
            emoji = "🟢" if action == "BUY" else "🔴"
            qty = trade.get("qty", 0)
            price = trade.get("price", 0)
            time_str = trade.get("time", "")

            message += f"{emoji} *{action}* {qty:.6f} @ ${price:,.2f}\n"
            message += f"  {time_str}\n\n"

        await update.message.reply_text(message, parse_mode="Markdown")

    async def _process_queue(self):
        """Process queued messages."""
        while self._running:
            try:
                message, parse_mode = self._message_queue.get(timeout=1)
                await self._send_message_async(message, parse_mode)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Telegram: Error processing queue: {e}")

    async def _send_startup_notification(self):
        """Send startup notification after bot is ready."""
        await self._send_message_async(
            "🚀 *Trading Bot Started*\n\n"
            "Notifications enabled. Use /help to see available commands.",
            "Markdown",
        )

    def start(self) -> bool:
        """Start the Telegram bot in a background thread.

        Returns:
            True if bot started successfully
        """
        if not self.enabled:
            print("Telegram: Bot not enabled or not configured")
            return False

        if self._running:
            print("Telegram: Bot already running")
            return True

        print("Telegram: Starting bot in background thread...")
        try:
            # Start in background thread
            self._running = True
            self._thread = threading.Thread(target=self._run_bot_thread, daemon=True)
            self._thread.start()

            # Wait for bot to signal it's ready
            if self._startup_event.wait(timeout=10):
                print("Telegram: Bot started successfully")
                return True
            else:
                print("Telegram: Bot failed to start within timeout")
                self._running = False
                return False

        except Exception as e:
            print(f"Telegram: Failed to start bot: {e}")
            self._running = False
            return False

    def _run_bot_thread(self):
        """Run the bot in a background thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Run the async bot function
            self._loop.run_until_complete(self._run_bot_async())

        except Exception as e:
            print(f"Telegram: Thread error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._running = False

    async def _run_bot_async(self):
        """Run the bot asynchronously."""
        print("Telegram: Bot starting...")
        try:
            self._running = True

            # Build application
            self.application = Application.builder().token(self.token).build()

            # Add handlers
            print("Telegram: Adding command handlers...")
            self.application.add_handler(CommandHandler("start", self._cmd_start))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("account", self._cmd_account))
            self.application.add_handler(
                CommandHandler("positions", self._cmd_positions)
            )
            self.application.add_handler(CommandHandler("trades", self._cmd_trades))

            # Add debug handler for ALL updates
            async def debug_all_updates(update, context):
                print(
                    f"Telegram: 🔄 RECEIVED UPDATE: type={type(update).__name__}, chat={getattr(update.effective_chat, 'id', 'unknown')}"
                )
                if hasattr(update, "message") and update.message:
                    print(
                        f"Telegram: 📨 MESSAGE: '{update.message.text}' from {update.effective_chat.id}"
                    )
                    if str(update.effective_chat.id) == str(self.chat_id):
                        await update.message.reply_text(
                            f"🤖 Bot received: {update.message.text}"
                        )
                    else:
                        print(
                            f"Telegram: ❌ Wrong chat: {update.effective_chat.id} != {self.chat_id}"
                        )
                elif hasattr(update, "callback_query") and update.callback_query:
                    print(f"Telegram: 🔘 CALLBACK: {update.callback_query.data}")
                else:
                    print(f"Telegram: ❓ OTHER UPDATE: {update}")

            # Add handlers for all updates
            from telegram.ext import MessageHandler, filters

            self.application.add_handler(
                MessageHandler(filters.ALL, debug_all_updates), group=0
            )

            print("Telegram: Handlers added")

            # Initialize and start
            print("Telegram: Initializing application...")
            await self.application.initialize()
            print("Telegram: Initializing updater...")
            await self.application.updater.initialize()
            print("Telegram: Starting application...")
            await self.application.start()

            # Signal that bot is ready
            self._startup_event.set()
            print("Telegram: Polling started, waiting for messages...")

            # Start polling
            await self.application.updater.start_polling(drop_pending_updates=True)
            print("Telegram: Polling active")

            # Send startup notification
            await self._send_startup_notification()

            # Keep the coroutine alive while polling (this will run forever)
            while self._running:
                await asyncio.sleep(1)

        except Exception as e:
            print(f"Telegram: Bot error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup
            try:
                print("Telegram: Stopping polling...")
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                print("Telegram: Cleanup completed")
            except Exception as e:
                print(f"Telegram: Error during cleanup: {e}")
            self._running = False
            self._startup_event.clear()

    def stop(self):
        """Stop the Telegram bot."""
        self._running = False
        if self._loop and self._loop.is_running():
            # Cancel all tasks in the loop
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


# Global bot instance
_bot_instance: Optional[TelegramBot] = None


def get_telegram_bot() -> TelegramBot:
    """Get or create the global Telegram bot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot()
    return _bot_instance


def send_trade_notification(
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    reason: str = "",
    indicators: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function to send trade notification."""
    bot = get_telegram_bot()
    return bot.send_trade_notification(
        symbol, action, quantity, price, reason, indicators
    )


def send_signal_notification(
    symbol: str, signal: str, price: float, indicators: Dict[str, Any]
) -> bool:
    """Convenience function to send signal notification."""
    bot = get_telegram_bot()
    return bot.send_signal_notification(symbol, signal, price, indicators)


def send_error_notification(error: str, context: str = "") -> bool:
    """Convenience function to send error notification."""
    bot = get_telegram_bot()
    return bot.send_error_notification(error, context)


def send_portfolio_notification(
    portfolio_summary: Dict[str, Any],
    positions: List[Dict[str, Any]],
) -> bool:
    """Convenience function to send portfolio notification."""
    bot = get_telegram_bot()
    return bot.send_portfolio_notification(portfolio_summary, positions)


def send_multi_asset_trade_notification(
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    portfolio_value: float,
    reason: str = "",
) -> bool:
    """Convenience function to send multi-asset trade notification."""
    bot = get_telegram_bot()
    return bot.send_multi_asset_trade_notification(
        symbol, action, quantity, price, portfolio_value, reason
    )
