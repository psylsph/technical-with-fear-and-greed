#!/usr/bin/env python3
"""
Test the Telegram bot standalone (without full trading system).
Run this to verify the bot is configured and responding correctly.
"""

import os
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from src.telegram_bot import get_telegram_bot


def main():
    print("=" * 60)
    print("Telegram Bot Test")
    print("=" * 60)

    # Get bot instance
    bot = get_telegram_bot()

    if not bot.is_enabled():
        print("\n❌ Bot is NOT enabled!")
        print("\nCheck your .env file:")
        print("  - TELEGRAM_BOT_TOKEN should be set")
        print("  - TELEGRAM_CHAT_ID should be set")
        print("\nYour current values:")
        print(f"  Token: {os.getenv('TELEGRAM_BOT_TOKEN', 'NOT SET')[:20]}...")
        print(f"  Chat ID: {os.getenv('TELEGRAM_CHAT_ID', 'NOT SET')}")
        return

    print("\n✅ Bot is enabled!")

    # Set a simple status callback for testing
    def get_test_status():
        return {
            "account": {
                "equity": 10000.00,
                "cash": 5000.00,
                "buying_power": 10000.00,
                "portfolio_value": 10000.00,
                "day_pnl": 150.50,
                "pnl": 500.00,
            },
            "positions": [
                {
                    "symbol": "ETH-USD",
                    "qty": 1.5,
                    "avg_entry": 3200.00,
                    "current_price": 3300.00,
                    "unrealized_pnl": 150.00,
                    "unrealized_pnl_pct": 3.13,
                }
            ],
            "recent_trades": [
                {
                    "side": "buy",
                    "qty": 1.5,
                    "price": 3200.00,
                    "time": "2026-01-11 10:30:00",
                },
                {
                    "side": "sell",
                    "qty": 0.5,
                    "price": 3250.00,
                    "time": "2026-01-11 12:15:00",
                },
            ],
        }

    bot.set_status_callback(get_test_status)

    # Start the bot
    print("\n🚀 Starting bot...")
    print("Bot will run in the background.")
    print("Try these commands in Telegram:")
    print("  /start - Start the bot")
    print("  /help - See available commands")
    print("  /status - See trading status")
    print("  /account - See account info")
    print("  /positions - See positions")
    print("  /trades - See recent trades")
    print("\nPress Ctrl+C to stop the bot\n")

    # Test sending a message first
    print("\n📤 Testing message sending...")
    result = bot.send_notification(
        "🧪 *Test Message*\n\nThis is a test from the Telegram bot test script."
    )
    if result:
        print("✅ Test message queued successfully!")
        print("   Check your Telegram bot for the test message.")
    else:
        print("❌ Failed to queue test message!")

    # Test status callback
    print("\n📊 Testing status callback...")
    if bot._status_callback:
        status = bot._status_callback()
        print("✅ Status callback works!")
        print(f"   Account equity: ${status.get('account', {}).get('equity', 0):,.2f}")
        positions = status.get("positions", [])
        print(f"   Positions: {len(positions)}")
    else:
        print("❌ Status callback not configured!")

    # Start the bot
    print("\n🚀 Starting bot...")
    try:
        if bot.start():
            print("✅ Bot started successfully!")
            print("Bot will run for 30 seconds to test command responses...")
            print("Try sending commands to your bot in Telegram now!\n")

            # Keep the main thread alive for 30 seconds
            import time

            start_time = time.time()
            while bot._running and (time.time() - start_time) < 30:
                time.sleep(1)

            print("\n⏰ Test period ended. Stopping bot...")
            bot.stop()
            print("Bot stopped.")
        else:
            print("❌ Failed to start bot!")
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping bot...")
        bot.stop()
        print("Bot stopped.")
    except Exception as e:
        print(f"\n❌ Error during bot operation: {e}")
        bot.stop()


if __name__ == "__main__":
    main()
