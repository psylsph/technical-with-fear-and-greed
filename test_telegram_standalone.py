#!/usr/bin/env python3
"""
Simple standalone Telegram bot to test message reception.
This isolates the Telegram polling functionality from the trading bot.
"""

import os
import asyncio
from pathlib import Path

# Load .env file
env_path = Path('.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Now import telegram after env is loaded
from telegram.ext import Application, CommandHandler, MessageHandler, filters

async def start_command(update, context):
    """Handle /start command."""
    await update.message.reply_text("🤖 Standalone Test Bot is running!")

async def help_command(update, context):
    """Handle /help command."""
    await update.message.reply_text("Available commands:\n/start - Start the bot\n/help - Show this help")

async def handle_message(update, context):
    """Handle any text message."""
    text = update.message.text
    chat_id = update.effective_chat.id
    user = update.effective_user.username or update.effective_user.first_name

    print(f"📨 RECEIVED: '{text}' from @{user} in chat {chat_id}")

    # Echo the message back
    response = f"🤖 Echo: {text}\n\nFrom: @{user}\nChat: {chat_id}"
    await update.message.reply_text(response)

async def main():
    """Main function to run the bot."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("❌ No TELEGRAM_BOT_TOKEN found in environment")
        return

    print(f"🚀 Starting standalone Telegram bot...")
    print(f"Bot Token: {token[:15]}...")

    # Create application
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Handlers added")
    print("🤖 Bot is ready! Send messages to test...")
    print("Will run for 30 seconds, then stop automatically...")

    # Start polling
    try:
        await application.initialize()
        await application.updater.initialize()
        await application.start()
        print("🔄 Polling started...")

        await application.updater.start_polling(drop_pending_updates=True)
        print("✅ Polling active!")
        print("⏰ Waiting 30 seconds for test messages...")

        # Keep running for 30 seconds
        await asyncio.sleep(30)
        print("⏰ Test period ended")

    except Exception as e:
        print(f"❌ Error during polling: {e}")
        import traceback
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n🛑 Stopping bot...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        print("👋 Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())