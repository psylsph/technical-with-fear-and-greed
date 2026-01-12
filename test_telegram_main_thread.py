#!/usr/bin/env python3
"""
Test script to verify Telegram bot commands work - with bot running in main thread.
"""

import os
import time
import json
import urllib.request
from pathlib import Path

# Load .env
env_path = Path('/home/stuart/repos/technical-with-fear-and-greed/.env')
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

print("=" * 60)
print("TELEGRAM BOT COMMAND TEST - MAIN THREAD")
print("=" * 60)

BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

from src.telegram_bot import TelegramBot

bot = TelegramBot()

# Set up status callback
def get_status():
    return {
        'account': {'equity': 10000, 'cash': 5000, 'pnl': 0, 'day_pnl': 0},
        'positions': [],
        'recent_trades': [],
    }
bot.set_status_callback(get_status)

# Get latest message ID before we start
print("Getting latest message ID...")
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?limit=1'
req = urllib.request.Request(url)
with urllib.request.urlopen(req, timeout=10) as response:
    data = json.loads(response.read().decode())
    if data.get('ok') and data.get('result'):
        last_msg_id = data['result'][-1].get('message', {}).get('message_id', 0)
        print(f"  Last message ID: {last_msg_id}")
    else:
        last_msg_id = 0
        print("  No previous messages")

print()
print("Starting bot (press Ctrl+C to stop)...")

# Start bot in main thread - run for limited time
import asyncio

async def run_bot_with_timeout():
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")

try:
    # Send a test command first
    print("\nSending /help command via API...")
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = json.dumps({
        'chat_id': CHAT_ID,
        'text': '/help',
    }).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode())
        if result.get('ok'):
            print(f"  Command sent (message {result['result']['message_id']})")

    # Run for 10 seconds to receive commands
    asyncio.run(asyncio.wait_for(run_bot_with_timeout(), timeout=10))
except asyncio.TimeoutError:
    print("\nTest timeout reached")

print()
print("Test complete - check your Telegram app for responses")
