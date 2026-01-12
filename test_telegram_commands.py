#!/usr/bin/env python3
"""
Test script to verify Telegram bot commands work.
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
print("TELEGRAM BOT COMMAND TEST")
print("=" * 60)

BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Get the latest message ID before we start
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
print("Starting bot...")
from src.telegram_bot import get_telegram_bot

bot = get_telegram_bot()

# Set up status callback
def get_status():
    return {
        'account': {'equity': 10000, 'cash': 5000, 'pnl': 0, 'day_pnl': 0},
        'positions': [],
        'recent_trades': [],
    }
bot.set_status_callback(get_status)

bot.start()
print("  Bot started")

# Wait for startup notification
time.sleep(2)

print()
print("Sending /help command via API...")
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

print()
print("Waiting 5 seconds for bot to respond...")
time.sleep(5)

print()
print("Checking for NEW messages from bot...")
# Only get messages AFTER the last known message
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_msg_id + 1}&limit=10'
req = urllib.request.Request(url)
with urllib.request.urlopen(req, timeout=10) as response:
    data = json.loads(response.read().decode())
    if data.get('ok'):
        updates = data.get('result', [])
        
        # Find messages FROM the bot
        bot_messages = [u.get('message', {}) for u in updates if u.get('message', {}).get('from', {}).get('is_bot')]
        
        if bot_messages:
            print(f"  Found {len(bot_messages)} messages from bot:")
            for msg in bot_messages:
                text = msg.get('text', '')[:100].replace('\n', ' ')
                msg_id = msg.get('message_id', 'N/A')
                print(f"    ID {msg_id}: {text}")
            
            # Check if help message is there
            help_msg = [m for m in bot_messages if 'Commands' in m.get('text', '') or 'help' in m.get('text', '').lower()]
            if help_msg:
                print()
                print("  ✅ HELP COMMAND WORKED!")
        else:
            print("  ❌ NO MESSAGES FROM BOT!")
            print()
            print("  All new updates:")
            for u in updates:
                msg = u.get('message', {})
                text = msg.get('text', 'N/A')
                from_user = msg.get('from', {}).get('username', 'N/A')
                is_bot = msg.get('from', {}).get('is_bot', False)
                print(f"    [{text}] from @{from_user} (is_bot={is_bot})")
    else:
        print(f"  Error: {data.get('description')}")

bot.stop()
print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
