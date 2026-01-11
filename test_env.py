#!/usr/bin/env python3
"""
Test script to verify environment variables are loaded correctly in Docker.
Run this to check if .env file is being loaded properly.
"""

import os
from pathlib import Path


def main():
    print("🔍 Environment Variables Test")
    print("=" * 50)

    # Check if .env file exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print(f"✅ .env file exists at: {env_path}")
        with open(env_path) as f:
            lines = f.readlines()
            print(f"📄 .env file has {len(lines)} lines")
    else:
        print("❌ .env file not found!")

    # Check critical environment variables
    critical_vars = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ]

    print("\n🔑 Critical Environment Variables:")
    print("-" * 40)

    all_present = True
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            # Show first/last few characters for security
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else value
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: NOT SET")
            all_present = False

    # Check Python path
    python_path = os.getenv("PYTHONPATH", "")
    print(f"\n🐍 PYTHONPATH: {python_path}")

    # Check working directory
    cwd = os.getcwd()
    print(f"📁 Working Directory: {cwd}")

    if all_present:
        print("\n🎉 All critical environment variables are loaded!")
        return True
    else:
        print("\n⚠️  Some environment variables are missing!")
        print("   Make sure your .env file contains all required variables.")
        return False


if __name__ == "__main__":
    main()
