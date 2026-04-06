#!/usr/bin/env python
"""Test Telegram alerts"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.alerts import TelegramAlert, BetAlert, AlertPriority
from datetime import datetime, timedelta


async def test_telegram():
    """Test all alert types"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        print("   Please add them to your .env file")
        return

    print(f"🤖 Bot Token: {bot_token[:15]}...")
    print(f"👤 Chat ID: {chat_id}")

    alerts = TelegramAlert(bot_token, chat_id, enabled=True)

    print("\n📤 Testing Telegram alerts...")
    print("-" * 40)

    # Test simple message
    result = await alerts.send_message("✅ VIT Bot is working! This is a test message.", AlertPriority.SUCCESS)
    print(f"Simple message: {'✅ SENT' if result else '❌ FAILED'}")

    await asyncio.sleep(2)

    # Test bet alert
    bet_alert = BetAlert(
        match_id=12345,
        home_team="Manchester City",
        away_team="Liverpool",
        prediction="Home Win",
        probability=0.62,
        edge=0.084,
        stake=0.035,
        odds=1.85,
        confidence=0.78,
        kickoff_time=datetime.now() + timedelta(days=2)
    )

    result2 = await alerts.send_bet_alert(bet_alert)
    print(f"Bet alert: {'✅ SENT' if result2 else '❌ FAILED'}")

    await asyncio.sleep(2)

    # Test daily report
    stats = {
        'total_bets': 42,
        'winning_bets': 26,
        'losing_bets': 16,
        'win_rate': 0.619,
        'roi': 0.084,
        'avg_clv': 0.032,
        'bankroll': 10840.00,
        'model_accuracy': 0.65,
        'avg_confidence': 0.72
    }

    top_edges = [
        {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'edge': 0.072},
        {'home_team': 'Man City', 'away_team': 'Liverpool', 'edge': 0.065},
    ]

    result3 = await alerts.send_daily_report(stats, top_edges)
    print(f"Daily report: {'✅ SENT' if result3 else '❌ FAILED'}")

    print("\n" + "-" * 40)
    print("✅ Telegram test complete! Check your Telegram bot for messages.")
    print("\n📱 If you didn't receive messages:")
    print("   1. Make sure you started a chat with @vit_betting_bot")
    print("   2. Check that your CHAT_ID is correct")
    print("   3. Verify the bot token is correct")


if __name__ == "__main__":
    asyncio.run(test_telegram())