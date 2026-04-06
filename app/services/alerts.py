# app/services/alerts.py
"""
Telegram Alert System - Real-time betting notifications.

Features:
- Send bet recommendations when edge > threshold
- Daily performance reports
- Critical anomaly alerts
- Match result notifications
"""

import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    CRITICAL = "🚨"
    BET = "🎯"


@dataclass
class BetAlert:
    """Bet recommendation alert data"""
    match_id: int
    home_team: str
    away_team: str
    prediction: str
    probability: float
    edge: float
    stake: float
    odds: float
    confidence: float
    kickoff_time: datetime


class TelegramAlert:
    """
    Telegram bot for real-time alerts.

    Setup:
    1. Create a bot with @BotFather on Telegram
    2. Get your bot token
    3. Get your chat ID (send a message to @userinfobot)
    4. Add to .env
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._last_message_time = None
        self._rate_limit_remaining = 20  # Telegram allows ~20 messages per minute

    async def send_message(
        self,
        text: str,
        priority: AlertPriority = AlertPriority.INFO,
        parse_mode: str = "HTML"
    ) -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            logger.debug("Telegram alerts disabled")
            return False

        # Rate limiting
        if self._last_message_time:
            elapsed = (datetime.now() - self._last_message_time).total_seconds()
            if elapsed < 3:  # Max 20 per minute = 1 every 3 seconds
                logger.warning("Rate limit hit, skipping message")
                return False

        formatted_text = f"{priority.value} {text}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": formatted_text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True
                    }
                )

                if response.status_code == 200:
                    self._last_message_time = datetime.now()
                    logger.info(f"Telegram message sent: {text[:50]}...")
                    return True
                else:
                    logger.error(f"Telegram error: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_bet_alert(self, alert: BetAlert) -> bool:
        """Send bet recommendation alert"""
        # Format kickoff time
        kickoff_str = alert.kickoff_time.strftime("%Y-%m-%d %H:%M UTC") if alert.kickoff_time else "TBD"

        # Determine emoji based on edge strength
        if alert.edge > 0.08:
            edge_emoji = "🔥🔥🔥"
        elif alert.edge > 0.05:
            edge_emoji = "🔥🔥"
        elif alert.edge > 0.02:
            edge_emoji = "🔥"
        else:
            edge_emoji = "📊"

        message = f"""
<b>🎯 VIT BET RECOMMENDATION</b>
━━━━━━━━━━━━━━━━━━━━━

<b>⚽ Match:</b> {alert.home_team} vs {alert.away_team}
<b>🕐 Kickoff:</b> {kickoff_str}

<b>📊 Prediction:</b> <b>{alert.prediction.upper()}</b>
<b>📈 Probability:</b> {alert.probability:.1%}
<b>💰 Edge:</b> {alert.edge:.2%} {edge_emoji}
<b>🎲 Odds:</b> {alert.odds:.2f}
<b>💵 Stake:</b> {alert.stake:.1%} of bankroll
<b>🎯 Confidence:</b> {alert.confidence:.0%}

<i>VIT Sports Intelligence Network</i>
        """

        return await self.send_message(message.strip(), AlertPriority.BET)

    async def send_daily_report(
        self,
        stats: Dict[str, Any],
        top_edges: List[Dict] = None
    ) -> bool:
        """Send daily performance report"""
        date = datetime.now().strftime("%Y-%m-%d")

        # Determine performance emoji
        roi = stats.get('roi', 0)
        if roi > 0.05:
            performance_emoji = "📈🚀"
        elif roi > 0:
            performance_emoji = "📈"
        elif roi > -0.05:
            performance_emoji = "📉"
        else:
            performance_emoji = "📉💀"

        message = f"""
<b>📊 VIT DAILY REPORT</b>
━━━━━━━━━━━━━━━━━━━━━

<b>📅 Date:</b> {date}
<b>{performance_emoji} Performance:</b>

<b>💰 Total Bets:</b> {stats.get('total_bets', 0)}
<b>✅ Winning Bets:</b> {stats.get('winning_bets', 0)}
<b>❌ Losing Bets:</b> {stats.get('losing_bets', 0)}
<b>📊 Win Rate:</b> {stats.get('win_rate', 0):.1%}
<b>💵 ROI:</b> {stats.get('roi', 0):.2%}
<b>📈 CLV:</b> {stats.get('avg_clv', 0):.4f}
<b>💼 Bankroll:</b> ${stats.get('bankroll', 0):.2f}

<b>📊 Model Health:</b>
<b>🎯 Accuracy:</b> {stats.get('model_accuracy', 0):.1%}
<b>⚡ Confidence:</b> {stats.get('avg_confidence', 0):.1%}
        """

        # Add top edges if provided
        if top_edges:
            message += "\n<b>🔥 Top Edges Today:</b>\n"
            for edge in top_edges[:3]:
                message += f"• {edge.get('home_team')} vs {edge.get('away_team')}: {edge.get('edge', 0):.2%} edge\n"

        message += "\n<i>VIT Sports Intelligence Network</i>"

        return await self.send_message(message.strip())

    async def send_match_result(
        self,
        match_id: int,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        was_correct: bool,
        profit: float
    ) -> bool:
        """Send match result notification"""
        result_emoji = "✅" if was_correct else "❌"
        score_str = f"{home_goals} - {away_goals}"

        message = f"""
<b>{result_emoji} MATCH RESULT</b>
━━━━━━━━━━━━━━━━━━━━━

<b>⚽ Match:</b> {home_team} vs {away_team}
<b>📊 Score:</b> {score_str}
<b>🎯 Prediction:</b> {'CORRECT' if was_correct else 'INCORRECT'}
<b>💰 Profit/Loss:</b> ${profit:.2f}
        """

        return await self.send_message(message.strip())

    async def send_anomaly_alert(
        self,
        anomaly_type: str,
        details: Dict[str, Any],
        severity: str = "warning"
    ) -> bool:
        """Send anomaly detection alert"""
        priority = AlertPriority.CRITICAL if severity == "critical" else AlertPriority.WARNING

        message = f"""
<b>⚠️ ANOMALY DETECTED</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Type:</b> {anomaly_type}
<b>Severity:</b> {severity.upper()}

<b>Details:</b>
"""
        for key, value in details.items():
            message += f"• {key}: {value}\n"

        message += "\n<i>Action may be required</i>"

        return await self.send_message(message.strip(), priority)

    async def send_model_performance_alert(
        self,
        model_name: str,
        old_weight: float,
        new_weight: float,
        reason: str
    ) -> bool:
        """Send model weight change alert"""
        direction = "⬆️" if new_weight > old_weight else "⬇️"

        message = f"""
<b>🤖 MODEL WEIGHT UPDATE</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Model:</b> {model_name}
<b>Weight:</b> {old_weight:.2%} → {new_weight:.2%} {direction}
<b>Reason:</b> {reason}

<i>Automatic weight decay applied</i>
        """

        return await self.send_message(message.strip())

    async def send_startup_message(self) -> bool:
        """Send system startup notification"""
        message = f"""
<b>🚀 VIT NETWORK STARTED</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Status:</b> OPERATIONAL
<b>Alerts:</b> ENABLED

<i>Monitoring for betting opportunities...</i>
        """

        return await self.send_message(message.strip())

    async def send_shutdown_message(self) -> bool:
        """Send system shutdown notification"""
        message = f"""
<b>🛑 VIT NETWORK SHUTDOWN</b>
━━━━━━━━━━━━━━━━━━━━━

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>System stopped. No alerts will be sent.</i>
        """

        return await self.send_message(message.strip())