"""
Email Notifications: Send trade alerts via email.
Configured to work with Gmail, Outlook, or any SMTP server.
"""

import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List

from ..config import PROJECT_ROOT


class EmailNotifier:
    """Send email notifications for trading events."""

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        sender_email: str = None,
        sender_password: str = None,
        enabled: bool = True
    ):
        """
        Args:
            smtp_server: SMTP server address (default: smtp.gmail.com)
            smtp_port: SMTP port (default: 587 for TLS)
            sender_email: Sender email address
            sender_password: Sender email password or app password
            enabled: Whether notifications are enabled
        """
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")
        self.enabled = enabled and bool(self.sender_email and self.sender_password)

        # Load recipient list
        self.recipients = self._load_recipients()

        # Notification history
        self.notification_log_file = os.path.join(PROJECT_ROOT, "notification_log.json")
        self.notification_log = self._load_notification_log()

    def _load_recipients(self) -> List[str]:
        """Load recipient email addresses from file or env."""
        # First check environment variable
        env_recipients = os.getenv("ALERT_EMAIL_RECIPIENTS")
        if env_recipients:
            return [email.strip() for email in env_recipients.split(",")]

        # Then check file
        recipient_file = os.path.join(PROJECT_ROOT, "alert_recipients.txt")
        if os.path.exists(recipient_file):
            try:
                with open(recipient_file) as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception:
                pass

        return []

    def _load_notification_log(self) -> List[Dict]:
        """Load notification history."""
        if os.path.exists(self.notification_log_file):
            try:
                import json
                with open(self.notification_log_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_notification_log(self):
        """Save notification history."""
        import json
        # Keep only last 100 notifications
        log = self.notification_log[-100:]
        with open(self.notification_log_file, "w") as f:
            json.dump(log, f, indent=2)

    def _send_email(
        self,
        subject: str,
        body: str,
        html_body: str = None
    ) -> bool:
        """
        Send email via SMTP.

        Args:
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body

        Returns:
            True if email sent successfully
        """
        if not self.enabled or not self.recipients:
            return False

        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipients)

            # Add plain text part
            text_part = MIMEText(body, "plain")
            message.attach(text_part)

            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                message.attach(html_part)

            # Connect to SMTP server and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipients, message.as_string())

            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def _log_notification(self, notification_type: str, data: dict):
        """Log notification to history."""
        self.notification_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": notification_type,
            "data": data,
            "recipients": self.recipients,
        })
        self._save_notification_log()

    def send_trade_alert(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        reason: str = None
    ) -> bool:
        """
        Send trade execution alert.

        Args:
            symbol: Trading symbol
            action: "buy" or "sell"
            qty: Quantity traded
            price: Execution price
            reason: Optional reason for trade

        Returns:
            True if alert sent successfully
        """
        emoji = "🟢" if action == "buy" else "🔴"
        subject = f"{emoji} {action.upper()} Alert: {symbol}"

        body = f"""
Trade Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Action: {action.upper()}
Symbol: {symbol}
Quantity: {qty:.6f}
Price: ${price:.2f}
Reason: {reason or "N/A"}

---
This is an automated message from your trading system.
"""

        html_body = f"""
<html>
<body>
    <h2>{emoji} {action.upper()} Alert: {symbol}</h2>
    <p><strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <table border="1" cellpadding="5" style="border-collapse: collapse;">
        <tr>
            <td><strong>Action</strong></td>
            <td>{action.upper()}</td>
        </tr>
        <tr>
            <td><strong>Symbol</strong></td>
            <td>{symbol}</td>
        </tr>
        <tr>
            <td><strong>Quantity</strong></td>
            <td>{qty:.6f}</td>
        </tr>
        <tr>
            <td><strong>Price</strong></td>
            <td>${price:.2f}</td>
        </tr>
        <tr>
            <td><strong>Reason</strong></td>
            <td>{reason or "N/A"}</td>
        </tr>
    </table>

    <p><em>This is an automated message from your trading system.</em></p>
</body>
</html>
"""

        success = self._send_email(subject, body, html_body)

        self._log_notification("trade_alert", {
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": price,
            "reason": reason,
            "sent": success,
        })

        return success

    def send_signal_alert(
        self,
        symbol: str,
        signal: str,
        price: float,
        indicators: dict = None
    ) -> bool:
        """
        Send trading signal alert.

        Args:
            symbol: Trading symbol
            signal: "buy", "sell", or "hold"
            price: Current price
            indicators: Optional indicator values

        Returns:
            True if alert sent successfully
        """
        emoji = "🟢" if signal == "buy" else "🔴" if signal == "sell" else "⚪"
        subject = f"{emoji} Signal Alert: {signal.upper()} {symbol}"

        body = f"""
Signal Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Signal: {signal.upper()}
Symbol: {symbol}
Price: ${price:.2f}
"""

        if indicators:
            body += "\nIndicators:\n"
            for key, value in indicators.items():
                body += f"  {key}: {value}\n"

        body += "\n---\nThis is an automated message from your trading system."

        success = self._send_email(subject, body)

        self._log_notification("signal_alert", {
            "symbol": symbol,
            "signal": signal,
            "price": price,
            "indicators": indicators,
            "sent": success,
        })

        return success

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: dict = None
    ) -> bool:
        """
        Send risk management alert.

        Args:
            alert_type: Type of risk alert (e.g., "stop_loss", "daily_limit")
            message: Alert message
            details: Optional additional details

        Returns:
            True if alert sent successfully
        """
        subject = f"⚠️ Risk Alert: {alert_type}"

        body = f"""
Risk Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Type: {alert_type}
Message: {message}
"""

        if details:
            body += "\nDetails:\n"
            for key, value in details.items():
                body += f"  {key}: {value}\n"

        body += "\n---\nThis is an automated message from your trading system."

        success = self._send_email(subject, body)

        self._log_notification("risk_alert", {
            "type": alert_type,
            "message": message,
            "details": details,
            "sent": success,
        })

        return success

    def send_daily_summary(
        self,
        summary: dict
    ) -> bool:
        """
        Send daily trading summary.

        Args:
            summary: Dict with daily stats (pnl, trades, etc.)

        Returns:
            True if alert sent successfully
        """
        subject = f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"

        pnl = summary.get("daily_pnl_pct", 0)
        emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

        body = f"""
Daily Trading Summary - {datetime.now().strftime("%Y-%m-%d")}

{emoji} Daily P&L: {pnl:.2%}
Trades: {summary.get("num_trades", 0)}
Final Equity: ${summary.get("final_equity", 0):,.2f}
"""

        if summary.get("trades_today"):
            body += "\nTrades:\n"
            for trade in summary.get("trades_today", []):
                body += f"  {trade}\n"

        body += "\n---\nThis is an automated message from your trading system."

        success = self._send_email(subject, body)

        self._log_notification("daily_summary", {
            "summary": summary,
            "sent": success,
        })

        return success

    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: dict = None
    ) -> bool:
        """
        Send error alert.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional context information

        Returns:
            True if alert sent successfully
        """
        subject = f"🚨 Error Alert: {error_type}"

        body = f"""
Error Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Type: {error_type}
Message: {error_message}
"""

        if context:
            body += "\nContext:\n"
            for key, value in context.items():
                body += f"  {key}: {value}\n"

        body += "\n---\nThis is an automated message from your trading system."

        success = self._send_email(subject, body)

        self._log_notification("error_alert", {
            "type": error_type,
            "message": error_message,
            "context": context,
            "sent": success,
        })

        return success

    def is_enabled(self) -> bool:
        """Check if email notifications are enabled."""
        return self.enabled and len(self.recipients) > 0

    def add_recipient(self, email: str):
        """Add a recipient to the notification list."""
        if email not in self.recipients:
            self.recipients.append(email)
            # Save to file
            recipient_file = os.path.join(PROJECT_ROOT, "alert_recipients.txt")
            with open(recipient_file, "w") as f:
                for recipient in self.recipients:
                    f.write(f"{recipient}\n")

    def remove_recipient(self, email: str):
        """Remove a recipient from the notification list."""
        if email in self.recipients:
            self.recipients.remove(email)
            # Update file
            recipient_file = os.path.join(PROJECT_ROOT, "alert_recipients.txt")
            with open(recipient_file, "w") as f:
                for recipient in self.recipients:
                    f.write(f"{recipient}\n")

    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get recent notification history."""
        return self.notification_log[-limit:] if self.notification_log else []
