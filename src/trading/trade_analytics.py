"""
Trade Analytics: Post-trade analysis and quality scoring.
Analyzes executed trades to measure performance, identify patterns, and improve strategy.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from ..config import PROJECT_ROOT


class TradeQuality(Enum):
    """Trade quality classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ExitReason(Enum):
    """Reasons for trade exit."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    TIME_EXIT = "time_exit"
    DAILY_LIMIT = "daily_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class Trade:
    """Represents a single completed trade."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    entry_cost: float
    exit_proceeds: float
    gross_profit: float
    fees: float
    net_profit: float
    net_profit_pct: float
    hold_duration_hours: float
    exit_reason: str
    quality: str
    max_favorable_adverse_move: float  # Max favorable move (profit), max adverse move (loss)
    max_drawdown_during_trade: float


class TradeAnalyzer:
    """
    Analyze completed trades to measure performance and identify patterns.

    Features:
    - Trade quality scoring based on entry timing and exit efficiency
    - Performance metrics by exit reason, time of day, day of week
    - Hold duration analysis
    - Win/loss ratio tracking
    - Slippage analysis
    """

    def __init__(
        self,
        trade_log_file: str = None,
        min_quality_score: float = 0.6,
    ):
        """
        Args:
            trade_log_file: Path to trade log JSON file
            min_quality_score: Minimum score for "good" quality trade
        """
        self.trade_log_file = trade_log_file or os.path.join(
            PROJECT_ROOT, "trade_log.json"
        )
        self.min_quality_score = min_quality_score

        self.trades: List[Trade] = []
        self._load_trades()

    def _load_trades(self) -> None:
        """Load trades from log file."""
        if os.path.exists(self.trade_log_file):
            try:
                with open(self.trade_log_file) as f:
                    data = json.load(f)

                for trade_data in data.get("trades", []):
                    trade = Trade(**trade_data)
                    self.trades.append(trade)
            except Exception as e:
                print(f"Error loading trades: {e}")

    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        entry_time: str,
        exit_time: str,
        quantity: float,
        entry_cost: float,
        exit_proceeds: float,
        fees: float,
        exit_reason: str = ExitReason.UNKNOWN.value,
        price_history: List[float] = None,
    ) -> Trade:
        """
        Log a completed trade.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp (ISO format)
            exit_time: Exit timestamp (ISO format)
            quantity: Position size
            entry_cost: Total cost including fees
            exit_proceeds: Total proceeds minus fees
            fees: Total fees paid
            exit_reason: Reason for exit
            price_history: Optional list of prices during trade (for MAE/MFA analysis)

        Returns:
            Trade object
        """
        # Calculate profit metrics
        if side == "long":
            gross_profit = (exit_price - entry_price) * quantity
        else:  # short
            gross_profit = (entry_price - exit_price) * quantity

        net_profit = gross_profit - fees
        net_profit_pct = (net_profit / entry_cost) * 100 if entry_cost > 0 else 0

        # Calculate hold duration
        entry_dt = datetime.fromisoformat(entry_time)
        exit_dt = datetime.fromisoformat(exit_time)
        hold_duration_hours = (exit_dt - entry_dt).total_seconds() / 3600

        # Calculate max favorable/adverse moves if price history provided
        max_favorable_move = 0.0
        max_adverse_move = 0.0
        max_drawdown = 0.0

        if price_history:
            if side == "long":
                # For long: favorable = price went up, adverse = price went down
                max_favorable_move = max(price_history) - entry_price
                max_adverse_move = entry_price - min(price_history)

                # Max drawdown from peak
                cumulative_max = 0
                current_dd = 0
                for price in price_history:
                    pnl = (price - entry_price) / entry_price
                    cumulative_max = max(cumulative_max, pnl)
                    current_dd = pnl - cumulative_max
                    max_drawdown = min(max_drawdown, current_dd)
                max_drawdown = abs(max_drawdown) * 100
            else:  # short
                # For short: favorable = price went down, adverse = price went up
                max_favorable_move = entry_price - min(price_history)
                max_adverse_move = max(price_history) - entry_price

        # Calculate trade quality score
        quality = self._calculate_quality(
            net_profit_pct,
            hold_duration_hours,
            max_favorable_move,
            max_adverse_move,
            exit_reason,
        )

        trade = Trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            quantity=quantity,
            entry_cost=entry_cost,
            exit_proceeds=exit_proceeds,
            gross_profit=gross_profit,
            fees=fees,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            hold_duration_hours=hold_duration_hours,
            exit_reason=exit_reason,
            quality=quality.value,
            max_favorable_adverse_move=max_favorable_move,
            max_drawdown_during_trade=max_drawdown,
        )

        self.trades.append(trade)
        self._save_trades()

        return trade

    def _calculate_quality(
        self,
        net_profit_pct: float,
        hold_hours: float,
        max_favorable: float,
        max_adverse: float,
        exit_reason: str,
    ) -> TradeQuality:
        """
        Calculate trade quality score.

        Args:
            net_profit_pct: Net profit as percentage
            hold_hours: Hours position was held
            max_favorable: Maximum favorable price move
            max_adverse: Maximum adverse price move
            exit_reason: Reason for exit

        Returns:
            TradeQuality enum
        """
        score = 0.0

        # Profitability score (40% weight)
        if net_profit_pct > 10:
            score += 0.4
        elif net_profit_pct > 5:
            score += 0.3
        elif net_profit_pct > 2:
            score += 0.2
        elif net_profit_pct > 0:
            score += 0.1
        elif net_profit_pct > -2:
            score += 0.0
        else:
            score -= 0.2

        # Exit efficiency score (30% weight)
        if exit_reason == ExitReason.TAKE_PROFIT.value:
            score += 0.3
        elif exit_reason == ExitReason.TRAILING_STOP.value:
            score += 0.2
        elif exit_reason == ExitReason.SIGNAL_REVERSAL.value:
            score += 0.1
        elif exit_reason == ExitReason.STOP_LOSS.value:
            score += 0.0
        elif exit_reason == ExitReason.TIME_EXIT.value:
            score -= 0.1
        else:
            score += 0.0

        # Favorable move capture score (20% weight)
        if max_favorable > 0:
            capture_ratio = net_profit_pct / ((max_favorable / max_favorable if max_favorable > 0 else 1) * 100)
            if capture_ratio > 0.8:
                score += 0.2
            elif capture_ratio > 0.5:
                score += 0.15
            elif capture_ratio > 0.3:
                score += 0.1
            elif capture_ratio > 0.1:
                score += 0.05

        # Adverse move management score (10% weight)
        entry_price_proxy = 1.0
        if max_adverse > 0 and entry_price_proxy > 0:
            adverse_pct = (max_adverse / entry_price_proxy) * 100
            if adverse_pct < 2:
                score += 0.1
            elif adverse_pct < 5:
                score += 0.05

        # Duration score (bonus for quick profits, penalty for long holds)
        if net_profit_pct > 0 and hold_hours < 24:
            score += 0.05  # Bonus for quick profitable trades
        elif hold_hours > 720:  # 30 days
            score -= 0.1

        # Convert to quality enum
        if score >= 0.8:
            return TradeQuality.EXCELLENT
        elif score >= 0.6:
            return TradeQuality.GOOD
        elif score >= 0.4:
            return TradeQuality.FAIR
        else:
            return TradeQuality.POOR

    def _save_trades(self) -> None:
        """Save trades to log file."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_trades": len(self.trades),
            "trades": [asdict(t) for t in self.trades],
        }

        with open(self.trade_log_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_performance_metrics(self, days: int = 30) -> Dict:
        """
        Calculate overall performance metrics.

        Args:
            days: Number of days to analyze (default 30)

        Returns:
            Dict with performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
            }

        # Filter trades by date
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t.exit_time) > cutoff
        ]

        if not recent_trades:
            recent_trades = self.trades

        winning_trades = [t for t in recent_trades if t.net_profit > 0]
        losing_trades = [t for t in recent_trades if t.net_profit < 0]

        total_trades = len(recent_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_profit_pct = (
            np.mean([t.net_profit_pct for t in winning_trades])
            if winning_trades else 0
        )
        avg_loss_pct = (
            np.mean([t.net_profit_pct for t in losing_trades])
            if losing_trades else 0
        )

        gross_profit = sum(t.gross_profit for t in winning_trades)
        gross_loss = abs(sum(t.gross_profit for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_hold_hours = (
            np.mean([t.hold_duration_hours for t in recent_trades])
            if recent_trades else 0
        )

        max_profit = max([t.net_profit_pct for t in recent_trades]) if recent_trades else 0
        max_loss = min([t.net_profit_pct for t in recent_trades]) if recent_trades else 0

        return {
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_profit_pct": avg_profit_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "avg_hold_hours": avg_hold_hours,
            "max_profit_pct": max_profit,
            "max_loss_pct": max_loss,
            "total_net_profit": sum(t.net_profit for t in recent_trades),
            "total_fees": sum(t.fees for t in recent_trades),
        }

    def get_metrics_by_exit_reason(self, days: int = 90) -> Dict:
        """
        Analyze performance by exit reason.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with metrics by exit reason
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t.exit_time) > cutoff
        ]

        metrics_by_reason = {}
        for reason in ExitReason:
            reason_trades = [
                t for t in recent_trades
                if t.exit_reason == reason.value
            ]

            if reason_trades:
                winning = [t for t in reason_trades if t.net_profit > 0]
                win_rate = len(winning) / len(reason_trades)

                avg_profit = (
                    np.mean([t.net_profit_pct for t in reason_trades])
                    if reason_trades else 0
                )

                metrics_by_reason[reason.value] = {
                    "count": len(reason_trades),
                    "win_rate": win_rate,
                    "avg_profit_pct": avg_profit,
                    "total_profit": sum(t.net_profit for t in reason_trades),
                }

        return metrics_by_reason

    def get_metrics_by_time_period(
        self, days: int = 30
    ) -> Dict:
        """
        Analyze performance by time period (hour of day, day of week).

        Args:
            days: Number of days to analyze

        Returns:
            Dict with metrics by time period
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t.exit_time) > cutoff
        ]

        # By hour of day
        by_hour = {}
        for hour in range(24):
            hour_trades = [
                t for t in recent_trades
                if datetime.fromisoformat(t.exit_time).hour == hour
            ]
            if hour_trades:
                by_hour[hour] = {
                    "count": len(hour_trades),
                    "win_rate": (
                        sum(1 for t in hour_trades if t.net_profit > 0) / len(hour_trades)
                    ),
                    "avg_profit_pct": np.mean([t.net_profit_pct for t in hour_trades]),
                }

        # By day of week
        by_day = {}
        days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days_map):
            day_trades = [
                t for t in recent_trades
                if datetime.fromisoformat(t.exit_time).weekday() == i
            ]
            if day_trades:
                by_day[day] = {
                    "count": len(day_trades),
                    "win_rate": (
                        sum(1 for t in day_trades if t.net_profit > 0) / len(day_trades)
                    ),
                    "avg_profit_pct": np.mean([t.net_profit_pct for t in day_trades]),
                }

        return {
            "by_hour": by_hour,
            "by_day": by_day,
        }

    def get_quality_distribution(self) -> Dict:
        """Get distribution of trade quality scores."""
        if not self.trades:
            return {}

        quality_counts = {}
        for quality in TradeQuality:
            count = sum(1 for t in self.trades if t.quality == quality.value)
            quality_counts[quality.value] = {
                "count": count,
                "pct": count / len(self.trades) if self.trades else 0,
            }

        return quality_counts

    def get_analytics_report(self, days: int = 30) -> str:
        """
        Generate comprehensive analytics report.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted report string
        """
        metrics = self.get_performance_metrics(days)
        by_reason = self.get_metrics_by_exit_reason(days)
        by_time = self.get_metrics_by_time_period(days)
        quality_dist = self.get_quality_distribution()

        report = f"Trade Analytics Report (Last {days} Days)\n"
        report += f"{'='*60}\n\n"

        # Overall performance
        report += "Overall Performance:\n"
        report += f"  Total Trades: {metrics['total_trades']}\n"
        report += f"  Win Rate: {metrics['win_rate']:.1%}\n"
        report += f"  Avg Profit: {metrics['avg_profit_pct']:.2f}%\n"
        report += f"  Avg Loss: {metrics['avg_loss_pct']:.2f}%\n"
        report += f"  Profit Factor: {metrics['profit_factor']:.2f}\n"
        report += f"  Avg Hold: {metrics['avg_hold_hours']:.1f} hours\n"
        report += f"  Max Profit: {metrics['max_profit_pct']:.2f}%\n"
        report += f"  Max Loss: {metrics['max_loss_pct']:.2f}%\n"
        report += f"  Net Profit: ${metrics['total_net_profit']:.2f}\n"
        report += f"  Total Fees: ${metrics['total_fees']:.2f}\n\n"

        # Quality distribution
        report += "Trade Quality Distribution:\n"
        for quality, data in quality_dist.items():
            emoji = {
                "excellent": "⭐",
                "good": "✅",
                "fair": "🟠",
                "poor": "🔴",
            }.get(quality, "•")
            report += f"  {emoji} {quality.title()}: {data['count']} ({data['pct']:.1%})\n"
        report += "\n"

        # By exit reason
        if by_reason:
            report += "Performance by Exit Reason:\n"
            for reason, data in sorted(
                by_reason.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            ):
                report += (
                    f"  {reason}: {data['count']} trades, "
                    f"Win Rate: {data['win_rate']:.1%}, "
                    f"Avg: {data['avg_profit_pct']:.2f}%\n"
                )
            report += "\n"

        # Best/worst hours
        if by_time["by_hour"]:
            sorted_hours = sorted(
                by_time["by_hour"].items(),
                key=lambda x: x[1]["avg_profit_pct"],
                reverse=True
            )
            best_hour = sorted_hours[0]
            worst_hour = sorted_hours[-1]

            report += f"Best Trading Hour: {best_hour[0]:02d}:00 "
            report += f"(Win: {best_hour[1]['win_rate']:.1%}, Avg: {best_hour[1]['avg_profit_pct']:.2f}%)\n"
            report += f"Worst Trading Hour: {worst_hour[0]:02d}:00 "
            report += f"(Win: {worst_hour[1]['win_rate']:.1%}, Avg: {worst_hour[1]['avg_profit_pct']:.2f}%)\n\n"

        return report

    def export_trades_csv(self, output_file: str = None) -> str:
        """
        Export trades to CSV format.

        Args:
            output_file: Optional output file path

        Returns:
            Path to exported CSV file
        """
        if output_file is None:
            output_file = os.path.join(PROJECT_ROOT, "trades_export.csv")

        # Convert trades to DataFrame
        df = pd.DataFrame([asdict(t) for t in self.trades])

        df.to_csv(output_file, index=False)
        return output_file


def analyze_trade_log(
    trade_log_file: str = None,
    days: int = 30
) -> str:
    """
    Convenience function to analyze trade log and generate report.

    Args:
        trade_log_file: Path to trade log file
        days: Number of days to analyze

    Returns:
        Analytics report string
    """
    analyzer = TradeAnalyzer(trade_log_file=trade_log_file)
    return analyzer.get_analytics_report(days=days)
