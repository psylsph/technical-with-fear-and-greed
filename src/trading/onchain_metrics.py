"""
On-Chain Metrics: Track exchange flows, whale activity, and blockchain data.
Provides on-chain analysis for cryptocurrency trading decisions.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import requests

from ..config import PROJECT_ROOT


class MetricType(Enum):
    """Types of on-chain metrics."""
    EXCHANGE_FLOW = "exchange_flow"
    WHALE_TRANSACTION = "whale_transaction"
    ACTIVE_ADDRESSES = "active_addresses"
    TRANSACTION_COUNT = "transaction_count"
    MVRV_RATIO = "mvrv_ratio"
    NVT_RATIO = "nvt_ratio"
    EXCHANGE_RESERVE = "exchange_reserve"
    LONG_SHORT_RATIO = "long_short_ratio"
    LIQUIDATION = "liquidation"


class FlowDirection(Enum):
    """Direction of exchange flow."""
    INFLOW = "inflow"  # Into exchange (bearish signal)
    OUTFLOW = "outflow"  # Out of exchange (bullish signal)


class WhaleCategory(Enum):
    """Categories of whale transactions."""
    EXCHANGE_TO_WALLET = "exchange_to_wallet"  # Accumulation (bullish)
    WALLET_TO_EXCHANGE = "wallet_to_exchange"  # Distribution (bearish)
    WALLET_TO_WALLET = "wallet_to_wallet"  # Redistribution
    UNKNOWN = "unknown"


@dataclass
class ExchangeFlow:
    """Exchange flow data point."""
    timestamp: str
    exchange: str
    symbol: str
    direction: FlowDirection
    amount_usd: float
    amount_tokens: float
    tx_count: int


@dataclass
class WhaleTransaction:
    """Large whale transaction."""
    timestamp: str
    symbol: str
    category: WhaleCategory
    amount_usd: float
    amount_tokens: float
    from_address: str
    to_address: str
    tx_hash: str


@dataclass
class OnChainMetric:
    """Generic on-chain metric data point."""
    timestamp: str
    symbol: str
    metric_type: MetricType
    value: float
    normalized_value: float  # Z-score or percentile
    signal: str  # "bullish", "bearish", "neutral"


class OnChainDataFetcher:
    """
    Fetch on-chain data from various sources.

    Note: This is a framework implementation. Real on-chain data requires
    API subscriptions from providers like:
    - Glassnode
    - CryptoQuant
    - CoinMetrics
    - Messari
    - Whale Alert
    """

    def __init__(
        self,
        api_key: str = None,
        cache_dir: str = None,
    ):
        """
        Args:
            api_key: API key for on-chain data provider
            cache_dir: Directory for caching data
        """
        self.api_key = api_key or os.environ.get("ONCHAIN_API_KEY")
        self.cache_dir = cache_dir or os.path.join(
            PROJECT_ROOT, "cache", "onchain"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Base URLs for different providers (would need real endpoints)
        self.endpoints = {
            "glassnode": "https://api.glassnode.com/v1",
            "cryptoquant": "https://api.cryptoquant.com/v1",
            "whale_alert": "https://api.whale-alert.io/v1",
        }

    def _make_request(
        self,
        provider: str,
        endpoint: str,
        params: Dict = None,
    ) -> Optional[Dict]:
        """
        Make API request to on-chain data provider.

        Args:
            provider: Provider name
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response JSON or None
        """
        if not self.api_key:
            return None

        base_url = self.endpoints.get(provider)
        if not base_url:
            return None

        url = f"{base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {provider} data: {e}")
            return None

    def get_exchange_flows(
        self,
        symbol: str,
        exchange: str = "all",
        hours: int = 24,
    ) -> List[ExchangeFlow]:
        """
        Get exchange inflow/outflow data.

        Args:
            symbol: Trading symbol (e.g., "ETH")
            exchange: Exchange name or "all"
            hours: Number of hours to look back

        Returns:
            List of ExchangeFlow objects
        """
        # This would call actual API in production
        # For now, return simulated data structure
        flows = []

        # Simulate recent flows
        now = datetime.now()
        for i in range(hours):
            timestamp = (now - timedelta(hours=i)).isoformat()

            # Simulate some flow activity
            inflow_usd = np.random.uniform(1000000, 50000000) if np.random.random() > 0.7 else 0
            outflow_usd = np.random.uniform(1000000, 50000000) if np.random.random() > 0.7 else 0

            if inflow_usd > 0:
                flows.append(ExchangeFlow(
                    timestamp=timestamp,
                    exchange=exchange,
                    symbol=symbol,
                    direction=FlowDirection.INFLOW,
                    amount_usd=inflow_usd,
                    amount_tokens=inflow_usd / 2000,  # Approximate price
                    tx_count=np.random.randint(10, 100),
                ))

            if outflow_usd > 0:
                flows.append(ExchangeFlow(
                    timestamp=timestamp,
                    exchange=exchange,
                    symbol=symbol,
                    direction=FlowDirection.OUTFLOW,
                    amount_usd=outflow_usd,
                    amount_tokens=outflow_usd / 2000,
                    tx_count=np.random.randint(10, 100),
                ))

        return flows

    def get_whale_transactions(
        self,
        symbol: str,
        min_amount_usd: float = 1000000,
        hours: int = 24,
    ) -> List[WhaleTransaction]:
        """
        Get large whale transactions.

        Args:
            symbol: Trading symbol
            min_amount_usd: Minimum transaction size in USD
            hours: Number of hours to look back

        Returns:
            List of WhaleTransaction objects
        """
        transactions = []

        # Simulate whale transactions
        now = datetime.now()
        for i in range(np.random.randint(5, 20)):
            timestamp = (now - timedelta(hours=np.random.uniform(0, hours))).isoformat()
            amount_usd = np.random.uniform(min_amount_usd, min_amount_usd * 50)

            # Determine transaction type
            categories = list(WhaleCategory)
            category = categories[np.random.randint(0, len(categories))]

            transactions.append(WhaleTransaction(
                timestamp=timestamp,
                symbol=symbol,
                category=category,
                amount_usd=amount_usd,
                amount_tokens=amount_usd / 2000,
                from_address=f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
                to_address=f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
                tx_hash=f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
            ))

        return sorted(transactions, key=lambda x: x.timestamp, reverse=True)

    def get_active_addresses(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get active address count over time.

        Args:
            symbol: Trading symbol
            days: Number of days

        Returns:
            DataFrame with timestamp and active_addresses columns
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Simulate trending data
        base_count = 500000 if symbol == "ETH" else 800000
        trend = np.linspace(0, 0.2, days)  # 20% increase trend
        noise = np.random.normal(0, 0.05, days)

        addresses = base_count * (1 + trend + noise)

        return pd.DataFrame({
            "timestamp": dates,
            "active_addresses": addresses.astype(int),
        })

    def get_mvrv_ratio(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get Market Value to Realized Value (MVRV) ratio.

        Args:
            symbol: Trading symbol
            days: Number of days

        Returns:
            DataFrame with MVRV ratio
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # MVRV > 3: overvalued, potential sell signal
        # MVRV < 1: undervalued, potential buy signal
        base_mvrv = 1.5
        mvrv_values = base_mvrv + np.random.normal(0, 0.3, days)

        return pd.DataFrame({
            "timestamp": dates,
            "mvrv_ratio": mvrv_values,
        })

    def get_exchange_reserves(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get exchange reserve data.

        Args:
            symbol: Trading symbol
            days: Number of days

        Returns:
            DataFrame with reserve amounts
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Simulate reserves (decreasing = bullish)
        base_reserve = 15000000 if symbol == "ETH" else 2000000
        trend = np.linspace(0, -0.05, days)  # 5% decrease trend
        noise = np.random.normal(0, 0.01, days)

        reserves = base_reserve * (1 + trend + noise)

        return pd.DataFrame({
            "timestamp": dates,
            "exchange_reserve": reserves.astype(int),
        })


class OnChainAnalyzer:
    """
    Analyze on-chain metrics for trading signals.

    Features:
    - Exchange flow analysis
    - Whale activity tracking
    - Momentum indicators
    - Signal generation
    """

    def __init__(
        self,
        fetcher: OnChainDataFetcher = None,
    ):
        """
        Args:
            fetcher: OnChainDataFetcher instance
        """
        self.fetcher = fetcher or OnChainDataFetcher()

    def analyze_exchange_flows(
        self,
        symbol: str,
        hours: int = 24,
    ) -> Dict:
        """
        Analyze exchange flows for bullish/bearish signals.

        Args:
            symbol: Trading symbol
            hours: Hours to analyze

        Returns:
            Analysis dict with signals
        """
        flows = self.fetcher.get_exchange_flows(symbol, hours=hours)

        if not flows:
            return {"signal": "neutral", "reason": "No flow data available"}

        # Calculate net flow
        inflow_total = sum(f.amount_usd for f in flows if f.direction == FlowDirection.INFLOW)
        outflow_total = sum(f.amount_usd for f in flows if f.direction == FlowDirection.OUTFLOW)

        net_flow = outflow_total - inflow_total
        net_flow_pct = (net_flow / (inflow_total + outflow_total) * 100) if (inflow_total + outflow_total) > 0 else 0

        # Generate signal
        # Outflow > Inflow = bullish (people moving coins off exchange)
        # Inflow > Outflow = bearish (people moving coins to exchange to sell)
        signal = "neutral"
        strength = "weak"

        if net_flow_pct > 20:
            signal = "bullish"
            strength = "strong" if net_flow_pct > 40 else "moderate"
        elif net_flow_pct > 10:
            signal = "bullish"
            strength = "weak"
        elif net_flow_pct < -20:
            signal = "bearish"
            strength = "strong" if net_flow_pct < -40 else "moderate"
        elif net_flow_pct < -10:
            signal = "bearish"
            strength = "weak"

        return {
            "signal": signal,
            "strength": strength,
            "inflow_usd": inflow_total,
            "outflow_usd": outflow_total,
            "net_flow_usd": net_flow,
            "net_flow_pct": round(net_flow_pct, 2),
            "flow_count": len(flows),
            "reason": f"Net {signal} flow ({strength}): {net_flow_pct:+.1f}%",
        }

    def analyze_whale_activity(
        self,
        symbol: str,
        min_amount_usd: float = 1000000,
        hours: int = 24,
    ) -> Dict:
        """
        Analyze whale transactions for signals.

        Args:
            symbol: Trading symbol
            min_amount_usd: Minimum whale transaction size
            hours: Hours to analyze

        Returns:
            Analysis dict with signals
        """
        transactions = self.fetcher.get_whale_transactions(
            symbol, min_amount_usd=min_amount_usd, hours=hours
        )

        if not transactions:
            return {"signal": "neutral", "reason": "No whale activity"}

        # Categorize transactions
        exchange_to_wallet = [t for t in transactions if t.category == WhaleCategory.EXCHANGE_TO_WALLET]
        wallet_to_exchange = [t for t in transactions if t.category == WhaleCategory.WALLET_TO_EXCHANGE]

        # Calculate totals
        accumulation_usd = sum(t.amount_usd for t in exchange_to_wallet)
        distribution_usd = sum(t.amount_usd for t in wallet_to_exchange)

        net_position = accumulation_usd - distribution_usd
        total_volume = sum(t.amount_usd for t in transactions)

        # Generate signal
        # Exchange -> Wallet = accumulation (bullish)
        # Wallet -> Exchange = distribution (bearish)
        signal = "neutral"
        strength = "weak"

        if net_position > total_volume * 0.3:
            signal = "bullish"
            strength = "strong" if net_position > total_volume * 0.5 else "moderate"
        elif net_position > total_volume * 0.1:
            signal = "bullish"
            strength = "weak"
        elif net_position < -total_volume * 0.3:
            signal = "bearish"
            strength = "strong" if net_position < -total_volume * 0.5 else "moderate"
        elif net_position < -total_volume * 0.1:
            signal = "bearish"
            strength = "weak"

        return {
            "signal": signal,
            "strength": strength,
            "transaction_count": len(transactions),
            "accumulation_usd": accumulation_usd,
            "distribution_usd": distribution_usd,
            "net_position_usd": net_position,
            "total_volume_usd": total_volume,
            "largest_transaction_usd": max(t.amount_usd for t in transactions),
            "reason": f"Whale {signal} ({strength}): {len(transactions)} large txs",
        }

    def analyze_mvrv(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict:
        """
        Analyze MVRV ratio for valuation signals.

        Args:
            symbol: Trading symbol
            days: Historical days for comparison

        Returns:
            Analysis dict with signals
        """
        mvrv_data = self.fetcher.get_mvrv_ratio(symbol, days=days)

        if mvrv_data.empty:
            return {"signal": "neutral", "reason": "No MVRV data"}

        current_mvrv = mvrv_data["mvrv_ratio"].iloc[-1]
        percentile = (mvrv_data["mvrv_ratio"] < current_mvrv).sum() / len(mvrv_data) * 100

        # MVRV interpretation
        # < 1: Undervalued (bullish)
        # 1-2: Fair value
        # 2-3: Overvalued
        # > 3: Significantly overvalued (bearish)
        signal = "neutral"
        strength = "weak"

        if current_mvrv < 1.0:
            signal = "bullish"
            strength = "strong" if current_mvrv < 0.8 else "moderate"
        elif current_mvrv > 3.0:
            signal = "bearish"
            strength = "strong"
        elif current_mvrv > 2.5:
            signal = "bearish"
            strength = "moderate"
        elif current_mvrv > 2.0:
            signal = "bearish"
            strength = "weak"

        return {
            "signal": signal,
            "strength": strength,
            "current_mvrv": round(current_mvrv, 3),
            "percentile": round(percentile, 1),
            "mvrv_mean": round(mvrv_data["mvrv_ratio"].mean(), 3),
            "mvrv_std": round(mvrv_data["mvrv_ratio"].std(), 3),
            "reason": f"MVRV {signal} ({strength}): {current_mvrv:.2f}",
        }

    def analyze_exchange_reserves(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict:
        """
        Analyze exchange reserve trends.

        Args:
            symbol: Trading symbol
            days: Historical days

        Returns:
            Analysis dict with signals
        """
        reserve_data = self.fetcher.get_exchange_reserves(symbol, days=days)

        if reserve_data.empty:
            return {"signal": "neutral", "reason": "No reserve data"}

        current_reserve = reserve_data["exchange_reserve"].iloc[-1]
        start_reserve = reserve_data["exchange_reserve"].iloc[0]

        # Calculate change
        reserve_change = current_reserve - start_reserve
        reserve_change_pct = (reserve_change / start_reserve) * 100

        # Calculate trend (linear regression slope)
        x = np.arange(len(reserve_data))
        y = reserve_data["exchange_reserve"].values
        slope = np.polyfit(x, y, 1)[0]
        slope_pct = (slope / start_reserve) * 100

        # Decreasing reserves = bullish (coins leaving exchanges)
        # Increasing reserves = bearish (coins entering exchanges)
        signal = "neutral"
        strength = "weak"

        if reserve_change_pct < -10:
            signal = "bullish"
            strength = "strong" if reserve_change_pct < -20 else "moderate"
        elif reserve_change_pct < -5:
            signal = "bullish"
            strength = "weak"
        elif reserve_change_pct > 10:
            signal = "bearish"
            strength = "strong" if reserve_change_pct > 20 else "moderate"
        elif reserve_change_pct > 5:
            signal = "bearish"
            strength = "weak"

        return {
            "signal": signal,
            "strength": strength,
            "current_reserve": round(current_reserve),
            "reserve_change": round(reserve_change),
            "reserve_change_pct": round(reserve_change_pct, 2),
            "trend_pct": round(slope_pct, 3),
            "reason": f"Reserves {signal} ({strength}): {reserve_change_pct:+.1f}%",
        }

    def get_composite_signal(
        self,
        symbol: str,
        hours: int = 24,
        days: int = 30,
    ) -> Dict:
        """
        Get composite on-chain signal combining all metrics.

        Args:
            symbol: Trading symbol
            hours: Hours for flow/whale analysis
            days: Days for trend analysis

        Returns:
            Composite signal dict
        """
        # Get individual signals
        flow_signal = self.analyze_exchange_flows(symbol, hours=hours)
        whale_signal = self.analyze_whale_activity(symbol, hours=hours)
        mvrv_signal = self.analyze_mvrv(symbol, days=days)
        reserve_signal = self.analyze_exchange_reserves(symbol, days=days)

        # Score each signal
        signal_scores = {
            "bullish": 1,
            "neutral": 0,
            "bearish": -1,
        }

        strength_multipliers = {
            "strong": 2,
            "moderate": 1.5,
            "weak": 1,
        }

        flow_score = signal_scores[flow_signal["signal"]] * strength_multipliers.get(flow_signal.get("strength", "weak"), 1)
        whale_score = signal_scores[whale_signal["signal"]] * strength_multipliers.get(whale_signal.get("strength", "weak"), 1)
        mvrv_score = signal_scores[mvrv_signal["signal"]] * strength_multipliers.get(mvrv_signal.get("strength", "weak"), 1)
        reserve_score = signal_scores[reserve_signal["signal"]] * strength_multipliers.get(reserve_signal.get("strength", "weak"), 1)

        # Calculate composite score
        composite_score = (flow_score + whale_score + mvrv_score + reserve_score) / 4

        # Determine composite signal
        composite_signal = "neutral"
        composite_strength = "weak"

        if composite_score > 1.0:
            composite_signal = "bullish"
            composite_strength = "strong"
        elif composite_score > 0.5:
            composite_signal = "bullish"
            composite_strength = "moderate"
        elif composite_score > 0:
            composite_signal = "bullish"
            composite_strength = "weak"
        elif composite_score < -1.0:
            composite_signal = "bearish"
            composite_strength = "strong"
        elif composite_score < -0.5:
            composite_signal = "bearish"
            composite_strength = "moderate"
        elif composite_score < 0:
            composite_signal = "bearish"
            composite_strength = "weak"

        return {
            "composite_signal": composite_signal,
            "composite_strength": composite_strength,
            "composite_score": round(composite_score, 2),
            "individual_signals": {
                "exchange_flows": flow_signal,
                "whale_activity": whale_signal,
                "mvrv_ratio": mvrv_signal,
                "exchange_reserves": reserve_signal,
            },
            "summary": f"On-chain: {composite_signal.upper()} ({composite_strength}) | Score: {composite_score:+.2f}",
        }


def get_onchain_signal(
    symbol: str = "ETH",
    hours: int = 24,
    days: int = 30,
) -> Dict:
    """
    Convenience function to get on-chain trading signal.

    Args:
        symbol: Trading symbol
        hours: Hours for recent activity analysis
        days: Days for trend analysis

    Returns:
        Signal dict with trading recommendation
    """
    analyzer = OnChainAnalyzer()
    return analyzer.get_composite_signal(symbol, hours=hours, days=days)


def generate_onchain_report(symbol: str = "ETH") -> str:
    """
    Generate comprehensive on-chain analysis report.

    Args:
        symbol: Trading symbol

    Returns:
        Formatted report string
    """
    analyzer = OnChainAnalyzer()
    composite = analyzer.get_composite_signal(symbol)

    report = f"On-Chain Analysis Report: {symbol}\n"
    report += f"{'=' * 50}\n\n"

    # Composite signal
    report += f"Composite Signal: {composite['composite_signal'].upper()}\n"
    report += f"Strength: {composite['composite_strength']}\n"
    report += f"Score: {composite['composite_score']:+.2f}\n\n"

    # Individual signals
    signals = composite["individual_signals"]

    report += "Exchange Flows:\n"
    flow = signals["exchange_flows"]
    report += f"  Signal: {flow['signal']}\n"
    report += f"  Net Flow: ${flow['net_flow_usd']:,.0f} ({flow['net_flow_pct']:+.1f}%)\n"
    report += f"  Inflow: ${flow['inflow_usd']:,.0f}\n"
    report += f"  Outflow: ${flow['outflow_usd']:,.0f}\n\n"

    report += "Whale Activity:\n"
    whale = signals["whale_activity"]
    report += f"  Signal: {whale['signal']}\n"
    report += f"  Transactions: {whale['transaction_count']}\n"
    report += f"  Net Position: ${whale['net_position_usd']:,.0f}\n"
    report += f"  Total Volume: ${whale['total_volume_usd']:,.0f}\n"
    report += f"  Largest Tx: ${whale['largest_transaction_usd']:,.0f}\n\n"

    report += "MVRV Ratio:\n"
    mvrv = signals["mvrv_ratio"]
    report += f"  Signal: {mvrv['signal']}\n"
    report += f"  Current: {mvrv['current_mvrv']:.2f}\n"
    report += f"  Percentile: {mvrv['percentile']:.1f}%\n"
    report += f"  Mean: {mvrv['mvrv_mean']:.2f}\n"
    report += f"  Std: {mvrv['mvrv_std']:.2f}\n\n"

    report += "Exchange Reserves:\n"
    reserve = signals["exchange_reserves"]
    report += f"  Signal: {reserve['signal']}\n"
    report += f"  Current: {reserve['current_reserve']:,}\n"
    report += f"  Change: {reserve['reserve_change_pct']:+.1f}%\n"
    report += f"  Trend: {reserve['trend_pct']:+.3f}%/period\n"

    return report
