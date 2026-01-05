# Alpaca Paper Trading Implementation Plan

## Overview

This document outlines the implementation of Alpaca paper trading integration for the existing Fear & Greed Index trading strategy. The implementation will support both backtesting (existing vectorbt mode) and live paper trading with the Alpaca API.

**Data Sources:**
- **Historical Prices**: Coinbase API (public endpoints)
- **Real-time Prices**: Coinbase WebSocket feed
- **Order Execution**: Alpaca API

## Requirements Summary

| Requirement | Decision |
|-------------|----------|
| Order types | Limit orders |
| Execution mode | Continuous (24/7) |
| Position sizing | Percentage based on FGI tier |
| Notifications | None |
| Capital (test mode) | Fixed $1000 |
| Capital (live mode) | Fetch from API (persists across restarts) |
| Historical data | Coinbase API |
| Real-time data | Coinbase WebSocket |

## Fear & Greed Index Position Sizing

| FGI Range | Classification | Position Size (% of Available Capital) |
|-----------|----------------|----------------------------------------|
| 0-20 | Extreme Fear | 60% |
| 21-35 | Fear | 45% |
| 36-65 | Neutral | 30% |
| 66-80 | Greed | 20% |
| 81-100 | Extreme Greed | 10% |

**Profit reduction**: If existing positions have >10% unrealized gain, reduce new position sizes by 50%.

## Limit Order Strategy

### Entry Orders
- **Placement**: 1.5% below current ask price
- **Timeout**: Cancel/reprice if no fill after 6 hours
- **Reprice trigger**: If price moves 2%+ past your limit
- **Validation**: Recheck FGI signal before repricing

### Exit Orders
- **Type**: Trailing stop (5% trailing, -3% trigger)
- **Alternative**: Take profit at +10% (whichever triggers first exits)

## Risk Controls

| Control | Value | Purpose |
|---------|-------|---------|
| Max concurrent positions | 3 | Diversification without overtrading |
| Max order size | $200,000 | Alpaca API limit |
| Limit order discount | 1.5% below ask | Better entry price |
| Limit order timeout | 6 hours | Avoid stale orders |
| Trailing stop | 5% trail, -3% trigger | Capture gains, limit losses |
| Profit target | +10% auto-sell | Take profits systematically |
| FGI extreme greed | 10% size | Reduce buying at tops |

## Alpaca Constraints

- **Trading hours**: 24/7 for crypto
- **Order types**: Market, Limit, Stop Limit
- **Time in force**: `gtc`, `ioc`
- **No margin**: Cannot leverage or short crypto
- **Fractional**: Supported (min 0.0001 BTC)
- **Max order**: $200,000 notional

## File Structure

```
technical-with-fear-and-greed/
├── trading_strategy.py           # Main entry point with CLI args
├── IMPLEMENTATION_PLAN.md        # This document
├── config/
│   ├── settings.py              # All tunable parameters
│   └── risk_limits.py           # Position sizing rules
├── alpaca/
│   ├── client.py                # RESTClient wrapper (order execution)
│   └── streaming.py             # WebSocket for order status updates
├── coinbase/
│   ├── client.py                # REST API for historical data
│   └── streaming.py             # WebSocket for real-time prices
├── strategy/
│   ├── signals.py               # FGI + trend signal generation
│   └── position_sizer.py        # Fear-based sizing calculator
├── execution/
│   ├── executor.py              # Main continuous loop
│   └── order_handler.py         # Limit order lifecycle management
├── models/
│   └── types.py                 # Typed classes
├── .env                         # API keys
├── requirements.txt             # Updated dependencies
└── logs/
    └── trading.log              # Trade activity log
```

## State Persistence Strategy

| Component | Test Mode ($1000 fixed) | Live Mode (API) |
|-----------|-------------------------|-----------------|
| Cash | Fixed $1000 | `client.get_account().cash` |
| Open positions | Memory only | `client.get_positions()` |
| Pending orders | Memory only | `client.get_orders()` |
| P&L tracking | Calculate from fills | Sum positions + cash vs. basis |

On startup in live mode:
1. Fetch cash balance from Alpaca
2. Fetch all open positions
3. Fetch all open orders
4. Reconstruct internal state
5. Resume execution loop

## Environment Variables

```bash
# .env file (see .env.example for template)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## Dependencies to Add

```
alpaca-py>=0.30.0
python-dotenv>=1.0.0
websockets>=15.0
```

**Note**: Coinbase data uses public API endpoints (no authentication required for historical candles and ticker data).

## CLI Usage

```bash
# Backtest mode (existing behavior)
python trading_strategy.py --mode backtest

# Paper trading with $1000 fixed capital
python trading_strategy.py --mode paper

# Live trading with API state
python trading_strategy.py --mode live
```

## Execution Flow (Continuous Mode)

```
Main Loop (while True):
1. Get State
   - Fetch FGI from alternative.me
   - Fetch BTC price from Coinbase WebSocket (real-time) or REST (fallback)
   - Get account cash/positions from Alpaca

2. Signal Generation
   - Calculate trend (price vs MA50/200) using Coinbase historical data
   - Evaluate FGI tier
   - Generate entry/exit signals

3. Order Management
   - Place limit orders 1.5% below ask on Alpaca
   - Monitor fills every 30 seconds
   - Reprice stale orders (6h timeout)

4. Logging & Sleep
   - Log trade, price, FGI, P&L
   - Sleep 60 seconds before next iteration
```

## Coinbase API Integration

### Historical Data (REST API)
- **Endpoint**: `https://api.exchange.coinbase.com/products/BTC-USD/candles`
- **Parameters**: granularity (seconds), start, end
- **Granularity**: 86400 (daily) for trend calculation
- **Use case**: MA50, MA200 calculation

### Real-time Data (WebSocket)
- **Endpoint**: `wss://ws-feed.exchange.coinbase.com`
- **Subscription**: `{"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}`
- **Use case**: Live price updates for order management

## Estimated Effort

| Phase | Hours | Deliverables |
|-------|-------|--------------|
| 1. Dependencies & Config | 1 | requirements.txt, settings.py |
| 2. Alpaca Client | 2 | client.py, streaming.py |
| 3. Strategy Layer | 2 | signals.py, position_sizer.py |
| 4. Execution Layer | 3 | executor.py, order_handler.py |
| 5. Integration | 2 | Refactored trading_strategy.py |
| 6. Testing & Polish | 2 | Bug fixes, logging, docs |
| **Total** | **12** | |

## Implementation Order

1. Update `requirements.txt` with new dependencies
2. Create `config/settings.py` with all tunable parameters
3. Create `config/risk_limits.py` with position sizing rules
4. Create `coinbase/client.py` for historical data (REST API)
5. Create `coinbase/streaming.py` for real-time prices (WebSocket)
6. Create `alpaca/client.py` with RESTClient wrapper for order execution
7. Create `strategy/signals.py` with FGI + trend signal generation
8. Create `strategy/position_sizer.py` with fear-based sizing
9. Create `execution/order_handler.py` for limit order lifecycle
10. Create `execution/executor.py` for main continuous loop
11. Refactor `trading_strategy.py` with CLI args and mode selection
12. Add Alpaca keys to `.env` (user to provide)
13. Test and verify

## Verification Commands

After implementation, run:
```bash
ruff check .
ruff format .
pyright .
python trading_strategy.py --mode paper
```

## Notes

- **Paper trading endpoint**: `https://paper-api.alpaca.markets`
- **Live trading endpoint**: `https://api.alpaca.markets`
- **Crypto symbol**: `BTC/USD`
- **All orders use**: `time_in_force='gtc'` (good till cancelled)
- **Environment variables**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- **Coinbase historical**: `https://api.exchange.coinbase.com/products/BTC-USD/candles`
- **Coinbase websocket**: `wss://ws-feed.exchange.coinbase.com`