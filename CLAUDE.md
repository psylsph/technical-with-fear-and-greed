# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A quantitative cryptocurrency trading system using the Fear & Greed Index (FGI) combined with RSI for ETH-USD trading. The system supports backtesting, paper trading, and live trading with Alpaca.

**Current Focus**: Single-asset (ETH-USD) trading with extensive risk controls and validation framework.

## Development Commands

```bash
# Linting
python -m ruff check <file>           # Check a file
python -m ruff check src/              # Check all src files

# Testing
python -m pytest tests/ -v             # Run all tests
python -m pytest tests/test_trading_strategy.py -v  # Run specific test file
python -m pytest tests/test_trading_strategy.py::TestPortfolio -v  # Run specific test class

# Run trading modes
python main.py --live                  # Live trading with Alpaca paper trading
python main.py --test                  # Paper trading with local state persistence
python main.py --test -q               # Quiet mode (minimal output)

# Backtesting
python backtest_suite.py --asset ETH-USD --start 2024-01-01 --end 2025-01-01
python backtest_suite.py --walk-forward
python backtest_suite.py --validate     # Parameter validation
```

**Important**: After making code changes, always run: `ruff check` → `pytest` → `python main.py --test -q`

## Architecture

### Core Signal Flow

```
Data Fetch → Indicators → Signal Generation → Trading Decision → Execution
     ↓            ↓              ↓                  ↓              ↓
  FGI/API    RSI/ADX/ATR    generate_signal()  should_trade()  Alpaca
```

### Key Modules

**Entry Points**:
- `main.py` - CLI entry point with modes: --live, --test, --multi-tf, --walk-forward, --optimize
- `backtest_suite.py` - Backtesting with walk-forward validation

**Strategy Core**:
- `src/strategy.py::generate_signal()` - Central signal generation using FGI + RSI + ML
- `src/strategy.py::run_strategy()` - Full strategy execution with vectorbt

**Trading Engine** (`src/trading/trading_engine.py`):
- `should_trade()` / `should_trade_test()` - Position sizing with Kelly criterion
- `get_position()` - Get Alpaca position with entry price and P&L (manual calculation)
- `execute_trade()` - Trade execution with IOC time_in_force for crypto

**Portfolio** (`src/portfolio.py`):
- State persistence in `test_portfolio_state.json` (test mode)
- Tracks: `cash`, `eth_held`, `entry_price`, `trades`
- `simulate_trade()` - Updates state with average entry price calculation

**Data** (`src/data/data_fetchers.py`):
- `fetch_fear_greed_index()` - FGI from alternative.me
- `fetch_unified_price_data()` - Price data from various exchanges
- Database cache at `cache/market_data.db`

### Risk Controls (`src/trading/risk_controls.py`)

All risk controls persist state in JSON files:
- `DailyLossLimit` - Stop trading if daily loss > 2%
- `TimeBasedExit` - Close position after 14 days if no profit
- `TrailingStop` - Trail stop at 3% to lock in profits
- `PositionSizeLimit` - Cap positions at 5% of portfolio

Position tracking consolidated in: `position_tracking.json`

### Key Configuration (`src/config.py`)

- `DEFAULT_ASSET = "ETH-USD"` - Single asset trading
- `BEST_PARAMS` - Optimized parameters for live trading
- `INITIAL_CAPITAL = 1000`
- `MAKER_FEE = 0.0015`, `TAKER_FEE = 0.0025`

## Important Constraints

### Position Tracking
- Entry price is tracked in `portfolio_state["entry_price"]`
- Average entry price is updated when adding to positions
- P&L is calculated manually: `((current_price - entry_price) / entry_price) * 100`

### Trading Rules
- **Pyramiding**: Only add to long positions if position is profitable (≥2% unrealized P&L)
- **Max position size**: 5% of portfolio (configurable)
- **Crypto orders**: Must use `time_in_force=TimeInForce.IOC` (DAY is invalid)
- **Max drawdown**: 5% for crypto volatility (reduced from 8%)

### Signal Types
- `buy` - Enter long (or add to profitable long)
- `sell` - Exit long position
- `short` - Enter short (if enabled)
- `cover` - Cover short position
- `hold` - No action

### Multi-timeframe Support
- Higher timeframe (daily) indicators: trend, RSI
- Used for filtering: only buy when higher_tf trend is BULLISH

## Testing & Validation Modules

Located in `src/trading/`:
- `walk_forward_analysis.py` - Rolling optimization windows (6mo train, 1mo test)
- `monte_carlo_simulation.py` - 1000+ bootstrap/permutation simulations
- `out_of_sample_testing.py` - Train/test split validation
- `stress_testing.py` - Black swan and flash crash scenarios
- `property_based_testing.py` - Hypothesis testing for edge cases

## ML Model Status

**INTENTIONALLY DISABLED** - Technical analysis outperformed ML in backtests.
- `src/ml/ml_model.py` exists but is not used in live trading
- Keep ML disabled; focus on signal quality and risk management

## State Files

JSON files for persistence (in project root):
- `test_portfolio_state.json` - Paper trading portfolio state
- `position_tracking.json` - Unified position tracking
- `daily_pnl_state.json` - Daily P&L tracking
- `api_circuit_breaker_state.json` - API failure tracking
- `pending_signals.json` - Entry confirmation signals
- `partial_exit_positions.json` - Partial exit tracking
- `notification_log.json` - Email notification history
- `api_latency_state.json` - API latency tracking
- `anomaly_detection_state.json` - Anomaly detection history
- `system_health_state.json` - System health status

## Common Issues

### "Order failed: invalid crypto time_in_force"
Use `TimeInForce.IOC` for crypto orders, not `TimeInForce.DAY`

### "P&L: -0.00%"
Alpaca's `unrealized_plpc` field is unreliable. Calculate manually:
```python
pnl_pct = ((current_price - entry_price) / entry_price) * 100
```

### Position reading errors
Format: "ETH-USD" → "ETHUSD" for Alpaca (remove "/")

### JSON serialization errors with numpy types
Convert numpy int64/float64 to native Python types before logging

### TypeError: 'float' object has no attribute 'get'
Old tests use outdated function signatures. Core functionality works.
