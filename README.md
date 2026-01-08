# Fear & Greed Trading Strategy

A quantitative trading strategy that uses the Crypto Fear & Greed Index combined with RSI-based sentiment for cryptocurrency trading. Supports both long and short positions.

## Features

- **Fear & Greed Index Trading**: Buy when FGI is low (fear), sell when FGI is high (greed)
- **RSI Confirmation**: Use RSI oversold/overbought conditions for entry timing
- **Short Selling**: Short when FGI > 75 and RSI > 70 (extreme greed + overbought)
- **Trailing Stops**: ATR-based dynamic trailing stops
- **Regime Filtering**: Avoid trading in unfavorable market conditions
- **Multi-Asset Support**: BTC, ETH, and XRP with RSI sentiment proxy

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python backtest_suite.py --asset ETH-USD --start 2024-01-01 --end 2025-01-01

# Compare all assets
python backtest_suite.py --compare

# Walk-forward analysis
python backtest_suite.py --walk-forward

# Parameter validation
python backtest_suite.py --validate
```

## Results (2024-2025)

| Asset | Return | Sharpe | Win Rate | Drawdown | Trades |
|-------|--------|--------|----------|----------|--------|
| **ETH-USD** | **74.99%** | **1.75** | **72.7%** | **18.86%** | **23** |

ETH-USD is recommended for the best balance of risk-adjusted returns.

## Parameters

Default parameters (conservative):
- `rsi_window`: 14
- `trail_pct`: 8%
- `buy_quantile`: 0.20 (bottom 20% of FGI values)
- `sell_quantile`: 0.80 (top 20% of FGI values)
- `atr_multiplier`: 2.5

## Strategy Logic

### Long Entry
- FGI <= buy_quantile threshold AND RSI < 30

### Short Entry
- FGI >= 75 AND RSI > 70

### Exit Conditions
- FGI returns to neutral (< 45 for longs, > 75 exits shorts)
- RSI crosses extreme levels
- 25% profit target (longs) / 15% profit target (shorts)
- Trailing stop hit
- Max drawdown (15%) reached

## Files

- `main.py` - Main trading script
- `backtest_suite.py` - Comprehensive backtesting tool
- `src/strategy.py` - Core strategy implementation
- `src/config.py` - Configuration parameters
- `src/data/data_fetchers.py` - Data fetching from exchanges
- `src/sentiment.py` - RSI-based sentiment for non-BTC assets
- `src/parameter_optimizer.py` - Parameter optimization module

## Requirements

- Python 3.10+
- pandas
- numpy
- vectorbt
- requests

## License

MIT
