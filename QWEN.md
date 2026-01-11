# Technical Analysis with Fear and Greed - Project Overview

## Project Description

This is a quantitative cryptocurrency trading strategy that combines the Crypto Fear & Greed Index with RSI-based sentiment analysis for algorithmic trading. The system supports both long and short positions and includes advanced features like trailing stops, regime filtering, and multi-asset support.

## Key Features

- **Fear & Greed Index Trading**: Buys when FGI is low (fear), sells when FGI is high (greed)
- **RSI Confirmation**: Uses RSI oversold/overbought conditions for entry timing
- **Short Selling**: Shorts when FGI > 75 and RSI > 70 (extreme greed + overbought)
- **Trailing Stops**: ATR-based dynamic trailing stops
- **Regime Filtering**: Avoids trading in unfavorable market conditions
- **Multi-Asset Support**: Supports BTC, ETH, XRP, SOL, DOGE and other major cryptocurrencies
- **Machine Learning Integration**: Includes ML models (Random Forest, LSTM, Transformer) for enhanced predictions
- **Real-time Trading**: WebSocket-based real-time trading system
- **Walk-forward Analysis**: Validates strategy performance across rolling time windows
- **Parameter Optimization**: Grid, random, and walk-forward parameter optimization

## Architecture

### Main Components

- `main.py`: Primary entry point with multiple modes (backtest, live, test, optimize)
- `backtest_suite.py`: Comprehensive backtesting and comparison tool
- `src/strategy.py`: Core strategy implementation
- `src/config.py`: Configuration parameters and constants
- `src/data/data_fetchers.py`: Data fetching from exchanges (Coinbase, Binance, Kraken, Bybit)
- `src/sentiment.py`: RSI-based sentiment for non-BTC assets
- `src/parameter_optimizer.py`: Parameter optimization module
- `src/ml/ml_model.py`: Machine learning model implementations
- `src/multi_asset_trading.py`: Multi-asset trading engine

### Directory Structure

```
├── main.py                 # Main application entry point
├── backtest_suite.py       # Comprehensive backtesting tool
├── paper_trading.py        # Paper trading implementation
├── debug_live.py           # Live trading debugging
├── src/
│   ├── config.py           # Configuration constants
│   ├── strategy.py         # Core trading strategy
│   ├── indicators.py       # Technical indicators
│   ├── data/
│   │   └── data_fetchers.py # Data fetching utilities
│   ├── ml/                 # Machine learning models
│   ├── trading/            # Trading engine components
│   ├── utils/              # Utility functions
│   └── ...
├── docker-compose.yml      # Docker orchestration
├── Dockerfile             # Container specification
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── ...
```

## Building and Running

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run backtest
python backtest_suite.py --asset ETH-USD --start 2024-01-01 --end 2025-01-01

# Compare all assets
python backtest_suite.py --compare

# Walk-forward analysis
python backtest_suite.py --walk-forward

# Parameter validation
python backtest_suite.py --validate
```

### Docker Setup

```bash
# Build and run with Docker
docker-compose up --build

# Or run specific commands
docker-compose run trading-app python main.py --test
```

### Available Commands

- `python main.py --test`: Run paper trading simulation
- `python main.py --live`: Run live trading with real money
- `python main.py --optimize`: Run parameter optimization
- `python main.py --walk-forward`: Run walk-forward analysis
- `python main.py --multi-tf`: Run multi-timeframe backtesting
- `python main.py --realtime`: Run real-time trading system
- `python main.py --multi-asset`: Enable multi-asset trading
- `python main.py --ml-status`: Show ML model status
- `python main.py --train-advanced`: Train advanced ML models

## Strategy Logic

### Long Entry Conditions
- FGI <= buy_quantile threshold (default: bottom 20% of FGI values) AND RSI < 30

### Short Entry Conditions
- FGI >= 75 AND RSI > 70

### Exit Conditions
- FGI returns to neutral (< 45 for longs, > 75 exits shorts)
- RSI crosses extreme levels
- 25% profit target (longs) / 15% profit target (shorts)
- Trailing stop hit
- Max drawdown (15%) reached

### Risk Management
- Dynamic position sizing based on volatility
- Adaptive thresholds per market regime
- ATR-based trailing stops
- Maximum drawdown limits

## Default Parameters

- `rsi_window`: 14
- `trail_pct`: 8%
- `buy_quantile`: 0.20 (bottom 20% of FGI values)
- `sell_quantile`: 0.80 (top 20% of FGI values)
- `atr_multiplier`: 2.5

## Performance Results (2024-2025)

| Asset | Return | Sharpe | Win Rate | Drawdown | Trades |
|-------|--------|--------|----------|----------|--------|
| **ETH-USD** | **74.99%** | **1.75** | **72.7%** | **18.86%** | **23** |

ETH-USD showed the best balance of risk-adjusted returns for the 2024-2025 period.

## Development Conventions

- Code follows PEP 8 style guidelines
- Type hints are used throughout
- Comprehensive error handling and logging
- Modular architecture with clear separation of concerns
- Configuration-driven approach for parameters
- Extensive testing with pytest

## Dependencies

Key Python packages:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- vectorbt: Vectorized backtesting
- scikit-learn: Machine learning algorithms
- requests: HTTP requests for API integration
- yfinance: Financial data fetching
- coinbase-advanced-py: Coinbase API integration
- alpaca-py: Alpaca trading API
- plotly: Interactive visualization
- python-telegram-bot: Telegram bot integration

## Testing

The project includes unit tests in the `tests/` directory and uses pytest for test execution. Coverage is monitored and maintained at acceptable levels.

## Deployment

The system can be deployed using Docker containers with the provided Dockerfile and docker-compose.yml. It's designed to run continuously with restart policies and volume mounts for persistent state.