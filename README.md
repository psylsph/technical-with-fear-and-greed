# BTC Fear & Greed Index Trading Strategy

A sophisticated algorithmic trading system that uses Fear & Greed Index (FGI), technical indicators, and machine learning to trade Bitcoin.

## Features

- **Multi-timeframe backtesting** (Daily, Hourly)
- **Machine learning predictions** for FGI direction
- **Dynamic FGI thresholds** based on market conditions
- **Technical indicators** (RSI, Trailing stops, Take profit)
- **Live trading** via Alpaca API
- **Paper trading** for risk-free testing
- **Portfolio persistence** across sessions

## Project Structure

```
├── src/
│   ├── config.py              # Configuration constants
│   ├── indicators.py          # Technical analysis indicators
│   ├── strategy.py            # Trading strategy logic
│   ├── ml/
│   │   └── ml_model.py        # ML model training and prediction
│   ├── data/
│   │   └── data_fetchers.py   # Data fetching utilities
│   ├── trading/
│   │   └── trading_engine.py  # Live/test trading execution
│   └── portfolio.py           # Portfolio management
├── tests/
│   └── test_trading_strategy.py  # Comprehensive test suite
├── main.py                    # Entry point
├── pyproject.toml            # Project configuration
├── requirements.txt          # Dependencies
└── dev-check.sh             # Development tools script
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys (optional):
   ```bash
   # For Coinbase data (optional)
   cp cdp_api_key.json.example cdp_api_key.json

   # For live trading (optional)
   export ALPACA_API_KEY=your_key
   export ALPACA_SECRET_KEY=your_secret
   ```

## Usage

### Backtesting
```bash
python main.py
```

### Paper Trading (Simulated)
```bash
python main.py --test
```

### Live Trading
```bash
python main.py --live
```

## Development

Run all development checks:
```bash
./dev-check.sh
```

This will:
- Format code with Black
- Sort imports with isort
- Lint with Ruff
- Type check with MyPy
- Run tests with 80% coverage requirement

### Running Tests Manually
```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

### Code Quality
- **Linting**: Ruff
- **Formatting**: Black
- **Type checking**: MyPy
- **Import sorting**: isort

## Strategy Details

### Entry Signals
- FGI ≤ dynamic buy threshold (20th percentile)
- RSI < 30 (oversold)
- ML prediction > threshold (bullish outlook)

### Exit Signals
- FGI ≥ dynamic sell threshold (80th percentile)
- RSI > 70 (overbought)
- 25% take profit
- 10% trailing stop loss

### ML Model
- **Type**: Random Forest Classifier
- **Features**: FGI, Price, RSI, Volume, Lagged FGI
- **Target**: Next day FGI direction (up/down)
- **Training**: Daily data from 2024-2025

## Risk Management

- Maximum position size: 10% of portfolio per trade
- Transaction fees: 0.25% taker, 0.15% maker
- Daily signal checking with 5-minute intervals
- Portfolio state persistence for paper trading

## Data Sources

- **Fear & Greed Index**: alternative.me API
- **Price Data**: Yahoo Finance (fallback to Coinbase)
- **Live Trading**: Alpaca API (paper mode available)

## License

MIT License