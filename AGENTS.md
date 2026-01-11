# AGENTS.md - Guidelines for Agentic Coding Assistants

## Project Overview

Fear & Greed Trading Strategy - A quantitative trading system using FGI and RSI sentiment for cryptocurrency trading with short selling support.

**Recommended Assets:** 
- **ETH-USD**: Best overall risk-adjusted returns (Sharpe 2.26)
- **XRP-USD**: Exceptional performance (172% return, Sharpe 2.39) - Now enabled for live trading

## Active Work Items

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for outstanding tasks (21 items pending across High/Medium/Low priority).

### Current Focus: Multi-Asset Enhancement Phase
1. **Multi-Asset Telegram Integration** - Extend Telegram bot for multi-asset trading
2. **Configuration Migration** - Move hardcoded parameters to YAML config files  
3. **Code Cleanup** - Remove unused files and consolidate duplicate code
4. **Multi-Asset Alpaca Integration** - Extend Alpaca exchange support to multi-asset

## Build/Lint/Test Commands

### Running the Trading Strategy
```bash
source venv/bin/activate && python main.py
```

### Backtest Suite
```bash
python backtest_suite.py                          # Single asset backtest
python backtest_suite.py --compare                # Multi-asset comparison
python backtest_suite.py --walk-forward           # Walk-forward analysis
python backtest_suite.py --validate               # Train/validation split
python backtest_suite.py --asset ETH-USD --start 2024-01-01 --end 2025-01-01
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Parameter Optimization
```bash
python main.py --optimize --optimization-type grid        # Grid search
python main.py --optimize --optimization-type random      # Random search
```

### Configuration Migration & Cleanup Commands
```bash
# Generate new YAML config structure
python scripts/generate_configs.py

# Validate configuration files
python scripts/validate_configs.py

# Clean up unused files (dry run)
python scripts/cleanup.py --dry-run

# Clean up unused files (actual)
python scripts/cleanup.py

# Test configuration loading
python scripts/test_config_loading.py
```

### Linting and Formatting (REQUIRED before commit)
```bash
ruff check .              # Run ruff linter
ruff check --fix .        # Auto-fix linting issues
ruff format .             # Format code
mypy .                    # Type checking (optional but recommended)
```

### Running Tests
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/test_*.py    # Run specific test file
pytest -k "test_name"     # Run specific test by name
pytest --cov             # With coverage report
```

### Required Verification After Every Change
```bash
ruff check . && python main.py
```

## Code Style Guidelines

### General Principles
- Follow PEP 8 style guide
- Write clean, readable, single-purpose functions
- Use meaningful variable and function names
- Keep functions under 50 lines when possible

### Imports
Organize in three sections with blank lines:
1. Standard library (os, sys, logging, etc.)
2. Third-party (pandas, numpy, vectorbt, etc.)
3. Local application (src.* modules)

```python
import logging
from typing import Optional

import pandas as pd
import numpy as np
import vectorbt as vbt

from src.strategy import TradingStrategy
```

### Formatting
- Use 4 spaces for indentation (no tabs)
- Line length: 88 characters (ruff default)
- Use `ruff format` for all formatting
- Blank lines between functions and around major blocks

### Type Hints
- Use type hints for all function parameters and returns
- Use `typing` module for complex types (Optional, List, Dict, Union)
- Example:
```python
from typing import Optional, List

def calculate_indicator(prices: pd.Series, window: int) -> pd.Series:
```

### Naming Conventions
- **Functions/variables**: snake_case (`calculate_returns`, `close_prices`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_WINDOW`, `MAX_POSITION`)
- **Classes**: PascalCase (`TradingStrategy`, `PortfolioAnalyzer`)
- **Private methods**: `_leading_underscore`

### Error Handling
- Use specific exception types, not bare `except:`
- Log errors with meaningful messages before raising
- Handle exceptions at the appropriate level
```python
try:
    data = vbt.YFData.download(symbol, start=start_date, end=end_date)
except ValueError as e:
    logger.error(f"Failed to download data for {symbol}: {e}")
    raise
```

### Constants and Configuration
- Define constants at module level in UPPER_SNAKE_CASE
- Avoid magic numbers; use named constants
- Use `src/config.py` for strategy parameters

### Documentation
- Use Google-style docstrings for all public functions and classes
- Include Args and Returns sections
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate the annualized Sharpe ratio.

    Args:
        returns: Series of asset returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
```

### Vectorbt Patterns
- Use `pd.DataFrame.vbt.signals.empty_like()` for signal placeholders
- Use `Portfolio.from_signals()` for backtesting
- Chain method calls for readability

### File Structure
- Keep related functionality together in `src/trading/`, `src/ml/`, `src/data/`
- Tests in `tests/` directory
- Use relative imports for local modules
- Split large files into logical components

### Code Review Checklist
- [ ] `ruff check` passes
- [ ] `ruff format` applied
- [ ] Type hints present and correct
- [ ] No bare `except:` clauses
- [ ] Docstrings on public functions
- [ ] Constants extracted to module level
- [ ] Meaningful variable names

## Short Selling Strategy
- **Long Entry:** RSI < 30, FGI/Sentiment < buy_quantile
- **Short Entry:** RSI > 70, FGI/Sentiment > 75
- **Short Exit:** RSI < 30, 15% profit target, or trailing stop
