# AGENTS.md - Guidelines for Agentic Coding Assistants

This document provides guidelines for agents working in this repository.

## Project Overview

Fear & Greed Trading Strategy - A quantitative trading system using FGI and RSI sentiment for cryptocurrency trading with short selling support.

**Recommended Asset:** ETH-USD (best risk-adjusted returns)

## Build/Lint/Test Commands

### Running the Trading Strategy
```bash
source venv/bin/activate && python main.py
```

### Backtest Suite (Recommended)
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
python main.py --optimize --optimization-type walk_forward  # Walk-forward
```

### Linting and Formatting (REQUIRED)
```bash
ruff check .      # Run ruff linter on entire codebase
ruff check --fix .  # Auto-fix linting issues
ruff format .     # Format code with ruff
```

### Running Tests
```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest -k "test_name"  # Run specific test
```

### Required Verification After Every Change
```bash
ruff check . && python main.py
```

## Short Selling

The strategy now supports both long and short positions:
- **Long Entry:** RSI < 30, FGI/Sentiment < buy_quantile
- **Short Entry:** RSI > 70, FGI/Sentiment > 75
- **Short Exit:** RSI < 30, or 15% profit target, or trailing stop

## Code Style Guidelines

### General Principles
- Follow PEP 8 style guide
- Write clean, readable, and maintainable code
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Imports
Organize imports in three sections separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
import pandas as pd
import numpy as np

import vectorbt as vbt
```

### Formatting
- Use 4 spaces for indentation (no tabs)
- Line length: 88 characters (ruff default)
- Use ruff format for all code formatting
- Add blank lines between function definitions and around major code blocks

### Type Hints
- Use type hints for function parameters and return values
- Use the `typing` module for complex types (Optional, List, Dict, Union, etc.)
- Example:
```python
from typing import Optional, List

def calculate_indicator(prices: pd.Series, window: int) -> pd.Series:
```

### Naming Conventions
- **Functions and variables**: snake_case (e.g., `calculate_returns`, `close_prices`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_WINDOW`, `MAX_POSITION`)
- **Classes**: PascalCase (e.g., `TradingStrategy`, `PortfolioAnalyzer`)
- Use descriptive names that indicate purpose

### Error Handling
- Use specific exception types rather than bare `except:` clauses
- Handle exceptions at the appropriate level
- Log errors with meaningful messages
- Example:
```python
try:
    data = vbt.YFData.download(symbol, start=start_date, end=end_date)
except ValueError as e:
    logger.error(f"Failed to download data for {symbol}: {e}")
    raise
```

### Constants and Configuration
- Define constants at the module level
- Use ALL_CAPS for constant names
- Avoid magic numbers; use named constants

### Documentation
- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstrings:
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Series of asset returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
```

### Vectorbt Specific Patterns
- Use `pd.DataFrame.vbt.signals.empty_like()` for signal placeholders
- Use `Portfolio.from_signals()` for backtesting
- Chain method calls where appropriate for readability
- Handle vectorbt data fetching with type checking

### File Structure
- Keep related functionality together
- Use relative imports for local modules
- Limit file length; split large files into logical components

### Code Review Checklist
- [ ] ruff check passes
- [ ] ruff format applied
- [ ] Type hints present and correct
- [ ] No bare except clauses
- [ ] Docstrings on public functions
- [ ] Constants extracted to module level
- [ ] Meaningful variable names
