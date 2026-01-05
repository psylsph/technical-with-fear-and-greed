# AGENTS.md - Guidelines for Agentic Coding Assistants

This document provides guidelines for agents working in this repository.

## Build/Lint/Test Commands

### Running the Trading Strategy
```bash
source venv/bin/activate && python trading_strategy.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Linting and Formatting (REQUIRED)
```bash
ruff check .      # Run ruff linter on entire codebase
ruff check --fix .  # Auto-fix linting issues
ruff format .     # Format code with ruff
```

### Running Tests
No formal test framework is configured. To add tests:
```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest -k "test_name"  # Run specific test
```

### Required Verification After Every Change
Run this command after making any code changes:
```bash
ruff check . && pyright . && python trading_strategy.py
```

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
- Handle vectorbt data fetching with type checking as shown in trading_strategy.py

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
