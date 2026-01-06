#!/bin/bash
# Development tools script

set -e

echo "Running development checks..."

# Format code
echo "Formatting code with black..."
black .

# Sort imports
echo "Sorting imports with isort..."
isort .

# Lint with ruff
echo "Linting with ruff..."
ruff check . --fix

# Type check with mypy
echo "Type checking with mypy..."
mypy src/

# Run tests with coverage
echo "Running tests with coverage..."
pytest --cov=src --cov-report=term-missing --cov-fail-under=80

echo "All checks passed! ✅"