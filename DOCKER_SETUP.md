c# Docker Setup for Technical Trading Strategy

This Docker setup provides a clean, isolated environment running Python 3.13, which resolves the numba/coverage compatibility issues encountered in the local environment.

## Quick Start

### Build and Run with Docker Compose (Recommended)

```bash
# Build the Docker image
docker compose build

# Run the trading strategy with test mode
docker compose up

# Run without test mode
docker compose run trading-app python main.py
```

### Build and Run with Docker CLI

```bash
# Build the image
docker build -t technical-trading .

# Run with test mode
docker run -it --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/main.py:/app/main.py \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/cdp_api_key.json:/app/cdp_api_key.json \
  -e PYTHONUNBUFFERED=1 \
  technical-trading

# Run without test mode
docker run -it --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/main.py:/app/main.py \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/cdp_api_key.json:/app/cdp_api_key.json \
  -e PYTHONUNBUFFERED=1 \
  technical-trading python main.py
```

## Configuration

### Environment Variables

The container automatically loads environment variables from the `.env` file mounted as a volume.

### Required Files

- `.env` - Environment variables for API keys and configuration
- `cdp_api_key.json` - Coinbase API key configuration
- `main.py` - Main application entry point
- `trading_strategy.py` - Trading strategy implementation
- `src/` - Source code directory

## Troubleshooting

### Rebuilding the Image

If you make changes to `requirements.txt` or `Dockerfile`:

```bash
docker compose build --no-cache
```

### Removing the Container

```bash
docker compose down
```

### Checking Logs

```bash
docker compose logs -f
```

### Running Commands Inside the Container

```bash
docker compose run trading-app bash
```

## Why Docker?

The local virtual environment had Python version compatibility issues:
- Original venv was created with Python 3.12
- Current system uses Python 3.13.5
- This caused numba/coverage incompatibility errors

Docker provides:
- Consistent Python 3.13 environment
- Isolated dependencies
- Reproducible builds
- No local environment conflicts
