FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for running the application
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Default command with test parameter
CMD ["python", "main.py", "--live", "--multi-asset", "--assets", "ETH-USD,BTC-USD,XRP-USD,SOL-USD"]
