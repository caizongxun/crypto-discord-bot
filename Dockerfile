FROM python:3.11-slim

LABEL maintainer="caizongxun"
LABEL description="Crypto Discord Bot - Real-time Price Prediction using LSTM"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/hf_cache templates

# Expose dashboard port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Run bot by default
CMD ["python", "bot.py"]
