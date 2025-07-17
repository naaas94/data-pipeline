# Enterprise Data Pipeline Dockerfile
# Multi-stage build for optimized production image

# Base stage with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    black \
    flake8 \
    mypy

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config.yaml .
COPY great_expectations.yml .

# Create necessary directories
RUN mkdir -p output logs checkpoints

# Set development environment
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY config.yaml .
COPY great_expectations.yml .

# Create necessary directories
RUN mkdir -p output logs checkpoints

# Set production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/data_pipeline.py", "--config", "config.yaml"] 