# Dataset Generation Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config*.yaml ./

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Create output directory
RUN mkdir -p output

# Run dataset generation
CMD ["python", "src/data/dataset_generator.py"] 