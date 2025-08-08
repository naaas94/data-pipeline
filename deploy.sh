#!/bin/bash

# Simple deployment script for Enterprise Data Pipeline
set -e

echo "🚀 Starting deployment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Are you in the right directory?"
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile -t enterprise-data-pipeline:latest .

# Run tests in container
echo "🧪 Running tests in container..."
docker run --rm enterprise-data-pipeline:latest python -m pytest tests/ -v

# Optional: Push to registry if REGISTRY is set
if [ ! -z "$REGISTRY" ]; then
    echo "📤 Pushing to registry: $REGISTRY"
    docker tag enterprise-data-pipeline:latest $REGISTRY/enterprise-data-pipeline:latest
    docker push $REGISTRY/enterprise-data-pipeline:latest
fi

echo "✅ Deployment completed successfully!"
