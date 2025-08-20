#!/bin/bash

# EKS Deployment Script for Dataset Generation
set -e

# Configuration
REGISTRY=${REGISTRY:-""}
IMAGE_NAME="dataset-generator"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"data-pipeline"}

echo "🚀 Starting EKS deployment for dataset generation..."

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: docker not found. Please install docker."
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag and push to registry if specified
if [ ! -z "$REGISTRY" ]; then
    echo "📤 Tagging and pushing to registry..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    IMAGE_FULL_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    IMAGE_FULL_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Update deployment with correct image and apply
echo "🔄 Deploying with image: $IMAGE_FULL_NAME"
sed "s|dataset-generator:latest|$IMAGE_FULL_NAME|g" k8s/deployment.yaml | kubectl apply -f - -n $NAMESPACE

# Monitor job status
echo "⏳ Monitoring job status..."
kubectl wait --for=condition=complete job/dataset-generation-job -n $NAMESPACE --timeout=300s

# Show job status and logs
echo "📊 Job status:"
kubectl get jobs -n $NAMESPACE
kubectl get pods -n $NAMESPACE

echo "📋 Job logs:"
kubectl logs job/dataset-generation-job -n $NAMESPACE

echo "✅ Dataset generation deployment completed!"
echo ""
echo "🔗 Useful commands:"
echo "  - Check job status: kubectl get jobs -n $NAMESPACE"
echo "  - View logs: kubectl logs job/dataset-generation-job -n $NAMESPACE"
echo "  - Re-run job: kubectl delete job dataset-generation-job -n $NAMESPACE && kubectl apply -f k8s/deployment.yaml -n $NAMESPACE"
