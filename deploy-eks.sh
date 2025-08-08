#!/bin/bash

# EKS Deployment Script for Enterprise Data Pipeline
set -e

# Configuration
REGISTRY=${REGISTRY:-"your-registry"}
IMAGE_NAME="privacy-intent-pipeline"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"data-pipeline"}
CLUSTER_NAME=${CLUSTER_NAME:-"your-eks-cluster"}

echo "🚀 Starting EKS deployment..."

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: docker not found. Please install docker."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Are you in the right directory?"
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile.optimized -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag for registry
if [ ! -z "$REGISTRY" ]; then
    echo "📤 Tagging image for registry..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    
    # Push to registry
    echo "📤 Pushing to registry: $REGISTRY"
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    
    # Update deployment to use registry image
    IMAGE_FULL_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    IMAGE_FULL_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply persistent volume claims
echo "💾 Applying persistent volume claims..."
kubectl apply -f k8s/persistent-volumes.yaml -n $NAMESPACE

# Update deployment with correct image
echo "🔄 Updating deployment with image: $IMAGE_FULL_NAME"
sed "s|privacy-intent-pipeline:latest|$IMAGE_FULL_NAME|g" k8s/deployment.yaml | kubectl apply -f - -n $NAMESPACE

# Apply services
echo "🔌 Applying services..."
kubectl apply -f k8s/services.yaml -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/privacy-intent-pipeline -n $NAMESPACE

# Show deployment status
echo "📊 Deployment status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Show logs
echo "📋 Recent logs from main pipeline:"
kubectl logs -f deployment/privacy-intent-pipeline -n $NAMESPACE --tail=50 &

echo "✅ EKS deployment completed successfully!"
echo ""
echo "🔗 Access points:"
echo "  - Pipeline logs: kubectl logs -f deployment/privacy-intent-pipeline -n $NAMESPACE"
echo "  - Pod status: kubectl get pods -n $NAMESPACE"
echo "  - Services: kubectl get services -n $NAMESPACE"
echo ""
echo "🚀 To scale the pipeline:"
echo "  kubectl scale deployment privacy-intent-pipeline --replicas=3 -n $NAMESPACE"
echo ""
echo "🔧 To update the deployment:"
echo "  ./deploy-eks.sh"
