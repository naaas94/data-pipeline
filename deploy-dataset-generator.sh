#!/bin/bash

# Dataset Generator Deployment Script
set -e

# Configuration
REGISTRY=${REGISTRY:-"your-registry"}
IMAGE_NAME="dataset-generator"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"data-pipeline"}
BUCKET_NAME=${BUCKET_NAME:-"pcc-datasets"}

echo "🚀 Starting Dataset Generator deployment..."

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

# Check for GCP credentials
if [ ! -f "credentials/service-account-key.json" ]; then
    echo "⚠️  Warning: GCP credentials not found at credentials/service-account-key.json"
    echo "   Please ensure you have Google Cloud credentials set up for GCS upload."
    echo "   You can either:"
    echo "   1. Place your service account key at credentials/service-account-key.json"
    echo "   2. Use gcloud auth application-default login"
    echo "   3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
fi

# Build Docker image
echo "🐳 Building Dataset Generator Docker image..."
docker build -f Dockerfile.dataset-generator -t ${IMAGE_NAME}:${IMAGE_TAG} .

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

# Create GCP credentials secret if credentials file exists
if [ -f "credentials/service-account-key.json" ]; then
    echo "🔐 Creating GCP credentials secret..."
    kubectl create secret generic gcp-service-account-key \
        --from-file=service-account-key.json=credentials/service-account-key.json \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
else
    echo "⚠️  No GCP credentials found. Please ensure authentication is set up."
fi

# Apply persistent volume claims
echo "💾 Applying persistent volume claims..."
kubectl apply -f k8s/dataset-generator-job.yaml -n $NAMESPACE

# Update job with correct image
echo "🔄 Updating job with image: $IMAGE_FULL_NAME"
sed "s|dataset-generator:latest|$IMAGE_FULL_NAME|g" k8s/dataset-generator-job.yaml | kubectl apply -f - -n $NAMESPACE

# Show job status
echo "📊 Job status:"
kubectl get jobs -n $NAMESPACE

# Show pod status
echo "📊 Pod status:"
kubectl get pods -n $NAMESPACE -l job-name=dataset-generator-job

# Show logs
echo "📋 Recent logs from dataset generator:"
kubectl logs -f job/dataset-generator-job -n $NAMESPACE --tail=50 &

echo "✅ Dataset Generator deployment completed!"
echo ""
echo "🔗 Access points:"
echo "  - Job logs: kubectl logs -f job/dataset-generator-job -n $NAMESPACE"
echo "  - Pod status: kubectl get pods -n $NAMESPACE -l job-name=dataset-generator-job"
echo "  - Job status: kubectl get jobs -n $NAMESPACE"
echo ""
echo "📁 Generated datasets will be:"
echo "  - Locally: ./output/df_YYYYMMDD.csv"
echo "  - GCS: gs://$BUCKET_NAME/balanced_dataset_YYYYMMDD.csv"
echo ""
echo "🔄 To run the job again:"
echo "  kubectl delete job dataset-generator-job -n $NAMESPACE"
echo "  kubectl apply -f k8s/dataset-generator-job.yaml -n $NAMESPACE"
