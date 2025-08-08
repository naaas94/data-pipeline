# EKS Deployment Guide for Enterprise Data Pipeline

This guide provides step-by-step instructions for deploying the Privacy Intent Classification pipeline to Amazon EKS.

## System Overview

The pipeline consists of two main components:
1. **Main Data Pipeline** (`src/pcc_pipeline.py`) - Privacy intent classification with advanced NLP features
2. **Daily Conversations Pipeline** (`src/daily_conversations_pipeline.py`) - Vector store-based conversation processing

### Key Features
- **Text Feature Engineering**: 25+ NLP features (sentiment, privacy keywords, linguistic patterns)
- **Multi-Modal Embeddings**: Sentence transformers + TF-IDF (584 dimensions)
- **Synthetic Data Generation**: Template-based with realistic variations
- **Data Validation**: Enterprise-grade quality checks
- **Lineage Tracking**: Complete data provenance
- **Vector Store Integration**: Local vector store for conversation search

## Prerequisites

### Required Tools
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Docker
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### EKS Cluster Setup
```bash
# Create EKS cluster (if not exists)
eksctl create cluster \
  --name your-cluster-name \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name your-cluster-name
```

## Quick Deployment

### 1. Build and Deploy
```bash
# Make deployment script executable
chmod +x deploy-eks.sh

# Deploy to EKS
./deploy-eks.sh
```

### 2. Deploy with Custom Registry
```bash
# Deploy with specific registry
REGISTRY=123456789012.dkr.ecr.us-west-2.amazonaws.com ./deploy-eks.sh

# Deploy with specific tag
IMAGE_TAG=v1.0.0 ./deploy-eks.sh
```

### 3. Deploy to Custom Namespace
```bash
# Deploy to specific namespace
NAMESPACE=ml-pipeline ./deploy-eks.sh
```

## Manual Deployment Steps

### 1. Build Docker Image
```bash
# Build optimized image
docker build -f Dockerfile.optimized -t privacy-intent-pipeline:latest .

# Tag for registry (if using ECR)
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker tag privacy-intent-pipeline:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/privacy-intent-pipeline:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/privacy-intent-pipeline:latest
```

### 2. Create Namespace
```bash
kubectl create namespace data-pipeline
```

### 3. Apply Persistent Volume Claims
```bash
kubectl apply -f k8s/persistent-volumes.yaml -n data-pipeline
```

### 4. Deploy Applications
```bash
# Update image name in deployment if using registry
sed 's|privacy-intent-pipeline:latest|123456789012.dkr.ecr.us-west-2.amazonaws.com/privacy-intent-pipeline:latest|g' k8s/deployment.yaml | kubectl apply -f - -n data-pipeline

# Apply services
kubectl apply -f k8s/services.yaml -n data-pipeline
```

### 5. Verify Deployment
```bash
# Check pod status
kubectl get pods -n data-pipeline

# Check services
kubectl get services -n data-pipeline

# View logs
kubectl logs -f deployment/privacy-intent-pipeline -n data-pipeline
```

## Configuration

### Environment Variables
The pipeline supports the following environment variables:

```yaml
# Pipeline Configuration
ENVIRONMENT: "production"
LOG_LEVEL: "INFO"
PYTHONPATH: "/app"

# External Services
KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
MLFLOW_TRACKING_URI: "http://mlflow:5000"
PROMETHEUS_ENDPOINT: "http://prometheus:9090"
```

### Resource Requirements
```yaml
# Main Pipeline
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Daily Conversations Pipeline
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

## Monitoring and Observability

### Access Monitoring Dashboards
```bash
# Port forward to access services
kubectl port-forward service/grafana-service 3000:3000 -n data-pipeline
kubectl port-forward service/prometheus-service 9090:9090 -n data-pipeline
kubectl port-forward service/mlflow-service 5000:5000 -n data-pipeline
```

### Monitoring URLs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

### Key Metrics
- **Data Quality Score**: Overall pipeline health
- **Processing Throughput**: Records per second
- **Feature Coverage**: Percentage of successful feature extraction
- **Embedding Quality**: Statistical measures of embedding space

## Scaling and Management

### Scale Pipeline
```bash
# Scale main pipeline
kubectl scale deployment privacy-intent-pipeline --replicas=3 -n data-pipeline

# Scale daily conversations pipeline
kubectl scale deployment daily-conversations-pipeline --replicas=2 -n data-pipeline
```

### Update Deployment
```bash
# Update image
kubectl set image deployment/privacy-intent-pipeline data-pipeline=privacy-intent-pipeline:v1.1.0 -n data-pipeline

# Restart deployment
kubectl rollout restart deployment/privacy-intent-pipeline -n data-pipeline
```

### Backup and Restore
```bash
# Backup persistent volumes
kubectl exec -it <pod-name> -n data-pipeline -- tar czf /tmp/backup.tar.gz /app/output

# Copy backup from pod
kubectl cp data-pipeline/<pod-name>:/tmp/backup.tar.gz ./backup.tar.gz
```

## Troubleshooting

### Common Issues

#### 1. Pod Stuck in Pending
```bash
# Check events
kubectl describe pod <pod-name> -n data-pipeline

# Check node resources
kubectl describe nodes
```

#### 2. Pipeline Failing
```bash
# Check logs
kubectl logs -f deployment/privacy-intent-pipeline -n data-pipeline

# Check config
kubectl exec -it <pod-name> -n data-pipeline -- cat /app/config.yaml
```

#### 3. Storage Issues
```bash
# Check PVC status
kubectl get pvc -n data-pipeline

# Check storage class
kubectl get storageclass
```

### Debug Commands
```bash
# Get pod details
kubectl describe pod <pod-name> -n data-pipeline

# Execute into pod
kubectl exec -it <pod-name> -n data-pipeline -- /bin/bash

# Check resource usage
kubectl top pods -n data-pipeline
```

## Security Considerations

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pipeline-network-policy
  namespace: data-pipeline
spec:
  podSelector:
    matchLabels:
      app: privacy-intent-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: data-pipeline
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: data-pipeline
    ports:
    - protocol: TCP
      port: 9092  # Kafka
    - protocol: TCP
      port: 5000  # MLflow
    - protocol: TCP
      port: 9090  # Prometheus
```

### RBAC Configuration
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pipeline-sa
  namespace: data-pipeline
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pipeline-role
  namespace: data-pipeline
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pipeline-role-binding
  namespace: data-pipeline
subjects:
- kind: ServiceAccount
  name: pipeline-sa
  namespace: data-pipeline
roleRef:
  kind: Role
  name: pipeline-role
  apiGroup: rbac.authorization.k8s.io
```

## Cost Optimization

### Resource Optimization
```yaml
# Optimize resource requests
resources:
  requests:
    memory: "256Mi"  # Start with lower requests
    cpu: "250m"
  limits:
    memory: "1Gi"    # Set reasonable limits
    cpu: "500m"
```

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pipeline-hpa
  namespace: data-pipeline
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: privacy-intent-pipeline
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Cleanup

### Remove Deployment
```bash
# Delete deployments
kubectl delete deployment privacy-intent-pipeline -n data-pipeline
kubectl delete deployment daily-conversations-pipeline -n data-pipeline

# Delete services
kubectl delete service privacy-intent-pipeline-service -n data-pipeline
kubectl delete service daily-conversations-pipeline-service -n data-pipeline

# Delete PVCs (optional - will delete data)
kubectl delete pvc --all -n data-pipeline

# Delete namespace
kubectl delete namespace data-pipeline
```

### Remove EKS Cluster
```bash
# Delete cluster
eksctl delete cluster --name your-cluster-name --region us-west-2
```

## Support

For issues and questions:
1. Check the logs: `kubectl logs -f deployment/privacy-intent-pipeline -n data-pipeline`
2. Review the configuration: `kubectl exec -it <pod-name> -n data-pipeline -- cat /app/config.yaml`
3. Check resource usage: `kubectl top pods -n data-pipeline`
4. Verify network connectivity: `kubectl exec -it <pod-name> -n data-pipeline -- curl -v http://kafka:9092`
noteId: "95799a80744711f084faa31cc6172271"
tags: []

---

