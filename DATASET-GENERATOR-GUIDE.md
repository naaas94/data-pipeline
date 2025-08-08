# Dataset Generator Guide

## Overview

The `dataset_generator.py` script is a standalone tool that:
1. **Generates synthetic privacy intent data** using `EnhancedSyntheticDataGenerator`
2. **Creates embeddings** using `EmbeddingGenerator` 
3. **Saves locally** as CSV with date stamp (`df_YYYYMMDD.csv`)
4. **Uploads to Google Cloud Storage** using `upload_to_gcs`

## What You Need

### 1. Google Cloud Setup
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create service account (if needed)
gcloud iam service-accounts create dataset-generator \
    --display-name="Dataset Generator Service Account"

# Grant storage permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:dataset-generator@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Download service account key
gcloud iam service-accounts keys create credentials/service-account-key.json \
    --iam-account=dataset-generator@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 2. GCS Bucket Setup
```bash
# Create bucket
gsutil mb gs://pcc-datasets

# Set bucket permissions
gsutil iam ch allUsers:objectViewer gs://pcc-datasets
```

## Quick Start

### Option 1: Docker Compose (Local)
```bash
# 1. Set up credentials
mkdir -p credentials
# Place your service-account-key.json in credentials/

# 2. Create output directory
mkdir -p output logs

# 3. Run with Docker Compose
docker-compose -f docker-compose.dataset-generator.yml up --build

# 4. Check results
ls -la output/
gsutil ls gs://pcc-datasets/
```

### Option 2: Kubernetes Job (EKS)
```bash
# 1. Set up credentials
mkdir -p credentials
# Place your service-account-key.json in credentials/

# 2. Deploy to EKS
chmod +x deploy-dataset-generator.sh
./deploy-dataset-generator.sh

# 3. Monitor job
kubectl get jobs -n data-pipeline
kubectl logs -f job/dataset-generator-job -n data-pipeline
```

### Option 3: Direct Docker Run
```bash
# 1. Build image
docker build -f Dockerfile.dataset-generator -t dataset-generator:latest .

# 2. Run container
docker run --rm \
  -v $(pwd)/credentials:/app/credentials:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json \
  dataset-generator:latest
```

## Configuration

### Environment Variables
```bash
# Required for GCS upload
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json

# Optional
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONPATH=/app
```

### Resource Requirements
```yaml
# Docker Compose
resources:
  limits:
    memory: 4G
    cpus: '2.0'
  reservations:
    memory: 2G
    cpus: '1.0'

# Kubernetes Job
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Output Files

### Local Files
```
output/
├── df_20241201.csv          # Generated dataset with embeddings
└── logs/
    └── dataset_generator.log # Processing logs
```

### GCS Files
```
gs://pcc-datasets/
├── balanced_dataset_20241201.csv  # Uploaded dataset
└── training_20241201.csv          # If using dataset splits
```

## Dataset Structure

The generated dataset includes:

### Core Columns
- **`text`**: Privacy intent text (e.g., "Please delete my data")
- **`intent`**: Classification label (`privacy_request`, `data_deletion`, `opt_out`, `other`)
- **`confidence`**: Confidence score (0.0-1.0)
- **`timestamp`**: Generation timestamp
- **`embeddings`**: Pre-computed embeddings (JSON array)

### Metadata Columns
- **`formality_score`**: Text formality level
- **`urgency_score`**: Urgency level
- **`text_length`**: Character count
- **`word_count`**: Word count

## Customization

### Modify Dataset Size
```python
# In dataset_generator.py
if __name__ == "__main__":
    generator = EnhancedSyntheticDataGenerator()
    # Change n_samples for different dataset size
    df = generator.generate_balanced_dataset(n_samples=50000)
    process_and_upload_dataset(df)
```

### Change Intent Distribution
```python
# Custom intent distribution
intent_distribution = {
    'privacy_request': 0.4,
    'data_deletion': 0.3,
    'opt_out': 0.2,
    'other': 0.1
}
df = generator.generate_dataset(n_samples=10000, 
                              intent_distribution=intent_distribution)
```

### Custom GCS Bucket
```python
# Change bucket name
process_and_upload_dataset(df, bucket_name='my-custom-bucket')
```

## Monitoring & Debugging

### Check Job Status (Kubernetes)
```bash
# Job status
kubectl get jobs -n data-pipeline

# Pod status
kubectl get pods -n data-pipeline -l job-name=dataset-generator-job

# Logs
kubectl logs -f job/dataset-generator-job -n data-pipeline

# Describe job for details
kubectl describe job dataset-generator-job -n data-pipeline
```

### Check Container Logs (Docker)
```bash
# View logs
docker logs dataset-generator

# Execute into container
docker exec -it dataset-generator /bin/bash
```

### Verify GCS Upload
```bash
# List files in bucket
gsutil ls gs://pcc-datasets/

# Download and verify
gsutil cp gs://pcc-datasets/balanced_dataset_20241201.csv ./
head -5 balanced_dataset_20241201.csv
```

## Troubleshooting

### Common Issues

#### 1. GCS Authentication Failed
```bash
# Check credentials
ls -la credentials/service-account-key.json

# Test authentication
gcloud auth application-default print-access-token

# Verify service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:dataset-generator"
```

#### 2. Memory Issues
```bash
# Increase memory limits in docker-compose
resources:
  limits:
    memory: 8G  # Increase from 4G
    cpus: '4.0' # Increase from 2.0
```

#### 3. Job Stuck in Pending
```bash
# Check node resources
kubectl describe nodes

# Check PVC status
kubectl get pvc -n data-pipeline

# Check events
kubectl get events -n data-pipeline --sort-by='.lastTimestamp'
```

### Debug Commands
```bash
# Check if all dependencies are available
docker run --rm dataset-generator:latest python -c "
import pandas as pd
import numpy as np
import sklearn
import sentence_transformers
from google.cloud import storage
print('All dependencies available')
"

# Test GCS connection
docker run --rm \
  -v $(pwd)/credentials:/app/credentials:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json \
  dataset-generator:latest python -c "
from google.cloud import storage
client = storage.Client()
buckets = list(client.list_buckets())
print(f'Connected to GCS, found {len(buckets)} buckets')
"
```

## Scheduling

### Cron Job (Kubernetes)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dataset-generator-cron
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dataset-generator
            image: dataset-generator:latest
            # ... rest of container spec
          restartPolicy: Never
```

### Airflow DAG
```python
from airflow import DAG
from airflow.operators.kubernetes_pod_operator import KubernetesPodOperator

dag = DAG('dataset_generator', schedule_interval='@daily')

generate_dataset = KubernetesPodOperator(
    task_id='generate_dataset',
    namespace='data-pipeline',
    image='dataset-generator:latest',
    cmds=['python', 'src/data/dataset_generator.py'],
    dag=dag
)
```

## Security Considerations

### GCP Service Account
- Use least privilege principle
- Rotate keys regularly
- Monitor usage with Cloud Audit Logs

### Container Security
- Non-root user execution
- Read-only credentials mount
- Resource limits to prevent abuse

### Network Security
- No external network access needed
- All processing is local + GCS upload

## Cost Optimization

### Resource Optimization
- Use spot instances for Kubernetes jobs
- Right-size memory/CPU based on dataset size
- Clean up old datasets from GCS

### Storage Optimization
- Compress datasets before upload
- Set lifecycle policies on GCS bucket
- Use appropriate storage classes

## Integration with Main Pipeline

The dataset generator can be integrated with the main pipeline:

```python
# In main pipeline
from src.data.dataset_generator import process_and_upload_dataset

# Generate fresh dataset
generator = EnhancedSyntheticDataGenerator()
df = generator.generate_balanced_dataset()
process_and_upload_dataset(df, bucket_name='pcc-datasets')

# Use generated dataset for training
# ... rest of pipeline
```

This provides a complete solution for generating and uploading datasets that can be used by your ML training pipeline.
noteId: "e449ecd0744911f084faa31cc6172271"
tags: []

---

