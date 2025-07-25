# PCC Data Pipeline - Data Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An enterprise-grade data preparation pipeline for privacy intent classification, designed as part of a microservices ML architecture.**

## 🎯 What This Project IS

This is the **first pipeline** in a **3-pipeline PCC Data Pipeline ecosystem**:

```
📊 Data Pipeline (THIS) → 🤖 Training Pipeline → 🚀 Inference Pipeline
    ↓                        ↓                     ↓
Data Preparation        Model Training        Real-time Classification
Feature Engineering     Hyperparameter Tuning    Model Serving
Quality Validation      Model Evaluation      Prediction API
Synthetic Data Gen.     Model Versioning      A/B Testing
```

### **Core Purpose**
- **Generate** high-quality synthetic privacy intent data
- **Extract** advanced NLP features for downstream ML training
- **Validate** data quality with enterprise-grade checks
- **Prepare** training-ready datasets with embeddings
- **Track** complete data lineage for compliance and debugging

### **Key Capabilities**
✅ **Advanced Text Features** - 25+ NLP features (sentiment, privacy keywords, linguistic patterns)  
✅ **Multi-Modal Embeddings** - Sentence transformers + TF-IDF with domain-specific weighting  
✅ **Enhanced Synthetic Data** - Template-based generation with realistic variations  
✅ **Complete Data Lineage** - Full provenance tracking for regulatory compliance  
✅ **Pipeline Contracts** - Type-safe interfaces between microservices  
✅ **Enterprise Validation** - Schema + quality + business rules checking  
✅ **Distributed Processing** - Spark, Ray, Beam support for scale  

## 🚫 What This Project is NOT

- ❌ **Not a complete ML solution** - This handles data prep only
- ❌ **Not model training** - Use the companion training pipeline
- ❌ **Not inference/serving** - Use the companion inference pipeline  
- ❌ **Not a monolithic ML platform** - Designed for microservices architecture

## 🏗️ Architecture & Design Philosophy

### **Microservices ML Pattern**
This follows **production ML best practices** by separating concerns:
- **Data Pipeline**: Focus on data quality and preparation
- **Training Pipeline**: Focus on model development and evaluation  
- **Inference Pipeline**: Focus on serving and monitoring

### **Enterprise-Grade Features**
- **Observability**: Complete lineage tracking with Prometheus metrics
- **Quality Gates**: Multi-layer validation with configurable thresholds
- **Contracts**: Type-safe interfaces between pipeline boundaries
- **Scalability**: Multiple processing engines (Spark, Ray, Beam, Pandas)
- **Compliance**: Audit trails and metadata for regulatory requirements

## 🚀 Quick Start

### **1. Installation**
```bash
git clone <repository-url>
cd enterprise-data-pipeline

# Install dependencies
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### **2. Basic Usage**
```bash
# Generate training dataset with all features
python src/data_pipeline.py --config config.yaml

# Extract only text features
python src/data_pipeline.py --config config.yaml --features-only

# Generate only embeddings  
python src/data_pipeline.py --config config.yaml --embeddings-only

# Validate data quality
python src/data_pipeline.py --config config.yaml --validate-only
```

### **3. Outputs**
The pipeline generates:
- **Training Dataset**: `output/curated_training_data.parquet`
- **Embeddings**: `output/embeddings/` (multiple formats)
- **Lineage Data**: `metadata/lineage/` (JSON reports)
- **Contracts**: `contracts/` (JSON schemas)
- **Documentation**: `metadata/contract_documentation.md`

## 📊 Data Features

### **Generated Text Features (25+)**
```python
# Privacy-specific features
- deletion_keywords, request_keywords, opt_out_keywords
- data_keywords, urgency_keywords, privacy_keyword_density

# Sentiment & emotion
- sentiment_positive, sentiment_negative, sentiment_compound
- formal_language_score, urgency_score

# Linguistic patterns  
- text_length, word_count, avg_word_length, sentence_count
- stopword_ratio, unique_word_ratio, readability_score

# Security & PII detection
- personal_info_detected, email_pattern, phone_pattern
```

### **Embedding Types**
- **Sentence Transformers**: `all-MiniLM-L6-v2` (384 dimensions)
- **TF-IDF**: Statistical features with SVD reduction (200 dimensions)  
- **Privacy Domain**: Weighted combination (584 dimensions)

### **Synthetic Data Realism**
- **Template-based generation** with intent-specific patterns
- **Realistic variations**: typos, urgency, formality levels
- **Temporal patterns**: business hours, weekday bias
- **Confidence modeling**: based on text characteristics

## ⚙️ Configuration

```yaml
# Enhanced Features
features:
  extract_text_features: true
  feature_extractor:
    privacy_keywords: true
    sentiment_analysis: true
    linguistic_features: true

embeddings:
  generate_embeddings: true
  embedding_type: "privacy_domain"
  models:
    sentence_transformer:
      model_name: "all-MiniLM-L6-v2"

synthetic_data:
  variation_level: 0.3
  intent_distribution:
    privacy_request: 0.25
    data_deletion: 0.20
    opt_out: 0.25
    other: 0.30

# Enterprise Features
lineage:
  enabled: true
  track_datasets: true
  track_stages: true

contracts:
  enabled: true
  validation_enabled: true
```

## 🔗 PCC Ecosystem Integration

### **Pipeline Contracts**
This pipeline implements **type-safe contracts** for seamless integration:

```python
# Training Data Contract (Output)
{
  "schema": [
    {"name": "text", "type": "string", "nullable": false},
    {"name": "intent", "type": "string", "constraints": {"allowed_values": [...]}},
    {"name": "embeddings", "type": "array", "description": "Pre-computed features"}
  ],
  "quality_requirements": {
    "completeness": 0.99,
    "validity": 0.98
  }
}
```

### **Data Lineage**
Complete tracking for downstream pipelines:
- **Dataset fingerprinting** for version control
- **Quality metrics** for training pipeline validation  
- **Processing lineage** for debugging and compliance
- **Feature provenance** for model interpretability

## 🏢 Production Deployment

### **Docker**
```bash
# Build production image
docker build -t pcc-data-pipeline:latest .

# Run with volume mounts
docker run -v $(pwd)/output:/app/output pcc-data-pipeline:latest
```

### **Docker Compose**
```bash
# Full development stack
docker-compose --profile development up -d

# Production deployment
docker-compose up -d
```

### **Kubernetes**
```bash
# Deploy to cluster
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -n pcc-pipeline
```

## 📈 Monitoring & Observability

### **Metrics Dashboard**
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

### **Key Metrics**
- **Data Quality Score**: Overall pipeline health
- **Processing Throughput**: Records per second
- **Feature Coverage**: Percentage of successful feature extraction
- **Embedding Quality**: Statistical measures of embedding space

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html

# Performance benchmarks
pytest tests/test_performance.py -v
```

## 📦 Project Structure

```
enterprise-data-pipeline/
├── src/
│   ├── data_pipeline.py          # Main enhanced pipeline
│   ├── features/
│   │   ├── text_features.py      # Advanced NLP features
│   │   └── embeddings.py         # Multi-modal embeddings
│   ├── data/
│   │   └── synthetic_generator.py # Enhanced synthetic data
│   ├── contracts/
│   │   └── pipeline_contracts.py # Inter-pipeline contracts
│   ├── utils/
│   │   ├── lineage.py            # Data provenance tracking
│   │   ├── sampling.py           # Advanced sampling
│   │   └── logger.py             # Enterprise logging
│   └── validators/
│       ├── schema_validator.py   # Schema validation
│       └── quality_checks.py     # Data quality checks
├── output/                       # Generated datasets
├── metadata/                     # Lineage and contracts
├── contracts/                    # Pipeline interface definitions
├── config.yaml                  # Enhanced configuration
└── docker-compose.yml           # Development stack
```

## 🎓 Learning & Portfolio Value

### **Demonstrates ML Engineering Skills**
- **System Design**: Microservices ML architecture understanding
- **Data Engineering**: Enterprise-grade data processing pipelines
- **Quality Engineering**: Multi-layer validation and monitoring
- **DevOps Integration**: Container-ready, cloud-native design
- **Compliance**: Audit trails and regulatory considerations

### **Production-Ready Patterns**
- **Contract-Driven Development**: Type-safe pipeline interfaces
- **Observability**: Complete lineage and metrics collection
- **Scalability**: Multiple processing engines and caching
- **Maintainability**: Modular design with clear separation of concerns

## 🤝 Contributing

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### **Adding Features**
1. **Text Features**: Extend `TextFeatureEngineer` class
2. **Embeddings**: Add new embedding types to `