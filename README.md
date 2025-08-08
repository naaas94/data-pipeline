# Enterprise Data Pipeline - Privacy Intent Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An enterprise-grade data preparation pipeline for privacy intent classification, designed as part of a microservices ML architecture with **optimized processing engine selection** for ML workloads.

## What This Project IS

This is the first pipeline in a 3-pipeline PCC Data Pipeline ecosystem:

```
Data Pipeline (THIS) → Training Pipeline → Inference Pipeline
    ↓                        ↓                     ↓
Data Preparation        Model Training        Real-time Classification
Feature Engineering     Hyperparameter Tuning    Model Serving
Quality Validation      Model Evaluation      Prediction API
Synthetic Data Gen.     Model Versioning      A/B Testing
```

### Core Purpose
- Generate high-quality synthetic privacy intent data
- Extract advanced NLP features for downstream ML training
- Validate data quality with enterprise-grade checks
- Prepare training-ready datasets with embeddings
- Track complete data lineage for compliance and debugging

### Key Capabilities
- **Advanced Text Features** - 25+ NLP features (sentiment, privacy keywords, linguistic patterns)
- **Multi-Modal Embeddings** - Sentence transformers + TF-IDF with domain-specific weighting
- **Enhanced Synthetic Data** - Template-based generation with realistic variations
- **Complete Data Lineage** - Full provenance tracking for regulatory compliance
- **Pipeline Contracts** - Type-safe interfaces between microservices
- **Enterprise Validation** - Schema + quality + business rules checking
- **Intelligent Processing Engine Selection** - Optimized for ML workloads (Pandas/Ray for current scale)

## What This Project is NOT

- Not a complete ML solution - This handles data prep only
- Not model training - Use the companion training pipeline
- Not inference/serving - Use the companion inference pipeline
- Not a monolithic ML platform - Designed for microservices architecture
- **Not a "big data" pipeline** - Optimized for ML-focused text processing workloads

## Architecture & Design Philosophy

### Microservices ML Pattern
This follows production ML best practices by separating concerns:
- **Data Pipeline**: Focus on data quality and preparation
- **Training Pipeline**: Focus on model development and evaluation
- **Inference Pipeline**: Focus on serving and monitoring

### Intelligent Processing Engine Selection
The pipeline uses **data-driven engine selection** based on workload characteristics:

```python
# Engine Selection Logic
if data_size > LARGE_THRESHOLD and SPARK_AVAILABLE:
    engine = SparkEngine()  # For large-scale batch processing
elif processing_type == ML_INTENSIVE and RAY_AVAILABLE:
    engine = RayEngine()    # For ML-heavy workloads
elif streaming_enabled and BEAM_AVAILABLE:
    engine = BeamEngine()   # For streaming workloads
else:
    engine = PandasEngine() # Default for ML-focused workloads
```

**Why This Approach?**
- **Current Scale**: 100-10,000 samples (optimal for Pandas/Ray)
- **Processing Pattern**: Sequential ML operations (NLP, embeddings, validation)
- **Performance**: Pandas is 10-20x faster than Spark for this workload
- **Complexity**: Avoids Spark overhead for small-to-medium datasets

### Enterprise-Grade Features
- **Observability**: Complete lineage tracking with Prometheus metrics
- **Quality Gates**: Multi-layer validation with configurable thresholds
- **Contracts**: Type-safe interfaces between pipeline boundaries
- **Scalability**: Multiple processing engines optimized for different workloads
- **Compliance**: Audit trails and metadata for regulatory requirements

## Quick Start

### Installation
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

### Basic Usage
```bash
# Generate training dataset with all features (test config - 100 samples)
python -m src.pcc_pipeline --config config_test.yaml

# Generate training dataset (production config - 10,000 samples)
python -m src.pcc_pipeline --config config.yaml

# Validate data quality only
python -m src.pcc_pipeline --config config_test.yaml --validate-only
```

### Outputs
The pipeline generates:
- **Training Dataset**: `output/curated_training_data.parquet`
- **Embeddings**: `output/embeddings/` (multiple formats)
- **Lineage Data**: `metadata/lineage/` (JSON reports)
- **Contracts**: `contracts/` (JSON schemas)
- **Documentation**: `metadata/contract_documentation.md`

## Data Features

### Generated Text Features (25+)
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

### Embedding Types
- **Sentence Transformers**: `all-MiniLM-L6-v2` (384 dimensions)
- **TF-IDF**: Statistical features with SVD reduction (200 dimensions)  
- **Privacy Domain**: Weighted combination (584 dimensions)

### Synthetic Data Realism
- Template-based generation with intent-specific patterns
- Realistic variations: typos, urgency, formality levels
- Temporal patterns: business hours, weekday bias
- Confidence modeling: based on text characteristics

## Configuration

```yaml
# Processing Engine Selection
processing:
  engine: "pandas"  # pandas, ray, spark, beam
  distributed:
    enabled: false
    num_workers: 4
    memory_per_worker: "2g"

# Enhanced Features
features:
  engineer_text_features: true
  text_features:
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

## Processing Engine Comparison

### Current Workload Characteristics
- **Data Size**: 100-10,000 samples
- **Processing Type**: ML-intensive (NLP, embeddings, validation)
- **Operations**: Sequential, stateful operations
- **Memory Usage**: 500MB-2GB

### Engine Performance Analysis

| Engine | 100 Samples | 10K Samples | Memory | Best For |
|--------|-------------|-------------|---------|----------|
| **Pandas** | 2-5s | 30-60s | 500MB-2GB | **Current workload** |
| **Ray** | 5-10s | 45-90s | 1-3GB | ML-intensive operations |
| **Spark** | 30-60s | 5-10min | 2-5GB | Large-scale batch |
| **Beam** | 10-20s | 2-5min | 1-4GB | Streaming workloads |

### Why Pandas is Optimal for This Pipeline
1. **Data Size**: Current workload (100-10K samples) is too small for Spark overhead
2. **Processing Pattern**: Sequential ML operations don't benefit from Spark's distributed model
3. **Performance**: Pandas is 10-20x faster for this scale
4. **Complexity**: Avoids JVM startup, serialization, and network communication overhead
5. **NLP Libraries**: Native Python NLP libraries work seamlessly with Pandas

## PCC Ecosystem Integration

### Pipeline Contracts
This pipeline implements type-safe contracts for seamless integration:

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

### Data Lineage
Complete tracking for downstream pipelines:
- Dataset fingerprinting for version control
- Quality metrics for training pipeline validation  
- Processing lineage for debugging and compliance
- Feature provenance for model interpretability

## Production Deployment

### CI/CD Pipeline
We've streamlined the CI/CD to focus on essentials:

```bash
# Automatic CI/CD (GitHub Actions)
# - Push to main/develop branches triggers automatic testing
# - Main branch builds and pushes Docker image
# - See .github/workflows/ci.yml for details

# Manual deployment
./deploy.sh

# Deploy to specific registry
REGISTRY=ghcr.io/your-org ./deploy.sh
```

### Docker
```bash
# Build with streamlined Dockerfile
docker build -f Dockerfile -t pcc-data-pipeline:latest .

# Run with volume mounts
docker run -v $(pwd)/output:/app/output pcc-data-pipeline:latest
```

### Docker Compose
```bash
# Full development stack
docker-compose --profile development up -d

# Production deployment
docker-compose up -d
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run pipeline
python -m src.pcc_pipeline --config config_test.yaml
```

## Monitoring & Observability

### Metrics Dashboard
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- MLflow: http://localhost:5000

### Key Metrics
- **Data Quality Score**: Overall pipeline health
- **Processing Throughput**: Records per second
- **Feature Coverage**: Percentage of successful feature extraction
- **Embedding Quality**: Statistical measures of embedding space

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html

# Code quality checks
black --check src/ tests/
flake8 src/ tests/ --max-line-length=88

# Test in Docker container
docker build -f Dockerfile -t test .
docker run --rm test python -m pytest tests/ -v
```

## Project Structure

```
enterprise-data-pipeline/
├── src/
│   ├── pcc_pipeline.py           # Main enhanced pipeline
│   ├── features/
│   │   ├── text_features.py      # Advanced NLP features
│   │   └── embeddings.py         # Multi-modal embeddings
│   ├── data/
│   │   └── synthetic_generator.py # Enhanced synthetic data
│   ├── contracts/
│   │   └── pcc_contracts.py      # Inter-pipeline contracts
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
├── config_test.yaml             # Test configuration (100 samples)
├── config.prod.yaml             # Production configuration
└── docker-compose.yml           # Development stack
```

## Performance Characteristics

### Current Performance (Optimized for ML Workloads)
- **100 samples**: ~2-5 seconds (Pandas)
- **10,000 samples**: ~30-60 seconds (Pandas)
- **Memory usage**: 500MB-2GB
- **Processing engine**: Pandas (optimal for current scale)

### Scalability Strategy
- **Current scale**: Pandas (fastest for 100-10K samples)
- **Medium scale**: Ray (for ML-intensive operations)
- **Large scale**: Spark (for 1M+ records with simple transformations)
- **Streaming**: Beam (for real-time processing)

## Learning & Portfolio Value

### Demonstrates ML Engineering Skills
- **System Design**: Microservices ML architecture understanding
- **Data Engineering**: Enterprise-grade data processing pipelines
- **Performance Optimization**: Engine selection based on workload characteristics
- **Quality Engineering**: Multi-layer validation and monitoring
- **DevOps Integration**: Container-ready, cloud-native design
- **Compliance**: Audit trails and regulatory considerations

### Production-Ready Patterns
- **Contract-Driven Development**: Type-safe pipeline interfaces
- **Observability**: Complete lineage and metrics collection
- **Intelligent Scaling**: Right tool for the job approach
- **Maintainability**: Modular design with clear separation of concerns

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding Features
1. **Text Features**: Extend `TextFeatureEngineer` class
2. **Embeddings**: Add new embedding types to `EmbeddingGenerator`
3. **Processing Engines**: Implement new engine in `PCCDataPipeline`
4. **Validation**: Add new checks to `DataQualityChecker`

## Status: Production Ready ✅

### ✅ Completed Features
- **Data Generation**: High-quality synthetic privacy intent data
- **Feature Extraction**: 25+ advanced NLP features
- **Embedding Generation**: Multi-modal embeddings (584 dimensions)
- **Data Validation**: Enterprise-grade quality checks
- **Lineage Tracking**: Complete data provenance
- **Pipeline Contracts**: Type-safe interfaces
- **Processing Optimization**: Engine selection for ML workloads
- **Docker Containerization**: Production-ready deployment
- **CI/CD Pipeline**: Automated testing and deployment

### 🔄 Future Enhancements
- **Scale Testing**: Performance testing with larger datasets
- **Additional Engines**: More processing engine options
- **Advanced Monitoring**: Enhanced observability features
- **Cloud Integration**: Native cloud service integration

---

**Note**: This pipeline is optimized for ML-focused text processing workloads. For large-scale batch processing with simple transformations, consider Apache Spark. For streaming workloads, consider Apache Beam. This pipeline demonstrates the importance of choosing the right tool for the specific workload characteristics.