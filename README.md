# Enterprise Data Pipeline for Privacy Intent Classification

A **production-ready, enterprise-grade data pipeline** for generating, validating, and curating training datasets for privacy intent classification. Built with **distributed processing**, **streaming capabilities**, **advanced validation**, and **cloud integration**.

## 🏗️ **Enterprise Architecture**

### **✅ Distributed Processing**
- **Apache Spark**: Large-scale data processing with SQL, MLlib, and Streaming
- **Apache Beam**: Portable, unified batch and streaming processing
- **Ray**: Distributed computing for ML workloads and data processing

### **✅ Streaming Capabilities**
- **Apache Kafka**: Real-time data streaming and event processing
- **Apache Pulsar**: Multi-tenant, high-performance messaging
- **Apache Flink**: Stateful stream processing with exactly-once semantics

### **✅ Data Validation**
- **Pandera**: Statistical data validation with pandas
- **Great Expectations**: Data quality monitoring and validation
- **Pydantic**: Data validation using Python type annotations

### **✅ Partitioning & Cloud Integration**
- **BigQuery**: Serverless, highly scalable data warehouse
- **dbt**: Data transformation and modeling
- **Dagster**: Data orchestration and pipeline management

## 🚀 **Quick Start**

### **1. Local Development**
```bash
# Clone repository
git clone <repository-url>
cd data-pipeline

# Install dependencies
pip install -r requirements.txt

# Run pipeline with default config
python src/data_pipeline.py --config config.yaml

# Run tests
pytest tests/ -v

# Run with specific options
python src/data_pipeline.py --config config.yaml --validate-only
python src/data_pipeline.py --config config.yaml --sample-only
```

### **2. Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up -d

# Run development environment
docker-compose --profile development up -d

# Access services
# - MLflow: http://localhost:5000
# - Spark UI: http://localhost:8080
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### **3. Kubernetes Deployment**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n data-pipeline
```

## 📊 **Pipeline Features**

### **Data Sources**
- **Synthetic Data Generation**: Realistic privacy intent data for testing
- **CSV/Parquet Files**: Local and cloud storage support
- **Apache Kafka**: Real-time streaming data ingestion
- **BigQuery**: Cloud data warehouse integration
- **Custom Connectors**: Extensible for any data source

### **Processing Engines**
- **Pandas**: Fast in-memory processing for small datasets
- **Apache Spark**: Distributed processing for large datasets
- **Ray**: Parallel processing for ML workloads
- **Apache Beam**: Portable batch and streaming processing

### **Data Validation**
- **Schema Validation**: Type checking, constraints, and business rules
- **Quality Checks**: Completeness, uniqueness, validity, consistency
- **Statistical Analysis**: Outlier detection, distribution analysis
- **Great Expectations**: Automated data quality monitoring

### **Sampling & Balancing**
- **Stratified Sampling**: Maintain class distribution
- **Class Balancing**: SMOTE, ADASYN, undersampling
- **Cross-Validation**: Train/validation/test splits
- **Custom Strategies**: Extensible sampling algorithms

### **Output Formats**
- **Parquet**: Columnar storage for analytics
- **CSV**: Universal format for compatibility
- **BigQuery**: Cloud data warehouse
- **Custom Sinks**: Extensible output formats

## 🔧 **Configuration**

### **Basic Configuration**
```yaml
# config.yaml
data_source:
  type: "synthetic"  # synthetic, csv, parquet, kafka, bigquery
  path: null

processing:
  engine: "spark"  # spark, beam, ray, pandas
  distributed:
    enabled: true
    num_workers: 4

validation:
  schema:
    - column: "text"
      type: "string"
      nullable: false
  quality_checks:
    - check_type: "completeness"
      threshold: 0.95

sampling:
  strategy: "stratified"
  by: "intent"
  n: 10000
  balance_classes: true
```

### **Advanced Configuration**
```yaml
# Enterprise features
streaming:
  enabled: true
  kafka:
    bootstrap_servers: ["localhost:9092"]
    topic: "privacy-intent-data"

monitoring:
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "privacy-intent-pipeline"
  
  prometheus:
    enabled: true
    port: 9090

partitioning:
  enabled: true
  strategy: "date"
  column: "timestamp"
```

## 📈 **Monitoring & Observability**

### **MLflow Integration**
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and deploy models
- **UI Dashboard**: Visualize experiments and results

### **Prometheus Metrics**
- **Processing Metrics**: Records processed, processing time
- **Quality Metrics**: Data quality scores, validation results
- **System Metrics**: Memory usage, CPU utilization

### **Grafana Dashboards**
- **Pipeline Health**: Real-time monitoring dashboard
- **Data Quality**: Quality metrics and trends
- **Performance**: Processing performance and bottlenecks

### **Structured Logging**
- **JSON Logs**: Machine-readable log format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Context Tracking**: Request tracing and correlation

## 🧪 **Testing**

### **Unit Tests**
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_pipeline.py -v
pytest tests/test_sampling.py -v
pytest tests/test_validation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### **Integration Tests**
```bash
# Test with real data sources
pytest tests/test_integration.py -v

# Test distributed processing
pytest tests/test_distributed.py -v

# Test streaming capabilities
pytest tests/test_streaming.py -v
```

### **Performance Tests**
```bash
# Benchmark processing performance
pytest tests/test_performance.py -v

# Load testing
pytest tests/test_load.py -v
```

## 🚀 **Deployment Options**

### **Docker**
```bash
# Production build
docker build -t privacy-intent-pipeline:latest .

# Run container
docker run -v $(pwd)/output:/app/output privacy-intent-pipeline:latest
```

### **Docker Compose**
```bash
# Full stack deployment
docker-compose up -d

# Development environment
docker-compose --profile development up -d
```

### **Kubernetes**
```bash
# Deploy to cluster
kubectl apply -f k8s/

# Scale pipeline
kubectl scale deployment data-pipeline --replicas=3
```

### **Cloud Platforms**
- **Google Cloud Platform**: BigQuery, Dataflow, Cloud Run
- **Amazon Web Services**: EMR, Kinesis, Lambda
- **Microsoft Azure**: Databricks, Event Hubs, Functions

## 📚 **API Reference**

### **Pipeline Configuration**
```python
from src.data_pipeline import EnterpriseDataPipeline

# Initialize pipeline
pipeline = EnterpriseDataPipeline("config.yaml")

# Run complete pipeline
success = pipeline.run_pipeline()

# Run specific stages
df = pipeline.load_data()
processed_df = pipeline.process_data(df)
validation_results = pipeline.validate_data(processed_df)
sampled_df = pipeline.sample_data(processed_df)
pipeline.save_data(sampled_df)
```

### **Sampling**
```python
from src.utils.sampling import AdvancedSampler

sampler = AdvancedSampler(config)
sampled_df = sampler.stratified_sample(df, 'intent', 1000)
balanced_df = sampler.balance_classes(df, 'intent')
```

### **Validation**
```python
from src.validators.schema_validator import SchemaValidator
from src.validators.quality_checks import DataQualityChecker

# Schema validation
validator = SchemaValidator(config)
schema_results = validator.validate_schema(df)

# Quality checks
checker = DataQualityChecker(config)
quality_results = checker.check_quality(df)
```

## 🔒 **Security & Compliance**

### **Data Security**
- **Encryption**: Data encryption at rest and in transit
- **Access Control**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive audit trails
- **Data Masking**: PII protection and anonymization

### **Compliance**
- **GDPR**: Privacy-by-design and data protection
- **CCPA**: California Consumer Privacy Act compliance
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd data-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### **Code Quality**
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: [Wiki](wiki-url)
- **Issues**: [GitHub Issues](issues-url)
- **Discussions**: [GitHub Discussions](discussions-url)
- **Email**: support@company.com

---

**Built with ❤️ by the ML Engineering Team**

 