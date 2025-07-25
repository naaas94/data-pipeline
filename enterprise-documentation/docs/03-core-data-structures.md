# Enhanced Core Data Structures

## PCC Pipeline Data Models

The enhanced Privacy Case Classifier data pipeline uses sophisticated data models that support advanced feature engineering, multi-modal embeddings, and enterprise-grade validation.

### **1. Enhanced Training Data Schema**

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Dict, Any, Optional

class PrivacyIntentRecord(BaseModel):
    """Enhanced record for privacy intent classification."""
    
    # Core fields
    text: str = Field(..., min_length=1, max_length=10000, description="Privacy intent text")
    intent: str = Field(..., pattern="^(privacy_request|data_deletion|opt_out|other)$")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence score")
    timestamp: datetime = Field(..., description="Record generation timestamp")
    
    # Enhanced text features (25+ features)
    text_length: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=1)
    avg_word_length: float = Field(..., ge=0.0)
    
    # Privacy-specific features
    deletion_keywords: int = Field(..., ge=0)
    request_keywords: int = Field(..., ge=0)
    opt_out_keywords: int = Field(..., ge=0)
    data_keywords: int = Field(..., ge=0)
    urgency_keywords: int = Field(..., ge=0)
    privacy_keyword_density: float = Field(..., ge=0.0, le=1.0)
    
    # Sentiment features
    sentiment_positive: float = Field(..., ge=0.0, le=1.0)
    sentiment_negative: float = Field(..., ge=0.0, le=1.0)
    sentiment_neutral: float = Field(..., ge=0.0, le=1.0)
    sentiment_compound: float = Field(..., ge=-1.0, le=1.0)
    
    # Linguistic features
    formality_score: float = Field(..., ge=0.0, le=1.0)
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    readability_score: float = Field(..., ge=0.0)
    unique_word_ratio: float = Field(..., ge=0.0, le=1.0)
    
    # Multi-modal embeddings
    embeddings: Optional[List[float]] = Field(None, description="Pre-computed embeddings")
    
    # Metadata
    has_personal_info: bool = Field(default=False)
    synthetic_generated: bool = Field(default=False)
    
    @field_validator('embeddings')
    def validate_embeddings(cls, v):
        if v is not None and len(v) != 584:  # Expected dimension for privacy domain embeddings
            raise ValueError('Embeddings must be 584-dimensional')
        return v
```

### **2. Data Lineage Models**

```python
class DatasetMetadata(BaseModel):
    """Complete metadata for dataset tracking."""
    
    dataset_id: str = Field(..., description="SHA-256 hash of dataset content")
    name: str = Field(..., description="Human-readable dataset name")
    stage: str = Field(..., description="Pipeline stage that produced this dataset")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Dataset characteristics
    shape: List[int] = Field(..., description="[rows, columns]")
    columns: List[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Column data types")
    memory_usage: int = Field(..., description="Memory usage in bytes")
    
    # Quality metrics
    null_counts: Dict[str, int] = Field(..., description="Null counts per column")
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Quality check results")
    
    # Processing metadata
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    engine_used: Optional[str] = Field(None, description="Processing engine")
    
    # Sample data for reference
    sample_data: Dict[str, Any] = Field(..., description="Small sample for debugging")

class StageExecution(BaseModel):
    """Pipeline stage execution tracking."""
    
    stage_id: str = Field(..., description="Unique stage identifier")
    name: str = Field(..., description="Stage name")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Dependencies
    inputs: List[str] = Field(..., description="Input dataset IDs")
    outputs: List[str] = Field(..., description="Output dataset IDs")
    
    # Execution details
    parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed)$")
    
    # Error handling
    error_info: Optional[str] = Field(None, description="Error details if failed")
    retry_count: int = Field(default=0, description="Number of retries")
```

### **3. Pipeline Contracts**

```python
class ColumnSchema(BaseModel):
    """Schema definition for pipeline contract columns."""
    
    name: str = Field(..., description="Column name")
    type: str = Field(..., pattern="^(string|int|float|datetime|boolean|array|object)$")
    nullable: bool = Field(default=True, description="Whether nulls are allowed")
    description: str = Field(..., description="Column description")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Validation constraints")

class QualityRequirements(BaseModel):
    """Data quality requirements for contract validation."""
    
    completeness: float = Field(..., ge=0.0, le=1.0, description="Minimum completeness ratio")
    validity: float = Field(..., ge=0.0, le=1.0, description="Minimum validity ratio")
    uniqueness: float = Field(..., ge=0.0, le=1.0, description="Minimum uniqueness ratio")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Minimum consistency ratio")
    timeliness: Optional[int] = Field(None, description="Maximum age in hours")

class TrainingDataContract(BaseModel):
    """Contract between data pipeline and training pipeline."""
    
    contract_id: str = Field(..., description="Unique contract identifier")
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Semantic version")
    pipeline_source: str = Field(default="data_pipeline")
    pipeline_target: str = Field(default="training_pipeline")
    
    # Schema definition
    schema: List[ColumnSchema] = Field(..., description="Expected data schema")
    quality_requirements: QualityRequirements = Field(..., description="Quality thresholds")
    
    # Size expectations
    expected_size_range: Dict[str, int] = Field(..., description="Expected dataset size")
    target_column: str = Field(..., description="ML target column")
    feature_columns: List[str] = Field(..., description="Feature columns")
    
    # Embeddings metadata
    embeddings_info: Optional[Dict[str, Any]] = Field(None, description="Embedding specifications")
    
    @field_validator('expected_size_range')
    def validate_size_range(cls, v):
        if 'min' not in v or 'max' not in v:
            raise ValueError('Size range must have min and max keys')
        if v['min'] > v['max']:
            raise ValueError('Min size cannot be greater than max size')
        return v
```

### **4. Enhanced Configuration Models**

```python
class ProcessingConfig(BaseModel):
    """Processing engine configuration."""
    
    engine: str = Field(default="pandas", pattern="^(pandas|spark|ray|beam)$")
    distributed: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "num_workers": 4,
        "memory_per_worker": "2g"
    })

class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    extract_text_features: bool = Field(default=True)
    feature_extractor: Dict[str, bool] = Field(default_factory=lambda: {
        "privacy_keywords": True,
        "sentiment_analysis": True,
        "linguistic_features": True,
        "statistical_features": True
    })

class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""
    
    generate_embeddings: bool = Field(default=True)
    embedding_type: str = Field(default="privacy_domain", 
                               pattern="^(sentence_transformer|tfidf|word2vec|privacy_domain)$")
    models: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "sentence_transformer": {
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "tfidf": {
            "max_features": 5000,
            "ngram_range": [1, 2]
        }
    })
    cache_dir: str = Field(default="cache/embeddings")

class EnhancedPipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    # Core configuration
    version: str = Field(..., description="Pipeline version")
    pipeline_name: str = Field(..., description="Pipeline identifier")
    description: str = Field(..., description="Pipeline description")
    
    # Processing configuration
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Enhanced synthetic data
    synthetic_data: Dict[str, Any] = Field(default_factory=lambda: {
        "variation_level": 0.3,
        "intent_distribution": {
            "privacy_request": 0.25,
            "data_deletion": 0.20,
            "opt_out": 0.25,
            "other": 0.30
        }
    })
    
    # Lineage and contracts
    lineage: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "storage_dir": "metadata/lineage",
        "track_datasets": True,
        "track_stages": True
    })
    
    contracts: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "validation_enabled": True,
        "contracts_dir": "contracts"
    })
```

### **5. Validation Results Models**

```python
class ValidationResults(BaseModel):
    """Comprehensive validation results."""
    
    # Schema validation
    schema_validation: Dict[str, Any] = Field(..., description="Schema check results")
    
    # Quality validation
    quality_checks: Dict[str, Any] = Field(..., description="Quality check results")
    
    # Contract validation
    contract_validation: Dict[str, Any] = Field(..., description="Contract compliance")
    
    # Overall result
    overall_valid: bool = Field(..., description="Combined validation result")
    validation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Detailed metrics
    metrics: Dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

class FeatureImportanceResults(BaseModel):
    """Feature importance analysis results."""
    
    feature_scores: Dict[str, float] = Field(..., description="Feature importance scores")
    top_features: List[str] = Field(..., description="Top N most important features")
    method: str = Field(..., description="Importance calculation method")
    target_column: str = Field(..., description="Target column for importance")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
```

## Data Processing Workflows

### **Enhanced Data Generation Workflow**

```python
@dataclass
class SyntheticDataGenerationParams:
    """Parameters for enhanced synthetic data generation."""
    
    n_samples: int = 10000
    variation_level: float = 0.3
    intent_distribution: Dict[str, float] = field(default_factory=lambda: {
        "privacy_request": 0.25,
        "data_deletion": 0.20,
        "opt_out": 0.25,
        "other": 0.30
    })
    include_metadata: bool = True
    business_hours_bias: bool = True
    temporal_patterns: bool = True
```

### **Feature Engineering Workflow**

```python
@dataclass
class FeatureExtractionResult:
    """Results from feature extraction process."""
    
    original_columns: int
    extracted_features: int
    total_columns: int
    extraction_time: float
    feature_categories: Dict[str, List[str]]
    feature_importance: Optional[Dict[str, float]] = None
```

### **Embedding Generation Workflow**

```python
@dataclass
class EmbeddingGenerationResult:
    """Results from embedding generation process."""
    
    embedding_type: str
    dimensions: int
    generation_time: float
    model_info: Dict[str, Any]
    cache_path: str
    statistics: Dict[str, float]
```

## Usage Examples

### **Creating Training Data Records**

```python
# Enhanced record creation
record = PrivacyIntentRecord(
    text="I would like to delete my personal data immediately",
    intent="data_deletion",
    confidence=0.95,
    timestamp=datetime.now(),
    text_length=47,
    word_count=9,
    deletion_keywords=2,
    urgency_score=0.8,
    sentiment_compound=0.2,
    embeddings=[0.1, -0.2, 0.3, ...],  # 584 dimensions
    has_personal_info=False,
    synthetic_generated=True
)
```

### **Contract Validation**

```python
# Validate data against contract
contract = TrainingDataContract(
    contract_id="pcc-training-data-v1",
    version="1.0.0",
    schema=[...],
    quality_requirements=QualityRequirements(
        completeness=0.99,
        validity=0.98,
        uniqueness=0.95,
        consistency=0.96
    )
)

validation_results = contracts_manager.validate_training_data(df)
```

### **Lineage Tracking**

```python
# Track dataset in lineage
dataset_id = lineage_tracker.track_dataset(
    data=df,
    dataset_name="enhanced_features",
    stage="feature_engineering",
    metadata={
        "feature_count": 62,
        "embedding_dimensions": 584,
        "quality_score": 0.97
    }
)
```

These enhanced data structures provide the foundation for enterprise-grade data processing with complete traceability, type safety, and quality assurance.

 