# Enhanced Ingestion Layer

The enhanced ingestion layer in the PCC data pipeline provides sophisticated data generation and ingestion capabilities designed specifically for privacy intent classification training data preparation.

## Enhanced Input Sources

### **1. Advanced Synthetic Data Generation**

The pipeline's primary strength lies in its sophisticated synthetic data generation capabilities, designed to create highly realistic privacy intent classification training data.

#### **Template-Based Generation**
```python
class PrivacyTextGenerator:
    """Advanced privacy intent text generator with templates and variations."""
    
    intent_templates = {
        'privacy_request': {
            'formal': [
                "I would like to request access to my personal data",
                "Please provide me with a copy of all personal information",
                "I am requesting to see what personal data you have collected"
            ],
            'informal': [
                "What info do you have about me?",
                "Can I see my data please?",
                "Show me my personal info"
            ],
            'urgent': [
                "I need to urgently access my personal data",
                "URGENT: Please send me all my information immediately"
            ]
        }
        # Similar templates for other intents...
    }
```

**Enhanced Features:**
- **30+ templates per intent** with realistic variations
- **Style diversity**: Formal, informal, urgent, angry variations
- **Realistic noise**: Typos, abbreviations, emotional language
- **Personal information injection**: Emails, phone numbers, account IDs
- **Temporal realism**: Business hours, weekday patterns, seasonal variation

#### **Confidence Modeling**
```python
def _generate_confidence_score(self, intent: str, text: str) -> float:
    """Generate realistic confidence based on text characteristics."""
    base_confidence = self.confidence_patterns[intent]['mean']
    
    # Adjust for formal language indicators
    formal_count = sum(1 for indicator in ['please', 'would like', 'gdpr'] 
                      if indicator in text.lower())
    confidence += formal_count * 0.05
    
    # Adjust for privacy keywords
    privacy_count = sum(1 for keyword in ['delete', 'access', 'data'] 
                       if keyword in text.lower())
    confidence += privacy_count * 0.02
    
    return max(0.0, min(1.0, confidence))
```

### **2. Traditional Data Sources**

#### **File-Based Ingestion**
```python
# Enhanced CSV/Parquet loading with lineage tracking
def load_data(self) -> pd.DataFrame:
    with self.logger.time_operation("data_loading"):
        if source_type == 'csv':
            df = pd.read_csv(path)
            
            # Track loaded dataset with metadata
            dataset_id = self.lineage_tracker.track_dataset(
                df, "csv_data", "data_loading", 
                metadata={
                    "source_path": path,
                    "file_size": os.path.getsize(path),
                    "encoding": "utf-8"
                }
            )
```

#### **Streaming Data Ingestion**
```python
# Kafka integration with enhanced error handling
def _load_from_kafka(self, kafka_config: Dict[str, Any]) -> pd.DataFrame:
    consumer = KafkaConsumer(
        kafka_config.get('topic'),
        bootstrap_servers=kafka_config.get('bootstrap_servers'),
        auto_offset_reset='earliest',
        value_deserializer=lambda x: x.decode('utf-8')
    )
    
    data = []
    for message in consumer:
        try:
            # Parse and validate each message
            parsed_data = self._parse_kafka_message(message.value)
            if self._validate_message(parsed_data):
                data.append(parsed_data)
        except Exception as e:
            self.logger.warning("Failed to parse Kafka message", error=str(e))
```

#### **Cloud Data Warehouse Integration**
```python
# BigQuery integration with query optimization
def _load_from_bigquery(self, bq_config: Dict[str, Any]) -> pd.DataFrame:
    client = bigquery.Client(project=bq_config.get('project_id'))
    
    query = f"""
    SELECT 
        text,
        intent,
        confidence,
        timestamp,
        metadata
    FROM `{bq_config.get('dataset')}.{bq_config.get('table')}`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    AND intent IN ('privacy_request', 'data_deletion', 'opt_out', 'other')
    ORDER BY timestamp DESC
    LIMIT {bq_config.get('limit', 10000)}
    """
    
    return client.query(query).to_dataframe()
```

## Enhanced Validation and Preprocessing

### **Multi-Layer Validation Pipeline**

#### **1. Schema Validation with Pydantic**
```python
class PrivacyIntentSchema(BaseModel):
    """Comprehensive schema for privacy intent data."""
    
    text: str = Field(..., min_length=1, max_length=10000)
    intent: str = Field(..., pattern="^(privacy_request|data_deletion|opt_out|other)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    
    @field_validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        
        # Check for suspicious patterns
        if len(re.findall(r'[A-Z]{10,}', v)) > 0:
            raise ValueError('Text contains excessive capital letters')
        
        return v.strip()
    
    @field_validator('timestamp')
    def validate_timestamp_range(cls, v):
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        
        # Check if timestamp is too old (> 5 years)
        if v < datetime.now() - timedelta(days=5*365):
            raise ValueError('Timestamp is too old')
        
        return v
```

#### **2. Advanced Quality Checks**
```python
class EnhancedDataQualityChecker:
    """Advanced quality checking with privacy-specific rules."""
    
    def check_privacy_specific_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Privacy intent specific quality checks."""
        quality_issues = []
        
        # Check intent distribution
        intent_distribution = df['intent'].value_counts(normalize=True)
        if intent_distribution.max() > 0.8:
            quality_issues.append("Intent distribution is highly imbalanced")
        
        # Check for realistic confidence patterns
        high_conf_mask = df['confidence'] > 0.9
        if high_conf_mask.sum() > len(df) * 0.8:
            quality_issues.append("Too many high confidence predictions")
        
        # Validate text-confidence correlation
        privacy_intents = df[df['intent'].isin(['privacy_request', 'data_deletion', 'opt_out'])]
        if privacy_intents['confidence'].mean() < 0.6:
            quality_issues.append("Privacy intents have unexpectedly low confidence")
        
        return {
            'privacy_quality_passed': len(quality_issues) == 0,
            'issues': quality_issues,
            'intent_distribution': intent_distribution.to_dict(),
            'avg_confidence_by_intent': df.groupby('intent')['confidence'].mean().to_dict()
        }
```

#### **3. Business Rule Validation**
```python
class BusinessRuleValidator:
    """Privacy-specific business rule validation."""
    
    def validate_privacy_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data meets privacy compliance requirements."""
        violations = []
        
        # Check for PII in text (should be anonymized)
        pii_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        texts_with_pii = df[df['text'].str.contains(pii_pattern, regex=True)]
        if len(texts_with_pii) > 0:
            violations.append(f"Found {len(texts_with_pii)} texts containing email addresses")
        
        # Check temporal patterns for realism
        if 'timestamp' in df.columns:
            weekend_ratio = self._calculate_weekend_ratio(df['timestamp'])
            if weekend_ratio > 0.4:  # Too many weekend requests
                violations.append("Unrealistic temporal distribution (too many weekend requests)")
        
        return {
            'compliance_passed': len(violations) == 0,
            'violations': violations,
            'pii_detection_count': len(texts_with_pii) if 'texts_with_pii' in locals() else 0
        }
```

### **Advanced Preprocessing Pipeline**

#### **1. Text Preprocessing with Privacy Focus**
```python
class PrivacyTextPreprocessor:
    """Specialized text preprocessing for privacy intents."""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize privacy intent text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle common abbreviations in privacy context
        privacy_abbreviations = {
            'gdpr': 'General Data Protection Regulation',
            'ccpa': 'California Consumer Privacy Act',
            'pii': 'personally identifiable information',
            'asap': 'as soon as possible'
        }
        
        for abbrev, full_form in privacy_abbreviations.items():
            text = re.sub(rf'\b{abbrev}\b', full_form, text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation while preserving urgency markers
        text = re.sub(r'[!]{3,}', '!!', text)  # Limit exclamation marks
        text = re.sub(r'[?]{2,}', '?', text)   # Limit question marks
        
        return text
    
    def detect_privacy_urgency(self, text: str) -> float:
        """Detect urgency level in privacy requests."""
        urgency_indicators = [
            'urgent', 'immediately', 'asap', 'emergency', 'right away',
            'time sensitive', 'deadline', 'expires', 'urgent!'
        ]
        
        urgency_count = sum(1 for indicator in urgency_indicators 
                           if indicator in text.lower())
        
        # Check for ALL CAPS (indicates urgency)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        
        urgency_score = min(1.0, (urgency_count * 0.3) + (caps_ratio * 0.7))
        return urgency_score
```

#### **2. Feature-Rich Data Preparation**
```python
def preprocess_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive preprocessing for training data."""
    
    # Track preprocessing stage
    input_dataset_id = self.lineage_tracker.track_dataset(
        df, "raw_input", "preprocessing"
    )
    
    stage_id = self.lineage_tracker.track_stage(
        "comprehensive_preprocessing",
        inputs=[input_dataset_id],
        outputs=[],
        parameters={
            "text_preprocessing": True,
            "feature_engineering": True,
            "quality_validation": True
        }
    )
    
    try:
        # Text preprocessing
        df['text'] = df['text'].apply(self.privacy_preprocessor.preprocess_text)
        
        # Extract comprehensive features
        if self.config.get('features', {}).get('extract_text_features', True):
            features_df = self.text_feature_extractor.extract_all_features(df['text'].tolist())
            df = pd.concat([df, features_df], axis=1)
        
        # Generate embeddings
        if self.config.get('embeddings', {}).get('generate_embeddings', True):
            embeddings = self.embedding_generator.generate_privacy_domain_embeddings(df['text'].tolist())
            df['embeddings'] = [emb.tolist() for emb in embeddings]
        
        # Add metadata features
        df['processing_timestamp'] = datetime.now()
        df['pipeline_version'] = self.config.get('version', '1.0.0')
        
        # Track processed dataset
        output_dataset_id = self.lineage_tracker.track_dataset(
            df, "preprocessed_training_data", "preprocessing",
            metadata={
                "features_added": len(features_df.columns) if 'features_df' in locals() else 0,
                "embeddings_generated": 'embeddings' in df.columns,
                "preprocessing_time": time.time() - start_time
            }
        )
        
        self.lineage_tracker.update_stage_completion(stage_id, processing_time, 'completed')
        return df
        
    except Exception as e:
        self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
        raise
```

## Integration with Downstream Pipelines

### **Training Pipeline Contract Validation**
```python
def validate_for_training_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data meets training pipeline requirements."""
    
    # Load training pipeline contract
    training_contract = self.contracts_manager.training_data_contract
    
    # Validate schema compliance
    schema_validation = self.contracts_manager.validate_training_data(df)
    
    # Additional training-specific checks
    training_specific_checks = {
        'sufficient_samples_per_intent': self._check_sample_distribution(df),
        'embedding_quality': self._validate_embedding_quality(df),
        'feature_completeness': self._check_feature_completeness(df),
        'temporal_distribution': self._validate_temporal_patterns(df)
    }
    
    return {
        'contract_validation': schema_validation,
        'training_specific': training_specific_checks,
        'ready_for_training': all([
            schema_validation['passed'],
            all(training_specific_checks.values())
        ])
    }
```

### **Output Format Optimization**
```python
def save_for_training_pipeline(self, df: pd.DataFrame, output_path: str):
    """Save data optimized for training pipeline consumption."""
    
    # Save main dataset
    df.to_parquet(output_path, compression='snappy', index=False)
    
    # Save embeddings separately for memory efficiency
    if 'embeddings' in df.columns:
        embeddings_array = np.array(df['embeddings'].tolist())
        np.save(output_path.replace('.parquet', '_embeddings.npy'), embeddings_array)
    
    # Save feature metadata
    feature_metadata = {
        'feature_names': [col for col in df.columns if col not in ['text', 'intent', 'embeddings']],
        'embedding_dimension': len(df['embeddings'].iloc[0]) if 'embeddings' in df.columns else None,
        'intent_mapping': {intent: idx for idx, intent in enumerate(df['intent'].unique())},
        'generation_timestamp': datetime.now().isoformat(),
        'pipeline_version': self.config.get('version', '1.0.0')
    }
    
    with open(output_path.replace('.parquet', '_metadata.json'), 'w') as f:
        json.dump(feature_metadata, f, indent=2)
```

The enhanced ingestion layer provides a sophisticated foundation for high-quality training data preparation with complete traceability, validation, and optimization for downstream ML pipelines.

 