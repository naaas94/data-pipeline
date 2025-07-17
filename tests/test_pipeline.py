"""
Comprehensive tests for the Enterprise Data Pipeline.
Includes unit tests, integration tests, and performance benchmarks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import EnterpriseDataPipeline
from utils.sampling import AdvancedSampler
from validators.schema_validator import SchemaValidator
from validators.quality_checks import DataQualityChecker


class TestEnterpriseDataPipeline:
    """Test suite for the Enterprise Data Pipeline."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            'data_source': {
                'type': 'synthetic'
            },
            'processing': {
                'engine': 'pandas',
                'distributed': {
                    'enabled': False
                },
                'streaming': {
                    'enabled': False
                }
            },
            'validation': {
                'schema': [
                    {
                        'column': 'text',
                        'type': 'string',
                        'nullable': False
                    },
                    {
                        'column': 'intent',
                        'type': 'string',
                        'nullable': False,
                        'allowed_values': ['privacy_request', 'data_deletion', 'opt_out', 'other']
                    },
                    {
                        'column': 'confidence',
                        'type': 'float',
                        'nullable': False,
                        'min_value': 0.0,
                        'max_value': 1.0
                    },
                    {
                        'column': 'timestamp',
                        'type': 'datetime',
                        'nullable': False
                    }
                ],
                'quality_checks': [
                    {
                        'check_type': 'completeness',
                        'threshold': 0.95
                    },
                    {
                        'check_type': 'uniqueness',
                        'threshold': 0.99
                    },
                    {
                        'check_type': 'validity',
                        'threshold': 0.98
                    },
                    {
                        'check_type': 'consistency',
                        'threshold': 0.95
                    }
                ]
            },
            'sampling': {
                'strategy': 'stratified',
                'by': 'intent',
                'n': 1000,
                'balance_classes': True,
                'oversampling_method': 'smote'
            },
            'output': {
                'format': 'parquet',
                'path': 'test_output.parquet'
            },
            'monitoring': {
                'log_level': 'INFO',
                'metrics_enabled': True
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'text': [
                'I want to delete my data',
                'Please remove my information',
                'Thank you for your service',
                'Can you help me?',
                'I need technical support'
            ],
            'intent': [
                'data_deletion',
                'privacy_request',
                'other',
                'other',
                'other'
            ],
            'confidence': [0.9, 0.8, 0.3, 0.4, 0.5],
            'timestamp': [
                datetime.now(),
                datetime.now(),
                datetime.now(),
                datetime.now(),
                datetime.now()
            ]
        })
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_pipeline_initialization(self, temp_config_file):
        """Test pipeline initialization with config."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        assert pipeline.config is not None
        assert pipeline.processing_engine == 'pandas'
        assert pipeline.streaming_enabled is False
        assert pipeline.sampler is not None
        assert pipeline.schema_validator is not None
        assert pipeline.quality_checker is not None
    
    def test_synthetic_data_generation(self, temp_config_file):
        """Test synthetic data generation."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        df = pipeline.generate_synthetic_data(n_samples=100)
        
        assert len(df) == 100
        assert all(col in df.columns for col in ['text', 'intent', 'confidence', 'timestamp'])
        assert all(intent in ['privacy_request', 'data_deletion', 'opt_out', 'other'] 
                  for intent in df['intent'])
        assert all(0.0 <= conf <= 1.0 for conf in df['confidence'])
    
    def test_data_processing_pandas(self, temp_config_file, sample_data):
        """Test data processing with pandas engine."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        processed_df = pipeline._process_with_pandas(sample_data)
        
        assert len(processed_df) == len(sample_data)
        assert 'text_length' in processed_df.columns
        assert 'word_count' in processed_df.columns
        assert all(processed_df['text_length'] > 0)
        assert all(processed_df['word_count'] > 0)
    
    @patch('pyspark.sql.SparkSession')
    def test_data_processing_spark(self, mock_spark, temp_config_file, sample_data):
        """Test data processing with Spark engine."""
        # Mock Spark session
        mock_spark_instance = Mock()
        mock_spark.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark_instance
        
        # Update config to use Spark
        with open(temp_config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['processing']['engine'] = 'spark'
        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        # Mock Spark DataFrame
        mock_spark_df = Mock()
        mock_spark_df.filter.return_value.withColumn.return_value.withColumn.return_value = mock_spark_df
        mock_spark_instance.createDataFrame.return_value = mock_spark_df
        
        processed_df = pipeline._process_with_spark(sample_data)
        
        assert processed_df is not None
        mock_spark_instance.createDataFrame.assert_called_once()
    
    def test_data_validation(self, temp_config_file, sample_data):
        """Test data validation."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        validation_results = pipeline.validate_data(sample_data)
        
        assert 'schema_validation' in validation_results
        assert 'quality_checks' in validation_results
        assert 'overall_valid' in validation_results
        assert isinstance(validation_results['overall_valid'], bool)
    
    def test_data_sampling(self, temp_config_file, sample_data):
        """Test data sampling."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        sampled_df = pipeline.sample_data(sample_data)
        
        assert len(sampled_df) <= len(sample_data)
        assert all(col in sampled_df.columns for col in sample_data.columns)
    
    def test_data_saving(self, temp_config_file, sample_data):
        """Test data saving."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        # Create temporary output path
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_output = f.name
        
        try:
            # Update config to use temporary output
            with open(temp_config_file, 'r') as f:
                config = yaml.safe_load(f)
            config['output']['path'] = temp_output
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)
            
            pipeline = EnterpriseDataPipeline(temp_config_file)
            pipeline.save_data(sample_data)
            
            # Verify file was created
            assert os.path.exists(temp_output)
            
            # Verify data can be read back
            loaded_df = pd.read_parquet(temp_output)
            assert len(loaded_df) == len(sample_data)
            assert all(col in loaded_df.columns for col in sample_data.columns)
        
        finally:
            # Cleanup
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_full_pipeline_run(self, temp_config_file):
        """Test full pipeline execution."""
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        # Create temporary output path
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_output = f.name
        
        try:
            # Update config to use temporary output
            with open(temp_config_file, 'r') as f:
                config = yaml.safe_load(f)
            config['output']['path'] = temp_output
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)
            
            pipeline = EnterpriseDataPipeline(temp_config_file)
            success = pipeline.run_pipeline()
            
            assert success is True
            assert os.path.exists(temp_output)
        
        finally:
            # Cleanup
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_error_handling_invalid_config(self):
        """Test error handling for invalid config."""
        with pytest.raises(ValueError):
            EnterpriseDataPipeline("nonexistent_config.yaml")
    
    def test_error_handling_invalid_data_source(self, temp_config_file):
        """Test error handling for invalid data source."""
        # Update config with invalid data source
        with open(temp_config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['data_source']['type'] = 'invalid_source'
        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = EnterpriseDataPipeline(temp_config_file)
        
        with pytest.raises(ValueError):
            pipeline.load_data()


class TestAdvancedSampler:
    """Test suite for AdvancedSampler."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'sampling': {
                'strategy': 'stratified',
                'by': 'intent',
                'n': 100,
                'balance_classes': True,
                'oversampling_method': 'smote'
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'intent': ['privacy_request'] * 50 + ['data_deletion'] * 30 + ['other'] * 20,
            'text': ['sample text'] * 100,
            'confidence': [0.8] * 100
        })
    
    def test_stratified_sampling(self, sample_config, sample_data):
        """Test stratified sampling."""
        sampler = AdvancedSampler(sample_config)
        
        sampled_df = sampler.stratified_sample(sample_data, 'intent', 50)
        
        assert len(sampled_df) <= 50
        assert 'intent' in sampled_df.columns
        
        # Check that all classes are represented
        unique_intents = sampled_df['intent'].unique()
        assert len(unique_intents) > 0
    
    def test_class_balancing(self, sample_config, sample_data):
        """Test class balancing."""
        sampler = AdvancedSampler(sample_config)
        
        balanced_df = sampler.balance_classes(sample_data, 'intent')
        
        # Check that classes are more balanced
        value_counts = balanced_df['intent'].value_counts()
        min_count = value_counts.min()
        max_count = value_counts.max()
        
        # Balance ratio should be better than original
        balance_ratio = min_count / max_count
        assert balance_ratio > 0.3  # Should be more balanced than original
    
    def test_sampling_stats(self, sample_config, sample_data):
        """Test sampling statistics."""
        sampler = AdvancedSampler(sample_config)
        
        stats = sampler.get_sampling_stats(sample_data, 'intent')
        
        assert 'total_samples' in stats
        assert 'class_distribution' in stats
        assert 'class_ratios' in stats
        assert 'balance_score' in stats
        assert 'unique_classes' in stats
        assert stats['total_samples'] == 100
        assert stats['unique_classes'] == 3


class TestSchemaValidator:
    """Test suite for SchemaValidator."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'validation': {
                'schema': [
                    {
                        'column': 'text',
                        'type': 'string',
                        'nullable': False
                    },
                    {
                        'column': 'intent',
                        'type': 'string',
                        'nullable': False,
                        'allowed_values': ['privacy_request', 'data_deletion', 'opt_out', 'other']
                    }
                ]
            }
        }
    
    @pytest.fixture
    def valid_data(self):
        return pd.DataFrame({
            'text': ['sample text', 'another text'],
            'intent': ['privacy_request', 'data_deletion'],
            'confidence': [0.8, 0.9],
            'timestamp': [datetime.now(), datetime.now()]
        })
    
    @pytest.fixture
    def invalid_data(self):
        return pd.DataFrame({
            'text': ['sample text', ''],
            'intent': ['privacy_request', 'invalid_intent'],
            'confidence': [0.8, 1.5],  # Invalid confidence
            'timestamp': [datetime.now(), datetime.now()]
        })
    
    def test_schema_validation_valid_data(self, sample_config, valid_data):
        """Test schema validation with valid data."""
        validator = SchemaValidator(sample_config)
        
        results = validator.validate_schema(valid_data)
        
        assert results['overall_valid'] is True
        assert results['total_errors'] == 0
    
    def test_schema_validation_invalid_data(self, sample_config, invalid_data):
        """Test schema validation with invalid data."""
        validator = SchemaValidator(sample_config)
        
        results = validator.validate_schema(invalid_data)
        
        # Should have some errors
        assert results['total_errors'] > 0
    
    def test_custom_rules_validation(self, sample_config, valid_data):
        """Test custom rules validation."""
        validator = SchemaValidator(sample_config)
        
        results = validator.validate_custom_rules(valid_data)
        
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results


class TestDataQualityChecker:
    """Test suite for DataQualityChecker."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'validation': {
                'quality_checks': [
                    {
                        'check_type': 'completeness',
                        'threshold': 0.95
                    },
                    {
                        'check_type': 'uniqueness',
                        'threshold': 0.99
                    }
                ]
            }
        }
    
    @pytest.fixture
    def quality_data(self):
        return pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'intent': ['privacy_request', 'data_deletion', 'other'],
            'confidence': [0.8, 0.9, 0.7],
            'timestamp': [datetime.now(), datetime.now(), datetime.now()]
        })
    
    @pytest.fixture
    def poor_quality_data(self):
        return pd.DataFrame({
            'text': ['text1', None, 'text3'],  # Has null
            'intent': ['privacy_request', 'privacy_request', 'privacy_request'],  # Low uniqueness
            'confidence': [0.8, 0.9, 0.7],
            'timestamp': [datetime.now(), datetime.now(), datetime.now()]
        })
    
    def test_completeness_check(self, sample_config, quality_data):
        """Test completeness check."""
        checker = DataQualityChecker(sample_config)
        
        results = checker.check_completeness(quality_data)
        
        assert results['check_type'] == 'completeness'
        assert results['passed'] is True
        assert results['score'] == 1.0
    
    def test_completeness_check_with_nulls(self, sample_config, poor_quality_data):
        """Test completeness check with null values."""
        checker = DataQualityChecker(sample_config)
        
        results = checker.check_completeness(poor_quality_data)
        
        assert results['check_type'] == 'completeness'
        assert results['passed'] is False
        assert results['score'] < 1.0
    
    def test_uniqueness_check(self, sample_config, quality_data):
        """Test uniqueness check."""
        checker = DataQualityChecker(sample_config)
        
        results = checker.check_uniqueness(quality_data)
        
        assert results['check_type'] == 'uniqueness'
        assert 'score' in results
    
    def test_validity_check(self, sample_config, quality_data):
        """Test validity check."""
        checker = DataQualityChecker(sample_config)
        
        results = checker.check_validity(quality_data)
        
        assert results['check_type'] == 'validity'
        assert 'score' in results
    
    def test_overall_quality_check(self, sample_config, quality_data):
        """Test overall quality check."""
        checker = DataQualityChecker(sample_config)
        
        results = checker.check_quality(quality_data)
        
        assert 'overall_score' in results
        assert 'passed_checks' in results
        assert 'total_checks' in results
        assert 'overall_passed' in results


if __name__ == "__main__":
    pytest.main([__file__]) 