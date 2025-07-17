 esto l"""
Enterprise Data Pipeline for Privacy Intent Classification
Supports distributed processing, streaming, validation, and cloud integration.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.sampling import AdvancedSampler
from validators.schema_validator import SchemaValidator
from validators.quality_checks import DataQualityChecker

# Import distributed processing libraries
try:
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("PySpark not available. Using pandas for processing.")

try:
    import ray
    from ray import data
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Using pandas for processing.")

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka not available. Streaming disabled.")

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("BigQuery not available. Cloud integration disabled.")


class EnterpriseDataPipeline:
    """Enterprise-grade data pipeline with multiple processing engines."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = get_logger("data_pipeline", self.config)
        self.processing_engine = self.config.get('processing', {}).get('engine', 'pandas')
        self.streaming_enabled = self.config.get('processing', {}).get('streaming', {}).get('enabled', False)
        
        # Initialize processing engines
        self._init_processing_engines()
        
        # Initialize components
        self.sampler = AdvancedSampler(self.config)
        self.schema_validator = SchemaValidator(self.config)
        self.quality_checker = DataQualityChecker(self.config)
        
        self.logger.info("Enterprise Data Pipeline initialized", 
                        processing_engine=self.processing_engine,
                        streaming_enabled=self.streaming_enabled)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    def _init_processing_engines(self):
        """Initialize distributed processing engines."""
        if self.processing_engine == 'spark' and SPARK_AVAILABLE:
            self.spark = SparkSession.builder \
                .appName("PrivacyIntentPipeline") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            self.logger.info("Spark session initialized")
        
        elif self.processing_engine == 'ray' and RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init()
            self.logger.info("Ray initialized")
        
        elif self.processing_engine == 'pandas':
            self.logger.info("Using pandas for processing")
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic privacy intent classification data."""
        np.random.seed(42)
        
        # Define intent categories and their probabilities
        intents = ['privacy_request', 'data_deletion', 'opt_out', 'other']
        intent_probs = [0.3, 0.25, 0.25, 0.2]
        
        # Generate synthetic texts
        privacy_texts = [
            "I want to delete my personal data",
            "Please remove my information from your database",
            "I request the deletion of my account",
            "Delete all my personal information",
            "I want to opt out of data collection",
            "Please stop collecting my data",
            "I don't want my data to be processed",
            "Remove me from your mailing list",
            "I want to know what data you have about me",
            "Please provide a copy of my personal data"
        ]
        
        other_texts = [
            "Thank you for your service",
            "I have a question about your products",
            "Can you help me with my account?",
            "I need technical support",
            "What are your business hours?",
            "How do I contact customer service?",
            "I want to update my information",
            "Please send me more details",
            "I'm interested in your services",
            "Can you explain your privacy policy?"
        ]
        
        data = []
        for i in range(n_samples):
            intent = np.random.choice(intents, p=intent_probs)
            
            if intent in ['privacy_request', 'data_deletion', 'opt_out']:
                text = np.random.choice(privacy_texts)
                confidence = np.random.beta(2, 1)  # Higher confidence for privacy-related
            else:
                text = np.random.choice(other_texts)
                confidence = np.random.beta(1, 2)  # Lower confidence for other
            
            # Add some noise to texts
            if np.random.random() < 0.3:
                text += f" {np.random.choice(['urgently', 'asap', 'immediately', 'please'])}"
            
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            data.append({
                'text': text,
                'intent': intent,
                'confidence': confidence,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} synthetic samples")
        return df
    
    def load_data(self) -> Union[pd.DataFrame, Any]:
        """Load data from configured source."""
        data_source = self.config.get('data_source', {})
        source_type = data_source.get('type', 'synthetic')
        
        with self.logger.time_operation("data_loading"):
            if source_type == 'synthetic':
                n_samples = self.config.get('sampling', {}).get('n', 10000)
                return self.generate_synthetic_data(n_samples)
            
            elif source_type == 'csv':
                path = data_source.get('path')
                if not path:
                    raise ValueError("CSV path not specified in config")
                return pd.read_csv(path)
            
            elif source_type == 'parquet':
                path = data_source.get('path')
                if not path:
                    raise ValueError("Parquet path not specified in config")
                return pd.read_parquet(path)
            
            elif source_type == 'kafka' and KAFKA_AVAILABLE:
                return self._load_from_kafka(data_source)
            
            elif source_type == 'bigquery' and BIGQUERY_AVAILABLE:
                return self._load_from_bigquery(data_source)
            
            else:
                raise ValueError(f"Unsupported data source type: {source_type}")
    
    def _load_from_kafka(self, kafka_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from Kafka stream."""
        consumer = KafkaConsumer(
            kafka_config.get('topic'),
            bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
            group_id=kafka_config.get('group_id', 'data-pipeline-consumer'),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: x.decode('utf-8')
        )
        
        data = []
        max_messages = 10000  # Limit for demo
        
        for message in consumer:
            if len(data) >= max_messages:
                break
            # Parse message and add to data list
            # This is a simplified version - in production you'd have proper message parsing
            data.append({'text': message.value, 'timestamp': datetime.now()})
        
        consumer.close()
        return pd.DataFrame(data)
    
    def _load_from_bigquery(self, bq_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from BigQuery."""
        client = bigquery.Client(project=bq_config.get('project_id'))
        
        query = f"""
        SELECT *
        FROM `{bq_config.get('project_id')}.{bq_config.get('dataset')}.{bq_config.get('table')}`
        LIMIT 10000
        """
        
        return client.query(query).to_dataframe()
    
    def process_data(self, df: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
        """Process data using configured engine."""
        with self.logger.time_operation("data_processing"):
            if self.processing_engine == 'spark' and SPARK_AVAILABLE:
                return self._process_with_spark(df)
            elif self.processing_engine == 'ray' and RAY_AVAILABLE:
                return self._process_with_ray(df)
            else:
                return self._process_with_pandas(df)
    
    def _process_with_spark(self, df: pd.DataFrame) -> Any:
        """Process data using Spark."""
        # Convert pandas DataFrame to Spark DataFrame
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("intent", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
        
        spark_df = self.spark.createDataFrame(df, schema)
        
        # Apply transformations
        processed_df = spark_df \
            .filter(F.col("text").isNotNull()) \
            .filter(F.length(F.col("text")) > 0) \
            .withColumn("text_length", F.length(F.col("text"))) \
            .withColumn("word_count", F.size(F.split(F.col("text"), " ")))
        
        return processed_df
    
    def _process_with_ray(self, df: pd.DataFrame) -> Any:
        """Process data using Ray."""
        # Convert to Ray dataset
        ray_df = data.from_pandas(df)
        
        # Apply transformations
        processed_df = ray_df \
            .filter(lambda row: row["text"] is not None and len(row["text"]) > 0) \
            .map(lambda row: {
                **row,
                "text_length": len(row["text"]),
                "word_count": len(row["text"].split())
            })
        
        return processed_df
    
    def _process_with_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data using pandas."""
        # Basic cleaning
        processed_df = df.copy()
        processed_df = processed_df.dropna(subset=['text'])
        processed_df = processed_df[processed_df['text'].str.len() > 0]
        
        # Add derived features
        processed_df['text_length'] = processed_df['text'].str.len()
        processed_df['word_count'] = processed_df['text'].str.split().str.len()
        
        return processed_df
    
    def validate_data(self, df: Union[pd.DataFrame, Any]) -> Dict[str, Any]:
        """Validate data using schema and quality checks."""
        with self.logger.time_operation("data_validation"):
            # Convert to pandas if needed
            if hasattr(df, 'toPandas'):
                df_pandas = df.toPandas()
            elif hasattr(df, 'to_pandas'):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
            
            # Schema validation
            schema_results = self.schema_validator.validate_schema(df_pandas)
            
            # Quality checks
            quality_results = self.quality_checker.check_quality(df_pandas)
            
            # Log results
            self.logger.info("Validation completed",
                           schema_valid=schema_results['overall_valid'],
                           quality_score=quality_results['overall_score'])
            
            return {
                'schema_validation': schema_results,
                'quality_checks': quality_results,
                'overall_valid': schema_results['overall_valid'] and quality_results['overall_passed']
            }
    
    def sample_data(self, df: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
        """Sample data using configured strategy."""
        with self.logger.time_operation("data_sampling"):
            # Convert to pandas if needed
            if hasattr(df, 'toPandas'):
                df_pandas = df.toPandas()
            elif hasattr(df, 'to_pandas'):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
            
            sampled_df = self.sampler.sample_data(df_pandas)
            
            # Log sampling statistics
            stats = self.sampler.get_sampling_stats(sampled_df, 'intent')
            self.logger.info("Sampling completed", **stats)
            
            return sampled_df
    
    def save_data(self, df: Union[pd.DataFrame, Any]):
        """Save processed data to configured output."""
        output_config = self.config.get('output', {})
        output_format = output_config.get('format', 'parquet')
        output_path = output_config.get('path', 'output/curated_training_data.parquet')
        
        with self.logger.time_operation("data_saving"):
            # Convert to pandas if needed
            if hasattr(df, 'toPandas'):
                df_pandas = df.toPandas()
            elif hasattr(df, 'to_pandas'):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_format == 'csv':
                df_pandas.to_csv(output_path, index=False)
            elif output_format == 'parquet':
                df_pandas.to_parquet(output_path, index=False, compression='snappy')
            elif output_format == 'bigquery' and BIGQUERY_AVAILABLE:
                self._save_to_bigquery(df_pandas, output_config.get('bigquery', {}))
            else:
                df_pandas.to_parquet(output_path, index=False, compression='snappy')
            
            self.logger.info(f"Data saved to {output_path}", 
                           format=output_format,
                           rows=len(df_pandas))
    
    def _save_to_bigquery(self, df: pd.DataFrame, bq_config: Dict[str, Any]):
        """Save data to BigQuery."""
        client = bigquery.Client(project=bq_config.get('project_id'))
        
        table_id = f"{bq_config.get('project_id')}.{bq_config.get('dataset')}.{bq_config.get('table')}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bq_config.get('write_disposition', 'WRITE_TRUNCATE')
        )
        
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
    
    def run_pipeline(self):
        """Run the complete data pipeline."""
        try:
            self.logger.info("Starting enterprise data pipeline")
            
            # Load data
            df = self.load_data()
            self.logger.log_metric('records_processed', len(df) if hasattr(df, '__len__') else 0, {'stage': 'load'})
            
            # Process data
            processed_df = self.process_data(df)
            self.logger.log_metric('records_processed', len(processed_df) if hasattr(processed_df, '__len__') else 0, {'stage': 'process'})
            
            # Validate data
            validation_results = self.validate_data(processed_df)
            if not validation_results['overall_valid']:
                self.logger.error("Data validation failed", validation_results=validation_results)
                return False
            
            # Sample data
            sampled_df = self.sample_data(processed_df)
            self.logger.log_metric('records_processed', len(sampled_df) if hasattr(sampled_df, '__len__') else 0, {'stage': 'sample'})
            
            # Save data
            self.save_data(sampled_df)
            
            self.logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("Pipeline failed", error=str(e), error_type=type(e).__name__)
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()


def main():
    """Main entry point for the data pipeline."""
    parser = argparse.ArgumentParser(description="Enterprise Data Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--sample-only", action="store_true", help="Only run sampling")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnterpriseDataPipeline(args.config)
    
    try:
        if args.validate_only:
            df = pipeline.load_data()
            processed_df = pipeline.process_data(df)
            results = pipeline.validate_data(processed_df)
            print("Validation Results:", results)
        elif args.sample_only:
            df = pipeline.load_data()
            processed_df = pipeline.process_data(df)
            sampled_df = pipeline.sample_data(processed_df)
            stats = pipeline.sampler.get_sampling_stats(sampled_df, 'intent')
            print("Sampling Results:", stats)
        else:
            success = pipeline.run_pipeline()
            if not success:
                sys.exit(1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main() 