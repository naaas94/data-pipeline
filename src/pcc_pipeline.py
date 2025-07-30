"""
Enterprise Data Pipeline for Privacy Intent Classification
Supports distributed processing, streaming, validation, and cloud integration.
Enhanced with advanced features: text features, embeddings, lineage tracking, and contracts.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import os
import sys

# Setup project paths for imports
from src.utils.path_setup import setup_project_paths
setup_project_paths()

from src.utils.logger import get_logger
from src.utils.sampling import AdvancedSampler
from src.utils.lineage import DataLineageTracker, DataProvenanceManager
from src.validators.schema_validator import SchemaValidator
from src.validators.quality_checks import DataQualityChecker
from src.features.text_features import TextFeatureEngineer
from src.features.embeddings import EmbeddingGenerator, save_embeddings_for_training
from src.data.synthetic_generator import EnhancedSyntheticDataGenerator
from src.contracts.pcc_contracts import PCCEcosystemContracts

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


class PCCDataPipeline:
    """Enhanced enterprise-grade data pipeline with advanced ML features."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = get_logger("data_pipeline", self.config)
        self.processing_engine = self.config.get('processing', {}).get('engine', 'pandas')
        self.streaming_enabled = self.config.get('processing', {}).get('streaming', {}).get('enabled', False)
        
        # Initialize advanced components
        self.lineage_tracker = DataLineageTracker(self.config)
        self.contracts_manager = PCCEcosystemContracts()
        self.text_feature_engineer = TextFeatureEngineer(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.enhanced_synthetic_generator = EnhancedSyntheticDataGenerator(self.config)
        
        # Initialize processing engines
        self._init_processing_engines()
        
        # Initialize traditional components
        self.sampler = AdvancedSampler(self.config)
        self.schema_validator = SchemaValidator(self.config)
        self.quality_checker = DataQualityChecker(self.config)
        
        self.logger.info("Enhanced Enterprise Data Pipeline initialized", 
                        processing_engine=self.processing_engine,
                        streaming_enabled=self.streaming_enabled,
                        lineage_tracking=True,
                        contracts_enabled=True)
    
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
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate enhanced synthetic privacy intent classification data."""
        self.logger.info("Generating enhanced synthetic data", n_samples=n_samples)
        
        # Track data generation stage
        stage_id = self.lineage_tracker.track_stage(
            "synthetic_data_generation",
            inputs=[],
            outputs=[],
            parameters={"n_samples": n_samples, "generator": "enhanced"}
        )
        
        try:
            # Use enhanced synthetic generator
            df = self.enhanced_synthetic_generator.generate_dataset(
                n_samples=n_samples,
                variation_level=self.config.get('synthetic_data', {}).get('variation_level', 0.3),
                include_metadata=True
            )
            
            # Track dataset
            dataset_id = self.lineage_tracker.track_dataset(
                df, "synthetic_data", "data_generation"
            )
            
            # Update stage completion
            import time
            self.lineage_tracker.update_stage_completion(stage_id, 0.5, 'completed')
            
            # Update stage outputs
            for stage in self.lineage_tracker.lineage_data['stages']:
                if stage['stage_id'] == stage_id:
                    stage['outputs'] = [dataset_id]
                    break
            
            self.logger.info(f"Generated {len(df)} enhanced synthetic samples")
            return df
            
        except Exception as e:
            self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
            raise
    
    def load_data(self) -> Union[pd.DataFrame, Any]:
        """Load data from configured source with lineage tracking."""
        self.logger.info("Starting enhanced data loading")
        
        data_source = self.config.get('data_source', {})
        source_type = data_source.get('type', 'synthetic')
        
        with self.logger.time_operation("data_loading"):
            if source_type == 'synthetic':
                n_samples = self.config.get('sampling', {}).get('n', 10000)
                df = self.generate_synthetic_data(n_samples)
                self.logger.info("Loaded enhanced synthetic data")
                return df
            
            elif source_type == 'csv':
                path = data_source.get('path')
                if not path:
                    raise ValueError("CSV path not specified in config")
                df = pd.read_csv(path)
                
                # Track loaded dataset
                dataset_id = self.lineage_tracker.track_dataset(
                    df, "csv_data", "data_loading", 
                    metadata={"source_path": path}
                )
                
                self.logger.info("Loaded CSV data")
                return df
            
            elif source_type == 'parquet':
                path = data_source.get('path')
                if not path:
                    raise ValueError("Parquet path not specified in config")
                df = pd.read_parquet(path)
                
                # Track loaded dataset
                dataset_id = self.lineage_tracker.track_dataset(
                    df, "parquet_data", "data_loading",
                    metadata={"source_path": path}
                )
                
                self.logger.info("Loaded Parquet data")
                return df
            
            elif source_type == 'kafka' and KAFKA_AVAILABLE:
                df = self._load_from_kafka(data_source)
                
                # Track loaded dataset
                dataset_id = self.lineage_tracker.track_dataset(
                    df, "kafka_data", "data_loading",
                    metadata={"kafka_config": data_source}
                )
                
                self.logger.info("Loaded data from Kafka")
                return df
            
            elif source_type == 'bigquery' and BIGQUERY_AVAILABLE:
                df = self._load_from_bigquery(data_source)
                
                # Track loaded dataset
                dataset_id = self.lineage_tracker.track_dataset(
                    df, "bigquery_data", "data_loading",
                    metadata={"bigquery_config": data_source}
                )
                
                self.logger.info("Loaded data from BigQuery")
                return df
            
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
    
    def engineer_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced text features from the data."""
        self.logger.info("Starting text feature engineering")
        
        # Track feature engineering stage
        input_dataset_id = self.lineage_tracker.track_dataset(
            df, "input_for_features", "feature_engineering"
        )
        
        stage_id = self.lineage_tracker.track_stage(
            "text_feature_engineering",
            inputs=[input_dataset_id],
            outputs=[],
            parameters={"feature_engineer": "advanced_nlp"}
        )
        
        try:
            with self.logger.time_operation("text_feature_engineering"):
                # Engineer text features
                texts = df['text'].tolist()
                features_df = self.text_feature_engineer.extract_all_features(texts)
                
                # Combine with original data
                enhanced_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
                
                # Track output dataset
                output_dataset_id = self.lineage_tracker.track_dataset(
                    enhanced_df, "features_engineered", "feature_engineering"
                )
                
                # Update stage completion
                self.lineage_tracker.update_stage_completion(stage_id, 1.0, 'completed')
                
                # Update stage outputs
                for stage in self.lineage_tracker.lineage_data['stages']:
                    if stage['stage_id'] == stage_id:
                        stage['outputs'] = [output_dataset_id]
                        break
                
                self.logger.info(f"Extracted {len(features_df.columns)} text features")
                return enhanced_df
                
        except Exception as e:
            self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
            raise
    
    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for the text data."""
        self.logger.info("Starting embedding generation")
        
        # Track embedding generation stage
        input_dataset_id = self.lineage_tracker.track_dataset(
            df, "input_for_embeddings", "embedding_generation"
        )
        
        stage_id = self.lineage_tracker.track_stage(
            "embedding_generation",
            inputs=[input_dataset_id],
            outputs=[],
            parameters={"embedding_type": "privacy_domain"}
        )
        
        try:
            with self.logger.time_operation("embedding_generation"):
                texts = df['text'].tolist()
                
                # Generate privacy domain embeddings
                embeddings = self.embedding_generator.generate_privacy_domain_embeddings(texts)
                
                if embeddings.size > 0:
                    # Add embeddings to dataframe
                    df_with_embeddings = df.copy()
                    
                    # Store embeddings as a list of arrays for each row
                    df_with_embeddings['embeddings'] = [emb.tolist() for emb in embeddings]
                    
                    # Save embeddings separately for training pipeline
                    saved_files = save_embeddings_for_training(
                        df, text_column='text', 
                        output_dir='output/embeddings',
                        config=self.config
                    )
                    
                    # Track output dataset
                    output_dataset_id = self.lineage_tracker.track_dataset(
                        df_with_embeddings, "embeddings_generated", "embedding_generation",
                        metadata={
                            "embedding_shape": embeddings.shape,
                            "embedding_files": saved_files,
                            "embedding_stats": self.embedding_generator.get_embedding_stats(embeddings)
                        }
                    )
                    
                    # Update stage completion
                    self.lineage_tracker.update_stage_completion(stage_id, 2.0, 'completed')
                    
                    # Update stage outputs
                    for stage in self.lineage_tracker.lineage_data['stages']:
                        if stage['stage_id'] == stage_id:
                            stage['outputs'] = [output_dataset_id]
                            break
                    
                    self.logger.info(f"Generated embeddings: {embeddings.shape}")
                    return df_with_embeddings
                else:
                    self.logger.warning("No embeddings generated")
                    return df
                    
        except Exception as e:
            self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
            self.logger.error("Embedding generation failed", error=str(e))
            return df  # Return original data if embedding fails
    
    def process_data(self, df: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
        """Process data using configured engine with enhanced features."""
        with self.logger.time_operation("data_processing"):
            if self.processing_engine == 'spark' and SPARK_AVAILABLE:
                processed_df = self._process_with_spark(df)
            elif self.processing_engine == 'ray' and RAY_AVAILABLE:
                processed_df = self._process_with_ray(df)
            else:
                processed_df = self._process_with_pandas(df)
            
            # Extract features if enabled
            if self.config.get('features', {}).get('engineer_text_features', True):
                processed_df = self.engineer_text_features(processed_df)
            
            # Generate embeddings if enabled
            if self.config.get('embeddings', {}).get('generate_embeddings', True):
                processed_df = self.generate_embeddings(processed_df)
            
            return processed_df
    
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
        """Process data using pandas with enhanced logging."""
        self.logger.info("Starting enhanced pandas processing")
        
        # Track processing stage
        input_dataset_id = self.lineage_tracker.track_dataset(
            df, "input_for_processing", "data_processing"
        )
        
        stage_id = self.lineage_tracker.track_stage(
            "pandas_processing",
            inputs=[input_dataset_id],
            outputs=[],
            parameters={"engine": "pandas", "operations": ["cleaning", "feature_engineering"]}
        )
        
        try:
            # Basic cleaning
            processed_df = df.copy()
            original_len = len(processed_df)
            
            processed_df = processed_df.dropna(subset=['text'])
            processed_df = processed_df[processed_df['text'].str.len() > 0]
            
            cleaned_len = len(processed_df)
            self.logger.info("Completed basic cleaning", 
                           original_count=original_len, 
                           cleaned_count=cleaned_len,
                           removed_count=original_len - cleaned_len)
            
            # Add derived features (if not already present)
            if 'text_length' not in processed_df.columns:
                processed_df['text_length'] = processed_df['text'].str.len()
            if 'word_count' not in processed_df.columns:
                processed_df['word_count'] = processed_df['text'].str.split().str.len()
            
            self.logger.info("Added derived features")
            
            # Track output dataset
            output_dataset_id = self.lineage_tracker.track_dataset(
                processed_df, "processed_data", "data_processing",
                metadata={
                    "cleaning_stats": {
                        "original_count": original_len,
                        "final_count": cleaned_len,
                        "removal_rate": (original_len - cleaned_len) / original_len if original_len > 0 else 0
                    }
                }
            )
            
            # Update stage completion
            self.lineage_tracker.update_stage_completion(stage_id, 0.8, 'completed')
            
            # Update stage outputs
            for stage in self.lineage_tracker.lineage_data['stages']:
                if stage['stage_id'] == stage_id:
                    stage['outputs'] = [output_dataset_id]
                    break
            
            self.logger.info("Enhanced pandas processing completed")
            return processed_df
            
        except Exception as e:
            self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
            raise
    
    def validate_data(self, df: Union[pd.DataFrame, Any]) -> Dict[str, Any]:
        """Validate data using schema and quality checks with contract validation."""
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
            
            # Contract validation
            contract_results = self.contracts_manager.validate_training_data(df_pandas)
            
            # Track quality metrics in lineage
            if hasattr(df_pandas, 'columns') and 'text' in df_pandas.columns:
                input_dataset_id = self.lineage_tracker.track_dataset(
                    df_pandas, "validation_input", "data_validation"
                )
                self.lineage_tracker.track_data_quality_metrics(
                    input_dataset_id, quality_results
                )
            
            # Log results
            self.logger.info("Enhanced validation completed",
                           schema_valid=schema_results['overall_valid'],
                           quality_score=quality_results['overall_score'],
                           contract_valid=contract_results['passed'])
            
            return {
                'schema_validation': schema_results,
                'quality_checks': quality_results,
                'contract_validation': contract_results,
                'overall_valid': (schema_results['overall_valid'] and 
                                quality_results['overall_passed'] and 
                                contract_results['passed'])
            }
    
    def sample_data(self, df: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
        """Sample data using configured strategy with lineage tracking."""
        with self.logger.time_operation("data_sampling"):
            # Convert to pandas if needed
            if hasattr(df, 'toPandas'):
                df_pandas = df.toPandas()
            elif hasattr(df, 'to_pandas'):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
            
            # Track sampling stage
            input_dataset_id = self.lineage_tracker.track_dataset(
                df_pandas, "sampling_input", "data_sampling"
            )
            
            sampling_config = self.config.get('sampling', {})
            stage_id = self.lineage_tracker.track_stage(
                "data_sampling",
                inputs=[input_dataset_id],
                outputs=[],
                parameters=sampling_config
            )
            
            try:
                sampled_df = self.sampler.sample_data(df_pandas)
                
                # Log sampling statistics
                stats = self.sampler.get_sampling_stats(sampled_df, 'intent')
                
                # Track output dataset
                output_dataset_id = self.lineage_tracker.track_dataset(
                    sampled_df, "sampled_data", "data_sampling",
                    metadata={"sampling_stats": stats}
                )
                
                # Update stage completion
                self.lineage_tracker.update_stage_completion(stage_id, 0.3, 'completed')
                
                # Update stage outputs
                for stage in self.lineage_tracker.lineage_data['stages']:
                    if stage['stage_id'] == stage_id:
                        stage['outputs'] = [output_dataset_id]
                        break
                
                self.logger.info("Sampling completed", **stats)
                return sampled_df
                
            except Exception as e:
                self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
                raise
    
    def save_data(self, df: Union[pd.DataFrame, Any]):
        """Save processed data with enhanced metadata and contracts."""
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
            
            # Track saving stage
            input_dataset_id = self.lineage_tracker.track_dataset(
                df_pandas, "final_dataset", "data_saving"
            )
            
            stage_id = self.lineage_tracker.track_stage(
                "data_saving",
                inputs=[input_dataset_id],
                outputs=[],
                parameters={"output_format": output_format, "output_path": output_path}
            )
            
            try:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save main data
                if output_format == 'csv':
                    df_pandas.to_csv(output_path, index=False)
                elif output_format == 'parquet':
                    df_pandas.to_parquet(output_path, index=False, compression='snappy')
                elif output_format == 'bigquery' and BIGQUERY_AVAILABLE:
                    self._save_to_bigquery(df_pandas, output_config.get('bigquery', {}))
                else:
                    df_pandas.to_parquet(output_path, index=False, compression='snappy')
                
                # Save metadata and contracts
                self._save_metadata(df_pandas, output_path)
                
                # Update stage completion
                self.lineage_tracker.update_stage_completion(stage_id, 0.5, 'completed')
                
                self.logger.info(f"Enhanced data saved to {output_path}", 
                               format=output_format,
                               rows=len(df_pandas),
                               columns=len(df_pandas.columns))
                               
            except Exception as e:
                self.lineage_tracker.update_stage_completion(stage_id, 0, 'failed', str(e))
                raise
    
    def _save_metadata(self, df: pd.DataFrame, output_path: str):
        """Save enhanced metadata including lineage and contracts."""
        metadata_dir = os.path.join(os.path.dirname(output_path), 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save lineage data
        lineage_path = self.lineage_tracker.save_lineage()
        
        # Save contracts
        self.contracts_manager.save_contracts()
        
        # Generate and save lineage report
        lineage_report = self.lineage_tracker.generate_lineage_report()
        report_path = os.path.join(metadata_dir, 'lineage_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(lineage_report, f, indent=2, default=str)
        
        # Generate and save contract documentation
        contract_docs = self.contracts_manager.generate_contract_documentation()
        docs_path = os.path.join(metadata_dir, 'contract_documentation.md')
        with open(docs_path, 'w') as f:
            f.write(contract_docs)
        
        self.logger.info("Metadata saved", 
                        lineage_path=lineage_path,
                        contracts_saved=True,
                        documentation_generated=True)
    
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
        """Run the complete enhanced data pipeline."""
        try:
            self.logger.info("Starting enhanced enterprise data pipeline")
            
            # Load data
            df = self.load_data()
            
            # Process data (includes feature extraction and embeddings)
            processed_df = self.process_data(df)
            
            # Validate data
            validation_results = self.validate_data(processed_df)
            if not validation_results['overall_valid']:
                self.logger.error("Enhanced data validation failed")
                return False
            
            # Sample data
            sampled_df = self.sample_data(processed_df)
            
            # Save data with metadata
            self.save_data(sampled_df)
            
            self.logger.info("Enhanced pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error("Enhanced pipeline failed", error=str(e), error_type=type(e).__name__)
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        
        # Save final lineage data
        self.lineage_tracker.save_lineage()


def main():
    """Main entry point for the enhanced data pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced Enterprise Data Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--sample-only", action="store_true", help="Only run sampling")
    parser.add_argument("--features-only", action="store_true", help="Only engineer features")
    parser.add_argument("--embeddings-only", action="store_true", help="Only generate embeddings")
    
    args = parser.parse_args()
    
    # Initialize enhanced pipeline
    pipeline = PCCDataPipeline(args.config)
    
    try:
        if args.validate_only:
            df = pipeline.load_data()
            processed_df = pipeline.process_data(df)
            results = pipeline.validate_data(processed_df)
            print("Enhanced Validation Results:", results)
        elif args.sample_only:
            df = pipeline.load_data()
            processed_df = pipeline.process_data(df)
            sampled_df = pipeline.sample_data(processed_df)
            stats = pipeline.sampler.get_sampling_stats(sampled_df, 'intent')
            print("Enhanced Sampling Results:", stats)
        elif args.features_only:
            df = pipeline.load_data()
            processed_df = pipeline.process_data(df)
            features_df = pipeline.engineer_text_features(processed_df)
            print(f"Features engineered: {features_df.shape}")
        elif args.embeddings_only:
            df = pipeline.load_data()
            embeddings_df = pipeline.generate_embeddings(df)
            print(f"Embeddings generated: {embeddings_df.shape}")
        else:
            success = pipeline.run_pipeline()
            if not success:
                sys.exit(1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main() 