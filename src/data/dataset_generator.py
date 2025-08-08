import sys
import os
import yaml
from datetime import date

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.synthetic_generator import EnhancedSyntheticDataGenerator, upload_to_gcs
from src.features.embeddings import EmbeddingGenerator
import pandas as pd

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Use main config file
        config_path = 'config.yaml'
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        return {}

def process_and_upload_dataset(df, config: dict = None):
    """Process dataset and upload to GCS with configurable settings."""
    if config is None:
        config = {}
    
    # Get dataset generation settings from config
    dataset_config = config.get('dataset_generation', {})
    bucket_name = dataset_config.get('gcs_bucket', 'pcc-datasets')
    dataset_type = dataset_config.get('gcs_dataset_type', 'balanced_dataset')
    
    # Initialize the embedding generator with config
    embedding_generator = EmbeddingGenerator(config)
    
    print(f"Generating embeddings for {len(df)} samples...")
    
    # Generate embeddings for the dataset
    embeddings = embedding_generator.generate_privacy_domain_embeddings(df['text'].tolist())
    
    # Store embeddings as a JSON string in a single column
    df['embeddings'] = embeddings.tolist()
    
    # Save dataset locally with configurable settings
    today = date.today().strftime('%Y%m%d')
    file_prefix = dataset_config.get('local_file_prefix', 'df')
    file_format = 'csv'  # Default format
    dataset_file = f"{file_prefix}_{today}.{file_format}"
    
    print(f"Saving dataset to: {dataset_file}")
    df.to_csv(dataset_file, index=False)
    
    # Upload to GCS
    print(f"Uploading to GCS: gs://{bucket_name}/{dataset_type}_{today}.{file_format}")
    upload_to_gcs(dataset_file, bucket_name, dataset_type, file_format)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {len(df)}")
    print(f"- Intent distribution: {df['intent'].value_counts().to_dict()}")
    print(f"- Embedding dimension: {embeddings.shape[1]}")
    print(f"- Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")

def generate_dataset_with_config(config: dict = None):
    """Generate dataset using configuration settings."""
    if config is None:
        config = {}
    
    # Get dataset generation settings
    dataset_config = config.get('dataset_generation', {})
    n_samples = dataset_config.get('n_samples', 10000)
    min_samples_per_class = dataset_config.get('min_samples_per_class', None)
    
    print(f"Generating synthetic dataset with {n_samples} samples...")
    
    # Initialize generator with config
    generator = EnhancedSyntheticDataGenerator(config)
    
    # Generate balanced dataset
    if min_samples_per_class:
        df = generator.generate_balanced_dataset(n_samples, min_samples_per_class)
    else:
        df = generator.generate_balanced_dataset(n_samples)
    
    return df

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Generate dataset
    df = generate_dataset_with_config(config)
    
    # Process and upload
    process_and_upload_dataset(df, config)
    
    print("Dataset generation completed successfully!") 