import sys
import os
from datetime import date

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.synthetic_generator import EnhancedSyntheticDataGenerator, upload_to_gcs
from src.features.embeddings import EmbeddingGenerator
import pandas as pd

def process_and_upload_dataset(df, bucket_name='pcc-datasets'):
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()

    # Generate embeddings for the dataset
    embeddings = embedding_generator.generate_privacy_domain_embeddings(df['text'].tolist())

    # Store embeddings as a JSON string in a single column
    df['embeddings'] = embeddings.tolist()

    # Save dataset locally
    today = date.today().strftime('%Y%m%d')
    dataset_file = f"df_{today}.csv"

    df.to_csv(dataset_file, index=False)

    # Upload to GCS
    upload_to_gcs(dataset_file, bucket_name, 'balanced_dataset')

if __name__ == "__main__":
    # Example usage
    generator = EnhancedSyntheticDataGenerator()
    df = generator.generate_balanced_dataset()
    process_and_upload_dataset(df) 