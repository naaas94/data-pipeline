from datetime import date
from src.data.synthetic_generator import EnhancedSyntheticDataGenerator, upload_to_gcs
from src.features.embeddings import EmbeddingGenerator
import pandas as pd

def process_and_upload_dataset(df, bucket_name='pcc-datasets'):
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()

    # Generate embeddings for the dataset
    embeddings = embedding_generator.generate_privacy_domain_embeddings(df['text'].tolist())

    # Add embeddings to the DataFrame
    df = df.assign(embeddings=list(embeddings))

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