from datetime import date
from src.data.synthetic_generator import EnhancedSyntheticDataGenerator, upload_to_gcs
from src.features.embeddings import EmbeddingGenerator
import pandas as pd

def process_and_upload_datasets(train_df, val_df, test_df, bucket_name='pcc-datasets'):
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()

    # Generate embeddings for each dataset
    train_embeddings = embedding_generator.generate_privacy_domain_embeddings(train_df['text'].tolist())
    val_embeddings = embedding_generator.generate_privacy_domain_embeddings(val_df['text'].tolist())
    test_embeddings = embedding_generator.generate_privacy_domain_embeddings(test_df['text'].tolist())

    # Add embeddings to the DataFrames
    train_df = train_df.assign(embeddings=list(train_embeddings))
    val_df = val_df.assign(embeddings=list(val_embeddings))
    test_df = test_df.assign(embeddings=list(test_embeddings))

    # Save datasets locally
    today = date.today().strftime('%Y%m%d')
    train_file = f"training_{today}.csv"
    val_file = f"validation_{today}.csv"
    test_file = f"test_{today}.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Upload to GCS
    upload_to_gcs(train_file, bucket_name, 'training')
    upload_to_gcs(val_file, bucket_name, 'validation')
    upload_to_gcs(test_file, bucket_name, 'test')

if __name__ == "__main__":
    # Example usage
    generator = EnhancedSyntheticDataGenerator()
    train_df, val_df, test_df = generator.generate_dataset_splits()
    process_and_upload_datasets(train_df, val_df, test_df) 