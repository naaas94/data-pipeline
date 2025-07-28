from datetime import date
from src.data.synthetic_generator import EnhancedSyntheticDataGenerator, upload_to_gcs
from src.features.embeddings import EmbeddingGenerator
import pandas as pd
import tempfile

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

    # Use temporary files to avoid printing
    with tempfile.NamedTemporaryFile(delete=False) as train_file, \
         tempfile.NamedTemporaryFile(delete=False) as val_file, \
         tempfile.NamedTemporaryFile(delete=False) as test_file:

        train_df.to_csv(train_file.name, index=False)
        val_df.to_csv(val_file.name, index=False)
        test_df.to_csv(test_file.name, index=False)

        # Upload to GCS
        today = date.today().strftime('%Y%m%d')
        upload_to_gcs(train_file.name, bucket_name, f'training_{today}.csv')
        upload_to_gcs(val_file.name, bucket_name, f'validation_{today}.csv')
        upload_to_gcs(test_file.name, bucket_name, f'test_{today}.csv')

if __name__ == "__main__":
    # Example usage
    generator = EnhancedSyntheticDataGenerator()
    train_df, val_df, test_df = generator.generate_dataset_splits()
    process_and_upload_datasets(train_df, val_df, test_df) 