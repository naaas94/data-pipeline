"""
Audit test for the generate_synthetic_data function.
Generates synthetic data, logs details, and saves to CSV for manual review.

This test ensures that the generated synthetic data is valid and meaningful by:
- Generating a specified number of synthetic samples.
- Logging the structure and a sample of the generated data.
- Saving the data to a CSV file for manual exploration.

The CSV file 'synthetic_data_audit.csv' is saved in the 'tests' directory,
allowing you to explore the generated text and verify its quality.
"""

import pandas as pd
import os
import sys
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import EnterpriseDataPipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("synthetic_data_audit")


def test_generate_synthetic_data_audit():
    """Audit synthetic data generation and save to CSV using config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    print(f"Using configuration file at: {config_path}")
    pipeline = EnterpriseDataPipeline(config_path)
    
    # Generate synthetic data
    n_samples = 1000  # Adjust the number of samples as needed
    print(f"Generating {n_samples} synthetic samples...")
    df = pipeline.generate_synthetic_data(n_samples=n_samples)
    
    # Log details about the generated data
    print(f"Generated DataFrame with {len(df)} rows.")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Save the generated data to a CSV file
    output_csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data_audit.csv')
    print(f"Saving synthetic data to: {output_csv_path}")
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Synthetic data successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

    # Assertions to ensure data integrity
    assert len(df) == n_samples
    assert all(col in df.columns for col in ['text', 'intent', 'confidence', 'timestamp'])
    assert all(intent in ['privacy_request', 'data_deletion', 'opt_out', 'other'] for intent in df['intent'])
    assert all(0.0 <= conf <= 1.0 for conf in df['confidence'])

    print("Audit test completed successfully.") 

if __name__ == "__main__":
    test_generate_synthetic_data_audit() 