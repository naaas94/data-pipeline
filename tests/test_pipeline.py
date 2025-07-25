"""
Comprehensive tests for the Enterprise Data Pipeline.
Includes unit tests, integration tests, and performance benchmarks.
"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import yaml
import os
import prometheus_client

from src.pcc_pipeline import PCCDataPipeline
from src.features.text_features import TextFeatureEngineer
from src.utils.sampling import AdvancedSampler

class TestPCCDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary config file and a sample pipeline instance."""
        # Clear the Prometheus registry to avoid errors
        for collector in list(prometheus_client.REGISTRY._collector_to_names.keys()):
            prometheus_client.REGISTRY.unregister(collector)

        self.config_data = {
            'processing': {'engine': 'pandas'},
            'data_source': {'type': 'synthetic'},
            'output': {'format': 'parquet', 'path': 'test_output.parquet'},
            'features': {'engineer_text_features': True},
            'embeddings': {'generate_embeddings': False},
            'sampling': {'strategy': 'random', 'n': 100},
            'logging': {'level': 'INFO'},
            'monitoring': {
                'metrics_enabled': True
            }
        }
        self.config_path = 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
            
        self.pipeline = PCCDataPipeline(self.config_path)

    def tearDown(self):
        """Remove the temporary config file."""
        os.remove(self.config_path)

    def test_pipeline_initialization(self):
        """Test if the pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, PCCDataPipeline)
        self.assertEqual(self.pipeline.processing_engine, 'pandas')

    @patch('src.pcc_pipeline.EnhancedSyntheticDataGenerator.generate_dataset')
    def test_synthetic_data_generation(self, mock_generate):
        """Test the synthetic data generation step."""
        mock_df = pd.DataFrame({'text': ["sample text"], 'intent': ["deletion"]})
        mock_generate.return_value = mock_df
        
        df = self.pipeline.generate_synthetic_data(n_samples=1)
        self.assertTrue(mock_generate.called)
        self.assertEqual(len(df), 1)

    @patch('src.pcc_pipeline.TextFeatureEngineer.extract_all_features')
    def test_feature_engineering(self, mock_extract_all_features):
        """Test the feature engineering logic."""
        mock_extract_all_features.return_value = pd.DataFrame()
        df = pd.DataFrame({'text': ["delete my account", "i need my data"]})
        self.pipeline.process_data(df)
        self.assertTrue(mock_extract_all_features.called)


    @patch('src.pcc_pipeline.PCCDataPipeline.save_data')
    @patch('src.pcc_pipeline.PCCDataPipeline.sample_data')
    @patch('src.pcc_pipeline.PCCDataPipeline.validate_data')
    @patch('src.pcc_pipeline.PCCDataPipeline.process_data')
    @patch('src.pcc_pipeline.PCCDataPipeline.load_data')
    def test_run_pipeline(self, mock_load, mock_process, mock_validate, mock_sample, mock_save):
        """Test the full pipeline run."""
        mock_load.return_value = pd.DataFrame({'text': ['test']})
        mock_process.return_value = pd.DataFrame({'text': ['test'], 'text_length': [4]})
        mock_validate.return_value = {'overall_valid': True}
        mock_sample.return_value = pd.DataFrame({'text': ['test'], 'text_length': [4]})
        
        success = self.pipeline.run_pipeline()
        
        self.assertTrue(success)


class TestAdvancedSampler(unittest.TestCase):

    def setUp(self):
        self.config = {
            'sampling': {
                'strategy': 'stratified',
                'n': 3,
                'stratify_by': 'intent',
                'random_state': 42
            }
        }
        self.sampler = AdvancedSampler(self.config)
        self.df = pd.DataFrame({
            'intent': ['A', 'A', 'B', 'B', 'C', 'C'],
            'data': range(6)
        })

    def test_stratified_sampling(self):
        """Test stratified sampling logic."""
        sampled_df = self.sampler.sample_data(self.df)
        self.assertEqual(len(sampled_df), 3)
        self.assertEqual(len(sampled_df['intent'].unique()), 3)


if __name__ == '__main__':
    unittest.main() 