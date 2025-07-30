"""
Daily Conversations Pipeline with Local Vector Store
Simple pipeline for storing and searching daily conversations.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import os
import sys

# Setup project paths for imports
from src.utils.path_setup import setup_project_paths
setup_project_paths()

from src.utils.logger import get_logger
from src.utils.vector_store import LocalVectorStore, create_daily_conversation_data
from src.features.embeddings import EmbeddingGenerator
from src.validators.schema_validator import SchemaValidator
from src.validators.quality_checks import DataQualityChecker


class DailyConversationsPipeline:
    """Simple pipeline for daily conversations with local vector store."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = get_logger("daily_conversations", self.config)
        
        # Initialize components
        self.vector_store = LocalVectorStore(
            self.config.get('vector_store', {}).get('store_path', 'vector_store')
        )
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.schema_validator = SchemaValidator(self.config)
        self.quality_checker = DataQualityChecker(self.config)
        
        self.logger.info("Daily Conversations Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    def generate_daily_conversations(self, n_conversations: Optional[int] = None) -> List[Dict]:
        """Generate daily conversations."""
        if n_conversations is None:
            n_conversations = self.config.get('source', {}).get('conversations_per_day', 50)
        
        self.logger.info(f"Generating {n_conversations} daily conversations")
        
        # Use the utility function to create conversation data
        conversations, embeddings = create_daily_conversation_data(n_conversations)
        
        # Add conversations to vector store
        total_conversations = self.vector_store.add_conversations(conversations, embeddings)
        
        self.logger.info(f"Added {len(conversations)} conversations to vector store. Total: {total_conversations}")
        
        return conversations
    
    def search_similar_conversations(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for similar conversations."""
        if top_k is None:
            top_k = self.config.get('vector_store', {}).get('search_top_k', 5)
        
        self.logger.info(f"Searching for conversations similar to: {query_text[:50]}...")
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_sentence_embeddings([query_text])
        
        if query_embedding.size == 0:
            self.logger.warning("Failed to generate embedding for query")
            return []
        
        # Search vector store
        results = self.vector_store.search_similar(query_embedding[0], top_k)
        
        self.logger.info(f"Found {len(results)} similar conversations")
        return results
    
    def get_daily_stats(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily statistics."""
        stats = self.vector_store.get_daily_stats(target_date)
        self.logger.info(f"Daily stats for {stats['date']}: {stats['total_conversations']} conversations")
        return stats
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get vector store information."""
        info = self.vector_store.get_store_info()
        self.logger.info(f"Vector store info: {info['current_conversations']} conversations, {info['current_embeddings']} embeddings")
        return info
    
    def cleanup_old_conversations(self, days_to_keep: Optional[int] = None):
        """Clean up old conversations."""
        if days_to_keep is None:
            days_to_keep = self.config.get('vector_store', {}).get('cleanup_old_days', 30)
        
        remaining = self.vector_store.cleanup_old_conversations(days_to_keep)
        self.logger.info(f"Cleanup completed. {remaining} conversations remaining")
        return remaining
    
    def validate_conversations(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Validate conversation data."""
        if not conversations:
            return {'valid': False, 'error': 'No conversations provided'}
        
        # Convert to DataFrame for validation
        df = pd.DataFrame(conversations)
        
        # Schema validation
        schema_results = self.schema_validator.validate_schema(df)
        
        # Quality checks
        quality_results = self.quality_checker.check_quality(df)
        
        return {
            'schema_validation': schema_results,
            'quality_checks': quality_results,
            'valid': schema_results['overall_valid'] and quality_results['overall_passed']
        }
    
    def run_daily_pipeline(self):
        """Run the complete daily pipeline."""
        try:
            self.logger.info("Starting daily conversations pipeline")
            
            # Generate daily conversations
            conversations = self.generate_daily_conversations()
            
            # Validate conversations
            validation_results = self.validate_conversations(conversations)
            
            if not validation_results['valid']:
                self.logger.error("Conversation validation failed")
                return False
            
            # Get daily stats
            stats = self.get_daily_stats()
            
            # Cleanup old conversations (optional)
            if self.config.get('vector_store', {}).get('auto_cleanup', False):
                self.cleanup_old_conversations()
            
            self.logger.info("Daily conversations pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Daily conversations pipeline failed: {e}")
            return False
    
    def run_search_demo(self):
        """Run a search demonstration."""
        self.logger.info("Running search demonstration")
        
        # Sample queries
        sample_queries = [
            "I need help with my account",
            "How do I delete my data?",
            "What's your privacy policy?",
            "I want to opt out of emails",
            "Can you help me with billing?"
        ]
        
        for query in sample_queries:
            print(f"\n--- Searching for: '{query}' ---")
            results = self.search_similar_conversations(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity_score']:.3f}")
                print(f"   Text: {result['text']}")
                print(f"   User: {result['user_id']}")
                print()


def main():
    """Main entry point for daily conversations pipeline."""
    parser = argparse.ArgumentParser(description="Daily Conversations Vector Store Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--generate", action="store_true", help="Generate daily conversations")
    parser.add_argument("--search", type=str, help="Search for similar conversations")
    parser.add_argument("--stats", action="store_true", help="Show daily statistics")
    parser.add_argument("--info", action="store_true", help="Show vector store information")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old conversations")
    parser.add_argument("--demo", action="store_true", help="Run search demonstration")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DailyConversationsPipeline(args.config)
    
    try:
        if args.generate:
            pipeline.run_daily_pipeline()
        elif args.search:
            results = pipeline.search_similar_conversations(args.search)
            print(f"Found {len(results)} similar conversations:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['text']} (similarity: {result['similarity_score']:.3f})")
        elif args.stats:
            stats = pipeline.get_daily_stats()
            print("Daily Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        elif args.info:
            info = pipeline.get_store_info()
            print("Vector Store Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        elif args.cleanup:
            remaining = pipeline.cleanup_old_conversations()
            print(f"Cleanup completed. {remaining} conversations remaining.")
        elif args.demo:
            pipeline.run_search_demo()
        else:
            # Default: run daily pipeline
            success = pipeline.run_daily_pipeline()
            if not success:
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 