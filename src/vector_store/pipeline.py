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
from .core import LocalVectorStore, create_daily_conversation_data
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
    
    def validate_conversations(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Validate conversation data quality."""
        self.logger.info(f"Validating {len(conversations)} conversations")
        
        # Convert to DataFrame for validation
        df = pd.DataFrame(conversations)
        
        # Run schema validation
        schema_results = self.schema_validator.validate_schema(df)
        
        # Run quality checks
        quality_results = self.quality_checker.run_quality_checks(df)
        
        validation_results = {
            "schema_validation": schema_results,
            "quality_checks": quality_results,
            "total_conversations": len(conversations),
            "validation_passed": schema_results.get("valid", False) and quality_results.get("overall_score", 0) > 0.9
        }
        
        self.logger.info(f"Validation results: {validation_results['validation_passed']}")
        return validation_results
    
    def run_daily_pipeline(self):
        """Run the complete daily pipeline."""
        self.logger.info("Starting daily conversations pipeline")
        
        try:
            # Generate conversations
            conversations = self.generate_daily_conversations()
            
            # Validate data quality
            validation_results = self.validate_conversations(conversations)
            
            # Get final statistics
            stats = self.get_daily_stats()
            store_info = self.get_store_info()
            
            self.logger.info("Daily pipeline completed successfully")
            self.logger.info(f"Final stats: {stats}")
            self.logger.info(f"Store info: {store_info}")
            
            return {
                "conversations_generated": len(conversations),
                "validation_results": validation_results,
                "daily_stats": stats,
                "store_info": store_info
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_search_demo(self):
        """Run a search demonstration."""
        self.logger.info("Running search demonstration")
        
        # Sample queries for demonstration
        demo_queries = [
            "I need help with my account",
            "How do I delete my data?",
            "What's your privacy policy?",
            "I want to opt out of emails",
            "Can you help me with billing?"
        ]
        
        results = {}
        for query in demo_queries:
            search_results = self.search_similar_conversations(query, top_k=3)
            results[query] = search_results
        
        self.logger.info("Search demonstration completed")
        return results


def main():
    """Main entry point for the daily conversations pipeline."""
    parser = argparse.ArgumentParser(description="Daily Conversations Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--generate", action="store_true", help="Generate daily conversations")
    parser.add_argument("--search", type=str, help="Search for similar conversations")
    parser.add_argument("--stats", action="store_true", help="Show daily statistics")
    parser.add_argument("--info", action="store_true", help="Show store information")
    parser.add_argument("--demo", action="store_true", help="Run search demonstration")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old conversations")
    parser.add_argument("--n-conversations", type=int, help="Number of conversations to generate")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DailyConversationsPipeline(args.config)
    
    try:
        if args.generate:
            n_conv = args.n_conversations or None
            conversations = pipeline.generate_daily_conversations(n_conv)
            print(f"Generated {len(conversations)} conversations")
            
        elif args.search:
            results = pipeline.search_similar_conversations(args.search)
            print(f"Found {len(results)} similar conversations:")
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity_score']:.3f}")
                print(f"   Text: {result['text']}")
                print(f"   User: {result['user_id']}")
                
        elif args.stats:
            stats = pipeline.get_daily_stats()
            print("Daily Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        elif args.info:
            info = pipeline.get_store_info()
            print("Store Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
                
        elif args.demo:
            results = pipeline.run_search_demo()
            print("Search Demonstration Results:")
            for query, search_results in results.items():
                print(f"\nQuery: '{query}'")
                for i, result in enumerate(search_results, 1):
                    print(f"  {i}. Similarity: {result['similarity_score']:.3f} - {result['text']}")
                    
        elif args.cleanup:
            days_to_keep = pipeline.config.get('vector_store', {}).get('cleanup_old_days', 30)
            pipeline.vector_store.cleanup_old_conversations(days_to_keep)
            print(f"Cleaned up conversations older than {days_to_keep} days")
            
        else:
            # Run full pipeline
            results = pipeline.run_daily_pipeline()
            print("Pipeline completed successfully")
            print(f"Generated: {results['conversations_generated']} conversations")
            print(f"Validation passed: {results['validation_results']['validation_passed']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 