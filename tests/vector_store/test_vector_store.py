"""
Test script for the local vector store functionality.
Demonstrates basic operations and search capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vector_store.core import LocalVectorStore, create_daily_conversation_data
from src.features.embeddings import EmbeddingGenerator


def test_vector_store():
    """Test the vector store functionality."""
    print("🧪 Testing Local Vector Store")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = LocalVectorStore("test_vector_store")
    
    # Create sample conversations
    print("📝 Creating sample conversations...")
    conversations, embeddings = create_daily_conversation_data(20)
    
    # Add to vector store
    total = vector_store.add_conversations(conversations, embeddings)
    print(f"✅ Added {len(conversations)} conversations. Total: {total}")
    
    # Get store info
    info = vector_store.get_store_info()
    print(f"📊 Store info: {info['current_conversations']} conversations, {info['current_embeddings']} embeddings")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    embedding_generator = EmbeddingGenerator()
    
    test_queries = [
        "I need help with my account",
        "How do I delete my data?",
        "What's your privacy policy?",
        "I want to opt out of emails"
    ]
    
    for query in test_queries:
        print(f"\n--- Searching for: '{query}' ---")
        
        # Generate embedding for query
        query_embedding = embedding_generator.generate_sentence_embeddings([query])
        
        if query_embedding.size > 0:
            # Search for similar conversations
            results = vector_store.search_similar(query_embedding[0], top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity_score']:.3f}")
                print(f"   Text: {result['text']}")
                print(f"   User: {result['user_id']}")
        else:
            print("❌ Failed to generate embedding for query")
    
    # Test daily stats
    print("\n📈 Daily Statistics:")
    stats = vector_store.get_daily_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test recent conversations
    print("\n🕒 Recent Conversations:")
    recent = vector_store.get_recent_conversations(5)
    for i, conv in enumerate(recent, 1):
        print(f"{i}. {conv['text'][:50]}... (User: {conv['user_id']})")
    
    print("\n✅ Vector store test completed successfully!")


def test_daily_pipeline():
    """Test the daily conversations pipeline."""
    print("\n🚀 Testing Daily Conversations Pipeline")
    print("=" * 50)
    
    try:
        from src.vector_store.pipeline import DailyConversationsPipeline
        
        # Initialize pipeline
        pipeline = DailyConversationsPipeline("config_daily_conversations.yaml")
        
        # Generate daily conversations
        print("📝 Generating daily conversations...")
        conversations = pipeline.generate_daily_conversations(10)
        print(f"✅ Generated {len(conversations)} conversations")
        
        # Test search
        print("\n🔍 Testing search...")
        results = pipeline.search_similar_conversations("I need help with my account", top_k=3)
        print(f"✅ Found {len(results)} similar conversations")
        
        # Get stats
        print("\n📊 Getting statistics...")
        stats = pipeline.get_daily_stats()
        print(f"✅ Daily stats: {stats['total_conversations']} conversations")
        
        # Get store info
        print("\n📋 Getting store info...")
        info = pipeline.get_store_info()
        print(f"✅ Store info: {info['current_conversations']} conversations")
        
        print("\n✅ Daily pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")


def test_demo():
    """Test the demo functionality."""
    print("\n🎯 Testing Vector Store Demo")
    print("=" * 50)
    
    try:
        from src.vector_store.demo import demo_vector_store
        demo_vector_store()
        print("\n✅ Demo test completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")


if __name__ == "__main__":
    # Run all tests
    test_vector_store()
    test_daily_pipeline()
    test_demo()
    
    print("\n🎉 All vector store tests completed!") 