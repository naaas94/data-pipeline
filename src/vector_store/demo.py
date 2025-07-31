"""
Vector Store Demo Functionality
Simple demo of the local vector store functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .core import LocalVectorStore, create_daily_conversation_data
from src.features.embeddings import EmbeddingGenerator


def demo_vector_store():
    """Demonstrate vector store functionality."""
    print("🎯 Local Vector Store Demo")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = LocalVectorStore("demo_vector_store")
    
    # Create sample conversations
    print("📝 Creating sample conversations...")
    conversations, embeddings = create_daily_conversation_data(30)
    
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
        "I want to opt out of emails",
        "Can you help me with billing?"
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
    
    print("\n✅ Vector store demo completed successfully!")
    print("\n💡 You can now say you have a local vector store!")
    print("   - No external dependencies")
    print("   - No cloud service fees")
    print("   - All data stored locally")
    print("   - Similarity search working")
    print("   - Daily conversation tracking")


if __name__ == "__main__":
    demo_vector_store() 