"""
Core Vector Store Implementation
Local vector store for storing and searching daily conversations.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import uuid
import random

from src.features.embeddings import EmbeddingGenerator


class LocalVectorStore:
    """Local vector store for daily conversations with similarity search."""
    
    def __init__(self, store_path: str = "vector_store"):
        """Initialize vector store with local storage path."""
        self.store_path = store_path
        self.metadata_file = os.path.join(store_path, "metadata.json")
        self.embeddings_file = os.path.join(store_path, "embeddings.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Load or initialize store
        self._load_or_initialize_store()
    
    def _load_or_initialize_store(self):
        """Load existing store or initialize new one."""
        if os.path.exists(self.metadata_file) and os.path.exists(self.embeddings_file):
            self._load_store()
        else:
            self._initialize_store()
    
    def _initialize_store(self):
        """Initialize new vector store."""
        self.conversations = []
        self.embeddings = np.array([])
        self.metadata = {
            "total_conversations": 0,
            "last_updated": datetime.now().isoformat(),
            "daily_counts": {},
            "embedding_dimension": 384  # Default for all-MiniLM-L6-v2
        }
        self._save_store()
    
    def _load_store(self):
        """Load existing vector store from disk."""
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.conversations = data.get('conversations', [])
                self.embeddings = data.get('embeddings', np.array([]))
            
            print(f"Loaded vector store with {len(self.conversations)} conversations")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self._initialize_store()
    
    def _save_store(self):
        """Save vector store to disk."""
        try:
            # Update metadata
            self.metadata["total_conversations"] = len(self.conversations)
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save conversations and embeddings
            data = {
                'conversations': self.conversations,
                'embeddings': self.embeddings
            }
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def add_conversations(self, conversations: List[Dict], embeddings: np.ndarray) -> int:
        """Add conversations and their embeddings to the store."""
        if len(conversations) != len(embeddings):
            raise ValueError("Number of conversations must match number of embeddings")
        
        # Add conversations
        for conv in conversations:
            if 'id' not in conv:
                conv['id'] = f"conv_{uuid.uuid4().hex[:8]}"
            if 'timestamp' not in conv:
                conv['timestamp'] = datetime.now().isoformat()
            if 'date' not in conv:
                conv['date'] = datetime.now().date().isoformat()
            
            self.conversations.append(conv)
        
        # Add embeddings
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Update daily counts
        for conv in conversations:
            conv_date = conv.get('date', datetime.now().date().isoformat())
            if conv_date not in self.metadata['daily_counts']:
                self.metadata['daily_counts'][conv_date] = 0
            self.metadata['daily_counts'][conv_date] += 1
        
        # Save to disk
        self._save_store()
        
        return len(self.conversations)
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar conversations using cosine similarity."""
        if len(self.conversations) == 0:
            return []
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results with similarity scores
        results = []
        for idx in top_indices:
            result = self.conversations[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        return results
    
    def get_daily_stats(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily statistics."""
        if target_date is None:
            target_date = datetime.now().date().isoformat()
        
        daily_count = self.metadata['daily_counts'].get(target_date, 0)
        
        return {
            "date": target_date,
            "total_conversations": daily_count,
            "total_conversations_all_time": len(self.conversations),
            "embedding_dimension": self.metadata['embedding_dimension']
        }
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get vector store information."""
        return {
            "current_conversations": len(self.conversations),
            "current_embeddings": self.embeddings.shape[0] if self.embeddings.size > 0 else 0,
            "embedding_dimension": self.metadata['embedding_dimension'],
            "last_updated": self.metadata['last_updated'],
            "store_path": self.store_path
        }
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations."""
        return self.conversations[-limit:] if self.conversations else []
    
    def cleanup_old_conversations(self, days_to_keep: int = 30):
        """Remove conversations older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
        
        # Filter conversations
        new_conversations = []
        new_embeddings = []
        
        for i, conv in enumerate(self.conversations):
            conv_date = datetime.fromisoformat(conv['timestamp']).date()
            if conv_date >= cutoff_date:
                new_conversations.append(conv)
                new_embeddings.append(self.embeddings[i])
        
        # Update store
        self.conversations = new_conversations
        self.embeddings = np.array(new_embeddings) if new_embeddings else np.array([])
        
        # Update daily counts
        self.metadata['daily_counts'] = {}
        for conv in self.conversations:
            conv_date = conv.get('date', datetime.fromisoformat(conv['timestamp']).date().isoformat())
            if conv_date not in self.metadata['daily_counts']:
                self.metadata['daily_counts'][conv_date] = 0
            self.metadata['daily_counts'][conv_date] += 1
        
        self._save_store()
        print(f"Cleaned up old conversations. Remaining: {len(self.conversations)}")


def create_daily_conversation_data(n_conversations: int = 50) -> Tuple[List[Dict], np.ndarray]:
    """Create sample daily conversation data with embeddings."""
    
    # Conversation templates
    templates = [
        "Hi, I need help with my account",
        "Can you help me understand your privacy policy?",
        "I want to delete my data",
        "How do I opt out of marketing emails?",
        "I have a question about data collection",
        "Can you explain your terms of service?",
        "I need to update my personal information",
        "How do I access my data?",
        "I want to know what information you have about me",
        "Can you help me with a billing issue?"
    ]
    
    # User IDs for variety
    user_ids = [f"user_{i:04d}" for i in range(1, 101)]
    
    # Generate conversations
    conversations = []
    texts = []
    
    for i in range(n_conversations):
        # Select random template and user
        template = random.choice(templates)
        user_id = random.choice(user_ids)
        
        # Add some variation to the template
        variations = [
            f"{template}",
            f"{template} please",
            f"Hello, {template.lower()}",
            f"Hi there, {template.lower()}",
            f"Good morning, {template.lower()}",
            f"Excuse me, {template.lower()}"
        ]
        
        text = random.choice(variations)
        
        conversation = {
            'id': f"conv_{uuid.uuid4().hex[:8]}",
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().date().isoformat(),
            'user_id': user_id,
            'session_id': f"session_{uuid.uuid4().hex[:8]}"
        }
        
        conversations.append(conversation)
        texts.append(text)
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_sentence_embeddings(texts)
    
    return conversations, embeddings 