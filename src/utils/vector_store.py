"""
Simple local vector store for daily conversations.
Provides basic CRUD operations and similarity search without external dependencies.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path


class LocalVectorStore:
    """Simple local vector store for daily conversations."""
    
    def __init__(self, store_path: str = "vector_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        self.metadata_file = self.store_path / "metadata.json"
        self.embeddings_file = self.store_path / "embeddings.pkl"
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file or create new."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "total_conversations": 0,
            "last_updated": None,
            "daily_counts": {},
            "embedding_dimension": None
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """Load embeddings and conversation data."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    return data['embeddings'], data['conversations']
            except Exception:
                pass
        
        return np.array([]), []
    
    def _save_embeddings(self, embeddings: np.ndarray, conversations: List[Dict]):
        """Save embeddings and conversation data."""
        data = {
            'embeddings': embeddings,
            'conversations': conversations,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_conversations(self, conversations: List[Dict], embeddings: np.ndarray):
        """Add new conversations to the vector store."""
        if len(conversations) != len(embeddings):
            raise ValueError("Number of conversations must match number of embeddings")
        
        # Load existing data
        existing_embeddings, existing_conversations = self._load_embeddings()
        
        # Combine with new data
        if existing_embeddings.size > 0:
            all_embeddings = np.vstack([existing_embeddings, embeddings])
            all_conversations = existing_conversations + conversations
        else:
            all_embeddings = embeddings
            all_conversations = conversations
        
        # Update metadata
        self.metadata["total_conversations"] = len(all_conversations)
        self.metadata["embedding_dimension"] = all_embeddings.shape[1] if all_embeddings.size > 0 else None
        
        # Update daily counts
        today = date.today().isoformat()
        daily_count = len(conversations)
        self.metadata["daily_counts"][today] = self.metadata["daily_counts"].get(today, 0) + daily_count
        
        # Save data
        self._save_embeddings(all_embeddings, all_conversations)
        self._save_metadata()
        
        return len(all_conversations)
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar conversations using cosine similarity."""
        embeddings, conversations = self._load_embeddings()
        
        if embeddings.size == 0:
            return []
        
        # Calculate cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results with similarity scores
        results = []
        for idx in top_indices:
            results.append({
                **conversations[idx],
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def get_daily_stats(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific date or overall."""
        if target_date is None:
            target_date = date.today().isoformat()
        
        embeddings, conversations = self._load_embeddings()
        
        # Filter conversations by date
        daily_conversations = [
            conv for conv in conversations 
            if str(conv.get('date', '')).startswith(target_date)
        ]
        
        return {
            'date': target_date,
            'total_conversations': len(daily_conversations),
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else None
        }
    
    def get_recent_conversations(self, limit: int = 50) -> List[Dict]:
        """Get most recent conversations."""
        embeddings, conversations = self._load_embeddings()
        
        # Sort by timestamp (assuming conversations have 'timestamp' field)
        sorted_conversations = sorted(
            conversations, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        return sorted_conversations[:limit]
    
    def cleanup_old_conversations(self, days_to_keep: int = 30):
        """Remove conversations older than specified days."""
        embeddings, conversations = self._load_embeddings()
        
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).date()
        
        # Filter conversations
        recent_conversations = []
        recent_embeddings = []
        
        for i, conv in enumerate(conversations):
            try:
                conv_date = datetime.fromisoformat(conv.get('timestamp', '')).date()
                if conv_date >= cutoff_date:
                    recent_conversations.append(conv)
                    recent_embeddings.append(embeddings[i])
            except (ValueError, TypeError):
                # Skip conversations with invalid timestamps
                continue
        
        # Update data
        if recent_embeddings:
            all_embeddings = np.array(recent_embeddings)
        else:
            all_embeddings = np.array([])
        
        # Update metadata
        self.metadata["total_conversations"] = len(recent_conversations)
        
        # Save updated data
        self._save_embeddings(all_embeddings, recent_conversations)
        self._save_metadata()
        
        return len(recent_conversations)
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        embeddings, conversations = self._load_embeddings()
        
        return {
            **self.metadata,
            'store_path': str(self.store_path),
            'current_embeddings': len(embeddings),
            'current_conversations': len(conversations),
            'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else None
        }


def create_daily_conversation_data(n_conversations: int = 50) -> Tuple[List[Dict], np.ndarray]:
    """Create sample daily conversation data for testing."""
    from src.features.embeddings import EmbeddingGenerator
    
    # Sample conversation templates
    conversation_templates = [
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
    
    conversations = []
    texts = []
    
    for i in range(n_conversations):
        # Random conversation template
        template = np.random.choice(conversation_templates)
        
        # Add some variation
        variations = [
            f"{template} please",
            f"Hello, {template.lower()}",
            f"Good morning, {template.lower()}",
            f"Hi there, {template.lower()}",
            template
        ]
        
        text = np.random.choice(variations)
        
        conversation = {
            'id': f"conv_{i+1:04d}",
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'date': date.today().isoformat(),
            'user_id': f"user_{np.random.randint(1000, 9999)}",
            'session_id': f"session_{np.random.randint(10000, 99999)}"
        }
        
        conversations.append(conversation)
        texts.append(text)
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_sentence_embeddings(texts)
    
    return conversations, embeddings 