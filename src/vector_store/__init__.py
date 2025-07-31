"""
Vector Store Module for Enterprise Data Pipeline

This module provides local vector store functionality for storing and searching
daily conversations with similarity search capabilities.
"""

from .core import LocalVectorStore, create_daily_conversation_data
from .pipeline import DailyConversationsPipeline
from .demo import demo_vector_store

__all__ = [
    'LocalVectorStore',
    'create_daily_conversation_data', 
    'DailyConversationsPipeline',
    'demo_vector_store'
]

__version__ = "1.0.0" 