"""
Embedding generation for privacy intent classification.
Supports multiple embedding models including sentence transformers, word2vec, and TF-IDF.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

# Optional imports for advanced embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None  # Define placeholder to avoid NameError
    print("gensim not available. Install with: pip install gensim")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingGenerator:
    """Advanced embedding generator for privacy intent classification."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embedding_config = self.config.get('embeddings', {})
        self.cache_dir = self.embedding_config.get('cache_dir', 'cache/embeddings')
        self.models = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default embedding configurations
        self.default_configs = {
            'sentence_transformer': {
                'model_name': 'all-MiniLM-L6-v2',  # Fast and good quality
                'max_seq_length': 512,
                'batch_size': 32
            },
            'tfidf': {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95,
                'svd_components': 200
            },
            'word2vec': {
                'vector_size': 100,
                'window': 5,
                'min_count': 1,
                'workers': 4,
                'epochs': 10
            }
        }
    
    def _get_cache_path(self, model_type: str, model_name: str) -> str:
        """Get cache path for a specific model."""
        safe_name = model_name.replace('/', '_').replace('-', '_')
        return os.path.join(self.cache_dir, f"{model_type}_{safe_name}.pkl")
    
    def load_sentence_transformer(self, model_name: str = None) -> Optional[SentenceTransformer]:
        """Load sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn("sentence-transformers not available")
            return None
        
        model_name = model_name or self.default_configs['sentence_transformer']['model_name']
        
        try:
            if model_name not in self.models:
                print(f"Loading sentence transformer model: {model_name}")
                model = SentenceTransformer(model_name)
                self.models[model_name] = model
            return self.models[model_name]
        except Exception as e:
            warnings.warn(f"Failed to load sentence transformer {model_name}: {e}")
            return None
    
    def generate_sentence_embeddings(self, texts: List[str], 
                                   model_name: str = None,
                                   batch_size: int = None) -> np.ndarray:
        """Generate embeddings using sentence transformers."""
        model = self.load_sentence_transformer(model_name)
        if model is None:
            return np.array([])
        
        batch_size = batch_size or self.default_configs['sentence_transformer']['batch_size']
        
        try:
            # Clean texts
            cleaned_texts = [text if text and not pd.isna(text) else "" for text in texts]
            
            # Generate embeddings
            embeddings = model.encode(
                cleaned_texts, 
                batch_size=batch_size,
                show_progress_bar=True if len(cleaned_texts) > 100 else False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            warnings.warn(f"Failed to generate sentence embeddings: {e}")
            return np.array([])
    
    def train_tfidf_model(self, texts: List[str], config: Dict[str, Any] = None) -> TfidfVectorizer:
        """Train TF-IDF model on texts."""
        config = config or self.default_configs['tfidf']
        
        # Clean texts
        cleaned_texts = [text if text and not pd.isna(text) else "" for text in texts]
        
        # Create and train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        vectorizer.fit(cleaned_texts)
        
        # Cache the model
        cache_path = self._get_cache_path('tfidf', 'vectorizer')
        joblib.dump(vectorizer, cache_path)
        
        return vectorizer
    
    def generate_tfidf_embeddings(self, texts: List[str], 
                                vectorizer: TfidfVectorizer = None,
                                use_svd: bool = True,
                                n_components: int = None) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        if vectorizer is None:
            # Try to load cached vectorizer
            cache_path = self._get_cache_path('tfidf', 'vectorizer')
            if os.path.exists(cache_path):
                vectorizer = joblib.load(cache_path)
            else:
                # Train new vectorizer
                vectorizer = self.train_tfidf_model(texts)
        
        # Clean texts
        cleaned_texts = [text if text and not pd.isna(text) else "" for text in texts]
        
        # Generate TF-IDF features
        tfidf_matrix = vectorizer.transform(cleaned_texts)
        
        if use_svd:
            # Apply dimensionality reduction
            n_components = n_components or self.default_configs['tfidf']['svd_components']
            n_components = min(n_components, tfidf_matrix.shape[1])
            
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)
            
            # Cache SVD model
            cache_path = self._get_cache_path('tfidf', 'svd')
            joblib.dump(svd, cache_path)
        else:
            embeddings = tfidf_matrix.toarray()
        
        return embeddings
    
    def train_word2vec_model(self, texts: List[str], config: Dict[str, Any] = None) -> Optional[Word2Vec]:
        """Train Word2Vec model on texts."""
        if not GENSIM_AVAILABLE:
            warnings.warn("gensim not available for Word2Vec")
            return None
        
        config = config or self.default_configs['word2vec']
        
        # Tokenize texts
        from nltk.tokenize import word_tokenize
        tokenized_texts = []
        for text in texts:
            if text and not pd.isna(text):
                tokens = word_tokenize(text.lower())
                tokenized_texts.append(tokens)
        
        if not tokenized_texts:
            warnings.warn("No valid texts for Word2Vec training")
            return None
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            workers=config['workers'],
            epochs=config['epochs']
        )
        
        # Cache the model
        cache_path = self._get_cache_path('word2vec', 'model')
        model.save(cache_path)
        
        return model
    
    def generate_word2vec_embeddings(self, texts: List[str], 
                                   model: Word2Vec = None,
                                   aggregation: str = 'mean') -> np.ndarray:
        """Generate Word2Vec embeddings by aggregating word vectors."""
        if not GENSIM_AVAILABLE:
            warnings.warn("gensim not available for Word2Vec")
            return np.array([])
        
        if model is None:
            # Try to load cached model
            cache_path = self._get_cache_path('word2vec', 'model')
            if os.path.exists(cache_path):
                model = Word2Vec.load(cache_path)
            else:
                # Train new model
                model = self.train_word2vec_model(texts)
                if model is None:
                    return np.array([])
        
        from nltk.tokenize import word_tokenize
        embeddings = []
        
        for text in texts:
            if not text or pd.isna(text):
                # Handle empty text
                embeddings.append(np.zeros(model.vector_size))
                continue
            
            tokens = word_tokenize(text.lower())
            word_vectors = []
            
            for token in tokens:
                if token in model.wv:
                    word_vectors.append(model.wv[token])
            
            if word_vectors:
                if aggregation == 'mean':
                    text_embedding = np.mean(word_vectors, axis=0)
                elif aggregation == 'sum':
                    text_embedding = np.sum(word_vectors, axis=0)
                elif aggregation == 'max':
                    text_embedding = np.max(word_vectors, axis=0)
                else:
                    text_embedding = np.mean(word_vectors, axis=0)
            else:
                text_embedding = np.zeros(model.vector_size)
            
            embeddings.append(text_embedding)
        
        return np.array(embeddings)
    
    def generate_privacy_domain_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate privacy domain-specific embeddings using weighted combinations."""
        embeddings_list = []
        
        # Generate multiple types of embeddings
        sentence_emb = self.generate_sentence_embeddings(texts)
        tfidf_emb = self.generate_tfidf_embeddings(texts)
        
        # Combine embeddings if both are available
        if sentence_emb.size > 0 and tfidf_emb.size > 0:
            # Normalize embeddings
            from sklearn.preprocessing import StandardScaler
            
            scaler_sent = StandardScaler()
            sentence_emb_norm = scaler_sent.fit_transform(sentence_emb)
            
            scaler_tfidf = StandardScaler()
            tfidf_emb_norm = scaler_tfidf.fit_transform(tfidf_emb)
            
            # Weighted combination (favor sentence transformers for semantic understanding)
            combined_emb = np.concatenate([
                sentence_emb_norm * 0.7,  # Higher weight for semantic embeddings
                tfidf_emb_norm * 0.3      # Lower weight for statistical embeddings
            ], axis=1)
            
            return combined_emb
        
        elif sentence_emb.size > 0:
            return sentence_emb
        elif tfidf_emb.size > 0:
            return tfidf_emb
        else:
            warnings.warn("No embeddings could be generated")
            return np.array([])
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       texts: List[str],
                       embedding_type: str,
                       metadata: Dict[str, Any] = None) -> str:
        """Save embeddings with metadata."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{embedding_type}_embeddings_{timestamp}.pkl"
        filepath = os.path.join(self.cache_dir, filename)
        
        data = {
            'embeddings': embeddings,
            'texts': texts,
            'embedding_type': embedding_type,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'shape': embeddings.shape
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Embeddings saved to: {filepath}")
        return filepath
    
    def load_embeddings(self, filepath: str) -> Dict[str, Any]:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        if embeddings.size == 0:
            return {'error': 'No embeddings provided'}
        
        return {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'sparsity': float(np.mean(embeddings == 0))
        }


def generate_embeddings(texts: List[str], 
                       embedding_type: str = 'sentence_transformer',
                       config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Convenience function for embedding generation."""
    generator = EmbeddingGenerator(config)
    
    if embedding_type == 'sentence_transformer':
        return generator.generate_sentence_embeddings(texts)
    elif embedding_type == 'tfidf':
        return generator.generate_tfidf_embeddings(texts)
    elif embedding_type == 'word2vec':
        return generator.generate_word2vec_embeddings(texts)
    elif embedding_type == 'privacy_domain':
        return generator.generate_privacy_domain_embeddings(texts)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def save_embeddings_for_training(df: pd.DataFrame, 
                                text_column: str = 'text',
                                output_dir: str = 'output/embeddings',
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Generate and save multiple types of embeddings for training pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    generator = EmbeddingGenerator(config)
    
    texts = df[text_column].tolist()
    saved_files = {}
    
    # Generate different types of embeddings
    embedding_types = ['sentence_transformer', 'tfidf', 'privacy_domain']
    
    for emb_type in embedding_types:
        try:
            embeddings = generate_embeddings(texts, emb_type, config)
            if embeddings.size > 0:
                # Create metadata
                metadata = {
                    'text_column': text_column,
                    'num_samples': len(texts),
                    'embedding_type': emb_type,
                    'stats': generator.get_embedding_stats(embeddings)
                }
                
                # Save embeddings
                filepath = generator.save_embeddings(embeddings, texts, emb_type, metadata)
                saved_files[emb_type] = filepath
                
                print(f"Generated {emb_type} embeddings: {embeddings.shape}")
            else:
                print(f"Failed to generate {emb_type} embeddings")
                
        except Exception as e:
            warnings.warn(f"Error generating {emb_type} embeddings: {e}")
    
    return saved_files 