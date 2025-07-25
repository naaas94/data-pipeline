"""
Advanced text feature extraction for privacy intent classification.
Includes domain-specific privacy features and NLP-based feature engineering.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Any, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


class TextFeatureEngineer:
    """Advanced text feature extractor for privacy intent classification."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Privacy-specific keywords and patterns
        self.privacy_keywords = {
            'deletion': [
                'delete', 'remove', 'erase', 'clear', 'purge', 'eliminate',
                'destroy', 'wipe', 'cancel', 'terminate', 'discontinue'
            ],
            'request': [
                'request', 'ask', 'need', 'want', 'require', 'demand',
                'please', 'could', 'would', 'can', 'may'
            ],
            'opt_out': [
                'opt out', 'opt-out', 'unsubscribe', 'stop', 'cease',
                'withdraw', 'revoke', 'refuse', 'decline', 'reject'
            ],
            'data_terms': [
                'data', 'information', 'details', 'records', 'account',
                'profile', 'personal', 'private', 'confidential'
            ],
            'urgency': [
                'urgent', 'immediately', 'asap', 'urgent', 'quickly',
                'right away', 'now', 'soon', 'fast'
            ]
        }
        
        # Compile regex patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def extract_basic_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract basic text statistics."""
        features = {
            'text_length': [],
            'word_count': [],
            'sentence_count': [],
            'avg_word_length': [],
            'punctuation_count': [],
            'capital_letters_count': [],
            'question_marks': [],
            'exclamation_marks': []
        }
        
        for text in texts:
            if pd.isna(text) or text == '':
                # Handle missing text
                for key in features:
                    features[key].append(0.0)
                continue
            
            # Basic counts
            features['text_length'].append(len(text))
            
            words = word_tokenize(text.lower())
            features['word_count'].append(len(words))
            
            sentences = sent_tokenize(text)
            features['sentence_count'].append(len(sentences))
            
            # Average word length
            if words:
                avg_word_len = np.mean([len(word) for word in words if word.isalpha()])
                features['avg_word_length'].append(avg_word_len if not np.isnan(avg_word_len) else 0.0)
            else:
                features['avg_word_length'].append(0.0)
            
            # Punctuation and formatting
            features['punctuation_count'].append(sum(1 for char in text if char in string.punctuation))
            features['capital_letters_count'].append(sum(1 for char in text if char.isupper()))
            features['question_marks'].append(text.count('?'))
            features['exclamation_marks'].append(text.count('!'))
        
        return features
    
    def extract_privacy_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract privacy-specific features."""
        features = {
            'deletion_keywords': [],
            'request_keywords': [],
            'opt_out_keywords': [],
            'data_keywords': [],
            'urgency_keywords': [],
            'personal_info_detected': [],
            'privacy_keyword_density': [],
            'formal_language_score': []
        }
        
        for text in texts:
            if pd.isna(text) or text == '':
                for key in features:
                    features[key].append(0.0)
                continue
            
            text_lower = text.lower()
            words = word_tokenize(text_lower)
            
            # Count privacy-related keywords
            deletion_count = sum(1 for keyword in self.privacy_keywords['deletion'] 
                               if keyword in text_lower)
            features['deletion_keywords'].append(deletion_count)
            
            request_count = sum(1 for keyword in self.privacy_keywords['request'] 
                              if keyword in text_lower)
            features['request_keywords'].append(request_count)
            
            opt_out_count = sum(1 for keyword in self.privacy_keywords['opt_out'] 
                              if keyword in text_lower)
            features['opt_out_keywords'].append(opt_out_count)
            
            data_count = sum(1 for keyword in self.privacy_keywords['data_terms'] 
                           if keyword in text_lower)
            features['data_keywords'].append(data_count)
            
            urgency_count = sum(1 for keyword in self.privacy_keywords['urgency'] 
                              if keyword in text_lower)
            features['urgency_keywords'].append(urgency_count)
            
            # Personal information detection
            has_email = bool(self.email_pattern.search(text))
            has_phone = bool(self.phone_pattern.search(text))
            has_url = bool(self.url_pattern.search(text))
            features['personal_info_detected'].append(float(has_email or has_phone or has_url))
            
            # Privacy keyword density
            total_privacy_keywords = deletion_count + request_count + opt_out_count + data_count
            keyword_density = total_privacy_keywords / len(words) if words else 0.0
            features['privacy_keyword_density'].append(keyword_density)
            
            # Formal language indicators
            formal_indicators = ['please', 'thank you', 'sincerely', 'regards', 'kindly']
            formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
            features['formal_language_score'].append(formal_count)
        
        return features
    
    def extract_sentiment_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract sentiment-based features."""
        features = {
            'sentiment_positive': [],
            'sentiment_negative': [],
            'sentiment_neutral': [],
            'sentiment_compound': []
        }
        
        for text in texts:
            if pd.isna(text) or text == '':
                for key in features:
                    features[key].append(0.0)
                continue
            
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                features['sentiment_positive'].append(sentiment_scores['pos'])
                features['sentiment_negative'].append(sentiment_scores['neg'])
                features['sentiment_neutral'].append(sentiment_scores['neu'])
                features['sentiment_compound'].append(sentiment_scores['compound'])
            except Exception as e:
                warnings.warn(f"Sentiment analysis failed for text: {e}")
                for key in features:
                    features[key].append(0.0)
        
        return features
    
    def extract_linguistic_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract linguistic and structural features."""
        features = {
            'stopword_ratio': [],
            'unique_word_ratio': [],
            'readability_score': [],  # Simple readability approximation
            'verb_count': [],
            'noun_count': [],
            'adjective_count': []
        }
        
        for text in texts:
            if pd.isna(text) or text == '':
                for key in features:
                    features[key].append(0.0)
                continue
            
            words = word_tokenize(text.lower())
            
            # Stopword ratio
            stopword_count = sum(1 for word in words if word in self.stop_words)
            stopword_ratio = stopword_count / len(words) if words else 0.0
            features['stopword_ratio'].append(stopword_ratio)
            
            # Unique word ratio (vocabulary richness)
            unique_ratio = len(set(words)) / len(words) if words else 0.0
            features['unique_word_ratio'].append(unique_ratio)
            
            # Simple readability score (average sentence length)
            sentences = sent_tokenize(text)
            avg_sentence_length = len(words) / len(sentences) if sentences else 0.0
            features['readability_score'].append(avg_sentence_length)
            
            # Simple POS approximation (without full POS tagging)
            # These are rough approximations for performance
            verb_indicators = ['ing', 'ed', 'er', 'es']
            verb_count = sum(1 for word in words 
                           if any(word.endswith(suffix) for suffix in verb_indicators))
            features['verb_count'].append(verb_count)
            
            noun_indicators = ['ion', 'tion', 'ness', 'ment']
            noun_count = sum(1 for word in words 
                           if any(word.endswith(suffix) for suffix in noun_indicators))
            features['noun_count'].append(noun_count)
            
            adj_indicators = ['ly', 'ful', 'less', 'ous']
            adj_count = sum(1 for word in words 
                          if any(word.endswith(suffix) for suffix in adj_indicators))
            features['adjective_count'].append(adj_count)
        
        return features
    
    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract all features and return as DataFrame."""
        all_features = {}
        
        # Extract different feature groups
        basic_features = self.extract_basic_features(texts)
        privacy_features = self.extract_privacy_features(texts)
        sentiment_features = self.extract_sentiment_features(texts)
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Combine all features
        all_features.update(basic_features)
        all_features.update(privacy_features)
        all_features.update(sentiment_features)
        all_features.update(linguistic_features)
        
        return pd.DataFrame(all_features)
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Analyze feature importance for privacy intent classification."""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].fillna(0)
        
        # Encode target if needed
        le = LabelEncoder()
        y = le.fit_transform(df[target_column])
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_columns, mi_scores))
        
        # Sort by importance
        sorted_features = dict(sorted(feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return sorted_features


def engineer_text_features(texts: List[str], config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convenience function for text feature engineering."""
    extractor = TextFeatureEngineer(config)
    return extractor.extract_all_features(texts) 