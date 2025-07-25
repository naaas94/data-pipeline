"""
Enhanced synthetic data generation for privacy intent classification.
Includes sophisticated text templates, variations, and realistic scenarios.
"""

import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import defaultdict


class PrivacyTextGenerator:
    """Advanced privacy intent text generator with templates and variations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Intent-specific templates and variations
        self.intent_templates = {
            'privacy_request': {
                'formal': [
                    "I would like to request access to my personal data",
                    "Please provide me with a copy of all personal information you have about me",
                    "I am requesting to see what personal data you have collected about me",
                    "Could you please send me all the information you have stored about me",
                    "I would like to exercise my right to access my personal data",
                    "Please provide me with a data subject access request",
                    "I need to see what personal information you have on file for me",
                    "I am entitled to know what data you have about me under GDPR"
                ],
                'informal': [
                    "What info do you have about me?",
                    "Can I see my data please?",
                    "I want to know what you know about me",
                    "Show me my personal info",
                    "What data do you have on me?",
                    "I need to see my information",
                    "Can you tell me what you have about me?",
                    "I want access to my data"
                ],
                'urgent': [
                    "I need to urgently access my personal data",
                    "URGENT: Please send me all my personal information immediately",
                    "I need my data ASAP for legal purposes",
                    "This is urgent - I need all my personal data right away",
                    "Please prioritize my data access request - it's urgent"
                ]
            },
            
            'data_deletion': {
                'formal': [
                    "I would like to request the deletion of my personal data",
                    "Please delete all personal information you have about me",
                    "I am requesting the complete removal of my personal data from your systems",
                    "I would like to exercise my right to erasure under GDPR",
                    "Please permanently delete my account and all associated data",
                    "I request that you erase all my personal information",
                    "I want to invoke my right to be forgotten",
                    "Please remove all traces of my personal data from your database"
                ],
                'informal': [
                    "Delete my data please",
                    "I want my info removed",
                    "Remove all my personal stuff",
                    "Delete everything about me",
                    "I want my data gone",
                    "Please remove my information",
                    "Get rid of my data",
                    "I don't want you to have my info anymore"
                ],
                'angry': [
                    "DELETE MY DATA NOW!",
                    "I DEMAND you delete all my information immediately",
                    "Remove my data RIGHT NOW or I'll take legal action",
                    "I'm sick of this - delete everything about me!",
                    "DELETE EVERYTHING - I don't want you to have ANY of my data"
                ]
            },
            
            'opt_out': {
                'formal': [
                    "I would like to opt out of all data collection",
                    "Please remove me from all data processing activities",
                    "I withdraw my consent for data processing",
                    "I no longer consent to the processing of my personal data",
                    "Please stop collecting and processing my personal information",
                    "I want to opt out of all marketing and data collection",
                    "I revoke my consent for data usage",
                    "Please cease all data collection related to my account"
                ],
                'informal': [
                    "Stop collecting my data",
                    "I want to opt out",
                    "Don't use my info anymore",
                    "Stop tracking me",
                    "I don't want you to collect my data",
                    "Please stop using my information",
                    "I want out of data collection",
                    "Stop processing my data"
                ],
                'direct': [
                    "Unsubscribe me from everything",
                    "Stop all data collection",
                    "I opt out",
                    "Remove me from all lists",
                    "Stop tracking",
                    "No more data collection"
                ]
            },
            
            'other': {
                'customer_service': [
                    "I have a question about my account",
                    "Can you help me with my order?",
                    "I need assistance with your service",
                    "I'm having trouble with my account",
                    "Can someone help me please?",
                    "I need customer support",
                    "Could you help me understand your terms?",
                    "I have a billing question"
                ],
                'general': [
                    "Thank you for your excellent service",
                    "I love using your platform",
                    "Your product is amazing",
                    "When are you open?",
                    "What are your business hours?",
                    "How can I contact you?",
                    "I'd like to learn more about your services",
                    "Do you offer discounts?"
                ],
                'technical': [
                    "I'm experiencing technical difficulties",
                    "The website isn't working properly",
                    "I can't log into my account",
                    "There seems to be a bug",
                    "The app keeps crashing",
                    "I'm getting an error message",
                    "Something is wrong with the system",
                    "The page won't load"
                ]
            }
        }
        
        # Modifiers to add variation
        self.modifiers = {
            'politeness': ["please", "kindly", "if possible", "thank you"],
            'urgency': ["urgently", "immediately", "as soon as possible", "ASAP", "right away"],
            'legal': ["under GDPR", "according to data protection law", "as per my rights", "legally"],
            'emotion': ["I'm concerned about", "I'm worried about", "I'm frustrated with", "I'm confused about"],
            'time': ["today", "this week", "within 30 days", "by tomorrow", "soon"]
        }
        
        # Personal information patterns
        self.personal_info_patterns = {
            'email': ['user@example.com', 'john.doe@email.com', 'privacy@user.org'],
            'phone': ['555-123-4567', '(555) 987-6543', '555.111.2222'],
            'names': ['John Smith', 'Jane Doe', 'Alex Johnson', 'Maria Garcia'],
            'account_ids': ['#12345', 'Account: 98765', 'ID: ABC123']
        }
    
    def _add_variations(self, text: str, intent: str, variation_level: float = 0.3) -> str:
        """Add variations to make text more realistic."""
        if random.random() > variation_level:
            return text
        
        variations = []
        
        # Add politeness modifiers
        if random.random() < 0.4:
            politeness = random.choice(self.modifiers['politeness'])
            if text.endswith('.'):
                text = text[:-1] + f", {politeness}."
            else:
                text = f"{text}, {politeness}"
        
        # Add urgency for certain intents
        if intent in ['data_deletion', 'privacy_request'] and random.random() < 0.3:
            urgency = random.choice(self.modifiers['urgency'])
            text = f"I need to {urgency} " + text.lower()
        
        # Add legal references
        if intent in ['privacy_request', 'data_deletion'] and random.random() < 0.2:
            legal = random.choice(self.modifiers['legal'])
            text = f"{text} {legal}"
        
        # Add emotional context
        if random.random() < 0.2:
            emotion = random.choice(self.modifiers['emotion'])
            text = f"{emotion} my privacy. {text}"
        
        # Add typos occasionally
        if random.random() < 0.1:
            text = self._add_typos(text)
        
        # Add personal information sometimes
        if random.random() < 0.15:
            text = self._add_personal_info(text)
        
        return text
    
    def _add_typos(self, text: str) -> str:
        """Add realistic typos to text."""
        typo_patterns = [
            ('the', 'teh'),
            ('you', 'u'),
            ('please', 'plz'),
            ('information', 'info'),
            ('and', '&'),
            ('to', '2'),
            ('for', '4')
        ]
        
        for correct, typo in typo_patterns:
            if random.random() < 0.3 and correct in text.lower():
                text = re.sub(r'\b' + correct + r'\b', typo, text, flags=re.IGNORECASE)
                break  # Only one typo per text
        
        return text
    
    def _add_personal_info(self, text: str) -> str:
        """Add personal information to make requests more realistic."""
        info_type = random.choice(list(self.personal_info_patterns.keys()))
        info_value = random.choice(self.personal_info_patterns[info_type])
        
        if info_type == 'email':
            text = f"{text} My email is {info_value}."
        elif info_type == 'phone':
            text = f"{text} You can reach me at {info_value}."
        elif info_type == 'names':
            text = f"Hi, this is {info_value}. {text}"
        elif info_type == 'account_ids':
            text = f"{text} My account number is {info_value}."
        
        return text
    
    def generate_text(self, intent: str, style: str = None, variation_level: float = 0.3) -> str:
        """Generate a single text for the given intent and style."""
        if intent not in self.intent_templates:
            raise ValueError(f"Unknown intent: {intent}")
        
        # Choose style or random
        available_styles = list(self.intent_templates[intent].keys())
        if style is None or style not in available_styles:
            style = random.choice(available_styles)
        
        # Choose base template
        templates = self.intent_templates[intent][style]
        base_text = random.choice(templates)
        
        # Add variations
        varied_text = self._add_variations(base_text, intent, variation_level)
        
        return varied_text
    
    def generate_texts(self, intent: str, n: int, 
                      style_distribution: Optional[Dict[str, float]] = None,
                      variation_level: float = 0.3) -> List[str]:
        """Generate multiple texts for an intent."""
        texts = []
        
        # Default style distribution
        if style_distribution is None:
            if intent == 'other':
                style_distribution = {'customer_service': 0.4, 'general': 0.4, 'technical': 0.2}
            else:
                available_styles = list(self.intent_templates[intent].keys())
                equal_prob = 1.0 / len(available_styles)
                style_distribution = {style: equal_prob for style in available_styles}
        
        for _ in range(n):
            # Choose style based on distribution
            style = np.random.choice(
                list(style_distribution.keys()),
                p=list(style_distribution.values())
            )
            
            text = self.generate_text(intent, style, variation_level)
            texts.append(text)
        
        return texts


class EnhancedSyntheticDataGenerator:
    """Enhanced synthetic data generator with realistic scenarios and features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.text_generator = PrivacyTextGenerator(config)
        self.random_state = self.config.get('random_state', 42)
        np.random.seed(self.random_state)
        
        # Realistic intent distribution
        self.intent_distribution = self.config.get('intent_distribution', {
            'privacy_request': 0.25,
            'data_deletion': 0.20,
            'opt_out': 0.25,
            'other': 0.30
        })
        
        # Time patterns for realistic timestamps
        self.time_patterns = {
            'business_hours': (9, 17),  # 9 AM to 5 PM
            'weekdays_only': True,
            'peak_hours': [10, 11, 14, 15],  # Peak request hours
            'seasonal_variation': True
        }
        
        # Confidence patterns based on intent and text characteristics
        self.confidence_patterns = {
            'privacy_request': {'mean': 0.8, 'std': 0.15, 'min': 0.4},
            'data_deletion': {'mean': 0.85, 'std': 0.12, 'min': 0.5},
            'opt_out': {'mean': 0.75, 'std': 0.18, 'min': 0.3},
            'other': {'mean': 0.4, 'std': 0.25, 'min': 0.1}
        }
    
    def _generate_realistic_timestamp(self) -> datetime:
        """Generate realistic timestamp based on business patterns."""
        # Random date within last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Adjust for business hours if configured
        if self.time_patterns['business_hours']:
            start_hour, end_hour = self.time_patterns['business_hours']
            
            # Bias towards business hours
            if random.random() < 0.7:  # 70% during business hours
                hour = random.randint(start_hour, end_hour - 1)
            else:
                # Off hours
                off_hours = list(range(0, start_hour)) + list(range(end_hour, 24))
                hour = random.choice(off_hours)
            
            # Bias towards peak hours
            if hour in self.time_patterns['peak_hours'] and random.random() < 0.8:
                pass  # Keep the peak hour
            
            random_date = random_date.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
        
        # Adjust for weekdays if configured
        if self.time_patterns['weekdays_only'] and random.random() < 0.8:
            # Move to weekday if weekend
            while random_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                random_date += timedelta(days=1)
        
        return random_date
    
    def _generate_confidence_score(self, intent: str, text: str) -> float:
        """Generate realistic confidence score based on intent and text characteristics."""
        patterns = self.confidence_patterns[intent]
        
        # Base confidence from patterns
        confidence = np.random.normal(patterns['mean'], patterns['std'])
        
        # Adjust based on text characteristics
        text_lower = text.lower()
        
        # Higher confidence for formal language
        formal_indicators = ['please', 'would like', 'request', 'gdpr', 'data protection']
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        confidence += formal_count * 0.05
        
        # Higher confidence for specific privacy keywords
        privacy_keywords = ['delete', 'remove', 'access', 'data', 'information', 'personal']
        privacy_count = sum(1 for keyword in privacy_keywords if keyword in text_lower)
        confidence += privacy_count * 0.02
        
        # Lower confidence for unclear or typo-filled text
        if len(text.split()) < 5:  # Very short text
            confidence -= 0.1
        
        typo_indicators = ['u', '2', '4', 'plz', 'teh']
        typo_count = sum(1 for typo in typo_indicators if typo in text_lower)
        confidence -= typo_count * 0.05
        
        # Ensure confidence is within bounds
        confidence = max(patterns['min'], min(1.0, confidence))
        
        return round(confidence, 3)
    
    def generate_dataset(self, n_samples: int = 10000,
                        intent_distribution: Optional[Dict[str, float]] = None,
                        variation_level: float = 0.3,
                        include_metadata: bool = True) -> pd.DataFrame:
        """Generate enhanced synthetic dataset."""
        if intent_distribution is None:
            intent_distribution = self.intent_distribution
        
        # Validate distribution
        if abs(sum(intent_distribution.values()) - 1.0) > 0.01:
            raise ValueError("Intent distribution must sum to 1.0")
        
        data = []
        
        # Generate samples for each intent
        for intent, proportion in intent_distribution.items():
            n_intent_samples = int(n_samples * proportion)
            
            # Generate texts for this intent
            texts = self.text_generator.generate_texts(
                intent, n_intent_samples, variation_level=variation_level
            )
            
            # Generate corresponding data
            for text in texts:
                timestamp = self._generate_realistic_timestamp()
                confidence = self._generate_confidence_score(intent, text)
                
                sample = {
                    'text': text,
                    'intent': intent,
                    'confidence': confidence,
                    'timestamp': timestamp
                }
                
                # Add metadata if requested
                if include_metadata:
                    sample.update({
                        'text_length': len(text),
                        'word_count': len(text.split()),
                        'has_personal_info': any(pattern in text.lower() 
                                               for patterns in self.text_generator.personal_info_patterns.values()
                                               for pattern in patterns),
                        'formality_score': self._calculate_formality_score(text),
                        'urgency_score': self._calculate_urgency_score(text)
                    })
                
                data.append(sample)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return df
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score for text."""
        formal_indicators = [
            'please', 'would like', 'request', 'kindly', 'thank you',
            'sincerely', 'regards', 'could you', 'i am writing'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        
        # Normalize by text length
        formality_score = formal_count / len(text.split()) if text.split() else 0
        return min(1.0, formality_score * 5)  # Scale up and cap at 1.0
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score for text."""
        urgency_indicators = [
            'urgent', 'immediately', 'asap', 'right away', 'now',
            'emergency', 'quickly', 'soon', '!', 'urgent:'
        ]
        
        text_lower = text.lower()
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in text_lower)
        
        # Check for ALL CAPS (indicates urgency)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        urgency_score = (urgency_count * 0.3) + (caps_ratio * 0.7)
        return min(1.0, urgency_score)
    
    def generate_balanced_dataset(self, n_samples: int = 10000,
                                 min_samples_per_class: int = None) -> pd.DataFrame:
        """Generate a balanced dataset with equal representation of all intents."""
        intents = list(self.intent_distribution.keys())
        n_per_intent = n_samples // len(intents)
        
        if min_samples_per_class and n_per_intent < min_samples_per_class:
            n_per_intent = min_samples_per_class
            n_samples = n_per_intent * len(intents)
            print(f"Adjusted total samples to {n_samples} to meet minimum per class requirement")
        
        balanced_distribution = {intent: 1.0 / len(intents) for intent in intents}
        
        return self.generate_dataset(
            n_samples=n_samples,
            intent_distribution=balanced_distribution
        )
    
    def generate_dataset_splits(self, n_samples: int = 10000,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate dataset and split into train/val/test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Generate full dataset
        df = self.generate_dataset(n_samples)
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['intent'],
            random_state=self.random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['intent'],
            random_state=self.random_state
        )
        
        return train_df, val_df, test_df
    
    def get_generation_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the generated dataset."""
        stats = {
            'total_samples': len(df),
            'intent_distribution': df['intent'].value_counts().to_dict(),
            'intent_proportions': df['intent'].value_counts(normalize=True).to_dict(),
            'confidence_stats': {
                'mean': df['confidence'].mean(),
                'std': df['confidence'].std(),
                'min': df['confidence'].min(),
                'max': df['confidence'].max()
            },
            'text_length_stats': {
                'mean': df['text'].str.len().mean(),
                'std': df['text'].str.len().std(),
                'min': df['text'].str.len().min(),
                'max': df['text'].str.len().max()
            },
            'temporal_stats': {
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'hour_distribution': df['timestamp'].dt.hour.value_counts().to_dict(),
                'weekday_distribution': df['timestamp'].dt.dayofweek.value_counts().to_dict()
            }
        }
        
        # Add metadata stats if available
        if 'formality_score' in df.columns:
            stats['formality_stats'] = {
                'mean': df['formality_score'].mean(),
                'distribution': df['formality_score'].describe().to_dict()
            }
        
        if 'urgency_score' in df.columns:
            stats['urgency_stats'] = {
                'mean': df['urgency_score'].mean(),
                'distribution': df['urgency_score'].describe().to_dict()
            }
        
        return stats


def generate_enhanced_synthetic_data(n_samples: int = 10000,
                                   config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convenience function to generate enhanced synthetic data."""
    generator = EnhancedSyntheticDataGenerator(config)
    return generator.generate_dataset(n_samples) 