"""
Enterprise-grade sampling utilities with stratified sampling, class balancing, and SMOTE.
Supports advanced sampling strategies for ML data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings


class AdvancedSampler:
    """Enterprise sampler with multiple sampling strategies and class balancing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampling_config = config.get('sampling', {})
        self.strategy = self.sampling_config.get('strategy', 'stratified')
        self.balance_classes = self.sampling_config.get('balance_classes', True)
        self.oversampling_method = self.sampling_config.get('oversampling_method', 'smote')
        self.random_state = 42
    
    def stratified_sample(self, df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
        """Perform stratified sampling based on specified column."""
        if by not in df.columns:
            raise ValueError(f"Column '{by}' not found in DataFrame")
        
        # Get unique values and their counts
        value_counts = df[by].value_counts()
        
        # Calculate sampling ratios
        total_samples = min(n, len(df))
        sampling_ratios = {}
        
        for value, count in value_counts.items():
            # Proportional sampling
            ratio = min(count / len(df), 1.0)
            sampling_ratios[value] = int(ratio * total_samples)
        
        # Ensure we get at least one sample from each category
        remaining_samples = total_samples - sum(sampling_ratios.values())
        if remaining_samples > 0:
            # Distribute remaining samples proportionally
            for value in value_counts.index:
                if remaining_samples <= 0:
                    break
                additional = min(remaining_samples, value_counts[value] - sampling_ratios[value])
                sampling_ratios[value] += additional
                remaining_samples -= additional
        
        # Perform stratified sampling
        sampled_dfs = []
        for value, sample_size in sampling_ratios.items():
            if sample_size > 0:
                subset = df[df[by] == value]
                if len(subset) >= sample_size:
                    sampled = subset.sample(n=sample_size, random_state=self.random_state)
                else:
                    # If not enough samples, take all available
                    sampled = subset
                sampled_dfs.append(sampled)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def random_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Perform random sampling."""
        return df.sample(n=min(n, len(df)), random_state=self.random_state)
    
    def systematic_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Perform systematic sampling."""
        if n >= len(df):
            return df
        
        step = len(df) // n
        indices = list(range(0, len(df), step))[:n]
        return df.iloc[indices].reset_index(drop=True)
    
    def cluster_sample(self, df: pd.DataFrame, n: int, cluster_column: str) -> pd.DataFrame:
        """Perform cluster sampling based on a categorical column."""
        if cluster_column not in df.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found")
        
        clusters = df[cluster_column].unique()
        n_clusters = min(len(clusters), n)
        
        # Randomly select clusters
        selected_clusters = np.random.choice(clusters.tolist(), size=n_clusters, replace=False)
        
        # Sample from selected clusters
        sampled_dfs = []
        samples_per_cluster = n // n_clusters
        
        for cluster in selected_clusters:
            cluster_data = df[df[cluster_column] == cluster]
            if len(cluster_data) > 0:
                sampled = cluster_data.sample(
                    n=min(samples_per_cluster, len(cluster_data)),
                    random_state=self.random_state
                )
                sampled_dfs.append(sampled)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def balance_classes_method(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Balance classes using various techniques."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        value_counts = df[target_column].value_counts()
        min_count = value_counts.min()
        max_count = value_counts.max()
        
        if max_count / min_count < 2:  # Already reasonably balanced
            return df
        
        balanced_dfs = []
        
        for value in value_counts.index:
            subset = df[df[target_column] == value]
            count = value_counts[value]
            
            if count > min_count:
                # Undersample majority classes
                balanced_subset = subset.sample(n=min_count, random_state=self.random_state)
            else:
                # Keep minority classes as is
                balanced_subset = subset
            
            balanced_dfs.append(balanced_subset)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def apply_smote(self, df: pd.DataFrame, target_column: str, text_column: str = None) -> pd.DataFrame:
        """Apply SMOTE for oversampling minority classes."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare features for SMOTE
        feature_columns = [col for col in df.columns if col not in [target_column, text_column]]
        
        if len(feature_columns) == 0:
            warnings.warn("No numeric features available for SMOTE. Returning original data.")
            return df
        
        X = df[feature_columns].fillna(0)  # Fill NaN with 0 for SMOTE
        y = df[target_column]
        
        # Apply SMOTE
        if self.oversampling_method == 'smote':
            smote = SMOTE(random_state=self.random_state)
        elif self.oversampling_method == 'adasyn':
            smote = ADASYN(random_state=self.random_state)
        elif self.oversampling_method == 'smoteenn':
            smote = SMOTEENN(random_state=self.random_state)
        elif self.oversampling_method == 'smotetomek':
            smote = SMOTETomek(random_state=self.random_state)
        else:
            smote = SMOTE(random_state=self.random_state)
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Create new DataFrame with resampled data
            resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
            resampled_df[target_column] = y_resampled
            
            # Add text column if it exists (duplicate from original data)
            if text_column and text_column in df.columns:
                # For text data, we'll duplicate existing texts for synthetic samples
                original_texts = df[text_column].tolist()
                resampled_texts = []
                
                for i, label in enumerate(y_resampled):
                    if i < len(original_texts):
                        resampled_texts.append(original_texts[i])
                    else:
                        # For synthetic samples, use a placeholder or duplicate
                        resampled_texts.append(f"Synthetic sample for {label}")
                
                resampled_df[text_column] = resampled_texts
            
            return resampled_df
            
        except Exception as e:
            warnings.warn(f"SMOTE failed: {e}. Returning original data.")
            return df
    
    def create_train_test_split(self, df: pd.DataFrame, target_column: str, 
                               test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation/test splits."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df[target_column],
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val[target_column],
            random_state=self.random_state
        )
        
        return train, val, test
    
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main sampling method that applies configured strategy."""
        by = self.sampling_config.get('by', 'intent')
        n = self.sampling_config.get('n', 10000)
        
        # Apply sampling strategy
        if self.strategy == 'stratified':
            sampled_df = self.stratified_sample(df, by, n)
        elif self.strategy == 'random':
            sampled_df = self.random_sample(df, n)
        elif self.strategy == 'systematic':
            sampled_df = self.systematic_sample(df, n)
        elif self.strategy == 'cluster':
            cluster_column = self.sampling_config.get('cluster_column', by)
            sampled_df = self.cluster_sample(df, n, cluster_column)
        else:
            sampled_df = self.stratified_sample(df, by, n)  # Default to stratified
        
        # Apply class balancing if requested
        if self.balance_classes:
            if self.oversampling_method in ['smote', 'adasyn', 'smoteenn', 'smotetomek']:
                sampled_df = self.apply_smote(sampled_df, by, text_column='text')
            else:
                sampled_df = self.balance_classes_method(sampled_df, by)
        
        return sampled_df
    
    def get_sampling_stats(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get statistics about the sampling results."""
        if target_column not in df.columns:
            return {'error': f"Target column '{target_column}' not found"}
        
        value_counts = df[target_column].value_counts()
        total_samples = len(df)
        
        return {
            'total_samples': total_samples,
            'class_distribution': value_counts.to_dict(),
            'class_ratios': (value_counts / total_samples).to_dict(),
            'balance_score': value_counts.min() / value_counts.max() if value_counts.max() > 0 else 0,
            'unique_classes': len(value_counts)
        }


def stratified_sample(df: pd.DataFrame, by: str, n: int, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Convenience function for stratified sampling."""
    if config is None:
        config = {'sampling': {'strategy': 'stratified', 'by': by, 'n': n}}
    
    sampler = AdvancedSampler(config)
    return sampler.stratified_sample(df, by, n) 