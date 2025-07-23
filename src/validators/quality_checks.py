"""
Enterprise-grade data quality checks using great_expectations and statistical analysis.
Supports comprehensive quality monitoring, outlier detection, and automated reporting.
"""

import pandas as pd
import numpy as np
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import FileDataContext
from great_expectations.data_context.types.resource_identifiers import GXCloudIdentifier
from typing import Any, Dict, List, Optional, Tuple
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataQualityChecker:
    """Enterprise data quality checker with multiple validation engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_config = config.get('validation', {}).get('quality_checks', [])
        self.thresholds = {check['check_type']: check['threshold'] for check in self.quality_config}
        self.quality_results = []
        
        # Initialize Great Expectations context
        try:
            self.ge_context = FileDataContext(project_config_dir="./great_expectations")
        except:
            self.ge_context = None
            warnings.warn("Great Expectations context not available. Using basic quality checks only.")
    
    def check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness (non-null values)."""
        completeness_scores = {}
        overall_completeness = 1.0
        
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            total_count = len(df)
            completeness = non_null_count / total_count if total_count > 0 else 0
            completeness_scores[column] = completeness
            overall_completeness = min(overall_completeness, completeness)
        
        threshold = self.thresholds.get('completeness', 0.95)
        passed = overall_completeness >= threshold
        
        return {
            'check_type': 'completeness',
            'passed': passed,
            'score': overall_completeness,
            'threshold': threshold,
            'details': completeness_scores,
            'failed_columns': [col for col, score in completeness_scores.items() if score < threshold]
        }
    
    def check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness."""
        uniqueness_scores = {}
        overall_uniqueness = 1.0
        
        for column in df.columns:
            unique_count = df[column].nunique()
            total_count = len(df)
            uniqueness = unique_count / total_count if total_count > 0 else 0
            uniqueness_scores[column] = uniqueness
            overall_uniqueness = min(overall_uniqueness, uniqueness)
        
        threshold = self.thresholds.get('uniqueness', 0.99)
        passed = overall_uniqueness >= threshold
        
        return {
            'check_type': 'uniqueness',
            'passed': passed,
            'score': overall_uniqueness,
            'threshold': threshold,
            'details': uniqueness_scores,
            'failed_columns': [col for col, score in uniqueness_scores.items() if score < threshold]
        }
    
    def check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity based on business rules."""
        validity_scores = {}
        overall_validity = 1.0
        
        for column in df.columns:
            valid_count = 0
            total_count = len(df)
            
            if column == 'intent':
                valid_values = ['privacy_request', 'data_deletion', 'opt_out', 'other']
                valid_count = df[column].isin(valid_values).sum()
            elif column == 'confidence':
                valid_count = ((df[column] >= 0.0) & (df[column] <= 1.0)).sum()
            elif column == 'text':
                valid_count = (df[column].str.len() > 0).sum()
            elif column == 'timestamp':
                valid_count = df[column].notna().sum()
            else:
                valid_count = total_count  # Assume valid for unknown columns
            
            validity = valid_count / total_count if total_count > 0 else 0
            validity_scores[column] = validity
            overall_validity = min(overall_validity, validity)
        
        threshold = self.thresholds.get('validity', 0.98)
        passed = overall_validity >= threshold
        
        return {
            'check_type': 'validity',
            'passed': passed,
            'score': overall_validity,
            'threshold': threshold,
            'details': validity_scores,
            'failed_columns': [col for col, score in validity_scores.items() if score < threshold]
        }
    
    def check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency across columns."""
        consistency_issues = []
        
        # Check for logical inconsistencies
        if 'intent' in df.columns and 'confidence' in df.columns:
            # Check if high confidence predictions have consistent patterns
            high_conf_mask = df['confidence'] > 0.8
            if high_conf_mask.sum() > 0:
                high_conf_data = df[high_conf_mask]
                intent_distribution = high_conf_data['intent'].value_counts()
                
                # Flag if one intent dominates too much (>90%)
                max_ratio = intent_distribution.max() / intent_distribution.sum()
                if max_ratio > 0.9:
                    consistency_issues.append(f"High confidence predictions are too concentrated: {max_ratio:.2%}")
        
        # Check timestamp consistency
        if 'timestamp' in df.columns:
            # Check for future timestamps
            future_timestamps = df['timestamp'] > pd.Timestamp.now()
            if future_timestamps.sum() > 0:
                consistency_issues.append(f"Found {future_timestamps.sum()} future timestamps")
            
            # Check for very old timestamps (older than 5 years)
            old_timestamps = df['timestamp'] < (pd.Timestamp.now() - pd.DateOffset(years=5))
            if old_timestamps.sum() > 0:
                consistency_issues.append(f"Found {old_timestamps.sum()} very old timestamps")
        
        overall_consistency = 1.0 if len(consistency_issues) == 0 else 0.5
        threshold = self.thresholds.get('consistency', 0.95)
        passed = overall_consistency >= threshold
        
        return {
            'check_type': 'consistency',
            'passed': passed,
            'score': overall_consistency,
            'threshold': threshold,
            'issues': consistency_issues
        }
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using statistical methods."""
        outlier_results = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == 'confidence':  # Skip confidence as it's bounded
                continue
                
            data = df[column].dropna()
            if len(data) == 0:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            
            # Isolation Forest
            if len(data) > 10:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_forest.fit(data.values.reshape(-1, 1))
                    iso_outliers = (iso_forest.predict(data.values.reshape(-1, 1)) == -1).sum()
                except:
                    iso_outliers = 0
            else:
                iso_outliers = 0
            
            outlier_results[column] = {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers,
                'isolation_forest_outliers': iso_outliers,
                'total_records': len(data)
            }
        
        return {
            'check_type': 'outlier_detection',
            'passed': True,  # Outliers don't necessarily mean failure
            'details': outlier_results
        }
    
    def check_with_great_expectations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run quality checks using Great Expectations."""
        if self.ge_context is None:
            return {
                'check_type': 'great_expectations',
                'passed': False,
                'error': 'Great Expectations context not available'
            }
        
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="privacy_intent_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Run expectations
            results = []
            
            # Expect column values to not be null
            result = self.ge_context.run_expectation_suite(
                expectation_suite_name="privacy_intent_suite",
                batch_request=batch_request
            )
            
            return {
                'check_type': 'great_expectations',
                'passed': result.success,
                'success_rate': result.statistics.get('success_percent', 0),
                'details': result.statistics
            }
            
        except Exception as e:
            return {
                'check_type': 'great_expectations',
                'passed': False,
                'error': str(e)
            }
    
    def check_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive quality check using all available methods."""
        checks = [
            self.check_completeness(df),
            self.check_uniqueness(df),
            self.check_validity(df),
            self.check_consistency(df),
            self.detect_outliers(df),
            self.check_with_great_expectations(df)
        ]
        
        # Calculate overall quality score
        passed_checks = sum(1 for check in checks if check.get('passed', False))
        total_checks = len(checks)
        overall_score = passed_checks / total_checks if total_checks > 0 else 0
        
        results = {
            'checks': checks,
            'overall_score': overall_score,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'overall_passed': overall_score >= 0.8  # 80% threshold for overall pass
        }
        
        self.quality_results.append(results)
        return results
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of all quality check runs."""
        if not self.quality_results:
            return {'message': 'No quality checks performed'}
        
        total_runs = len(self.quality_results)
        successful_runs = sum(1 for r in self.quality_results if r['overall_passed'])
        avg_score = np.mean([r['overall_score'] for r in self.quality_results])
        
        return {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            'average_quality_score': avg_score,
            'quality_trend': [r['overall_score'] for r in self.quality_results]
        }


def check_quality(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for quality checking."""
    checker = DataQualityChecker(config)
    return checker.check_quality(df) 