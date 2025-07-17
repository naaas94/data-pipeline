"""
Enterprise-grade schema validation using pandera and pydantic.
Supports complex validation rules, custom checks, and detailed reporting.
"""

import pandas as pd
import pandera as pa
from pandera.typing import Series
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime


class PrivacyIntentSchema(pa.SchemaModel):
    """Pandera schema for privacy intent classification data."""
    
    text: Series[str] = pa.Field(
        nullable=False,
        description="Input text for classification"
    )
    
    intent: Series[str] = pa.Field(
        nullable=False,
        isin=["privacy_request", "data_deletion", "opt_out", "other"],
        description="Target intent classification"
    )
    
    confidence: Series[float] = pa.Field(
        nullable=False,
        ge=0.0,
        le=1.0,
        description="Confidence score for the intent"
    )
    
    timestamp: Series[datetime] = pa.Field(
        nullable=False,
        description="Timestamp of the data point"
    )
    
    @pa.check("text")
    def text_not_empty(cls, series: pd.Series) -> bool:
        """Check that text is not empty."""
        return series.str.len() > 0
    
    @pa.check("text")
    def text_max_length(cls, series: pd.Series) -> bool:
        """Check that text doesn't exceed maximum length."""
        return series.str.len() <= 10000
    
    @pa.check("confidence")
    def confidence_valid_range(cls, series: pd.Series) -> bool:
        """Check that confidence is in valid range."""
        return (series >= 0.0) & (series <= 1.0)


class PydanticDataPoint(BaseModel):
    """Pydantic model for individual data point validation."""
    
    text: str = Field(..., min_length=1, max_length=10000)
    intent: str = Field(..., regex="^(privacy_request|data_deletion|opt_out|other)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        return v


class SchemaValidator:
    """Enterprise schema validator with multiple validation engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schema_config = config.get('validation', {}).get('schema', [])
        self.pandera_schema = PrivacyIntentSchema
        self.validation_results = []
    
    def validate_pandera(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame using Pandera schema."""
        try:
            # Apply pandera validation
            validated_df = self.pandera_schema.validate(df)
            
            return {
                'valid': True,
                'validated_rows': len(validated_df),
                'errors': [],
                'warnings': []
            }
        except pa.errors.SchemaError as e:
            return {
                'valid': False,
                'validated_rows': 0,
                'errors': [str(e)],
                'warnings': []
            }
    
    def validate_pydantic(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data using Pydantic models."""
        errors = []
        valid_count = 0
        
        for i, item in enumerate(data):
            try:
                PydanticDataPoint(**item)
                valid_count += 1
            except Exception as e:
                errors.append(f"Row {i}: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'validated_rows': valid_count,
            'errors': errors,
            'warnings': []
        }
    
    def validate_custom_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply custom validation rules from config."""
        errors = []
        warnings = []
        
        for rule in self.schema_config:
            column = rule.get('column')
            rule_type = rule.get('type')
            
            if column not in df.columns:
                errors.append(f"Required column '{column}' not found")
                continue
            
            if rule_type == 'string':
                if not df[column].dtype == 'object':
                    errors.append(f"Column '{column}' should be string type")
                
                # Check allowed values if specified
                allowed_values = rule.get('allowed_values')
                if allowed_values:
                    invalid_values = df[~df[column].isin(allowed_values)][column].unique()
                    if len(invalid_values) > 0:
                        errors.append(f"Column '{column}' contains invalid values: {invalid_values}")
            
            elif rule_type == 'float':
                if not pd.api.types.is_numeric_dtype(df[column]):
                    errors.append(f"Column '{column}' should be numeric type")
                
                # Check range if specified
                min_value = rule.get('min_value')
                max_value = rule.get('max_value')
                
                if min_value is not None and df[column].min() < min_value:
                    warnings.append(f"Column '{column}' contains values below minimum {min_value}")
                
                if max_value is not None and df[column].max() > max_value:
                    warnings.append(f"Column '{column}' contains values above maximum {max_value}")
            
            elif rule_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    errors.append(f"Column '{column}' should be datetime type")
        
        return {
            'valid': len(errors) == 0,
            'validated_rows': len(df) if len(errors) == 0 else 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive schema validation using multiple engines."""
        results = {
            'pandera': self.validate_pandera(df),
            'custom_rules': self.validate_custom_rules(df),
            'overall_valid': True,
            'total_errors': 0,
            'total_warnings': 0
        }
        
        # Aggregate results
        for engine_result in [results['pandera'], results['custom_rules']]:
            if not engine_result['valid']:
                results['overall_valid'] = False
            results['total_errors'] += len(engine_result['errors'])
            results['total_warnings'] += len(engine_result['warnings'])
        
        self.validation_results.append(results)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs."""
        if not self.validation_results:
            return {'message': 'No validation runs performed'}
        
        total_runs = len(self.validation_results)
        successful_runs = sum(1 for r in self.validation_results if r['overall_valid'])
        
        return {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            'total_errors': sum(r['total_errors'] for r in self.validation_results),
            'total_warnings': sum(r['total_warnings'] for r in self.validation_results)
        }


def validate_schema(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for schema validation."""
    validator = SchemaValidator(config)
    return validator.validate_schema(df) 