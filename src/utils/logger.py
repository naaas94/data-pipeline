"""
Enterprise-grade logging utility for the data pipeline.
Supports structured logging, metrics collection, and MLflow integration.
"""

import logging
import structlog
import time
from typing import Any, Dict, Optional
from prometheus_client import Counter, Histogram, Gauge
import mlflow


class PipelineLogger:
    """Enterprise logger with structured logging and metrics."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.metrics_enabled = config.get('monitoring', {}).get('metrics_enabled', True)
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(name)
        
        # Initialize metrics if enabled
        if self.metrics_enabled:
            self._init_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.metrics = {
            'records_processed': Counter(
                'pipeline_records_processed_total',
                'Total number of records processed',
                ['pipeline_name', 'stage']
            ),
            'processing_time': Histogram(
                'pipeline_processing_duration_seconds',
                'Time spent processing data',
                ['pipeline_name', 'stage']
            ),
            'data_quality_score': Gauge(
                'pipeline_data_quality_score',
                'Data quality score (0-1)',
                ['pipeline_name']
            ),
            'errors_total': Counter(
                'pipeline_errors_total',
                'Total number of errors',
                ['pipeline_name', 'error_type']
            )
        }
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, **kwargs)
        if self.metrics_enabled:
            error_type = kwargs.get('error_type', 'unknown')
            self.metrics['errors_total'].labels(
                pipeline_name=self.name, 
                error_type=error_type
            ).inc()
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, **kwargs)
    
    def log_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Log custom metric."""
        if self.metrics_enabled and metric_name in self.metrics:
            label_dict = labels or {}
            label_dict['pipeline_name'] = self.name
            self.metrics[metric_name].labels(**label_dict).observe(value)
    
    def log_mlflow_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log metric to MLflow."""
        try:
            mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            self.warning(f"Failed to log MLflow metric: {e}")
    
    def log_mlflow_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
        except Exception as e:
            self.warning(f"Failed to log MLflow params: {e}")
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: PipelineLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.info(
                f"Completed operation: {self.operation_name}",
                duration_seconds=duration
            )
            
            if self.logger.metrics_enabled:
                self.logger.log_metric(
                    'processing_time',
                    duration,
                    {'stage': self.operation_name}
                )


def get_logger(name: str, config: Dict[str, Any]) -> PipelineLogger:
    """Get configured pipeline logger."""
    return PipelineLogger(name, config) 