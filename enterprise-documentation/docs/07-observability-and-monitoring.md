# Enhanced Observability and Monitoring

The enhanced PCC data pipeline provides comprehensive observability through advanced lineage tracking, detailed metrics collection, and enterprise-grade monitoring capabilities.

## Advanced Metrics Collection

### **Pipeline Performance Metrics**

The pipeline collects comprehensive metrics across all stages using Prometheus with structured logging:

```python
class EnhancedPipelineMetrics:
    """Comprehensive metrics collection for the data pipeline."""
    
    def __init__(self):
        # Processing metrics
        self.records_processed = Counter(
            'pipeline_records_processed_total',
            'Total records processed by stage',
            ['pipeline_name', 'stage', 'intent']
        )
        
        self.processing_duration = Histogram(
            'pipeline_processing_duration_seconds',
            'Processing time per stage',
            ['pipeline_name', 'stage'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'pipeline_data_quality_score',
            'Overall data quality score (0-1)',
            ['pipeline_name', 'dataset_type']
        )
        
        self.validation_errors = Counter(
            'pipeline_validation_errors_total',
            'Number of validation errors',
            ['pipeline_name', 'validation_type', 'error_type']
        )
        
        # Feature engineering metrics
        self.features_extracted = Gauge(
            'pipeline_features_extracted_count',
            'Number of features extracted',
            ['pipeline_name', 'feature_category']
        )
        
        self.embedding_generation_time = Histogram(
            'pipeline_embedding_generation_seconds',
            'Time to generate embeddings',
            ['pipeline_name', 'embedding_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')]
        )
        
        # Resource utilization
        self.memory_usage = Gauge(
            'pipeline_memory_usage_bytes',
            'Memory usage during processing',
            ['pipeline_name', 'stage']
        )
```

### **Data Quality Tracking**

```python
class DataQualityMetrics:
    """Specialized metrics for data quality monitoring."""
    
    def track_quality_metrics(self, dataset_id: str, quality_results: Dict[str, Any]):
        """Track comprehensive quality metrics."""
        
        # Overall quality score
        overall_score = quality_results.get('overall_score', 0.0)
        self.data_quality_score.labels(
            pipeline_name='pcc_data_pipeline',
            dataset_type='training_data'
        ).set(overall_score)
        
        # Individual check results
        for check in quality_results.get('checks', []):
            check_type = check.get('check_type')
            passed = check.get('passed', False)
            score = check.get('score', 0.0)
            
            # Track check success rate
            self.quality_check_success.labels(
                pipeline_name='pcc_data_pipeline',
                check_type=check_type
            ).set(1.0 if passed else 0.0)
            
            # Track detailed scores
            self.quality_check_score.labels(
                pipeline_name='pcc_data_pipeline',
                check_type=check_type
            ).set(score)
        
        # Privacy-specific metrics
        if 'privacy_checks' in quality_results:
            privacy_metrics = quality_results['privacy_checks']
            
            self.privacy_keyword_density.labels(
                pipeline_name='pcc_data_pipeline'
            ).set(privacy_metrics.get('avg_keyword_density', 0.0))
            
            self.pii_detection_rate.labels(
                pipeline_name='pcc_data_pipeline'
            ).set(privacy_metrics.get('pii_detection_rate', 0.0))
```

### **Feature Engineering Metrics**

```python
def track_feature_engineering_metrics(self, extraction_result: FeatureExtractionResult):
    """Track feature extraction performance and quality."""
    
    # Feature count by category
    for category, features in extraction_result.feature_categories.items():
        self.features_extracted.labels(
            pipeline_name='pcc_data_pipeline',
            feature_category=category
        ).set(len(features))
    
    # Extraction performance
    self.processing_duration.labels(
        pipeline_name='pcc_data_pipeline',
        stage='feature_engineering'
    ).observe(extraction_result.extraction_time)
    
    # Feature importance distribution
    if extraction_result.feature_importance:
        top_features = sorted(
            extraction_result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for feature, importance in top_features:
            self.feature_importance_score.labels(
                pipeline_name='pcc_data_pipeline',
                feature_name=feature
            ).set(importance)
```

## Complete Data Lineage Tracking

### **Advanced Lineage System**

```python
class EnhancedDataLineageTracker:
    """Complete data lineage tracking with rich metadata."""
    
    def track_dataset_with_quality(self, data: pd.DataFrame, name: str, 
                                  stage: str, quality_results: Dict = None) -> str:
        """Track dataset with integrated quality metrics."""
        
        dataset_id = self._generate_dataset_id(data)
        
        # Basic dataset information
        dataset_info = {
            'dataset_id': dataset_id,
            'name': name,
            'stage': stage,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'shape': list(data.shape),
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'memory_usage': data.memory_usage(deep=True).sum(),
            'row_count': len(data),
            'null_counts': data.isnull().sum().to_dict()
        }
        
        # Add quality metrics if available
        if quality_results:
            dataset_info['quality_metrics'] = quality_results
            dataset_info['quality_score'] = quality_results.get('overall_score', 0.0)
        
        # Privacy-specific metadata
        if 'intent' in data.columns:
            intent_distribution = data['intent'].value_counts().to_dict()
            dataset_info['intent_distribution'] = intent_distribution
            dataset_info['class_balance_score'] = self._calculate_balance_score(intent_distribution)
        
        # Feature metadata
        feature_columns = [col for col in data.columns 
                          if col not in ['text', 'intent', 'timestamp', 'embeddings']]
        if feature_columns:
            dataset_info['feature_count'] = len(feature_columns)
            dataset_info['feature_categories'] = self._categorize_features(feature_columns)
        
        # Embedding metadata
        if 'embeddings' in data.columns:
            sample_embedding = data['embeddings'].iloc[0]
            if isinstance(sample_embedding, list):
                dataset_info['embedding_dimension'] = len(sample_embedding)
                dataset_info['embedding_stats'] = self._calculate_embedding_stats(data['embeddings'])
        
        self.lineage_data['datasets'][dataset_id] = dataset_info
        return dataset_id
    
    def generate_lineage_visualization(self) -> Dict[str, Any]:
        """Generate data for lineage graph visualization."""
        
        nodes = []
        edges = []
        
        # Dataset nodes with rich metadata
        for dataset_id, dataset_info in self.lineage_data['datasets'].items():
            node = {
                'id': dataset_id,
                'type': 'dataset',
                'label': dataset_info['name'],
                'shape': dataset_info.get('shape', [0, 0]),
                'quality_score': dataset_info.get('quality_score', 0.0),
                'stage': dataset_info['stage'],
                'timestamp': dataset_info['timestamp']
            }
            
            # Color coding based on quality score
            if dataset_info.get('quality_score', 0) > 0.9:
                node['color'] = 'green'
            elif dataset_info.get('quality_score', 0) > 0.7:
                node['color'] = 'yellow'
            else:
                node['color'] = 'red'
                
            nodes.append(node)
        
        # Stage nodes with performance metrics
        for stage in self.lineage_data['stages']:
            node = {
                'id': stage['stage_id'],
                'type': 'stage',
                'label': stage['name'],
                'execution_time': stage.get('execution_time', 0),
                'status': stage.get('status', 'unknown'),
                'timestamp': stage['timestamp']
            }
            nodes.append(node)
            
            # Create edges with metadata
            for input_id in stage['inputs']:
                edges.append({
                    'from': input_id,
                    'to': stage['stage_id'],
                    'type': 'input',
                    'label': 'processes'
                })
            
            for output_id in stage['outputs']:
                edges.append({
                    'from': stage['stage_id'],
                    'to': output_id,
                    'type': 'output',
                    'label': 'produces'
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_datasets': len(self.lineage_data['datasets']),
                'total_stages': len(self.lineage_data['stages']),
                'pipeline_version': self.lineage_data.get('pipeline_version', '1.0.0'),
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
```

### **Compliance and Audit Reporting**

```python
class ComplianceReporter:
    """Generate compliance reports from lineage data."""
    
    def generate_gdpr_compliance_report(self, lineage_tracker: DataLineageTracker) -> Dict[str, Any]:
        """Generate GDPR compliance report from lineage data."""
        
        report = {
            'report_type': 'GDPR_Compliance',
            'generation_date': datetime.now().isoformat(),
            'pipeline_version': lineage_tracker.lineage_data.get('pipeline_version'),
            'data_processing_activities': []
        }
        
        # Document all data processing activities
        for stage in lineage_tracker.lineage_data['stages']:
            activity = {
                'activity_name': stage['name'],
                'processing_date': stage['timestamp'],
                'purpose': self._map_stage_to_purpose(stage['name']),
                'legal_basis': 'Legitimate interest - ML model training',
                'data_categories': self._extract_data_categories(stage),
                'retention_period': '2 years',
                'processing_time': stage.get('execution_time', 0),
                'data_subjects_count': self._estimate_data_subjects(stage)
            }
            report['data_processing_activities'].append(activity)
        
        # Data quality and accuracy measures
        report['data_quality_measures'] = {
            'validation_checks_performed': len([s for s in lineage_tracker.lineage_data['stages'] 
                                              if 'validation' in s['name'].lower()]),
            'overall_quality_score': self._calculate_overall_quality_score(lineage_tracker),
            'data_accuracy_measures': [
                'Schema validation with Pydantic',
                'Quality checks with Great Expectations',
                'Business rule validation',
                'Privacy-specific validation'
            ]
        }
        
        return report
```

## Enhanced Health Checks and Alerting

### **Comprehensive Health Monitoring**

```python
class EnhancedHealthChecker:
    """Comprehensive health monitoring for the data pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checks = {
            'pipeline_status': self.check_pipeline_status,
            'data_quality': self.check_data_quality_trends,
            'resource_utilization': self.check_resource_utilization,
            'lineage_tracking': self.check_lineage_tracking,
            'contract_compliance': self.check_contract_compliance,
            'embedding_generation': self.check_embedding_health
        }
    
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check across all components."""
        
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'alerts': [],
            'recommendations': []
        }
        
        for check_name, check_function in self.health_checks.items():
            try:
                check_result = check_function()
                health_status['checks'][check_name] = check_result
                
                if not check_result.get('healthy', True):
                    health_status['overall_status'] = 'degraded'
                    health_status['alerts'].extend(check_result.get('alerts', []))
                
            except Exception as e:
                health_status['checks'][check_name] = {
                    'healthy': False,
                    'error': str(e),
                    'alerts': [f"Health check {check_name} failed: {str(e)}"]
                }
                health_status['overall_status'] = 'unhealthy'
        
        return health_status
    
    def check_data_quality_trends(self) -> Dict[str, Any]:
        """Monitor data quality trends over time."""
        
        # Query recent quality metrics from Prometheus
        quality_scores = self._query_prometheus_metrics(
            'pipeline_data_quality_score',
            time_range='1h'
        )
        
        if not quality_scores:
            return {
                'healthy': False,
                'alerts': ['No quality metrics available'],
                'score': None
            }
        
        current_score = quality_scores[-1]['value']
        trend = self._calculate_trend(quality_scores)
        
        alerts = []
        if current_score < 0.8:
            alerts.append(f"Data quality score below threshold: {current_score:.3f}")
        
        if trend < -0.05:  # Declining trend
            alerts.append(f"Data quality declining: {trend:.3f} per hour")
        
        return {
            'healthy': len(alerts) == 0,
            'current_score': current_score,
            'trend': trend,
            'alerts': alerts,
            'recommendations': self._generate_quality_recommendations(current_score, trend)
        }
    
    def check_embedding_health(self) -> Dict[str, Any]:
        """Check embedding generation health and performance."""
        
        # Check embedding cache status
        cache_dir = self.config.get('embeddings', {}).get('cache_dir', 'cache/embeddings')
        cache_files = list(Path(cache_dir).glob('*.pkl')) if os.path.exists(cache_dir) else []
        
        # Check recent embedding generation metrics
        embedding_times = self._query_prometheus_metrics(
            'pipeline_embedding_generation_seconds',
            time_range='6h'
        )
        
        alerts = []
        recommendations = []
        
        if not cache_files:
            alerts.append("No embedding cache files found")
            recommendations.append("Run embedding generation to populate cache")
        
        if embedding_times:
            avg_time = sum(float(t['value']) for t in embedding_times) / len(embedding_times)
            if avg_time > 60:  # More than 1 minute
                alerts.append(f"Embedding generation slow: {avg_time:.1f}s average")
                recommendations.append("Consider using cached embeddings or GPU acceleration")
        
        return {
            'healthy': len(alerts) == 0,
            'cache_files_count': len(cache_files),
            'avg_generation_time': avg_time if embedding_times else None,
            'alerts': alerts,
            'recommendations': recommendations
        }
```

### **Intelligent Alerting System**

```python
class IntelligentAlerting:
    """Smart alerting based on pipeline performance and trends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_thresholds = config.get('alerting', {
            'data_quality_threshold': 0.8,
            'processing_time_threshold': 300,  # 5 minutes
            'error_rate_threshold': 0.05,
            'trend_degradation_threshold': -0.1
        })
    
    def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate current metrics and generate intelligent alerts."""
        
        alerts = []
        
        # Data quality alerts
        current_quality = metrics.get('data_quality_score', 1.0)
        if current_quality < self.alert_thresholds['data_quality_threshold']:
            severity = 'critical' if current_quality < 0.6 else 'warning'
            alerts.append({
                'type': 'data_quality',
                'severity': severity,
                'message': f"Data quality score {current_quality:.3f} below threshold {self.alert_thresholds['data_quality_threshold']}",
                'timestamp': datetime.now().isoformat(),
                'recommended_actions': [
                    'Check data source quality',
                    'Review validation rules',
                    'Investigate recent data changes'
                ],
                'metrics': {'current_score': current_quality}
            })
        
        # Performance alerts
        processing_times = metrics.get('processing_times', [])
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            if avg_time > self.alert_thresholds['processing_time_threshold']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f"Average processing time {avg_time:.1f}s exceeds threshold",
                    'timestamp': datetime.now().isoformat(),
                    'recommended_actions': [
                        'Check resource utilization',
                        'Consider scaling up processing engine',
                        'Optimize feature extraction pipeline'
                    ],
                    'metrics': {'avg_processing_time': avg_time}
                })
        
        # Embedding generation alerts
        embedding_failures = metrics.get('embedding_failures', 0)
        total_embeddings = metrics.get('total_embedding_attempts', 1)
        failure_rate = embedding_failures / total_embeddings
        
        if failure_rate > 0.1:  # 10% failure rate
            alerts.append({
                'type': 'embedding_generation',
                'severity': 'critical' if failure_rate > 0.5 else 'warning',
                'message': f"Embedding generation failure rate {failure_rate:.1%}",
                'timestamp': datetime.now().isoformat(),
                'recommended_actions': [
                    'Check embedding model availability',
                    'Verify model cache integrity',
                    'Review memory allocation',
                    'Consider fallback embedding strategy'
                ],
                'metrics': {'failure_rate': failure_rate, 'failures': embedding_failures}
            })
        
        return alerts
```

## Monitoring Dashboards and Visualization

### **Grafana Dashboard Configuration**

```yaml
# grafana/dashboards/pcc-pipeline-dashboard.yaml
dashboard:
  title: "PCC Data Pipeline Monitoring"
  panels:
    - title: "Pipeline Health Overview"
      type: "stat"
      targets:
        - expr: "pipeline_data_quality_score"
        - expr: "rate(pipeline_records_processed_total[5m])"
        - expr: "pipeline_processing_duration_seconds"
    
    - title: "Data Quality Trends"
      type: "time-series"
      targets:
        - expr: "pipeline_data_quality_score"
          legend: "Overall Quality"
        - expr: "pipeline_quality_check_score{check_type='completeness'}"
          legend: "Completeness"
        - expr: "pipeline_quality_check_score{check_type='validity'}"
          legend: "Validity"
    
    - title: "Feature Engineering Performance"
      type: "time-series"
      targets:
        - expr: "pipeline_features_extracted_count"
          legend: "Features Extracted"
        - expr: "rate(pipeline_embedding_generation_seconds[5m])"
          legend: "Embedding Generation Rate"
    
    - title: "Lineage and Compliance"
      type: "table"
      targets:
        - expr: "pipeline_datasets_tracked_total"
        - expr: "pipeline_stages_completed_total"
        - expr: "pipeline_validation_errors_total"
```

### **Real-time Monitoring Interface**

```python
class RealTimeMonitoringAPI:
    """REST API for real-time pipeline monitoring."""
    
    @app.route('/api/v1/pipeline/status')
    def get_pipeline_status(self):
        """Get current pipeline status with all metrics."""
        
        return {
            'pipeline_status': self.health_checker.comprehensive_health_check(),
            'current_metrics': self.metrics_collector.get_current_metrics(),
            'recent_lineage': self.lineage_tracker.get_recent_activity(hours=24),
            'quality_trends': self.quality_monitor.get_quality_trends(hours=24),
            'alerts': self.alerting_system.get_active_alerts()
        }
    
    @app.route('/api/v1/lineage/visualization')
    def get_lineage_visualization(self):
        """Get lineage data formatted for visualization."""
        
        return self.lineage_tracker.generate_lineage_visualization()
    
    @app.route('/api/v1/compliance/report')
    def get_compliance_report(self):
        """Generate compliance report for auditing."""
        
        return self.compliance_reporter.generate_gdpr_compliance_report(
            self.lineage_tracker
        )
```

This enhanced observability system provides complete visibility into pipeline operations, enabling proactive monitoring, compliance reporting, and performance optimization.

 