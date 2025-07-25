"""
Data lineage tracking for privacy intent classification pipeline.
Tracks data provenance, metadata, and pipeline execution history.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
import uuid
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path


class DataLineageTracker:
    """Advanced data lineage tracker for ML pipelines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lineage_config = self.config.get('lineage', {})
        self.storage_dir = self.lineage_config.get('storage_dir', 'metadata/lineage')
        self.experiment_id = str(uuid.uuid4())
        self.lineage_data = {
            'experiment_id': self.experiment_id,
            'pipeline_version': self.config.get('version', '1.0.0'),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'stages': [],
            'datasets': {},
            'models': {},
            'metrics': {},
            'artifacts': {},
            'config': self.config
        }
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _generate_dataset_id(self, data: Union[pd.DataFrame, np.ndarray, str]) -> str:
        """Generate unique ID for dataset based on content hash."""
        if isinstance(data, pd.DataFrame):
            # Hash dataframe content
            content = data.to_string().encode('utf-8')
        elif isinstance(data, np.ndarray):
            # Hash array content
            content = data.tobytes()
        elif isinstance(data, str):
            # Hash string content (file path or identifier)
            content = data.encode('utf-8')
        else:
            # Fallback to string representation
            content = str(data).encode('utf-8')
        
        return hashlib.sha256(content).hexdigest()[:16]
    
    def track_dataset(self, data: Union[pd.DataFrame, np.ndarray], 
                     dataset_name: str,
                     stage: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a dataset and its metadata."""
        dataset_id = self._generate_dataset_id(data)
        
        dataset_info = {
            'dataset_id': dataset_id,
            'name': dataset_name,
            'stage': stage,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': type(data).__name__,
            'metadata': metadata or {}
        }
        
        # Add dataset-specific information
        if isinstance(data, pd.DataFrame):
            dataset_info.update({
                'shape': list(data.shape),
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'memory_usage': data.memory_usage(deep=True).sum(),
                'null_counts': data.isnull().sum().to_dict(),
                'sample_data': data.head(3).to_dict()  # Small sample for reference
            })
        elif isinstance(data, np.ndarray):
            dataset_info.update({
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'memory_usage': data.nbytes,
                'stats': {
                    'mean': float(np.mean(data)) if data.size > 0 else None,
                    'std': float(np.std(data)) if data.size > 0 else None,
                    'min': float(np.min(data)) if data.size > 0 else None,
                    'max': float(np.max(data)) if data.size > 0 else None
                }
            })
        
        self.lineage_data['datasets'][dataset_id] = dataset_info
        return dataset_id
    
    def track_stage(self, stage_name: str, 
                   inputs: List[str], 
                   outputs: List[str],
                   parameters: Optional[Dict[str, Any]] = None,
                   metrics: Optional[Dict[str, Any]] = None,
                   artifacts: Optional[Dict[str, str]] = None) -> str:
        """Track a pipeline stage execution."""
        stage_id = str(uuid.uuid4())
        
        stage_info = {
            'stage_id': stage_id,
            'name': stage_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'inputs': inputs,
            'outputs': outputs,
            'parameters': parameters or {},
            'metrics': metrics or {},
            'artifacts': artifacts or {},
            'execution_time': None  # Will be updated when stage completes
        }
        
        self.lineage_data['stages'].append(stage_info)
        return stage_id
    
    def update_stage_completion(self, stage_id: str, 
                              execution_time: float,
                              status: str = 'completed',
                              error_info: Optional[str] = None):
        """Update stage completion information."""
        for stage in self.lineage_data['stages']:
            if stage['stage_id'] == stage_id:
                stage['execution_time'] = execution_time
                stage['status'] = status
                stage['completed_at'] = datetime.now(timezone.utc).isoformat()
                if error_info:
                    stage['error_info'] = error_info
                break
    
    def track_model_artifact(self, model_info: Dict[str, Any], 
                            model_path: Optional[str] = None) -> str:
        """Track model artifacts and metadata."""
        model_id = str(uuid.uuid4())
        
        model_data = {
            'model_id': model_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'path': model_path,
            'info': model_info
        }
        
        self.lineage_data['models'][model_id] = model_data
        return model_id
    
    def track_data_quality_metrics(self, dataset_id: str, 
                                  quality_metrics: Dict[str, Any]):
        """Track data quality metrics for a dataset."""
        if dataset_id in self.lineage_data['datasets']:
            self.lineage_data['datasets'][dataset_id]['quality_metrics'] = quality_metrics
    
    def track_pipeline_metrics(self, metrics: Dict[str, Any]):
        """Track overall pipeline metrics."""
        self.lineage_data['metrics'].update(metrics)
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """Get lineage information for a specific dataset."""
        lineage = {
            'dataset_info': self.lineage_data['datasets'].get(dataset_id, {}),
            'upstream_stages': [],
            'downstream_stages': []
        }
        
        # Find stages that produced or consumed this dataset
        for stage in self.lineage_data['stages']:
            if dataset_id in stage['inputs']:
                lineage['downstream_stages'].append(stage)
            if dataset_id in stage['outputs']:
                lineage['upstream_stages'].append(stage)
        
        return lineage
    
    def export_lineage_graph(self) -> Dict[str, Any]:
        """Export lineage as a graph structure."""
        nodes = []
        edges = []
        
        # Add dataset nodes
        for dataset_id, dataset_info in self.lineage_data['datasets'].items():
            nodes.append({
                'id': dataset_id,
                'type': 'dataset',
                'label': dataset_info['name'],
                'metadata': dataset_info
            })
        
        # Add stage nodes and edges
        for stage in self.lineage_data['stages']:
            stage_id = stage['stage_id']
            nodes.append({
                'id': stage_id,
                'type': 'stage',
                'label': stage['name'],
                'metadata': stage
            })
            
            # Add edges from inputs to stage
            for input_id in stage['inputs']:
                edges.append({
                    'from': input_id,
                    'to': stage_id,
                    'type': 'input'
                })
            
            # Add edges from stage to outputs
            for output_id in stage['outputs']:
                edges.append({
                    'from': stage_id,
                    'to': output_id,
                    'type': 'output'
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'experiment_id': self.experiment_id,
                'created_at': self.lineage_data['created_at'],
                'pipeline_version': self.lineage_data['pipeline_version']
            }
        }
    
    def save_lineage(self, filename: Optional[str] = None) -> str:
        """Save lineage data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lineage_{self.experiment_id}_{timestamp}.json"
        
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.lineage_data, f, indent=2, default=str)
        
        print(f"Lineage data saved to: {filepath}")
        return filepath
    
    def load_lineage(self, filepath: str):
        """Load lineage data from file."""
        with open(filepath, 'r') as f:
            self.lineage_data = json.load(f)
        
        self.experiment_id = self.lineage_data.get('experiment_id', str(uuid.uuid4()))
    
    def generate_lineage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive lineage report."""
        report = {
            'experiment_summary': {
                'experiment_id': self.experiment_id,
                'pipeline_version': self.lineage_data['pipeline_version'],
                'created_at': self.lineage_data['created_at'],
                'total_stages': len(self.lineage_data['stages']),
                'total_datasets': len(self.lineage_data['datasets']),
                'total_models': len(self.lineage_data['models'])
            },
            'stage_summary': [],
            'dataset_summary': [],
            'quality_summary': {},
            'performance_summary': {}
        }
        
        # Stage summary
        for stage in self.lineage_data['stages']:
            report['stage_summary'].append({
                'name': stage['name'],
                'status': stage.get('status', 'unknown'),
                'execution_time': stage.get('execution_time'),
                'inputs_count': len(stage['inputs']),
                'outputs_count': len(stage['outputs'])
            })
        
        # Dataset summary
        for dataset_id, dataset_info in self.lineage_data['datasets'].items():
            summary = {
                'name': dataset_info['name'],
                'stage': dataset_info['stage'],
                'type': dataset_info['type'],
                'shape': dataset_info.get('shape'),
                'memory_usage': dataset_info.get('memory_usage')
            }
            
            # Add quality metrics if available
            if 'quality_metrics' in dataset_info:
                summary['quality_score'] = dataset_info['quality_metrics'].get('overall_score')
            
            report['dataset_summary'].append(summary)
        
        # Performance summary
        execution_times = [s.get('execution_time', 0) for s in self.lineage_data['stages'] 
                          if s.get('execution_time') is not None]
        
        if execution_times:
            report['performance_summary'] = {
                'total_execution_time': sum(execution_times),
                'average_stage_time': sum(execution_times) / len(execution_times),
                'slowest_stage_time': max(execution_times),
                'fastest_stage_time': min(execution_times)
            }
        
        return report
    
    def create_data_contract(self, dataset_id: str) -> Dict[str, Any]:
        """Create a data contract for a dataset."""
        if dataset_id not in self.lineage_data['datasets']:
            raise ValueError(f"Dataset {dataset_id} not found in lineage")
        
        dataset_info = self.lineage_data['datasets'][dataset_id]
        
        contract = {
            'dataset_id': dataset_id,
            'version': '1.0.0',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'schema': {
                'columns': dataset_info.get('columns', []),
                'dtypes': dataset_info.get('dtypes', {}),
                'shape': dataset_info.get('shape', [])
            },
            'quality_requirements': {
                'completeness': 0.95,
                'validity': 0.98,
                'uniqueness': 0.99,
                'consistency': 0.95
            },
            'metadata': {
                'description': f"Data contract for {dataset_info['name']}",
                'stage': dataset_info['stage'],
                'lineage_experiment': self.experiment_id
            }
        }
        
        # Add quality metrics if available
        if 'quality_metrics' in dataset_info:
            contract['current_quality'] = dataset_info['quality_metrics']
        
        return contract


class DataProvenanceManager:
    """Manager for data provenance across the entire pipeline ecosystem."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.trackers = {}
        self.ecosystem_metadata = {
            'ecosystem_name': 'Privacy Case Classifier (PCC)',
            'pipelines': {
                'data_pipeline': {'status': 'active', 'version': '1.0.0'},
                'training_pipeline': {'status': 'pending', 'version': '1.0.0'},
                'inference_pipeline': {'status': 'pending', 'version': '1.0.0'}
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def create_pipeline_tracker(self, pipeline_name: str) -> DataLineageTracker:
        """Create a new lineage tracker for a pipeline."""
        pipeline_config = self.config.copy()
        pipeline_config['pipeline_name'] = pipeline_name
        
        tracker = DataLineageTracker(pipeline_config)
        self.trackers[pipeline_name] = tracker
        
        return tracker
    
    def get_ecosystem_summary(self) -> Dict[str, Any]:
        """Get summary of the entire ecosystem."""
        summary = self.ecosystem_metadata.copy()
        summary['trackers'] = {}
        
        for pipeline_name, tracker in self.trackers.items():
            summary['trackers'][pipeline_name] = {
                'experiment_id': tracker.experiment_id,
                'stages_count': len(tracker.lineage_data['stages']),
                'datasets_count': len(tracker.lineage_data['datasets']),
                'last_updated': max([s.get('timestamp', '') for s in tracker.lineage_data['stages']] + [''])
            }
        
        return summary
    
    def export_ecosystem_lineage(self) -> Dict[str, Any]:
        """Export lineage for the entire ecosystem."""
        ecosystem_lineage = {
            'ecosystem_metadata': self.ecosystem_metadata,
            'pipelines': {}
        }
        
        for pipeline_name, tracker in self.trackers.items():
            ecosystem_lineage['pipelines'][pipeline_name] = tracker.export_lineage_graph()
        
        return ecosystem_lineage


def create_lineage_tracker(config: Optional[Dict[str, Any]] = None) -> DataLineageTracker:
    """Convenience function to create a lineage tracker."""
    return DataLineageTracker(config)


def track_pipeline_execution(tracker: DataLineageTracker, 
                           pipeline_func,
                           stage_name: str,
                           inputs: Dict[str, Any],
                           parameters: Dict[str, Any] = None) -> Any:
    """Decorator-like function to track pipeline stage execution."""
    import time
    
    # Track input datasets
    input_ids = []
    for name, data in inputs.items():
        dataset_id = tracker.track_dataset(data, name, stage_name)
        input_ids.append(dataset_id)
    
    # Start stage tracking
    stage_id = tracker.track_stage(stage_name, input_ids, [], parameters)
    start_time = time.time()
    
    try:
        # Execute pipeline function
        result = pipeline_func(**inputs)
        
        # Track output datasets
        output_ids = []
        if isinstance(result, (pd.DataFrame, np.ndarray)):
            output_id = tracker.track_dataset(result, f"{stage_name}_output", stage_name)
            output_ids.append(output_id)
        elif isinstance(result, dict):
            for name, data in result.items():
                if isinstance(data, (pd.DataFrame, np.ndarray)):
                    output_id = tracker.track_dataset(data, name, stage_name)
                    output_ids.append(output_id)
        
        # Update stage with outputs
        execution_time = time.time() - start_time
        tracker.update_stage_completion(stage_id, execution_time, 'completed')
        
        # Update stage outputs
        for stage in tracker.lineage_data['stages']:
            if stage['stage_id'] == stage_id:
                stage['outputs'] = output_ids
                break
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        tracker.update_stage_completion(stage_id, execution_time, 'failed', str(e))
        raise 