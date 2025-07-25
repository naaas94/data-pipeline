"""
Pipeline contracts for the Privacy Case Classifier (PCC) ecosystem.
Defines clear interfaces and data contracts between data, training, and inference pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import json
from pathlib import Path


class DataContract(BaseModel):
    """Base data contract for pipeline interfaces."""
    
    contract_id: str = Field(..., description="Unique identifier for the contract")
    version: str = Field(..., description="Contract version (semantic versioning)")
    pipeline_source: str = Field(..., description="Source pipeline name")
    pipeline_target: str = Field(..., description="Target pipeline name")
    created_at: datetime = Field(default_factory=datetime.now)
    description: str = Field(..., description="Description of the data contract")
    
    @field_validator('version')
    def validate_version(cls, v):
        """Validate semantic versioning format."""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(pattern, v):
            raise ValueError('Version must follow semantic versioning (e.g., 1.0.0)')
        return v


class ColumnSchema(BaseModel):
    """Schema definition for a data column."""
    
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Data type (string, int, float, datetime, etc.)")
    nullable: bool = Field(default=True, description="Whether column can contain null values")
    description: str = Field(..., description="Description of the column")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Column constraints")
    
    @field_validator('type')
    def validate_type(cls, v):
        """Validate column type."""
        valid_types = ['string', 'int', 'float', 'datetime', 'boolean', 'array', 'object']
        if v not in valid_types:
            raise ValueError(f'Type must be one of {valid_types}')
        return v


class QualityRequirements(BaseModel):
    """Data quality requirements for contract validation."""
    
    completeness: float = Field(ge=0, le=1, description="Minimum completeness ratio (0-1)")
    validity: float = Field(ge=0, le=1, description="Minimum validity ratio (0-1)")
    uniqueness: float = Field(ge=0, le=1, description="Minimum uniqueness ratio (0-1)")
    consistency: float = Field(ge=0, le=1, description="Minimum consistency ratio (0-1)")
    timeliness: Optional[int] = Field(default=None, description="Maximum age in hours")


class TrainingDataContract(DataContract):
    """Contract for training data from data pipeline to training pipeline."""
    
    schema: List[ColumnSchema] = Field(..., description="Data schema definition")
    quality_requirements: QualityRequirements = Field(..., description="Quality requirements")
    expected_size_range: Dict[str, int] = Field(..., description="Expected dataset size range")
    target_column: str = Field(..., description="Target column for ML training")
    feature_columns: List[str] = Field(..., description="Feature columns for training")
    embeddings_info: Optional[Dict[str, Any]] = Field(default=None, description="Embeddings metadata")
    
    @field_validator('expected_size_range')
    def validate_size_range(cls, v):
        """Validate size range has min and max."""
        if 'min' not in v or 'max' not in v:
            raise ValueError('Size range must have min and max keys')
        if v['min'] > v['max']:
            raise ValueError('Min size cannot be greater than max size')
        return v


class ModelContract(DataContract):
    """Contract for model artifacts from training pipeline to inference pipeline."""
    
    model_type: str = Field(..., description="Type of ML model")
    model_format: str = Field(..., description="Model serialization format")
    input_schema: List[ColumnSchema] = Field(..., description="Expected input schema")
    output_schema: List[ColumnSchema] = Field(..., description="Expected output schema")
    performance_requirements: Dict[str, float] = Field(..., description="Minimum performance metrics")
    inference_requirements: Dict[str, Any] = Field(..., description="Inference environment requirements")
    
    @field_validator('model_format')
    def validate_format(cls, v):
        """Validate model format."""
        valid_formats = ['pickle', 'joblib', 'onnx', 'tensorflow', 'pytorch', 'huggingface']
        if v not in valid_formats:
            raise ValueError(f'Model format must be one of {valid_formats}')
        return v


class InferenceDataContract(DataContract):
    """Contract for inference data from inference pipeline."""
    
    input_schema: List[ColumnSchema] = Field(..., description="Input data schema")
    output_schema: List[ColumnSchema] = Field(..., description="Output predictions schema")
    latency_requirements: Dict[str, float] = Field(..., description="Latency requirements")
    throughput_requirements: Dict[str, int] = Field(..., description="Throughput requirements")


class PCCEcosystemContracts:
    """Manager for all contracts in the PCC ecosystem."""
    
    def __init__(self, contracts_dir: str = "contracts"):
        self.contracts_dir = Path(contracts_dir)
        self.contracts_dir.mkdir(exist_ok=True)
        
        # Initialize standard contracts
        self.training_data_contract = self._create_training_data_contract()
        self.model_contract = self._create_model_contract()
        self.inference_contract = self._create_inference_contract()
    
    def _create_training_data_contract(self) -> TrainingDataContract:
        """Create the standard training data contract."""
        schema = [
            ColumnSchema(
                name="text",
                type="string",
                nullable=False,
                description="Privacy intent text content",
                constraints={"min_length": 5, "max_length": 10000}
            ),
            ColumnSchema(
                name="intent",
                type="string",
                nullable=False,
                description="Privacy intent category",
                constraints={"allowed_values": ["privacy_request", "data_deletion", "opt_out", "other"]}
            ),
            ColumnSchema(
                name="confidence",
                type="float",
                nullable=False,
                description="Confidence score for the intent",
                constraints={"min_value": 0.0, "max_value": 1.0}
            ),
            ColumnSchema(
                name="timestamp",
                type="datetime",
                nullable=False,
                description="Timestamp when the text was generated/collected"
            ),
            ColumnSchema(
                name="embeddings",
                type="array",
                nullable=True,
                description="Pre-computed text embeddings"
            )
        ]
        
        quality_requirements = QualityRequirements(
            completeness=0.99,
            validity=0.98,
            uniqueness=0.95,
            consistency=0.96,
            timeliness=24  # Data should be less than 24 hours old
        )
        
        return TrainingDataContract(
            contract_id="pcc-training-data-v1",
            version="1.0.0",
            pipeline_source="data_pipeline",
            pipeline_target="training_pipeline",
            description="Training data contract for privacy intent classification",
            schema=schema,
            quality_requirements=quality_requirements,
            expected_size_range={"min": 5000, "max": 1000000},
            target_column="intent",
            feature_columns=["text", "embeddings"],
            embeddings_info={
                "embedding_type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384
            }
        )
    
    def _create_model_contract(self) -> ModelContract:
        """Create the standard model contract."""
        input_schema = [
            ColumnSchema(
                name="text",
                type="string",
                nullable=False,
                description="Input text for classification"
            ),
            ColumnSchema(
                name="embeddings",
                type="array",
                nullable=True,
                description="Pre-computed embeddings (optional)"
            )
        ]
        
        output_schema = [
            ColumnSchema(
                name="predicted_intent",
                type="string",
                nullable=False,
                description="Predicted privacy intent"
            ),
            ColumnSchema(
                name="confidence_score",
                type="float",
                nullable=False,
                description="Prediction confidence"
            ),
            ColumnSchema(
                name="prediction_probabilities",
                type="object",
                nullable=False,
                description="Probabilities for each intent class"
            )
        ]
        
        return ModelContract(
            contract_id="pcc-model-v1",
            version="1.0.0",
            pipeline_source="training_pipeline",
            pipeline_target="inference_pipeline",
            description="Model contract for privacy intent classification",
            model_type="text_classifier",
            model_format="joblib",
            input_schema=input_schema,
            output_schema=output_schema,
            performance_requirements={
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.80,
                "f1_score": 0.80
            },
            inference_requirements={
                "max_latency_ms": 100,
                "memory_mb": 512,
                "cpu_cores": 1
            }
        )
    
    def _create_inference_contract(self) -> InferenceDataContract:
        """Create the standard inference contract."""
        input_schema = [
            ColumnSchema(
                name="text",
                type="string",
                nullable=False,
                description="Text to classify"
            ),
            ColumnSchema(
                name="request_id",
                type="string",
                nullable=True,
                description="Optional request identifier"
            )
        ]
        
        output_schema = [
            ColumnSchema(
                name="request_id",
                type="string",
                nullable=True,
                description="Request identifier (if provided)"
            ),
            ColumnSchema(
                name="predicted_intent",
                type="string",
                nullable=False,
                description="Predicted privacy intent"
            ),
            ColumnSchema(
                name="confidence_score",
                type="float",
                nullable=False,
                description="Prediction confidence"
            ),
            ColumnSchema(
                name="processing_time_ms",
                type="float",
                nullable=False,
                description="Processing time in milliseconds"
            ),
            ColumnSchema(
                name="model_version",
                type="string",
                nullable=False,
                description="Version of the model used"
            )
        ]
        
        return InferenceDataContract(
            contract_id="pcc-inference-v1",
            version="1.0.0",
            pipeline_source="inference_pipeline",
            pipeline_target="client_applications",
            description="Inference contract for privacy intent classification",
            input_schema=input_schema,
            output_schema=output_schema,
            latency_requirements={
                "p50_ms": 50,
                "p95_ms": 100,
                "p99_ms": 200
            },
            throughput_requirements={
                "requests_per_second": 100,
                "concurrent_requests": 50
            }
        )
    
    def validate_training_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate training data against contract."""
        contract = self.training_data_contract
        validation_results = {
            "contract_id": contract.contract_id,
            "validation_timestamp": datetime.now().isoformat(),
            "passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Schema validation
        required_columns = [col.name for col in contract.schema if not col.nullable]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            validation_results["passed"] = False
        
        # Size validation
        size_range = contract.expected_size_range
        if len(df) < size_range["min"]:
            validation_results["errors"].append(f"Dataset too small: {len(df)} < {size_range['min']}")
            validation_results["passed"] = False
        elif len(df) > size_range["max"]:
            validation_results["warnings"].append(f"Dataset very large: {len(df)} > {size_range['max']}")
        
        # Quality validation
        quality_req = contract.quality_requirements
        
        # Completeness check
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        validation_results["metrics"]["completeness"] = completeness
        if completeness < quality_req.completeness:
            validation_results["errors"].append(f"Completeness too low: {completeness:.3f} < {quality_req.completeness}")
            validation_results["passed"] = False
        
        # Intent validation
        if "intent" in df.columns:
            valid_intents = contract.schema[1].constraints["allowed_values"]
            invalid_intents = set(df["intent"].unique()) - set(valid_intents)
            if invalid_intents:
                validation_results["errors"].append(f"Invalid intent values: {invalid_intents}")
                validation_results["passed"] = False
        
        # Confidence validation
        if "confidence" in df.columns:
            confidence_constraints = contract.schema[2].constraints
            if df["confidence"].min() < confidence_constraints["min_value"]:
                validation_results["errors"].append("Confidence values below minimum")
                validation_results["passed"] = False
            if df["confidence"].max() > confidence_constraints["max_value"]:
                validation_results["errors"].append("Confidence values above maximum")
                validation_results["passed"] = False
        
        return validation_results
    
    def save_contracts(self):
        """Save all contracts to files."""
        contracts = {
            "training_data": self.training_data_contract,
            "model": self.model_contract,
            "inference": self.inference_contract
        }
        
        for name, contract in contracts.items():
            filepath = self.contracts_dir / f"{name}_contract.json"
            with open(filepath, 'w') as f:
                json.dump(contract.model_dump(), f, indent=2, default=str)
            print(f"Saved {name} contract to {filepath}")
    
    def load_contract(self, contract_type: str, filepath: str):
        """Load contract from file."""
        with open(filepath, 'r') as f:
            contract_data = json.load(f)
        
        if contract_type == "training_data":
            return TrainingDataContract(**contract_data)
        elif contract_type == "model":
            return ModelContract(**contract_data)
        elif contract_type == "inference":
            return InferenceDataContract(**contract_data)
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    
    def generate_contract_documentation(self) -> str:
        """Generate documentation for all contracts."""
        doc = "# PCC Ecosystem Contracts Documentation\n\n"
        doc += f"Generated on: {datetime.now().isoformat()}\n\n"
        
        contracts = [
            ("Training Data Contract", self.training_data_contract),
            ("Model Contract", self.model_contract),
            ("Inference Contract", self.inference_contract)
        ]
        
        for title, contract in contracts:
            doc += f"## {title}\n\n"
            doc += f"**Contract ID:** {contract.contract_id}\n"
            doc += f"**Version:** {contract.version}\n"
            doc += f"**Source:** {contract.pipeline_source}\n"
            doc += f"**Target:** {contract.pipeline_target}\n"
            doc += f"**Description:** {contract.description}\n\n"
            
            if hasattr(contract, 'schema'):
                doc += "### Schema\n\n"
                for col in contract.schema:
                    doc += f"- **{col.name}** ({col.type}): {col.description}\n"
                    if col.constraints:
                        doc += f"  - Constraints: {col.constraints}\n"
                doc += "\n"
            
            if hasattr(contract, 'quality_requirements'):
                quality = contract.quality_requirements
                doc += "### Quality Requirements\n\n"
                doc += f"- Completeness: {quality.completeness}\n"
                doc += f"- Validity: {quality.validity}\n"
                doc += f"- Uniqueness: {quality.uniqueness}\n"
                doc += f"- Consistency: {quality.consistency}\n"
                if quality.timeliness:
                    doc += f"- Timeliness: {quality.timeliness} hours\n"
                doc += "\n"
            
            doc += "---\n\n"
        
        return doc


def create_ecosystem_contracts(contracts_dir: str = "contracts") -> PCCEcosystemContracts:
    """Create and initialize PCC ecosystem contracts."""
    return PCCEcosystemContracts(contracts_dir)


def validate_data_contract(df: pd.DataFrame, contract: TrainingDataContract) -> Dict[str, Any]:
    """Validate DataFrame against training data contract."""
    ecosystem = PCCEcosystemContracts()
    ecosystem.training_data_contract = contract
    return ecosystem.validate_training_data(df) 