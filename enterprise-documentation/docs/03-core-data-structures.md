# Core Data Structures

## Data Models

Data models are crucial for defining the structure and format of data as it flows through the pipeline. They ensure consistency and facilitate validation and processing. Here are some key components and examples:

1. **Schemas for Input Data**:
   - **Purpose**: Define the expected structure of incoming data to ensure it meets the necessary criteria before processing.
   - **Example**: Using a JSON schema to validate incoming JSON data.
     ```json
     {
       "$schema": "http://json-schema.org/draft-07/schema#",
       "type": "object",
       "properties": {
         "id": {
           "type": "integer"
         },
         "name": {
           "type": "string"
         },
         "timestamp": {
           "type": "string",
           "format": "date-time"
         }
       },
       "required": ["id", "name", "timestamp"]
     }
     ```

2. **Validation Results**:
   - **Purpose**: Store the results of data validation checks to ensure data quality and integrity.
   - **Example**: A Python class to represent validation results.
     ```python
     class ValidationResult:
         def __init__(self, is_valid, errors=None):
             self.is_valid = is_valid
             self.errors = errors or []

         def add_error(self, error):
             self.errors.append(error)

         def __str__(self):
             return f"ValidationResult(is_valid={self.is_valid}, errors={self.errors})"
     ```

3. **Processed Outputs**:
   - **Purpose**: Define the structure of data after it has been processed by the pipeline.
   - **Example**: A data class in Python to represent processed data.
     ```python
     from dataclasses import dataclass
     from datetime import datetime

     @dataclass
     class ProcessedData:
         id: int
         name: str
         processed_time: datetime
         additional_info: dict

         def to_dict(self):
             return {
                 "id": self.id,
                 "name": self.name,
                 "processed_time": self.processed_time.isoformat(),
                 "additional_info": self.additional_info
             }
     ```

## Configuration Structures

Configuration structures define how the pipeline behaves under different conditions. They are typically stored in configuration files and can be adjusted to change the pipeline's behavior without altering the code.

1. **Configuration Files**:
   - **Purpose**: Store settings and parameters that control the pipeline's operation.
   - **Example**: A YAML configuration file for a data pipeline.
     ```yaml
     pipeline:
       name: "Data Ingestion Pipeline"
       version: "1.0"
       settings:
         batch_size: 100
         retry_attempts: 3
         timeout_seconds: 30
     ```

2. **Parameters**:
   - **Purpose**: Allow dynamic adjustment of the pipeline's behavior.
   - **Example**: Python code to load and use configuration parameters.
     ```python
     import yaml

     def load_config(file_path):
         with open(file_path, 'r') as file:
             return yaml.safe_load(file)

     config = load_config('config.yaml')
     batch_size = config['pipeline']['settings']['batch_size']
     retry_attempts = config['pipeline']['settings']['retry_attempts']
     ```

 