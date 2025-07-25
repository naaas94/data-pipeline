# Configuration Management

The configuration for the PCC Data Pipeline is managed through a central `config.yaml` file. This file is loaded at the initialization of the `PCCDataPipeline` class.

## Guiding Principles
- **Centralized**: All configuration settings are stored in a single `config.yaml` file.
- **Environment-Specific**: The configuration is designed to be environment-specific, allowing for different settings in development, staging, and production environments.
- **Versioned**: The `config.yaml` file is version-controlled with Git, ensuring that changes are tracked and can be rolled back if necessary.

## Configuration Structure
The configuration files are organized in a structured manner to facilitate easy access and modification. Here is a typical structure:

- **`config.yaml`**: The main configuration file that includes all the settings for the data pipeline. It is loaded at the initialization of the `PCCDataPipeline` class.
- **Sections**:
  - `processing`: Contains settings for the processing engine and streaming options.
  - `data_source`: Specifies the source of the data, such as a CSV file, Parquet file, or a streaming platform like Kafka.
  - `output`: Defines the output format and location for the processed data.
  - `validation`: Includes settings for data quality checks and schema validation.
  - `sampling`: Contains parameters for data sampling, such as the sampling rate and strategy.

## Best Practices
- **Use descriptive names**: Use clear and descriptive names for configuration keys to avoid ambiguity.
- **Avoid hardcoding**: Do not hardcode configuration values in the code. Always use the configuration file to store settings.
- **Validate configurations**: Implement validation checks to ensure that the configuration is valid before running the pipeline.

### Tools and APIs
- **YAML Libraries**: Use libraries like PyYAML to load and parse YAML configuration files. This is already implemented in the `_load_config` method of the `PCCDataPipeline` class.
- **Logging and Monitoring**: Utilize structured logging and metrics collection to monitor configuration changes and their impact on the pipeline's performance. The `PipelineLogger` class provides structured logging and integrates with Prometheus for metrics collection.

### Example Code
Here is an example of how the configuration is loaded and used in the `PCCDataPipeline` class:

```python
class PCCDataPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
```

In this example, the `config_path` is passed to the constructor of the `PCCDataPipeline` class, and the `_load_config` method is used to load the configuration from the YAML file.

### Further Reading
- [YAML Tutorial](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-to-get-started)
- [Python `yaml` library documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Best practices for configuration management](https://www.redhat.com/en/topics/automation/what-is-configuration-management)

 