# Architecture and Data Flow

![System Architecture Diagram](data-pipeline/enterprise-documentation/images/system_architecture.png)

## System Architecture

The diagram above illustrates the key components and data flow within the enterprise data pipeline. Each component plays a crucial role in ensuring efficient data processing and validation.

The enterprise data pipeline is designed to handle large-scale data processing with a focus on privacy intent classification. It is built with distributed processing, streaming capabilities, advanced validation, and cloud integration. The architecture consists of several key components:

1. **Distributed Processing Engines**:
   - **Apache Spark**: Used for large-scale data processing, offering SQL, MLlib, and Streaming capabilities.
   - **Apache Beam**: Provides a unified model for batch and streaming data processing.
   - **Ray**: Facilitates distributed computing for machine learning workloads and data processing.

2. **Streaming Capabilities**:
   - **Apache Kafka**: Manages real-time data streaming and event processing.
   - **Apache Pulsar**: Offers multi-tenant, high-performance messaging.
   - **Apache Flink**: Handles stateful stream processing with exactly-once semantics.

3. **Data Validation**:
   - **Pandera**: Performs statistical data validation with pandas.
   - **Great Expectations**: Monitors and validates data quality.
   - **Pydantic**: Utilizes Python type annotations for data validation.

4. **Partitioning & Cloud Integration**:
   - **BigQuery**: A serverless, highly scalable data warehouse.
   - **dbt**: Manages data transformation and modeling.
   - **Dagster**: Orchestrates data pipelines and manages workflows.

5. **Monitoring & Observability**:
   - **MLflow**: Tracks experiments, logs parameters, metrics, and artifacts.
   - **Prometheus**: Collects and stores metrics for monitoring.
   - **Grafana**: Visualizes metrics and provides dashboards for real-time monitoring.

## Data Flow

The data flow in the pipeline is structured to efficiently handle data from ingestion to output, ensuring high-quality data processing and validation:

1. **Data Ingestion**:
   - Data can be ingested from various sources such as synthetic data generation, CSV/Parquet files, Apache Kafka, and BigQuery.
   - The `load_data` method in `EnterpriseDataPipeline` class handles data loading based on the configured source type.

2. **Data Processing**:
   - The pipeline supports multiple processing engines (Spark, Ray, Pandas) to transform and process data.
   - Processing involves cleaning, feature extraction, and transformation, as seen in methods like `_process_with_spark`, `_process_with_ray`, and `_process_with_pandas`.

3. **Data Validation**:
   - Schema validation and quality checks are performed using `SchemaValidator` and `DataQualityChecker`.
   - Validation ensures data integrity and quality before further processing.

4. **Data Sampling**:
   - The pipeline includes advanced sampling techniques such as stratified sampling and class balancing.
   - The `sample_data` method applies these strategies to prepare data for training and analysis.

5. **Data Output**:
   - Processed data is saved in various formats like Parquet, CSV, or directly to BigQuery.
   - The `save_data` method manages data output, ensuring compatibility with downstream systems.

6. **Monitoring & Logging**:
   - The pipeline integrates structured logging and metrics collection using `PipelineLogger`.
   - Monitoring tools like Prometheus and Grafana provide insights into pipeline performance and data quality.

### Example Code Snippets

Here are some example code snippets to illustrate the pipeline's functionality:

- **Loading Data**:
  ```python
  df = pipeline.load_data()
  ```

- **Processing Data**:
  ```python
  processed_df = pipeline.process_data(df)
  ```

- **Validating Data**:
  ```python
  validation_results = pipeline.validate_data(processed_df)
  ```

- **Sampling Data**:
  ```python
  sampled_df = pipeline.sample_data(processed_df)
  ```

- **Saving Data**:
  ```python
  pipeline.save_data(sampled_df)
  ```

 
