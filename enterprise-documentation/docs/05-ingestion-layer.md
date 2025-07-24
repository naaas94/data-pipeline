### Ingestion Layer

The ingestion layer is a critical component of the enterprise data pipeline, responsible for efficiently and reliably bringing data into the system from various sources. This layer ensures that data is ready for processing, validation, and analysis.

#### Input Sources

The pipeline supports multiple input sources, each catering to different data ingestion needs:

1. **Files**: 
   - **CSV/Parquet**: These are common file formats used for data storage and exchange. The pipeline can ingest data from these files using the `load_data` method, which reads the files into a DataFrame for further processing.
   - **Example**:
     ```python
     df = pipeline.load_data(source_type='csv', file_path='data/input.csv')
     ```

2. **APIs**:
   - The pipeline can connect to external APIs to fetch data. This is useful for real-time data ingestion or when integrating with third-party services.
   - **Example**:
     ```python
     df = pipeline.load_data(source_type='api', endpoint='https://api.example.com/data')
     ```

3. **Streaming**:
   - **Apache Kafka**: Used for real-time data streaming, allowing the pipeline to process data as it arrives.
   - **Example**:
     ```python
     df = pipeline.load_data(source_type='kafka', topic='data-stream')
     ```

4. **Databases**:
   - **BigQuery**: The pipeline can ingest data directly from BigQuery, leveraging its scalability and performance.
   - **Example**:
     ```python
     df = pipeline.load_data(source_type='bigquery', query='SELECT * FROM dataset.table')
     ```

#### Validation and Preprocessing

Before data is processed, it undergoes validation and preprocessing to ensure quality and consistency:

1. **Validation Rules**:
   - **Schema Validation**: Using tools like Pandera and Pydantic, the pipeline validates the data schema to ensure it meets the expected structure and types.
   - **Quality Checks**: Great Expectations is used to define and enforce data quality rules, such as checking for missing values or ensuring data falls within expected ranges.
   - **Example**:
     ```python
     validation_results = pipeline.validate_data(df)
     ```

2. **Preprocessing Steps**:
   - **Data Cleaning**: This involves removing duplicates, handling missing values, and correcting data types.
   - **Feature Extraction**: The pipeline extracts relevant features from raw data, which are essential for downstream analysis and machine learning tasks.
   - **Transformation**: Data is transformed into a suitable format for processing, which may include normalization, encoding categorical variables, and aggregating data.
   - **Example**:
     ```python
     processed_df = pipeline.process_data(df)
     ```

 