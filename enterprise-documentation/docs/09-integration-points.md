# Integration Points

## External Services

### Local Vector Store
The pipeline includes a built-in local vector store for daily conversations. This provides similarity search capabilities without external dependencies or cloud service costs.

**Features:**
- **Local Storage**: All data stored locally in `vector_store/` directory
- **Similarity Search**: Cosine similarity with configurable top-k results
- **Daily Conversations**: Store up to 50 conversations per day
- **Automatic Cleanup**: Configurable retention policies (default: 30 days)
- **Statistics Tracking**: Daily conversation counts and store metrics

**Example:**
```python
from src.utils.vector_store import LocalVectorStore

# Initialize vector store
store = LocalVectorStore("my_vector_store")

# Add conversations with embeddings
conversations = [{"text": "I need help", "user_id": "user_123"}]
embeddings = generate_embeddings([conv["text"] for conv in conversations])
store.add_conversations(conversations, embeddings)

# Search similar conversations
query_embedding = generate_embeddings(["I need help with my account"])
results = store.search_similar(query_embedding[0], top_k=5)

# Get daily statistics
stats = store.get_daily_stats()
```

### Google Cloud BigQuery
The pipeline integrates with Google Cloud BigQuery for data storage and retrieval. This integration is enabled through the `google.cloud.bigquery` library. The pipeline can load data from BigQuery using the `_load_from_bigquery` method, which executes a SQL query and returns the results as a DataFrame. Similarly, data can be saved to BigQuery using the `_save_to_bigquery` method.

**Example:**
```python
from google.cloud import bigquery

client = bigquery.Client(project='your_project_id')
query = "SELECT * FROM your_dataset.your_table LIMIT 1000"
df = client.query(query).to_dataframe()
```

### Apache Spark
Apache Spark is used for distributed data processing. If the `pyspark` library is available, a Spark session is initialized, allowing the pipeline to leverage Spark's powerful data processing capabilities.

**Example:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataPipeline").getOrCreate()
df = spark.read.csv("path/to/csv")
```

### Ray
Ray provides an alternative for distributed processing. The pipeline initializes Ray if the library is available, enabling parallel data processing.

**Example:**
```python
import ray

ray.init()
# Use Ray for distributed processing
```

### Kafka
Kafka is used for streaming data ingestion. The pipeline can consume messages from a Kafka topic, allowing it to process real-time data streams.

**Example:**
```python
from kafka import KafkaConsumer

consumer = KafkaConsumer('your_topic', bootstrap_servers=['localhost:9092'])
for message in consumer:
    print(message.value)
```

## API Endpoints

Currently, the pipeline does not expose any API endpoints. To enable API interactions, consider using a web framework like Flask or FastAPI to define routes that correspond to pipeline operations.

**Example with Flask:**
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    # Code to run the pipeline
    return "Pipeline executed successfully"

if __name__ == '__main__':
    app.run(debug=True)
```

 