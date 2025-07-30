# Local Vector Store for Daily Conversations

A simple, local vector store implementation for storing and searching daily conversations. No external dependencies or cloud services required.

## Features

- ✅ **Local Storage**: All data stored locally, no cloud dependencies
- ✅ **Daily Conversations**: Store up to 50 conversations per day
- ✅ **Similarity Search**: Find similar conversations using embeddings
- ✅ **Automatic Cleanup**: Remove old conversations (configurable retention)
- ✅ **Simple API**: Easy-to-use interface for adding and searching
- ✅ **Statistics**: Track daily conversation counts and store metrics

## Quick Start

### 1. Generate Daily Conversations

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --generate
```

### 2. Search Similar Conversations

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --search "I need help with my account"
```

### 3. View Statistics

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --stats
```

### 4. Run Demo

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --demo
```

## Configuration

The `config_daily_conversations.yaml` file controls:

- **Conversations per day**: Default 50
- **Vector store path**: Where data is stored
- **Cleanup retention**: How many days to keep conversations
- **Search results**: Number of similar conversations to return
- **Embedding model**: Which model to use for embeddings

## API Usage

### Basic Vector Store Operations

```python
from src.utils.vector_store import LocalVectorStore

# Initialize vector store
store = LocalVectorStore("my_vector_store")

# Add conversations
conversations = [
    {
        'id': 'conv_001',
        'text': 'I need help with my account',
        'timestamp': '2025-07-30T10:00:00',
        'user_id': 'user_1234'
    }
]
embeddings = generate_embeddings([conv['text'] for conv in conversations])
store.add_conversations(conversations, embeddings)

# Search similar conversations
query_embedding = generate_embeddings(['I need help'])
results = store.search_similar(query_embedding[0], top_k=5)

# Get statistics
stats = store.get_daily_stats()
info = store.get_store_info()
```

### Daily Pipeline Usage

```python
from src.daily_conversations_pipeline import DailyConversationsPipeline

# Initialize pipeline
pipeline = DailyConversationsPipeline("config_daily_conversations.yaml")

# Generate daily conversations
conversations = pipeline.generate_daily_conversations(50)

# Search for similar conversations
results = pipeline.search_similar_conversations("How do I delete my data?")

# Get daily statistics
stats = pipeline.get_daily_stats()
```

## File Structure

```
vector_store/
├── metadata.json          # Store metadata and statistics
└── embeddings.pkl         # Embeddings and conversation data
```

## Data Format

### Conversation Object
```json
{
    "id": "conv_001",
    "text": "I need help with my account",
    "timestamp": "2025-07-30T10:00:00",
    "date": "2025-07-30",
    "user_id": "user_1234",
    "session_id": "session_56789"
}
```

### Search Result
```json
{
    "id": "conv_001",
    "text": "I need help with my account",
    "timestamp": "2025-07-30T10:00:00",
    "user_id": "user_1234",
    "similarity_score": 0.95
}
```

## Testing

Run the test script to verify functionality:

```bash
python test_vector_store.py
```

## Benefits

- **No External Dependencies**: Everything runs locally
- **Cost Effective**: No cloud service fees
- **Privacy**: All data stays on your infrastructure
- **Simple**: Easy to understand and modify
- **Scalable**: Can handle daily conversation volumes
- **Reliable**: No network dependencies or API limits

## Use Cases

- **Customer Support**: Store and search support conversations
- **FAQ Generation**: Find similar questions and answers
- **Training Data**: Build datasets for ML models
- **Analytics**: Track conversation patterns and trends
- **Compliance**: Maintain conversation logs for regulatory requirements

## Maintenance

### Cleanup Old Conversations

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --cleanup
```

### Monitor Storage

```bash
python src/daily_conversations_pipeline.py --config config_daily_conversations.yaml --info
```

## Integration

The vector store can be easily integrated into existing systems:

- **Web Applications**: Add conversation storage to chat interfaces
- **Data Pipelines**: Include in ETL processes
- **Analytics Platforms**: Feed conversation data to BI tools
- **ML Systems**: Use as training data for conversation models

## Performance

- **Storage**: ~1KB per conversation (including embeddings)
- **Search**: O(n) complexity for similarity search
- **Memory**: Loads all embeddings into memory for fast search
- **Scalability**: Suitable for daily conversation volumes (50-100 per day)

## Security

- **Local Storage**: No data leaves your infrastructure
- **No External APIs**: No risk of data exposure to third parties
- **Configurable Retention**: Control how long data is kept
- **Access Control**: Standard file system permissions apply 
noteId: "b5ce89c06d6d11f0bc81c55a457885ea"
tags: []

---

 