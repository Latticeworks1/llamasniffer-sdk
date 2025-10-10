# LlamaSniffer

Drop-in replacement for the Ollama Python client with enhanced global discovery, distributed inference, and semantic model matching. Discover and interact with Ollama instances across local networks and globally via Shodan integration.

## Installation

```bash
uv add llamasniffer
```

## Quick Start

### Basic Single Instance Usage
```python
from llamasniffer import discover_ollama_instances, connect_to_ollama

# Discover Ollama instances on your network
instances = discover_ollama_instances()
print(f"Found {len(instances)} Ollama instances")

# Connect to the first instance found
if instances:
    client = connect_to_ollama(instances[0]['host'], instances[0]['port'])
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Generate text
    if models:
        response = client.generate(models[0], "Hello, world!")
        print(response.get('response', 'No response'))
```

### Drop-in Ollama Replacement (Recommended)
```python
# BEFORE: Standard ollama
# import ollama

# AFTER: LlamaSniffer drop-in replacement  
from llamasniffer import ollama

# Configure with Shodan API for global discovery
ollama.configure(
    shodan_api_key='your-shodan-api-key',
    network_prefix='192.168.1'  # Local network range
)

# Your existing code works unchanged! Just with semantic model selection:
response = ollama.chat(
    model='reasoning',  # Natural language model selection!
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
)
print(response['message']['content'])

# Inspect what model was actually selected
resolution = ollama.inspect_resolution()
print(f"Model used: {resolution['last_resolution']['model']}")
print(f"Confidence: {resolution['last_resolution']['confidence']:.3f}")
```

### Complete Ollama API Compatibility
```python
from llamasniffer import ollama

# Configure once with discovery settings
ollama.configure(shodan_api_key='your-key')

# All standard Ollama methods work with distributed instances:

# Chat completions
response = ollama.chat(
    model='coding',  # Semantic model matching
    messages=[{'role': 'user', 'content': 'Write a Python function'}],
    stream=True  # Streaming support
)

# Text generation
response = ollama.generate(model='creative', prompt='Write a story')

# Model management
models = ollama.list()  # List across all instances
ollama.pull('llama3')   # Pull to distributed instances
ollama.delete('old-model')  # Remove from instances

# Embeddings
embeddings = ollama.embed('fast', ['text1', 'text2'])

# Custom client with authentication
client = ollama.Client(
    host='http://discovered-instance:11434',
    headers={'Authorization': 'Bearer token'}
)

# Async support
async def chat():
    response = await ollama.achat(model='reasoning', messages=[...])
    async for chunk in await ollama.achat(model='creative', messages=[...], stream=True):
        print(chunk['message']['content'], end='')
```

### Traditional SDK Usage
```python
from llamasniffer import create_distributed_manager

# Direct access to distributed manager
manager = create_distributed_manager(strategy="fastest")
response = manager.generate_distributed("reasoning", "Explain quantum computing")
```

### Advanced Distributed Usage
```python
# Manual instance specification with global discovery
from llamasniffer import discover_ollama_shodan, create_distributed_manager

# Discover global instances via Shodan
global_instances = discover_ollama_shodan("your_shodan_key", limit=5)
local_instances = discover_ollama_instances()

# Combine local and global instances
all_instances = local_instances + global_instances

# Create distributed manager with load balancing
manager = create_distributed_manager(
    instances=all_instances,
    strategy="least_loaded"
)

# Parallel ensemble inference
response = manager.generate_distributed(
    "llama2", 
    "What is the capital of France?",
    parallel_requests=3  # Get responses from 3 instances
)

# Best response selected automatically
print(response['response'])
print(f"Parallel results: {len(response['parallel_results'])}")
```

## Features

### Discovery
- **Local Network Scanning**: Multi-threaded discovery across IP ranges
- **Global Discovery**: Shodan integration for internet-facing instances
- **Instance Verification**: API endpoint validation and performance testing
- **Model Enumeration**: Automatic model availability detection

### Distributed Inference
- **Semantic Model Matching**: Natural language model selection using latterworks/ollama-embeddings
- **Load Balancing**: Multiple strategies (fastest, round_robin, least_loaded)
- **Automatic Failover**: Health checking and instance recovery
- **Performance Monitoring**: Response time tracking and statistics
- **Parallel Execution**: Ensemble inference across multiple instances
- **Conservative Resource Usage**: Smart credit management for Shodan API

### Complete Ollama API Compatibility
- **Zero Code Changes**: Replace `import ollama` with `from llamasniffer import ollama`
- **Full API Compatibility**: All standard methods (`chat`, `generate`, `list`, `show`, `pull`, `delete`, `create`, `copy`, `push`, `embed`)
- **Streaming Support**: Both sync and async streaming with `stream=True`
- **Custom Client Support**: Headers, authentication, and httpx configuration
- **Enhanced with Semantics**: Natural language model selection with existing syntax
- **Async/Await Support**: Full `achat()`, `agenerate()` async compatibility
- **Inspection Tools**: `get_cluster_status()`, `get_cached_endpoints()`, `refresh_endpoint_health()`

### Semantic Model Selection
- **Natural Language Queries**: Use "reasoning", "coding", "creative" instead of exact model names
- **Automatic Model Resolution**: AI-powered matching using sentence transformers
- **Confidence Scoring**: See how well your query matched available models
- **Fallback Support**: Falls back to exact name matching if embeddings unavailable
- **Configurable Behavior**: Adjust semantic thresholds and resolution strategies

### Endpoint Management & Monitoring
- **Cluster Status**: Real-time health and performance monitoring via `get_cluster_status()`
- **Endpoint Inspection**: Detailed cache info with `get_cached_endpoints()`
- **Health Refresh**: Force health checks with `refresh_endpoint_health()`
- **Dataset Backup**: Automatic Hugging Face dataset integration
- **Performance Tracking**: Success rates, response times, load balancing stats

## Requirements for Semantic Model Matching

The semantic model matching feature requires a local embedding model running via LM Studio:

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai)
2. **Load Embedding Model**: Download and load `text-embedding-nomic-embed-text-v1.5`
3. **Start Local Server**: Run the model on `http://127.0.0.1:1234/v1/embeddings`

The system gracefully falls back to exact name matching if the embedding model is unavailable.

## Testing & Monitoring

### Basic Testing
```python
# Test discovery and API compatibility
from llamasniffer import ollama

# Configure with your Shodan API key
ollama.configure(shodan_api_key='your-shodan-key')

# Check discovered instances
status = ollama.get_cluster_status()
print(f"Instances: {status['cluster_health']['healthy_instances']}")

# Test inference
response = ollama.chat(model='small', messages=[{'role': 'user', 'content': 'Hi'}])
print(response['message']['content'])
```

### Endpoint Monitoring
```python
# Monitor your cached endpoints
endpoints = ollama.get_cached_endpoints()
print(f"Total endpoints: {endpoints['total_endpoints']}")
print(f"Discovery methods: {endpoints['discovery_summary']}")

# Refresh failed endpoints
refresh_result = ollama.refresh_endpoint_health()
print(f"Refreshed: {refresh_result['refreshed_endpoints']}")
```

## Development

```bash
# Install with uv
uv sync

# Run tests
uv run pytest
```