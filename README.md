# LlamaSniffer

**Distributed Semantic Routing for Ollama — Global Model Discovery and Inference Coordination**

LlamaSniffer is the unified control layer for the distributed Ollama ecosystem.  
It discovers, ranks, and routes to active Ollama-compatible instances worldwide using verified discovery channels and semantic intent matching.  
Every request is intelligently directed to the optimal node, selected by capability, latency, and contextual fit.

⸻

## Overview

LlamaSniffer connects inference infrastructure across geographic and organizational boundaries.  
It provides global visibility, semantic routing, and parallel coordination for AI workloads using the Ollama protocol.  
Each endpoint operates as part of a continuously optimized network of reasoning systems — collectively known as **the flock**.

### Core capabilities
• **Global Discovery** – Integration with verified discovery sources and Shodan indexing to locate active Ollama-compatible nodes  
• **Semantic Model Matching** – Natural-language model selection driven by embeddings and confidence metrics  
• **Distributed Inference** – Load-balanced or parallel execution across multiple nodes with automatic health-based routing  
• **Zero-Code Integration** – Identical interface to the Ollama Python client for immediate compatibility  
• **Flock Intelligence** – Real-time tracking of connected endpoints and global inference availability  
• **Observability and Control** – Endpoint-level metrics, semantic resolution history, and adaptive routing transparency

⸻

## Installation

```bash
uv add llamasniffer
```

or

```bash
pip install llamasniffer
```

⸻

## Drop-In Replacement for Ollama

LlamaSniffer provides a **zero-code replacement** for the standard Ollama Python client. Just swap the import:

```python
# Before: Standard Ollama
# from ollama import Client
# client = Client(host='http://localhost:11434')

# After: LlamaSniffer (distributed + semantic routing)
from llamasniffer import ollama

# Same API, global reach
response = ollama.chat(
    model='llama3',  # Or use semantic: 'reasoning', 'coding', etc.
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### Using with OpenAI-Style Code

Since LlamaSniffer is Ollama-compatible, you can use it anywhere that accepts Ollama:

```python
# Works with LiteLLM, LangChain, or any Ollama-compatible library
from llamasniffer import ollama

# Configure once for distributed discovery
ollama.configure(shodan_api_key='your-key')

# Now all ollama.* calls route through the global flock
response = ollama.chat(model='llama3', messages=[...])
```

### Environment Variable Configuration

Set up LlamaSniffer to auto-configure:

```bash
export SHODAN_API_KEY="your-shodan-key"
export HF_TOKEN="your-huggingface-token"  # Optional, for dataset uploads
```

Then use without explicit configuration:

```python
from llamasniffer import ollama

# Auto-discovers instances via Shodan
response = ollama.generate(model='general', prompt='Hello world')
```

⸻

## Quick Start

```python
from llamasniffer import ollama

# Configure with your Shodan API key or verified discovery index
ollama.configure(shodan_api_key='your-shodan-api-key')

# Query with intent; LlamaSniffer handles model resolution and routing
response = ollama.chat(
    model='reasoning',
    messages=[{'role': 'user', 'content': 'Explain gravitational lensing in precise terms.'}]
)

print(response['message']['content'])

resolution = ollama.inspect_resolution()
print(f"Model used: {resolution['last_resolution']['model']}")
print(f"Confidence: {resolution['last_resolution']['confidence']:.3f}")
```

⸻

## Distributed Inference

```python
from llamasniffer import discover_remote_instances, create_distributed_manager

instances = discover_remote_instances('your_shodan_key', limit=5)

manager = create_distributed_manager(instances, strategy='least_loaded')

response = manager.generate_distributed(
    model='general',
    prompt='Summarize the economic policies of 1930s America.',
    parallel_requests=3
)

print(response['response'])
```

Each inference call evaluates latency, model availability, and reliability before selecting the ideal node or ensemble within the flock.

⸻

## Semantic Routing

LlamaSniffer interprets intent and dynamically resolves the most appropriate model across the global flock.  
Specify purpose — reasoning, creative writing, analysis, or summarization — and LlamaSniffer determines the optimal execution route.

```python
from llamasniffer import ollama

response = ollama.generate(
    model='creative',
    prompt='Write a short speculative fiction piece about Mars terraforming.'
)
print(response['response'])
```

⸻

## Flock Monitoring and Control

```python
from llamasniffer import ollama

flock = ollama.get_flock_status()
print(flock['flock_health'])

nodes = ollama.get_flock_nodes()
print(f"Active nodes in flock: {len(nodes['active_nodes'])}")

ollama.refresh_flock_health()
```

LlamaSniffer continuously monitors its flock — the collection of active, verified endpoints participating in distributed inference.  
It tracks uptime, latency, and semantic match accuracy across the global network.

⸻

## Complete Ollama API Compatibility

LlamaSniffer implements the full Ollama Python client API. Use it as a **drop-in replacement** in existing code:

```python
# Replace this:
# import ollama

# With this:
from llamasniffer import ollama

# Everything else stays the same!
```

All Ollama methods are supported with global reach and distributed intelligence:

```python
from llamasniffer import ollama

# Configure once with discovery settings
ollama.configure(shodan_api_key='your-key')

# Chat completions with semantic routing
response = ollama.chat(
    model='coding',  # Semantic model matching
    messages=[{'role': 'user', 'content': 'Write a Python function'}],
    stream=True  # Streaming support
)

# Text generation
response = ollama.generate(model='creative', prompt='Write a story')

# Model management across the flock
models = ollama.list()  # List across all nodes
ollama.pull('general')   # Pull to distributed nodes
ollama.delete('old-model')  # Remove from nodes

# Embeddings with intelligent routing
embeddings = ollama.embed('fast', ['text1', 'text2'])

# Custom client with authentication
client = ollama.Client(
    host='http://discovered-node:11434',
    headers={'Authorization': 'Bearer token'}
)

# Async support
async def chat():
    response = await ollama.achat(model='reasoning', messages=[...])
    async for chunk in await ollama.achat(model='creative', messages=[...], stream=True):
        print(chunk['message']['content'], end='')
```

**Supported Methods:**
• `chat()` • `generate()` • `embed()` • `list()` • `pull()` • `delete()` • `create()` • `copy()` • `push()` • `show()`

Same syntax. Global scale. Coordinated intelligence.

⸻

## Advanced Flock Management

### Node Discovery and Health

```python
# Monitor your flock nodes
nodes = ollama.get_flock_nodes()
print(f"Total nodes: {nodes['total_nodes']}")
print(f"Discovery methods: {nodes['discovery_summary']}")

# Inspect individual nodes
for node in nodes['nodes'][:3]:
    print(f"Node: {node['host']}:{node['port']}")
    print(f"  Health: {'✓' if node.get('is_healthy', True) else '✗'}")
    print(f"  Models: {len(node['models'])}")
    print(f"  Discovery: {node['discovery_method']}")
    print(f"  Response Time: {node['response_time_ms']}ms")
```

### Flock Intelligence Analytics

```python
# Get comprehensive flock status
status = ollama.get_flock_status()
print(f"Flock Health: {status['flock_health']['health_percentage']}%")
print(f"Available Models: {status['model_availability']['unique_models']}")
print(f"Geographic Distribution: {status['performance_stats']}")

# Force health refresh across the flock
refresh = ollama.refresh_flock_health()
print(f"Refreshed nodes: {refresh['refreshed_nodes']}")
```

⸻

## Parallel Task Queue

```python
import asyncio
from llamasniffer import ParallelTaskQueue, TaskPriority

async def run_batch():
    queue = ParallelTaskQueue(max_workers=20, strategy="fastest")
    await queue.start()

    tasks = [
        {'model': 'coding', 'messages': [{'role': 'user', 'content': 'Write Python code'}]},
        {'model': 'analysis', 'messages': [{'role': 'user', 'content': 'Summarize this document'}]},
        {'model': 'creative', 'messages': [{'role': 'user', 'content': 'Tell a short story'}]},
    ]

    task_ids = await queue.submit_batch(tasks, priority=TaskPriority.HIGH)
    results = await queue.wait_for_batch(task_ids)

    stats = queue.get_stats()
    print(f"Completed: {stats['tasks']['completed']}/{stats['tasks']['submitted']}")
    print(f"Success rate: {stats['tasks']['success_rate']}%")
    print(f"Avg latency: {stats['performance']['avg_latency']:.2f}s")
    print(f"Active workers: {stats['workers']['active']}/{stats['workers']['configured']}")

    await queue.stop()
    return results

asyncio.run(run_batch())
```

- Priority-aware scheduling (CRITICAL → LOW) with automatic retries
- Async worker pool backed by `DistributedOllamaManager` load balancing
- Dataset helper via `process_dataset()` for 100k+ record fan-out
- Real-time stats with throughput, latency, and queue depth
- Legacy FlockShepherd/FlockHerder APIs have been removed in favor of this queue

⸻

## Synthetic Dataset Generation

Generate high-quality synthetic datasets using your distributed flock:

```python
import asyncio
from llamasniffer import DatasetForge, DatasetConfig, DatasetType, QualityLevel

async def generate_dataset():
    config = DatasetConfig(
        dataset_type=DatasetType.REASONING,
        target_size=1000,
        quality_level=QualityLevel.HIGH,
        models=['llama3', 'deepseek-r1', 'qwen'],
        batch_size=50,
        require_consensus=True,  # Multiple models must agree
        deduplicate=True
    )

    forge = DatasetForge(config)
    dataset = await forge.forge_dataset()

    print(f"Generated {len(dataset['data'])} samples")
    print(f"Quality score: {dataset['metadata']['quality_score']}")

    return dataset

asyncio.run(generate_dataset())
```

### Available Dataset Types

- **QA_PAIRS** - Question and answer pairs
- **CONVERSATIONS** - Multi-turn dialogues
- **INSTRUCTIONS** - Task instructions and responses
- **CODE_COMPLETION** - Code generation examples
- **REASONING** - Step-by-step problem solving
- **CLASSIFICATION** - Labeled classification data
- **SUMMARIZATION** - Text summarization pairs
- **TRANSLATION** - Multi-language translation

### Quality Levels

- **BASIC** - Fast generation, minimal validation
- **STANDARD** - Balanced quality and speed
- **HIGH** - Multiple model consensus, strict validation
- **PREMIUM** - Maximum quality, extensive validation

### CLI Tool

```bash
# Generate from YAML config
python -m llamasniffer.dataset_cli generate config.yaml

# Create example configurations
python -m llamasniffer.dataset_cli examples

# With HuggingFace auto-upload
python -m llamasniffer.dataset_cli generate config.yaml --upload
```

See `config_examples/` for ready-to-use templates.

### HuggingFace Integration

Auto-upload datasets to HuggingFace Hub:

```python
from llamasniffer import HuggingFaceUploader

uploader = HuggingFaceUploader(
    username="your-hf-username",
    token="your-hf-token"  # Auto-detected if configured
)

uploader.upload_dataset(
    dataset=dataset,
    repo_name="my-reasoning-dataset",
    private=False
)
```

⸻

## Semantic Model Resolution

### Natural Language Model Selection

```python
# Use intent-based model selection
queries = [
    'coding assistant',
    'creative writing', 
    'mathematical reasoning',
    'fast small model',
    'multilingual support'
]

for query in queries:
    response = ollama.chat(
        model=query,
        messages=[{'role': 'user', 'content': f'Help with {query}'}]
    )
    
    resolution = ollama.inspect_resolution()
    print(f"'{query}' → {resolution['last_resolution']['model']}")
```

### Configuration and Tuning

```python
# Fine-tune semantic matching behavior
ollama.configure(
    semantic_enabled=True,
    semantic_threshold=0.4,
    strategy="fastest",
    verbose=True,  # See model resolution decisions
    shodan_api_key='your-key'  # For global remote discovery
)
```

⸻

## Requirements for Semantic Model Matching

The semantic model matching feature requires a local embedding model running via LM Studio:

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai)
2. **Load Embedding Model**: Download and load `text-embedding-nomic-embed-text-v1.5`
3. **Start Local Server**: Run the model on `http://127.0.0.1:1234/v1/embeddings`

The system gracefully falls back to exact name matching if the embedding model is unavailable.

⸻

## Development

```bash
uv sync
uv run pytest
```

⸻

## Philosophy

LlamaSniffer builds an autonomous inference mesh where every node contributes to collective intelligence.  
The flock operates as a distributed organism — aware, adaptive, and constantly optimizing.  
Every request joins the herd, every endpoint adds strength, and the system itself learns where intelligence resides.
