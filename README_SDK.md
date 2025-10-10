# LlamaSniffer SDK

SDK for discovering and interacting with Ollama instances on local networks.

## Installation

```bash
uv add llamasniffer-sdk
```

## Quick Start

```python
from llamasniffer_sdk import discover_ollama_instances, connect_to_ollama

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

## Features

- Network discovery of Ollama instances
- Model listing and management
- Text generation
- Threaded network scanning for speed

## Development

```bash
# Install with uv
uv sync

# Run tests
uv run pytest
```