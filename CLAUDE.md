# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LlamaSniffer is the unified control layer for the distributed Ollama ecosystem — providing distributed semantic routing for global model discovery and inference coordination. The project creates an autonomous inference mesh where every node contributes to collective intelligence, operating as a distributed organism that is aware, adaptive, and constantly optimizing.

## Architecture

### Core Components

- **OllamaDiscovery**: Global discovery engine for locating active Ollama-compatible nodes worldwide
- **DistributedOllamaManager**: Flock coordination and semantic routing intelligence  
- **OllamaClient**: Zero-code replacement for ollama client with distributed flock integration
- **ParallelTaskQueue**: Async priority task queue (replaces legacy FlockShepherd/FlockHerder)
- **Semantic Model Matching**: Natural language model selection across the flock
- **Flock Intelligence**: Real-time monitoring and adaptive routing across the autonomous mesh

### Key Features

- Multi-threaded local network scanning
- Shodan integration for global discovery (any port, not just 11434)
- Instance verification via API endpoint testing
- Hugging Face dataset backup capability
- Conservative credit usage for Shodan API calls

## Development Commands

### Setup and Dependencies
```bash
# Install with uv (preferred)
uv pip install -e .

# Install with pip
pip install -e .

# Install specific dependencies
pip install shodan datasets huggingface-hub requests
```

### Testing
```bash
# Test unified configuration
python3 -c "
import llamasniffer
config = llamasniffer.OllamaConfig()
print('Config loaded from:', config.config_path)
print('Shodan enabled:', bool(config.shodan_api_key))
"

# Test remote discovery with auto-detected keys
python3 -c "import llamasniffer; print(llamasniffer.discover_remote_instances(limit=5))"

# Test distributed client with auto-configuration
python3 -c "
import llamasniffer
# Keys auto-detected from environment/files
response = llamasniffer.chat(model='reasoning', messages=[{'role': 'user', 'content': 'test'}])
"

# Test parallel task queue
python3 -c "
import llamasniffer
import asyncio

async def test_queue():
    queue = llamasniffer.ParallelTaskQueue(max_workers=5)
    await queue.start()
    task_ids = await queue.submit_batch([
        {'model': 'llama3', 'messages': [{'role': 'user', 'content': 'What is AI?'}]},
        {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'What is ML?'}]}
    ])
    results = await queue.wait_for_batch(task_ids)
    await queue.stop()
    return results

asyncio.run(test_queue())
"
```

### Build and Distribution
```bash
# Build package
python -m build

# Install locally
pip install -e .
```

## Code Patterns and Conventions

### Discovery Pattern
- All discovery methods return standardized instance dictionaries
- Instance verification occurs before adding to results
- Conservative resource usage (credits, bandwidth)
- Threaded operations for performance

### Error Handling
- Graceful degradation on API failures
- Comprehensive exception catching in network operations
- Informative error messages without exposing internals

### API Integration
- Shodan: Search "ollama" without port restrictions (instances can run on any port)
- Hugging Face: Automatic dataset creation and updates
- Ollama API: Standard endpoint verification and interaction

## Configuration

The project uses a unified configuration file `llamasniffer_config.json` that consolidates all settings for remote-only operation.

### Configuration File Locations
1. `./llamasniffer_config.json` (current directory)
2. `~/.llamasniffer/config.json` (user home)
3. `/etc/llamasniffer/config.json` (system-wide)

### Remote-Only Architecture
- **No local network scanning** - all instances are remote
- **Shodan discovery** for global Ollama instance location
- **Manual instance configuration** for known remote endpoints
- **Distributed flock intelligence** across verified remote nodes

### API Keys (Auto-Detection)
The system automatically detects API keys from multiple sources:

**Shodan API Key** (Required for global discovery):
- Environment variables: `SHODAN_API_KEY`, `SHODAN_KEY`, `SHODAN_TOKEN`
- Files: `~/.shodan/api_key`, `~/.config/shodan/api_key`, `~/.shodan_key`, `./shodan_key.txt`

**Hugging Face Token** (Required for dataset backup):
- Environment variables: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
- Files: `~/.huggingface/token`, `~/.cache/huggingface/token`, `~/.hf_token`, `./hf_token.txt`

Keys are automatically populated on startup with clear detection messages.

### Default Values
- Shodan query limit: 10 (conservative credit usage)  
- Instance verification timeout: 5.0 seconds
- Health check interval: 60 seconds
- Dataset owner: "latterworks"

## Important Implementation Notes

### Shodan Usage
- Search by "ollama" term, not port-specific queries
- Ollama instances can run on any port, not just 11434
- Always verify instances before including in results
- Use conservative limits to preserve API credits

### Remote Instance Management
- All instances are verified before inclusion in flock
- Health checks maintain connection quality
- Automatic failover to healthy instances

### Dataset Backup
- Automatic Hugging Face dataset creation
- Timestamped commits for tracking
- Handles both new and existing repositories

## Development Guidelines

- Maintain conservative resource usage patterns
- Implement proper timeout handling for all network operations
- Use descriptive docstrings with Args/Returns sections
- Follow type hints throughout codebase
- Graceful error handling without exposed internals

## Naming Conventions

- Never name things "enhanced" or "advanced" (misspellings or overly generic terms)

## Warnings and Restrictions

- Dont use local discover
