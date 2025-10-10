# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LlamaSniffer SDK is a Python library for discovering and interacting with Ollama LLM instances across local networks and globally via Shodan. The project enables automated discovery, verification, and interaction with distributed Ollama services.

## Architecture

### Core Components

- **OllamaDiscovery**: Main discovery engine supporting local network scanning and Shodan-based global discovery
- **OllamaClient**: High-level client for interacting with discovered Ollama instances
- **Dataset Integration**: Automatic backup of discoveries to Hugging Face datasets

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
# Test local discovery
python3 -c "import llamasniffer_sdk; print(llamasniffer_sdk.discover_ollama_instances())"

# Test Shodan discovery
python3 -c "import llamasniffer_sdk; print(llamasniffer_sdk.discover_ollama_shodan(query='ollama', limit=5))"

# Test client interaction
python3 -c "
import llamasniffer_sdk
client = llamasniffer_sdk.connect_to_ollama('127.0.0.1', 11434)
print(client.list_models())
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

### API Keys
- Shodan API key: Required for global discovery
- Hugging Face token: Required for dataset backup
- Store in environment variables or pass as parameters

### Default Values
- Shodan query limit: 10 (conservative credit usage)
- Network timeout: 2.0 seconds
- Local network prefix: "192.168.1"
- Dataset owner: "latterworks"

## Important Implementation Notes

### Shodan Usage
- Search by "ollama" term, not port-specific queries
- Ollama instances can run on any port, not just 11434
- Always verify instances before including in results
- Use conservative limits to preserve API credits

### Local Network Scanning
- Threaded scanning for performance
- Port range 11434-11450 for local discovery
- Verification required for all discovered services

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