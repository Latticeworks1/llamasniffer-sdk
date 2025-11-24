# LlamaSniffer Codebase Restructuring Plan

## Current State Analysis

### What the code ACTUALLY does (vs. what names suggest):

| Current File | Claimed Purpose | Actual Functionality |
|--------------|-----------------|---------------------|
| `core.py` | "Core" functionality | - Network port scanner<br>- Shodan API client<br>- Basic HTTP client for Ollama<br>- Embedding-based model matcher<br>- Multi-instance load balancer |
| `ollama.py` | Ollama compatibility | - Config file management<br>- API key detection<br>- Ollama API wrapper<br>- HuggingFace backup functions<br>- Synthetic data generation |
| `shepherd.py` | "Shepherding" tasks | Async task queue + worker pool |
| `dataset_forge.py` | Dataset creation | Distributed synthetic dataset generator |
| `schema_compiler.py` | Schema compilation | YAML/JSON schema parser for datasets |
| `hf_uploader.py` | HF uploads | HuggingFace dataset upload client |

### Core Issues:

1. **Naming confusion**: "sniffer", "shepherd", "flock", "herder" are metaphors that obscure actual function
2. **Duplicate classes**: Two `OllamaClient` classes (one in core.py, one in ollama.py)
3. **God objects**: `core.py` (1040 lines) and `ollama.py` (1529 lines) do too much
4. **Poor separation**: Discovery, client, routing, and queueing mixed together
5. **Not PyPI standard**: Flat structure instead of organized subpackages

## Proposed PyPI-Standard Structure

```
llamasniffer/
├── __init__.py           # Public API exports
├── _version.py           # Version info (keep as-is)
├── config.py             # Configuration management
├── cli.py                # CLI entry point
│
├── discovery/            # Instance discovery subsystem
│   ├── __init__.py
│   ├── scanner.py        # Network port scanning
│   └── shodan_search.py  # Shodan API integration
│
├── client/               # HTTP client subsystem
│   ├── __init__.py
│   ├── http.py           # Basic Ollama HTTP client
│   └── distributed.py    # Multi-instance load-balanced client
│
├── routing/              # Request routing subsystem
│   ├── __init__.py
│   ├── semantic.py       # Embedding-based model resolution
│   └── strategies.py     # Load balancing strategies
│
├── tasks/                # Async task management
│   ├── __init__.py
│   ├── queue.py          # Priority task queue
│   └── worker_pool.py    # Worker pool executor
│
├── datasets/             # Synthetic dataset generation
│   ├── __init__.py
│   ├── generator.py      # Dataset generator
│   └── schema.py         # Schema compiler
│
└── integrations/         # External service integrations
    ├── __init__.py
    └── huggingface.py    # HuggingFace upload client
```

## Detailed Refactoring Map

### 1. `config.py` (NEW)
**Source**: Extract from `ollama.py` lines 25-254
- `OllamaConfig` class
- `_find_config_file()`
- `_load_config()`
- `_find_shodan_key()`
- `_find_huggingface_token()`
- `_default_config()`

### 2. `discovery/scanner.py` (NEW)
**Source**: Extract from `core.py` lines 21-176
- `NetworkScanner` class (rename from `OllamaDiscovery`)
- `scan_port_range()`
- `_verify_ollama_instance()`
- `scan_network()`
- `get_models()`

### 3. `discovery/shodan_search.py` (NEW)
**Source**: Extract from `core.py` lines 178-233
- `ShodanDiscovery` class
- `scan_shodan()`
- Helper: `discover_ollama_shodan()` function

### 4. `client/http.py` (NEW)
**Source**: Extract from `core.py` lines 304-378
- `OllamaHTTPClient` class (rename from first `OllamaClient`)
- `generate()`
- `list_models()`
- `pull_model()`

### 5. `client/distributed.py` (NEW)
**Source**: Extract from `core.py` lines 576-905
- `DistributedClient` class (rename from `DistributedOllamaManager`)
- `_get_available_instances_for_model()`
- `_select_optimal_instance()`
- `_update_instance_stats()`
- `_health_check_instance()`
- `generate_distributed()`
- `get_cluster_status()`

### 6. `routing/semantic.py` (NEW)
**Source**: Extract from `core.py` lines 380-574
- `SemanticModelMatcher` class (keep name)
- `_get_embedding()`
- `_cosine_similarity()`
- `_extract_model_features()`
- `find_best_model()`
- `explain_model_choice()`

### 7. `routing/strategies.py` (NEW)
**Source**: Extract from `core.py` lines 662-678
- `LoadBalancingStrategy` enum
- `select_instance_round_robin()`
- `select_instance_fastest()`
- `select_instance_least_loaded()`

### 8. `tasks/queue.py` (NEW)
**Source**: Extract from `shepherd.py` lines 22-60, 69-100
- `TaskPriority` enum (keep)
- `TaskStatus` enum (keep)
- `Task` dataclass (keep)
- `PriorityTaskQueue` class (new)

### 9. `tasks/worker_pool.py` (NEW)
**Source**: Extract from `shepherd.py` lines 62-360
- `ParallelTaskQueue` class (successor to `FlockShepherd`)
- `start_workers()`
- `stop_workers()`
- `submit_task()`
- `submit_batch()`
- `await_task()`
- `await_batch()`

### 10. `datasets/generator.py` (NEW)
**Source**: Extract from `dataset_forge.py`
- `DatasetGenerator` class (rename from `DatasetForge`)
- `DatasetConfig` dataclass (keep)
- `DatasetType` enum (keep)
- `QualityLevel` enum (keep)
- `DataPoint` dataclass (keep)

### 11. `datasets/schema.py` (NEW)
**Source**: Extract from `schema_compiler.py`
- `SchemaCompiler` class (keep)
- `FieldType` enum (keep)
- `ValidationRule` enum (keep)
- `FieldSchema` dataclass (keep)
- `DatasetSchema` dataclass (keep)

### 12. `integrations/huggingface.py` (NEW)
**Source**: Extract from `hf_uploader.py` + `ollama.py` lines 1281-1485
- `HuggingFaceClient` class (rename from `HuggingFaceUploader`)
- `backup_dataset()`
- `setup_auth()`
- Functions from ollama.py: `setup_hf_auth()`, `backup_flock_data()`, etc.

### 13. `cli.py` (NEW)
**Source**: Extract from `core.py` lines 968-1039
- `main()` function
- CLI argument parsing
- Keep as separate file for clean entry point

### 14. `__init__.py` (REFACTOR)
**Source**: Current `__init__.py`
- Import from new structure
- Maintain backward compatibility
- Export clean public API

## Backward Compatibility Layer

Create compatibility aliases in `__init__.py`:

```python
# Backward compatibility
from .client.distributed import DistributedClient as DistributedOllamaManager
from .discovery.scanner import NetworkScanner as OllamaDiscovery
from .client.http import OllamaHTTPClient as OllamaClient
from .tasks import ParallelTaskQueue
from .datasets.generator import DatasetGenerator as DatasetForge
```

## Migration Steps

1. Create new directory structure
2. Extract and refactor code into new files
3. Update imports in new files
4. Create backward compatibility layer in `__init__.py`
5. Update tests to use new structure
6. Update documentation
7. Run full test suite
8. Remove old files after confirming tests pass

## Benefits

- **Clear separation of concerns**: Each module has one responsibility
- **Standard PyPI structure**: Follows Python packaging best practices
- **Better discoverability**: Clear module names explain functionality
- **Easier testing**: Can test individual components in isolation
- **Maintainability**: Smaller files are easier to understand and modify
- **Backward compatible**: Old imports still work via compatibility layer
