"""
LlamaSniffer - Discover and interact with Ollama instances locally and globally.

A comprehensive Python SDK for discovering Ollama LLM instances across local networks
and internet-facing services, with advanced distributed inference capabilities and
semantic model matching.
"""

from ._version import __version__, __version_info__

__author__ = "Latticeworks1"
__email__ = "dev@latticeworks1.com"

from .core import (
    # Main classes
    RemoteDiscovery,
    _OllamaClient as BaseOllamaClient,
    DistributedOllamaManager,
    SemanticModelMatcher,
    # Convenience functions
    discover_remote_instances,
    connect_to_ollama,
    create_distributed_manager,
)

# Import ollama interface
from .ollama import (
    ollama,
    chat,
    achat,
    generate,
    agenerate,
    list as list_models,
    show,
    ps,
    configure,
    inspect_resolution,
    auto_discover_instances,
    get_flock_status,
    OllamaConfig,
)

# Import distributed task queue
from .tasks import ParallelTaskQueue, TaskPriority, TaskStatus, Task

# Import dataset creation
from .dataset_forge import (
    DatasetForge,
    DatasetConfig,
    DatasetType,
    QualityLevel,
    create_synthetic_dataset,
)

# Import HF integration
from .hf_uploader import (
    HuggingFaceUploader,
)

# Import endpoint caching
from .endpoint_cache import (
    EndpointCache,
    get_cached_endpoints,
    cache_endpoints,
    get_cache_info,
    clear_cache,
)

__all__ = [
    # Core Classes
    "RemoteDiscovery",
    "BaseOllamaClient",
    "DistributedOllamaManager",
    "SemanticModelMatcher",
    # Core Functions
    "discover_remote_instances",
    "connect_to_ollama",
    "create_distributed_manager",
    # Ollama Interface
    "ollama",
    "chat",
    "achat",
    "generate",
    "agenerate",
    "list_models",
    "show",
    "ps",
    "configure",
    "inspect_resolution",
    "auto_discover_instances",
    "get_flock_status",
    "OllamaConfig",
    # Distributed Task Queue
    "ParallelTaskQueue",
    "TaskPriority",
    "TaskStatus",
    "Task",
    # Dataset Creation
    "DatasetForge",
    "DatasetConfig",
    "DatasetType", 
    "QualityLevel",
    "create_synthetic_dataset",
    # HF Integration
    "HuggingFaceUploader",
    # Endpoint Caching
    "EndpointCache",
    "get_cached_endpoints",
    "cache_endpoints",
    "get_cache_info",
    "clear_cache",
    # Metadata
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
]
