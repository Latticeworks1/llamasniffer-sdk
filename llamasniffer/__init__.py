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
    OllamaDiscovery,
    OllamaClient,
    DistributedOllamaManager,
    SemanticModelMatcher,
    # Convenience functions
    discover_ollama_instances,
    discover_ollama_shodan,
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
    get_cluster_status,
    OllamaConfig,
)

__all__ = [
    # Core Classes
    "OllamaDiscovery",
    "OllamaClient",
    "DistributedOllamaManager",
    "SemanticModelMatcher",
    # Core Functions
    "discover_ollama_instances",
    "discover_ollama_shodan",
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
    "get_cluster_status",
    "OllamaConfig",
    # Metadata
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
]
