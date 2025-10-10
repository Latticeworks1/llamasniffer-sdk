# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-10-10

### Added
- **Complete Ollama API Compatibility**: Full drop-in replacement with zero code changes
  - All standard methods: `chat()`, `generate()`, `list()`, `show()`, `pull()`, `delete()`, `create()`, `copy()`, `push()`, `embed()`
  - Streaming support: Both sync and async streaming with `stream=True`
  - Custom client support: Headers, authentication, httpx configuration
  - Async/await support: `achat()`, `agenerate()` with async generators
- **Endpoint Management & Monitoring**: 
  - `get_cached_endpoints()`: Detailed info about discovered instances
  - `refresh_endpoint_health()`: Force health checks on failed endpoints
  - Performance tracking: Success rates, response times, load statistics
- **Enhanced Custom Client Support**:
  - Custom headers propagate through distributed infrastructure
  - Authentication support across all discovered instances
  - httpx integration for advanced HTTP configuration
- **Improved Discovery Integration**:
  - Automatic Shodan + network discovery in client initialization
  - Configurable network prefixes and discovery parameters
  - Better error handling for discovery failures

### Changed
- Discovery methods now properly integrated into client auto-discovery
- Custom headers flow through entire distributed system
- Enhanced semantic model resolution with better confidence scoring
- Improved performance tracking and health monitoring

## [0.2.0] - 2024-10-10

### Added
- **Drop-in Ollama Replacement**: Complete ollama-compatible interface for seamless integration
  - `from llamasniffer import ollama` works exactly like standard ollama
  - All standard methods: `chat()`, `generate()`, `list()`, `show()`, `ps()`
  - Async/await support: `achat()`, `agenerate()`
  - Enhanced methods: `configure()`, `inspect_resolution()`, `get_cluster_status()`
- **Semantic Model Selection**: Natural language model queries in ollama syntax
  - `ollama.chat(model='reasoning', messages=[...])`
  - Automatic model resolution with confidence scoring
  - Configurable semantic thresholds and strategies
- **Configuration Management**: Fine-grained control over semantic matching behavior
- **Inspection Tools**: Full transparency into model resolution decisions
- **Backward Compatibility**: All v0.1.0 functionality preserved

### Changed
- Enhanced distributed manager with semantic model resolution
- Improved error handling and fallback mechanisms
- Better integration between discovery and inference components

## [0.1.0] - 2024-10-10

### Added
- Initial release of LlamaSniffer
- **Network Discovery**: Multi-threaded local network scanning for Ollama instances
- **Global Discovery**: Shodan integration for internet-facing Ollama services
- **Distributed Inference**: Load balancing across multiple instances with strategies:
  - Fastest response time
  - Round robin
  - Least loaded
- **Semantic Model Matching**: Natural language model selection using embeddings
  - Integration with latterworks/ollama-embeddings model
  - Support for queries like "reasoning", "coding", "creative"
  - Confidence scoring and fallback to exact matching
- **Performance Monitoring**: Response time tracking and health checking
- **Automatic Failover**: Instance recovery and error handling
- **Parallel Execution**: Ensemble inference across multiple instances
- **Dataset Integration**: Automatic backup to Hugging Face datasets
- **CLI Interface**: Command-line tools for discovery and testing
- **Comprehensive Documentation**: Usage examples and API reference

### Features
- Conservative resource usage for Shodan API credits
- Graceful degradation when embedding models unavailable
- Extensible architecture for additional discovery methods
- Type hints throughout codebase
- Comprehensive error handling and logging