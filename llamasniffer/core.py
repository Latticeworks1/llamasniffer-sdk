"""
LlamaSniffer - Discover and interact with remote Ollama instances globally.

Provides remote discovery for Ollama LLM instances via Shodan integration.
Remote-only architecture - no local network scanning.
"""

import logging
import threading
import requests
import json
import time
import shodan
import random
import numpy as np
from typing import List, Dict, Optional, Any
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

logger = logging.getLogger(__name__)


class RemoteDiscovery:
    """Discovers remote Ollama instances globally via Shodan API.

    Remote-only architecture - discovers internet-facing Ollama instances
    on any port worldwide. No local network scanning.

    Args:
        timeout: Connection timeout for verification requests in seconds
        shodan_api_key: Shodan API key for global discovery
        hf_token: Optional Hugging Face token for dataset backups
    """

    def __init__(
        self,
        timeout: float = 5.0,
        shodan_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_ssl: bool = False,
    ):
        self.timeout = timeout
        self.discovered_instances = []
        self.shodan_api_key = shodan_api_key
        self.shodan_client = shodan.Shodan(shodan_api_key) if shodan_api_key else None
        self.hf_token = hf_token
        self.hf_api = HfApi(token=hf_token) if hf_token else None
        self.use_ssl = use_ssl

    def _verify_ollama_instance(self, host: str, port: int, use_ssl: Optional[bool] = None) -> Dict[str, Any]:
        """Fast Ollama instance verification with response time measurement.

        Optimized verification process:
        1. Quick health check via version endpoint
        2. Fast model enumeration via tags endpoint
        3. Response time measurement

        Args:
            host: Target host address
            port: Target port number
            use_ssl: Use HTTPS instead of HTTP (defaults to instance setting)

        Returns:
            Dictionary with verification results and performance metrics
        """
        protocol = "https" if (use_ssl if use_ssl is not None else self.use_ssl) else "http"
        base_url = f"{protocol}://{host}:{port}"
        verification_data = {}
        start_time = time.time()

        try:
            # Fast version check with response time
            version_start = time.time()
            version_response = requests.get(f"{base_url}/api/version", timeout=self.timeout)
            version_time = (time.time() - version_start) * 1000  # ms

            if version_response.status_code != 200:
                return {}

            version_data = version_response.json()
            verification_data["version"] = version_data.get("version", "unknown")
            verification_data["version_response_time_ms"] = round(version_time, 2)

            # Fast model enumeration
            tags_start = time.time()
            tags_response = requests.get(f"{base_url}/api/tags", timeout=self.timeout)
            tags_time = (time.time() - tags_start) * 1000  # ms

            if tags_response.status_code != 200:
                return {}

            tags_data = tags_response.json()
            models = [model["name"] for model in tags_data.get("models", [])]
            verification_data["models"] = models
            verification_data["model_count"] = len(models)
            verification_data["tags_response_time_ms"] = round(tags_time, 2)

            # Skip heartbeat test for speed - just mark as responsive if we got this far
            verification_data["responsive"] = True
            verification_data["total_verification_time_ms"] = round(
                (time.time() - start_time) * 1000, 2
            )
            verification_data["verified"] = True
            verification_data["verified_at"] = time.time()

            return verification_data

        except requests.Timeout:
            logger.debug(f"Verification timeout for {host}:{port}")
            return {}
        except requests.RequestException as e:
            logger.debug(f"Verification failed for {host}:{port}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error verifying {host}:{port}: {e}")
            return {}

    def search(self, query: str = "ollama", limit: int = 100) -> List[Dict[str, Any]]:
        """Discover internet-facing Ollama instances via Shodan search.

        Uses conservative credit consumption with single targeted query.
        Verifies discovered instances before returning results.

        Args:
            query: Search term to combine with port filter
            limit: Maximum results to return (credit conservation)

        Returns:
            List of verified Ollama instances with enriched metadata

        Raises:
            ValueError: When Shodan API key not configured
        """
        if not self.shodan_client:
            raise ValueError("Shodan API key required for global discovery")

        instances = []
        # Search for Shodan-identified Ollama product (much more accurate than text search)
        search_query = 'product:"ollama"' if query == "ollama" else query
        print(f"Searching Shodan: '{search_query}' (limit: {limit})")

        try:
            results = self.shodan_client.search(search_query, limit=limit)

            for result in results["matches"]:
                # Perform comprehensive verification
                verification = self._verify_ollama_instance(result["ip_str"], result["port"])

                if verification.get("verified", False):
                    instance = {
                        "host": result["ip_str"],
                        "port": result["port"],
                        "url": f"http://{result['ip_str']}:{result['port']}",
                        "country": result.get("location", {}).get("country_name", "Unknown"),
                        "org": result.get("org", "Unknown"),
                        "hostnames": result.get("hostnames", []),
                        "discovered_at": time.time(),
                        "discovery_method": "shodan",
                        "shodan_data": {
                            "timestamp": result.get("timestamp"),
                            "product": result.get("product"),
                            "version": result.get("version"),
                            "banner": result.get("data", "")[:200],
                        },
                        **verification,  # Include comprehensive verification data
                    }
                    instances.append(instance)

        except shodan.APIError as e:
            logger.error(f"Shodan API error: {e}")
            print(f"Shodan API error: {e}")
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            print(f"Discovery error: {e}")

        return instances

    def get_models(self, instance: Dict[str, Any]) -> List[str]:
        """Retrieve available model list from discovered Ollama instance.

        Args:
            instance: Instance dictionary from discovery results

        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{instance['url']}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except requests.Timeout:
            logger.debug(f"Timeout getting models from {instance['url']}")
        except requests.RequestException as e:
            logger.debug(f"Request failed getting models from {instance['url']}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting models from {instance['url']}: {e}")
        return []

    def backup_to_dataset(
        self,
        instances: List[Dict[str, Any]],
        dataset_name: str = "ollama-instances",
        repo_owner: str = "latterworks",
    ) -> bool:
        """Backup discovered instances to Hugging Face dataset.

        Args:
            instances: List of discovered instances to backup
            dataset_name: Name of the dataset repository
            repo_owner: Hugging Face username or organization

        Returns:
            True if backup successful, False otherwise
        """
        if not self.hf_api:
            print("Hugging Face token required for dataset backup")
            return False

        if not instances:
            print("No instances to backup")
            return False

        try:
            repo_id = f"{repo_owner}/{dataset_name}"

            try:
                create_repo(
                    repo_id=repo_id, repo_type="dataset", token=self.hf_token, exist_ok=True
                )
                print(f"Created/verified dataset repository: {repo_id}")
            except Exception as e:
                print(f"Repository creation info: {e}")

            dataset = Dataset.from_list(instances)

            dataset.push_to_hub(
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Update Ollama instances discovery - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )

            print(f"Successfully backed up {len(instances)} instances to {repo_id}")
            return True

        except Exception as e:
            logger.error(f"Dataset backup error: {e}")
            print(f"Dataset backup error: {e}")
            return False


class _OllamaClient:
    """High-level client for Ollama instance interaction and management.

    Provides streamlined interface for model operations, text generation,
    and instance management across discovered Ollama services.

    Args:
        host: Target Ollama host address
        port: Target Ollama port (default: 11434)
        use_ssl: Use HTTPS instead of HTTP (default: False)
    """

    def __init__(self, host: str, port: int = 11434, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        protocol = "https" if use_ssl else "http"
        self.base_url = f"{protocol}://{host}:{port}"
        self.headers = {}
        self.httpx_kwargs = {}

    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Execute text generation request using specified model.

        Args:
            model: Model name for generation
            prompt: Text prompt for generation
            stream: Enable streaming response mode

        Returns:
            Generation response with text and metadata
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": stream},
                headers=self.headers,
                timeout=30,
            )
            return response.json()
        except requests.Timeout:
            logger.warning(f"Timeout generating with model {model}")
            return {"error": "Request timeout"}
        except requests.RequestException as e:
            logger.warning(f"Request failed for generate: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            return {"error": str(e)}

    def list_models(self) -> List[str]:
        """Retrieve all available models from this instance.

        Returns:
            List of model names available for generation
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except requests.Timeout:
            logger.debug(f"Timeout listing models from {self.base_url}")
        except requests.RequestException as e:
            logger.debug(f"Request failed listing models: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error listing models: {e}")
        return []

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download and install model on this Ollama instance.

        Args:
            model_name: Name of model to download and install

        Returns:
            Pull operation result with status and metadata
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                headers=self.headers,
                timeout=300
            )
            return response.json()
        except requests.Timeout:
            logger.warning(f"Timeout pulling model {model_name}")
            return {"error": "Request timeout"}
        except requests.RequestException as e:
            logger.warning(f"Request failed pulling model: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error pulling model: {e}")
            return {"error": str(e)}


class SemanticModelMatcher:
    """Semantic model name resolution using embeddings for natural language queries.

    Enables users to specify models using natural language descriptions like
    'reasoning', 'creative writing', 'coding' instead of exact model names.
    Uses the latterworks/ollama-embeddings model for semantic similarity.

    Args:
        embedding_endpoint: Embedding model endpoint URL (default: LM Studio on localhost:1234)
        embedding_model: Model name for embeddings (default: text-embedding-nomic-embed-text-v1.5)
        model_descriptions: Custom model descriptions for semantic matching
    """

    DEFAULT_EMBEDDING_ENDPOINT = "http://127.0.0.1:1234/v1/embeddings"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

    def __init__(
        self,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.embedding_endpoint = embedding_endpoint or self.DEFAULT_EMBEDDING_ENDPOINT
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.model_cache = {}
        self.embedding_cache = {}

        # Default semantic descriptions for common model types
        self.semantic_descriptions = {
            "reasoning": "logical thinking analysis problem solving step by step reasoning mathematical calculations",
            "creative": "creative writing storytelling imagination narrative fiction poetry artistic expression",
            "coding": "programming code development software engineering debugging algorithms",
            "general": "general purpose conversation helpful assistant question answering",
            "chat": "conversational chatbot friendly dialogue interactive communication",
            "instruct": "instruction following task completion commands directives structured responses",
            "vision": "image analysis visual processing computer vision multimodal",
            "fast": "quick responses lightweight efficient speed performance optimized",
            "large": "comprehensive detailed thorough extensive knowledge advanced capabilities",
            "small": "lightweight compact efficient minimal resource usage",
        }

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using the local embedding model."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            response = requests.post(
                self.embedding_endpoint,
                headers={"Content-Type": "application/json"},
                json={"model": self.embedding_model, "input": text.lower()},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["data"][0]["embedding"])
                self.embedding_cache[text] = embedding
                return embedding

        except requests.Timeout:
            logger.debug(f"Embedding timeout for text: {text[:50]}...")
        except requests.RequestException as e:
            logger.debug(f"Embedding request failed: {e}")
        except Exception as e:
            logger.warning(f"Embedding error for '{text[:50]}...': {e}")

        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _extract_model_features(self, model_name: str) -> str:
        """Extract semantic features from model name for embedding."""
        model_lower = model_name.lower()

        # Extract features based on common naming patterns
        features = []

        # Size indicators
        if any(x in model_lower for x in ["1b", "3b", "7b", "8b", "13b", "34b", "70b"]):
            size_match = next(
                (x for x in ["1b", "3b", "7b", "8b", "13b", "34b", "70b"] if x in model_lower), ""
            )
            if size_match in ["1b", "3b", "7b"]:
                features.append("small lightweight efficient")
            elif size_match in ["13b", "34b", "70b"]:
                features.append("large comprehensive advanced")

        # Model family features
        if "llama" in model_lower:
            features.append("general purpose conversational instruction following")
        if "codellama" in model_lower or "code" in model_lower:
            features.append("programming coding software development")
        if "mistral" in model_lower:
            features.append("efficient balanced performance general purpose")
        if "gemma" in model_lower:
            features.append("instruction following helpful assistant")
        if "deepseek" in model_lower:
            features.append("reasoning analytical thinking problem solving")
        if "qwen" in model_lower:
            features.append("multilingual reasoning instruction following")
        if "phi" in model_lower:
            features.append("small efficient reasoning coding")
        if "yi" in model_lower:
            features.append("bilingual reasoning general purpose")

        # Capability indicators
        if "instruct" in model_lower or "chat" in model_lower:
            features.append("instruction following conversational chat")
        if "uncensored" in model_lower:
            features.append("creative writing unrestricted")
        if "vision" in model_lower:
            features.append("multimodal image analysis visual processing")

        return " ".join(features) + f" {model_name}"

    def find_best_model(
        self, query: str, available_models: List[str], similarity_threshold: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching model for a natural language query.

        Args:
            query: Natural language description of desired model
            available_models: List of available model names
            similarity_threshold: Minimum similarity score to consider

        Returns:
            Dictionary with matched model and confidence score
        """
        if not available_models:
            return None

        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            # Fallback to exact name matching
            query_lower = query.lower()
            for model in available_models:
                if query_lower in model.lower():
                    return {"model": model, "confidence": 1.0, "method": "exact_match"}
            return None

        best_match = None
        best_score = similarity_threshold

        # Check semantic descriptions first
        for semantic_key, description in self.semantic_descriptions.items():
            if semantic_key in query.lower():
                desc_embedding = self._get_embedding(description)
                if desc_embedding is not None:
                    for model in available_models:
                        model_features = self._extract_model_features(model)
                        model_embedding = self._get_embedding(model_features)

                        if model_embedding is not None:
                            similarity = self._cosine_similarity(desc_embedding, model_embedding)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = {
                                    "model": model,
                                    "confidence": similarity,
                                    "method": "semantic_description",
                                    "matched_concept": semantic_key,
                                }

        # Direct query to model comparison
        for model in available_models:
            model_features = self._extract_model_features(model)
            model_embedding = self._get_embedding(model_features)

            if model_embedding is not None:
                similarity = self._cosine_similarity(query_embedding, model_embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        "model": model,
                        "confidence": similarity,
                        "method": "direct_semantic",
                        "features_used": model_features,
                    }

        return best_match

    def explain_model_choice(self, query: str, available_models: List[str]) -> Dict[str, Any]:
        """Provide detailed explanation of model matching process."""
        result = self.find_best_model(query, available_models)

        explanation = {
            "query": query,
            "total_models": len(available_models),
            "embedding_available": self._get_embedding("test") is not None,
        }

        if result:
            explanation.update(
                {
                    "selected_model": result["model"],
                    "confidence": round(result["confidence"], 3),
                    "selection_method": result["method"],
                    "reasoning": result.get("matched_concept", "Direct semantic similarity"),
                }
            )
        else:
            explanation["error"] = "No suitable model found above confidence threshold"

        return explanation


class DistributedOllamaManager:
    """Manages distributed inference across multiple Ollama instances.

    Provides load balancing, failover, and performance optimization for
    queries across discovered Ollama instances. Routes requests based on
    model availability, response times, and instance health.

    Args:
        instances: List of discovered Ollama instances
        strategy: Load balancing strategy ('round_robin', 'fastest', 'least_loaded')
        timeout: Request timeout for inference operations
    """

    def __init__(
        self,
        instances: List[Dict[str, Any]],
        strategy: str = "fastest",
        timeout: float = 30.0,
        enable_semantic_matching: bool = True,
        headers: Dict[str, str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
        **httpx_kwargs,
    ):
        self.instances = instances
        self.strategy = strategy
        self.timeout = timeout
        self.headers = headers or {}
        self.httpx_kwargs = httpx_kwargs
        self.clients = {}
        self.instance_stats = {}
        self.failed_instances = set()

        # Initialize semantic model matcher
        self.semantic_matcher = (
            SemanticModelMatcher(
                embedding_endpoint=embedding_endpoint,
                embedding_model=embedding_model,
            )
            if enable_semantic_matching
            else None
        )

        # Initialize clients and stats for each instance
        for instance in instances:
            host, port = instance["host"], instance["port"]
            # Create client with custom headers and auth
            client = _OllamaClient(host, port)
            client.headers = self.headers
            client.httpx_kwargs = self.httpx_kwargs
            self.clients[f"{host}:{port}"] = client
            self.instance_stats[f"{host}:{port}"] = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": instance.get("version_response_time_ms", 1000),
                "current_load": 0,
                "models": instance.get("models", []),
                "last_health_check": instance.get("last_health_check") or 0,
            }

    def _get_available_instances_for_model(self, model: str) -> List[str]:
        """Get instances that have the specified model available."""
        available: List[str] = []
        unknown: List[str] = []

        for instance_key, stats in self.instance_stats.items():
            if instance_key in self.failed_instances:
                continue

            models = stats.get("models", [])

            if models:
                if model in models:
                    available.append(instance_key)
                continue

            # No cached model info – attempt a health check to refresh inventory
            last_check = stats.get("last_health_check", 0)
            if not last_check or (time.time() - last_check) >= max(self.timeout / 2, 1):
                if self._health_check_instance(instance_key):
                    models = self.instance_stats[instance_key].get("models", [])
                    if models and model in models:
                        available.append(instance_key)
                        continue
                    if models:
                        # Model inventory refreshed but desired model absent
                        continue

            # Still no information about available models; treat as unknown capability
            unknown.append(instance_key)

        # If we couldn't confirm any compatible nodes but have unknown-capability nodes,
        # surface them so the caller can attempt execution rather than failing fast.
        return available if available else unknown

    def _select_optimal_instance(self, available_instances: List[str]) -> Optional[str]:
        """Select the optimal instance based on the configured strategy."""
        if not available_instances:
            return None

        if self.strategy == "round_robin":
            return random.choice(available_instances)

        elif self.strategy == "fastest":
            return min(
                available_instances, key=lambda x: self.instance_stats[x]["average_response_time"]
            )

        elif self.strategy == "least_loaded":
            return min(available_instances, key=lambda x: self.instance_stats[x]["current_load"])

        return available_instances[0]

    def _update_instance_stats(self, instance_key: str, response_time: float, success: bool):
        """Update performance statistics for an instance."""
        stats = self.instance_stats[instance_key]
        stats["total_requests"] += 1

        if success:
            stats["successful_requests"] += 1
            # Update moving average of response time
            current_avg = stats["average_response_time"]
            stats["average_response_time"] = (current_avg * 0.8) + (response_time * 0.2)

        stats["current_load"] = max(0, stats["current_load"] - 1)

    def _health_check_instance(self, instance_key: str) -> bool:
        """Perform health check on a specific instance."""
        client = self.clients.get(instance_key)
        if not client:
            return False

        try:
            start_time = time.time()
            models = client.list_models()
            response_time = (time.time() - start_time) * 1000

            if models is None:
                models = []

            stats = self.instance_stats.get(instance_key)
            if stats is not None:
                stats["models"] = models
                stats["last_health_check"] = time.time()
                current_avg = stats.get("average_response_time", response_time)
                stats["average_response_time"] = (current_avg * 0.8) + (response_time * 0.2)

            self.failed_instances.discard(instance_key)
            return True

        except requests.Timeout:
            logger.debug(f"Health check timeout for {instance_key}")
        except requests.RequestException as e:
            logger.debug(f"Health check failed for {instance_key}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in health check for {instance_key}: {e}")

        self.failed_instances.add(instance_key)
        return False

    def _resolve_model_name(self, model_query: str) -> Optional[Dict[str, Any]]:
        """Resolve natural language model query to actual model name."""
        if not self.semantic_matcher:
            return {"model": model_query, "method": "exact"}

        # Get all available models across instances
        all_models = set()
        for stats in self.instance_stats.values():
            if stats["models"]:
                all_models.update(stats["models"])

        # Try semantic matching first
        match_result = self.semantic_matcher.find_best_model(model_query, list(all_models))
        if match_result:
            return match_result

        # Fallback to exact matching
        return {"model": model_query, "method": "exact"}

    def generate_distributed(
        self, model_query: str, prompt: str, max_retries: int = 2, parallel_requests: int = 1
    ) -> Dict[str, Any]:
        """Execute distributed inference with automatic failover and semantic model matching.

        Args:
            model_query: Natural language model description or exact model name
            prompt: Text prompt for generation
            max_retries: Maximum retry attempts on failure
            parallel_requests: Number of parallel requests (for ensemble)

        Returns:
            Generation response with metadata about execution and model resolution
        """
        # Resolve model name using semantic matching
        model_resolution = self._resolve_model_name(model_query)
        if not model_resolution:
            return {
                "error": f"Could not resolve model query: '{model_query}'",
                "query": model_query,
            }

        model = model_resolution["model"]
        available_instances = self._get_available_instances_for_model(model)

        if not available_instances:
            # Try health check on failed instances
            for instance_key in list(self.failed_instances):
                if self._health_check_instance(instance_key):
                    available_instances = self._get_available_instances_for_model(model)
                    break

        if not available_instances:
            return {
                "error": f"No available instances found for model '{model}'",
                "available_instances": len(self.instances) - len(self.failed_instances),
                "failed_instances": len(self.failed_instances),
            }

        # Single request mode
        if parallel_requests == 1:
            for attempt in range(max_retries + 1):
                instance_key = self._select_optimal_instance(available_instances)
                if not instance_key:
                    continue

                try:
                    client = self.clients[instance_key]
                    self.instance_stats[instance_key]["current_load"] += 1

                    start_time = time.time()
                    result = client.generate(model, prompt)
                    response_time = (time.time() - start_time) * 1000

                    if "error" not in result:
                        self._update_instance_stats(instance_key, response_time, True)
                        result["execution_metadata"] = {
                            "instance": instance_key,
                            "response_time_ms": round(response_time, 2),
                            "attempt": attempt + 1,
                            "strategy": self.strategy,
                            "model_resolution": model_resolution,
                        }
                        return result
                    else:
                        self._update_instance_stats(instance_key, response_time, False)

                except Exception as e:
                    self._update_instance_stats(instance_key, 0, False)
                    self.failed_instances.add(instance_key)
                    available_instances = [x for x in available_instances if x != instance_key]

        # Parallel/ensemble mode
        else:
            selected_instances = available_instances[:parallel_requests]
            results = []
            threads = []

            def execute_request(instance_key: str):
                try:
                    client = self.clients[instance_key]
                    start_time = time.time()
                    result = client.generate(model, prompt)
                    response_time = (time.time() - start_time) * 1000

                    result["execution_metadata"] = {
                        "instance": instance_key,
                        "response_time_ms": round(response_time, 2),
                        "strategy": "parallel",
                    }
                    results.append(result)
                    self._update_instance_stats(instance_key, response_time, "error" not in result)

                except Exception as e:
                    self._update_instance_stats(instance_key, 0, False)
                    results.append({"error": str(e), "instance": instance_key})

            for instance_key in selected_instances:
                thread = threading.Thread(target=execute_request, args=(instance_key,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Return first successful result or aggregate
            successful_results = [r for r in results if "error" not in r]
            if successful_results:
                best_result = min(
                    successful_results,
                    key=lambda x: x.get("execution_metadata", {}).get(
                        "response_time_ms", float("inf")
                    ),
                )
                best_result["parallel_results"] = results
                return best_result

        return {
            "error": "All instances failed after retries",
            "attempts": max_retries + 1,
            "failed_instances": list(self.failed_instances),
        }

    def get_flock_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the distributed flock."""
        total_instances = len(self.instances)
        healthy_instances = total_instances - len(self.failed_instances)

        # Aggregate model availability
        all_models = set()
        model_distribution = {}

        for instance_key, stats in self.instance_stats.items():
            if instance_key not in self.failed_instances:
                for model in stats["models"]:
                    all_models.add(model)
                    model_distribution[model] = model_distribution.get(model, 0) + 1

        return {
            "flock_health": {
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "failed_instances": len(self.failed_instances),
                "health_percentage": round((healthy_instances / total_instances) * 100, 1),
            },
            "model_availability": {
                "unique_models": len(all_models),
                "models": list(all_models),
                "distribution": model_distribution,
            },
            "performance_stats": {
                instance_key: {
                    "success_rate": round(
                        (stats["successful_requests"] / max(stats["total_requests"], 1)) * 100, 1
                    ),
                    "avg_response_time_ms": round(stats["average_response_time"], 2),
                    "current_load": stats["current_load"],
                    "model_count": len(stats["models"]),
                }
                for instance_key, stats in self.instance_stats.items()
                if instance_key not in self.failed_instances
            },
        }


def discover_remote_instances(
    shodan_api_key: str, query: str = "ollama", limit: int = 100
) -> List[Dict[str, Any]]:
    """Discover Ollama instances globally via Shodan with comprehensive verification.

    Performs sequential verification on each discovered instance:
    - Version/health check via /api/version
    - Model enumeration via /api/tags

    Args:
        shodan_api_key: Valid Shodan API key for global search
        query: Search term (searches any port for "ollama")
        limit: Maximum results (conservative credit usage)

    Returns:
        List of verified instances with comprehensive metadata
    """
    discovery = RemoteDiscovery(shodan_api_key=shodan_api_key)
    return discovery.search(query, limit)


def connect_to_ollama(host: str, port: int = 11434, use_ssl: bool = False) -> _OllamaClient:
    """Convenience function to create an Ollama client.

    Args:
        host: Target host address
        port: Target port (default: 11434)
        use_ssl: Use HTTPS instead of HTTP (default: False)

    Returns:
        Configured _OllamaClient instance
    """
    return _OllamaClient(host, port, use_ssl)


def create_distributed_manager(
    instances: List[Dict[str, Any]], strategy: str = "fastest"
) -> DistributedOllamaManager:
    """Create a distributed Ollama manager for remote instances.

    Args:
        instances: List of remote instances (required)
        strategy: Load balancing strategy ('fastest', 'round_robin', 'least_loaded')

    Returns:
        Configured DistributedOllamaManager ready for inference
    """
    if not instances:
        raise ValueError("No Ollama instances provided. Use discover_remote_instances() to find instances.")

    print(
        f"Initializing distributed manager with {len(instances)} remote instances using '{strategy}' strategy"
    )
    return DistributedOllamaManager(instances, strategy=strategy)


def main():
    """Main CLI entry point."""
    import argparse
    import os
    from ._version import __version__

    parser = argparse.ArgumentParser(
        description="LlamaSniffer - Discover remote Ollama instances globally."
    )
    parser.add_argument(
        "--discover", action="store_true", help="Discover remote Ollama instances via Shodan."
    )
    parser.add_argument(
        "--test-semantic", action="store_true", help="Test semantic model matching."
    )
    parser.add_argument("--query", type=str, default="ollama", help="Shodan search query.")
    parser.add_argument("--limit", type=int, default=10, help="Limit for Shodan results (default: 10).")
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Connection timeout for verification."
    )
    parser.add_argument("--version", action="version", version=f"llamasniffer {__version__}")

    args = parser.parse_args()

    if not any([args.discover, args.test_semantic]):
        parser.print_help()
        return

    if args.discover:
        shodan_api_key = os.environ.get("SHODAN_API_KEY")
        if not shodan_api_key:
            print("Error: SHODAN_API_KEY environment variable not set.")
        else:
            print(f"Discovering remote instances via Shodan (query: '{args.query}', limit: {args.limit})...")
            instances = discover_remote_instances(
                shodan_api_key=shodan_api_key, query=args.query, limit=args.limit
            )
            print(f"Discovered {len(instances)} remote instances:")
            for instance in instances:
                print(json.dumps(instance, indent=2))

    if args.test_semantic:
        print("Testing semantic model matching...")
        matcher = SemanticModelMatcher()
        test_models = ["llama2:7b", "codellama:13b", "deepseek-coder:6.7b", "phi3:3.8b"]
        test_queries = ["reasoning", "coding", "creative", "fast"]

        for query in test_queries:
            result = matcher.find_best_model(query, test_models)
            if result:
                print(f"'{query}' → {result['model']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"'{query}' → No match found")


if __name__ == "__main__":
    main()
