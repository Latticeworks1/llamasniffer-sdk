"""
Ollama client with distributed inference and semantic model matching.

Drop-in replacement for the standard ollama Python client with enhanced
global discovery, load balancing, and natural language model selection.

Usage:
    from llamasniffer import ollama

    response = await ollama.chat({
        'model': 'reasoning',  # Semantic model selection
        'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}]
    })
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Union
from .core import DistributedOllamaManager, SemanticModelMatcher, discover_ollama_instances, discover_ollama_shodan


class OllamaConfig:
    """Configuration management for Ollama client."""

    def __init__(self):
        self.semantic_enabled = True
        self.semantic_threshold = 0.3
        self.auto_discovery = True  # Use LlamaSniffer discovery methods
        self.strategy = "fastest"
        self.verbose_resolution = False
        self.fallback_to_first = True
        self.instances = None
        self.shodan_api_key = None
        self.network_prefix = "192.168.1"  # Default network range
        self._manager = None

    def set_semantic_matching(self, enabled: bool = True, threshold: float = 0.3):
        """Configure semantic model matching behavior."""
        self.semantic_enabled = enabled
        self.semantic_threshold = threshold

    def set_load_balancing(self, strategy: str = "fastest"):
        """Set load balancing strategy: 'fastest', 'round_robin', 'least_loaded'."""
        self.strategy = strategy

    def set_verbose(self, enabled: bool = True):
        """Enable verbose model resolution logging."""
        self.verbose_resolution = enabled

    def set_instances(self, instances: List[Dict] = None):
        """Manually configure Ollama instances."""
        self.instances = instances
        self._manager = None  # Reset manager to use new instances


class OllamaClient:
    """Ollama client with distributed inference and semantic matching."""

    def __init__(self, config: OllamaConfig = None, host: str = None, headers: Dict[str, str] = None, **httpx_kwargs):
        self.config = config or OllamaConfig()
        self.host = host
        self.headers = headers or {}
        self.httpx_kwargs = httpx_kwargs
        self._manager = None
        self._last_resolution = {}

    def _get_manager(self) -> DistributedOllamaManager:
        """Get or create distributed manager instance."""
        if self._manager is None:
            if self.host:
                # Single host mode with custom client settings
                instances = [{"host": self.host.split('://')[1].split(':')[0] if '://' in self.host else self.host.split(':')[0], 
                            "port": int(self.host.split(':')[-1]) if ':' in self.host else 11434,
                            "verified": True, "models": [], "response_time_ms": 0}]
            elif self.config.instances:
                instances = self.config.instances
            elif self.config.auto_discovery:
                # Use LlamaSniffer discovery methods
                instances = []
                
                # Network discovery first
                if self.config.network_prefix:
                    network_instances = discover_ollama_instances(
                        network_prefix=self.config.network_prefix
                    )
                    instances.extend(network_instances)
                
                # Shodan discovery if API key provided
                if self.config.shodan_api_key:
                    try:
                        shodan_instances = discover_ollama_shodan(
                            shodan_api_key=self.config.shodan_api_key,
                            query="ollama",
                            limit=10
                        )
                        instances.extend(shodan_instances)
                    except Exception as e:
                        print(f"Shodan discovery failed: {e}")
                
                if not instances:
                    raise ConnectionError(
                        "No Ollama instances discovered. Configure Shodan API key or set explicit instances."
                    )
            else:
                raise ValueError("No instances configured and auto-discovery disabled")

            self._manager = DistributedOllamaManager(
                instances=instances,
                strategy=self.config.strategy,
                enable_semantic_matching=self.config.semantic_enabled,
                headers=self.headers,
                **self.httpx_kwargs,
            )

        return self._manager

    def _resolve_model(self, model: str) -> Dict[str, Any]:
        """Resolve model name using semantic matching with transparency."""
        manager = self._get_manager()

        if not self.config.semantic_enabled:
            return {"model": model, "method": "exact", "confidence": 1.0}

        resolution = manager._resolve_model_name(model)

        if not resolution and self.config.fallback_to_first:
            # Fallback to first available model
            status = manager.get_cluster_status()
            available = status["model_availability"]["models"]
            if available:
                resolution = {"model": available[0], "method": "fallback", "confidence": 0.0}

        if self.config.verbose_resolution and resolution:
            print(
                f"Model resolution: '{model}' → '{resolution['model']}' "
                f"(method: {resolution['method']}, confidence: {resolution.get('confidence', 'N/A')})"
            )

        self._last_resolution = resolution or {}
        return resolution

    def chat(
        self, model: str = None, messages: List[Dict] = None, stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Ollama chat method with semantic model resolution.

        Args:
            model: Model name or semantic description
            messages: List of message objects with 'role' and 'content'
            stream: Enable streaming responses
            **kwargs: Additional parameters

        Returns:
            Ollama response object or generator if streaming
        """
        if isinstance(model, dict):
            # Handle chat({model: 'name', messages: [...]}) format
            chat_params = model
            model = chat_params.get("model")
            messages = chat_params.get("messages", messages)
            stream = chat_params.get("stream", stream)
            kwargs.update(
                {k: v for k, v in chat_params.items() if k not in ["model", "messages", "stream"]}
            )

        if not model or not messages:
            raise ValueError("Both 'model' and 'messages' are required")

        # Resolve model name
        resolution = self._resolve_model(model)
        if not resolution:
            raise ValueError(f"Could not resolve model: '{model}'")

        resolved_model = resolution["model"]

        # Convert messages to prompt (Ollama generate format)
        prompt = self._messages_to_prompt(messages)

        if stream:
            return self._chat_stream(resolved_model, prompt, messages, resolution)

        # Execute distributed inference
        manager = self._get_manager()
        response = manager.generate_distributed(
            resolved_model, prompt, max_retries=2, parallel_requests=1
        )

        if "error" in response:
            raise RuntimeError(f"Generation failed: {response['error']}")

        # Convert to Ollama format
        return self._format_chat_response(response, messages, resolution)

    def _chat_stream(self, model: str, prompt: str, messages: List[Dict], resolution: Dict):
        """Generate streaming chat responses."""
        import requests
        manager = self._get_manager()
        
        # Get best instance for this model
        available_instances = manager._get_available_instances_for_model(model)
        if not available_instances:
            raise RuntimeError(f"No instances available for model: {model}")
        
        instance = available_instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            yield {
                                "model": model,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                "message": {"role": "assistant", "content": data["response"]},
                                "done": data.get("done", False),
                                "model_resolution": resolution
                            }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {str(e)}")

    async def achat(
        self, model: str = None, messages: List[Dict] = None, stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Async version of chat method."""
        if stream:
            # Return async generator for streaming
            return self._achat_stream(model, messages, **kwargs)
        
        # Run synchronous chat in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat(model, messages, stream, **kwargs)
        )

    async def _achat_stream(self, model: str, messages: List[Dict], **kwargs):
        """Async streaming chat generator."""
        loop = asyncio.get_event_loop()
        for chunk in await loop.run_in_executor(None, lambda: self.chat(model, messages, stream=True, **kwargs)):
            yield chunk

    def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Ollama generate method.

        Args:
            model: Model name or semantic description
            prompt: Text prompt
            stream: Enable streaming
            **kwargs: Additional parameters

        Returns:
            Ollama response or generator if streaming
        """
        resolution = self._resolve_model(model)
        if not resolution:
            raise ValueError(f"Could not resolve model: '{model}'")

        resolved_model = resolution["model"]

        if stream:
            return self._generate_stream(resolved_model, prompt, resolution)

        manager = self._get_manager()
        response = manager.generate_distributed(
            resolved_model, prompt, max_retries=2, parallel_requests=1
        )

        if "error" in response:
            raise RuntimeError(f"Generation failed: {response['error']}")

        # Add resolution metadata
        response["model_resolution"] = resolution
        return response

    def _generate_stream(self, model: str, prompt: str, resolution: Dict):
        """Generate streaming responses."""
        import requests
        manager = self._get_manager()
        
        # Get best instance for this model
        available_instances = manager._get_available_instances_for_model(model)
        if not available_instances:
            raise RuntimeError(f"No instances available for model: {model}")
        
        instance = available_instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        data["model_resolution"] = resolution
                        yield data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {str(e)}")

    async def agenerate(
        self, model: str, prompt: str, stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Async version of generate method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(model, prompt, stream, **kwargs)
        )

    def list(self) -> Dict[str, List[Dict]]:
        """List all available models across instances."""
        manager = self._get_manager()
        status = manager.get_cluster_status()

        models = []
        for model_name in status["model_availability"]["models"]:
            instance_count = status["model_availability"]["distribution"].get(model_name, 0)
            models.append(
                {
                    "name": model_name,
                    "size": 0,  # Size not available in our discovery
                    "digest": "",  # Digest not available
                    "instances": instance_count,
                    "modified_at": time.time(),
                }
            )

        return {"models": models}

    def show(self, model: str) -> Dict[str, Any]:
        """Show model information with semantic resolution details."""
        resolution = self._resolve_model(model)
        manager = self._get_manager()
        status = manager.get_cluster_status()

        if resolution and resolution["model"] in status["model_availability"]["models"]:
            return {
                "license": "",
                "modelfile": "",
                "parameters": "",
                "template": "",
                "details": {
                    "format": "ggu",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0",
                },
                "model_resolution": resolution,
                "available_instances": status["model_availability"]["distribution"].get(
                    resolution["model"], 0
                ),
            }
        else:
            raise ValueError(f"Model not found: {model}")

    def ps(self) -> Dict[str, List[Dict]]:
        """Show running models (cluster status)."""
        manager = self._get_manager()
        status = manager.get_cluster_status()

        models = []
        for model_name in status["model_availability"]["models"]:
            models.append(
                {
                    "name": model_name,
                    "size": 0,
                    "processor": "gpu",
                    "until": time.time() + 3600,  # 1 hour from now
                    "instances": status["model_availability"]["distribution"].get(model_name, 0),
                }
            )

        return {"models": models, "cluster_status": status}

    def embed(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings for text input."""
        import requests
        
        resolution = self._resolve_model(model)
        if not resolution:
            raise ValueError(f"Could not resolve model: '{model}'")

        resolved_model = resolution["model"]
        manager = self._get_manager()
        
        # Get best instance for this model
        available_instances = manager._get_available_instances_for_model(resolved_model)
        if not available_instances:
            raise RuntimeError(f"No instances available for model: {resolved_model}")
        
        instance = available_instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/embed"
        
        payload = {
            "model": resolved_model,
            "input": input
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            result["model_resolution"] = resolution
            return result
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")

    def pull(self, model: str, **kwargs) -> Dict[str, Any]:
        """Pull/download a model."""
        import requests
        
        manager = self._get_manager()
        instances = manager.instances
        
        if not instances:
            raise RuntimeError("No instances available")
        
        instance = instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/pull"
        
        payload = {"model": model}
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Pull failed: {str(e)}")

    def delete(self, model: str, **kwargs) -> Dict[str, Any]:
        """Delete a model."""
        import requests
        
        manager = self._get_manager()
        instances = manager.instances
        
        results = []
        for instance in instances:
            url = f"http://{instance['host']}:{instance['port']}/api/delete"
            payload = {"model": model}
            
            try:
                headers = {"Content-Type": "application/json"}
                headers.update(manager.headers)
                
                response = requests.delete(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                results.append({"instance": f"{instance['host']}:{instance['port']}", "success": True})
            except Exception as e:
                results.append({"instance": f"{instance['host']}:{instance['port']}", "success": False, "error": str(e)})
        
        return {"results": results}

    def create(self, model: str, from_: str = None, modelfile: str = None, **kwargs) -> Dict[str, Any]:
        """Create a new model."""
        import requests
        
        manager = self._get_manager()
        instances = manager.instances
        
        if not instances:
            raise RuntimeError("No instances available")
        
        instance = instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/create"
        
        payload = {"model": model}
        if from_:
            payload["from"] = from_
        if modelfile:
            payload["modelfile"] = modelfile
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Create failed: {str(e)}")

    def copy(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        """Copy a model."""
        import requests
        
        manager = self._get_manager()
        instances = manager.instances
        
        if not instances:
            raise RuntimeError("No instances available")
        
        instance = instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/copy"
        
        payload = {"source": source, "destination": destination}
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Copy failed: {str(e)}")

    def push(self, model: str, **kwargs) -> Dict[str, Any]:
        """Push a model to registry."""
        import requests
        
        manager = self._get_manager()
        instances = manager.instances
        
        if not instances:
            raise RuntimeError("No instances available")
        
        instance = instances[0]
        url = f"http://{instance['host']}:{instance['port']}/api/push"
        
        payload = {"model": model}
        
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(manager.headers)
            
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Push failed: {str(e)}")

    def inspect_last_resolution(self) -> Dict[str, Any]:
        """Inspect the last model resolution for debugging."""
        return {
            "last_resolution": self._last_resolution,
            "config": {
                "semantic_enabled": self.config.semantic_enabled,
                "semantic_threshold": self.config.semantic_threshold,
                "strategy": self.config.strategy,
                "verbose_resolution": self.config.verbose_resolution,
            },
        }

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role.title()}: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def _format_chat_response(
        self, response: Dict, messages: List[Dict], resolution: Dict
    ) -> Dict[str, Any]:
        """Format response to match Ollama chat API."""
        message_content = response.get("response", "")

        return {
            "model": resolution["model"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {"role": "assistant", "content": message_content},
            "done": True,
            "total_duration": int(
                response["execution_metadata"]["response_time_ms"] * 1000000
            ),  # nanoseconds
            "load_duration": 0,
            "prompt_eval_count": len(" ".join(msg["content"] for msg in messages).split()),
            "prompt_eval_duration": 0,
            "eval_count": len(message_content.split()),
            "eval_duration": int(response["execution_metadata"]["response_time_ms"] * 1000000),
            "model_resolution": resolution,
            "execution_metadata": response["execution_metadata"],
        }


# Global client instance and configuration
_config = OllamaConfig()
_client = OllamaClient(_config)


# Ollama-compatible module-level functions
def chat(
    model: str = None, messages: List[Dict] = None, stream: bool = False, **kwargs
) -> Dict[str, Any]:
    """Module-level chat function for drop-in compatibility."""
    return _client.chat(model, messages, stream, **kwargs)


async def achat(
    model: str = None, messages: List[Dict] = None, stream: bool = False, **kwargs
) -> Dict[str, Any]:
    """Async module-level chat function."""
    return await _client.achat(model, messages, stream, **kwargs)


def generate(model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any]:
    """Module-level generate function."""
    return _client.generate(model, prompt, stream, **kwargs)


async def agenerate(model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any]:
    """Async module-level generate function."""
    return await _client.agenerate(model, prompt, stream, **kwargs)


def list() -> Dict[str, List[Dict]]:
    """Module-level list function."""
    return _client.list()


def show(model: str) -> Dict[str, Any]:
    """Module-level show function."""
    return _client.show(model)


def ps() -> Dict[str, List[Dict]]:
    """Module-level ps function."""
    return _client.ps()


def embed(model: str, input: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
    """Module-level embed function."""
    return _client.embed(model, input, **kwargs)


def pull(model: str, **kwargs) -> Dict[str, Any]:
    """Module-level pull function."""
    return _client.pull(model, **kwargs)


def delete(model: str, **kwargs) -> Dict[str, Any]:
    """Module-level delete function."""
    return _client.delete(model, **kwargs)


def create(model: str, from_: str = None, modelfile: str = None, **kwargs) -> Dict[str, Any]:
    """Module-level create function."""
    return _client.create(model, from_, modelfile, **kwargs)


def copy(source: str, destination: str, **kwargs) -> Dict[str, Any]:
    """Module-level copy function."""
    return _client.copy(source, destination, **kwargs)


def push(model: str, **kwargs) -> Dict[str, Any]:
    """Module-level push function."""
    return _client.push(model, **kwargs)


# Configuration functions
def configure(
    semantic_enabled: bool = True,
    semantic_threshold: float = 0.3,
    strategy: str = "fastest",
    verbose: bool = False,
    instances: List[Dict] = None,
    shodan_api_key: str = None,
    network_prefix: str = "192.168.1",
) -> None:
    """Configure the global Ollama client."""
    global _client, _config

    _config.set_semantic_matching(semantic_enabled, semantic_threshold)
    _config.set_load_balancing(strategy)
    _config.set_verbose(verbose)
    if instances:
        _config.set_instances(instances)
    if shodan_api_key:
        _config.shodan_api_key = shodan_api_key
    _config.network_prefix = network_prefix

    # Reset client to use new configuration
    _client = OllamaClient(_config)


def inspect_resolution() -> Dict[str, Any]:
    """Inspect the last model resolution."""
    return _client.inspect_last_resolution()


def get_cluster_status() -> Dict[str, Any]:
    """Get distributed cluster status."""
    manager = _client._get_manager()
    return manager.get_cluster_status()


def get_cached_endpoints() -> Dict[str, Any]:
    """Get detailed info about all cached/discovered endpoints."""
    manager = _client._get_manager()
    
    endpoint_details = []
    for instance in manager.instances:
        endpoint_info = {
            "host": instance["host"],
            "port": instance["port"],
            "url": f"http://{instance['host']}:{instance['port']}",
            "verified": instance.get("verified", False),
            "models": instance.get("models", []),
            "discovery_method": instance.get("discovery_method", "unknown"),
            "discovered_at": instance.get("discovered_at", 0),
            "response_time_ms": instance.get("response_time_ms", 0),
            "version": instance.get("version", "unknown"),
            "last_updated": instance.get("last_health_check", 0),
        }
        
        # Add performance stats if available
        instance_key = f"{instance['host']}:{instance['port']}"
        if instance_key in manager.instance_stats:
            stats = manager.instance_stats[instance_key]
            endpoint_info.update({
                "total_requests": stats["total_requests"],
                "successful_requests": stats["successful_requests"],
                "success_rate": round((stats["successful_requests"] / max(stats["total_requests"], 1)) * 100, 1),
                "avg_response_time": round(stats["average_response_time"], 2),
                "current_load": stats["current_load"],
                "is_healthy": instance_key not in manager.failed_instances,
            })
        
        endpoint_details.append(endpoint_info)
    
    return {
        "total_endpoints": len(endpoint_details),
        "healthy_endpoints": len([e for e in endpoint_details if e.get("is_healthy", True)]),
        "failed_endpoints": len(manager.failed_instances),
        "discovery_summary": {
            method: len([e for e in endpoint_details if e["discovery_method"] == method])
            for method in set(e["discovery_method"] for e in endpoint_details)
        },
        "endpoints": endpoint_details,
    }


def refresh_endpoint_health() -> Dict[str, Any]:
    """Force refresh health check on all cached endpoints."""
    manager = _client._get_manager()
    
    refreshed = 0
    errors = []
    
    for instance_key in list(manager.failed_instances):
        try:
            if manager._health_check_instance(instance_key):
                manager.failed_instances.discard(instance_key)
                refreshed += 1
        except Exception as e:
            errors.append(f"{instance_key}: {str(e)}")
    
    return {
        "refreshed_endpoints": refreshed,
        "errors": errors,
        "current_status": get_cluster_status(),
    }


# Create ollama-compatible object for import ollama syntax
class OllamaModule:
    """Module object to support 'import ollama' syntax."""

    def __init__(self):
        self.chat = chat
        self.achat = achat
        self.generate = generate
        self.agenerate = agenerate
        self.list = list
        self.show = show
        self.ps = ps
        self.embed = embed
        self.pull = pull
        self.delete = delete
        self.create = create
        self.copy = copy
        self.push = push
        self.configure = configure
        self.inspect_resolution = inspect_resolution
        self.get_cluster_status = get_cluster_status
        self.get_cached_endpoints = get_cached_endpoints
        self.refresh_endpoint_health = refresh_endpoint_health
        self.Client = OllamaClient
        self.AsyncClient = OllamaClient  # Same class handles both sync/async
        self.Config = OllamaConfig


# Create the module instance
ollama = OllamaModule()
