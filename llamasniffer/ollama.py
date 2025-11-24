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
import os
from pathlib import Path
from typing import Dict, List, Any, Union
from .core import DistributedOllamaManager, SemanticModelMatcher, discover_remote_instances


class OllamaConfig:
    """Unified configuration management for distributed Ollama client."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config_file()
        self._config = self._load_config()
        self._manager = None

    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        search_paths = [
            "llamasniffer_config.json",
            os.path.expanduser("~/.llamasniffer/config.json"),
            "/etc/llamasniffer/config.json"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Return default path in current directory
        return "llamasniffer_config.json"

    def _load_config(self) -> Dict:
        """Load configuration from JSON file with automatic key detection."""
        config = self._default_config()
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge file config with defaults
                    self._deep_merge(config, file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
        
        # Auto-detect and populate API keys
        self._detect_api_keys(config)
        
        return config

    def _deep_merge(self, base: Dict, updates: Dict):
        """Deep merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _detect_api_keys(self, config: Dict):
        """Automatically detect and populate API keys from environment and files."""
        # Shodan API key detection
        if not config["shodan"]["api_key"]:
            shodan_key = self._find_shodan_key()
            if shodan_key:
                config["shodan"]["api_key"] = shodan_key
                print(f"Auto-detected Shodan API key from system")
        
        # Hugging Face token detection
        if not config["huggingface"]["token"]:
            hf_token = self._find_huggingface_token()
            if hf_token:
                config["huggingface"]["token"] = hf_token
                print(f"Auto-detected Hugging Face token from system")

    def _find_shodan_key(self) -> str:
        """Find Shodan API key from environment variables and common file locations."""
        # Check environment variables
        env_vars = ['SHODAN_API_KEY', 'SHODAN_KEY', 'SHODAN_TOKEN']
        for var in env_vars:
            key = os.environ.get(var)
            if key and len(key.strip()) > 10:  # Basic validation
                return key.strip()
        
        # Check common file locations
        key_files = [
            os.path.expanduser("~/.shodan/api_key"),
            os.path.expanduser("~/.config/shodan/api_key"), 
            os.path.expanduser("~/.shodan_key"),
            "./shodan_key.txt",
            "./api_keys/shodan.txt"
        ]
        
        for filepath in key_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        key = f.read().strip()
                        if len(key) > 10:  # Basic validation
                            return key
                except IOError:
                    continue
        
        return None

    def _find_huggingface_token(self) -> str:
        """Find Hugging Face token from environment variables and common file locations."""
        # Check environment variables
        env_vars = ['HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGING_FACE_HUB_TOKEN']
        for var in env_vars:
            token = os.environ.get(var)
            if token and len(token.strip()) > 10:  # Basic validation
                return token.strip()
        
        # Check common file locations
        token_files = [
            os.path.expanduser("~/.huggingface/token"),
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.expanduser("~/.hf_token"),
            "./hf_token.txt",
            "./api_keys/huggingface.txt"
        ]
        
        for filepath in token_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        token = f.read().strip()
                        if len(token) > 10:  # Basic validation
                            return token
                except IOError:
                    continue
        
        return None

    def _default_config(self) -> Dict:
        """Return default configuration for remote-only operation."""
        return {
            "shodan": {
                "api_key": None,
                "query": "ollama",
                "limit": 10,
                "search_timeout": 30
            },
            "flock": {
                "instances": [],
                "verification_timeout": 5.0,
                "health_check_interval": 60,
                "max_retries": 3
            },
            "semantic": {
                "enabled": True,
                "threshold": 0.3,
                "embedding_model": "text-embedding-nomic-embed-text-v1.5",
                "embedding_api_url": "http://127.0.0.1:1234/v1/embeddings"
            },
            "routing": {
                "strategy": "fastest",
                "fallback_to_first": True,
                "load_balancing": "round_robin"
            },
            "huggingface": {
                "token": None,
                "username": None,
                "auto_backup": False,
                "repo_prefix": "llamasniffer",
                "dataset_owner": "latterworks"
            },
            "client": {
                "verbose_resolution": False,
                "custom_headers": {},
                "timeout": 30
            }
        }

    def save(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    # Properties for backward compatibility
    @property
    def semantic_enabled(self) -> bool:
        return self._config["semantic"]["enabled"]
    
    @property
    def semantic_threshold(self) -> float:
        return self._config["semantic"]["threshold"]
    
    @property
    def strategy(self) -> str:
        return self._config["routing"]["strategy"]
    
    @property
    def verbose_resolution(self) -> bool:
        return self._config["client"]["verbose_resolution"]
    
    @property
    def fallback_to_first(self) -> bool:
        return self._config["routing"]["fallback_to_first"]
    
    @property
    def instances(self) -> List[Dict]:
        return self._config["flock"]["instances"]
    
    @property
    def shodan_api_key(self) -> str:
        return self._config["shodan"]["api_key"]

    # Configuration methods
    def set_semantic_matching(self, enabled: bool = True, threshold: float = 0.3):
        """Configure semantic model matching behavior."""
        self._config["semantic"]["enabled"] = enabled
        self._config["semantic"]["threshold"] = threshold

    def set_load_balancing(self, strategy: str = "fastest"):
        """Set load balancing strategy: 'fastest', 'round_robin', 'least_loaded'."""
        self._config["routing"]["strategy"] = strategy

    def set_verbose(self, enabled: bool = True):
        """Enable verbose model resolution logging."""
        self._config["client"]["verbose_resolution"] = enabled

    def set_instances(self, instances: List[Dict] = None):
        """Manually configure remote Ollama instances."""
        self._config["flock"]["instances"] = instances or []
        self._manager = None  # Reset manager to use new instances
    
    def set_huggingface(self, token: str = None, username: str = None, auto_backup: bool = False, repo_prefix: str = "llamasniffer"):
        """Configure Hugging Face integration."""
        self._config["huggingface"]["token"] = token
        self._config["huggingface"]["username"] = username
        self._config["huggingface"]["auto_backup"] = auto_backup
        self._config["huggingface"]["repo_prefix"] = repo_prefix

    def set_shodan_key(self, api_key: str):
        """Configure Shodan API key for global discovery."""
        self._config["shodan"]["api_key"] = api_key


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
            else:
                # Remote-only discovery via Shodan
                instances = []
                
                if self.config.shodan_api_key:
                    try:
                        shodan_instances = discover_remote_instances(
                            shodan_api_key=self.config.shodan_api_key,
                            query="ollama",
                            limit=self.config._config["shodan"]["limit"]
                        )
                        instances.extend(shodan_instances)
                    except Exception as e:
                        print(f"Shodan discovery failed: {e}")
                
                if not instances:
                    raise ConnectionError(
                        "No Ollama instances discovered. Configure Shodan API key or set explicit remote instances."
                    )

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

        instance_key = manager._select_optimal_instance(available_instances) or available_instances[0]
        client = manager.clients.get(instance_key)
        if not client:
            raise RuntimeError(f"Instance client not found for: {instance_key}")

        url = f"http://{client.host}:{client.port}/api/generate"
        
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

        instance_key = manager._select_optimal_instance(available_instances) or available_instances[0]
        client = manager.clients.get(instance_key)
        if not client:
            raise RuntimeError(f"Instance client not found for: {instance_key}")

        url = f"http://{client.host}:{client.port}/api/generate"
        
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

        instance_key = manager._select_optimal_instance(available_instances) or available_instances[0]
        client = manager.clients.get(instance_key)
        if not client:
            raise RuntimeError(f"Instance client not found for: {instance_key}")

        url = f"http://{client.host}:{client.port}/api/embed"
        
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
    hf_token: str = None,
    hf_username: str = None,
    hf_auto_backup: bool = False,
    hf_repo_prefix: str = "llamasniffer",
) -> None:
    """Configure the global Ollama client for remote-only operation."""
    global _client, _config

    _config.set_semantic_matching(semantic_enabled, semantic_threshold)
    _config.set_load_balancing(strategy)
    _config.set_verbose(verbose)
    if instances:
        _config.set_instances(instances)
    if shodan_api_key:
        _config.set_shodan_key(shodan_api_key)
    
    # Configure Hugging Face integration
    if hf_token or hf_username:
        _config.set_huggingface(hf_token, hf_username, hf_auto_backup, hf_repo_prefix)

    # Reset client to use new configuration
    _client = OllamaClient(_config)


def inspect_resolution() -> Dict[str, Any]:
    """Inspect the last model resolution."""
    return _client.inspect_last_resolution()


def auto_discover_instances(limit: int = 10, use_cache: bool = True, ttl_hours: int = 24,
                          quality_check: bool = True, min_quality: float = 0.5) -> List[Dict]:
    """Discover remote Ollama instances using auto-detected Shodan API key from config.

    Convenience function that automatically detects the Shodan API key from:
    - Environment variables (SHODAN_API_KEY, SHODAN_KEY, SHODAN_TOKEN)
    - Config files (~/.shodan/api_key, ./shodan_key.txt, etc.)
    - llamasniffer config file

    By default, uses cached endpoints if available (refreshes every 24 hours).
    Also performs quality checks to filter out broken/garbled models.

    For explicit API key control, use discover_remote_instances() from core module.

    Args:
        limit: Maximum number of instances to discover (default: 10)
        use_cache: Use cached endpoints if available (default: True)
        ttl_hours: Cache time-to-live in hours (default: 24)
        quality_check: Test endpoints and filter out low-quality ones (default: True)
        min_quality: Minimum quality score to accept (0.0-1.0, default: 0.5)

    Returns:
        List of discovered, quality-validated remote Ollama instances
    """
    global _config

    # Try to use cache first
    if use_cache:
        from .endpoint_cache import EndpointCache
        cache = EndpointCache(ttl_hours=ttl_hours)
        cached_endpoints = cache.get_endpoints()

        if cached_endpoints:
            cache_info = cache.get_cache_info()
            print(f"Using cached endpoints ({len(cached_endpoints)} instances, age: {cache_info['age_hours']:.1f}h)")
            return cached_endpoints[:limit]  # Return up to limit from cache

    # No valid cache, need to scan
    if not _config.shodan_api_key:
        raise ValueError("No Shodan API key found. Set SHODAN_API_KEY environment variable or create shodan_key.txt file")

    try:
        from .core import discover_remote_instances
        from .endpoint_cache import EndpointCache
        from .quality_control import filter_quality_endpoints

        print(f"Scanning Shodan for fresh endpoints...")
        endpoints = discover_remote_instances(
            shodan_api_key=_config.shodan_api_key,
            query="ollama",
            limit=limit * 2  # Get more since we'll filter some out
        )

        # Quality check endpoints if requested
        if quality_check and endpoints:
            print(f"Quality testing {len(endpoints)} endpoints...")
            endpoints = filter_quality_endpoints(
                endpoints,
                min_score=min_quality,
                timeout=10.0,
                verbose=True
            )
            print(f"✅ {len(endpoints)} endpoints passed quality checks")

        # Cache the results
        if use_cache and endpoints:
            cache = EndpointCache(ttl_hours=ttl_hours)
            cache.save(endpoints, metadata={
                'query': 'ollama',
                'limit': limit,
                'source': 'shodan',
                'quality_checked': quality_check,
                'min_quality': min_quality
            })
            print(f"Cached {len(endpoints)} quality endpoints (expires in {ttl_hours}h)")

        return endpoints[:limit]  # Return up to requested limit
    except Exception as e:
        print(f"Remote discovery failed: {e}")
        return []


def get_cluster_status() -> Dict[str, Any]:
    """Get distributed cluster status."""
    manager = _client._get_manager()
    return manager.get_cluster_status()


def get_flock_status() -> Dict[str, Any]:
    """Get distributed flock status and intelligence."""
    manager = _client._get_manager()
    status = manager.get_cluster_status()
    
    # Transform cluster terminology to flock terminology
    flock_status = {
        "flock_health": status["cluster_health"],
        "model_availability": status["model_availability"],
        "performance_stats": status["performance_stats"]
    }
    
    return flock_status


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


def get_flock_nodes() -> Dict[str, Any]:
    """Get detailed info about all nodes in the flock."""
    manager = _client._get_manager()
    
    node_details = []
    for instance in manager.instances:
        node_info = {
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
            node_info.update({
                "total_requests": stats["total_requests"],
                "successful_requests": stats["successful_requests"],
                "success_rate": round((stats["successful_requests"] / max(stats["total_requests"], 1)) * 100, 1),
                "avg_response_time": round(stats["average_response_time"], 2),
                "current_load": stats["current_load"],
                "is_healthy": instance_key not in manager.failed_instances,
            })
        
        node_details.append(node_info)
    
    return {
        "total_nodes": len(node_details),
        "active_nodes": len([n for n in node_details if n.get("is_healthy", True)]),
        "failed_nodes": len(manager.failed_instances),
        "discovery_summary": {
            method: len([n for n in node_details if n["discovery_method"] == method])
            for method in set(n["discovery_method"] for n in node_details)
        },
        "nodes": node_details,
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


def refresh_flock_health() -> Dict[str, Any]:
    """Force refresh health check across the flock."""
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
        "refreshed_nodes": refreshed,
        "errors": errors,
        "current_status": get_flock_status(),
    }


def parallel_generate_flock(
    model: str, 
    prompt: str, 
    count: int = 10, 
    max_nodes: int = None,
    timeout: float = 60.0,
    backup_to_hf: bool = False,
    hf_repo: str = None
) -> List[Dict[str, Any]]:
    """Generate multiple synthetic data samples in parallel across the flock.
    
    Args:
        model: Model name or semantic description
        prompt: Generation prompt to send to all nodes
        count: Number of parallel generations desired
        max_nodes: Maximum nodes to use (defaults to min(count, available_nodes))
        timeout: Timeout per generation in seconds
        backup_to_hf: Whether to backup results to Hugging Face
        hf_repo: HF repository (format: 'username/repo-name')
        
    Returns:
        Dictionary with generation results and metadata
    """
    import threading
    import time
    
    manager = _client._get_manager()
    
    # Resolve model semantically
    resolution = manager._resolve_model_name(model)
    if not resolution:
        raise ValueError(f"Could not resolve model: '{model}'")
    
    resolved_model = resolution["model"]
    available_instances = manager._get_available_instances_for_model(resolved_model)
    
    if not available_instances:
        raise RuntimeError(f"No available nodes for model: {resolved_model}")
    
    # Determine how many nodes to use
    nodes_to_use = min(count, len(available_instances))
    if max_nodes:
        nodes_to_use = min(nodes_to_use, max_nodes)
    
    selected_instances = available_instances[:nodes_to_use]
    
    # If we need more generations than nodes, some nodes will do multiple
    generations_per_node = count // len(selected_instances)
    extra_generations = count % len(selected_instances)
    
    results = []
    threads = []
    results_lock = threading.Lock()
    
    def execute_generations(instance_key: str, num_generations: int):
        """Execute multiple generations on a single node."""
        try:
            client = manager.clients[instance_key]
            
            for i in range(num_generations):
                start_time = time.time()
                result = client.generate(resolved_model, prompt)
                response_time = (time.time() - start_time) * 1000
                
                if "error" not in result:
                    generation_result = {
                        "response": result.get("response", ""),
                        "model": resolved_model,
                        "model_resolution": resolution,
                        "execution_metadata": {
                            "instance": instance_key,
                            "generation_id": f"{instance_key}_{i}",
                            "response_time_ms": round(response_time, 2),
                            "strategy": "parallel_synthetic",
                            "node_generation_index": i
                        }
                    }
                    
                    with results_lock:
                        results.append(generation_result)
                    
                    manager._update_instance_stats(instance_key, response_time, True)
                else:
                    with results_lock:
                        results.append({
                            "error": result.get("error", "Unknown error"),
                            "instance": instance_key,
                            "generation_id": f"{instance_key}_{i}"
                        })
                    manager._update_instance_stats(instance_key, response_time, False)
                        
        except Exception as e:
            with results_lock:
                results.append({
                    "error": str(e),
                    "instance": instance_key
                })
            manager._update_instance_stats(instance_key, 0, False)
    
    # Distribute work across nodes
    for i, instance_key in enumerate(selected_instances):
        num_gens = generations_per_node
        if i < extra_generations:  # Distribute extra generations
            num_gens += 1
            
        thread = threading.Thread(
            target=execute_generations, 
            args=(instance_key, num_gens),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all generations with timeout
    start_time = time.time()
    for thread in threads:
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 0:
            thread.join(timeout=remaining_time)
        else:
            break
    
    # Filter successful results
    successful_results = [r for r in results if "error" not in r]
    
    result_data = {
        "synthetic_data": successful_results,
        "total_requested": count,
        "total_generated": len(successful_results),
        "nodes_used": len(selected_instances),
        "generation_summary": {
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "success_rate": round(len(successful_results) / max(len(results), 1) * 100, 1)
        },
        "flock_metadata": {
            "model_resolution": resolution,
            "nodes_utilized": selected_instances,
            "total_response_time": sum(r.get("execution_metadata", {}).get("response_time_ms", 0) 
                                     for r in successful_results),
            "average_response_time": round(
                sum(r.get("execution_metadata", {}).get("response_time_ms", 0) for r in successful_results) 
                / max(len(successful_results), 1), 2
            )
        }
    }
    
    # Backup to Hugging Face if requested
    if backup_to_hf and successful_results:
        try:
            backup_result = backup_flock_data(
                data=successful_results,
                data_type="synthetic_generation",
                hf_repo=hf_repo,
                metadata={
                    "prompt": prompt,
                    "model": resolved_model,
                    "generation_count": len(successful_results),
                    "flock_nodes": selected_instances
                }
            )
            result_data["hf_backup"] = backup_result
        except Exception as e:
            result_data["hf_backup"] = {"error": str(e)}
    
    return result_data


def setup_hf_auth(token: str = None) -> bool:
    """Setup Hugging Face authentication using CLI or provided token.
    
    Args:
        token: Optional HF token. If not provided, uses huggingface-cli login
        
    Returns:
        True if authentication successful
    """
    try:
        from huggingface_hub import login, whoami
        import subprocess
        import os
        
        if token:
            # Use provided token
            login(token=token)
            print("✅ Hugging Face authentication successful (token)")
        else:
            # Check if already logged in
            try:
                user_info = whoami()
                print(f"✅ Already logged in to Hugging Face as: {user_info['name']}")
                return True
            except:
                # Need to login via CLI
                print("🔐 Please authenticate with Hugging Face...")
                print("Run: huggingface-cli login")
                
                # Try automatic CLI login
                try:
                    result = subprocess.run(
                        ["huggingface-cli", "login"],
                        capture_output=False,
                        text=True
                    )
                    if result.returncode == 0:
                        print("✅ Hugging Face CLI authentication successful")
                    else:
                        print("❌ CLI authentication failed. Please run manually: huggingface-cli login")
                        return False
                except FileNotFoundError:
                    print("❌ huggingface-cli not found. Install with: pip install huggingface_hub[cli]")
                    return False
        
        # Verify authentication
        whoami()
        return True
        
    except Exception as e:
        print(f"❌ HF authentication error: {e}")
        return False


def backup_flock_data(
    data: List[Dict], 
    data_type: str,
    hf_repo: str = None,
    metadata: Dict = None
) -> Dict[str, Any]:
    """Backup flock data to Hugging Face with automatic authentication.
    
    Args:
        data: Data to backup (flock nodes, synthetic generations, etc.)
        data_type: Type of data ('flock_discovery', 'synthetic_generation', 'flock_health')
        hf_repo: HF repository (format: 'username/repo-name')
        metadata: Additional metadata to include
        
    Returns:
        Backup result with repository info
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi, create_repo, whoami
        import time
        
        # Check authentication
        try:
            user_info = whoami()
            username = user_info['name']
        except:
            print("❌ Not authenticated with Hugging Face. Run setup_hf_auth() first.")
            return {"error": "Authentication required"}
        
        # Determine repository
        if not hf_repo:
            hf_repo = f"{username}/llamasniffer-{data_type}"
        
        # Prepare data for dataset
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        dataset_data = []
        
        for i, item in enumerate(data):
            record = {
                "id": f"{data_type}_{timestamp}_{i}",
                "timestamp": timestamp,
                "data_type": data_type,
                **item
            }
            
            # Add metadata if provided
            if metadata:
                record["metadata"] = metadata
                
            dataset_data.append(record)
        
        # Create dataset
        dataset = Dataset.from_list(dataset_data)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=hf_repo,
                repo_type="dataset",
                exist_ok=True
            )
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Push to hub
        commit_message = f"LlamaSniffer {data_type} backup - {timestamp}"
        if metadata and "prompt" in metadata:
            commit_message += f" | Prompt: {metadata['prompt'][:50]}..."
            
        dataset.push_to_hub(
            repo_id=hf_repo,
            commit_message=commit_message
        )
        
        backup_url = f"https://huggingface.co/datasets/{hf_repo}"
        
        return {
            "success": True,
            "repository": hf_repo,
            "url": backup_url,
            "records_uploaded": len(dataset_data),
            "timestamp": timestamp,
            "commit_message": commit_message
        }
        
    except Exception as e:
        return {"error": str(e)}


def backup_flock_nodes(hf_repo: str = None) -> Dict[str, Any]:
    """Backup current flock discovery data to Hugging Face.
    
    Args:
        hf_repo: HF repository (optional, auto-generated if not provided)
        
    Returns:
        Backup result
    """
    try:
        nodes = get_flock_nodes()
        
        return backup_flock_data(
            data=nodes["nodes"],
            data_type="flock_discovery",
            hf_repo=hf_repo,
            metadata={
                "total_nodes": nodes["total_nodes"],
                "active_nodes": nodes["active_nodes"],
                "discovery_methods": nodes["discovery_summary"]
            }
        )
        
    except Exception as e:
        return {"error": str(e)}


def backup_flock_health(hf_repo: str = None) -> Dict[str, Any]:
    """Backup current flock health and performance data to Hugging Face.
    
    Args:
        hf_repo: HF repository (optional, auto-generated if not provided)
        
    Returns:
        Backup result
    """
    try:
        flock_status = get_flock_status()
        
        # Convert to list format for dataset
        health_data = [{
            "flock_health_percentage": flock_status["flock_health"]["health_percentage"],
            "total_instances": flock_status["flock_health"]["total_instances"],
            "healthy_instances": flock_status["flock_health"]["healthy_instances"],
            "unique_models": flock_status["model_availability"]["unique_models"],
            "model_distribution": flock_status["model_availability"]["distribution"],
            "performance_stats": flock_status["performance_stats"]
        }]
        
        return backup_flock_data(
            data=health_data,
            data_type="flock_health",
            hf_repo=hf_repo,
            metadata={
                "backup_type": "health_snapshot",
                "models_available": flock_status["model_availability"]["models"]
            }
        )
        
    except Exception as e:
        return {"error": str(e)}


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
        # Flock-based methods
        self.get_flock_status = get_flock_status
        self.get_flock_nodes = get_flock_nodes
        self.refresh_flock_health = refresh_flock_health
        self.parallel_generate_flock = parallel_generate_flock
        # Hugging Face integration
        self.setup_hf_auth = setup_hf_auth
        self.backup_flock_data = backup_flock_data
        self.backup_flock_nodes = backup_flock_nodes
        self.backup_flock_health = backup_flock_health
        # Backward compatibility aliases
        self.get_cluster_status = get_cluster_status
        self.get_cached_endpoints = get_cached_endpoints
        self.refresh_endpoint_health = refresh_endpoint_health
        self.Client = OllamaClient
        self.AsyncClient = OllamaClient  # Same class handles both sync/async
        self.Config = OllamaConfig


# Create the module instance
ollama = OllamaModule()
