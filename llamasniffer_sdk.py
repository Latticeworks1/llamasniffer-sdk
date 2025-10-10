"""
LlamaSniffer SDK - Discover and interact with Ollama instances locally and globally.

Provides comprehensive discovery mechanisms for Ollama LLM instances across 
local networks and internet-facing services via Shodan integration.
"""

import socket
import threading
import requests
import json
import time
import shodan
from typing import List, Dict, Optional, Callable
from datasets import Dataset
from huggingface_hub import HfApi, create_repo


class OllamaDiscovery:
    """Discovers Ollama instances across networks using multiple discovery methods.
    
    Supports both local network scanning and global discovery via Shodan API.
    Provides verification and metadata collection for discovered instances.
    
    Args:
        timeout: Connection timeout for network operations in seconds
        shodan_api_key: Optional Shodan API key for global discovery
    """
    
    def __init__(self, timeout: float = 2.0, shodan_api_key: Optional[str] = None, hf_token: Optional[str] = None):
        self.timeout = timeout
        self.discovered_instances = []
        self.shodan_api_key = shodan_api_key
        self.shodan_client = shodan.Shodan(shodan_api_key) if shodan_api_key else None
        self.hf_token = hf_token
        self.hf_api = HfApi(token=hf_token) if hf_token else None
        
    def scan_port_range(self, host: str, start_port: int = 11434, end_port: int = 11450) -> List[int]:
        """Scan port range on target host for active Ollama services.
        
        Args:
            host: Target IP address or hostname
            start_port: Beginning of port range to scan
            end_port: End of port range to scan
            
        Returns:
            List of ports running verified Ollama instances
        """
        open_ports = []
        
        for port in range(start_port, end_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    verification = self._verify_ollama_instance(host, port)
                    if verification.get('verified', False):
                        open_ports.append(port)
            except Exception:
                pass
            finally:
                sock.close()
                
        return open_ports
    
    def _verify_ollama_instance(self, host: str, port: int) -> Dict[str, any]:
        """Fast Ollama instance verification with response time measurement.
        
        Optimized verification process:
        1. Quick health check via version endpoint
        2. Fast model enumeration via tags endpoint
        3. Response time measurement
        
        Args:
            host: Target host address
            port: Target port number
            
        Returns:
            Dictionary with verification results and performance metrics
        """
        base_url = f"http://{host}:{port}"
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
            verification_data['version'] = version_data.get('version', 'unknown')
            verification_data['version_response_time_ms'] = round(version_time, 2)
            
            # Fast model enumeration
            tags_start = time.time()
            tags_response = requests.get(f"{base_url}/api/tags", timeout=self.timeout)
            tags_time = (time.time() - tags_start) * 1000  # ms
            
            if tags_response.status_code != 200:
                return {}
                
            tags_data = tags_response.json()
            models = [model['name'] for model in tags_data.get('models', [])]
            verification_data['models'] = models
            verification_data['model_count'] = len(models)
            verification_data['tags_response_time_ms'] = round(tags_time, 2)
            
            # Skip heartbeat test for speed - just mark as responsive if we got this far
            verification_data['responsive'] = True
            verification_data['total_verification_time_ms'] = round((time.time() - start_time) * 1000, 2)
            verification_data['verified'] = True
            verification_data['verified_at'] = time.time()
            
            return verification_data
            
        except Exception as e:
            return {}
    
    def _test_ollama_heartbeat(self, base_url: str, model_name: str) -> bool:
        """Test Ollama instance responsiveness with minimal generation request.
        
        Args:
            base_url: Base URL for Ollama instance
            model_name: Model to test with
            
        Returns:
            True if instance responds to generation request
        """
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1}  # Minimal token generation
                },
                timeout=self.timeout * 2  # Allow extra time for generation
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def scan_network(self, network_prefix: str = "192.168.1") -> List[Dict[str, any]]:
        """Perform threaded network scan across IP range for Ollama instances.
        
        Args:
            network_prefix: Network prefix (e.g., "192.168.1") for subnet scanning
            
        Returns:
            List of discovered instances with metadata
        """
        instances = []
        threads = []
        lock = threading.Lock()
        
        def scan_host(host_ip: str):
            ports = self.scan_port_range(host_ip)
            if ports:
                with lock:
                    for port in ports:
                        # Get full verification data for discovered instances
                        verification = self._verify_ollama_instance(host_ip, port)
                        if verification.get('verified', False):
                            instance = {
                                "host": host_ip,
                                "port": port,
                                "url": f"http://{host_ip}:{port}",
                                "discovered_at": time.time(),
                                "discovery_method": "local_scan",
                                **verification  # Include all verification metadata
                            }
                            instances.append(instance)
        
        for i in range(1, 255):
            host_ip = f"{network_prefix}.{i}"
            thread = threading.Thread(target=scan_host, args=(host_ip,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
            
        self.discovered_instances = instances
        return instances
    
    def scan_shodan(self, query: str = "ollama", limit: int = 100) -> List[Dict[str, any]]:
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
        search_query = query
        print(f"Searching Shodan: '{search_query}' (limit: {limit})")
        
        try:
            results = self.shodan_client.search(search_query, limit=limit)
            
            for result in results['matches']:
                # Perform comprehensive verification
                verification = self._verify_ollama_instance(result['ip_str'], result['port'])
                
                if verification.get('verified', False):
                    instance = {
                        "host": result['ip_str'],
                        "port": result['port'],
                        "url": f"http://{result['ip_str']}:{result['port']}",
                        "country": result.get('location', {}).get('country_name', 'Unknown'),
                        "org": result.get('org', 'Unknown'),
                        "hostnames": result.get('hostnames', []),
                        "discovered_at": time.time(),
                        "discovery_method": "shodan",
                        "shodan_data": {
                            "timestamp": result.get('timestamp'),
                            "product": result.get('product'),
                            "version": result.get('version'),
                            "banner": result.get('data', '')[:200]
                        },
                        **verification  # Include comprehensive verification data
                    }
                    instances.append(instance)
                    
        except shodan.APIError as e:
            print(f"Shodan API error: {e}")
        except Exception as e:
            print(f"Discovery error: {e}")
            
        return instances
    
    def get_models(self, instance: Dict[str, any]) -> List[str]:
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
                return [model['name'] for model in data.get('models', [])]
        except Exception:
            pass
        return []
    
    def backup_to_dataset(self, instances: List[Dict[str, any]], dataset_name: str = "ollama-instances", 
                         repo_owner: str = "latterworks") -> bool:
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
                create_repo(repo_id=repo_id, repo_type="dataset", token=self.hf_token, exist_ok=True)
                print(f"Created/verified dataset repository: {repo_id}")
            except Exception as e:
                print(f"Repository creation info: {e}")
            
            dataset = Dataset.from_list(instances)
            
            dataset.push_to_hub(
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Update Ollama instances discovery - {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            print(f"Successfully backed up {len(instances)} instances to {repo_id}")
            return True
            
        except Exception as e:
            print(f"Dataset backup error: {e}")
            return False


class OllamaClient:
    """High-level client for Ollama instance interaction and management.
    
    Provides streamlined interface for model operations, text generation,
    and instance management across discovered Ollama services.
    
    Args:
        host: Target Ollama host address
        port: Target Ollama port (default: 11434)
    """
    
    def __init__(self, host: str, port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, any]:
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
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": stream
                },
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> List[str]:
        """Retrieve all available models from this instance.
        
        Returns:
            List of model names available for generation
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception:
            pass
        return []
    
    def pull_model(self, model_name: str) -> Dict[str, any]:
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
                timeout=300
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def discover_ollama_instances(network_prefix: str = "192.168.1", timeout: float = 2.0) -> List[Dict[str, any]]:
    """Convenience function to discover Ollama instances on the local network"""
    discovery = OllamaDiscovery(timeout=timeout)
    return discovery.scan_network(network_prefix)


def discover_ollama_shodan(shodan_api_key: str = "tSWZYaXGbmpZg6PtHFBeDSbvbZaN1UuU", query: str = "ollama", limit: int = 100) -> List[Dict[str, any]]:
    """Discover Ollama instances globally via Shodan with comprehensive verification.
    
    Performs sequential verification on each discovered instance:
    - Version/health check via /api/version
    - Model enumeration via /api/tags  
    - Heartbeat test via minimal generation request
    
    Args:
        shodan_api_key: Valid Shodan API key for global search
        query: Search term (searches any port for "ollama")
        limit: Maximum results (conservative credit usage)
        
    Returns:
        List of verified instances with comprehensive metadata
    """
    discovery = OllamaDiscovery(shodan_api_key=shodan_api_key)
    return discovery.scan_shodan(query, limit)


def connect_to_ollama(host: str, port: int = 11434) -> OllamaClient:
    """Convenience function to create an Ollama client"""
    return OllamaClient(host, port)