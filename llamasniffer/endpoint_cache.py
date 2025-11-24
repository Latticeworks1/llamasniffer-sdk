"""
Endpoint caching for discovered Ollama instances.

Stores discovered endpoints locally to avoid repeated Shodan scans.
Implements periodic refresh and health-based expiry.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class EndpointCache:
    """Local cache for discovered Ollama endpoints.

    Features:
    - Stores endpoints in JSON file
    - Automatic expiry after configurable TTL
    - Health-based filtering (removes dead endpoints)
    - Periodic refresh tracking
    """

    def __init__(self,
                 cache_path: Optional[str] = None,
                 ttl_hours: int = 24,
                 min_health_threshold: float = 0.7):
        """Initialize endpoint cache.

        Args:
            cache_path: Path to cache file (default: ~/.llamasniffer/endpoint_cache.json)
            ttl_hours: Time-to-live in hours before re-scan needed
            min_health_threshold: Minimum health score to keep endpoint (0.0-1.0)
        """
        self.cache_path = cache_path or self._default_cache_path()
        self.ttl_hours = ttl_hours
        self.min_health_threshold = min_health_threshold
        self._ensure_cache_dir()

    def _default_cache_path(self) -> str:
        """Get default cache file path."""
        cache_dir = Path.home() / ".llamasniffer"
        return str(cache_dir / "endpoint_cache.json")

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        cache_dir = Path(self.cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[Dict]:
        """Load cached endpoints from disk.

        Returns:
            Cache data dict or None if cache doesn't exist/is invalid
        """
        if not os.path.exists(self.cache_path):
            return None

        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'endpoints' not in cache_data:
                return None

            return cache_data
        except (json.JSONDecodeError, IOError):
            return None

    def save(self, endpoints: List[Dict], metadata: Optional[Dict] = None):
        """Save endpoints to cache.

        Args:
            endpoints: List of discovered endpoint dicts
            metadata: Optional metadata about discovery (query, limit, etc.)
        """
        cache_data = {
            'endpoints': endpoints,
            'cached_at': time.time(),
            'cached_at_human': datetime.now().isoformat(),
            'ttl_hours': self.ttl_hours,
            'count': len(endpoints),
            'metadata': metadata or {}
        }

        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def is_valid(self, cache_data: Optional[Dict] = None) -> bool:
        """Check if cache is still valid (not expired).

        Args:
            cache_data: Cache data dict (loads from disk if None)

        Returns:
            True if cache is valid and not expired
        """
        if cache_data is None:
            cache_data = self.load()

        if not cache_data:
            return False

        # Check if cache has expired
        cached_at = cache_data.get('cached_at', 0)
        age_hours = (time.time() - cached_at) / 3600

        return age_hours < self.ttl_hours

    def get_endpoints(self,
                     max_age_hours: Optional[int] = None,
                     filter_unhealthy: bool = True) -> Optional[List[Dict]]:
        """Get cached endpoints if valid.

        Args:
            max_age_hours: Override TTL for this request
            filter_unhealthy: Remove endpoints below health threshold

        Returns:
            List of endpoints or None if cache invalid/expired
        """
        cache_data = self.load()

        if not cache_data:
            return None

        # Check expiry
        ttl = max_age_hours or self.ttl_hours
        cached_at = cache_data.get('cached_at', 0)
        age_hours = (time.time() - cached_at) / 3600

        if age_hours >= ttl:
            return None

        endpoints = cache_data.get('endpoints', [])

        # Filter unhealthy endpoints if requested
        if filter_unhealthy and endpoints:
            endpoints = self._filter_healthy(endpoints)

        return endpoints

    def _filter_healthy(self, endpoints: List[Dict]) -> List[Dict]:
        """Filter out unhealthy endpoints based on health scores.

        Args:
            endpoints: List of endpoint dicts

        Returns:
            Filtered list of healthy endpoints
        """
        healthy = []
        for ep in endpoints:
            # Calculate health score based on:
            # - Has models (0.5 weight)
            # - Response time < 2000ms (0.3 weight)
            # - Recently verified (0.2 weight)

            has_models = len(ep.get('models', [])) > 0
            response_time = ep.get('total_verification_time_ms', 9999)
            verified_at = ep.get('verified_at', 0)
            age_hours = (time.time() - verified_at) / 3600

            health_score = 0.0
            if has_models:
                health_score += 0.5
            if response_time < 2000:
                health_score += 0.3
            if age_hours < 48:  # Verified within last 48 hours
                health_score += 0.2

            if health_score >= self.min_health_threshold:
                ep['health_score'] = health_score
                healthy.append(ep)

        return healthy

    def update_health(self, endpoint_url: str, is_healthy: bool):
        """Update health status of a specific endpoint.

        Args:
            endpoint_url: URL of endpoint to update
            is_healthy: Whether endpoint is currently healthy
        """
        cache_data = self.load()
        if not cache_data:
            return

        endpoints = cache_data.get('endpoints', [])
        updated = False

        for ep in endpoints:
            if ep.get('url') == endpoint_url:
                ep['last_health_check'] = time.time()
                ep['is_healthy'] = is_healthy
                if not is_healthy:
                    ep['consecutive_failures'] = ep.get('consecutive_failures', 0) + 1
                else:
                    ep['consecutive_failures'] = 0
                updated = True
                break

        if updated:
            self.save(endpoints, cache_data.get('metadata'))

    def remove_endpoint(self, endpoint_url: str):
        """Remove a specific endpoint from cache.

        Args:
            endpoint_url: URL of endpoint to remove
        """
        cache_data = self.load()
        if not cache_data:
            return

        endpoints = cache_data.get('endpoints', [])
        filtered = [ep for ep in endpoints if ep.get('url') != endpoint_url]

        if len(filtered) != len(endpoints):
            self.save(filtered, cache_data.get('metadata'))

    def get_cache_info(self) -> Dict:
        """Get information about the current cache state.

        Returns:
            Dict with cache statistics
        """
        cache_data = self.load()

        if not cache_data:
            return {
                'exists': False,
                'valid': False,
                'count': 0
            }

        cached_at = cache_data.get('cached_at', 0)
        age_hours = (time.time() - cached_at) / 3600

        return {
            'exists': True,
            'valid': self.is_valid(cache_data),
            'count': cache_data.get('count', 0),
            'cached_at': datetime.fromtimestamp(cached_at).isoformat(),
            'age_hours': round(age_hours, 2),
            'ttl_hours': self.ttl_hours,
            'expires_in_hours': max(0, self.ttl_hours - age_hours),
            'path': self.cache_path
        }

    def clear(self):
        """Clear the cache by deleting the cache file."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)


# Convenience functions

def get_cached_endpoints(ttl_hours: int = 24) -> Optional[List[Dict]]:
    """Get cached endpoints if available and valid.

    Args:
        ttl_hours: Maximum age of cache to accept

    Returns:
        List of cached endpoints or None if cache invalid
    """
    cache = EndpointCache(ttl_hours=ttl_hours)
    return cache.get_endpoints()


def cache_endpoints(endpoints: List[Dict], ttl_hours: int = 24):
    """Cache discovered endpoints.

    Args:
        endpoints: List of endpoint dicts to cache
        ttl_hours: Time-to-live for this cache
    """
    cache = EndpointCache(ttl_hours=ttl_hours)
    cache.save(endpoints, metadata={'source': 'manual_cache'})


def get_cache_info() -> Dict:
    """Get current cache status.

    Returns:
        Dict with cache information
    """
    cache = EndpointCache()
    return cache.get_cache_info()


def clear_cache():
    """Clear the endpoint cache."""
    cache = EndpointCache()
    cache.clear()
