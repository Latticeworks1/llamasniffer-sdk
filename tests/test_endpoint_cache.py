"""Tests for endpoint caching functionality."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from llamasniffer.endpoint_cache import (
    EndpointCache,
    get_cached_endpoints,
    cache_endpoints,
    get_cache_info,
    clear_cache,
)


@pytest.fixture
def temp_cache_path():
    """Create temporary cache file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_endpoints():
    """Sample endpoint data for testing."""
    return [
        {
            'host': '192.168.1.100',
            'port': 11434,
            'url': 'http://192.168.1.100:11434',
            'models': ['llama2', 'mistral'],
            'verified': True,
            'verified_at': time.time(),
            'total_verification_time_ms': 150.5
        },
        {
            'host': '192.168.1.101',
            'port': 11434,
            'url': 'http://192.168.1.101:11434',
            'models': ['codellama'],
            'verified': True,
            'verified_at': time.time(),
            'total_verification_time_ms': 200.3
        }
    ]


def test_endpoint_cache_save_and_load(temp_cache_path, sample_endpoints):
    """Test saving and loading endpoints from cache."""
    cache = EndpointCache(cache_path=temp_cache_path, ttl_hours=24)

    # Save endpoints
    cache.save(sample_endpoints, metadata={'source': 'test'})

    # Load endpoints
    cache_data = cache.load(use_memory_cache=False)

    assert cache_data is not None
    assert len(cache_data['endpoints']) == 2
    assert cache_data['count'] == 2
    assert cache_data['metadata']['source'] == 'test'
    assert 'cached_at' in cache_data
    assert 'cached_at_human' in cache_data


def test_endpoint_cache_memory_caching(temp_cache_path, sample_endpoints):
    """Test in-memory caching functionality."""
    cache = EndpointCache(cache_path=temp_cache_path, memory_cache_ttl=5)

    # Save and load with memory cache
    cache.save(sample_endpoints)

    # First load - from disk
    data1 = cache.load(use_memory_cache=False)
    assert cache._memory_cache is not None

    # Second load - from memory (should be instant)
    data2 = cache.load(use_memory_cache=True)
    assert data1 == data2

    # Verify memory cache is used
    assert cache._is_memory_cache_valid()


def test_endpoint_cache_expiry(temp_cache_path, sample_endpoints):
    """Test cache expiry based on TTL."""
    cache = EndpointCache(cache_path=temp_cache_path, ttl_hours=0)  # Expire immediately

    cache.save(sample_endpoints)

    # Cache should be expired
    endpoints = cache.get_endpoints()
    assert endpoints is None


def test_endpoint_cache_health_filtering(temp_cache_path):
    """Test filtering endpoints based on health scores."""
    endpoints = [
        {
            'url': 'http://good.com:11434',
            'models': ['llama2'],
            'total_verification_time_ms': 100,
            'verified_at': time.time()
        },
        {
            'url': 'http://slow.com:11434',
            'models': [],  # No models = unhealthy
            'total_verification_time_ms': 5000,
            'verified_at': time.time() - (100 * 3600)  # 100 hours ago
        }
    ]

    cache = EndpointCache(cache_path=temp_cache_path, min_health_threshold=0.7)
    cache.save(endpoints)

    # Get with health filtering
    healthy = cache.get_endpoints(filter_unhealthy=True)

    # Only the good endpoint should remain
    assert len(healthy) == 1
    assert healthy[0]['url'] == 'http://good.com:11434'


def test_endpoint_cache_update_health(temp_cache_path, sample_endpoints):
    """Test updating health status of specific endpoint."""
    cache = EndpointCache(cache_path=temp_cache_path)
    cache.save(sample_endpoints)

    # Update health
    cache.update_health('http://192.168.1.100:11434', is_healthy=False)

    # Load and check
    data = cache.load()
    endpoint = next(e for e in data['endpoints'] if e['url'] == 'http://192.168.1.100:11434')

    assert endpoint['is_healthy'] is False
    assert endpoint['consecutive_failures'] == 1


def test_endpoint_cache_remove_endpoint(temp_cache_path, sample_endpoints):
    """Test removing specific endpoint from cache."""
    cache = EndpointCache(cache_path=temp_cache_path)
    cache.save(sample_endpoints)

    # Remove one endpoint
    cache.remove_endpoint('http://192.168.1.100:11434')

    # Check only one remains
    data = cache.load()
    assert len(data['endpoints']) == 1
    assert data['endpoints'][0]['url'] == 'http://192.168.1.101:11434'


def test_endpoint_cache_info(temp_cache_path, sample_endpoints):
    """Test getting cache information."""
    cache = EndpointCache(cache_path=temp_cache_path, ttl_hours=24)
    cache.save(sample_endpoints)

    info = cache.get_cache_info()

    assert info['exists'] is True
    assert info['valid'] is True
    assert info['count'] == 2
    assert info['ttl_hours'] == 24
    assert info['age_hours'] >= 0
    assert 'expires_in_hours' in info


def test_endpoint_cache_clear(temp_cache_path, sample_endpoints):
    """Test clearing the cache."""
    cache = EndpointCache(cache_path=temp_cache_path)
    cache.save(sample_endpoints)

    # Verify cache exists
    assert os.path.exists(temp_cache_path)
    assert cache._memory_cache is not None

    # Clear cache
    cache.clear()

    # Verify cache is gone
    assert not os.path.exists(temp_cache_path)
    assert cache._memory_cache is None
    assert cache._memory_cache_loaded_at == 0


def test_convenience_functions(temp_cache_path, sample_endpoints):
    """Test convenience wrapper functions."""
    # Use default cache location for this test
    cache_endpoints(sample_endpoints, ttl_hours=24)

    # Get cached endpoints
    cached = get_cached_endpoints(ttl_hours=24)
    assert cached is not None
    assert len(cached) == 2

    # Get cache info
    info = get_cache_info()
    assert info['exists'] is True

    # Clear cache
    clear_cache()
    info = get_cache_info()
    assert info['exists'] is False


def test_invalid_cache_structure(temp_cache_path):
    """Test handling of invalid cache file structure."""
    # Write invalid JSON
    with open(temp_cache_path, 'w') as f:
        json.dump({'invalid': 'structure'}, f)

    cache = EndpointCache(cache_path=temp_cache_path)
    data = cache.load()

    assert data is None


def test_memory_cache_ttl_expiry(temp_cache_path, sample_endpoints):
    """Test memory cache TTL expiry."""
    cache = EndpointCache(cache_path=temp_cache_path, memory_cache_ttl=1)
    cache.save(sample_endpoints)

    # Load into memory
    cache.load()
    assert cache._is_memory_cache_valid()

    # Wait for TTL to expire
    time.sleep(1.1)
    assert not cache._is_memory_cache_valid()


def test_cache_with_no_file(temp_cache_path):
    """Test cache behavior when file doesn't exist."""
    cache = EndpointCache(cache_path=temp_cache_path)

    # Try to load non-existent cache
    data = cache.load()
    assert data is None

    # Get endpoints should return None
    endpoints = cache.get_endpoints()
    assert endpoints is None

    # Info should reflect cache doesn't exist
    info = cache.get_cache_info()
    assert info['exists'] is False
    assert info['valid'] is False
