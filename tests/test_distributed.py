import pytest

import llamasniffer
from llamasniffer.core import DistributedOllamaManager


class StubOllamaClient:
    """Minimal OllamaClient replacement for tests."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._models = ["demo-model"]

    def list_models(self):
        return self._models

    def generate(self, model, prompt, stream=False):
        return {"response": f"{model}:{prompt}", "instance": f"{self.host}:{self.port}"}


class StubDiscovery:
    def __init__(self, shodan_api_key: str, **_):
        self.key = shodan_api_key

    def search(self, query, limit):
        return [
            {
                "host": "1.2.3.4",
                "port": 11434,
                "response_time_ms": 50,
                "models": ["demo-model"],
                "discovery_method": "stub",
            }
        ]


@pytest.fixture(autouse=True)
def patch_ollama_client(monkeypatch):
    monkeypatch.setattr("llamasniffer.core.OllamaClient", StubOllamaClient)


def test_discover_remote_instances_uses_remote_discovery(monkeypatch):
    monkeypatch.setattr("llamasniffer.core.RemoteDiscovery", StubDiscovery)
    results = llamasniffer.discover_remote_instances("fake-key", query="ollama", limit=1)

    assert len(results) == 1
    assert results[0]["host"] == "1.2.3.4"


def test_create_distributed_manager_requires_instances():
    with pytest.raises(ValueError):
        llamasniffer.create_distributed_manager(instances=[], strategy="fastest")


def test_generate_distributed_returns_augmented_result():
    instances = [
        {"host": "localhost", "port": 11434, "models": ["demo-model"], "verified": True},
        {"host": "remote", "port": 11435, "models": ["demo-model"], "verified": True},
    ]

    manager = DistributedOllamaManager(
        instances=instances,
        strategy="fastest",
        enable_semantic_matching=False,
    )

    response = manager.generate_distributed("demo-model", "hello world", max_retries=0)

    assert "error" not in response
    assert response["response"] == "demo-model:hello world"
    assert response["execution_metadata"]["instance"] in {"localhost:11434", "remote:11435"}
