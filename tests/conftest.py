"""Shared test fixtures for llamasniffer tests."""

import pytest
from typing import Dict, List


class FakeQueue:
    """Fake task queue for testing dataset generation."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.started = False
        self.submits = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def submit_batch(self, tasks, priority=None):
        from llamasniffer.tasks import TaskPriority
        self.submits.append((tasks, priority or TaskPriority.NORMAL))
        base = len(self.submits)
        return [f"task-{base}-{i}" for i in range(len(tasks))]

    async def wait_for_batch(self, task_ids):
        return self.responses.pop(0)


class FakeManager:
    """Fake distributed manager for testing task queue."""

    def __init__(self, *, failures_before_success: int = 0):
        self.strategy = "fastest"
        self.failures_before_success = failures_before_success
        self.calls: List[Dict[str, str]] = []

    def generate_distributed(self, model_query: str, prompt: str, *_, **__):
        self.calls.append({"model": model_query, "prompt": prompt})
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("simulated failure")
        return {
            "result": f"{model_query}:{prompt}",
            "execution_metadata": {"instance": "fake-node"},
        }


class StubDiscovery:
    """Stub discovery for testing distributed operations."""

    def __init__(self, instances):
        self.instances = instances
        self.search_called = False

    def search(self, query, limit):
        self.search_called = True
        return self.instances[:limit]


@pytest.fixture
def fake_queue():
    """Fixture providing a fake queue with successful responses."""
    responses = [
        [{"status": "completed", "result": {"response": "test response"}}]
    ]
    return FakeQueue(responses)


@pytest.fixture
def fake_manager():
    """Fixture providing a fake manager."""
    return FakeManager()


@pytest.fixture
def fake_manager_with_failures():
    """Fixture providing a fake manager that fails before succeeding."""
    return FakeManager(failures_before_success=1)


@pytest.fixture
def stub_instances():
    """Fixture providing stub Ollama instances."""
    return [
        {
            "host": "192.168.1.100",
            "port": 11434,
            "url": "http://192.168.1.100:11434",
            "models": ["llama3", "qwen"],
            "verified": True,
            "discovery_method": "shodan",
        },
        {
            "host": "192.168.1.101",
            "port": 11434,
            "url": "http://192.168.1.101:11434",
            "models": ["deepseek-r1"],
            "verified": True,
            "discovery_method": "manual",
        },
    ]


@pytest.fixture
def stub_discovery(stub_instances):
    """Fixture providing a stub discovery instance."""
    return StubDiscovery(stub_instances)
