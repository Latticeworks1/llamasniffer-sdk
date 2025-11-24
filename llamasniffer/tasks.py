"""
Parallel task queue for distributed Ollama inference.

Offers a single, well-documented entry point (`ParallelTaskQueue`) for running large
volumes of chat/generate style requests across the global LlamaSniffer fleet with
priority-aware scheduling, retries, and real-time statistics.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .core import DistributedOllamaManager


class TaskPriority(Enum):
    """Priority levels for tasks submitted to the queue."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Lifecycle states for tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a distributed inference task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    assigned_instance: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout: float = 30.0

    def duration(self) -> Optional[float]:
        """Return total runtime if the task has completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class ParallelTaskQueue:
    """Distributed parallel task processing across remote Ollama instances."""

    def __init__(
        self,
        instances: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = 10,
        strategy: str = "fastest",
        *,
        manager: Optional["DistributedOllamaManager"] = None,
        config_path: Optional[str] = None,
        enable_semantic_matching: bool = True,
        default_timeout: float = 30.0,
        default_max_retries: int = 2,
    ):
        self.max_workers = max(1, max_workers)
        self.strategy = strategy
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries
        self.manager = manager or self._build_manager(
            instances=instances,
            strategy=strategy,
            config_path=config_path,
            enable_semantic_matching=enable_semantic_matching,
        )

        self.priority_queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque(),
        }

        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self._task_futures: Dict[str, asyncio.Future] = {}

        self._worker_tasks: List[asyncio.Task] = []
        self._new_task_event: Optional[asyncio.Event] = None
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._current_running = 0
        self._peak_running = 0
        self._start_time: Optional[float] = None

        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retries": 0,
            "total_latency": 0.0,
        }

    def _build_manager(
        self,
        *,
        instances: Optional[List[Dict[str, Any]]],
        strategy: str,
        config_path: Optional[str],
        enable_semantic_matching: bool,
    ) -> "DistributedOllamaManager":
        """Create a distributed manager if one was not provided."""
        from .core import DistributedOllamaManager
        if instances:
            return DistributedOllamaManager(
                instances=instances,
                strategy=strategy,
                enable_semantic_matching=enable_semantic_matching,
            )

        from .ollama import OllamaClient, OllamaConfig

        config = OllamaConfig(config_path)
        if strategy:
            config.set_load_balancing(strategy)

        client = OllamaClient(config)
        return client._get_manager()

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._new_task_event = asyncio.Event()
        self._running = True
        self._start_time = time.time()

        self._worker_tasks = []
        for idx in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{idx}"))
            self._worker_tasks.append(worker)

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        if not self._running:
            return

        self._running = False
        if self._new_task_event:
            self._new_task_event.set()

        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def submit(
        self,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """Submit a single task."""
        if not self._running or not self._loop:
            raise RuntimeError("ParallelTaskQueue must be started before submitting tasks")

        resolved_priority = self._normalize_priority(priority)
        loop = asyncio.get_running_loop()

        task = Task(
            model=model,
            messages=messages or [],
            priority=resolved_priority,
            timeout=timeout or self.default_timeout,
            max_retries=self.default_max_retries if max_retries is None else max_retries,
        )

        self.active_tasks[task.id] = task
        self.priority_queues[resolved_priority].append(task)
        self._metrics["submitted"] += 1

        future = loop.create_future()
        self._task_futures[task.id] = future

        if self._new_task_event:
            self._new_task_event.set()

        return task.id

    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[str]:
        """Submit multiple tasks for parallel processing."""
        task_ids = []
        for task_spec in tasks:
            task_id = await self.submit(
                model=task_spec["model"],
                messages=task_spec.get("messages", []),
                priority=task_spec.get("priority", priority),
                timeout=task_spec.get("timeout"),
                max_retries=task_spec.get("max_retries"),
            )
            task_ids.append(task_id)
        return task_ids

    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for a specific task to complete."""
        future = self._task_futures.get(task_id)
        if not future:
            raise KeyError(f"Unknown task id '{task_id}'")

        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            return {"status": "timeout", "error": "Task timed out", "task_id": task_id}
        return result

    async def wait_for_batch(
        self,
        task_ids: List[str],
        timeout: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """Wait for all tasks in batch to complete."""
        awaitables = [self.wait_for_task(task_id, timeout) for task_id in task_ids]
        return await asyncio.gather(*awaitables, return_exceptions=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics and performance metrics."""
        uptime = (time.time() - self._start_time) if self._start_time else 0.0
        avg_latency = (
            self._metrics["total_latency"] / self._metrics["completed"]
            if self._metrics["completed"]
            else 0.0
        )

        queue_lengths = {
            priority.name: len(queue) for priority, queue in self.priority_queues.items()
        }

        return {
            "uptime": round(uptime, 2),
            "workers": {
                "configured": self.max_workers,
                "active": sum(1 for t in self._worker_tasks if not t.done()),
                "current_running": self._current_running,
                "peak_concurrent": self._peak_running,
            },
            "tasks": {
                "submitted": self._metrics["submitted"],
                "completed": self._metrics["completed"],
                "failed": self._metrics["failed"],
                "retries": self._metrics["retries"],
                "active": len(self.active_tasks),
                "success_rate": (
                    (self._metrics["completed"] / max(1, self._metrics["submitted"])) * 100
                ),
            },
            "queue": {
                "per_priority": queue_lengths,
                "total": sum(queue_lengths.values()),
            },
            "performance": {
                "avg_latency": round(avg_latency, 2),
                "total_latency": round(self._metrics["total_latency"], 2),
            },
        }

    async def process_dataset(
        self,
        dataset: List[Dict[str, Any]],
        *,
        model: str,
        batch_size: int = 100,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[Dict[str, Any]]:
        """Process entire dataset in parallel."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        results: List[Dict[str, Any]] = []
        priority = self._normalize_priority(priority)

        for start_idx in range(0, len(dataset), batch_size):
            batch_records = dataset[start_idx : start_idx + batch_size]
            if not batch_records:
                continue

            task_specs = [
                {
                    "model": record.get("model", model),
                    "messages": self._record_to_messages(record),
                    "priority": record.get("priority", priority),
                    "timeout": record.get("timeout"),
                    "max_retries": record.get("max_retries"),
                }
                for record in batch_records
            ]

            task_ids = await self.submit_batch(task_specs, priority=priority)
            batch_results = await self.wait_for_batch(task_ids, timeout=self.default_timeout * 4)
            results.extend(batch_results)

        return results

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker loop that pulls work from the priority queues."""
        while self._running or self._has_pending_work():
            task = await self._next_task()
            if not task:
                continue

            await self._execute_task(task, worker_id)

    async def _next_task(self) -> Optional[Task]:
        """Fetch the next available task, waiting until one is available."""
        task = self._pop_next_task()
        if task:
            return task

        if not self._new_task_event:
            await asyncio.sleep(0.05)
            return None

        while True:
            task = self._pop_next_task()
            if task:
                return task

            if not self._running and not self._has_pending_work():
                return None

            self._new_task_event.clear()
            await self._new_task_event.wait()

    def _pop_next_task(self) -> Optional[Task]:
        for priority in (
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
        ):
            queue = self.priority_queues[priority]
            if queue:
                return queue.popleft()
        return None

    def _has_pending_work(self) -> bool:
        """Return True if queued or in-flight tasks remain."""
        return any(self.priority_queues[priority] for priority in self.priority_queues)

    async def _execute_task(self, task: Task, worker_id: str) -> None:
        """Execute a task and handle retries."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._current_running += 1
        self._peak_running = max(self._peak_running, self._current_running)

        prompt = self._messages_to_prompt(task.messages)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.manager.generate_distributed,
                    task.model,
                    prompt,
                    0,
                    1,
                ),
                timeout=task.timeout,
            )

            if not response or response.get("error"):
                raise RuntimeError(response.get("error", "Unknown inference error"))

            task.assigned_instance = (
                response.get("execution_metadata", {}).get("instance")
                if isinstance(response, dict)
                else None
            )
            task.result = response
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

            self._finalize_task(task)
        except asyncio.TimeoutError:
            await self._handle_task_failure(task, "Task timed out")
        except Exception as exc:
            await self._handle_task_failure(task, str(exc))
        finally:
            self._current_running = max(0, self._current_running - 1)

    async def _handle_task_failure(self, task: Task, error: str) -> None:
        """Retry or fail a task based on retry policy."""
        task.error = error
        task.retry_count += 1

        if task.retry_count <= task.max_retries:
            task.status = TaskStatus.PENDING
            task.started_at = None
            self._metrics["retries"] += 1
            self.priority_queues[task.priority].append(task)
            if self._new_task_event:
                self._new_task_event.set()
            return

        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        self._finalize_task(task, failed=True)

    def _finalize_task(self, task: Task, failed: bool = False) -> None:
        """Move task to completed/failed tracking and resolve waiters."""
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]

        payload = {
            "status": task.status.value,
            "task_id": task.id,
            "result": task.result,
            "error": task.error,
            "duration": task.duration(),
            "instance": task.assigned_instance,
        }

        if failed:
            self.failed_tasks[task.id] = task
            self._metrics["failed"] += 1
        else:
            self.completed_tasks[task.id] = task
            self._metrics["completed"] += 1
            if task.duration():
                self._metrics["total_latency"] += task.duration()

        future = self._task_futures.get(task.id)
        if future and not future.done():
            future.set_result(payload)

    def _normalize_priority(self, priority: TaskPriority) -> TaskPriority:
        if isinstance(priority, TaskPriority):
            return priority
        if isinstance(priority, str):
            normalized = priority.upper()
            return TaskPriority[normalized]
        raise ValueError(f"Unsupported priority type: {priority}")

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat-style messages into a single prompt string."""
        if not messages:
            return ""
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)

    def _record_to_messages(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dataset records to chat message format."""
        if "messages" in record and isinstance(record["messages"], list):
            return record["messages"]

        text = (
            record.get("prompt")
            or record.get("text")
            or record.get("content")
            or json.dumps(record)
        )
        return [{"role": "user", "content": text}]
