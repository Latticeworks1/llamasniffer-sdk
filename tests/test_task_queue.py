import asyncio

from llamasniffer.tasks import ParallelTaskQueue, TaskPriority
from .conftest import FakeManager


def test_queue_submits_and_collects_results():
    async def _run():
        manager = FakeManager()
        queue = ParallelTaskQueue(manager=manager, max_workers=1)
        await queue.start()

        task_id = await queue.submit("demo", [{"role": "user", "content": "hello"}])
        result = await queue.wait_for_task(task_id)
        assert result["status"] == "completed"
        assert result["result"]["result"] == "demo:hello"

        stats = queue.get_stats()
        assert stats["tasks"]["completed"] == 1

        await queue.stop()

    asyncio.run(_run())


def test_queue_retries_and_preserves_order():
    async def _run():
        manager = FakeManager(failures_before_success=1)
        queue = ParallelTaskQueue(manager=manager, max_workers=2, default_max_retries=2)
        await queue.start()

        batch = [
            {"model": "a", "messages": [{"role": "user", "content": "first"}]},
            {"model": "b", "messages": [{"role": "user", "content": "second"}]},
        ]
        task_ids = await queue.submit_batch(batch, priority=TaskPriority.HIGH)
        results = await queue.wait_for_batch(task_ids)

        assert all(result["status"] == "completed" for result in results)
        assert results[0]["result"]["result"].endswith("first")
        stats = queue.get_stats()
        assert stats["tasks"]["retries"] >= 1

        await queue.stop()

    asyncio.run(_run())


def test_task_failure_after_max_retries():
    async def _run():
        manager = FakeManager(failures_before_success=10)
        queue = ParallelTaskQueue(manager=manager, max_workers=1, default_max_retries=1)
        await queue.start()

        task_id = await queue.submit("demo", [{"role": "user", "content": "hello"}])
        result = await queue.wait_for_task(task_id, timeout=2)

        assert result["status"] == "failed"
        assert "error" in result
        stats = queue.get_stats()
        assert stats["tasks"]["failed"] == 1

        await queue.stop()

    asyncio.run(_run())


def test_process_dataset_converts_records_and_batches():
    async def _run():
        manager = FakeManager()
        queue = ParallelTaskQueue(manager=manager, max_workers=2)
        await queue.start()

        dataset = [
            {"text": "hello world"},
            {"messages": [{"role": "user", "content": "custom"}], "model": "custom-model"},
        ]

        results = await queue.process_dataset(dataset, model="default-model", batch_size=1)

        assert len(results) == 2
        assert results[0]["status"] == "completed"
        assert manager.calls[0]["model"] == "default-model"
        assert manager.calls[0]["prompt"] == "hello world"
        assert manager.calls[1]["model"] == "custom-model"

        await queue.stop()

    asyncio.run(_run())
