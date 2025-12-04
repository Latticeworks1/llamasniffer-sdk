import asyncio

from llamasniffer.dataset_forge import (
    DatasetForge,
    DatasetConfig,
    DatasetType,
    QualityLevel,
)
from llamasniffer.tasks import TaskPriority
from .conftest import FakeQueue


def _make_config():
    return DatasetConfig(
        dataset_type=DatasetType.QA_PAIRS,
        target_size=2,
        quality_level=QualityLevel.BASIC,
        models=["qa-model"],
        diversity_models=[],
        validation_model="validator",
        batch_size=1,
        max_retries=1,
        require_consensus=False,
        deduplicate=False,
    )


def test_dataset_forge_generates_dataset_with_fake_queue():
    async def _run():
        responses = [
            [{"status": "completed", "result": {"response": "Q: Sky? A: Blue"}}],
            [{"status": "completed", "result": {"response": "Q: Grass? A: Green"}}],
        ]
        queue = FakeQueue(responses)
        forge = DatasetForge(_make_config(), queue=queue)

        dataset = await forge.forge_dataset()

        assert dataset["metadata"]["size"] == 2
        assert len(dataset["data"]) == 2
        assert queue.started is True  # external queues remain running
        assert forge.stats["total_generated"] == 2
        assert all("question" in row["content"] for row in dataset["data"])

        await queue.stop()

    asyncio.run(_run())


def test_parse_basic_content_handles_missing_format():
    config = _make_config()
    forge = DatasetForge(config, queue=FakeQueue([]))
    parsed = forge._parse_basic_content("Just text", DatasetType.QA_PAIRS)
    assert parsed["content"] == "Just text"

    qa_parsed = forge._parse_basic_content("Q: Who? A: Me", DatasetType.QA_PAIRS)
    assert qa_parsed["question"] == "Who?"
    assert qa_parsed["answer"] == "Me"
