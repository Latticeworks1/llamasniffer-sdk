# Task Queue Redesign Proposal

## Current Problems

### 1. Confusing Names
- `FlockShepherd` - doesn't describe what it does (async task queue)
- `FlockHerder` - just a wrapper, shouldn't be separate class
- "Flock/Shepherd" metaphor obscures actual functionality

### 2. Over-Engineering
- Two classes when one would suffice
- FlockHerder is just a thin wrapper around FlockShepherd
- Adds unnecessary complexity

### 3. Unclear Purpose
Current code doesn't make it obvious this is for:
- Parallel dataset processing
- Distributed task queuing
- Load balancing across many models

## Proposed Redesign

### Single Class: `ParallelTaskQueue`

```python
class ParallelTaskQueue:
    """Distributed parallel task processing across remote Ollama instances.

    Use Cases:
    - Process large datasets in parallel (100k rows across 1000 models)
    - LLM-as-judge patterns (multiple models evaluating same input)
    - Agent handoff (route tasks to different specialized models)
    - Batch inference with automatic load balancing

    Features:
    - Priority-based task queue (CRITICAL, HIGH, NORMAL, LOW)
    - Async worker pool with configurable concurrency
    - Automatic retry on failure
    - Health-based routing to available instances
    - Real-time statistics and monitoring
    """

    def __init__(self,
                 instances: List[Dict] = None,
                 max_workers: int = 10,
                 strategy: str = 'fastest'):
        """Initialize parallel task queue.

        Args:
            instances: Remote Ollama instances (auto-discovered if None)
            max_workers: Number of concurrent workers
            strategy: Load balancing strategy
        """
        pass

    async def start(self):
        """Start the worker pool."""
        pass

    async def stop(self):
        """Stop all workers gracefully."""
        pass

    async def submit(self,
                     model: str,
                     messages: List[Dict],
                     priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a single task.

        Args:
            model: Model name or semantic description
            messages: Chat messages
            priority: Task priority level

        Returns:
            Task ID for tracking
        """
        pass

    async def submit_batch(self,
                          tasks: List[Dict],
                          priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """Submit multiple tasks for parallel processing.

        Args:
            tasks: List of task specs with 'model' and 'messages'
            priority: Priority level for all tasks

        Returns:
            List of task IDs
        """
        pass

    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> Dict:
        """Wait for a specific task to complete.

        Returns:
            {'status': 'completed', 'result': ..., 'duration': ...}
        """
        pass

    async def wait_for_batch(self, task_ids: List[str], timeout: float = 120.0) -> List[Dict]:
        """Wait for all tasks in batch to complete."""
        pass

    def get_stats(self) -> Dict:
        """Get queue statistics and performance metrics."""
        pass

    async def process_dataset(self,
                             dataset: List[Dict],
                             model: str,
                             batch_size: int = 100,
                             priority: TaskPriority = TaskPriority.NORMAL) -> List[Dict]:
        """Process entire dataset in parallel.

        High-level convenience method for dataset processing.

        Args:
            dataset: List of records to process
            model: Model to use for all records
            batch_size: Number of tasks to submit at once
            priority: Task priority

        Returns:
            List of results in same order as input
        """
        pass
```

## Usage Examples

### Example 1: Process Large Dataset (100k rows)

```python
import asyncio
from llamasniffer import ParallelTaskQueue, TaskPriority

async def process_large_dataset():
    # Your 100k row dataset
    dataset = [{'id': i, 'text': f'...'} for i in range(100000)]

    # Initialize queue with auto-discovered instances
    queue = ParallelTaskQueue(max_workers=50, strategy='fastest')
    await queue.start()

    # Process all 100k rows in parallel
    results = await queue.process_dataset(
        dataset=dataset,
        model='fast',  # Semantic model selection
        batch_size=1000,  # Submit 1000 at a time
        priority=TaskPriority.HIGH
    )

    # Get statistics
    stats = queue.get_stats()
    print(f"Processed {stats['completed']} tasks")
    print(f"Average time: {stats['avg_time']}s")
    print(f"Success rate: {stats['success_rate']}%")

    await queue.stop()
    return results

results = asyncio.run(process_large_dataset())
```

### Example 2: LLM-as-Judge (Multiple Models Evaluating)

```python
async def llm_as_judge():
    queue = ParallelTaskQueue()
    await queue.start()

    # Same prompt to multiple models for consensus
    prompt = "Is this text toxic: 'Hello friend!'"

    judge_tasks = [
        {'model': 'llama3', 'messages': [{'role': 'user', 'content': prompt}]},
        {'model': 'qwen', 'messages': [{'role': 'user', 'content': prompt}]},
        {'model': 'gemma', 'messages': [{'role': 'user', 'content': prompt}]},
    ]

    # Submit all judges in parallel
    task_ids = await queue.submit_batch(judge_tasks, priority=TaskPriority.CRITICAL)

    # Wait for consensus
    results = await queue.wait_for_batch(task_ids)

    # Analyze consensus
    votes = [r['result']['response'] for r in results if r['status'] == 'completed']
    consensus = most_common(votes)

    await queue.stop()
    return consensus
```

### Example 3: Agent Handoff (Route to Specialized Models)

```python
async def agent_handoff():
    queue = ParallelTaskQueue()
    await queue.start()

    # Route different tasks to specialized models
    tasks = [
        {'model': 'coding', 'messages': [{'role': 'user', 'content': 'Write Python code'}]},
        {'model': 'reasoning', 'messages': [{'role': 'user', 'content': 'Solve math problem'}]},
        {'model': 'creative', 'messages': [{'role': 'user', 'content': 'Write a story'}]},
    ]

    # Each task routed to best model via semantic matching
    task_ids = await queue.submit_batch(tasks)
    results = await queue.wait_for_batch(task_ids)

    await queue.stop()
    return results
```

## Migration Path

1. Create new `ParallelTaskQueue` class in `llamasniffer/tasks.py`
2. Remove `FlockShepherd` and `FlockHerder` aliases entirely
3. Update documentation with clear examples
4. Ensure downstream modules (dataset forge/CLI) reference the new queue directly

## Benefits

1. **Clear naming**: Immediately obvious what it does
2. **Single class**: No confusion about which to use
3. **Better docs**: Use cases are clear
4. **Easier to use**: One import, one class
5. **Maintained functionality**: All features preserved
