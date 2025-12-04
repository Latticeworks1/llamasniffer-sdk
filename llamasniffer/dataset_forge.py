"""
Distributed synthetic dataset creation using the global Ollama flock.

Leverages the shepherding system to generate large-scale, high-quality synthetic
datasets across multiple models and instances for training, evaluation, and research.
"""

import asyncio
import time
import json
import uuid
import random
import yaml
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict

from .tasks import ParallelTaskQueue, TaskPriority
from .schema_compiler import SchemaCompiler, DatasetSchema, GenerationTemplateEngine


class DatasetType(Enum):
    QA_PAIRS = "qa_pairs"
    CONVERSATIONS = "conversations"
    INSTRUCTIONS = "instructions"
    CODE_COMPLETION = "code_completion"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


class QualityLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    dataset_type: DatasetType
    target_size: int
    quality_level: QualityLevel = QualityLevel.STANDARD
    models: List[str] = field(default_factory=lambda: ["llama3", "openchat", "deepseek-r1"])
    diversity_models: List[str] = field(default_factory=lambda: ["qwen2.5", "gemma3"])
    validation_model: str = "deepseek-r1"
    batch_size: int = 50
    max_retries: int = 3
    require_consensus: bool = False
    consensus_threshold: float = 0.7
    deduplicate: bool = True
    output_format: str = "jsonl"


@dataclass
class DataPoint:
    """Individual synthetic data point."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_type: DatasetType = DatasetType.QA_PAIRS
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    validation_status: str = "pending"
    generated_by: str = ""
    generated_at: float = field(default_factory=time.time)
    hash: str = ""
    
    def __post_init__(self):
        """Generate content hash for deduplication."""
        content_str = json.dumps(self.content, sort_keys=True)
        self.hash = hashlib.md5(content_str.encode()).hexdigest()


class DatasetForge:
    """Distributed synthetic dataset creation engine."""
    
    def __init__(self, config: Union[DatasetConfig, Dict, str], queue: ParallelTaskQueue = None):
        # Handle different config input types
        if isinstance(config, str):
            # Load from YAML/JSON file
            self.raw_config = self._load_config_file(config)
            self.config = self._parse_config_dict(self.raw_config)
        elif isinstance(config, dict):
            # Parse from dictionary
            self.raw_config = config
            self.config = self._parse_config_dict(config)
        else:
            # Use DatasetConfig directly
            self.config = config
            self.raw_config = {}
        
        self.queue = queue or ParallelTaskQueue()
        self._queue_owner = queue is None
        
        # Initialize schema compiler and compiled schema
        self.schema_compiler = SchemaCompiler()
        self.compiled_schema: Optional[DatasetSchema] = None
        self.template_engine: Optional[GenerationTemplateEngine] = None
        
        # Compile schema if raw config available
        if self.raw_config:
            self.compiled_schema = self.schema_compiler.compile_schema(self.raw_config)
            self.template_engine = GenerationTemplateEngine(self.compiled_schema)
        
        # Dataset state
        self.generated_data: List[DataPoint] = []
        self.validation_queue: List[DataPoint] = []
        self.completed_data: List[DataPoint] = []
        self.rejected_data: List[DataPoint] = []
        
        # Deduplication tracking
        self.seen_hashes = set()
        
        # Statistics
        self.stats = {
            "total_generated": 0,
            "total_validated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "total_duplicates": 0,
            "generation_time": 0.0,
            "validation_time": 0.0,
            "quality_distribution": defaultdict(int)
        }

    async def forge_dataset(self) -> Dict[str, Any]:
        """Main orchestration method for dataset creation."""
        print(f"Starting distributed dataset forge: {self.config.dataset_type.value}")
        print(f"Target size: {self.config.target_size} samples")
        print(f"Quality level: {self.config.quality_level.value}")
        print(f"Generation models: {', '.join(self.config.models)}")
        
        await self.queue.start()
        
        try:
            # Phase 1: Distributed generation
            await self._generation_phase()
            
            # Phase 2: Quality validation
            await self._validation_phase()
            
            # Phase 3: Final processing
            dataset = await self._finalization_phase()
            
            return dataset
            
        finally:
            if self._queue_owner:
                await self.queue.stop()

    async def _generation_phase(self):
        """Distributed data generation across the flock."""
        print("Phase 1: Distributed Generation")
        print("--------------------------------")
        
        generation_start = time.time()
        
        # Calculate batches needed
        batches_needed = (self.config.target_size + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(batches_needed):
            print(f"Generating batch {batch_idx + 1}/{batches_needed}")
            
            # Create generation tasks for this batch
            generation_tasks = []
            samples_in_batch = min(self.config.batch_size, 
                                 self.config.target_size - len(self.completed_data))
            
            # Distribute across different models for diversity
            for i in range(samples_in_batch):
                model = random.choice(self.config.models)
                
                # Generate prompt based on dataset type
                prompt = self._generate_creation_prompt(self.config.dataset_type, i)
                
                task = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "timeout": 60.0
                }
                generation_tasks.append(task)
            
            # Execute batch in parallel
            task_ids = await self.queue.submit_batch(
                generation_tasks,
                priority=TaskPriority.HIGH,
            )
            batch_results = await self.queue.wait_for_batch(task_ids)
            
            # Process batch results
            await self._process_generation_batch(batch_results, batch_idx)
            
            # Stop if we have enough data
            if len(self.completed_data) >= self.config.target_size:
                break
        
        generation_time = time.time() - generation_start
        self.stats["generation_time"] = generation_time
        print(f"Generation phase completed in {generation_time:.1f}s")
        print(f"Generated {len(self.generated_data)} samples")

    async def _process_generation_batch(self, batch_results: List[Dict], batch_idx: int):
        """Process results from a generation batch."""
        for i, result in enumerate(batch_results):
            if result["status"] != "completed":
                continue
                
            try:
                # Extract generated content
                response_content = result["result"]["response"]
                
                # Parse and create data point
                data_point = self._parse_generated_content(
                    response_content, 
                    self.config.dataset_type,
                    batch_idx * self.config.batch_size + i
                )
                
                # Check for duplicates
                if self.config.deduplicate and data_point.hash in self.seen_hashes:
                    self.stats["total_duplicates"] += 1
                    continue
                
                self.seen_hashes.add(data_point.hash)
                self.generated_data.append(data_point)
                self.stats["total_generated"] += 1
                
            except Exception as e:
                print(f"Failed to parse generated content: {e}")
                continue

    async def _validation_phase(self):
        """Quality validation of generated data."""
        if not self.config.require_consensus:
            # Simple acceptance - move all to completed
            self.completed_data.extend(self.generated_data)
            self.stats["total_accepted"] = len(self.generated_data)
            return
            
        print("Phase 2: Quality Validation")
        print("---------------------------")
        
        validation_start = time.time()
        
        # Create validation tasks
        validation_tasks = []
        for data_point in self.generated_data:
            validation_prompt = self._generate_validation_prompt(data_point)
            
            task = {
                "model": self.config.validation_model,
                "messages": [{"role": "user", "content": validation_prompt}],
                "timeout": 30.0
            }
            validation_tasks.append((data_point, task))
        
        # Process validation in batches
        batch_size = 20
        for i in range(0, len(validation_tasks), batch_size):
            batch = validation_tasks[i:i + batch_size]
            batch_tasks = [task for _, task in batch]
            
            task_ids = await self.queue.submit_batch(batch_tasks, priority=TaskPriority.NORMAL)
            validation_results = await self.queue.wait_for_batch(task_ids)
            
            # Process validation results
            for j, result in enumerate(validation_results):
                data_point = batch[j][0]
                
                if result["status"] == "completed":
                    quality_score = self._parse_validation_result(
                        result["result"]["response"]
                    )
                    data_point.quality_score = quality_score
                    
                    # Accept or reject based on quality threshold
                    threshold = self._get_quality_threshold()
                    if quality_score >= threshold:
                        data_point.validation_status = "accepted"
                        self.completed_data.append(data_point)
                        self.stats["total_accepted"] += 1
                    else:
                        data_point.validation_status = "rejected"
                        self.rejected_data.append(data_point)
                        self.stats["total_rejected"] += 1
                else:
                    # Validation failed - default accept
                    data_point.validation_status = "validation_failed"
                    self.completed_data.append(data_point)
                    self.stats["total_accepted"] += 1
        
        validation_time = time.time() - validation_start
        self.stats["validation_time"] = validation_time
        print(f"Validation phase completed in {validation_time:.1f}s")
        print(f"Accepted: {self.stats['total_accepted']}")
        print(f"Rejected: {self.stats['total_rejected']}")

    async def _finalization_phase(self) -> Dict[str, Any]:
        """Final dataset processing and packaging."""
        print("Phase 3: Finalization")
        print("---------------------")
        
        # Sort by quality score (highest first)
        self.completed_data.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Trim to target size if we have excess
        if len(self.completed_data) > self.config.target_size:
            self.completed_data = self.completed_data[:self.config.target_size]
        
        # Calculate final statistics
        self._calculate_final_stats()
        
        # Package dataset
        dataset = {
            "metadata": {
                "dataset_type": self.config.dataset_type.value,
                "size": len(self.completed_data),
                "quality_level": self.config.quality_level.value,
                "generation_models": self.config.models,
                "validation_model": self.config.validation_model,
                "created_at": time.time(),
                "statistics": self.stats
            },
            "data": [self._serialize_datapoint(dp) for dp in self.completed_data]
        }
        
        print(f"Dataset forge completed:")
        print(f"- Final size: {len(self.completed_data)} samples")
        print(f"- Average quality: {self.stats.get('avg_quality', 0):.2f}")
        print(f"- Total time: {self.stats['generation_time'] + self.stats['validation_time']:.1f}s")
        
        return dataset

    def _load_config_file(self, config_path: str) -> Dict:
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _parse_config_dict(self, config_dict: Dict) -> DatasetConfig:
        """Parse dictionary configuration into DatasetConfig object."""
        return DatasetConfig(
            dataset_type=DatasetType(config_dict.get('dataset_type', 'qa_pairs')),
            target_size=config_dict.get('target_size', 100),
            quality_level=QualityLevel(config_dict.get('quality_level', 'standard')),
            models=config_dict.get('models', ['llama3', 'deepseek-r1']),
            validation_model=config_dict.get('validation_model', 'deepseek-r1'),
            batch_size=config_dict.get('batch_size', 50),
            max_retries=config_dict.get('max_retries', 3),
            require_consensus=config_dict.get('require_consensus', False),
            consensus_threshold=config_dict.get('consensus_threshold', 0.7),
            deduplicate=config_dict.get('deduplicate', True),
            output_format=config_dict.get('output_format', 'jsonl')
        )

    def _generate_creation_prompt(self, dataset_type: DatasetType, index: int) -> str:
        """Generate prompts for data creation based on dataset type."""
        
        # Use compiled schema template if available
        if self.template_engine:
            return self.template_engine.generate_prompt()
        
        # Fallback to static prompts
        prompts = {
            DatasetType.QA_PAIRS: [
                "Create a high-quality question and answer pair about science. Format as Q: [question] A: [answer]",
                "Generate an educational Q&A about technology. Use clear, informative language.",
                "Create a factual question-answer pair about history or geography.",
            ],
            DatasetType.CONVERSATIONS: [
                "Generate a natural conversation between two people discussing a common topic. Make it realistic and engaging.",
                "Create a dialogue about problem-solving between a mentor and student.",
                "Generate a conversation about current events or social topics.",
            ],
            DatasetType.INSTRUCTIONS: [
                "Create a clear instruction for completing a practical task, followed by step-by-step guidance.",
                "Generate instructions for a cooking recipe or DIY project.",
                "Create instructions for using a software tool or application.",
            ],
            DatasetType.CODE_COMPLETION: [
                "Write a Python function with docstring that solves a specific programming problem.",
                "Create a code snippet that demonstrates a programming concept with comments.",
                "Generate a complete class implementation for a common data structure.",
            ],
            DatasetType.REASONING: [
                "Create a logical reasoning problem with step-by-step solution.",
                "Generate a mathematical word problem with detailed explanation.",
                "Create a critical thinking scenario with analysis.",
            ]
        }
        
        prompt_list = prompts.get(dataset_type, prompts[DatasetType.QA_PAIRS])
        return random.choice(prompt_list)

    def _generate_validation_prompt(self, data_point: DataPoint) -> str:
        """Generate validation prompt for quality assessment."""
        content_preview = str(data_point.content)[:200]
        
        return f"""
Rate the quality of this generated content on a scale of 0.0 to 1.0:

Content: {content_preview}

Consider:
- Factual accuracy
- Clarity and coherence  
- Usefulness and relevance
- Grammar and formatting

Respond with only a number between 0.0 and 1.0.
"""

    def _parse_generated_content(self, content: str, dataset_type: DatasetType, index: int) -> DataPoint:
        """Parse generated content into structured data point."""
        
        # Use schema-based parsing if available
        if self.compiled_schema and self.template_engine:
            parsed_content = self._parse_structured_content(content)
        else:
            # Fallback to basic parsing
            parsed_content = self._parse_basic_content(content, dataset_type)
        
        return DataPoint(
            dataset_type=dataset_type,
            content=parsed_content,
            metadata={
                "index": index,
                "length": len(content),
                "generated_at": time.time(),
                "schema_parsed": bool(self.compiled_schema)
            }
        )
    
    def _parse_structured_content(self, content: str) -> Dict[str, Any]:
        """Parse content using compiled schema structure."""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                parsed = json.loads(content.strip())
                
                # Validate against schema
                validation_result = self.template_engine.validate_output(parsed)
                
                if validation_result['valid']:
                    return parsed
                else:
                    print(f"Schema validation failed: {validation_result['errors']}")
            
            # Fall back to pattern-based extraction for reasoning data
            if self.config.dataset_type == DatasetType.REASONING:
                return self._extract_reasoning_structure(content)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Failed to parse structured content: {e}")
        
        # Final fallback
        return {"content": content.strip()}
    
    def _extract_reasoning_structure(self, content: str) -> Dict[str, Any]:
        """Extract reasoning structure from text content."""
        result = {}
        
        # Try to identify problem statement
        lines = content.strip().split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if any(keyword in line.lower() for keyword in ['problem:', 'question:', 'solve']):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = 'problem'
                current_content = [line.split(':', 1)[-1].strip() if ':' in line else line]
            elif any(keyword in line.lower() for keyword in ['step', 'reasoning', 'solution']):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = 'reasoning_steps'
                current_content = [line]
            elif any(keyword in line.lower() for keyword in ['answer:', 'result:', 'final']):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = 'answer'
                current_content = [line.split(':', 1)[-1].strip() if ':' in line else line]
            else:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            result[current_section] = '\n'.join(current_content).strip()
        
        # Convert reasoning steps to list if it's a string
        if 'reasoning_steps' in result and isinstance(result['reasoning_steps'], str):
            steps = []
            for line in result['reasoning_steps'].split('\n'):
                line = line.strip()
                if line and (line.startswith('Step') or line.startswith('-') or line.startswith('*')):
                    # Clean up step markers
                    clean_step = line.lstrip('Step 0123456789.-* ').strip()
                    if clean_step:
                        steps.append(clean_step)
            result['reasoning_steps'] = steps if steps else [result['reasoning_steps']]
        
        # Ensure required fields exist
        if 'problem' not in result:
            result['problem'] = result.get('content', 'No problem statement found')
        if 'reasoning_steps' not in result:
            result['reasoning_steps'] = ['No reasoning steps found']
        if 'answer' not in result:
            result['answer'] = 'No answer found'
        
        return result
    
    def _parse_basic_content(self, content: str, dataset_type: DatasetType) -> Dict[str, Any]:
        """Basic parsing for backwards compatibility."""
        if dataset_type == DatasetType.QA_PAIRS:
            # Try to extract Q: and A: format
            if "Q:" in content and "A:" in content:
                parts = content.split("A:", 1)
                question = parts[0].replace("Q:", "").strip()
                answer = parts[1].strip()
                
                return {
                    "question": question,
                    "answer": answer
                }
            else:
                return {"content": content.strip()}
        
        elif dataset_type == DatasetType.CONVERSATIONS:
            return {"conversation": content.strip()}
            
        elif dataset_type == DatasetType.INSTRUCTIONS:
            return {"instruction": content.strip()}
            
        else:
            return {"content": content.strip()}

    def _parse_validation_result(self, validation_response: str) -> float:
        """Parse validation model response to extract quality score."""
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', validation_response)
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        # Default to medium quality if parsing fails
        return 0.5

    def _get_quality_threshold(self) -> float:
        """Get quality threshold based on quality level."""
        thresholds = {
            QualityLevel.BASIC: 0.3,
            QualityLevel.STANDARD: 0.5,
            QualityLevel.HIGH: 0.7,
            QualityLevel.PREMIUM: 0.8
        }
        return thresholds[self.config.quality_level]

    def _calculate_final_stats(self):
        """Calculate final dataset statistics."""
        if self.completed_data:
            qualities = [dp.quality_score for dp in self.completed_data if dp.quality_score > 0]
            if qualities:
                self.stats["avg_quality"] = sum(qualities) / len(qualities)
                self.stats["min_quality"] = min(qualities)
                self.stats["max_quality"] = max(qualities)

    def _serialize_datapoint(self, data_point: DataPoint) -> Dict:
        """Serialize data point for output."""
        return {
            "id": data_point.id,
            "content": data_point.content,
            "metadata": data_point.metadata,
            "quality_score": data_point.quality_score,
            "validation_status": data_point.validation_status
        }


async def create_synthetic_dataset(dataset_type: DatasetType, 
                                 target_size: int,
                                 quality_level: QualityLevel = QualityLevel.STANDARD,
                                 **kwargs) -> Dict[str, Any]:
    """Convenience function for creating synthetic datasets."""
    
    config = DatasetConfig(
        dataset_type=dataset_type,
        target_size=target_size,
        quality_level=quality_level,
        **kwargs
    )
    
    forge = DatasetForge(config)
    dataset = await forge.forge_dataset()
    
    return dataset


# Example usage patterns
async def example_qa_dataset():
    """Example: Create Q&A dataset using distributed generation."""
    
    dataset = await create_synthetic_dataset(
        dataset_type=DatasetType.QA_PAIRS,
        target_size=1000,
        quality_level=QualityLevel.HIGH,
        models=["llama3", "openchat", "deepseek-r1"],
        require_consensus=True,
        batch_size=50
    )
    
    return dataset


async def example_conversation_dataset():
    """Example: Create conversation dataset with diversity models."""
    
    dataset = await create_synthetic_dataset(
        dataset_type=DatasetType.CONVERSATIONS,
        target_size=500,
        quality_level=QualityLevel.STANDARD,
        models=["llama3", "openchat"],
        diversity_models=["qwen2.5", "gemma3"],
        batch_size=25
    )
    
    return dataset
