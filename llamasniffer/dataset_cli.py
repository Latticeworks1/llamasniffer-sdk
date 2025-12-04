#!/usr/bin/env python3
"""
Interactive synthetic dataset generation CLI tool.

Provides OpenAI-style configuration-driven dataset generation using the
distributed Ollama flock with YAML/JSON configuration support.
"""

import argparse
import asyncio
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
import signal

from .dataset_forge import DatasetForge, DatasetConfig, DatasetType, QualityLevel
from .tasks import ParallelTaskQueue
from .hf_uploader import HuggingFaceUploader


class DatasetGenerationConfig:
    """Configuration loader and validator for dataset generation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML or JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                return json.load(f)
            else:
                # Try to detect format
                content = f.read()
                try:
                    return yaml.safe_load(content)
                except:
                    return json.loads(content)
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        required_fields = ['name', 'dataset_type', 'target_size']
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Validate dataset type
        try:
            DatasetType(self.config['dataset_type'])
        except ValueError:
            valid_types = [t.value for t in DatasetType]
            raise ValueError(f"Invalid dataset_type. Valid options: {valid_types}")
        
        # Validate quality level if specified
        if 'quality_level' in self.config:
            try:
                QualityLevel(self.config['quality_level'])
            except ValueError:
                valid_levels = [q.value for q in QualityLevel]
                raise ValueError(f"Invalid quality_level. Valid options: {valid_levels}")
    
    def to_dataset_config(self) -> DatasetConfig:
        """Convert to DatasetConfig object."""
        return DatasetConfig(
            dataset_type=DatasetType(self.config['dataset_type']),
            target_size=self.config['target_size'],
            quality_level=QualityLevel(self.config.get('quality_level', 'standard')),
            models=self.config.get('models', ['llama3', 'openchat']),
            diversity_models=self.config.get('diversity_models', []),
            validation_model=self.config.get('validation_model', 'deepseek-r1'),
            batch_size=self.config.get('batch_size', 50),
            max_retries=self.config.get('max_retries', 3),
            require_consensus=self.config.get('require_consensus', False),
            consensus_threshold=self.config.get('consensus_threshold', 0.7),
            deduplicate=self.config.get('deduplicate', True),
            output_format=self.config.get('output_format', 'jsonl')
        )


class ProgressTracker:
    """Real-time progress tracking for dataset generation."""
    
    def __init__(self, target_size: int):
        self.target_size = target_size
        self.start_time = time.time()
        self.last_update = self.start_time
        self.completed = 0
        self.failed = 0
        self.duplicates = 0
        
    def update(self, completed: int, failed: int = 0, duplicates: int = 0):
        """Update progress statistics."""
        self.completed = completed
        self.failed = failed
        self.duplicates = duplicates
        self.last_update = time.time()
        
    def get_progress_info(self) -> Dict:
        """Get current progress information."""
        elapsed = time.time() - self.start_time
        
        if self.completed > 0:
            rate = self.completed / elapsed
            eta = (self.target_size - self.completed) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0
            
        return {
            'completed': self.completed,
            'failed': self.failed,
            'duplicates': self.duplicates,
            'target': self.target_size,
            'progress_pct': (self.completed / self.target_size * 100) if self.target_size > 0 else 0,
            'elapsed': elapsed,
            'rate': rate,
            'eta': eta
        }
    
    def print_progress(self):
        """Print formatted progress information."""
        info = self.get_progress_info()
        
        progress_bar = self._create_progress_bar(info['progress_pct'])
        
        print(f"\\r{progress_bar} {info['completed']}/{info['target']} "
              f"({info['progress_pct']:.1f}%) "
              f"Rate: {info['rate']:.1f}/s "
              f"ETA: {info['eta']:.0f}s "
              f"Failed: {info['failed']} "
              f"Dupes: {info['duplicates']}", end='', flush=True)
    
    def _create_progress_bar(self, percent: float, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '▒' * (width - filled)
        return f"[{bar}]"


class InteractiveDatasetGenerator:
    """Interactive dataset generation with real-time monitoring and HF upload."""
    
    def __init__(self, config_path: str, upload_to_hf: bool = True, private: bool = False):
        self.config_loader = DatasetGenerationConfig(config_path)
        self.config = self.config_loader.to_dataset_config()
        self.raw_config = self.config_loader.config
        self.upload_to_hf = upload_to_hf
        self.private = private
        
        self.queue = ParallelTaskQueue()
        self.forge = DatasetForge(self.config, self.queue)
        self.tracker = ProgressTracker(self.config.target_size)
        
        # Initialize HF uploader if enabled
        if self.upload_to_hf:
            self.hf_uploader = HuggingFaceUploader()
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\\nShutdown signal received. Finishing current batch...")
        self.shutdown = True
    
    async def generate_dataset(self):
        """Main dataset generation with interactive monitoring."""
        print("LLAMASNIFFER SYNTHETIC DATASET GENERATOR")
        print("========================================")
        print()
        
        # Display configuration
        self._print_configuration()
        
        # Confirm generation
        if not self._confirm_generation():
            print("Generation cancelled.")
            return
        
        print("\\nStarting distributed dataset generation...")
        print("Press Ctrl+C to gracefully stop generation")
        print()
        
        # Start generation with monitoring
        dataset = await self._generate_with_monitoring()
        
        if dataset:
            # Save/upload dataset
            if self.upload_to_hf:
                dataset_url = await self._upload_to_huggingface(dataset)
                self._print_completion_summary(dataset, dataset_url, uploaded=True)
            else:
                output_path = await self._save_dataset_locally(dataset)
                self._print_completion_summary(dataset, output_path, uploaded=False)
        else:
            print("\\nGeneration was interrupted or failed.")
    
    def _print_configuration(self):
        """Print generation configuration."""
        print(f"Dataset Name: {self.raw_config['name']}")
        print(f"Type: {self.config.dataset_type.value}")
        print(f"Target Size: {self.config.target_size:,} samples")
        print(f"Quality Level: {self.config.quality_level.value}")
        print(f"Generation Models: {', '.join(self.config.models)}")
        
        if self.config.diversity_models:
            print(f"Diversity Models: {', '.join(self.config.diversity_models)}")
        
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Consensus Validation: {'Yes' if self.config.require_consensus else 'No'}")
        print(f"Deduplication: {'Yes' if self.config.deduplicate else 'No'}")
        
        # Estimate time and cost
        self._print_estimates()
    
    def _print_estimates(self):
        """Print time and resource estimates."""
        # Get cluster status for estimates
        try:
            import llamasniffer
            status = llamasniffer.get_flock_status()
            instances = status['flock_health']['total_instances']
            
            # Rough estimates based on batch size and instance count
            concurrent_tasks = min(instances * 3, self.config.batch_size)
            estimated_batches = (self.config.target_size + self.config.batch_size - 1) // self.config.batch_size
            estimated_time_minutes = (estimated_batches * 30) / concurrent_tasks  # ~30s per batch
            
            print(f"\\nEstimates:")
            print(f"Available Instances: {instances}")
            print(f"Concurrent Tasks: {concurrent_tasks}")
            print(f"Estimated Time: {estimated_time_minutes:.1f} minutes")
            
        except Exception:
            print(f"\\nNote: Connect to flock for accurate time estimates")
    
    def _confirm_generation(self) -> bool:
        """Confirm generation start with user."""
        if '--yes' in sys.argv or '-y' in sys.argv:
            return True
            
        while True:
            response = input("\\nProceed with generation? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    async def _generate_with_monitoring(self) -> Optional[Dict]:
        """Generate dataset with real-time progress monitoring."""
        # Start the monitoring task
        monitor_task = asyncio.create_task(self._monitor_progress())
        
        try:
            # Start generation
            dataset = await self.forge.forge_dataset()
            return dataset
            
        except KeyboardInterrupt:
            print("\\nGeneration interrupted by user.")
            return None
        except Exception as e:
            print(f"\\nGeneration failed: {e}")
            return None
        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_progress(self):
        """Monitor and display progress in real-time."""
        while not self.shutdown:
            try:
                # Update progress from forge statistics
                stats = self.forge.stats
                self.tracker.update(
                    completed=stats.get('completed_tasks', 0),
                    failed=stats.get('failed_tasks', 0),
                    duplicates=stats.get('total_duplicates', 0)
                )
                
                # Print progress
                self.tracker.print_progress()
                
                # Check if complete
                if self.tracker.completed >= self.tracker.target_size:
                    break
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)
    
    async def _upload_to_huggingface(self, dataset: Dict) -> str:
        """Upload dataset to Hugging Face Hub."""
        print("\\nUploading dataset to Hugging Face Hub...")
        
        try:
            dataset_url = self.hf_uploader.upload_dataset(
                dataset=dataset,
                config=self.raw_config,
                private=self.private
            )
            return dataset_url
        except Exception as e:
            print(f"HF upload failed: {e}")
            print("Falling back to local save...")
            return await self._save_dataset_locally(dataset)
    
    async def _save_dataset_locally(self, dataset: Dict) -> str:
        """Save generated dataset locally as fallback."""
        output_dir = Path("datasets")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.raw_config['name'].replace(' ', '_').lower()
        
        filename = f"{name}_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return str(output_path)
    
    def _print_completion_summary(self, dataset: Dict, path_or_url: str, uploaded: bool = False):
        """Print completion summary."""
        metadata = dataset['metadata']
        stats = metadata['statistics']
        
        print("\\n\\nGENERATION COMPLETE")
        print("==================")
        
        if uploaded:
            print(f"Dataset uploaded to Hugging Face: {path_or_url}")
            print("Dataset includes:")
            print("- Automatic versioning")
            print("- Comprehensive dataset card")  
            print("- Generation metadata")
            print("- Ready for public use")
        else:
            print(f"Dataset saved locally: {path_or_url}")
        
        print(f"\\nFinal size: {metadata['size']:,} samples")
        print(f"Generation time: {stats['generation_time']:.1f}s")
        
        if stats.get('validation_time', 0) > 0:
            print(f"Validation time: {stats['validation_time']:.1f}s")
        
        print(f"Success rate: {stats['completed_tasks'] / max(1, stats['total_tasks']) * 100:.1f}%")
        
        if stats.get('total_duplicates', 0) > 0:
            print(f"Duplicates removed: {stats['total_duplicates']}")
        
        if 'avg_quality' in stats:
            print(f"Average quality: {stats['avg_quality']:.2f}")
        
        if uploaded:
            print("\\nUsage:")
            repo_name = path_or_url.split('/')[-1]
            username = self.hf_uploader.username
            print(f"from datasets import load_dataset")
            print(f"dataset = load_dataset('{username}/{repo_name}')")


def create_example_configs():
    """Create example configuration files."""
    configs_dir = Path("config_examples")
    configs_dir.mkdir(exist_ok=True)
    
    # Q&A Dataset Configuration
    qa_config = {
        "name": "Science QA Dataset",
        "description": "High-quality science question-answer pairs for training",
        "dataset_type": "qa_pairs",
        "target_size": 10000,
        "quality_level": "high",
        "models": ["llama3", "openchat", "deepseek-r1"],
        "diversity_models": ["qwen2.5"],
        "validation_model": "deepseek-r1",
        "batch_size": 100,
        "require_consensus": True,
        "consensus_threshold": 0.8,
        "deduplicate": True,
        "max_retries": 3,
        "output_format": "jsonl",
        "generation_params": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "topics": ["physics", "chemistry", "biology", "earth_science", "astronomy"],
            "difficulty_levels": ["beginner", "intermediate", "advanced"]
        }
    }
    
    # Instruction Dataset Configuration  
    instruction_config = {
        "name": "Coding Instructions",
        "description": "Programming instruction dataset for code generation",
        "dataset_type": "instructions",
        "target_size": 5000,
        "quality_level": "premium",
        "models": ["codellama", "deepseek-r1"],
        "validation_model": "deepseek-r1",
        "batch_size": 50,
        "require_consensus": True,
        "consensus_threshold": 0.9,
        "deduplicate": True,
        "output_format": "json",
        "generation_params": {
            "languages": ["python", "javascript", "rust", "go"],
            "complexity": ["simple", "moderate", "complex"],
            "domains": ["web_dev", "data_science", "algorithms", "system_programming"]
        }
    }
    
    # Conversation Dataset Configuration
    conversation_config = {
        "name": "Customer Support Dialogues", 
        "description": "Realistic customer support conversations",
        "dataset_type": "conversations",
        "target_size": 2000,
        "quality_level": "standard",
        "models": ["llama3", "openchat"],
        "batch_size": 25,
        "require_consensus": False,
        "deduplicate": True,
        "output_format": "jsonl",
        "generation_params": {
            "scenarios": ["technical_support", "billing_inquiry", "product_question", "complaint_resolution"],
            "conversation_length": [3, 10],
            "tone": ["professional", "friendly", "patient"]
        }
    }
    
    # Save example configs
    with open(configs_dir / "science_qa.yaml", 'w') as f:
        yaml.dump(qa_config, f, default_flow_style=False)
    
    with open(configs_dir / "coding_instructions.yaml", 'w') as f:
        yaml.dump(instruction_config, f, default_flow_style=False)
    
    with open(configs_dir / "customer_support.json", 'w') as f:
        json.dump(conversation_config, f, indent=2)
    
    print(f"Example configurations created in {configs_dir}/")
    print("Available examples:")
    print("- science_qa.yaml: Science Q&A dataset")
    print("- coding_instructions.yaml: Programming instructions")
    print("- customer_support.json: Customer support dialogues")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LlamaSniffer Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset from config
  python -m llamasniffer.dataset_cli generate config.yaml
  
  # Create example configurations
  python -m llamasniffer.dataset_cli examples
  
  # Generate with custom output directory
  python -m llamasniffer.dataset_cli generate config.yaml --output datasets/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    generate_parser.add_argument('config', help='Configuration file (YAML or JSON)')
    generate_parser.add_argument('--local', action='store_true',
                                help='Save locally instead of uploading to HF')
    generate_parser.add_argument('--private', action='store_true',
                                help='Create private HF repository')
    generate_parser.add_argument('--yes', '-y', action='store_true',
                                help='Skip confirmation prompt')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Create example configurations')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        try:
            generator = InteractiveDatasetGenerator(
                config_path=args.config,
                upload_to_hf=not args.local,
                private=args.private
            )
            asyncio.run(generator.generate_dataset())
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'examples':
        create_example_configs()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
