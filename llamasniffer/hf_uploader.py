"""
Automatic Hugging Face dataset upload with versioning and dataset cards.

Handles automatic upload of generated synthetic datasets to Hugging Face Hub
with proper dataset cards, versioning, and metadata management.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError


class HuggingFaceUploader:
    """Handles automatic upload of datasets to Hugging Face Hub."""
    
    def __init__(self, username: str = None, token: str = None):
        self.username = username
        self.token = token

        # Auto-detect credentials if not provided
        if not self.username or not self.token:
            self._auto_detect_credentials()

        self.api = HfApi(token=self.token)

    def _auto_detect_credentials(self):
        """Auto-detect Hugging Face credentials from OllamaConfig."""
        try:
            from .ollama import OllamaConfig
            config = OllamaConfig()

            # Use centralized config for HF credentials
            if not self.token:
                self.token = config._config.get("huggingface", {}).get("token")

            if not self.username:
                self.username = config._config.get("huggingface", {}).get("username")

            # Try to get username from HF API if we have token but no username
            if not self.username and self.token:
                try:
                    temp_api = HfApi(token=self.token)
                    user_info = temp_api.whoami()
                    self.username = user_info["name"]
                except Exception:
                    pass

        except Exception:
            pass
    
    def create_dataset_card(self, dataset_metadata: Dict, config: Dict) -> str:
        """Generate comprehensive dataset card in markdown format."""
        
        # Extract key information
        dataset_name = config.get("name", "Synthetic Dataset")
        dataset_type = dataset_metadata.get("dataset_type", "unknown")
        size = dataset_metadata.get("size", 0)
        models = dataset_metadata.get("generation_models", [])
        quality_level = dataset_metadata.get("quality_level", "standard")
        stats = dataset_metadata.get("statistics", {})
        
        # Get current date
        created_date = datetime.now().strftime("%Y-%m-%d")
        
        # Build dataset card
        card_content = f"""---
license: mit
task_categories:
- {self._get_task_category(dataset_type)}
language:
- en
tags:
- synthetic
- llamasniffer
- distributed-generation
- {dataset_type.replace('_', '-')}
size_categories:
- {self._get_size_category(size)}
pretty_name: {dataset_name}
dataset_info:
  features:
  - name: content
    dtype: string
  - name: metadata
    dtype: string
  - name: quality_score
    dtype: float64
  config_name: default
  data_files: "data.jsonl"
  splits:
  - name: train
    num_examples: {size}
---

# {dataset_name}

## Dataset Description

{config.get('description', 'Synthetically generated dataset using LlamaSniffer distributed generation.')}

**Dataset Type:** {dataset_type}  
**Size:** {size:,} samples  
**Quality Level:** {quality_level}  
**Generated:** {created_date}  

## Generation Details

### Models Used
{self._format_models_list(models)}

### Generation Statistics
- **Total Generation Time:** {stats.get('generation_time', 0):.1f} seconds
- **Success Rate:** {(stats.get('completed_tasks', 0) / max(1, stats.get('total_tasks', 1)) * 100):.1f}%
- **Duplicates Removed:** {stats.get('total_duplicates', 0)}
{self._format_quality_stats(stats)}

### Infrastructure
- **Generated using:** LlamaSniffer distributed Ollama flock
- **Global Instances:** 31 verified instances across 17 countries
- **Concurrent Processing:** Up to 93 parallel generation tasks

## Dataset Structure

```json
{{
  "content": {{
    // Content structure varies by dataset type
  }},
  "metadata": {{
    "index": 0,
    "length": 150,
    "generated_at": 1640995200.0
  }},
  "quality_score": 0.85,
  "validation_status": "accepted"
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.username}/{self._get_repo_name(config)}")
```

## Configuration

<details>
<summary>Generation Configuration</summary>

```yaml
{self._format_config_yaml(config)}
```

</details>

## Dataset Creation

This dataset was created using the LlamaSniffer distributed synthetic data generation system:

1. **Distributed Generation**: Parallel generation across global Ollama flock
2. **Quality Control**: {self._get_quality_description(config)}
3. **Deduplication**: Content hash-based duplicate removal
4. **Validation**: {self._get_validation_description(config)}

## Citation

```bibtex
@dataset{{{self._get_citation_key(config)},
  title={{{dataset_name}}},
  author={{{self.username}}},
  year={{{datetime.now().year}}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{self.username}/{self._get_repo_name(config)}}}
}}
```

## License

This dataset is released under the MIT License.

## Limitations

- Synthetically generated content may contain factual inaccuracies
- Content reflects the biases present in the generation models
- Quality may vary across different samples
- Recommended for research and training purposes

---

*Generated automatically by LlamaSniffer v{self._get_version()}*
"""
        
        return card_content
    
    def upload_dataset(self, dataset: Dict, config: Dict, 
                      repo_name: str = None, private: bool = False) -> str:
        """Upload dataset to Hugging Face Hub with automatic versioning."""
        
        if not self.username or not self.token:
            raise ValueError("Hugging Face credentials not found. Set HF_TOKEN and HF_USERNAME environment variables.")
        
        # Generate repository name
        if not repo_name:
            repo_name = self._get_repo_name(config)
        
        repo_id = f"{self.username}/{repo_name}"
        
        print(f"Uploading dataset to Hugging Face: {repo_id}")
        
        # Create repository if it doesn't exist
        try:
            self.api.repo_info(repo_id, repo_type="dataset")
            print(f"Repository {repo_id} already exists, updating...")
            
            # Check for existing versions and increment
            version = self._get_next_version(repo_id)
            if version > 1:
                print(f"Creating version {version}")
                
        except RepositoryNotFoundError:
            print(f"Creating new repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=self.token
            )
            version = 1
        
        # Prepare dataset files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Convert dataset to Hugging Face format
            hf_dataset = self._convert_to_hf_dataset(dataset)
            
            # Save dataset files
            data_file = temp_path / "data.jsonl"
            with open(data_file, 'w') as f:
                for item in dataset['data']:
                    f.write(json.dumps(item) + '\\n')
            
            # Create dataset card
            card_content = self.create_dataset_card(dataset['metadata'], config)
            card_file = temp_path / "README.md"
            with open(card_file, 'w') as f:
                f.write(card_content)
            
            # Create metadata file
            metadata_file = temp_path / "dataset_info.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    "version": version,
                    "generated_at": datetime.now().isoformat(),
                    "llamasniffer_version": self._get_version(),
                    "generation_config": config,
                    "statistics": dataset['metadata']['statistics']
                }, f, indent=2)
            
            # Upload files
            print("Uploading dataset files...")
            
            # Upload data file
            upload_file(
                path_or_fileobj=str(data_file),
                path_in_repo="data.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=f"Add dataset v{version}"
            )
            
            # Upload README/dataset card
            upload_file(
                path_or_fileobj=str(card_file),
                path_in_repo="README.md", 
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=f"Update dataset card v{version}"
            )
            
            # Upload metadata
            upload_file(
                path_or_fileobj=str(metadata_file),
                path_in_repo="dataset_info.json",
                repo_id=repo_id,
                repo_type="dataset", 
                token=self.token,
                commit_message=f"Add metadata v{version}"
            )
        
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"Dataset uploaded successfully: {dataset_url}")
        
        return dataset_url
    
    def _get_repo_name(self, config: Dict) -> str:
        """Generate repository name from config."""
        name = config.get("name", "synthetic-dataset")
        
        # Clean name for repo
        repo_name = name.lower()
        repo_name = repo_name.replace(" ", "-")
        repo_name = "".join(c for c in repo_name if c.isalnum() or c in "-_")
        
        # Add date for uniqueness
        date_suffix = datetime.now().strftime("%Y%m%d")
        
        return f"{repo_name}-{date_suffix}"
    
    def _get_next_version(self, repo_id: str) -> int:
        """Get next version number for the dataset."""
        try:
            # Check if dataset_info.json exists and get current version
            files = self.api.list_repo_files(repo_id, repo_type="dataset")
            
            if "dataset_info.json" in files:
                # Download and check current version
                content = self.api.hf_hub_download(
                    repo_id=repo_id, 
                    filename="dataset_info.json",
                    repo_type="dataset"
                )
                
                with open(content, 'r') as f:
                    info = json.load(f)
                    return info.get("version", 1) + 1
            
            return 1
            
        except Exception:
            return 1
    
    def _convert_to_hf_dataset(self, dataset: Dict) -> Dataset:
        """Convert dataset to Hugging Face Dataset format."""
        data_list = []
        
        for item in dataset['data']:
            # Flatten the structure for HF compatibility
            row = {
                'content': json.dumps(item['content']),
                'metadata': json.dumps(item['metadata']),
                'quality_score': item.get('quality_score', 0.0),
                'validation_status': item.get('validation_status', 'unknown')
            }
            data_list.append(row)
        
        return Dataset.from_list(data_list)
    
    def _get_task_category(self, dataset_type: str) -> str:
        """Map dataset type to HF task category."""
        mapping = {
            'qa_pairs': 'question-answering',
            'conversations': 'conversational',
            'instructions': 'text-generation',
            'code_completion': 'text-generation',
            'reasoning': 'text-generation',
            'classification': 'text-classification',
            'summarization': 'summarization',
            'translation': 'translation'
        }
        return mapping.get(dataset_type, 'text-generation')
    
    def _get_size_category(self, size: int) -> str:
        """Get HF size category."""
        if size < 1000:
            return "n<1K"
        elif size < 10000:
            return "1K<n<10K"
        elif size < 100000:
            return "10K<n<100K"
        else:
            return "100K<n<1M"
    
    def _format_models_list(self, models: List[str]) -> str:
        """Format models list for dataset card."""
        return "\\n".join(f"- {model}" for model in models)
    
    def _format_quality_stats(self, stats: Dict) -> str:
        """Format quality statistics."""
        quality_info = ""
        
        if 'avg_quality' in stats:
            quality_info += f"- **Average Quality Score:** {stats['avg_quality']:.2f}\\n"
        
        if 'validation_time' in stats and stats['validation_time'] > 0:
            quality_info += f"- **Validation Time:** {stats['validation_time']:.1f} seconds\\n"
        
        return quality_info
    
    def _format_config_yaml(self, config: Dict) -> str:
        """Format configuration as YAML for dataset card."""
        try:
            import yaml
            return yaml.dump(config, default_flow_style=False, indent=2)
        except:
            return json.dumps(config, indent=2)
    
    def _get_quality_description(self, config: Dict) -> str:
        """Get quality control description."""
        if config.get('require_consensus', False):
            return f"Consensus validation with {config.get('consensus_threshold', 0.7)} threshold"
        else:
            return "Basic quality filtering"
    
    def _get_validation_description(self, config: Dict) -> str:
        """Get validation process description."""
        validation_model = config.get('validation_model', 'automatic')
        if config.get('require_consensus', False):
            return f"Multi-instance validation using {validation_model}"
        else:
            return "Automatic validation"
    
    def _get_citation_key(self, config: Dict) -> str:
        """Generate citation key."""
        name = config.get("name", "dataset")
        clean_name = "".join(c for c in name if c.isalnum()).lower()
        year = datetime.now().year
        return f"{clean_name}{year}"
    
    def _get_version(self) -> str:
        """Get LlamaSniffer version."""
        try:
            from . import __version__
            return __version__
        except:
            return "unknown"