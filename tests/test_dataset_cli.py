import pytest

from llamasniffer.dataset_cli import DatasetGenerationConfig, ProgressTracker
from llamasniffer.dataset_forge import QualityLevel


def test_dataset_generation_config_parses_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: sample
dataset_type: qa_pairs
target_size: 5
quality_level: basic
models:
  - m1
"""
    )

    loader = DatasetGenerationConfig(str(config_path))
    cfg = loader.to_dataset_config()

    assert cfg.target_size == 5
    assert cfg.quality_level == QualityLevel.BASIC
    assert cfg.models == ["m1"]


def test_dataset_generation_config_rejects_invalid_dataset(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        """
name: sample
dataset_type: nope
target_size: 3
"""
    )

    with pytest.raises(ValueError):
        DatasetGenerationConfig(str(bad_path))


def test_progress_tracker_reports_estimates(monkeypatch):
    tracker = ProgressTracker(target_size=10)
    tracker.update(completed=5, failed=1, duplicates=1)
    info = tracker.get_progress_info()

    assert info["completed"] == 5
    assert info["failed"] == 1
    assert info["duplicates"] == 1
    assert 0 < info["progress_pct"] <= 100

    bar = tracker._create_progress_bar(info["progress_pct"])
    assert bar.startswith("[") and bar.endswith("]")
