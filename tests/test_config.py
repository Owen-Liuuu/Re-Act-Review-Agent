"""Tests for configuration loading."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from react_review.core.config import AppConfig, load_config
from react_review.core.exceptions import ConfigError


def test_default_config():
    """AppConfig should have sensible defaults."""
    config = AppConfig()
    assert config.app_name == "react-review"
    assert config.mock_mode is True
    assert len(config.enabled_steps) == 4


def test_load_config_from_yaml(tmp_path: Path):
    """Should load and validate a YAML config file."""
    config_data = {
        "app_name": "test-app",
        "environment": "testing",
        "mock_mode": True,
        "enabled_steps": ["search_validation"],
        "llm": {"provider": "mock", "model": "test-model"},
    }
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")

    config = load_config(config_file)
    assert config.app_name == "test-app"
    assert config.environment == "testing"
    assert config.enabled_steps == ["search_validation"]
    assert config.llm.model == "test-model"


def test_load_config_file_not_found():
    """Should raise ConfigError for missing file."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("nonexistent.yaml"))


def test_load_config_empty_yaml(tmp_path: Path):
    """Should return defaults for an empty YAML file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")
    config = load_config(config_file)
    assert config.app_name == "react-review"
