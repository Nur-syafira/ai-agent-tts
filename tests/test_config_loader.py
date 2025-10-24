"""
Tests for config loader.
"""

import pytest
from pathlib import Path
from pydantic import BaseModel
from src.shared.config_loader import load_yaml_config, override_with_env
import os


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        # Create temp config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
server:
  host: "localhost"
  port: 8000
debug: true
        """)
        
        config = load_yaml_config(config_path)
        
        assert config["server"]["host"] == "localhost"
        assert config["server"]["port"] == 8000
        assert config["debug"] is True

    def test_override_with_env(self, monkeypatch):
        """Test overriding config with ENV variables."""
        config = {
            "server": {"host": "localhost", "port": 8000},
            "debug": False,
        }
        
        # Set ENV variables
        monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("SERVER_PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")
        
        updated_config = override_with_env(config)
        
        assert updated_config["server"]["host"] == "0.0.0.0"
        assert updated_config["server"]["port"] == 9000
        assert updated_config["debug"] is True

    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("/nonexistent/config.yaml"))

