"""A script defining paths for the project structure."""

from pathlib import Path

__all__ = ["PROJECT_ROOT", "DATA_DIR", "MODEL_DIR", "CONFIG_DIR"]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
