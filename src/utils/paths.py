"""A script defining paths for the project structure."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
