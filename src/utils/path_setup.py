"""
Path setup utility for the enterprise data pipeline.
Handles Python path configuration to ensure imports work correctly from any directory.
"""

import sys
import os
from pathlib import Path


def setup_project_paths():
    """
    Add the project root to Python path to ensure imports work correctly.
    This should be called at the beginning of any script that needs to import from src.
    """
    # Get the current file's directory
    current_file = Path(__file__)
    
    # Navigate to project root (3 levels up: utils -> src -> project_root)
    project_root = current_file.parent.parent.parent
    
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def get_project_root():
    """
    Get the absolute path to the project root directory.
    """
    return Path(__file__).parent.parent.parent


def ensure_cache_dirs():
    """
    Ensure all necessary cache directories exist.
    """
    project_root = get_project_root()
    
    cache_dirs = [
        project_root / "cache" / "embeddings",
        project_root / "output",
        project_root / "checkpoints",
        project_root / "metadata" / "lineage",
        project_root / "uncommitted",
    ]
    
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    return cache_dirs 