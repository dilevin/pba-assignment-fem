"""
Data directory utilities.
"""

from pathlib import Path

def get_data_directory() -> Path:
    """Get the absolute path to the data directory."""
    return Path(__file__).parent.absolute()
