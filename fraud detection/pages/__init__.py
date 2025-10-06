"""
Pages package initialization
Ensures proper imports for all page modules
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import all page modules
try:
    from pages import data_upload_page
    from pages import fhe_operations_page
    from pages import statistics_page
    from pages import key_management_page
except ImportError as e:
    print(f"Warning: Could not import all pages: {e}")

__all__ = [
    'data_upload_page',
    'fhe_operations_page',
    'statistics_page',
    'key_management_page'
]