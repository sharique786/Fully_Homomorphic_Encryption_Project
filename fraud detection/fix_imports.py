"""
Import Fix Script
Run this if you encounter import errors
"""

import os
import sys


def fix_imports():
    """Add project directory to Python path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Added {current_dir} to Python path")

    # Verify imports
    try:
        import config
        print("✅ config.py imported successfully")
    except ImportError as e:
        print(f"❌ Error importing config: {e}")

    try:
        import fhe_core
        print("✅ fhe_core.py imported successfully")
    except ImportError as e:
        print(f"❌ Error importing fhe_core: {e}")

    try:
        import data_manager
        print("✅ data_manager.py imported successfully")
    except ImportError as e:
        print(f"❌ Error importing data_manager: {e}")

    try:
        import ui_components
        print("✅ ui_components.py imported successfully")
    except ImportError as e:
        print(f"❌ Error importing ui_components: {e}")

    try:
        import analytics
        print("✅ analytics.py imported successfully")
    except ImportError as e:
        print(f"❌ Error importing analytics: {e}")

    print("\n✅ Import check complete!")


if __name__ == "__main__":
    fix_imports()