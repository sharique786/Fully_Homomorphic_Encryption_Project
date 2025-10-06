#!/usr/bin/env python
"""
Simple launcher script for FHE Financial Analytics
This ensures all imports work correctly
"""

# python run.py

import sys
import os
from pathlib import Path


def setup_environment():
    """Setup Python environment"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Add to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    print(f"✅ Added {script_dir} to Python path")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")

    return script_dir


def verify_files(script_dir):
    """Verify all required files exist"""
    required_files = [
        'app.py',
        'config.py',
        'fhe_core.py',
        'data_manager.py',
        'ui_components.py',
        'analytics.py'
    ]

    required_dirs = ['pages']

    print("\n📁 Checking files...")

    missing_files = []
    for file in required_files:
        file_path = script_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND!")
            missing_files.append(file)

    for dir_name in required_dirs:
        dir_path = script_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - NOT FOUND!")
            missing_files.append(f"{dir_name}/")

    return len(missing_files) == 0


def test_imports():
    """Test if all modules can be imported"""
    print("\n🔍 Testing imports...")

    imports_ok = True

    try:
        import config
        print("✅ config")
    except Exception as e:
        print(f"❌ config: {e}")
        imports_ok = False

    try:
        from fhe_core import FHEKeyManager, FHEProcessor
        print("✅ fhe_core (FHEKeyManager, FHEProcessor)")
    except Exception as e:
        print(f"❌ fhe_core: {e}")
        imports_ok = False

    try:
        from data_manager import DataManager
        print("✅ data_manager (DataManager)")
    except Exception as e:
        print(f"❌ data_manager: {e}")
        imports_ok = False

    try:
        from ui_components import render_sidebar
        print("✅ ui_components")
    except Exception as e:
        print(f"❌ ui_components: {e}")
        imports_ok = False

    try:
        from analytics import AnalyticsEngine
        print("✅ analytics (AnalyticsEngine)")
    except Exception as e:
        print(f"❌ analytics: {e}")
        imports_ok = False

    return imports_ok


def launch_streamlit():
    """Launch Streamlit app"""
    import subprocess

    print("\n🚀 Launching Streamlit...")
    print("=" * 50)

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run app.py")


def main():
    """Main function"""
    print("=" * 50)
    print("🔐 FHE Financial Analytics - Launcher")
    print("=" * 50)

    # Setup environment
    script_dir = setup_environment()

    # Verify files
    if not verify_files(script_dir):
        print("\n❌ Some required files are missing!")
        print("Please ensure all files are in the correct location.")
        return

    # Test imports
    if not test_imports():
        print("\n❌ Import errors detected!")
        print("\n💡 Troubleshooting steps:")
        print("1. Make sure all .py files are in the same directory")
        print("2. Check if files are named correctly (fhe_core.py not fhe-core.py)")
        print("3. Try running: pip install -r requirements.txt")
        return

    print("\n✅ All checks passed!")

    # Launch Streamlit
    try:
        launch_streamlit()
    except Exception as e:
        print(f"\n❌ Failed to launch: {e}")


if __name__ == "__main__":
    main()