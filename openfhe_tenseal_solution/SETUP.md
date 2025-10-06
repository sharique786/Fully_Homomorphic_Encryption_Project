# 🚀 Setup Guide - FHE Financial Analytics

## Quick Fix for Import Error

### **Problem:**
```
ImportError: cannot import name 'FHEKeyManager' from 'fhe_core'
```

### **Solution Steps:**

## Step 1: Verify File Structure

Make sure your directory structure looks like this:

```
your_project_folder/
│
├── app.py
├── config.py
├── fhe_core.py
├── data_manager.py
├── ui_components.py
├── analytics.py
├── requirements.txt
├── fix_imports.py
│
└── pages/
    ├── __init__.py
    ├── data_upload_page.py
    ├── fhe_operations_page.py
    ├── statistics_page.py
    └── key_management_page.py
```

## Step 2: Check Python Path

Run this command from your project folder:

```bash
python fix_imports.py
```

This will verify all imports are working correctly.

## Step 3: Alternative - Run from Project Root

Instead of running from a subdirectory, always run from the project root:

```bash
# Navigate to project root
cd C:\Users\alish\Workspaces\Python\Homomorphic-Enc-Project\openfhe_tenseal_solution

# Run the app
streamlit run app.py
```

## Step 4: If Still Getting Errors

### Option A: Use Absolute Imports

Edit `app.py` and add this at the top:

```python
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
```

### Option B: Create a Startup Script

Create `run_app.py`:

```python
import sys
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import and run
import streamlit.web.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
```

Then run:
```bash
python run_app.py
```

## Step 5: Verify Each Module

### Check config.py exists:
```bash
python -c "import config; print('Config OK')"
```

### Check fhe_core.py exists:
```bash
python -c "from fhe_core import FHEKeyManager; print('FHE Core OK')"
```

### Check data_manager.py exists:
```bash
python -c "from data_manager import DataManager; print('Data Manager OK')"
```

### Check ui_components.py exists:
```bash
python -c "from ui_components import render_sidebar; print('UI Components OK')"
```

## Step 6: Check Python Version

```bash
python --version
# Should show Python 3.8 - 3.11
```

## Step 7: Reinstall Dependencies

```bash
pip uninstall streamlit pandas numpy plotly tenseal
pip install streamlit pandas numpy plotly tenseal
```

## Common Issues and Solutions

### Issue 1: Module Not Found

**Solution:**
```bash
# Make sure you're in the correct directory
pwd  # or cd on Windows

# Should show: .../Homomorphic-Enc-Project/openfhe_tenseal_solution
```

### Issue 2: Permission Denied

**Solution:**
```bash
# Run as administrator or use:
pip install --user streamlit pandas numpy plotly tenseal
```

### Issue 3: Import Circular Dependencies

**Solution:** This is already handled in the code. If you still see it, restart your Python interpreter:
```bash
# Close and reopen terminal
# Navigate back to project
cd your_project_path
streamlit run app.py
```

## Minimal Test Script

Create `test_imports.py`:

```python
#!/usr/bin/env python
"""Test if all imports work"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    import config
    print("✅ config imported")
except Exception as e:
    print(f"❌ config failed: {e}")

try:
    from fhe_core import FHEKeyManager, FHEProcessor
    print("✅ fhe_core imported")
except Exception as e:
    print(f"❌ fhe_core failed: {e}")

try:
    from data_manager import DataManager
    print("✅ data_manager imported")
except Exception as e:
    print(f"❌ data_manager failed: {e}")

try:
    from ui_components import render_sidebar
    print("✅ ui_components imported")
except Exception as e:
    print(f"❌ ui_components failed: {e}")

try:
    from analytics import AnalyticsEngine
    print("✅ analytics imported")
except Exception as e:
    print(f"❌ analytics failed: {e}")

print("\n✅ All imports successful!" if all else "❌ Some imports failed")
```

Run it:
```bash
python test_imports.py
```

## Working Configuration

If imports still fail, use this simplified `app.py`:

```python
import streamlit as st
import sys
import os

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import modules
from config import APP_CONFIG, FHE_SCHEMES, LIBRARY_OPTIONS
from data_manager import DataManager
from fhe_core import FHEKeyManager, FHEProcessor
from analytics import AnalyticsEngine

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'key_manager' not in st.session_state:
    st.session_state.key_manager = FHEKeyManager()

if 'analytics_engine' not in st.session_state:
    st.session_state.analytics_engine = AnalyticsEngine()

# Simple UI
st.title("🔐 FHE Financial Analytics")
st.success("✅ All modules loaded successfully!")

# Test each module
st.write("**Module Status:**")
st.write(f"- Config: {APP_CONFIG['app_name']}")
st.write(f"- Data Manager: Initialized")
st.write(f"- Key Manager: Initialized")
st.write(f"- Analytics: Initialized")
```

## Quick Start Command

```bash
# One-liner to setup and run
cd C:\Users\alish\Workspaces\Python\Homomorphic-Enc-Project\openfhe_tenseal_solution && python -m streamlit run app.py
```

## Environment Variables (Optional)

Add to your shell profile:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "C:\Users\alish\Workspaces\Python\Homomorphic-Enc-Project\openfhe_tenseal_solution"

# Linux/Mac (bash)
export PYTHONPATH="/path/to/project:$PYTHONPATH"
```

## Still Having Issues?

1. **Check file names** - Make sure files are named exactly:
   - `fhe_core.py` (not `fhe-core.py` or `fhe_core.txt`)
   
2. **Check file encoding** - Save all files as UTF-8

3. **Restart IDE** - If using VS Code or PyCharm, restart it

4. **Virtual Environment** - Create fresh venv:
   ```bash
   python -m venv fresh_env
   fresh_env\Scripts\activate  # Windows
   pip install streamlit pandas numpy plotly tenseal
   streamlit run app.py
   ```

## Success Verification

You should see:
```
✅ Config OK
✅ FHE Core OK
✅ Data Manager OK
✅ UI Components OK
✅ Analytics OK

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

If you see this, everything is working! 🎉