"""
Fully Homomorphic Encryption Financial Data Analysis System
Main Application Entry Point

File Structure:
- app.py (this file): Main Streamlit application
- fhe_core.py: Core FHE operations and key management
- data_manager.py: Data handling and CSV operations
- ui_components.py: Reusable UI components
- analytics.py: Statistical analysis and visualizations
- openfhe_wrapper.py: OpenFHE C++ wrapper
- tenseal_wrapper.py: TenSEAL wrapper
- config.py: Configuration and constants
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from config import APP_CONFIG, FHE_SCHEMES, LIBRARY_OPTIONS
from ui_components import (
    render_sidebar,
    show_welcome_screen,
    render_key_management_section
)
from data_manager import DataManager
from fhe_core import FHEKeyManager, FHEProcessor
from analytics import AnalyticsEngine

import sys
import os

# Fix imports - ADD THESE LINES AT THE TOP
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Page configuration
st.set_page_config(
    page_title="FHE Financial Analytics",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        padding: 1rem;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()

    if 'key_manager' not in st.session_state:
        st.session_state.key_manager = FHEKeyManager()

    if 'fhe_processor' not in st.session_state:
        st.session_state.fhe_processor = None

    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = AnalyticsEngine()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Upload & Encryption"

    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

    if 'encrypted_data' not in st.session_state:
        st.session_state.encrypted_data = None

    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None

    if 'selected_library' not in st.session_state:
        st.session_state.selected_library = "TenSEAL"

    if 'selected_scheme' not in st.session_state:
        st.session_state.selected_scheme = "BFV"


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Render header
    st.markdown(
        '<div class="main-header">🔐 Fully Homomorphic Encryption Financial Analytics</div>',
        unsafe_allow_html=True
    )

    # Render sidebar for navigation
    page = render_sidebar()
    st.session_state.current_page = page

    # Route to appropriate page
    if page == "Data Upload & Encryption":
        from pages import data_upload_page
        data_upload_page.render()

    elif page == "FHE Operations & Analysis":
        from pages import fhe_operations_page
        fhe_operations_page.render()

    elif page == "Performance Statistics":
        from pages import statistics_page
        statistics_page.render()

    elif page == "Key Management":
        from pages import key_management_page
        key_management_page.render()


if __name__ == "__main__":
    main()