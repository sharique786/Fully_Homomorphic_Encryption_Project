import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ui.data_management import data_management_page
from ui.fhe_operations import fhe_operations_page
from ui.statistics import statistics_page
from utils.session_state import initialize_session_state


def main():
    st.set_page_config(
        page_title="FHE Financial Data Processor",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("🔐 FHE Financial Processor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Management", "🔒 FHE Operations", "📈 Statistics & Comparison"],
        key="navigation"
    )

    # Library selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("FHE Library Selection")
    library = st.sidebar.selectbox(
        "Select FHE Library",
        ["TenSEAL", "OpenFHE"],
        key="fhe_library"
    )

    st.sidebar.info(f"**Selected Library:** {library}")

    # Display current page
    if page == "📊 Data Management":
        data_management_page()
    elif page == "🔒 FHE Operations":
        fhe_operations_page()
    elif page == "📈 Statistics & Comparison":
        statistics_page()


if __name__ == "__main__":
    main()