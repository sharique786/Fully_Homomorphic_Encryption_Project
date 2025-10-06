"""
UI Components Module
Reusable Streamlit UI components
"""

import streamlit as st
from typing import Dict, List, Any
from config import FHE_SCHEMES, LIBRARY_OPTIONS


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page"""
    st.sidebar.title("🔐 FHE Analytics")

    # Custom radio buttons with icons only (no text labels)
    st.sidebar.markdown("### Navigation")

    # Use custom session state for page selection
    if 'current_page_index' not in st.session_state:
        st.session_state.current_page_index = 0

    # Page options with icons
    pages = [
        ("📊", "Data Upload & Encryption"),
        ("🧮", "FHE Operations & Analysis"),
        ("📈", "Performance Statistics"),
        ("🔑", "Key Management")
    ]

    # Create custom buttons for navigation
    selected_idx = st.session_state.current_page_index

    for idx, (icon, page_name) in enumerate(pages):
        if st.sidebar.button(
            f"{icon}",
            key=f"nav_{idx}",
            use_container_width=True,
            type="primary" if idx == selected_idx else "secondary"
        ):
            st.session_state.current_page_index = idx
            st.rerun()

    page = pages[selected_idx][1]

    st.sidebar.markdown("---")

    # Library selection
    st.sidebar.subheader("⚙️ FHE Configuration")

    library = st.sidebar.selectbox(
        "FHE Library:",
        list(LIBRARY_OPTIONS.keys()),
        index=0,
        help="Select the FHE library to use"
    )
    st.session_state.selected_library = library

    # Show OpenFHE path if selected
    if library == "OpenFHE":
        openfhe_path = st.sidebar.text_input(
            "OpenFHE Path:",
            value=r"C:\openfhe-development\build\bin\Release",
            help="Path to compiled OpenFHE executable"
        )
        st.session_state.openfhe_path = openfhe_path

        # Check if executable exists
        import os
        exe_path = os.path.join(openfhe_path, "fhe_wrapper.exe")
        if os.path.exists(exe_path):
            st.sidebar.success("✅ OpenFHE executable found")
        else:
            st.sidebar.warning("⚠️ Executable not found. Using simulation.")

    # Scheme selection based on library
    available_schemes = LIBRARY_OPTIONS[library]['schemes']
    scheme = st.sidebar.selectbox(
        "FHE Scheme:",
        available_schemes,
        index=0,
        help="Select the encryption scheme"
    )
    st.session_state.selected_scheme = scheme

    st.sidebar.markdown("---")

    # Show scheme info
    with st.sidebar.expander("📖 Scheme Info"):
        if scheme in FHE_SCHEMES:
            scheme_info = FHE_SCHEMES[scheme]
            st.write(f"**{scheme_info['name']}**")
            st.write(f"Type: {scheme_info['type']}")
            st.write(f"Precision: {scheme_info['precision']}")

    return page


def show_welcome_screen():
    """Display welcome screen with app information"""
    st.markdown("""
    ### Welcome to FHE Financial Analytics System
    
    This application demonstrates Fully Homomorphic Encryption (FHE) for secure financial data analysis.
    
    **Features:**
    - 🔐 Encrypt sensitive financial data
    - 🧮 Perform computations on encrypted data
    - 📊 Analyze results without decryption
    - 🔑 Comprehensive key management
    - 📈 Performance benchmarking
    
    **Get Started:**
    1. Upload or generate financial data
    2. Select columns to encrypt
    3. Generate or import encryption keys
    4. Perform homomorphic operations
    5. View results and statistics
    """)


def render_key_management_section(key_manager) -> Dict[str, Any]:
    """Render key management UI section"""
    st.subheader("🔑 Key Management")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Key Generation Options**")

        key_action = st.radio(
            "Choose action:",
            ["Generate New Keys", "Import Existing Keys", "View Current Keys"]
        )

    with col2:
        st.write("**Key Information**")
        if key_manager.key_metadata.get('generation_time'):
            st.info(f"✅ Keys generated on: {key_manager.key_metadata['generation_time']}")
            st.write(f"Scheme: {key_manager.key_metadata.get('scheme', 'N/A')}")
            st.write(f"Library: {key_manager.key_metadata.get('library', 'N/A')}")
        else:
            st.warning("⚠️ No keys generated yet")

    return {'action': key_action}


def render_data_preview(df, title: str = "Data Preview", max_rows: int = 10):
    """Render data preview with statistics"""
    st.subheader(title)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))

    st.dataframe(df.head(max_rows), use_container_width=True)


def render_encryption_config(scheme: str) -> Dict[str, Any]:
    """Render encryption configuration UI"""
    st.subheader("⚙️ Encryption Parameters")

    scheme_config = FHE_SCHEMES.get(scheme, {})

    col1, col2 = st.columns(2)

    with col1:
        poly_modulus_degree = st.selectbox(
            "Polynomial Modulus Degree:",
            scheme_config.get('poly_modulus_degrees', [8192]),
            index=scheme_config.get('poly_modulus_degrees', [8192]).index(
                scheme_config.get('default_poly_modulus', 8192)
            ),
            help="Higher values = more security but slower performance"
        )

        if scheme in ['BFV', 'BGV']:
            plain_modulus = st.selectbox(
                "Plain Modulus:",
                scheme_config.get('plain_modulus_options', [65537]),
                help="Modulus for plaintext space"
            )

    with col2:
        security_level = st.selectbox(
            "Security Level (bits):",
            scheme_config.get('security_levels', [128]),
            help="Higher = more secure but slower"
        )

        if scheme == 'CKKS':
            scale_factor = st.selectbox(
                "Scale Factor:",
                scheme_config.get('scale_factors', [40]),
                help="Precision for CKKS scheme"
            )

    parameters = {
        'poly_modulus_degree': poly_modulus_degree,
        'security_level': security_level
    }

    if scheme in ['BFV', 'BGV']:
        parameters['plain_modulus'] = plain_modulus
    elif scheme == 'CKKS':
        parameters['scale_factor'] = scale_factor

    return parameters


def render_operation_selector() -> Dict[str, Any]:
    """Render operation selection UI"""
    st.subheader("🧮 Select Operation")

    operation_category = st.selectbox(
        "Operation Category:",
        ["Basic Operations", "Aggregations", "Filters", "Advanced Analytics"]
    )

    if operation_category == "Basic Operations":
        operation = st.selectbox(
            "Operation:",
            ["Addition", "Subtraction", "Multiplication", "Division (Scalar)"]
        )
        operand = st.number_input("Operand:", value=1.0)

    elif operation_category == "Aggregations":
        operation = st.selectbox(
            "Aggregation:",
            ["Sum", "Mean", "Count", "Variance", "Standard Deviation"]
        )
        operand = None

    elif operation_category == "Filters":
        operation = st.selectbox(
            "Filter Type:",
            ["Greater Than", "Less Than", "Equal To", "Between"]
        )
        operand = st.number_input("Threshold:", value=0.0)

    else:  # Advanced Analytics
        operation = st.selectbox(
            "Analysis:",
            ["Group By Currency", "Transaction Patterns", "Fraud Detection Score"]
        )
        operand = None

    return {
        'category': operation_category,
        'operation': operation,
        'operand': operand
    }


def render_progress_indicator(message: str, percentage: float = None):
    """Render progress indicator"""
    if percentage is not None:
        st.progress(percentage)
    st.info(f"⏳ {message}")


def render_metrics_dashboard(metrics: Dict[str, Any]):
    """Render metrics dashboard"""
    st.subheader("📊 Performance Metrics")

    cols = st.columns(len(metrics))

    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[idx]:
            if isinstance(metric_value, dict):
                st.metric(
                    metric_name,
                    metric_value.get('value', 'N/A'),
                    delta=metric_value.get('delta')
                )
            else:
                st.metric(metric_name, metric_value)


def render_key_display(keys: Dict[str, str], include_private: bool = False):
    """Render key display with copy functionality"""
    st.subheader("🔑 Generated Keys")

    st.warning("⚠️ **Security Warning**: Save these keys securely. Never share private keys!")

    # Public Key
    with st.expander("📤 Public Key", expanded=True):
        st.text_area(
            "Public Key (Base64):",
            value=keys.get('public_key', 'Not generated'),
            height=100,
            key="public_key_display"
        )
        if st.button("📋 Copy Public Key"):
            st.success("✅ Copied to clipboard!")

    # Private Key (only if user wants to see it)
    if include_private:
        with st.expander("🔒 Private Key", expanded=False):
            st.error("⚠️ PRIVATE - Keep this secret!")
            st.text_area(
                "Private Key (Base64):",
                value=keys.get('private_key', 'Not generated'),
                height=100,
                key="private_key_display"
            )
            if st.button("📋 Copy Private Key"):
                st.success("✅ Copied to clipboard!")

    # Evaluation Keys
    if 'evaluation_key' in keys:
        with st.expander("⚙️ Evaluation Keys"):
            st.text_area(
                "Evaluation Key:",
                value=keys.get('evaluation_key', 'Not generated'),
                height=100
            )


def render_alert(message: str, alert_type: str = "info"):
    """Render colored alert box"""
    if alert_type == "success":
        st.success(f"✅ {message}")
    elif alert_type == "error":
        st.error(f"❌ {message}")
    elif alert_type == "warning":
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")


def render_column_selector(df, title: str = "Select Columns to Encrypt") -> List[str]:
    """Render column selector with recommendations"""
    st.subheader(title)

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Numeric Columns** (Recommended for FHE)")
        selected_numeric = st.multiselect(
            "Select numeric columns:",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )

    with col2:
        st.write("**Text/Categorical Columns**")
        selected_text = st.multiselect(
            "Select text columns:",
            text_cols,
            default=[]
        )

    return selected_numeric + selected_text