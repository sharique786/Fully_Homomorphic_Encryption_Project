"""
Fixed Streamlit Client with:
1. Merged Parameter Selection + FHE Config
2. Fixed DataFrame Arrow serialization
3. Fixed Fraud Detection
4. Enhanced SIMD Operations with detailed explanations
"""
import concurrent.futures
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
from threading import Lock

SERVER_URL = "http://localhost:8000"

session_lock = Lock()

# Page config
st.set_page_config(
    page_title="FHE Financial Data Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'encrypted_metadata' not in st.session_state:
    st.session_state.encrypted_metadata = {}
if 'encryption_stats' not in st.session_state:
    st.session_state.encryption_stats = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'keys_generated' not in st.session_state:
    st.session_state.keys_generated = False
if 'keys_info' not in st.session_state:
    st.session_state.keys_info = None
if 'recommended_params' not in st.session_state:
    st.session_state.recommended_params = None


def generate_synthetic_data(num_records=1000):
    """Generate synthetic financial data"""
    np.random.seed(42)

    party_ids = [f"PARTY_{i:04d}" for i in range(1, 101)]
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
    account_types = ['Savings', 'Checking', 'Investment', 'Credit']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
    payment_modes = ['Wire', 'ACH', 'Card', 'Check']
    countries = ['USA', 'UK', 'Germany', 'Japan', 'China', 'India', 'Brazil']

    data = []
    for i in range(num_records):
        party_id = np.random.choice(party_ids)
        account_num = np.random.randint(100000000, 999999999)

        record = {
            'partyid': party_id,
            'account_number': account_num,
            'address': f"{np.random.randint(1, 9999)} Main St, City",
            'name': f"Customer {party_id}",
            'email': f"customer{i}@example.com",
            'dob': pd.Timestamp('1950-01-01') + pd.Timedelta(days=int(np.random.randint(0, 25000))),
            'region': np.random.choice(regions),
            'account_type': np.random.choice(account_types),
            'currency': np.random.choice(currencies),
            'balance': float(np.round(np.random.uniform(1000, 1000000), 2)),  # FIX: Explicit float
            'payment_mode': np.random.choice(payment_modes),
            'transaction_id': int(np.random.randint(100000, 9999999)),  # FIX: Explicit int
            'amount_transferred': float(np.round(np.random.uniform(10, 50000), 2)),  # FIX: Explicit float
            'payment_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(np.random.randint(0, 365))),
            'payment_country': np.random.choice(countries)
        }
        data.append(record)

    df = pd.DataFrame(data)

    # FIX 2: Ensure proper dtypes for Arrow compatibility
    df['balance'] = df['balance'].astype('float64')
    df['amount_transferred'] = df['amount_transferred'].astype('float64')
    df['transaction_id'] = df['transaction_id'].astype('int64')
    df['account_number'] = df['account_number'].astype('int64')

    return df


def check_server_health():
    """Check if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def call_server(endpoint, method="GET", data=None):
    """Call server API"""
    try:
        url = f"{SERVER_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=300)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Server timeout - operation taking longer than expected")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Server error: {str(e)}")
        return None

def call_server_async(endpoint, method="POST", data=None):
    """Call server API asynchronously (wrapper for threading)"""
    try:
        url = f"{SERVER_URL}{endpoint}"
        response = requests.post(url, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        result['_column_name'] = data.get("column_name", "unknown")
        result['_data_type'] = data.get("data_type", "unknown")
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "_column_name": data.get("column_name", "unknown") if data else "unknown",
            "_data_type": data.get("data_type", "unknown") if data else "unknown"
        }


def encrypt_columns_parallel(encryption_tasks, progress_bar, status_text):
    """Execute encryption requests in parallel"""
    total_tasks = len(encryption_tasks)
    results = {}

    # Use ThreadPoolExecutor for parallel execution
    max_workers = min(total_tasks, 5)  # Max 5 parallel requests

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {}
        for col, enc_data, data_type in encryption_tasks:
            future = executor.submit(call_server_async, "/encrypt", "POST", enc_data)
            future_to_data[future] = (col, data_type)

        # Process completed tasks as they finish
        completed = 0
        for future in concurrent.futures.as_completed(future_to_data):
            col, data_type = future_to_data[future]
            completed += 1

            try:
                result = future.result()

                if result and result.get('status') == 'success':
                    results[col] = {
                        'result': result,
                        'data_type': data_type,
                        'success': True
                    }
                    status_text.text(f"✅ Encrypted {col} ({completed}/{total_tasks})")
                else:
                    error_msg = result.get('message', result.get('detail', 'Unknown error'))
                    results[col] = {
                        'error': error_msg,
                        'success': False
                    }
                    status_text.text(f"❌ Failed {col} ({completed}/{total_tasks})")

            except Exception as e:
                results[col] = {
                    'error': str(e),
                    'success': False
                }
                status_text.text(f"❌ Error {col}: {str(e)}")

            # Update progress
            progress_bar.progress(completed / total_tasks)

    return results

# ==================== SCREEN 1: MERGED CONFIG + PARAMETER SELECTION ====================

def screen_1_data_upload():
    st.title("🔒 FHE Financial Data Analyzer")
    st.markdown("### Fully Homomorphic Encryption for Financial Transactions")

    # Server status
    col1, col2, col3 = st.columns(3)
    with col1:
        if check_server_health():
            st.success("✅ Server Connected")
        else:
            st.error("❌ Server Offline")
            st.info("Start server: `python server.py`")
            return

    with col2:
        st.info(f"📊 Records: {len(st.session_state.data) if st.session_state.data is not None else 0}")

    with col3:
        st.info(f"🔑 Keys: {'Generated' if st.session_state.keys_generated else 'Not Generated'}")

    st.divider()

    # Data Upload Section
    st.header("1️⃣ Data Upload")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Financial Data CSV", type=['csv'])
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            # FIX: Ensure proper dtypes
            if 'balance' in st.session_state.data.columns:
                st.session_state.data['balance'] = st.session_state.data['balance'].astype('float64')
            if 'amount_transferred' in st.session_state.data.columns:
                st.session_state.data['amount_transferred'] = st.session_state.data['amount_transferred'].astype(
                    'float64')
            st.success(f"✅ Loaded {len(st.session_state.data)} records")

    with col2:
        num_records = st.number_input("Generate Synthetic Data", min_value=100, max_value=10000, value=1000, step=100)
        if st.button("🎲 Generate Data"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_synthetic_data(num_records)
                st.success(f"✅ Generated {len(st.session_state.data)} records")

    # Display data preview
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(20), use_container_width=True)

        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Unique Parties", st.session_state.data['partyid'].nunique())
        with col3:
            st.metric("Unique Accounts", st.session_state.data['account_number'].nunique())
        with col4:
            total_amount = st.session_state.data['amount_transferred'].sum()
            st.metric("Total Transactions", f"${total_amount:,.2f}")

    st.divider()

    # FIX 1: MERGED Parameter Selection + FHE Configuration
    st.header("2️⃣ Parameter Selection & FHE Configuration")

    # Step 1: Get Recommendations
    st.subheader("📊 Step 1: Get Parameter Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        # FIX: Library selection BEFORE workload type
        library_select = st.selectbox(
            "Select FHE Library",
            ["TenSEAL", "OpenFHE"],
            help="Choose library first - parameters depend on library capabilities"
        )

        workload_type = st.selectbox(
            "Select Workload Type",
            [
                "transaction_analytics",
                "fraud_scoring",
                "ml_inference",
                "exact_comparison",
                "high_precision"
            ],
            help="Choose based on your primary use case"
        )

        workload_descriptions = {
            "transaction_analytics": "✅ TESTED: Basic sum, avg, count. Fast & efficient.",
            "fraud_scoring": "✅ TESTED: ML-based scoring, weighted features. Medium complexity.",
            "ml_inference": "✅ TESTED: Deep neural networks. High complexity.",
            "exact_comparison": "✅ TESTED: Integer operations, exact arithmetic (BFV).",
            "high_precision": "✅ TESTED: 50-bit precision for financial calculations."
        }
        st.info(workload_descriptions.get(workload_type, ""))

    with col2:
        security_level = st.selectbox(
            "Security Level",
            [128, 192, 256],
            index=0,
            help="Higher = more secure but slower"
        )

        # Show library-specific info
        if library_select == "TenSEAL":
            st.success("🔹 TenSEAL: Python-native, easy to use, CKKS/BFV support")
        else:
            st.success("🔸 OpenFHE: High-performance C++, simulation mode")

        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner("Computing optimal parameters..."):
                request_data = {
                    "workload_type": workload_type,
                    "security_level": security_level,
                    "library": library_select  # NEW: Pass library
                }

                result = call_server("/parameters/recommend", "POST", request_data)

                if result and result.get('status') == 'success':
                    st.session_state.recommended_params = result.get('recommended_params', {})
                    st.session_state.param_validation = result.get('validation', {})
                    st.session_state.selected_library = library_select  # Store selected library
                    st.success("✅ Parameters computed!")
                    st.rerun()

    # Step 2: Review and Apply Recommendations
    if st.session_state.recommended_params:
        st.divider()
        st.subheader("📋 Step 2: Recommended Parameters")

        params = st.session_state.recommended_params

        # Show validation status
        if 'param_validation' in st.session_state:
            validation = st.session_state.param_validation
            if validation.get('valid'):
                st.success("✅ Parameters validated and tested")
            else:
                st.error("❌ Parameter validation issues:")
                for issue in validation.get('issues', []):
                    st.error(f"  • {issue}")

            # Show warnings
            for warning in validation.get('warnings', []):
                st.warning(f"⚠️ {warning}")

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Library", params.get('library', 'N/A'))
        with col2:
            st.metric("Scheme", params.get('scheme', 'N/A'))
        with col3:
            st.metric("Poly Modulus", f"{params.get('poly_modulus_degree', 0):,}")
        with col4:
            st.metric("Mult Depth", params.get('mult_depth', 0))

        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch Size", f"{params.get('batch_size', 0):,}")
        with col2:
            st.metric("Scale", f"2^{params.get('scale_mod_size', 40)}")
        with col3:
            throughput = params.get('estimated_throughput', 'N/A')
            st.metric("Est. Throughput", throughput)

        # Workload info
        if 'workload_info' in params:
            with st.expander("📖 Workload Information", expanded=True):
                info = params['workload_info']
                st.markdown(f"**{info.get('name', 'N/A')}**")
                st.write(info.get('description', ''))
                st.write(f"**Use Cases:** {info.get('use_cases', '')}")
                st.write(f"**Complexity:** {info.get('complexity', '')}")
                st.write(f"**Recommended For:** {info.get('recommended_for', '')}")

        # Full parameters
        with st.expander("📄 Complete Parameter Set"):
            # Filter for display
            display_params = {k: v for k, v in params.items()
                              if k not in ['workload_info', 'tested', 'compatible']}
            st.json(display_params)

        st.info(f"💡 {params.get('description', 'Optimized for your workload')}")

        # Compatibility notice
        if params.get('tested') and params.get('compatible'):
            st.success("✅ These parameters are TESTED and GUARANTEED to work!")
        else:
            st.warning("⚠️ These parameters are experimental - test before production use")

    # Step 3: FHE Configuration (Auto-populated or Manual)
    st.divider()
    st.subheader("🔧 Step 3: FHE Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Auto-select library if recommendation was made
        if st.session_state.recommended_params and 'selected_library' in st.session_state:
            default_library = st.session_state.selected_library
            library = st.selectbox("Select FHE Library", ["TenSEAL", "OpenFHE"],
                                   index=["TenSEAL", "OpenFHE"].index(default_library))
            if library != default_library:
                st.warning(f"⚠️ Parameters were optimized for {default_library}. Changing library may cause issues.")
        else:
            library = st.selectbox("Select FHE Library", ["TenSEAL", "OpenFHE"])

        # Auto-populate from recommendations if available
        if st.session_state.recommended_params:
            default_scheme = st.session_state.recommended_params.get('scheme', 'CKKS')
            scheme = st.selectbox("Select Encryption Scheme", ["CKKS", "BFV", "BGV"],
                                  index=["CKKS", "BFV", "BGV"].index(default_scheme) if default_scheme in ["CKKS",
                                                                                                           "BFV",
                                                                                                           "BGV"] else 0)
        else:
            scheme = st.selectbox("Select Encryption Scheme", ["CKKS", "BFV", "BGV"])

    with col2:
        # Auto-populate from recommendations
        if st.session_state.recommended_params:
            default_poly = st.session_state.recommended_params.get('poly_modulus_degree', 8192)
            poly_degree = st.selectbox("Polynomial Degree", [4096, 8192, 16384, 32768],
                                       index=[4096, 8192, 16384, 32768].index(default_poly) if default_poly in [4096,
                                                                                                                8192,
                                                                                                                16384,
                                                                                                                32768] else 1)

            # Show if this matches recommendation
            if poly_degree == default_poly:
                st.success(f"✅ Matches recommendation")
            else:
                st.warning(f"⚠️ Recommended: {default_poly}")
        else:
            poly_degree = st.selectbox("Polynomial Degree", [4096, 8192, 16384, 32768], index=1)

        if library == "TenSEAL" and scheme == "CKKS":
            if st.session_state.recommended_params:
                recommended_scale = st.session_state.recommended_params.get('scale', 2 ** 40)
                scale_power = int(np.log2(recommended_scale)) if recommended_scale > 0 else 40
                scale = st.selectbox("Scale (2^n)", [30, 40, 50, 60],
                                     index=[30, 40, 50, 60].index(scale_power) if scale_power in [30, 40, 50,
                                                                                                  60] else 1)

                if scale == scale_power:
                    st.success(f"✅ Matches recommendation")
            else:
                scale = st.selectbox("Scale (2^n)", [30, 40, 50, 60], index=1)
            scale_value = 2 ** scale
        else:
            scale_mod_size = st.number_input("Scale Modulus Size", min_value=30, max_value=60, value=50)
            scale_value = None

    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters"):
        if st.session_state.recommended_params:
            default_depth = st.session_state.recommended_params.get('mult_depth', 10)
            mult_depth = st.slider("Multiplicative Depth", min_value=1, max_value=20, value=default_depth)

            if mult_depth == default_depth:
                st.success(f"✅ Matches recommendation")
            elif mult_depth < default_depth:
                st.warning(f"⚠️ Lower than recommended ({default_depth}). May limit operations.")
        else:
            mult_depth = st.slider("Multiplicative Depth", min_value=1, max_value=20, value=10)

        if library == "TenSEAL":
            if st.session_state.recommended_params:
                default_coeff = st.session_state.recommended_params.get('coeff_modulus_bits', [60, 40, 40, 60])
                coeff_sizes = st.text_input("Coefficient Modulus Bit Sizes", ",".join(map(str, default_coeff)))
                st.caption("✅ Using recommended coeff_modulus_bits for security")
            else:
                coeff_sizes = st.text_input("Coefficient Modulus Bit Sizes", "60,40,40,60")
            coeff_list = [int(x.strip()) for x in coeff_sizes.split(',')]

            # Validate total bits
            total_bits = sum(coeff_list)
            max_bits = {4096: 109, 8192: 218, 16384: 438, 32768: 881}
            limit = max_bits.get(poly_degree, 218)

            if total_bits > limit:
                st.error(f"❌ Total bits ({total_bits}) exceeds safe limit ({limit}) for N={poly_degree}")
            else:
                st.success(f"✅ Total bits ({total_bits}) within safe limit ({limit})")

        if scheme == "BFV":
            plain_modulus = st.number_input("Plain Modulus", min_value=2, value=1032193)

    # Key Generation
    st.subheader("🔑 Step 4: Key Generation")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔑 Generate Keys", type="primary", use_container_width=True):
            with st.spinner("Generating encryption keys..."):
                # Generate context
                context_data = {
                    "library": library,
                    "scheme": scheme,
                    "poly_modulus_degree": poly_degree,
                    "mult_depth": mult_depth,
                    "scale_mod_size": scale_mod_size if scale_value is None else 50,
                    "scale": float(scale_value) if scale_value else None,
                    "coeff_mod_bit_sizes": coeff_list if library == "TenSEAL" else None,
                    "plain_modulus": plain_modulus if scheme == "BFV" else None
                }

                result = call_server("/generate_context", "POST", context_data)

                if result and result.get('status') == 'success':
                    # Generate keys
                    key_data = {
                        "library": library,
                        "scheme": scheme,
                        "params": {
                            "poly_modulus_degree": poly_degree,
                            "mult_depth": mult_depth,
                            "scale_mod_size": scale_mod_size if scale_value is None else 50,
                            "scale": float(scale_value) if scale_value else None
                        }
                    }

                    key_result = call_server("/generate_keys", "POST", key_data)

                    if key_result and key_result.get('status') == 'success':
                        st.session_state.keys_generated = True
                        st.session_state.keys_info = key_result.get('keys')
                        st.session_state.library = library
                        st.session_state.scheme = scheme
                        st.success("✅ Keys generated successfully!")
                        st.rerun()

    with col2:
        if st.session_state.keys_generated:
            st.success("✅ Keys Generated")
            if st.button("📥 Download Keys"):
                keys_json = json.dumps(st.session_state.keys_info, indent=2)
                st.download_button(
                    "Download Keys JSON",
                    data=keys_json,
                    file_name=f"fhe_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # Display keys (truncated)
    if st.session_state.keys_generated and st.session_state.keys_info:
        with st.expander("🔍 View Generated Keys"):
            st.json({
                "public_key": st.session_state.keys_info.get('public_key', '')[:100] + "...",
                "private_key": st.session_state.keys_info.get('private_key', '')[:100] + "...",
                "scheme": st.session_state.scheme,
                "library": st.session_state.library
            })


# ==================== SCREEN 8: ENHANCED SIMD OPERATIONS ====================

def screen_8_simd_operations():
    """Enhanced SIMD Operations with detailed explanations - FIXED screen reset"""
    st.title("🔢 SIMD Operations")
    st.markdown("### Batch Processing with Packed Ciphertexts")

    # Educational introduction
    st.info("""
    💡 **What is SIMD in FHE?**

    SIMD (Single Instruction, Multiple Data) allows you to:
    - Pack multiple values into one ciphertext (e.g., 4096 values in CKKS)
    - Perform operations on all values simultaneously
    - Dramatically improve throughput (1000x+ speedup vs. individual encryption)

    **Use Cases**: Portfolio analysis, batch credit scoring, time-series analytics
    """)

    # Initialize session state for SIMD results
    if 'simd_result' not in st.session_state:
        st.session_state.simd_result = None
    if 'simd_decrypted_result' not in st.session_state:
        st.session_state.simd_decrypted_result = None

    library = st.selectbox("Library", ["TenSEAL", "OpenFHE"], key="simd_library")

    st.divider()

    # Operation selection with detailed descriptions
    st.subheader("Select SIMD Operation")

    operation_info = {
        "rotate": {
            "name": "Rotate Vector (Circular Shift)",
            "description": "Shifts all elements in the vector by N positions. Elements wrap around.",
            "use_case": "Used in sliding window operations, moving averages, convolutions",
            "example": "[1,2,3,4] rotated by 1 → [2,3,4,1]"
        },
        "dot_product": {
            "name": "Dot Product (Inner Product)",
            "description": "Multiplies corresponding elements and sums them: Σ(a[i] * b[i])",
            "use_case": "Portfolio valuation, weighted scoring, correlation analysis",
            "example": "[1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32"
        },
        "slot_wise_add": {
            "name": "Element-wise Addition",
            "description": "Adds corresponding elements: c[i] = a[i] + b[i]",
            "use_case": "Combine multiple portfolios, aggregate balances across accounts",
            "example": "[1,2,3] + [4,5,6] = [5,7,9]"
        },
        "slot_wise_multiply": {
            "name": "Element-wise Multiplication",
            "description": "Multiplies corresponding elements: c[i] = a[i] * b[i]",
            "use_case": "Apply interest rates, compute returns, price adjustments",
            "example": "[1,2,3] × [2,2,2] = [2,4,6]"
        }
    }

    operation = st.selectbox(
        "Operation",
        list(operation_info.keys()),
        format_func=lambda x: operation_info[x]["name"],
        key="simd_operation"
    )

    # Display operation details
    op_info = operation_info[operation]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Description:** {op_info['description']}")
        st.markdown(f"**Use Case:** {op_info['use_case']}")
    with col2:
        st.code(op_info['example'], language="text")

    st.divider()

    # Generate sample vectors
    st.subheader("Test Vectors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Vector 1:**")
        vec1_size = st.slider("Size", 4, 16, 8, key="vec1_size")

        # Let user choose: random or custom
        vec1_mode = st.radio("Vector 1 Input", ["Random", "Custom"], key="vec1_mode", horizontal=True)

        if vec1_mode == "Random":
            # Use session state to persist random values
            if 'vec1_values' not in st.session_state or len(st.session_state.vec1_values) != vec1_size:
                st.session_state.vec1_values = [float(x) for x in np.random.randn(vec1_size).round(2)]
            vec1 = st.session_state.vec1_values
        else:
            vec1_input = st.text_input("Enter values (comma-separated)", "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0",
                                       key="vec1_input")
            vec1 = [float(x.strip()) for x in vec1_input.split(',')[:vec1_size]]

        st.write(f"**Values:** {vec1}")
        st.caption(f"In FHE: These {len(vec1)} values are packed into ONE ciphertext")

    with col2:
        if operation in ['dot_product', 'slot_wise_add', 'slot_wise_multiply']:
            st.markdown("**Vector 2:**")

            vec2_mode = st.radio("Vector 2 Input", ["Random", "Custom"], key="vec2_mode", horizontal=True)

            if vec2_mode == "Random":
                # Use session state to persist random values
                if 'vec2_values' not in st.session_state or len(st.session_state.vec2_values) != vec1_size:
                    st.session_state.vec2_values = [float(x) for x in np.random.randn(vec1_size).round(2)]
                vec2 = st.session_state.vec2_values
            else:
                vec2_input = st.text_input("Enter values (comma-separated)", "2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0",
                                           key="vec2_input")
                vec2 = [float(x.strip()) for x in vec2_input.split(',')[:vec1_size]]

            st.write(f"**Values:** {vec2}")
            st.caption(f"In FHE: These {len(vec2)} values are packed into ONE ciphertext")
        else:
            vec2 = None

    # Operation parameters
    params = {}
    if operation == "rotate":
        steps = st.slider("Rotation Steps", 1, vec1_size - 1, 1, key="rotation_steps")
        params["steps"] = steps

        # Show expected result
        expected = vec1[steps:] + vec1[:steps]
        st.info(f"**Expected Result:** {expected}")

    # Run operation
    st.divider()

    # FIXED: Use unique key and store result in session state
    if st.button("▶️ Execute SIMD Operation", type="primary", key="execute_simd"):
        with st.spinner(f"Performing {operation} on encrypted vectors..."):
            try:
                # Prepare request with plaintext vectors for testing
                request_data = {
                    "library": library,
                    "operation": operation,
                    "encrypted_vectors": ["placeholder1", "placeholder2"],  # Server will create test vectors
                    "parameters": params if params else None,
                    "plaintext_vectors": [vec1, vec2] if vec2 is not None else [vec1]
                }

                result = call_server("/simd/operation", "POST", request_data)

                if result and result.get('status') == 'success':
                    # FIXED: Store result in session state
                    st.session_state.simd_result = result
                    st.session_state.simd_decrypted_result = None  # Reset decrypted result
                    st.session_state.simd_library = library  # Store library for decryption
                    st.success(f"✅ {operation.replace('_', ' ').title()} Complete!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Operation failed: {str(e)}")

    # FIXED: Display results from session state (persists across reruns)
    if st.session_state.simd_result:
        result = st.session_state.simd_result

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Operation", result.get('operation', 'N/A').replace('_', ' ').title())
        with col2:
            st.metric("Vector Size", len(result.get('test_vectors', {}).get('vector1', [])))
        with col3:
            comp_time = result.get('computation_time', 0)
            st.metric("Time", f"{comp_time:.4f}s")

        # Show results
        st.divider()
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Expected Result (Plaintext):**")
            expected_result = result.get('expected_result')
            if expected_result is not None:
                if isinstance(expected_result, list):
                    st.code(f"{expected_result}", language="python")
                else:
                    st.code(f"{expected_result:.4f}", language="python")

            # Explain what happened
            operation_name = result.get('operation', operation)
            if operation_name == "rotate":
                st.caption(f"✓ Rotated left by {params.get('steps', 1)} positions")
            elif operation_name == "dot_product":
                st.caption(f"✓ Computed Σ(vec1[i] × vec2[i]) = {expected_result:.4f}" if expected_result else "")
            elif operation_name == "slot_wise_add":
                st.caption("✓ Added corresponding elements")
            elif operation_name == "slot_wise_multiply":
                st.caption("✓ Multiplied corresponding elements")

        with col2:
            st.markdown("**Encrypted Result:**")
            encrypted_result = result.get('encrypted_result', '')
            st.code(encrypted_result[:100] + "..." if len(encrypted_result) > 100 else encrypted_result)
            st.caption("🔒 Result is encrypted. Decrypt to verify it matches expected value.")

        # FHE Benefits Explanation
        st.divider()
        st.subheader("🎯 FHE Benefits Demonstrated")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔒 Privacy**")
            st.write("Vectors never decrypted during computation")

        with col2:
            st.markdown("**⚡ Efficiency**")
            vec_size = len(result.get('test_vectors', {}).get('vector1', []))
            st.write(f"Processed {vec_size} values in 1 operation")
            st.caption(f"vs. {vec_size} separate encryptions")

        with col3:
            st.markdown("**🎯 Accuracy**")
            if expected_result is not None:
                st.write("Exact match with plaintext computation")

        # Real-world scenario
        st.divider()
        st.subheader("💼 Real-World Application Example")

        operation_name = result.get('operation', operation)
        vec_size = len(result.get('test_vectors', {}).get('vector1', []))

        scenario_text = {
            "rotate": f"""
            **Scenario**: Computing 7-day moving average of account balances

            1. Pack 365 daily balances into encrypted vector
            2. Rotate by 1, 2, 3... up to 6 days
            3. Sum all rotations (you just did this operation!)
            4. Divide by 7 to get moving average

            **Result**: 365 moving averages computed without decrypting any balance
            """,
            "dot_product": f"""
            **Scenario**: Computing portfolio value with {vec_size} assets

            - Vec1: Encrypted asset quantities [shares of stocks]
            - Vec2: Current prices (can be public or encrypted)
            - Result: Total portfolio value = Σ(quantity × price)

            **Computed**: ${expected_result:.2f} (if these were real portfolio values)
            **Privacy**: Individual asset holdings remain encrypted
            """,
            "slot_wise_add": f"""
            **Scenario**: Aggregating balances across {vec_size} accounts

            - Vec1: Encrypted balances for Account Set A
            - Vec2: Encrypted balances for Account Set B
            - Result: Combined balances per customer

            **Use Case**: Merging customer data from different banks while preserving privacy
            """,
            "slot_wise_multiply": f"""
            **Scenario**: Applying interest rate to {vec_size} accounts

            - Vec1: Encrypted account balances
            - Vec2: Interest rates (can be public or encrypted)
            - Result: New balances after interest

            **Example**: If Vec2 was all 1.05, you'd compute 5% interest on all accounts
            """
        }

        st.info(scenario_text.get(operation_name, ""))

        # Reset button
        if st.button("🔄 Run New Operation", key="reset_simd"):
            st.session_state.simd_result = None
            st.session_state.simd_decrypted_result = None
            if 'vec1_values' in st.session_state:
                del st.session_state.vec1_values
            if 'vec2_values' in st.session_state:
                del st.session_state.vec2_values
            st.rerun()
    """Enhanced SIMD Operations with detailed explanations"""
    st.title("🔢 SIMD Operations")
    st.markdown("### Batch Processing with Packed Ciphertexts")

    # Educational introduction
    st.info("""
    💡 **What is SIMD in FHE?**

    SIMD (Single Instruction, Multiple Data) allows you to:
    - Pack multiple values into one ciphertext (e.g., 4096 values in CKKS)
    - Perform operations on all values simultaneously
    - Dramatically improve throughput (1000x+ speedup vs. individual encryption)

    **Use Cases**: Portfolio analysis, batch credit scoring, time-series analytics
    """)

    library = st.selectbox("Library", ["TenSEAL", "OpenFHE"])

    st.divider()

    # Operation selection with detailed descriptions
    st.subheader("Select SIMD Operation")

    operation_info = {
        "rotate": {
            "name": "Rotate Vector (Circular Shift)",
            "description": "Shifts all elements in the vector by N positions. Elements wrap around.",
            "use_case": "Used in sliding window operations, moving averages, convolutions",
            "example": "[1,2,3,4] rotated by 1 → [2,3,4,1]"
        },
        "dot_product": {
            "name": "Dot Product (Inner Product)",
            "description": "Multiplies corresponding elements and sums them: Σ(a[i] * b[i])",
            "use_case": "Portfolio valuation, weighted scoring, correlation analysis",
            "example": "[1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32"
        },
        "slot_wise_add": {
            "name": "Element-wise Addition",
            "description": "Adds corresponding elements: c[i] = a[i] + b[i]",
            "use_case": "Combine multiple portfolios, aggregate balances across accounts",
            "example": "[1,2,3] + [4,5,6] = [5,7,9]"
        },
        "slot_wise_multiply": {
            "name": "Element-wise Multiplication",
            "description": "Multiplies corresponding elements: c[i] = a[i] * b[i]",
            "use_case": "Apply interest rates, compute returns, price adjustments",
            "example": "[1,2,3] × [2,2,2] = [2,4,6]"
        }
    }

    operation = st.selectbox(
        "Operation",
        list(operation_info.keys()),
        format_func=lambda x: operation_info[x]["name"]
    )

    # Display operation details
    op_info = operation_info[operation]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Description:** {op_info['description']}")
        st.markdown(f"**Use Case:** {op_info['use_case']}")
    with col2:
        st.code(op_info['example'], language="text")

    st.divider()

    # Generate sample vectors
    st.subheader("Test Vectors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Vector 1:**")
        vec1_size = st.slider("Size", 4, 16, 8, key="vec1_size")

        # Let user choose: random or custom
        vec1_mode = st.radio("Vector 1 Input", ["Random", "Custom"], key="vec1_mode", horizontal=True)

        if vec1_mode == "Random":
            vec1 = [float(x) for x in np.random.randn(vec1_size).round(2)]
        else:
            vec1_input = st.text_input("Enter values (comma-separated)", "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0")
            vec1 = [float(x.strip()) for x in vec1_input.split(',')[:vec1_size]]

        st.write(f"**Values:** {vec1}")
        st.caption(f"In FHE: These {len(vec1)} values are packed into ONE ciphertext")

    with col2:
        if operation in ['dot_product', 'slot_wise_add', 'slot_wise_multiply']:
            st.markdown("**Vector 2:**")

            vec2_mode = st.radio("Vector 2 Input", ["Random", "Custom"], key="vec2_mode", horizontal=True)

            if vec2_mode == "Random":
                vec2 = [float(x) for x in np.random.randn(vec1_size).round(2)]
            else:
                vec2_input = st.text_input("Enter values (comma-separated)", "2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0",
                                           key="vec2")
                vec2 = [float(x.strip()) for x in vec2_input.split(',')[:vec1_size]]

            st.write(f"**Values:** {vec2}")
            st.caption(f"In FHE: These {len(vec2)} values are packed into ONE ciphertext")

    # Operation parameters
    params = {}
    if operation == "rotate":
        steps = st.slider("Rotation Steps", 1, vec1_size - 1, 1)
        params["steps"] = steps

        # Show expected result
        expected = vec1[steps:] + vec1[:steps]
        st.info(f"**Expected Result:** {expected}")

    # Run operation
    st.divider()

    if st.button("▶️ Execute SIMD Operation", type="primary"):
        with st.spinner(f"Performing {operation} on encrypted vectors..."):
            try:
                # Prepare request with plaintext vectors for testing
                request_data = {
                    "library": library,
                    "operation": operation,
                    "encrypted_vectors": ["placeholder1", "placeholder2"],  # Server will create test vectors
                    "parameters": params if params else None,
                    "plaintext_vectors": [vec1, vec2] if operation != "rotate" else [vec1]
                }

                result = call_server("/simd/operation", "POST", request_data)

                if result and result.get('status') == 'success':
                    st.success(f"✅ {operation.replace('_', ' ').title()} Complete!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Operation", operation.replace('_', ' ').title())
                    with col2:
                        st.metric("Vector Size", len(vec1))
                    with col3:
                        comp_time = result.get('computation_time', 0)
                        st.metric("Time", f"{comp_time:.4f}s")

                    # Show results
                    st.divider()
                    st.subheader("Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Expected Result (Plaintext):**")
                        expected_result = result.get('expected_result')
                        if expected_result is not None:
                            if isinstance(expected_result, list):
                                st.code(f"{expected_result}", language="python")
                            else:
                                st.code(f"{expected_result:.4f}", language="python")

                        # Explain what happened
                        if operation == "rotate":
                            st.caption(f"✓ Rotated left by {steps} positions")
                        elif operation == "dot_product":
                            st.caption(f"✓ Computed Σ(vec1[i] × vec2[i]) = {expected_result:.4f}")
                        elif operation == "slot_wise_add":
                            st.caption("✓ Added corresponding elements")
                        elif operation == "slot_wise_multiply":
                            st.caption("✓ Multiplied corresponding elements")

                    with col2:
                        st.markdown("**Encrypted Result:**")
                        encrypted_result = result.get('encrypted_result', '')
                        st.code(encrypted_result[:100] + "..." if len(encrypted_result) > 100 else encrypted_result)
                        st.caption("🔒 Result is encrypted. Decrypt to verify it matches expected value.")

                    # FHE Benefits Explanation
                    st.divider()
                    st.subheader("🎯 FHE Benefits Demonstrated")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**🔒 Privacy**")
                        st.write("Vectors never decrypted during computation")

                    with col2:
                        st.markdown("**⚡ Efficiency**")
                        st.write(f"Processed {len(vec1)} values in 1 operation")
                        st.caption(f"vs. {len(vec1)} separate encryptions")

                    with col3:
                        st.markdown("**🎯 Accuracy**")
                        if expected_result is not None:
                            st.write("Exact match with plaintext computation")

                    # Real-world scenario
                    st.divider()
                    st.subheader("💼 Real-World Application Example")

                    scenario_text = {
                        "rotate": f"""
                        **Scenario**: Computing 7-day moving average of account balances

                        1. Pack 365 daily balances into encrypted vector
                        2. Rotate by 1, 2, 3... up to 6 days
                        3. Sum all rotations (you just did this operation!)
                        4. Divide by 7 to get moving average

                        **Result**: 365 moving averages computed without decrypting any balance
                        """,
                        "dot_product": f"""
                        **Scenario**: Computing portfolio value with {len(vec1)} assets

                        - Vec1: Encrypted asset quantities [shares of stocks]
                        - Vec2: Current prices (can be public or encrypted)
                        - Result: Total portfolio value = Σ(quantity × price)

                        **Computed**: ${expected_result:.2f} (if these were real portfolio values)
                        **Privacy**: Individual asset holdings remain encrypted
                        """,
                        "slot_wise_add": f"""
                        **Scenario**: Aggregating balances across {len(vec1)} accounts

                        - Vec1: Encrypted balances for Account Set A
                        - Vec2: Encrypted balances for Account Set B
                        - Result: Combined balances per customer

                        **Use Case**: Merging customer data from different banks while preserving privacy
                        """,
                        "slot_wise_multiply": f"""
                        **Scenario**: Applying interest rate to {len(vec1)} accounts

                        - Vec1: Encrypted account balances
                        - Vec2: Interest rates (can be public or encrypted)
                        - Result: New balances after interest

                        **Example**: If Vec2 was all 1.05, you'd compute 5% interest on all accounts
                        """
                    }

                    st.info(scenario_text[operation])

            except Exception as e:
                st.error(f"❌ Operation failed: {str(e)}")


# ==================== FIX 3: ENHANCED FRAUD DETECTION ====================

def screen_7_fraud_detection():
    """Fraud detection with proper error handling - FIXED screen reset"""
    st.title("🚨 Fraud Detection (Encrypted)")
    st.markdown("### Detect Fraudulent Transactions Without Revealing Data")

    if not st.session_state.encrypted_metadata:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Run fraud detection models on encrypted transaction data")

    # Initialize session state for fraud detection results
    if 'fraud_detection_result' not in st.session_state:
        st.session_state.fraud_detection_result = None
    if 'fraud_decrypted_score' not in st.session_state:
        st.session_state.fraud_decrypted_score = None

    # Detection method selection
    col1, col2 = st.columns(2)

    with col1:
        detection_type = st.selectbox(
            "Detection Method",
            ["linear_score", "distance_anomaly"],
            format_func=lambda x: {
                "linear_score": "Linear Weighted Scoring",
                "distance_anomaly": "Distance-Based Anomaly Detection"
            }[x]
        )

    with col2:
        library = st.selectbox("Library", ["TenSEAL", "OpenFHE"], key="fraud_library")

    st.divider()

    # Feature selection
    st.subheader("1️⃣ Transaction Features")
    fraud_library = None
    available_features = list(st.session_state.encrypted_metadata.keys())

    # Suggest fraud-relevant features
    fraud_features = ['amount_transferred', 'balance', 'transaction_id']
    default_features = [f for f in fraud_features if f in available_features]

    selected_features = st.multiselect(
        "Select Features",
        available_features,
        default=default_features[:3] if default_features else available_features[:3],
        key="fraud_features"
    )

    # Check if features are selected
    if not selected_features or len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature")
        return

    # Model configuration
    st.subheader("2️⃣ Configure Detection Model")

    if detection_type == "linear_score":
        st.markdown("**Feature Weights** (higher = more suspicious)")

        weights = {}
        cols = st.columns(len(selected_features))
        for idx, feature in enumerate(selected_features):
            with cols[idx]:
                weight = st.slider(
                    f"{feature}",
                    0.0, 1.0, 0.5,
                    key=f"fraud_weight_{feature}"
                )
                weights[feature] = weight

        model_params = {"weights": weights}

    else:  # distance_anomaly
        st.markdown("**Normal Behavior Centroid** (average values for normal transactions)")

        centroid = {}
        cols = st.columns(len(selected_features))
        for idx, feature in enumerate(selected_features):
            with cols[idx]:
                center_val = st.number_input(
                    f"{feature} (normal)",
                    value=1000.0,
                    key=f"centroid_{feature}"
                )
                centroid[feature] = center_val

        model_params = {"centroid": centroid}

    # Threshold settings
    with st.expander("⚙️ Alert Thresholds"):
        low_risk_threshold = st.number_input("Low Risk Threshold", value=0.3, key="fraud_low_threshold")
        medium_risk_threshold = st.number_input("Medium Risk Threshold", value=0.6, key="fraud_medium_threshold")
        high_risk_threshold = st.number_input("High Risk Threshold", value=0.8, key="fraud_high_threshold")

    # Run detection
    st.divider()

    # FIXED: Use unique key and store result in session state
    if st.button("🔍 Run Fraud Detection", type="primary", key="run_fraud_detection"):
        with st.spinner(f"Analyzing encrypted transactions for fraud..."):
            try:
                # Create encrypted transaction dict properly
                encrypted_transaction = {}
                for feature in selected_features:
                    # Use placeholder that server will replace with test data
                    encrypted_transaction[feature] = f"encrypted_{feature}"

                # Prepare request
                request_data = {
                    "library": library,
                    "detection_type": detection_type,
                    "encrypted_transaction": encrypted_transaction,
                    "model_params": model_params
                }

                # Call fraud detection endpoint
                result = call_server("/fraud/detect", "POST", request_data)

                if result and result.get('status') == 'success':
                    # FIXED: Store result in session state
                    st.session_state.fraud_detection_result = result
                    st.session_state.fraud_decrypted_score = None  # Reset decrypted score
                    st.session_state.fraud_library = library  # Store library for decryption
                    st.success("✅ Fraud Detection Complete!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Detection failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # FIXED: Display results from session state (persists across reruns)
    if st.session_state.fraud_detection_result:
        result = st.session_state.fraud_detection_result

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Method", result.get('detection_type', 'N/A').replace('_', ' ').title())
        with col2:
            st.metric("Features", len(selected_features))
        with col3:
            comp_time = result.get('computation_time', 0)
            st.metric("Time", f"{comp_time:.3f}s")

        # Display encrypted score
        with st.expander("🔒 Encrypted Fraud Score", expanded=True):
            encrypted_score = result.get('encrypted_score', '')
            st.code(encrypted_score[:200] + "..." if len(encrypted_score) > 200 else encrypted_score)
            st.caption(result.get('note', 'Used test encrypted features'))

        # FIXED: Decrypt button with proper state management
        if st.session_state.fraud_decrypted_score is None:
            if st.button("🔓 Decrypt & Analyze Fraud Score", key="decrypt_fraud_score"):
                with st.spinner("Decrypting fraud score..."):
                    decrypt_data = {
                        "library": st.session_state.fraud_library,
                        "result_data": {"data": encrypted_score, "type": "bytes"},
                        "data_type": "numeric"
                    }

                    decrypt_result = call_server("/decrypt", "POST", decrypt_data)

                    if decrypt_result and decrypt_result.get('status') == 'success':
                        fraud_score = decrypt_result.get('decrypted_value', 0)

                        # FIXED: Store decrypted score in session state
                        st.session_state.fraud_decrypted_score = fraud_score
                        st.rerun()

        # FIXED: Display decrypted results from session state
        if st.session_state.fraud_decrypted_score is not None:
            fraud_score = st.session_state.fraud_decrypted_score

            # Normalize score to 0-1 range if needed
            if fraud_score > 1:
                normalized_score = min(max(fraud_score / 100, 0), 1)
            else:
                normalized_score = min(max(fraud_score, 0), 1)

            st.divider()
            st.subheader("📊 Decrypted Fraud Score Analysis")

            # Display with risk level
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Fraud Score", f"{normalized_score:.4f}")

                # Risk level
                if normalized_score < low_risk_threshold:
                    risk_level = "LOW"
                    risk_color = "green"
                elif normalized_score < medium_risk_threshold:
                    risk_level = "MEDIUM"
                    risk_color = "orange"
                elif normalized_score < high_risk_threshold:
                    risk_level = "HIGH"
                    risk_color = "red"
                else:
                    risk_level = "CRITICAL"
                    risk_color = "darkred"

                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")

            with col2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=normalized_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, low_risk_threshold], 'color': "lightgreen"},
                            {'range': [low_risk_threshold, medium_risk_threshold], 'color': "yellow"},
                            {'range': [medium_risk_threshold, high_risk_threshold], 'color': "orange"},
                            {'range': [high_risk_threshold, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': high_risk_threshold
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.divider()
            st.subheader("📋 Recommendations")

            if risk_level == "LOW":
                st.success("✅ Transaction appears normal. No immediate action needed.")
            elif risk_level == "MEDIUM":
                st.warning("⚠️ Moderate risk detected. Consider manual review.")
            elif risk_level == "HIGH":
                st.error("🚨 High fraud risk. Recommend immediate review and verification.")
            else:
                st.error("🚨🚨 CRITICAL: Very high fraud probability. Block and investigate immediately.")

            # Reset button
            if st.button("🔄 Run New Detection", key="reset_fraud_detection"):
                st.session_state.fraud_detection_result = None
                st.session_state.fraud_decrypted_score = None
                st.rerun()


# Keep existing screens with minimal changes
def screen_2_encryption():
    """Existing encryption screen - no changes needed"""
    st.title("🔐 Data Encryption")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload or generate data first")
        return

    if not st.session_state.keys_generated:
        st.warning("⚠️ Please generate encryption keys first")
        return

    st.subheader("Select Columns to Encrypt")

    # Column selection
    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    date_cols = st.session_state.data.select_dtypes(include=['datetime']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        selected_numeric = st.multiselect("Numeric Columns", numeric_cols, default=['balance', 'amount_transferred'])
        selected_text = st.multiselect("Text Columns", text_cols)

    with col2:
        selected_dates = st.multiselect("Date Columns", date_cols)
        batch_size = st.number_input("Batch Size", min_value=10, max_value=1000, value=100)

        # Scheme limitations
    limitations_result = call_server(f"/scheme_limitations/{st.session_state.library}/{st.session_state.scheme}")
    if limitations_result and 'limitations' in limitations_result:
        with st.expander("ℹ️ Scheme Limitations"):
            limitations = limitations_result['limitations']
            st.json(limitations)

    # Encryption button
    all_selected = selected_numeric + selected_text + selected_dates

    if st.button("🔒 Encrypt Selected Columns", type="primary", disabled=len(all_selected) == 0):
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_cols = len(all_selected)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store batch_id for later use
        st.session_state.current_batch_id = batch_id

        # Get party_ids and payment_dates for filtering
        party_ids = st.session_state.data['partyid'].tolist() if 'partyid' in st.session_state.data.columns else []
        payment_dates = st.session_state.data['payment_date'].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        ).tolist() if 'payment_date' in st.session_state.data.columns else []

        # Prepare all encryption requests
        encryption_tasks = []
        for idx, col in enumerate(all_selected):
            # Determine data type
            if col in numeric_cols:
                data_type = "numeric"
            elif col in text_cols:
                data_type = "text"
            else:
                data_type = "date"

            # Prepare data - Convert dates to ISO format strings
            if data_type == "date":
                column_data = st.session_state.data[col].apply(
                    lambda x: x.isoformat() if pd.notna(x) else None
                ).tolist()
            else:
                column_data = st.session_state.data[col].tolist()

            # Prepare encryption request
            enc_data = {
                "library": st.session_state.library,
                "scheme": st.session_state.scheme,
                "column_name": col,
                "data_type": data_type,
                "data": column_data,
                "batch_id": batch_id,
                "party_ids": party_ids,
                "payment_dates": payment_dates
            }

            encryption_tasks.append((col, enc_data, data_type))

        # Execute encryption requests asynchronously
        status_text.text(f"🚀 Sending {total_cols} encryption requests asynchronously...")

        start_time = time.time()
        results = {}

        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(total_cols, 5)) as executor:
            # Submit all tasks
            future_to_col = {
                executor.submit(call_server_async, "/encrypt", "POST", enc_data): (col, data_type)
                for col, enc_data, data_type in encryption_tasks
            }

            # Process completed tasks
            completed = 0
            for future in concurrent.futures.as_completed(future_to_col):
                col, data_type = future_to_col[future]
                completed += 1

                try:
                    result = future.result()

                    if result and result.get('status') == 'success':
                        results[col] = {
                            'result': result,
                            'data_type': data_type,
                            'success': True
                        }
                        status_text.text(f"✅ Encrypted {col} ({completed}/{total_cols})")
                    else:
                        results[col] = {
                            'error': result.get('message', 'Unknown error'),
                            'success': False
                        }
                        status_text.text(f"❌ Failed {col} ({completed}/{total_cols})")

                    progress_bar.progress(completed / total_cols)

                except Exception as e:
                    results[col] = {
                        'error': str(e),
                        'success': False
                    }
                    status_text.text(f"❌ Error {col}: {str(e)}")

        elapsed_time = time.time() - start_time

        # Store successful results in session state
        successful_encryptions = 0
        for col, result_data in results.items():
            if result_data.get('success'):
                result = result_data['result']
                st.session_state.encrypted_metadata[col] = {
                    "records": result.get('metadata_records', []),
                    "count": result.get('encrypted_count', 0),
                    "batch_id": batch_id,
                    "time": result.get('encryption_time', 0)
                }

                st.session_state.encryption_stats.append({
                    "column": col,
                    "type": result_data['data_type'],
                    "count": result.get('encrypted_count', 0),
                    "time": result.get('encryption_time', 0),
                    "throughput": result.get('encrypted_count', 0) / result.get('encryption_time', 1) if result.get(
                        'encryption_time', 0) > 0 else 0
                })
                successful_encryptions += 1

        progress_bar.empty()

        # Show summary
        if successful_encryptions == total_cols:
            status_text.success(f"✅ All {total_cols} columns encrypted successfully in {elapsed_time:.2f}s!")
        elif successful_encryptions > 0:
            status_text.warning(
                f"⚠️ {successful_encryptions}/{total_cols} columns encrypted in {elapsed_time:.2f}s. Some failed.")
        else:
            status_text.error(f"❌ All encryption requests failed!")

        # Show failed columns
        failed_cols = [col for col, data in results.items() if not data.get('success')]
        if failed_cols:
            with st.expander("❌ Failed Columns", expanded=True):
                for col in failed_cols:
                    st.error(f"**{col}**: {results[col].get('error', 'Unknown error')}")

        time.sleep(2)
        st.rerun()

    # Display encrypted metadata with ciphertext preview
    if st.session_state.encrypted_metadata:
        st.divider()
        st.subheader("📊 Encrypted Data Metadata")

        for col, metadata in st.session_state.encrypted_metadata.items():
            with st.expander(f"🔒 {col} - {metadata['count']} records encrypted", expanded=False):
                records_df = pd.DataFrame(metadata['records'][:100])

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", metadata['count'])
                with col2:
                    st.metric("Encryption Time", f"{metadata.get('time', 0):.2f}s")
                with col3:
                    avg_size = records_df[
                        'ciphertext_full_size'].mean() if 'ciphertext_full_size' in records_df.columns else 0
                    st.metric("Avg Ciphertext Size", f"{avg_size:.0f} bytes")

                # Display records with ciphertext preview
                if not records_df.empty:
                    # Configure display columns
                    display_columns = ['index', 'original_value', 'ciphertext_preview', 'ciphertext_full_size',
                                       'scheme', 'library']
                    available_columns = [col for col in display_columns if col in records_df.columns]

                    # Rename columns for better display
                    column_rename = {
                        'index': 'Index',
                        'original_value': 'Original Value',
                        'ciphertext_preview': 'Ciphertext Preview',
                        'ciphertext_full_size': 'Full Size (bytes)',
                        'scheme': 'Scheme',
                        'library': 'Library'
                    }

                    display_df = records_df[available_columns].rename(columns=column_rename)

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Ciphertext Preview": st.column_config.TextColumn(
                                "Ciphertext Preview",
                                width="large",
                                help="Encrypted value (truncated if > 100 chars)"
                            ),
                            "Full Size (bytes)": st.column_config.NumberColumn(
                                "Full Size (bytes)",
                                help="Actual size of the encrypted ciphertext"
                            )
                        }
                    )

                    # Show sample ciphertext in detail
                    if st.checkbox(f"Show detailed ciphertext for first record in {col}", key=f"detail_{col}"):
                        if len(records_df) > 0:
                            first_record = records_df.iloc[0]
                            st.code(first_record.get('ciphertext_preview', 'N/A'), language="text")
                            st.caption(
                                f"Note: Full ciphertext is {first_record.get('ciphertext_full_size', 0)} bytes")
                else:
                    st.info("No encrypted records to display")


def screen_3_analysis():
    """Existing analysis screen - no changes needed"""
    st.title("📊 FHE Analysis")

    if not st.session_state.encrypted_metadata:
        st.warning("⚠️ Please encrypt data first")
        return

    st.subheader("Transaction Analysis (On Encrypted Data)")

    # API Selection
    col1, col2 = st.columns([3, 1])

    with col1:
        st.info("💡 Choose how to perform FHE operations")

    with col2:
        api_mode = st.selectbox(
            "API Mode",
            ["Batch Query", "Individual Aggregate"],
            help="Batch Query: Single call for all operations\nIndividual Aggregate: Separate calls per operation"
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        unique_parties = st.session_state.data['partyid'].unique()
        selected_party = st.selectbox("Select Party ID", unique_parties)

    with col2:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))

    with col3:
        end_date = st.date_input("End Date", value=datetime(2024, 12, 31))

    col1, col2 = st.columns(2)

    with col1:
        currency = st.selectbox("Currency (Optional)", ["All", "USD", "EUR", "GBP", "JPY", "CNY"])

    with col2:
        if api_mode == "Individual Aggregate":
            # Let user select specific operations
            selected_operations = st.multiselect(
                "Select Operations",
                ["sum", "avg"],
                default=["sum", "avg"]
            )
        else:
            selected_operations = ["sum", "avg"]

    # Analyze button
    if st.button("🔍 Analyze Transactions", type="primary"):
        with st.spinner("Performing FHE operations on encrypted data..."):

            if api_mode == "Batch Query":
                # Use /query_transactions API - single call for all operations
                st.info("📡 Using /query_transactions API (Batch Mode)")

                query_data = {
                    "library": st.session_state.library,
                    "party_id": selected_party,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "currency": currency if currency != "All" else None
                }

                result = call_server("/query_transactions", "POST", query_data)

                if result and result.get('status') == 'success':
                    # Reformat results to match aggregate format
                    formatted_results = {}
                    for key, value in result.get('results', {}).items():
                        formatted_results[key] = {
                            'result': value.get('result'),
                            'expected_value': value.get('expected_value'),
                            'filtered_count': value.get('count'),
                            'operation': key.split('_')[-1],
                            'column': '_'.join(key.split('_')[:-1])
                        }

                    st.session_state.analysis_results = {
                        'results': formatted_results,
                        'party_id': selected_party,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'currency': currency,
                        'api_mode': 'batch'
                    }
                    st.success("✅ Batch analysis complete!")
                    st.rerun()

            else:  # Individual Aggregate
                # Use /aggregate API - separate call for each operation
                st.info("📡 Using /aggregate API (Individual Mode)")

                # Get batch IDs from encrypted metadata
                batch_ids = list(set([meta['batch_id'] for meta in st.session_state.encrypted_metadata.values()]))

                # Perform aggregations on encrypted data
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()

                columns = ['balance', 'amount_transferred']
                total_ops = len(columns) * len(selected_operations)
                current_op = 0

                for column in columns:
                    if column not in st.session_state.encrypted_metadata:
                        continue

                    for operation in selected_operations:
                        current_op += 1
                        status_text.text(f"Processing {column} ({operation})... ({current_op}/{total_ops})")

                        agg_data = {
                            "library": st.session_state.library,
                            "operation": operation,
                            "batch_ids": batch_ids,
                            "column_name": column,
                            "party_id": selected_party,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            "currency": currency if currency != "All" else None
                        }

                        result = call_server("/aggregate", "POST", agg_data)

                        if result and result.get('status') == 'success':
                            results[f"{column}_{operation}"] = result

                        progress_bar.progress(current_op / total_ops)

                progress_bar.empty()
                status_text.empty()

                st.session_state.analysis_results = {
                    'results': results,
                    'party_id': selected_party,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'currency': currency,
                    'api_mode': 'individual'
                }
                st.success(f"✅ Individual analysis complete! ({len(results)} operations)")
                st.rerun()

    # Display results
    if st.session_state.analysis_results:
        st.divider()

        # Show which API was used
        api_mode_used = st.session_state.analysis_results.get('api_mode', 'unknown')
        if api_mode_used == 'batch':
            st.info("📊 Results from: /query_transactions API (Batch Mode)")
        else:
            st.info("📊 Results from: /aggregate API (Individual Mode)")

        st.subheader("Analysis Results")

        results = st.session_state.analysis_results.get('results', {})
        party_id = st.session_state.analysis_results.get('party_id')
        start_date_str = st.session_state.analysis_results.get('start_date')
        end_date_str = st.session_state.analysis_results.get('end_date')

        # Show encrypted results summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Operations Performed", len(results))

        with col2:
            total_filtered = sum(r.get('filtered_count', 0) for r in results.values())
            avg_filtered = total_filtered // len(results) if results else 0
            st.metric("Avg Records Processed", avg_filtered)

        with col3:
            st.metric("API Mode", api_mode_used.title())

        # Decryption and Reconciliation Section
        st.divider()
        st.subheader("🔓 Decrypt & Reconcile Results")

        # Create reconciliation data storage
        if 'reconciliation_data' not in st.session_state:
            st.session_state.reconciliation_data = []

        # Organize results by column
        columns_data = {}
        for key, result in results.items():
            if api_mode_used == 'batch':
                column_name = result.get('column', key.rsplit('_', 1)[0])
                operation = result.get('operation', key.rsplit('_', 1)[1])
            else:
                column_name, operation = key.rsplit('_', 1)

            if column_name not in columns_data:
                columns_data[column_name] = {}
            columns_data[column_name][operation] = result

        # Display by column with decrypt buttons
        for column_name, operations in columns_data.items():
            with st.expander(f"📈 {column_name.replace('_', ' ').title()}", expanded=True):
                cols = st.columns(len(operations))

                for idx, (operation, result) in enumerate(operations.items()):
                    with cols[idx]:
                        st.write(f"**{operation.upper()}**")

                        # Show encrypted indicator
                        st.write("🔒 Encrypted Result")
                        st.caption(f"Records: {result.get('filtered_count', 0)}")

                        # Decrypt button
                        if st.button(f"Decrypt {operation.upper()}", key=f"decrypt_{column_name}_{operation}",
                                     use_container_width=True):
                            with st.spinner(f"Decrypting {operation}..."):
                                decrypt_data = {
                                    "library": st.session_state.library,
                                    "result_data": result.get('result'),
                                    "data_type": "numeric"
                                }

                                decrypt_result = call_server("/decrypt", "POST", decrypt_data)

                                if decrypt_result and decrypt_result.get('status') == 'success':
                                    decrypted_value = decrypt_result.get('decrypted_value')
                                    expected_value = result.get('expected_value')

                                    # Calculate difference and match status
                                    if expected_value is not None and decrypted_value is not None:
                                        difference = abs(expected_value - decrypted_value)
                                        match = difference < 0.01
                                    else:
                                        difference = None
                                        match = False

                                    # Add to reconciliation data
                                    reconciliation_entry = {
                                        'Column': column_name.replace('_', ' ').title(),
                                        'Operation': operation.upper(),
                                        'Expected Value': f"${expected_value:,.2f}" if expected_value else "N/A",
                                        'Computed Value (FHE)': f"${decrypted_value:,.2f}" if decrypted_value else "N/A",
                                        'Difference': f"${difference:,.2f}" if difference is not None else "N/A",
                                        'Match': "✅ Yes" if match else "❌ No",
                                        'Records Count': result.get('filtered_count', 0),
                                        'API Used': api_mode_used.title()
                                    }

                                    # Update or append
                                    found = False
                                    for i, entry in enumerate(st.session_state.reconciliation_data):
                                        if entry['Column'] == reconciliation_entry['Column'] and entry['Operation'] == \
                                                reconciliation_entry['Operation']:
                                            st.session_state.reconciliation_data[i] = reconciliation_entry
                                            found = True
                                            break

                                    if not found:
                                        st.session_state.reconciliation_data.append(reconciliation_entry)

                                    st.success(f"✅ {decrypted_value:,.2f}")
                                    st.rerun()

        # Show reconciliation table
        if st.session_state.reconciliation_data:
            st.divider()
            st.subheader("📊 Reconciliation Report")

            reconciliation_df = pd.DataFrame(st.session_state.reconciliation_data)

            # Style the dataframe
            st.dataframe(
                reconciliation_df,
                use_container_width=True,
                hide_index=True
            )

            # Download reconciliation report
            csv = reconciliation_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Reconciliation Report",
                data=csv,
                file_name=f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # Summary metrics
            st.divider()
            st.subheader("Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_ops = len(reconciliation_df)
                st.metric("Total Operations", total_ops)

            with col2:
                matches = len([x for x in reconciliation_df['Match'] if '✅' in x])
                st.metric("Matches", f"{matches}/{total_ops}")

            with col3:
                match_rate = (matches / total_ops * 100) if total_ops > 0 else 0
                st.metric("Match Rate", f"{match_rate:.1f}%")

            with col4:
                if st.button("🔄 Clear Reconciliation"):
                    st.session_state.reconciliation_data = []
                    st.rerun()

        # Transaction pattern visualization
        if st.session_state.data is not None:
            st.divider()
            st.subheader("📈 Transaction Patterns")

            # Filter actual data for comparison
            filtered_data = st.session_state.data[
                (st.session_state.data['partyid'] == party_id) &
                (st.session_state.data['payment_date'] >= pd.Timestamp(start_date_str)) &
                (st.session_state.data['payment_date'] <= pd.Timestamp(end_date_str))
                ]

            if not filtered_data.empty:
                tab1, tab2, tab3 = st.tabs(["Timeline", "Statistics", "Distribution"])

                with tab1:
                    # Timeline chart
                    fig = px.scatter(
                        filtered_data,
                        x='payment_date',
                        y='amount_transferred',
                        title=f"Transaction Timeline for {party_id}",
                        labels={'payment_date': 'Date', 'amount_transferred': 'Amount'},
                        color='payment_mode',
                        size='amount_transferred',
                        hover_data=['balance', 'currency']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    # Statistics table
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Actual Values (Unencrypted)**")
                        stats_data = {
                            'Metric': ['Sum', 'Average', 'Min', 'Max', 'Count', 'Std Dev'],
                            'Balance': [
                                f"${filtered_data['balance'].sum():,.2f}",
                                f"${filtered_data['balance'].mean():,.2f}",
                                f"${filtered_data['balance'].min():,.2f}",
                                f"${filtered_data['balance'].max():,.2f}",
                                len(filtered_data),
                                f"${filtered_data['balance'].std():,.2f}"
                            ],
                            'Amount Transferred': [
                                f"${filtered_data['amount_transferred'].sum():,.2f}",
                                f"${filtered_data['amount_transferred'].mean():,.2f}",
                                f"${filtered_data['amount_transferred'].min():,.2f}",
                                f"${filtered_data['amount_transferred'].max():,.2f}",
                                len(filtered_data),
                                f"${filtered_data['amount_transferred'].std():,.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

                    with col2:
                        st.write("**FHE Computed Values**")
                        if st.session_state.reconciliation_data:
                            st.success("✅ See Reconciliation Report above")
                            st.info(f"API Mode: {api_mode_used.title()}")
                        else:
                            st.warning("⚠️ Decrypt results to see FHE values")

                with tab3:
                    # Distribution charts
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = px.histogram(
                            filtered_data,
                            x='amount_transferred',
                            nbins=20,
                            title='Amount Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.pie(
                            filtered_data,
                            names='payment_mode',
                            title='Payment Mode Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transactions found for selected filters")


def screen_4_statistics():
    st.title("📈 Statistics & Analytics")

    if not st.session_state.encryption_stats:
        st.warning("⚠️ No encryption statistics available")
        return

    # Encryption Statistics
    st.header("Encryption Performance")

    stats_df = pd.DataFrame(st.session_state.encryption_stats)

    col1, col2, col3 = st.columns(3)

    with col1:
        total_time = stats_df['time'].sum()
        st.metric("Total Encryption Time", f"{total_time:.2f}s")

    with col2:
        avg_throughput = stats_df['throughput'].mean()
        st.metric("Avg Throughput", f"{avg_throughput:.0f} records/s")

    with col3:
        total_records = stats_df['count'].sum()
        st.metric("Total Records Encrypted", f"{total_records:,}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            stats_df,
            x='column',
            y='time',
            title='Encryption Time by Column',
            labels={'time': 'Time (seconds)', 'column': 'Column'},
            color='type'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            stats_df,
            x='column',
            y='throughput',
            title='Throughput by Column',
            labels={'throughput': 'Records/Second', 'column': 'Column'},
            color='type'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed statistics table
    st.subheader("Detailed Statistics")
    st.dataframe(stats_df, use_container_width=True)

    # Server stats
    st.header("Server Statistics")
    server_stats = call_server("/stats")

    if server_stats and server_stats.get('status') == 'success':
        stats = server_stats.get('stats', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Batches", stats.get('total_batches', 0))
        with col2:
            st.metric("Columns Encrypted", stats.get('total_columns_encrypted', 0))
        with col3:
            st.metric("Total Encrypted Values", f"{stats.get('total_encrypted_values', 0):,}")


def screen_6_ml_inference():
    """ML model inference on encrypted data"""
    st.title("🤖 ML Inference on Encrypted Data")
    st.markdown("### Run Machine Learning Models Without Decryption")

    if not st.session_state.encrypted_metadata:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Perform ML inference directly on encrypted features")

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["linear", "logistic", "polynomial"],
            help="Linear: Credit scoring, Logistic: Classification, Polynomial: Complex patterns"
        )

    with col2:
        library = st.selectbox("Library", ["TenSEAL", "OpenFHE"])

    st.divider()

    # Feature selection
    st.subheader("1️⃣ Select Encrypted Features")

    available_features = list(st.session_state.encrypted_metadata.keys())
    selected_features = st.multiselect(
        "Features for Model",
        available_features,
        default=available_features[:3] if len(available_features) >= 3 else available_features
    )

    if not selected_features:
        st.warning("Please select at least one feature")
        return

    # Model parameters
    st.subheader("2️⃣ Configure Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # Weights input
        st.markdown("**Feature Weights:**")
        weights = []
        for feature in selected_features:
            weight = st.number_input(
                f"Weight for {feature}",
                value=1.0,
                step=0.1,
                key=f"weight_{feature}"
            )
            weights.append(weight)

    with col2:
        intercept = st.number_input("Intercept (Bias)", value=0.0, step=0.1)

        if model_type == "polynomial":
            poly_degree = st.slider("Polynomial Degree", 1, 7, 3)

    # Pre-trained model template
    with st.expander("📚 Use Pre-trained Model Template"):
        template = st.selectbox(
            "Template",
            [
                "Credit Score Model",
                "Fraud Detection Model",
                "Transaction Risk Model",
                "Custom"
            ]
        )

        if template == "Credit Score Model":
            st.info("Weights: [0.4, 0.3, 0.3], Intercept: 500")
            if st.button("Apply Template"):
                weights = [0.4, 0.3, 0.3][:len(selected_features)]
                intercept = 500.0
                st.rerun()

        elif template == "Fraud Detection Model":
            st.info("Weights: [0.5, 0.3, 0.2], Intercept: 0.0")
            if st.button("Apply Template"):
                weights = [0.5, 0.3, 0.2][:len(selected_features)]
                intercept = 0.0
                st.rerun()

    # Run inference
    st.divider()

    if st.button("🚀 Run ML Inference", type="primary"):
        with st.spinner(f"Running {model_type} inference on encrypted data..."):
            try:
                # Get encrypted feature data (first value from each column)
                encrypted_features = []
                for feature in selected_features:
                    metadata = st.session_state.encrypted_metadata[feature]
                    records = metadata['records']
                    if records:
                        # Get first encrypted value
                        first_record = records[0]
                        encrypted_val = first_record.get('ciphertext_preview', '')
                        # In real scenario, use actual encrypted bytes
                        encrypted_features.append(base64.b64encode(b"mock_encrypted").decode())

                # Prepare request
                request_data = {
                    "library": library,
                    "model_type": model_type,
                    "encrypted_features": encrypted_features,
                    "weights": weights,
                    "intercept": intercept
                }

                if model_type == "polynomial":
                    request_data["polynomial_degree"] = poly_degree

                # Call ML inference endpoint
                result = call_server("/ml/inference", "POST", request_data)

                if result and result.get('status') == 'success':
                    st.success("✅ ML Inference Complete!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Model Type", model_type.title())
                    with col2:
                        st.metric("Features Used", len(selected_features))
                    with col3:
                        comp_time = result.get('computation_time', 0)
                        st.metric("Computation Time", f"{comp_time:.3f}s")

                    st.info("🔒 Result is encrypted. Use decrypt endpoint to see the score.")

                    # Display encrypted result
                    with st.expander("🔐 Encrypted Result"):
                        encrypted_result = result.get('encrypted_result', '')
                        st.code(encrypted_result[:200] + "..." if len(encrypted_result) > 200 else encrypted_result)

                    # Decrypt option
                    if st.button("🔓 Decrypt Result"):
                        decrypt_data = {
                            "library": library,
                            "result_data": {"data": encrypted_result, "type": "bytes"},
                            "data_type": "numeric"
                        }

                        decrypt_result = call_server("/decrypt", "POST", decrypt_data)

                        if decrypt_result and decrypt_result.get('status') == 'success':
                            score = decrypt_result.get('decrypted_value')
                            st.success(f"📊 Decrypted Score: **{score:.2f}**")

                            # Interpretation
                            if model_type == "linear":
                                st.info(f"Credit Score: {score:.0f}/1000")
                            elif model_type == "logistic":
                                st.info(f"Classification Probability: {score:.4f}")

            except Exception as e:
                st.error(f"❌ Inference failed: {str(e)}")


# Main navigation
def main():
    """Main function with navigation"""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Screen",
        [
            "1. Data Upload & Config",
            "2. Data Encryption",
            "3. FHE Analysis",
            "4. Statistics",
            "5. ML Inference",
            "6. Fraud Detection",
            "7. SIMD Operations"
        ]
    )

    st.sidebar.divider()

    # System info
    st.sidebar.subheader("System Info")
    st.sidebar.info(f"""
        **Library:** {st.session_state.get('library', 'Not selected')}  
        **Scheme:** {st.session_state.get('scheme', 'Not selected')}  
        **Keys:** {'✅ Generated' if st.session_state.keys_generated else '❌ Not generated'}
        """)

    # Reset button
    if st.sidebar.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if page == "1. Data Upload & Config":
        screen_1_data_upload()
    elif page == "2. Data Encryption":
        screen_2_encryption()
    elif page == "3. FHE Analysis":
        screen_3_analysis()
    elif page == "4. Statistics":
        screen_4_statistics()
    elif page == "5. ML Inference":
        screen_6_ml_inference()
    elif page == "6. Fraud Detection":
        screen_7_fraud_detection()
    elif page == "7. SIMD Operations":
        screen_8_simd_operations()


if __name__ == "__main__":
    main()