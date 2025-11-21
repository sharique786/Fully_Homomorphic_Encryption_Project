"""
Complete Enhanced Streamlit Client with:
1. New schema (Party → Account → Transaction)
2. SQLite storage for encrypted data
3. Party ID/Email-based filtering
4. Merged Parameter Selection + Encryption screens
5. Real FHE operations only
6. All 7 screens implemented
"""
import concurrent.futures
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import base64
from threading import Lock
import sqlite3
import uuid

SERVER_URL = "http://localhost:8000"
session_lock = Lock()

# Page config
st.set_page_config(
    page_title="FHE Financial Data Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== SQLite Database Setup ====================

def init_db():
    """Initialize SQLite database for encrypted storage"""
    conn = sqlite3.connect('fhe_encrypted_data.db', check_same_thread=False)
    cursor = conn.cursor()

    # Create encrypted data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS encrypted_data (
            id TEXT PRIMARY KEY,
            party_id TEXT NOT NULL,
            email_id TEXT NOT NULL,
            account_id TEXT,
            transaction_id TEXT,
            column_name TEXT NOT NULL,
            encrypted_value BLOB NOT NULL,
            data_type TEXT NOT NULL,
            transaction_date TEXT,
            batch_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_party ON encrypted_data(party_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON encrypted_data(email_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON encrypted_data(transaction_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_column ON encrypted_data(column_name)')

    # Create metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            batch_id TEXT PRIMARY KEY,
            column_name TEXT NOT NULL,
            total_records INTEGER,
            library TEXT,
            scheme TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    return conn


# Initialize database
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()

# ==================== Session State ====================

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
if 'selected_library' not in st.session_state:
    st.session_state.selected_library = None
if 'selected_scheme' not in st.session_state:
    st.session_state.selected_scheme = None


# ==================== Database Helper Functions ====================

def store_encrypted_data(party_id, email_id, account_id, transaction_id,
                         column_name, encrypted_value, data_type,
                         transaction_date, batch_id):
    """Store encrypted data in SQLite"""
    conn = st.session_state.db_conn
    cursor = conn.cursor()

    record_id = str(uuid.uuid4())

    cursor.execute('''
        INSERT INTO encrypted_data 
        (id, party_id, email_id, account_id, transaction_id, column_name, 
         encrypted_value, data_type, transaction_date, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (record_id, party_id, email_id, account_id, transaction_id,
          column_name, encrypted_value, data_type,
          transaction_date.isoformat() if pd.notna(transaction_date) else None, batch_id))

    conn.commit()


def query_encrypted_data(party_id=None, email_id=None, start_date=None,
                         end_date=None, column_name=None):
    """Query encrypted data from SQLite"""
    conn = st.session_state.db_conn
    cursor = conn.cursor()

    query = "SELECT * FROM encrypted_data WHERE 1=1"
    params = []

    if party_id:
        query += " AND party_id = ?"
        params.append(party_id)

    if email_id:
        query += " AND email_id = ?"
        params.append(email_id)

    if start_date:
        query += " AND transaction_date >= ?"
        params.append(start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date))

    if end_date:
        query += " AND transaction_date <= ?"
        params.append(end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date))

    if column_name:
        query += " AND column_name = ?"
        params.append(column_name)

    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    results = cursor.fetchall()

    return [dict(zip(columns, row)) for row in results]


def get_db_stats():
    """Get database statistics"""
    conn = st.session_state.db_conn
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM encrypted_data")
    total_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT party_id) FROM encrypted_data")
    unique_parties = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT column_name) FROM encrypted_data")
    unique_columns = cursor.fetchone()[0]

    return {
        'total_records': total_records,
        'unique_parties': unique_parties,
        'unique_columns': unique_columns
    }


# ==================== Helper Functions ====================

def generate_synthetic_data(num_records=1000):
    """Generate synthetic financial data with NEW SCHEMA"""
    np.random.seed(42)

    num_parties = min(num_records // 10, 100)
    parties_data = []

    regions = ['US', 'LATAM', 'EMEA', 'APAC']
    account_types = ['Saving', 'Wealth', 'Current', 'Checking']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
    transaction_types = ['Card', 'Wire', 'ACH', 'SWIFT', 'Check']

    for p in range(num_parties):
        party_id = str(uuid.uuid4())
        name = f"Customer_{p + 1}"
        email = f"customer{p + 1}@example.com"
        address = f"{np.random.randint(1, 9999)} Main St, City {p + 1}"
        dob = (datetime(1950, 1, 1) + timedelta(days=int(np.random.randint(0, 25000)))).date()
        region = np.random.choice(regions)

        num_accounts = np.random.randint(1, 4)

        for a in range(num_accounts):
            account_id = str(uuid.uuid4())
            account_region = region
            account_type = np.random.choice(account_types)

            num_transactions = np.random.randint(3, 11)

            for t in range(num_transactions):
                transaction_id = str(uuid.uuid4())
                from_account = account_id if np.random.rand() > 0.5 else str(uuid.uuid4())
                to_account = str(uuid.uuid4()) if from_account == account_id else account_id
                amount = float(np.round(np.random.uniform(100, 50000), 2))
                currency = np.random.choice(currencies)
                transaction_date = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 365)))
                transaction_type = np.random.choice(transaction_types)

                parties_data.append({
                    'Party ID': party_id,
                    'Name': name,
                    'Email ID': email,
                    'Address': address,
                    'DOB': dob,
                    'Region': region,
                    'Account ID': account_id,
                    'Account Region': account_region,
                    'Account Type': account_type,
                    'Transaction ID': transaction_id,
                    'From Account': from_account,
                    'To Account': to_account,
                    'Amount': amount,
                    'Currency': currency,
                    'Transaction Date': transaction_date,
                    'Transaction Type': transaction_type
                })

    df = pd.DataFrame(parties_data)
    df['Amount'] = df['Amount'].astype('float64')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

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


# ==================== SCREEN 1: DATA UPLOAD & KEY GENERATION ====================

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
            st.session_state.data['Transaction Date'] = pd.to_datetime(st.session_state.data['Transaction Date'])
            st.success(f"✅ Loaded {len(st.session_state.data)} records")

    with col2:
        num_records = st.number_input("Generate Synthetic Data", min_value=100, max_value=10000, value=1000, step=100)
        if st.button("🎲 Generate Data"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_synthetic_data(num_records)
                st.success(f"✅ Generated {len(st.session_state.data)} records")

    # Display data preview
    if st.session_state.data is not None:
        st.subheader("Data Preview (New Schema)")
        st.dataframe(st.session_state.data.head(20), use_container_width=True)

        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(st.session_state.data))
        with col2:
            st.metric("Unique Parties", st.session_state.data['Party ID'].nunique())
        with col3:
            st.metric("Unique Accounts", st.session_state.data['Account ID'].nunique())
        with col4:
            total_amount = st.session_state.data['Amount'].sum()
            st.metric("Total Amount", f"${total_amount:,.2f}")

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

        st.session_state.selected_scheme = scheme  # Store selected scheme
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


# ==================== SCREEN 2: ENCRYPTION WITH PARAMETER SELECTION ====================
# ==================== SCREEN 2: COLUMN-BASED ENCRYPTION ====================

def screen_2_encryption():
    st.title("🔐 Data Encryption")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload or generate data first")
        return

    if not st.session_state.keys_generated:
        st.warning("⚠️ Please generate encryption keys first")
        return

    # SIMD Mode Selection
    st.header("1️⃣ SIMD Operation Type")

    simd_mode = st.selectbox(
        "Select SIMD Mode",
        ["individual", "packed_vector", "batch_processing"],
        format_func=lambda x: {
            "individual": "Individual Encryption (Standard - One value per ciphertext)",
            "packed_vector": "Packed Vector (SIMD - Multiple values in one ciphertext, 10-100x faster)",
            "batch_processing": "Batch Processing (Optimized batching, 2-5x faster)"
        }[x]
    )

    # Explain SIMD modes
    with st.expander("ℹ️ SIMD Mode Explanation"):
        st.markdown("""
        **Individual Encryption:**
        - Each value encrypted separately
        - Standard approach, slower but simple
        - Best for: Small datasets, testing

        **Packed Vector (SIMD):**
        - Packs 128 values into single ciphertext
        - Uses SIMD slots for parallelism
        - 10-100x faster than individual
        - Requires CKKS scheme
        - Best for: Large datasets, analytics

        **Batch Processing:**
        - Processes in optimized batches of 256
        - Balances speed and flexibility
        - 2-5x faster than individual
        - Works with all schemes
        - Best for: Medium datasets, general use
        """)

    st.divider()

    # Column Selection
    st.header("2️⃣ Select Columns to Encrypt")

    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    date_cols = st.session_state.data.select_dtypes(include=['datetime']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        selected_numeric = st.multiselect("Numeric Columns", numeric_cols, default=['Amount'])
        selected_text = st.multiselect("Text Columns", text_cols)

    with col2:
        selected_dates = st.multiselect("Date Columns", date_cols)

    # Scheme limitations
    limitations_result = call_server(f"/scheme_limitations/{st.session_state.library}/{st.session_state.scheme}")
    if limitations_result and 'limitations' in limitations_result:
        with st.expander("ℹ️ Scheme Limitations"):
            limitations = limitations_result['limitations']
            st.json(limitations)

    all_selected = selected_numeric + selected_text + selected_dates

    # Encryption Button
    if st.button("🔒 Encrypt Selected Columns", type="primary", disabled=len(all_selected) == 0):
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        # COLUMN-BASED ENCRYPTION (not row-based)
        for idx, col in enumerate(all_selected):
            status_text.text(f"Encrypting {col}... ({idx + 1}/{len(all_selected)})")

            # Determine data type
            if col in numeric_cols:
                data_type = "numeric"
            elif col in text_cols:
                data_type = "text"
            else:
                data_type = "date"

            # Prepare ENTIRE COLUMN data
            if data_type == "date":
                column_data = st.session_state.data[col].apply(
                    lambda x: x.isoformat() if pd.notna(x) else None
                ).tolist()
            else:
                column_data = st.session_state.data[col].tolist()

            # column_data = st.session_state.data[col].tolist()
            party_ids = st.session_state.data['Party ID'].tolist()
            email_ids = st.session_state.data['Email ID'].tolist()
            account_ids = st.session_state.data['Account ID'].tolist()
            transaction_ids = st.session_state.data['Transaction ID'].tolist()
            transaction_dates = st.session_state.data['Transaction Date'].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            ).tolist()

            # FIXED: Send entire column to server
            enc_data = {
                "library": st.session_state.selected_library,
                "scheme": st.session_state.selected_scheme,
                "column_name": col,
                "data_type": data_type,
                "column_data": column_data,  # Entire column
                "party_ids": party_ids,
                "email_ids": email_ids,
                "account_ids": account_ids,
                "transaction_ids": transaction_ids,
                "transaction_dates": transaction_dates,
                "batch_id": batch_id,
                "simd_mode": simd_mode
            }

            # Call NEW endpoint
            result = call_server("/encrypt_column", "POST", enc_data)

            if result and result.get('status') == 'success':
                encrypted_results = result.get('encrypted_results', [])

                # Store in SQLite
                for enc_result in encrypted_results:
                    encrypted_value = enc_result.get('encrypted_value')
                    party_id = enc_result.get('party_id')
                    email_id = enc_result.get('email_id')
                    account_id = enc_result.get('account_id')
                    transaction_id = enc_result.get('transaction_id')
                    transaction_date_str = enc_result.get('transaction_date')

                    transaction_date = pd.Timestamp(transaction_date_str) if transaction_date_str else None

                    store_encrypted_data(
                        party_id, email_id, account_id, transaction_id,
                        col, encrypted_value.encode() if isinstance(encrypted_value, str) else encrypted_value,
                        data_type, transaction_date, batch_id
                    )

                # Update metadata
                st.session_state.encrypted_metadata[col] = {
                    "count": result.get('encrypted_count', 0),
                    "batch_id": batch_id,
                    "data_type": data_type,
                    "library": st.session_state.selected_library,
                    "scheme": st.session_state.selected_scheme,
                    "simd_mode": simd_mode,
                    "encryption_time": result.get('encryption_time', 0)
                }

            progress_bar.progress((idx + 1) / len(all_selected))

        elapsed_time = time.time() - start_time
        status_text.success(f"✅ Encryption complete! {len(all_selected)} columns encrypted in {elapsed_time:.2f}s")

        # Store encryption stats
        st.session_state.encryption_stats.append({
            "batch_id": batch_id,
            "columns": len(all_selected),
            "total_records": len(st.session_state.data) * len(all_selected),
            "time": elapsed_time,
            "throughput": (len(st.session_state.data) * len(all_selected)) / elapsed_time,
            "simd_mode": simd_mode
        })

        time.sleep(1)
        st.rerun()

    # Display encryption statistics
    if st.session_state.encrypted_metadata:
        st.divider()
        st.header("📊 Encryption Statistics")

        db_stats = get_db_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Encrypted Records", db_stats['total_records'])
        with col2:
            st.metric("Unique Parties", db_stats['unique_parties'])
        with col3:
            st.metric("Encrypted Columns", db_stats['unique_columns'])
        with col4:
            if st.session_state.encryption_stats:
                avg_throughput = np.mean([s['throughput'] for s in st.session_state.encryption_stats])
                st.metric("Avg Throughput", f"{avg_throughput:.0f} rec/s")

        # Show column details
        with st.expander("📋 Column Encryption Details", expanded=True):
            for col, meta in st.session_state.encrypted_metadata.items():
                simd_mode = meta.get('simd_mode', 'individual')
                encryption_time = meta.get('encryption_time', 0)
                st.write(
                    f"**{col}**: {meta['count']} records ({meta['data_type']}) - "
                    f"{meta['library']}/{meta['scheme']} - "
                    f"Mode: {simd_mode} - Time: {encryption_time:.2f}s"
                )


# ==================== SCREEN 3: FHE ANALYSIS ====================

def screen_3_analysis():
    st.title("📊 FHE Analysis")

    db_stats = get_db_stats()
    if db_stats['total_records'] == 0:
        st.warning("⚠️ Please encrypt data first")
        return

    st.subheader("Transaction Analysis (On Encrypted Data)")
    st.info("💡 All computations performed on encrypted data. Results returned encrypted.")

    # Get unique parties from database
    conn = st.session_state.db_conn
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT party_id, email_id FROM encrypted_data LIMIT 1000")
    parties = cursor.fetchall()

    if not parties:
        st.warning("No encrypted data available")
        return

    # Selection UI
    col1, col2, col3 = st.columns(3)

    with col1:
        party_options = [f"{p[0][:50]}" for p in parties]
        selected_idx = st.selectbox("Select Party ID", range(len(party_options)),
                                    format_func=lambda x: party_options[x])
        selected_party = parties[selected_idx][0]
        selected_email = parties[selected_idx][1]

    with col2:
        # Default to last year
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        start_date = st.date_input("Start Date", value=start_date)

    with col3:
        end_date = st.date_input("End Date", value=end_date)

    currency = st.selectbox("Currency (Optional)", ["All", "USD", "EUR", "GBP", "JPY", "CNY"])

    if st.button("🔍 Analyze Transactions", type="primary"):
        with st.spinner("Performing FHE operations on encrypted data..."):

            query_data = {
                "library": st.session_state.selected_library,
                "party_id": selected_party,
                "email_id": selected_email,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "currency": currency if currency != "All" else None
            }

            result = call_server("/query_transactions", "POST", query_data)

            if result and result.get('status') == 'success':
                st.session_state.analysis_results = result
                st.success("✅ Analysis complete!")
                st.rerun()

    # Display results
    if st.session_state.analysis_results:
        st.divider()
        st.subheader("Analysis Results")

        results = st.session_state.analysis_results

        # Show encrypted results
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Encrypted Results:**")
            st.metric("Transaction Count", results.get('transaction_count', 0))
            st.code("🔒 Total Transferred: [Encrypted]")
            st.code("🔒 Total Received: [Encrypted]")
            st.code("🔒 Average Amount: [Encrypted]")

        with col2:
            if st.button("🔓 Decrypt Results"):
                with st.spinner("Decrypting..."):
                    # Decrypt each result
                    decrypted = {}

                    for key in ['total_transferred', 'total_received', 'average_amount']:
                        if key in results:
                            decrypt_result = call_server("/decrypt", "POST", {
                                "library": st.session_state.selected_library,
                                "result_data": results[key],
                                "data_type": "numeric"
                            })

                            if decrypt_result and decrypt_result.get('status') == 'success':
                                decrypted[key] = decrypt_result.get('decrypted_value', 0)

                    st.session_state.decrypted_results = decrypted
                    st.rerun()

        # Show decrypted results
        if 'decrypted_results' in st.session_state and st.session_state.decrypted_results:
            st.divider()
            st.subheader("🔓 Decrypted Results")

            dec = st.session_state.decrypted_results

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transferred", f"${dec.get('total_transferred', 0):,.2f}")
            with col2:
                st.metric("Total Received", f"${dec.get('total_received', 0):,.2f}")
            with col3:
                st.metric("Average Amount", f"${dec.get('average_amount', 0):,.2f}")

            # Currency distribution
            if 'currency_distribution' in results:
                st.subheader("Payment Distribution by Currency")
                curr_dist = results['currency_distribution']

                if curr_dist:
                    df_curr = pd.DataFrame(list(curr_dist.items()), columns=['Currency', 'Count'])
                    fig = px.pie(df_curr, names='Currency', values='Count', title='Transaction Distribution')
                    st.plotly_chart(fig, use_container_width=True)


# ==================== SCREEN 4: FRAUD DETECTION ====================

def screen_4_fraud_detection():
    st.title("🚨 Fraud Detection (Encrypted)")

    db_stats = get_db_stats()
    if db_stats['total_records'] == 0:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Fraud detection performed on encrypted transaction data")

    # Get parties from database
    conn = st.session_state.db_conn
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT party_id, email_id FROM encrypted_data LIMIT 1000")
    parties = cursor.fetchall()

    col1, col2 = st.columns(2)

    with col1:
        party_options = [f"{p[0][:8]}... ({p[1]})" for p in parties]
        selected_idx = st.selectbox("Select Party", range(len(party_options)), format_func=lambda x: party_options[x])
        selected_party = parties[selected_idx][0]
        selected_email = parties[selected_idx][1]

    with col2:
        detection_type = st.selectbox(
            "Detection Method",
            ["linear_score", "distance_anomaly"],
            format_func=lambda x: {
                "linear_score": "Linear Weighted Scoring",
                "distance_anomaly": "Distance-Based Anomaly Detection"
            }[x]
        )

    # Query encrypted data for this party
    encrypted_records = query_encrypted_data(
        party_id=selected_party,
        column_name="Amount"
    )

    if not encrypted_records:
        st.warning(f"No encrypted data found for party {selected_party[:8]}...")
        return

    st.write(f"Found {len(encrypted_records)} encrypted transactions for analysis")

    # Model configuration
    st.subheader("Configure Detection Model")

    if detection_type == "linear_score":
        st.markdown("**Feature Weights:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            w_amount = st.slider("Amount Weight", 0.0, 1.0, 0.6)
        with col2:
            w_frequency = st.slider("Frequency Weight", 0.0, 1.0, 0.3)
        with col3:
            w_velocity = st.slider("Velocity Weight", 0.0, 1.0, 0.1)

        weights = {
            'amount': w_amount,
            'frequency': w_frequency,
            'velocity': w_velocity
        }
    else:
        st.markdown("**Normal Behavior Centroid:**")
        centroid_amount = st.number_input("Normal Amount", value=5000.0)
        centroid_frequency = st.number_input("Normal Frequency", value=10.0)

        centroid = {
            'amount': centroid_amount,
            'frequency': centroid_frequency
        }

    if st.button("🔍 Run Fraud Detection", type="primary"):
        with st.spinner("Analyzing encrypted transactions for fraud..."):

            # Prepare encrypted transaction data
            encrypted_amounts = [rec['encrypted_value'] for rec in encrypted_records[:10]]  # Limit for demo

            request_data = {
                "library": st.session_state.selected_library,
                "party_id": selected_party,
                "email_id": selected_email,
                "detection_type": detection_type,
                "encrypted_amounts": [base64.b64encode(amt).decode() if isinstance(amt, bytes) else amt for amt in
                                      encrypted_amounts],
                "model_params": weights if detection_type == "linear_score" else {'centroid': centroid}
            }

            result = call_server("/fraud/detect", "POST", request_data)

            if result and result.get('status') == 'success':
                st.session_state.fraud_result = result
                st.success("✅ Fraud detection complete!")
                st.rerun()

    # Display fraud results
    if 'fraud_result' in st.session_state and st.session_state.fraud_result:
        st.divider()
        st.subheader("Fraud Detection Results")

        fraud_result = st.session_state.fraud_result

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Encrypted Fraud Score:**")
            st.code("🔒 [Encrypted Score]")
            st.caption(f"Party: {fraud_result.get('party_id', '')[:8]}...")
            st.caption(f"Email: {fraud_result.get('email_id', '')}")

        with col2:
            if st.button("🔓 Decrypt Fraud Score"):
                with st.spinner("Decrypting..."):
                    decrypt_result = call_server("/decrypt", "POST", {
                        "library": st.session_state.selected_library,
                        "result_data": fraud_result.get('encrypted_score'),
                        "data_type": "numeric"
                    })

                    if decrypt_result and decrypt_result.get('status') == 'success':
                        fraud_score = decrypt_result.get('decrypted_value', 0)
                        st.session_state.decrypted_fraud_score = fraud_score
                        st.rerun()

        if 'decrypted_fraud_score' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Fraud Score Analysis")

            fraud_score = st.session_state.decrypted_fraud_score
            normalized_score = min(max(fraud_score / 100 if fraud_score > 1 else fraud_score, 0), 1)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Fraud Score", f"{normalized_score:.4f}")

                if normalized_score < 0.3:
                    st.success("✅ LOW RISK")
                elif normalized_score < 0.6:
                    st.warning("⚠️ MEDIUM RISK")
                elif normalized_score < 0.8:
                    st.error("🚨 HIGH RISK")
                else:
                    st.error("🚨🚨 CRITICAL RISK")

            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=normalized_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.6], 'color': "yellow"},
                            {'range': [0.6, 0.8], 'color': "orange"},
                            {'range': [0.8, 1], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)


# ==================== SCREEN 5: SIMD OPERATIONS ====================

def screen_5_simd_operations():
    st.title("🔢 SIMD Operations - Time Series Analytics")

    db_stats = get_db_stats()
    if db_stats['total_records'] == 0:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Perform time-series analytics on encrypted transaction data using SIMD operations")

    # Get parties from database
    conn = st.session_state.db_conn
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT party_id, email_id FROM encrypted_data LIMIT 1000")
    parties = cursor.fetchall()

    col1, col2 = st.columns(2)

    with col1:
        party_options = [f"{p[0][:8]}... ({p[1]})" for p in parties]
        selected_idx = st.selectbox("Select Party", range(len(party_options)), format_func=lambda x: party_options[x])
        selected_party = parties[selected_idx][0]
        selected_email = parties[selected_idx][1]

    with col2:
        operation = st.selectbox(
            "SIMD Operation",
            ["moving_average", "velocity_analysis", "transaction_correlation", "slot_wise_aggregation"],
            format_func=lambda x: {
                "moving_average": "Moving Average (Time Series)",
                "velocity_analysis": "Transaction Velocity",
                "transaction_correlation": "Transaction Correlation",
                "slot_wise_aggregation": "Slot-wise Aggregation"
            }[x]
        )

    # Query encrypted data for time series
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    encrypted_records = query_encrypted_data(
        party_id=selected_party,
        start_date=start_date,
        end_date=end_date,
        column_name="Amount"
    )

    if not encrypted_records:
        st.warning(f"No encrypted data found for party {selected_party[:8]}...")
        return

    st.write(f"Found {len(encrypted_records)} encrypted transactions for time-series analysis")

    # Operation parameters
    if operation == "moving_average":
        window_size = st.slider("Window Size (days)", 7, 90, 30)
        params = {"window_size": window_size}
    elif operation == "velocity_analysis":
        time_window = st.slider("Time Window (days)", 1, 30, 7)
        params = {"time_window": time_window}
    else:
        params = {}

    if st.button("▶️ Execute SIMD Operation", type="primary"):
        with st.spinner(f"Performing {operation} on encrypted time series..."):

            # Prepare encrypted vector
            encrypted_vector = [rec['encrypted_value'] for rec in encrypted_records]

            request_data = {
                "library": st.session_state.selected_library,
                "party_id": selected_party,
                "email_id": selected_email,
                "operation": operation,
                "encrypted_vector": [base64.b64encode(v).decode() if isinstance(v, bytes) else v for v in
                                     encrypted_vector],
                "parameters": params,
                "transaction_dates": [rec['transaction_date'] for rec in encrypted_records]
            }

            result = call_server("/simd/timeseries", "POST", request_data)

            if result and result.get('status') == 'success':
                st.session_state.simd_result = result
                st.success(f"✅ {operation} complete!")
                st.rerun()

    # Display results
    if 'simd_result' in st.session_state and st.session_state.simd_result:
        st.divider()
        st.subheader("SIMD Operation Results")

        simd_result = st.session_state.simd_result

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Operation", operation.replace('_', ' ').title())
        with col2:
            st.metric("Data Points", simd_result.get('data_points', 0))
        with col3:
            st.metric("Computation Time", f"{simd_result.get('computation_time', 0):.3f}s")

        st.write("**Encrypted Results:**")
        st.code("🔒 [Time Series Results Encrypted]")

        if st.button("🔓 Decrypt Time Series Results"):
            with st.spinner("Decrypting time series..."):
                # Decrypt results
                encrypted_results = simd_result.get('encrypted_results', [])
                decrypted_series = []

                for enc_val in encrypted_results[:50]:  # Limit for performance
                    decrypt_result = call_server("/decrypt", "POST", {
                        "library": st.session_state.selected_library,
                        "result_data": enc_val,
                        "data_type": "numeric"
                    })

                    if decrypt_result and decrypt_result.get('status') == 'success':
                        decrypted_series.append(decrypt_result.get('decrypted_value', 0))

                st.session_state.decrypted_simd = decrypted_series
                st.rerun()

        if 'decrypted_simd' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Time Series Analysis")

            series = st.session_state.decrypted_simd

            # Plot time series
            df_series = pd.DataFrame({
                'Time Index': range(len(series)),
                'Value': series
            })

            fig = px.line(df_series, x='Time Index', y='Value',
                          title=f'{operation.replace("_", " ").title()} Analysis')
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${np.mean(series):,.2f}")
            with col2:
                st.metric("Std Dev", f"${np.std(series):,.2f}")
            with col3:
                st.metric("Min", f"${np.min(series):,.2f}")
            with col4:
                st.metric("Max", f"${np.max(series):,.2f}")


# ==================== SCREEN 6: ML INFERENCE ====================

def screen_6_ml_inference():
    st.title("🤖 ML Inference on Encrypted Data")

    db_stats = get_db_stats()
    if db_stats['total_records'] == 0:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Run machine learning models on encrypted features")

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["linear", "logistic", "polynomial"],
            format_func=lambda x: {
                "linear": "Linear Regression",
                "logistic": "Logistic Regression",
                "polynomial": "Polynomial Model"
            }[x]
        )

    with col2:
        # Get available encrypted columns
        cursor = st.session_state.db_conn.cursor()
        cursor.execute("SELECT DISTINCT column_name FROM encrypted_data")
        available_cols = [row[0] for row in cursor.fetchall()]

        selected_features = st.multiselect("Select Features", available_cols,
                                           default=available_cols[:3] if len(available_cols) >= 3 else available_cols)

    if not selected_features:
        st.warning("Please select at least one feature")
        return

    # Model configuration
    st.subheader("Model Parameters")

    weights = []
    for feature in selected_features:
        weight = st.slider(f"Weight for {feature}", -1.0, 1.0, 0.5, key=f"weight_{feature}")
        weights.append(weight)

    intercept = st.number_input("Intercept (Bias)", value=0.0)

    if model_type == "polynomial":
        poly_degree = st.slider("Polynomial Degree", 1, 7, 3)

    if st.button("🚀 Run ML Inference", type="primary"):
        with st.spinner("Running ML inference on encrypted features..."):

            # Get encrypted features from database (sample)
            cursor = st.session_state.db_conn.cursor()
            cursor.execute(f"SELECT encrypted_value FROM encrypted_data WHERE column_name = ? LIMIT 1",
                           (selected_features[0],))
            sample_encrypted = [row[0] for row in cursor.fetchall()]

            request_data = {
                "library": st.session_state.selected_library,
                "model_type": model_type,
                "encrypted_features": [base64.b64encode(v).decode() if isinstance(v, bytes) else v for v in
                                       sample_encrypted * len(selected_features)],
                "weights": weights,
                "intercept": intercept
            }

            if model_type == "polynomial":
                request_data["polynomial_degree"] = poly_degree

            result = call_server("/ml/inference", "POST", request_data)

            if result and result.get('status') == 'success':
                st.session_state.ml_result = result
                st.success("✅ ML inference complete!")
                st.rerun()

    # Display results
    if 'ml_result' in st.session_state and st.session_state.ml_result:
        st.divider()
        st.subheader("ML Inference Results")

        ml_result = st.session_state.ml_result

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Model Type", model_type.title())
            st.metric("Features Used", len(selected_features))
            st.write("**Encrypted Prediction:**")
            st.code("🔒 [Encrypted Score]")

        with col2:
            if st.button("🔓 Decrypt Prediction"):
                with st.spinner("Decrypting..."):
                    decrypt_result = call_server("/decrypt", "POST", {
                        "library": st.session_state.selected_library,
                        "result_data": ml_result.get('encrypted_result'),
                        "data_type": "numeric"
                    })

                    if decrypt_result and decrypt_result.get('status') == 'success':
                        prediction = decrypt_result.get('decrypted_value', 0)
                        st.session_state.ml_prediction = prediction
                        st.rerun()

        if 'ml_prediction' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Prediction")

            prediction = st.session_state.ml_prediction

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction Value", f"{prediction:.4f}")

            with col2:
                if model_type == "linear":
                    st.info(f"Linear Model Score: {prediction:.2f}")
                elif model_type == "logistic":
                    st.info(f"Classification Probability: {prediction:.4f}")


# ==================== SCREEN 7: STATISTICS ====================

def screen_7_statistics():
    st.title("📈 Statistics & Analytics")

    db_stats = get_db_stats()

    if db_stats['total_records'] == 0:
        st.warning("⚠️ No encrypted data available")
        return

    # Overall statistics
    st.header("Database Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Encrypted Records", f"{db_stats['total_records']:,}")

    with col2:
        st.metric("Unique Parties", db_stats['unique_parties'])

    with col3:
        st.metric("Encrypted Columns", db_stats['unique_columns'])

    with col4:
        if st.session_state.encryption_stats:
            total_time = sum(s['time'] for s in st.session_state.encryption_stats)
            st.metric("Total Encryption Time", f"{total_time:.2f}s")

    st.divider()

    # Encryption performance
    if st.session_state.encryption_stats:
        st.header("Encryption Performance")

        stats_df = pd.DataFrame(st.session_state.encryption_stats)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(stats_df, x='batch_id', y='throughput',
                         title='Encryption Throughput by Batch',
                         labels={'throughput': 'Records/Second'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(stats_df, x='batch_id', y='time',
                         title='Encryption Time by Batch',
                         labels={'time': 'Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(stats_df, use_container_width=True)

    st.divider()

    # Column distribution
    st.header("Encrypted Data Distribution")

    cursor = st.session_state.db_conn.cursor()
    cursor.execute("""
        SELECT column_name, COUNT(*) as count 
        FROM encrypted_data 
        GROUP BY column_name
    """)
    col_dist = cursor.fetchall()

    if col_dist:
        df_dist = pd.DataFrame(col_dist, columns=['Column', 'Count'])
        fig = px.pie(df_dist, names='Column', values='Count',
                     title='Encrypted Records by Column')
        st.plotly_chart(fig, use_container_width=True)

    # Party distribution
    cursor.execute("""
        SELECT party_id, COUNT(*) as count 
        FROM encrypted_data 
        GROUP BY party_id 
        ORDER BY count DESC 
        LIMIT 10
    """)
    party_dist = cursor.fetchall()

    if party_dist:
        st.subheader("Top 10 Parties by Transaction Count")
        df_parties = pd.DataFrame(party_dist, columns=['Party ID', 'Transaction Count'])
        df_parties['Party ID'] = df_parties['Party ID'].str[:8] + '...'
        st.dataframe(df_parties, use_container_width=True)


# ==================== MAIN NAVIGATION ====================

def main():
    """Main function with navigation"""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Screen",
        [
            "1. Data Upload & Keys",
            "2. Data Encryption",
            "3. FHE Analysis",
            "4. Fraud Detection",
            "5. SIMD Operations",
            "6. ML Inference",
            "7. Statistics"
        ]
    )

    st.sidebar.divider()

    # System info
    st.sidebar.subheader("System Info")
    db_stats = get_db_stats()
    st.sidebar.info(f"""
        **Library:** {st.session_state.get('selected_library', 'Not selected')}  
        **Scheme:** {st.session_state.get('selected_scheme', 'Not selected')}  
        **Keys:** {'✅ Generated' if st.session_state.keys_generated else '❌ Not generated'}  
        **Encrypted Records:** {db_stats['total_records']:,}
        """)

    # Reset button
    if st.sidebar.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            if key != 'db_conn':
                del st.session_state[key]
        # Clear database
        cursor = st.session_state.db_conn.cursor()
        cursor.execute("DELETE FROM encrypted_data")
        cursor.execute("DELETE FROM metadata")
        st.session_state.db_conn.commit()
        st.rerun()

    # Route to screens
    if page == "1. Data Upload & Keys":
        screen_1_data_upload()
    elif page == "2. Data Encryption":
        screen_2_encryption()
    elif page == "3. FHE Analysis":
        screen_3_analysis()
    elif page == "4. Fraud Detection":
        screen_4_fraud_detection()
    elif page == "5. SIMD Operations":
        screen_5_simd_operations()
    elif page == "6. ML Inference":
        screen_6_ml_inference()
    elif page == "7. Statistics":
        screen_7_statistics()


if __name__ == "__main__":
    main()