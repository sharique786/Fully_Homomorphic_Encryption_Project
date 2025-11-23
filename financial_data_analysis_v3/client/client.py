"""
Complete Enhanced Streamlit Client
"""
import json
import concurrent.futures
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import base64
from threading import Lock
import sqlite3
import uuid
import gzip

SERVER_URL = "http://localhost:8000"
session_lock = Lock()

# Page config
st.set_page_config(
    page_title="FHE Financial Data Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

from client_key_mgr import ClientKeyManager

# ==================== SQLite Database Setup ====================

def init_db():
    """Initialize SQLite database for encrypted storage"""
    conn = sqlite3.connect('fhe_encrypted_data.db', check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute(''' 
            DROP TABLE IF EXISTS encrypted_data
        ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS encrypted_data (
            id TEXT PRIMARY KEY,
            party_id TEXT NOT NULL,
            email_id TEXT NOT NULL,
            account_id TEXT,
            transaction_id TEXT,
            column_name TEXT NOT NULL,
            encrypted_value BLOB NOT NULL,
            original_value TEXT,
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

# ==================== Initialize Client Key Manager ====================

if 'key_manager' not in st.session_state:
    st.session_state.key_manager = ClientKeyManager()

# ==================== Session State ====================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'encrypted_metadata' not in st.session_state:
    st.session_state.encrypted_metadata = {}
if 'encryption_stats' not in st.session_state:
    st.session_state.encryption_stats = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.expected_recon_data = None
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
                         column_name, encrypted_value, original_value, data_type,
                         transaction_date, batch_id):
    """Store encrypted data in SQLite with original value"""
    conn = st.session_state.db_conn
    cursor = conn.cursor()

    record_id = str(uuid.uuid4())

    cursor.execute('''
        INSERT INTO encrypted_data 
        (id, party_id, email_id, account_id, transaction_id, column_name, 
         encrypted_value, original_value, data_type, transaction_date, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (record_id, party_id, email_id, account_id, transaction_id,
          column_name, encrypted_value, str(original_value), data_type,
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


def calculate_transaction_metrics(email_or_party_id):
    """Calculate totals from synthetic data in session state."""

    if "data" not in st.session_state or st.session_state.data is None:
        return None

    # Convert to DataFrame if list
    df = st.session_state.data
    result = {}
    if isinstance(df, list):
        df = pd.DataFrame(df)

    print(f"Original DF head : {df.head()}")
    id_mask = (df["Email ID"] == email_or_party_id) | (df["Party ID"] == email_or_party_id)
    df_id = df[id_mask]
    account_id = df_id["Account ID"].iloc[0] if not df_id.empty else None

    if account_id:
        # Filter rows where user account is involved
        mask = (df["From Account"] == account_id) | (df["To Account"] == account_id)
        df_user = df[mask]

        total_transferred = df_user[df_user["From Account"] == account_id]["Amount"].sum()
        total_received = df_user[df_user["To Account"] == account_id]["Amount"].sum()
        total_amount = df_user["Amount"].sum()

        result = {
            "total_transferred": float(total_transferred),
            "total_received": float(total_received),
            "total_amount": float(total_amount)
        }

        # Save results to another session state object
        st.session_state.expected_recon_data = result

    return result


def check_server_health():
    """Check if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles date and datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

# def call_server(endpoint, method="GET", data=None):
#     """Call server API"""
#     try:
#         url = f"{SERVER_URL}{endpoint}"
#         if method == "GET":
#             response = requests.get(url, timeout=30)
#         else:
#             response = requests.post(url, json=data, timeout=300)
#
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.Timeout:
#         st.error("⏱️ Server timeout - operation taking longer than expected")
#         return None
#     except requests.exceptions.RequestException as e:
#         st.error(f"❌ Server error: {str(e)}")
#         return None

def call_server(endpoint, method="GET", data=None):
    """Call server API with custom JSON encoder for dates"""
    try:
        url = f"{SERVER_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            # FIX: Use custom encoder and convert data to JSON string first
            if data:
                json_data = json.dumps(data, cls=DateTimeEncoder)
                response = requests.post(
                    url,
                    data=json_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=300
                )
            else:
                response = requests.post(url, timeout=300)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Server timeout - operation taking longer than expected")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Server error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def compress_encrypted_data(data):
    """Compress encrypted data before sending"""
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            data = str(data).encode('utf-8')

        compressed = gzip.compress(data, compresslevel=6)
        # Return as base64 string for JSON transmission
        return base64.b64encode(compressed).decode('utf-8')
    except Exception as e:
        print(f"Compression error: {e}")
        return None


def compress_encrypted_list(data_list):
    """Compress a list of encrypted values"""
    compressed_list = []
    for item in data_list:
        if isinstance(item, bytes):
            compressed = gzip.compress(item, compresslevel=6)
            compressed_list.append(base64.b64encode(compressed).decode('utf-8'))
        elif isinstance(item, str):
            # If already base64, decode first, compress, then re-encode
            try:
                original = base64.b64decode(item)
                compressed = gzip.compress(original, compresslevel=6)
                compressed_list.append(base64.b64encode(compressed).decode('utf-8'))
            except:
                compressed_list.append(item)
        else:
            compressed_list.append(item)
    return compressed_list
# ==================== Custom JSON Encoder for dates ====================
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)



def list_to_csv(value):
    """
    Convert any list (flat or nested) into a comma-separated string.
    - Removes duplicates (keeps first occurrence)
    - Safely flattens nested lists
    - Converts all items to strings
    """

    if not isinstance(value, list):
        raise TypeError("Input must be a list")

    # Recursive flatten function
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    flat_items = list(flatten(value))

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in flat_items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    # Create comma-separated string
    return ",".join(str(x) for x in unique_items)

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
        num_records = st.number_input("Generate Synthetic Data", min_value=10, max_value=10000, value=1000, step=100)
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
            st.success(f"✅ These parameters are compatible with the selected work load: {workload_type}")
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
        # Step 1: Initialize server context
        if st.button("1️⃣ Initialize Server Context", type="primary"):
            with st.spinner("Setting up server context..."):
                context_data = {
                    "library": library,
                    "scheme": scheme,
                    "poly_modulus_degree": poly_degree,
                    "mult_depth": mult_depth,
                    "scale_mod_size": scale_mod_size if scale_value is None else 50,
                    "scale": float(scale_value) if scale_value else None,
                    "coeff_mod_bit_sizes": coeff_list,
                    "plain_modulus": plain_modulus if scheme == "BFV" else None
                }
                
                result = call_server("/generate_context", "POST", context_data)
                
                if result and result.get('status') == 'success':
                    st.session_state.server_context_ready = True
                    st.session_state.context_params = context_data
                    st.success("✅ Server context initialized!")
                    st.info("👉 Now generate keys locally (Step 2)")
                    st.rerun()
    
    with col2:
        # Step 2: Generate keys locally
        if st.session_state.get('server_context_ready', False):
            if st.button("2️⃣ Generate Keys Locally", type="primary"):
                with st.spinner("Generating keys on client..."):
                    context_params = st.session_state.context_params
                    
                    success = st.session_state.key_manager.generate_context_locally(
                        library=library,
                        scheme=scheme,
                        context_params=context_params
                    )
                    
                    if success:
                        st.session_state.keys_generated = True
                        st.session_state.library = library
                        st.session_state.scheme = scheme
                        st.success("✅ Keys generated locally!")
                        st.info("👉 Now upload public keys to server (Step 3)")
                        st.rerun()
        else:
            st.info("⏸️ Initialize server context first (Step 1)")
    
    # Step 3: Upload public keys to server
    if st.session_state.get('keys_generated', False):
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("3️⃣ Upload Public Keys to Server", type="primary"):
                with st.spinner("Uploading public keys..."):
                    public_keys = st.session_state.key_manager.get_public_keys_for_server()
                    
                    if public_keys:
                        # NEW API endpoint
                        result = call_server("/upload_public_keys", "POST", public_keys)
                        
                        if result and result.get('status') == 'success':
                            st.session_state.public_keys_uploaded = True
                            st.success("✅ Public keys uploaded to server!")
                            st.success("🎉 Setup complete! Ready for encryption.")
                            st.rerun()
        
        with col2:
            # Download private key backup
            if st.button("💾 Backup Private Key"):
                context_bytes = st.session_state.key_manager.context.serialize()
                
                st.download_button(
                    label="📥 Download Private Key",
                    data=context_bytes,
                    file_name=f"fhe_private_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin",
                    mime="application/octet-stream"
                )
                
                st.warning("⚠️ Keep this file secure! Anyone with this file can decrypt your data.")
    
    # Show current status
    if st.session_state.get('keys_generated', False):
        st.divider()
        st.subheader("🔐 Security Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Server Context: Ready")
        with col2:
            st.success("✅ Local Keys: Generated")
        with col3:
            if st.session_state.get('public_keys_uploaded', False):
                st.success("✅ Public Keys: Uploaded")
            else:
                st.warning("⏸️ Public Keys: Not Uploaded")

    # with col1:
    #     if st.button("🔑 Generate Keys", type="primary", use_container_width=True):
    #         with st.spinner("Generating encryption keys..."):
    #             # Generate context
    #             context_data = {
    #                 "library": library,
    #                 "scheme": scheme,
    #                 "poly_modulus_degree": poly_degree,
    #                 "mult_depth": mult_depth,
    #                 "scale_mod_size": scale_mod_size if scale_value is None else 50,
    #                 "scale": float(scale_value) if scale_value else None,
    #                 "coeff_mod_bit_sizes": coeff_list if library == "TenSEAL" else None,
    #                 "plain_modulus": plain_modulus if scheme == "BFV" else None
    #             }

    #             result = call_server("/generate_context", "POST", context_data)

    #             if result and result.get('status') == 'success':
    #                 # Generate keys
    #                 key_data = {
    #                     "library": library,
    #                     "scheme": scheme,
    #                     "params": {
    #                         "poly_modulus_degree": poly_degree,
    #                         "mult_depth": mult_depth,
    #                         "scale_mod_size": scale_mod_size if scale_value is None else 50,
    #                         "scale": float(scale_value) if scale_value else None
    #                     }
    #                 }

    #                 key_result = call_server("/generate_keys", "POST", key_data)

    #                 if key_result and key_result.get('status') == 'success':
    #                     st.session_state.keys_generated = True
    #                     st.session_state.keys_info = key_result.get('keys')
    #                     st.session_state.library = library
    #                     st.session_state.scheme = scheme
    #                     st.success("✅ Keys generated successfully!")
    #                     st.rerun()

    # with col2:
    #     if st.session_state.keys_generated:
    #         st.success("✅ Keys Generated")
    #         if st.button("📥 Download Keys"):
    #             keys_json = json.dumps(st.session_state.keys_info, indent=2)
    #             st.download_button(
    #                 "Download Keys JSON",
    #                 data=keys_json,
    #                 file_name=f"fhe_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #                 mime="application/json"
    #             )

    # # Display keys (truncated)
    # if st.session_state.keys_generated and st.session_state.keys_info:
    #     with st.expander("🔍 View Generated Keys"):
    #         st.json({
    #             "public_key": st.session_state.keys_info.get('public_key', '')[:100] + "...",
    #             "private_key": st.session_state.keys_info.get('private_key', '')[:100] + "...",
    #             "scheme": st.session_state.scheme,
    #             "library": st.session_state.library
    #         })


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

    # Call scheme limitations API
    limitations_result = call_server(f"/scheme_limitations/{st.session_state.library}/{st.session_state.scheme}")

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
        - **⚠️ Requires CKKS scheme and NUMERIC data only**
        - Best for: Large numeric datasets, analytics

        **Batch Processing:**
        - Processes in optimized batches of 256
        - Balances speed and flexibility
        - 2-5x faster than individual
        - Works with all schemes and data types
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
    if limitations_result and 'limitations' in limitations_result:
        with st.expander("ℹ️ Scheme Limitations"):
            limitations = limitations_result['limitations']
            st.json(limitations)

    all_selected = selected_numeric + selected_text + selected_dates

    # Validation for packed_vector mode
    if simd_mode == "packed_vector":
        non_numeric_selected = selected_text + selected_dates
        if non_numeric_selected:
            st.error(f"❌ Packed Vector mode only supports NUMERIC columns")
            st.error(f"Please remove these columns: {', '.join(non_numeric_selected)}")
            st.info("💡 Use 'Individual' or 'Batch Processing' mode for text/date columns")
            all_selected = []  # Disable encryption button

    # Encryption Button
    if st.button("🔒 Encrypt Selected Columns", type="primary", disabled=len(all_selected) == 0):
        # Clear previous session data
        if 'current_encryption_batch' in st.session_state:
            st.info("🔄 Starting new encryption session...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        # Store current batch ID
        st.session_state.current_encryption_batch = batch_id

        # COLUMN-BASED ENCRYPTION
        for idx, col in enumerate(all_selected):
            status_text.text(f"Encrypting {col}... ({idx + 1}/{len(all_selected)})")

            # Determine data type
            if col in numeric_cols:
                data_type = "numeric"
            elif col in text_cols:
                data_type = "text"
            else:
                data_type = "date"

            # FIX: Handle ALL date/datetime serialization types
            if data_type == "date":
                column_data = []
                for val in st.session_state.data[col]:
                    if pd.isna(val) or val is None:
                        column_data.append(None)
                    elif isinstance(val, str):
                        # Already a string
                        column_data.append(val)
                    elif isinstance(val, pd.Timestamp):
                        # Pandas Timestamp
                        column_data.append(val.isoformat())
                    elif isinstance(val, datetime):
                        # Python datetime
                        column_data.append(val.isoformat())
                    elif isinstance(val, date):
                        # Python date (DOB case)
                        column_data.append(val.isoformat())
                    elif hasattr(val, 'isoformat'):
                        # Any object with isoformat method
                        column_data.append(val.isoformat())
                    else:
                        # Fallback: convert to string
                        column_data.append(str(val))
            else:
                # Numeric or text columns - no special handling needed
                column_data = st.session_state.data[col].tolist()

            party_ids = st.session_state.data['Party ID'].tolist()
            email_ids = st.session_state.data['Email ID'].tolist()
            account_ids = st.session_state.data['Account ID'].tolist()
            transaction_ids = st.session_state.data['Transaction ID'].tolist()

            # FIX: Also handle transaction_dates serialization
            transaction_dates = []
            for val in st.session_state.data['Transaction Date']:
                if pd.isna(val) or val is None:
                    transaction_dates.append(None)
                elif isinstance(val, str):
                    transaction_dates.append(val)
                elif isinstance(val, (pd.Timestamp, datetime, date)):
                    transaction_dates.append(val.isoformat())
                elif hasattr(val, 'isoformat'):
                    transaction_dates.append(val.isoformat())
                else:
                    transaction_dates.append(str(val))

            enc_data = {
                "library": st.session_state.selected_library,
                "scheme": st.session_state.selected_scheme,
                "column_name": col,
                "data_type": data_type,
                "column_data": column_data,
                "party_ids": party_ids,
                "email_ids": email_ids,
                "account_ids": account_ids,
                "transaction_ids": transaction_ids,
                "transaction_dates": transaction_dates,
                "batch_id": batch_id,
                "simd_mode": simd_mode,
                "compress_response": True
            }

            result = call_server("/encrypt_column", "POST", enc_data)

            if result and result.get('status') == 'success':
                encrypted_results = result.get('encrypted_results', [])

                if simd_mode == "packed_vector":
                    for enc_result in encrypted_results:
                        encrypted_value = enc_result.get('encrypted_value')
                        batch_start = enc_result.get('batch_start', 0)
                        batch_size = enc_result.get('batch_size', 0)

                        if result.get('compressed', False):
                            try:
                                compressed_bytes = base64.b64decode(encrypted_value)
                                decompressed_bytes = gzip.decompress(compressed_bytes)
                                encrypted_value = decompressed_bytes
                            except Exception as e:
                                print(f"Decompression error: {e}")

                        record_id = str(uuid.uuid4())
                        conn = st.session_state.db_conn
                        cursor = conn.cursor()

                        flattened_party_ids = list_to_csv(party_ids)
                        flattened_email_ids = list_to_csv(email_ids)
                        flattened_account_ids = list_to_csv(account_ids)
                        flattened_transaction_ids = list_to_csv(transaction_ids)
                        flattened_transaction_dates = list_to_csv(transaction_dates)

                        cursor.execute('''
                                            INSERT INTO encrypted_data 
                                            (id, party_id, email_id, account_id, transaction_id, column_name, 
                                             encrypted_value, original_value, data_type, transaction_date, batch_id)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (record_id,
                                              flattened_party_ids,
                                              flattened_email_ids,
                                              flattened_account_ids, flattened_transaction_ids,
                                              col,
                                              encrypted_value.encode() if isinstance(encrypted_value,
                                                                                     str) else encrypted_value,
                                              f"Batch: {batch_start}-{batch_start + batch_size}",
                                              data_type,
                                              flattened_transaction_dates,
                                              batch_id))
                        conn.commit()
                else:
                    for enc_result in encrypted_results:
                        encrypted_value = enc_result.get('encrypted_value')

                        if result.get('compressed', False) and isinstance(encrypted_value, str):
                            try:
                                compressed_bytes = base64.b64decode(encrypted_value)
                                decompressed_bytes = gzip.decompress(compressed_bytes)
                                encrypted_value = decompressed_bytes
                            except Exception as e:
                                print(f"Decompression error: {e}")

                        party_id = enc_result.get('party_id')
                        email_id = enc_result.get('email_id')
                        account_id = enc_result.get('account_id')
                        transaction_id = enc_result.get('transaction_id')
                        transaction_date_str = enc_result.get('transaction_date')
                        idx_val = enc_result.get('index', 0)

                        original_value = st.session_state.data.iloc[idx_val][col]
                        transaction_date = pd.Timestamp(transaction_date_str) if transaction_date_str else None

                        store_encrypted_data(
                            party_id, email_id, account_id, transaction_id,
                            col,
                            encrypted_value if isinstance(encrypted_value, bytes) else base64.b64decode(
                                encrypted_value),
                            original_value,
                            data_type,
                            transaction_date,
                            batch_id
                        )

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

        with st.expander("📋 Column Encryption Details", expanded=True):
            for col, meta in st.session_state.encrypted_metadata.items():
                simd_mode = meta.get('simd_mode', 'individual')
                encryption_time = meta.get('encryption_time', 0)
                st.write(
                    f"**{col}**: {meta['count']} records ({meta['data_type']}) - "
                    f"{meta['library']}/{meta['scheme']} - "
                    f"Mode: {simd_mode} - Time: {encryption_time:.2f}s"
                )

        # Encrypted data preview
        st.divider()
        st.header("🔍 Encrypted Data Preview")

        col1, col2 = st.columns([2, 1])

        with col1:
            search_email = st.text_input("🔎 Search by Email ID", placeholder="Enter email to filter...")

        with col2:
            preview_limit = st.number_input("Max Rows", min_value=5, max_value=100, value=10)

        conn = st.session_state.db_conn
        cursor = conn.cursor()

        # Show only current batch data
        if 'current_encryption_batch' in st.session_state:
            batch_filter = st.session_state.current_encryption_batch

            if search_email:
                cursor.execute("""
                    SELECT email_id, column_name, encrypted_value, original_value, data_type, batch_id
                    FROM encrypted_data 
                    WHERE email_id LIKE ? AND batch_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (f"%{search_email}%", batch_filter, preview_limit))
            else:
                cursor.execute("""
                    SELECT email_id, column_name, encrypted_value, original_value, data_type, batch_id
                    FROM encrypted_data 
                    WHERE batch_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (batch_filter, preview_limit))
        else:
            if search_email:
                cursor.execute("""
                    SELECT email_id, column_name, encrypted_value, original_value, data_type, batch_id
                    FROM encrypted_data 
                    WHERE email_id LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (f"%{search_email}%", preview_limit))
            else:
                cursor.execute("""
                    SELECT email_id, column_name, encrypted_value, original_value, data_type, batch_id
                    FROM encrypted_data 
                    ORDER BY id DESC
                    LIMIT ?
                """, (preview_limit,))

        preview_data = cursor.fetchall()

        if preview_data:
            preview_df = pd.DataFrame(preview_data,
                                      columns=['Email ID', 'Column', 'Encrypted Value', 'Original Value', 'Data Type',
                                               'Batch ID'])

            preview_df['Encrypted Value'] = preview_df['Encrypted Value'].apply(
                lambda x: str(base64.b64encode(x[:50]) if isinstance(x, bytes) else x)[:50] + "..."
            )

            st.dataframe(preview_df, use_container_width=True)

            st.caption(
                f"Showing {len(preview_data)} encrypted records from current session (Batch: {st.session_state.get('current_encryption_batch', 'N/A')})")
        else:
            st.info("No encrypted data found. Please encrypt columns first.")

        # Clear data button
        if st.button("🔄 Clear All Encrypted Data"):
            # if st.checkbox("⚠️ Confirm: Delete all encrypted data from database?"):
            cursor = st.session_state.db_conn.cursor()
            cursor.execute("DELETE FROM encrypted_data")
            cursor.execute("DELETE FROM metadata")
            st.session_state.db_conn.commit()
            print("All encrypted data cleared from database")
            st.session_state.encrypted_metadata = {}
            st.session_state.encryption_stats = []
            if 'current_encryption_batch' in st.session_state:
                del st.session_state.current_encryption_batch
            st.success("✅ All encrypted data cleared")
            st.rerun()

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
        search_email = st.text_input("Enter Email ID")
        search_results = None
        if search_email:
            cursor.execute("SELECT DISTINCT party_id, email_id FROM encrypted_data WHERE email_id = ?", (search_email,))
            search_results = cursor.fetchall()
            # print(f"Search results by Email ID: {search_results}")

        # selected_idx = None
        # party_options = [f"{p[0][:50]}" for p in parties]
        # selected_idx = st.selectbox("Select Party ID", range(len(party_options)),
        #                             format_func=lambda x: party_options[x])

        if search_results:
            selected_party = search_results[0][0]
            selected_email = search_results[0][1]
            party_options = [f"{selected_party[0][:50]}"]
            selected_idx = st.selectbox("Select Party ID", [selected_party])
        else:
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

        if 'reconciliation_data' not in st.session_state:
            st.session_state.reconciliation_data = []

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
                with st.spinner("Decrypting (client side)..."):
                    # # Decrypt each result
                    # decrypted = {}

                    # for key in ['total_transferred', 'total_received', 'average_amount']:
                    #     if key in results:
                    #         decrypt_result = call_server("/decrypt", "POST", {
                    #             "library": st.session_state.selected_library,
                    #             "result_data": results[key],
                    #             "data_type": "numeric"
                    #         })

                    #         if decrypt_result and decrypt_result.get('status') == 'success':
                    #             decrypted[key] = decrypt_result.get('decrypted_value', 0)

                    # st.session_state.decrypted_results = decrypted
                    # st.rerun()
                    key_manager = st.session_state.key_manager
                    
                    decrypted = {}
                    
                    for key in ['total_transferred', 'total_received', 'average_amount']:
                        if key in results:
                            encrypted_value = results[key]
                            
                            # Decrypt locally (no server call)
                            decrypted_value = key_manager.decrypt_locally(
                                encrypted_value,
                                data_type='numeric'
                            )
                            
                            if decrypted_value is not None:
                                decrypted[key] = decrypted_value
                    st.session_state.decrypted_results = decrypted
                    st.rerun()
        # Show decrypted results
        if 'decrypted_results' in st.session_state and st.session_state.decrypted_results:
            st.divider()
            st.subheader("🔓 Decrypted Results")

            dec = st.session_state.decrypted_results
            email_or_account_id = selected_email if selected_email else selected_party

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transferred", f"${dec.get('total_transferred', 0):,.2f}")
            with col2:
                st.metric("Total Received", f"${dec.get('total_received', 0):,.2f}")
            with col3:
                st.metric("Average Amount", f"${dec.get('average_amount', 0):,.2f}")

            st.caption("🔒 Decryption performed entirely on client - private key never sent to server")
            # Currency distribution
            filtered_data = st.session_state.data[
                (st.session_state.data['Party ID'] == selected_party) |
                (st.session_state.data['Email ID'] == selected_email)
                ]
            currency_dist = filtered_data['Currency'].value_counts().to_dict()
            st.session_state.analysis_results['currency_distribution'] = currency_dist

            if 'currency_distribution' in results:
                st.subheader("Payment Distribution by Currency")
                curr_dist = results['currency_distribution']

                if curr_dist:
                    df_curr = pd.DataFrame(list(curr_dist.items()), columns=['Currency', 'Count'])
                    fig = px.pie(df_curr, names='Currency', values='Count', title='Transaction Distribution')
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🔄 Reconciliation: Encrypted vs Expected")

            if st.session_state.data is not None:
                # Get filtered data matching the query
                filtered_data = st.session_state.data[
                    (st.session_state.data['Party ID'] == results.get('party_id')) |
                    (st.session_state.data['Email ID'] == results.get('email_id'))
                    ]

                # Calculate expected values from original data
                expected_total = filtered_data['Amount'].sum()
                expected_avg = filtered_data['Amount'].mean()
                expected_count = len(filtered_data)

                # Get decrypted values
                decrypted_transferred = dec.get('total_transferred', 0)
                decrypted_received = dec.get('total_received', 0)
                decrypted_total = decrypted_transferred + decrypted_received
                decrypted_avg = dec.get('average_amount', 0)
                decrypted_count = results.get('transaction_count', 0)

                # Tolerance for floating point comparison (0.1%)
                tolerance = 0.001

                # Create reconciliation table
                reconciliation_data = []

                # Total Amount reconciliation
                total_diff = abs(expected_total - decrypted_total)
                total_diff_pct = (total_diff / max(expected_total, 1)) * 100
                total_match = (total_diff / max(expected_total, 1)) < tolerance
                reconciliation_data.append({
                    "Metric": "Total Amount",
                    "Expected": f"${expected_total:,.2f}",
                    "Decrypted (Transferred + Received)": f"${decrypted_total:,.2f}",
                    "Difference": f"${total_diff:,.2f} ({total_diff_pct:.4f}%)",
                    "Status": "✅ PASSED" if total_match else "❌ FAILED"
                })

                # Transferred Amount reconciliation (compare with half of expected)
                expected_transferred = expected_total / 2
                transferred_diff = abs(expected_transferred - decrypted_transferred)
                transferred_diff_pct = (transferred_diff / max(expected_transferred, 1)) * 100
                transferred_match = (transferred_diff / max(expected_transferred, 1)) < tolerance
                reconciliation_data.append({
                    "Metric": "Total Transferred",
                    "Expected": f"${expected_transferred:,.2f} (est.)",
                    "Decrypted (Transferred + Received)": f"${decrypted_transferred:,.2f}",
                    "Difference": f"${transferred_diff:,.2f} ({transferred_diff_pct:.4f}%)",
                    "Status": "✅ PASSED" if transferred_match else "⚠️ CHECK"
                })

                # Received Amount reconciliation (compare with half of expected)
                expected_received = expected_total / 2
                received_diff = abs(expected_received - decrypted_received)
                received_diff_pct = (received_diff / max(expected_received, 1)) * 100
                received_match = (received_diff / max(expected_received, 1)) < tolerance
                reconciliation_data.append({
                    "Metric": "Total Received",
                    "Expected": f"${expected_received:,.2f} (est.)",
                    "Decrypted (Transferred + Received)": f"${decrypted_received:,.2f}",
                    "Difference": f"${received_diff:,.2f} ({received_diff_pct:.4f}%)",
                    "Status": "✅ PASSED" if received_match else "⚠️ CHECK"
                })

                # Average Amount reconciliation
                avg_diff = abs(expected_avg - decrypted_avg)
                avg_diff_pct = (avg_diff / max(expected_avg, 1)) * 100
                avg_match = (avg_diff / max(expected_avg, 1)) < tolerance
                reconciliation_data.append({
                    "Metric": "Average Amount",
                    "Expected": f"${expected_avg:,.2f}",
                    "Decrypted (Transferred + Received)": f"${decrypted_avg:,.2f}",
                    "Difference": f"${avg_diff:,.2f} ({avg_diff_pct:.4f}%)",
                    "Status": "✅ PASSED" if avg_match else "❌ FAILED"
                })

                # Transaction Count reconciliation
                count_diff = abs(expected_count - decrypted_count)
                count_match = expected_count == decrypted_count
                reconciliation_data.append({
                    "Metric": "Transaction Count",
                    "Expected": f"{expected_count:,}",
                    "Decrypted (Transferred + Received)": f"{decrypted_count:,}",
                    "Difference": f"{count_diff:,}",
                    "Status": "✅ PASSED" if count_match else "❌ FAILED"
                })

                # Display reconciliation table
                recon_df = pd.DataFrame(reconciliation_data)
                st.dataframe(recon_df, use_container_width=True)

                # Overall status
                all_passed = all(row["Status"] == "✅ PASSED" for row in reconciliation_data
                                 if row["Status"] in ["✅ PASSED", "❌ FAILED"])

                if all_passed:
                    st.success("✅ All reconciliation checks PASSED! Encrypted computations are accurate.")
                else:
                    failed_checks = [row["Metric"] for row in reconciliation_data if row["Status"] == "❌ FAILED"]
                    st.error(f"❌ Reconciliation FAILED for: {', '.join(failed_checks)}")
                    st.info(
                        "💡 Note: Transferred/Received are estimated as 50/50 split. Check individual transactions for exact values.")
            else:
                st.warning("⚠️ Original data not available for reconciliation")



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
        party_options = [f"{p[0][:50]} ({p[1]})" for p in parties]
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
        st.warning(f"No encrypted data found for party {selected_party[:50]}")
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
                    encrypted_score = fraud_result.get('encrypted_score')

                    if encrypted_score is None:
                        st.error("❌ Fraud score is None. Detection may have failed on server.")
                        st.info("Possible reasons:")
                        st.info("• Fraud detection computation returned no result")
                        st.info("• All encrypted amounts failed to process")
                        st.info("• Check server logs for details")
                        st.session_state.decrypted_fraud_score = None
                    else:
                        key_manager = st.session_state.key_manager                        
                        fraud_score = key_manager.decrypt_locally(
                                    encrypted_score,
                                    data_type='numeric'
                                )
                                
                        if fraud_score is not None:                            
                            st.session_state.decrypted_fraud_score = fraud_score
                        else:
                            st.error("❌ Decryption failed")
                            st.session_state.decrypted_fraud_score = None
                                

                        # decrypt_result = call_server("/decrypt", "POST", {
                        #     "library": st.session_state.selected_library,
                        #     "result_data": encrypted_score,
                        #     "data_type": "numeric"
                        # })

                        # if decrypt_result and decrypt_result.get('status') == 'success':
                        #     fraud_score = decrypt_result.get('decrypted_value')
                        #     st.session_state.decrypted_fraud_score = fraud_score
                        # else:
                        #     st.error("❌ Decryption failed")
                        #     st.session_state.decrypted_fraud_score = None

                    st.rerun()

        if 'decrypted_fraud_score' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Fraud Score Analysis")

            fraud_score = st.session_state.decrypted_fraud_score

            # FIX: Complete None-safe handling
            if fraud_score is None:
                st.error("❌ Fraud score is None - cannot display analysis")
                st.info("**Troubleshooting:**")
                st.info("1. Check that encrypted amounts are valid")
                st.info("2. Verify party has encrypted transaction data")
                st.info("3. Review server logs for fraud detection errors")
                st.info("4. Try running fraud detection again")

                # Clear the session state so user can retry
                if st.button("Clear and Retry"):
                    del st.session_state.decrypted_fraud_score
                    del st.session_state.fraud_result
                    st.rerun()

            else:
                # Safe conversion to float
                try:
                    fraud_score_float = float(fraud_score)

                    # Normalize score to 0-1 range
                    if fraud_score_float == 0:
                        normalized_score = 0.0
                        st.info("ℹ️ Fraud score is zero - this may indicate no fraud signals detected")
                    elif fraud_score_float > 1:
                        # Score is already large, normalize by dividing by 100
                        normalized_score = min(max(fraud_score_float / 100.0, 0.0), 1.0)
                    else:
                        # Score is between 0-1
                        normalized_score = min(max(fraud_score_float, 0.0), 1.0)

                    st.write(f"**Raw fraud score:** {fraud_score_float:.6f}")
                    st.write(f"**Normalized score (0-1):** {normalized_score:.4f}")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric("Fraud Score", f"{normalized_score:.4f}")

                        # Risk classification
                        if normalized_score < 0.3:
                            st.success("✅ LOW RISK")
                            risk_level = "LOW"
                        elif normalized_score < 0.6:
                            st.warning("⚠️ MEDIUM RISK")
                            risk_level = "MEDIUM"
                        elif normalized_score < 0.8:
                            st.error("🚨 HIGH RISK")
                            risk_level = "HIGH"
                        else:
                            st.error("🚨🚨 CRITICAL RISK")
                            risk_level = "CRITICAL"

                        st.write(f"**Risk Level:** {risk_level}")

                    with col2:
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=normalized_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Fraud Risk Score"},
                            number={'suffix': "", 'font': {'size': 40}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickwidth': 1},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 0.3], 'color': 'lightgreen'},
                                    {'range': [0.3, 0.6], 'color': 'yellow'},
                                    {'range': [0.6, 0.8], 'color': 'orange'},
                                    {'range': [0.8, 1], 'color': 'red'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.8
                                }
                            }
                        ))

                        fig.update_layout(
                            height=300,
                            margin={'t': 50, 'b': 0, 'l': 0, 'r': 0}
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Additional analysis
                    st.divider()
                    st.subheader("📊 Fraud Analysis Details")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Detection Type", fraud_result.get('detection_type', 'N/A').replace('_', ' ').title())

                    with col2:
                        st.metric("Computation Time", f"{fraud_result.get('computation_time', 0):.3f}s")

                    with col3:
                        st.metric("Data Points Analyzed", len(fraud_result.get('encrypted_amounts',
                                                                               [])) if 'encrypted_amounts' in fraud_result else 'N/A')

                    # Recommendations based on risk level
                    st.subheader("🎯 Recommendations")

                    if normalized_score < 0.3:
                        st.success("✅ **Low Risk - No Action Required**")
                        st.write("• Transaction patterns appear normal")
                        st.write("• Continue standard monitoring")
                    elif normalized_score < 0.6:
                        st.warning("⚠️ **Medium Risk - Monitor Closely**")
                        st.write("• Review recent transaction history")
                        st.write("• Watch for unusual patterns")
                        st.write("• Consider additional verification")
                    elif normalized_score < 0.8:
                        st.error("🚨 **High Risk - Investigation Recommended**")
                        st.write("• Conduct detailed transaction review")
                        st.write("• Contact customer for verification")
                        st.write("• Consider temporary limits")
                    else:
                        st.error("🚨🚨 **Critical Risk - Immediate Action Required**")
                        st.write("• Suspend suspicious transactions immediately")
                        st.write("• Escalate to fraud investigation team")
                        st.write("• Notify customer and verify identity")

                except (ValueError, TypeError) as e:
                    st.error(f"❌ Error processing fraud score: {e}")
                    st.write(f"**Raw value received:** {repr(fraud_score)}")
                    st.write(f"**Type:** {type(fraud_score)}")
                    st.info("The fraud score value is not in the expected numeric format")


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
        party_options = [f"{p[0][:50]} ({p[1]})" for p in parties]
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
        st.warning(f"No encrypted data found for party {selected_party[:20]}")
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

                if not encrypted_results:
                    st.warning("⚠️ No encrypted results to decrypt")
                    st.session_state.decrypted_simd = []
                else:
                    key_manager = st.session_state.key_manager       
                    for enc_val in encrypted_results[:50]:                 
                        decrypt_result = key_manager.decrypt_locally(
                                    enc_val,
                                    data_type='numeric'
                                )
                            
                        if decrypt_result:
                                decrypted_series.append(decrypt_result)
                    

                    # for enc_val in encrypted_results[:50]:  # Limit for performance
                    #     decrypt_result = call_server("/decrypt", "POST", {
                    #         "library": st.session_state.selected_library,
                    #         "result_data": enc_val,
                    #         "data_type": "numeric"
                    #     })

                    #     if decrypt_result and decrypt_result.get('status') == 'success':
                    #         decrypted_series.append(decrypt_result.get('decrypted_value', 0))

                st.session_state.decrypted_simd = decrypted_series
                st.rerun()

        if 'decrypted_simd' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Time Series Analysis")

            series = st.session_state.decrypted_simd
            # print(f"Series : {series}")

            # Plot time series
            df_series = pd.DataFrame({
                'Time Index': range(len(series)),
                'Value': series
            })

            if not series or len(series) == 0:
                st.warning("⚠️ No data points to display. The decrypted series is empty.")
                st.info("This may occur if the SIMD operation produced no results...")
            else:
                fig = px.line(df_series, x='Time Index', y='Value',
                              title=f'{operation.replace("_", " ").title()} Analysis')
                st.plotly_chart(fig, use_container_width=True)
                clean_series = pd.Series(series).dropna().astype(float)

                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"${np.mean(clean_series):,.2f}")
                with col2:
                    st.metric("Std Dev", f"${np.std(clean_series):,.2f}")
                with col3:
                    st.metric("Min", f"${np.min(clean_series):,.2f}")
                with col4:
                    st.metric("Max", f"${np.max(clean_series):,.2f}")


# ==================== SCREEN 6: ML INFERENCE ====================
def cleanup_oversized_encrypted_data(max_size=500000):
    """Remove oversized encrypted values from database"""
    conn = st.session_state.db_conn
    cursor = conn.cursor()

    # Find oversized records
    cursor.execute("""
        SELECT COUNT(*), column_name, AVG(LENGTH(encrypted_value)) as avg_size
        FROM encrypted_data 
        WHERE LENGTH(encrypted_value) > ?
        GROUP BY column_name
    """, (max_size,))

    oversized = cursor.fetchall()

    if oversized:
        st.warning(f"Found {len(oversized)} column(s) with oversized encrypted data:")
        for count, col_name, avg_size in oversized:
            st.write(f"  • {col_name}: {count} records, avg size: {avg_size:,.0f} bytes")

        if st.button("🗑️ Remove Oversized Data"):
            cursor.execute("DELETE FROM encrypted_data WHERE LENGTH(encrypted_value) > ?", (max_size,))
            deleted = cursor.rowcount
            conn.commit()
            st.success(f"✅ Removed {deleted} oversized encrypted records")
            st.info("Please re-encrypt these columns with 'individual' or 'batch_processing' mode")
            st.rerun()
    else:
        st.success("✅ No oversized encrypted data found")

    return len(oversized) > 0


def screen_6_ml_inference():
    st.title("🤖 ML Inference on Encrypted Data")

    db_stats = get_db_stats()
    if db_stats['total_records'] == 0:
        st.warning("⚠️ Please encrypt data first")
        return

    st.info("💡 Run machine learning models on encrypted features")

    # Add data quality check
    with st.expander("🔧 Data Quality Check", expanded=False):
        st.write("**Check for oversized or corrupted encrypted data:**")
        has_oversized = cleanup_oversized_encrypted_data(max_size=500000)

        if has_oversized:
            st.warning("⚠️ Please clean up oversized data before proceeding with ML inference")

    # Model selection
    col1, col2 = st.columns(2)
    cursor = st.session_state.db_conn.cursor()
    searched_email = None
    with col1:
        search_email = st.text_input("Enter Email ID")
        if search_email:
            cursor.execute("SELECT DISTINCT email_id FROM encrypted_data WHERE email_id = ?", (search_email,))
            ret_val = cursor.fetchall()
            if ret_val:
                for res in ret_val:
                    searched_email = res[0]
                    # print(f"Searched Email {searched_email}")

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
        cursor.execute("SELECT DISTINCT column_name FROM encrypted_data")
        available_cols = [row[0] for row in cursor.fetchall()]

        selected_features = st.multiselect("Select Features", available_cols,
                                           default=available_cols[:3] if len(available_cols) >= 3 else available_cols)

    if not selected_features:
        st.warning("Please select at least one feature")
        return

    st.divider()
    st.subheader("⚙️ Encryption Method")

    encryption_method = st.radio(
        "Choose encryption method for ML inference:",
        ["Use Existing Encrypted Data", "Fresh Encryption (Recommended for large datasets)"],
        help="Fresh encryption creates new, properly-sized encrypted values for ML inference"
    )

    if encryption_method == "Fresh Encryption (Recommended for large datasets)":
        st.info("💡 This will encrypt a single sample value from each selected feature")

        if st.button("🔒 Encrypt Fresh Sample for ML", type="primary"):
            with st.spinner("Encrypting fresh samples..."):
                encrypted_features_list = []

                for feature_name in selected_features:
                    # Get one raw value from original data
                    if feature_name in st.session_state.data.columns:
                        sample_value = st.session_state.data[feature_name].iloc[0]

                        if pd.notna(sample_value):
                            # Prepare single value for encryption
                            enc_data = {
                                "library": st.session_state.selected_library,
                                "scheme": st.session_state.selected_scheme,
                                "column_name": feature_name,
                                "data_type": "numeric",  # ML typically uses numeric
                                "column_data": [float(sample_value)],
                                "party_ids": ["ML_INFERENCE"],
                                "email_ids": [searched_email],
                                "account_ids": ["ML_ACCOUNT"],
                                "transaction_ids": ["ML_TRANS"],
                                "transaction_dates": [datetime.now().isoformat()],
                                "batch_id": f"ML_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                "simd_mode": "individual",
                                "compress_response": True
                            }

                            result = call_server("/encrypt_column", "POST", enc_data)

                            if result and result.get('status') == 'success':
                                enc_results = result.get('encrypted_results', [])
                                if enc_results:
                                    enc_value = enc_results[0].get('encrypted_value')

                                    # Decompress if needed
                                    if result.get('compressed', False):
                                        compressed_bytes = base64.b64decode(enc_value)
                                        enc_bytes = gzip.decompress(compressed_bytes)
                                    else:
                                        enc_bytes = base64.b64decode(enc_value) if isinstance(enc_value, str) else enc_value

                                    # Re-compress for transmission
                                    compressed = gzip.compress(enc_bytes, compresslevel=9)
                                    encrypted_features_list.append(base64.b64encode(compressed).decode())

                                    st.write(f"  ✅ {feature_name}: {len(enc_bytes):,} → {len(compressed):,} bytes")
                        else:
                            st.error(f"❌ No valid data for {feature_name}")
                            return
                    else:
                        st.error(f"❌ Column {feature_name} not found in dataset")
                        return

                if len(encrypted_features_list) == len(selected_features):
                    st.success(f"✅ Fresh encryption complete for {len(selected_features)} features")

                    # Store in session for ML inference
                    st.session_state.fresh_ml_features = encrypted_features_list
                    st.info("Ready for ML inference! Configure model parameters below and click 'Run ML Inference'")
                else:
                    st.error("Failed to encrypt all features")
                    return

    # Model configuration
    st.divider()
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
            feature_details = []
            encrypted_features_list = []

            if encryption_method == "Fresh Encryption (Recommended for large datasets)":
                if 'fresh_ml_features' not in st.session_state:
                    st.error("Please encrypt fresh samples first using the button above")
                    return

                encrypted_features_list = st.session_state.fresh_ml_features
                st.info(f"Using freshly encrypted features ({len(encrypted_features_list)} features)")
            else:
                cursor = st.session_state.db_conn.cursor()
                st.info(f"🔍 Retrieving encrypted features for ML inference...")

                # Get one encrypted value per feature
                for feature_name in selected_features:
                    # Query with size limit and validation
                    cursor.execute("""
                        SELECT encrypted_value, LENGTH(encrypted_value) as size, party_id, email_id
                        FROM encrypted_data 
                        WHERE column_name = ? 
                          AND party_id != 'BATCH_'
                          AND LENGTH(encrypted_value) < 500000
                          AND email_id= ?
                        ORDER BY LENGTH(encrypted_value) ASC
                        LIMIT 1
                    """, (feature_name, searched_email,))

                    result = cursor.fetchone()

                    if result:
                        enc_value, value_size, party_id, email_id = result

                        # Validate size
                        if value_size > 500000:
                            st.error(f"❌ {feature_name}: Encrypted value too large ({value_size:,} bytes)")
                            st.error(f"   This suggests data concatenation or corruption")
                            st.info(f"   Expected size: 5KB-50KB per value")
                            st.info(f"   Try re-encrypting this column with 'individual' mode")
                            return

                        feature_details.append({
                            'name': feature_name,
                            'size': value_size,
                            'party_id': party_id[:30],
                            'email_id': email_id
                        })

                        # Compress the encrypted value
                        compressed = gzip.compress(enc_value, compresslevel=9)
                        compressed_size = len(compressed)

                        encrypted_features_list.append(base64.b64encode(compressed).decode())

                        st.write(f"  ✅ {feature_name}: {value_size:,} bytes → {compressed_size:,} bytes (compressed)")

                    else:
                        st.error(f"❌ No valid encrypted data found for: {feature_name}")
                        st.info("Possible issues:")
                        st.info("  • Column not encrypted yet")
                        st.info("  • All encrypted values exceed 500KB limit")
                        st.info("  • Try re-encrypting with 'individual' or 'batch_processing' mode")
                        return
                pass

            if len(encrypted_features_list) != len(selected_features):
                st.error(f"Failed to retrieve all features. Got {len(encrypted_features_list)}/{len(selected_features)}")
                return

            # Show feature summary
            st.success(f"✅ Retrieved {len(encrypted_features_list)} encrypted features")

            with st.expander("📋 Feature Details"):
                details_df = pd.DataFrame(feature_details)
                st.dataframe(details_df, use_container_width=True)

            # Calculate total payload
            total_original = sum(f['size'] for f in feature_details) if feature_details else 0
            total_compressed = sum(len(base64.b64decode(ef)) for ef in encrypted_features_list)
            compression_ratio = (1 - total_compressed / total_original) * 100 if feature_details else 0

            st.info(f"📦 Total: {total_original:,} bytes → {total_compressed:,} bytes ({compression_ratio:.1f}% reduction)")

            # Final size check
            if total_compressed > 5000000:  # 5MB limit
                st.error(f"⚠️ Payload too large: {total_compressed:,} bytes (limit: 5MB)")
                st.error("Cannot proceed with ML inference")
                st.info("Solutions:")
                st.info("  • Use fewer features")
                st.info("  • Re-encrypt columns individually")
                st.info("  • Check for data duplication in database")
                return

            # Prepare request
            request_data = {
                "library": st.session_state.selected_library,
                "model_type": model_type,
                "encrypted_features": encrypted_features_list,
                "weights": weights,
                "intercept": intercept,
                "compressed": True
            }

            if model_type == "polynomial":
                request_data["polynomial_degree"] = poly_degree

            # Send request
            with st.spinner("Running ML inference on server..."):
                result = call_server("/ml/inference", "POST", request_data)

            if result and result.get('status') == 'success':
                st.session_state.ml_result = result
                st.success("✅ ML inference complete!")
                st.rerun()
            else:
                st.error("❌ ML inference failed")
                if result:
                    st.error(f"Server message: {result.get('detail', 'Unknown error')}")
                st.info("Check server logs for detailed error information")

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
            encrypted_result = ml_result.get('encrypted_result')

            if encrypted_result is None:
                st.error("❌ Encrypted result is None. ML inference may have failed.")
                st.session_state.ml_prediction = None
            else:
                if st.button("🔓 Decrypt Prediction"):
                    with st.spinner("Decrypting..."):
                        key_manager = st.session_state.key_manager       
                        
                        prediction = key_manager.decrypt_locally(
                                    ml_result.get('encrypted_result'),
                                    data_type='numeric'
                                )
                            
                        if prediction:
                            st.session_state.ml_prediction = prediction
                            st.rerun()


                        # decrypt_result = call_server("/decrypt", "POST", {
                        #     "library": st.session_state.selected_library,
                        #     "result_data": ml_result.get('encrypted_result'),
                        #     "data_type": "numeric"
                        # })

                        # if decrypt_result and decrypt_result.get('status') == 'success':
                        #     prediction = decrypt_result.get('decrypted_value', 0)
                        #     st.session_state.ml_prediction = prediction
                        #     st.rerun()

        if 'ml_prediction' in st.session_state:
            st.divider()
            st.subheader("🔓 Decrypted Prediction")

            prediction = st.session_state.ml_prediction

            col1, col2 = st.columns(2)
            with col1:
                if prediction is None:
                    st.error("❌ Prediction is None. Check server logs for errors.")
                    st.info("Possible causes: encryption/decryption mismatch...")
                else:
                    st.metric("Prediction Value", f"{prediction:.4f}")

            with col2:
                if prediction is None:
                    st.error("❌ Prediction is None. Check server logs for errors.")
                    st.info("Possible causes: encryption/decryption mismatch...")
                else:
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
        SELECT party_id, email_id, transaction_date, COUNT(*) as count 
        FROM encrypted_data 
        GROUP BY party_id 
        ORDER BY count DESC 
        LIMIT 10
    """)
    party_dist = cursor.fetchall()

    if party_dist:
        st.subheader("Top 10 Parties by Transaction Count")
        df_parties = pd.DataFrame(party_dist, columns=['Party ID', 'Email ID', 'Transaction Date', 'Transaction Count'])
        df_parties['Party ID'] = df_parties['Party ID'].str[:50]
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