"""
Enhanced FHE Financial Transaction Client Application
Handles multiple datasets with user-controlled encryption and analysis
Python 3.11+
Required: streamlit, pandas, numpy, httpx, plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import httpx
import json
import base64
import time
from datetime import date, datetime, timedelta
from io import StringIO, BytesIO
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

# ==================== Configuration ====================

st.set_page_config(
    page_title="FHE Financial Transaction Client",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'server_url' not in st.session_state:
    st.session_state.server_url = "http://localhost:8000"

# Server configuration in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Server Configuration")
    SERVER_URL = st.text_input(
        "Server URL",
        value=st.session_state.server_url,
        key="server_url_input",
        help="URL of the FHE server"
    )
    if SERVER_URL != st.session_state.server_url:
        st.session_state.server_url = SERVER_URL
    SERVER_URL = st.session_state.server_url

# Constants
RESTRICTED_COUNTRIES = ['CN', 'TR', 'SA', 'KP', 'IR', 'RU', 'China', 'Turkey', 'Saudi Arabia']

SCHEME_CONFIGS = {
    'CKKS': {
        'name': 'CKKS',
        'description': 'Approximate arithmetic on encrypted real numbers',
        'supports': {
            'numeric': True,
            'text': 'limited',
            'date': True
        },
        'warnings': ['Approximate precision', 'No exact integer comparison']
    },
    'BFV': {
        'name': 'BFV',
        'description': 'Exact integer arithmetic',
        'supports': {
            'numeric': True,
            'text': False,
            'date': 'limited'
        },
        'warnings': ['Integer only', 'No floating point', 'Text not supported']
    },
    'BGV': {
        'name': 'BGV',
        'description': 'Exact integer arithmetic with rotation support',
        'supports': {
            'numeric': True,
            'text': 'limited',
            'date': 'limited'
        },
        'warnings': ['Integer only', 'Requires parameter tuning']
    }
}


# ==================== Helper Functions ====================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'df_users': None,
        'df_accounts': None,
        'df_transactions': None,
        'keys_info': None,
        'selected_columns': {'users': [], 'accounts': [], 'transactions': []},
        'encrypted_data': {},
        'analysis_results': None,
        'statistics': [],
        'encryption_times': {},
        'restricted_data_detected': False,
        'encryption_ready': False,
        'analysis_ready': False,
        'uploaded_file_info': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_server_health():
    """Check if server is healthy"""
    try:
        response = httpx.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)


def detect_data_type(series):
    """Detect data type of a pandas series"""
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    else:
        return 'text'


def check_scheme_compatibility(scheme, data_type):
    """Check if scheme supports data type"""
    config = SCHEME_CONFIGS.get(scheme, {})
    supports = config.get('supports', {})

    if data_type == 'text' and supports.get('text') == False:
        return False, f"{scheme} does not support text data"
    elif data_type == 'text' and supports.get('text') == 'limited':
        return True, f"{scheme} has limited text support (will encode as numeric)"

    return True, None


# ==================== Data Generation ====================

def generate_synthetic_data(num_users=100, num_accounts=150, num_transactions=1000):
    """Generate synthetic financial data"""
    import random

    countries = ['USA', 'UK', 'Germany', 'France', 'Japan', 'Canada', 'Australia',
                 'China', 'Turkey', 'Saudi Arabia', 'India', 'Brazil', 'Mexico']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY', 'TRY', 'SAR', 'INR']
    account_types = ['Checking', 'Savings', 'Credit', 'Investment']
    transaction_types = ['Purchase', 'Transfer', 'Withdrawal', 'Deposit', 'Payment']

    # Generate users
    users = []
    for i in range(num_users):
        country = random.choice(countries)
        users.append({
            'user_id': f'U{str(i + 1).zfill(5)}',
            'name': f'User {i + 1}',
            'email': f'user{i + 1}@example.com',
            'address': f'{random.randint(1, 9999)} Street {i + 1}',
            'city': f'City {random.randint(1, 100)}',
            'country': country,
            'is_restricted': country in RESTRICTED_COUNTRIES,
            'age': random.randint(18, 80),
            'phone': f'+1-555-{random.randint(1000, 9999)}'
        })

    df_users = pd.DataFrame(users)

    # Generate accounts
    accounts = []
    for i in range(num_accounts):
        user = random.choice(users)
        accounts.append({
            'account_id': f'A{str(i + 1).zfill(6)}',
            'user_id': user['user_id'],
            'account_number': str(random.randint(1000000000, 9999999999)),
            'account_type': random.choice(account_types),
            'balance': round(random.uniform(100, 100000), 2),
            'currency': random.choice(currencies),
            'created_date': (date.today() - timedelta(days=random.randint(1, 1825))).isoformat(),
            'status': 'Active' if random.random() > 0.1 else 'Inactive'
        })

    df_accounts = pd.DataFrame(accounts)

    # Generate transactions
    transactions = []
    for i in range(num_transactions):
        account = random.choice(accounts)
        user = next(u for u in users if u['user_id'] == account['user_id'])

        transactions.append({
            'transaction_id': f'T{str(i + 1).zfill(7)}',
            'user_id': account['user_id'],
            'account_id': account['account_id'],
            'amount': round(random.uniform(10, 10000), 2),
            'currency': account['currency'],
            'transaction_type': random.choice(transaction_types),
            'transaction_date': (date.today() - timedelta(days=random.randint(1, 365))).isoformat(),
            'description': f'Transaction {i + 1}',
            'country': user['country'],
            'is_restricted': user['is_restricted'],
            'merchant': f'Merchant {random.randint(1, 500)}',
            'status': 'Completed' if random.random() > 0.05 else 'Pending'
        })

    df_transactions = pd.DataFrame(transactions)

    return df_users, df_accounts, df_transactions


def load_user_dataset(uploaded_file, dataset_type):
    """Load user-uploaded dataset"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return None

        # Add metadata
        st.session_state.uploaded_file_info[dataset_type] = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'rows': len(df),
            'columns': list(df.columns)
        }

        # Auto-detect restricted countries
        if 'country' in df.columns:
            df['is_restricted'] = df['country'].isin(RESTRICTED_COUNTRIES)

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# ==================== Tab 1: Data & Configuration ====================

def render_data_configuration_tab():
    st.header("📊 Data Management & Configuration")

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Generate Synthetic Data", "Upload Custom Dataset"],
            key="data_source"
        )

        if data_source == "Generate Synthetic Data":
            with st.expander("🎲 Generate Synthetic Data", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    num_users = st.number_input("Number of Users", min_value=10, max_value=1000, value=100)
                with col_b:
                    num_accounts = st.number_input("Number of Accounts", min_value=10, max_value=2000, value=150)
                with col_c:
                    num_transactions = st.number_input("Number of Transactions", min_value=50, max_value=10000,
                                                       value=1000)

                if st.button("🎲 Generate Synthetic Data", type="primary"):
                    with st.spinner("Generating synthetic data..."):
                        df_users, df_accounts, df_transactions = generate_synthetic_data(
                            num_users, num_accounts, num_transactions
                        )
                        st.session_state.df_users = df_users
                        st.session_state.df_accounts = df_accounts
                        st.session_state.df_transactions = df_transactions

                        restricted_count = df_users['is_restricted'].sum()
                        if restricted_count > 0:
                            st.session_state.restricted_data_detected = True
                            st.warning(f"⚠️ Detected {restricted_count} users from restricted countries")

                        st.success("✅ Synthetic data generated successfully!")

        else:  # Upload Custom Dataset
            with st.expander("📤 Upload Custom Dataset", expanded=True):
                st.info("Upload CSV or Excel files for parties (users), accounts, and payments (transactions)")

                tab_u, tab_a, tab_t = st.tabs(["👥 Parties (Users)", "💳 Accounts", "💰 Payments (Transactions)"])

                with tab_u:
                    uploaded_users = st.file_uploader(
                        "Upload Parties/Users Dataset",
                        type=['csv', 'xlsx', 'xls'],
                        key="upload_users"
                    )
                    if uploaded_users:
                        df = load_user_dataset(uploaded_users, 'users')
                        if df is not None:
                            st.session_state.df_users = df
                            st.success(f"✅ Loaded {len(df)} users from {uploaded_users.name}")
                            st.dataframe(df.head(), use_container_width=True)

                with tab_a:
                    uploaded_accounts = st.file_uploader(
                        "Upload Accounts Dataset",
                        type=['csv', 'xlsx', 'xls'],
                        key="upload_accounts"
                    )
                    if uploaded_accounts:
                        df = load_user_dataset(uploaded_accounts, 'accounts')
                        if df is not None:
                            st.session_state.df_accounts = df
                            st.success(f"✅ Loaded {len(df)} accounts from {uploaded_accounts.name}")
                            st.dataframe(df.head(), use_container_width=True)

                with tab_t:
                    uploaded_transactions = st.file_uploader(
                        "Upload Payments/Transactions Dataset",
                        type=['csv', 'xlsx', 'xls'],
                        key="upload_transactions"
                    )
                    if uploaded_transactions:
                        df = load_user_dataset(uploaded_transactions, 'transactions')
                        if df is not None:
                            st.session_state.df_transactions = df
                            st.success(f"✅ Loaded {len(df)} transactions from {uploaded_transactions.name}")
                            st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.subheader("FHE Configuration")

        # Library selection
        library = st.selectbox(
            "FHE Library",
            ["TenSEAL", "OpenFHE"],
            key="library",
            help="Select the FHE library to use"
        )

        # Scheme selection
        available_schemes = ["CKKS", "BFV", "BGV"] if library == "OpenFHE" else ["CKKS", "BFV"]
        scheme = st.selectbox(
            "FHE Scheme",
            available_schemes,
            key="scheme",
            help="Select the encryption scheme"
        )

        # Display scheme info
        scheme_info = SCHEME_CONFIGS.get(scheme, {})
        st.info(f"**{scheme}**: {scheme_info.get('description', '')}")

        # Warnings
        warnings = scheme_info.get('warnings', [])
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")

        # Advanced Parameters
        with st.expander("⚙️ Advanced Parameters"):
            poly_modulus_degree = st.selectbox(
                "Polynomial Modulus Degree",
                [4096, 8192, 16384, 32768],
                index=1,
                help="Higher values = more security but slower"
            )

            if scheme == "CKKS":
                scale = st.selectbox(
                    "Scale",
                    [2 ** 30, 2 ** 40, 2 ** 50],
                    index=1,
                    help="Precision parameter for CKKS"
                )

            if library == "OpenFHE":
                mult_depth = st.number_input(
                    "Multiplicative Depth",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Number of multiplications supported"
                )
                scale_mod_size = st.number_input(
                    "Scaling Modulus Size",
                    min_value=30,
                    max_value=60,
                    value=50
                )

        # Generate keys button
        if st.button("🔑 Generate Encryption Keys", type="primary"):
            with st.spinner("Generating encryption keys on server..."):
                params = {'poly_modulus_degree': poly_modulus_degree}

                if scheme == "CKKS":
                    params['scale'] = scale

                if library == "OpenFHE":
                    params['mult_depth'] = mult_depth
                    params['scale_mod_size'] = scale_mod_size

                try:
                    response = httpx.post(
                        f"{SERVER_URL}/generate_keys",
                        json={
                            'scheme': scheme,
                            'library': library,
                            'params': params
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    keys_info = response.json()
                    st.session_state.keys_info = keys_info
                    st.session_state.encryption_ready = True

                    st.success("✅ Keys generated successfully!")
                    st.json({
                        'session_id': keys_info.get('session_id'),
                        'library': keys_info.get('library'),
                        'scheme': keys_info.get('scheme')
                    })

                    # Download keys
                    keys_json = json.dumps(keys_info, indent=2)
                    st.download_button(
                        "📥 Download Keys",
                        keys_json,
                        file_name=f"fhe_keys_{scheme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"❌ Error generating keys: {str(e)}")
                    st.session_state.encryption_ready = False

    # Display loaded data tables
    st.markdown("---")
    st.subheader("📋 Loaded Data Tables")

    if st.session_state.df_users is not None:
        with st.expander(f"👥 Users Table ({len(st.session_state.df_users)} records)", expanded=False):
            st.dataframe(st.session_state.df_users, use_container_width=True)

            if 'is_restricted' in st.session_state.df_users.columns:
                restricted_count = st.session_state.df_users['is_restricted'].sum()
                if restricted_count > 0:
                    st.warning(f"⚠️ {restricted_count} users from restricted countries")

    if st.session_state.df_accounts is not None:
        with st.expander(f"💳 Accounts Table ({len(st.session_state.df_accounts)} records)", expanded=False):
            st.dataframe(st.session_state.df_accounts, use_container_width=True)

    if st.session_state.df_transactions is not None:
        with st.expander(f"💰 Transactions Table ({len(st.session_state.df_transactions)} records)", expanded=False):
            st.dataframe(st.session_state.df_transactions, use_container_width=True)


# ==================== Tab 2: Encryption & Analysis ====================

def render_encryption_analysis_tab():
    st.header("🔒 Data Encryption & Analysis")

    if not st.session_state.encryption_ready or st.session_state.keys_info is None:
        st.warning("⚠️ Please generate encryption keys in Tab 1 first")
        return

    if st.session_state.df_users is None:
        st.warning("⚠️ Please load or generate data in Tab 1 first")
        return

    # Section 1: Column Selection
    st.subheader("1️⃣ Select Columns for Encryption")
    st.info("💡 Select columns you want to encrypt. Click 'Encrypt Selected Columns' when ready.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Users/Parties Table**")
        if st.session_state.df_users is not None:
            user_columns = list(st.session_state.df_users.columns)
            selected_user_cols = st.multiselect(
                "Select columns:",
                user_columns,
                default=st.session_state.selected_columns.get('users', []),
                key="selected_user_cols"
            )
            st.session_state.selected_columns['users'] = selected_user_cols

    with col2:
        st.write("**Accounts Table**")
        if st.session_state.df_accounts is not None:
            account_columns = list(st.session_state.df_accounts.columns)
            selected_account_cols = st.multiselect(
                "Select columns:",
                account_columns,
                default=st.session_state.selected_columns.get('accounts', []),
                key="selected_account_cols"
            )
            st.session_state.selected_columns['accounts'] = selected_account_cols

    with col3:
        st.write("**Transactions/Payments Table**")
        if st.session_state.df_transactions is not None:
            transaction_columns = list(st.session_state.df_transactions.columns)
            selected_transaction_cols = st.multiselect(
                "Select columns:",
                transaction_columns,
                default=st.session_state.selected_columns.get('transactions', []),
                key="selected_transaction_cols"
            )
            st.session_state.selected_columns['transactions'] = selected_transaction_cols

    # Batch size
    batch_size = st.slider(
        "Encryption Batch Size",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of records to encrypt per API call"
    )

    # Encrypt button - user must click this
    if st.button("🔒 Encrypt Selected Columns", type="primary", use_container_width=True):
        encrypt_selected_columns(batch_size)

    st.markdown("---")

    # Section 2: Analysis
    st.subheader("2️⃣ Run FHE Analysis")

    if not st.session_state.encrypted_data:
        st.info("ℹ️ Encrypt data first to perform analysis")
        return

    st.success(f"✅ {len(st.session_state.encrypted_data)} columns encrypted and ready for analysis")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.write("**Analysis Parameters**")

        operation_type = st.selectbox(
            "Analysis Type",
            ["Transaction Analysis", "Transaction Count", "Account Summary", "Country Analysis"],
            help="Select the type of analysis to perform on encrypted data"
        )

        col_x, col_y = st.columns(2)
        with col_x:
            start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
        with col_y:
            end_date = st.date_input("End Date", date.today())

        user_id_filter = st.text_input("User ID (optional)", "")
        country_filter = st.text_input("Country (optional)", "")

    with col_b:
        st.write("**Jurisdiction Info**")

        is_restricted = False
        if country_filter and country_filter in RESTRICTED_COUNTRIES:
            is_restricted = True
            st.error("🚨 RESTRICTED COUNTRY")
            st.warning("Data processed on-premises only")
        elif st.session_state.restricted_data_detected:
            st.warning("⚠️ Dataset has restricted data")
            is_restricted = st.checkbox("Process as restricted", value=False)
        else:
            st.success("✅ Non-restricted")

    # Run Analysis button - user must click
    if st.button("▶️ Run FHE Analysis", type="primary", use_container_width=True):
        run_fhe_analysis(operation_type, start_date, end_date, user_id_filter, country_filter, is_restricted)


def encrypt_selected_columns(batch_size):
    """Encrypt selected columns with user interaction"""
    keys_info = st.session_state.keys_info
    session_id = keys_info.get('session_id')
    library = keys_info.get('library')
    scheme = keys_info.get('scheme')

    total_columns = sum(len(cols) for cols in st.session_state.selected_columns.values())

    if total_columns == 0:
        st.warning("⚠️ No columns selected for encryption")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    current_column = 0

    try:
        client = httpx.Client(timeout=120)

        for table_name, df_key in [('users', 'df_users'), ('accounts', 'df_accounts'),
                                   ('transactions', 'df_transactions')]:
            df = st.session_state.get(df_key)
            if df is None:
                continue

            selected_cols = st.session_state.selected_columns[table_name]

            for column in selected_cols:
                current_column += 1
                progress = current_column / total_columns
                progress_bar.progress(progress)
                status_text.text(f"Encrypting {table_name}.{column}... ({current_column}/{total_columns})")

                column_data = df[column].tolist()
                data_type = detect_data_type(df[column])

                compatible, warning = check_scheme_compatibility(scheme, data_type)
                if not compatible:
                    st.warning(f"⚠️ Skipping {column}: {warning}")
                    continue

                start_time = time.time()
                encrypted_values = []

                for i in range(0, len(column_data), batch_size):
                    batch_data = column_data[i:i + batch_size]

                    response = client.post(
                        f"{SERVER_URL}/encrypt",
                        json={
                            'data': batch_data,
                            'column_name': column,
                            'data_type': data_type,
                            'keys': {'session_id': session_id},
                            'scheme': scheme,
                            'library': library,
                            'batch_id': i // batch_size
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get('success'):
                        encrypted_values.extend(result.get('encrypted_values', []))

                encryption_time = time.time() - start_time

                key = f"{table_name}:{column}"
                st.session_state.encrypted_data[key] = {
                    'encrypted_values': encrypted_values,
                    'data_type': data_type,
                    'library': library,
                    'scheme': scheme,
                    'original_count': len(column_data),
                    'encrypted_count': len(encrypted_values),
                    'encryption_time': encryption_time,
                    'table': table_name,
                    'column': column
                }

                st.session_state.statistics.append({
                    'table': table_name,
                    'column': column,
                    'data_type': data_type,
                    'library': library,
                    'scheme': scheme,
                    'original_count': len(column_data),
                    'encrypted_count': len(encrypted_values),
                    'encryption_time': encryption_time,
                    'rate': len(column_data) / encryption_time,
                    'timestamp': datetime.now().isoformat()
                })

        client.close()
        progress_bar.progress(1.0)
        status_text.text("✅ Encryption complete!")

        st.success(f"✅ Successfully encrypted {total_columns} columns")
        st.session_state.analysis_ready = True

    except Exception as e:
        st.error(f"❌ Encryption error: {str(e)}")


def run_fhe_analysis(operation_type, start_date, end_date, user_id_filter, country_filter, is_restricted):
    """Run FHE analysis with user confirmation"""
    keys_info = st.session_state.keys_info
    session_id = keys_info.get('session_id')
    library = keys_info.get('library')
    scheme = keys_info.get('scheme')

    with st.spinner("Running FHE analysis on encrypted data..."):
        try:
            encrypted_data_payload = {}
            for key, value in st.session_state.encrypted_data.items():
                encrypted_data_payload[key] = value['encrypted_values']

            query_params = {
                'operation_type': operation_type,
                'user_id': user_id_filter if user_id_filter else None,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'country': country_filter if country_filter else None,
                'is_restricted': is_restricted
            }

            response = httpx.post(
                f"{SERVER_URL}/fhe_query",
                json={
                    'encrypted_data': encrypted_data_payload,
                    'query_params': query_params,
                    'keys': {'session_id': session_id},
                    'library': library,
                    'scheme': scheme
                },
                timeout=60
            )
            response.raise_for_status()
            results = response.json()

            st.session_state.analysis_results = results

            st.success("✅ FHE Analysis complete!")

            if is_restricted:
                st.error("🚨 RESTRICTED - Processed on-premises")

        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")


# ==================== Tab 3: Results & Statistics ====================

def render_results_statistics_tab():
    st.header("📈 Results & Statistics")

    if st.session_state.analysis_results:
        st.subheader("🔍 Analysis Results")
        results = st.session_state.analysis_results

        if results.get('is_restricted'):
            st.error("🚨 RESTRICTED COUNTRY DATA")
            st.warning(f"**Location:** {results.get('processing_location')}")

        st.json(results)

    if st.session_state.statistics:
        st.subheader("📊 Encryption Statistics")
        df_stats = pd.DataFrame(st.session_state.statistics)
        st.dataframe(df_stats, use_container_width=True)


# ==================== Main Application ====================

def main():
    init_session_state()

    st.title("🔐 FHE Financial Transaction Analysis System")
    st.markdown("### Secure homomorphic encryption for financial data processing")

    # Check server
    is_healthy, health_info = check_server_health()
    if not is_healthy:
        st.error(f"⚠️ Cannot connect to server at {SERVER_URL}")
        st.info("Start server with: `python server.py`")
        return
    else:
        st.success(f"✅ Connected to server at {SERVER_URL}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 1. Data & Configuration",
        "🔒 2. Encryption & Analysis",
        "📈 3. Results & Statistics"
    ])

    with tab1:
        render_data_configuration_tab()

    with tab2:
        render_encryption_analysis_tab()

    with tab3:
        render_results_statistics_tab()


if __name__ == "__main__":
    main()