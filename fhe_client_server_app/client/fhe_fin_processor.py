"""
FHE Financial Data Processor - Enhanced Client Application
Complete implementation with key management, rotation, and CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import hashlib
from typing import Optional, Dict, List, Tuple
import os

# Configuration
SERVER_URL = os.environ.get('SERVER_URL', "http://localhost:8000")
RESTRICTED_COUNTRIES = ['CN', 'RU', 'KP', 'IR', 'SY']


class FHEClient:
    """Enhanced client for FHE operations with CSV export"""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = requests.Session()

    def check_server_health(self) -> bool:
        """Check if server is available"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_keys(self, scheme: str, library: str, params: Dict) -> Dict:
        """Generate FHE keys on server"""
        try:
            response = self.session.post(
                f"{self.server_url}/generate_keys",
                json={"scheme": scheme, "library": library, "params": params},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error generating keys: {str(e)}")
            return {}

    def rotate_keys(self, old_session_id: str, scheme: str, library: str, params: Dict) -> Dict:
        """Rotate keys with backward compatibility"""
        try:
            response = self.session.post(
                f"{self.server_url}/rotate_keys",
                json={
                    "old_session_id": old_session_id,
                    "scheme": scheme,
                    "library": library,
                    "params": params
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error rotating keys: {str(e)}")
            return {}

    def export_encrypted_csv(self, table_name: str, data: pd.DataFrame,
                            encrypted_columns: Dict) -> str:
        """Export data with encrypted columns as CSV"""
        df_export = data.copy()

        for col_name, enc_data in encrypted_columns.items():
            if col_name in df_export.columns:
                # Replace with encrypted representation
                enc_strings = [
                    json.dumps(enc) if enc else None
                    for enc in enc_data.get('encrypted', [])
                ]
                df_export[f"{col_name}_encrypted"] = enc_strings
                # Keep original for reference
                df_export[f"{col_name}_original"] = df_export[col_name]

        return df_export.to_csv(index=False)

    def encrypt_data_batch(self, data: List, column_name: str, data_type: str,
                          keys: Dict, scheme: str, library: str) -> Dict:
        """Encrypt data on server"""
        try:
            response = self.session.post(
                f"{self.server_url}/encrypt",
                json={
                    "data": data,
                    "column_name": column_name,
                    "data_type": data_type,
                    "keys": keys,
                    "scheme": scheme,
                    "library": library
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error encrypting data: {str(e)}")
            return {}

    def perform_fhe_query(self, query_request: Dict) -> Dict:
        """Perform FHE query on server"""
        try:
            response = self.session.post(
                f"{self.server_url}/fhe_query",
                json=query_request,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error performing FHE query: {str(e)}")
            return {}


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'user_data': None,
        'account_data': None,
        'transaction_data': None,
        'encrypted_data': {},
        'keys': {},
        'key_history': [],
        'fhe_results': None,
        'statistics': [],
        'restricted_data_flags': {},
        'selected_columns': {'user': [], 'account': [], 'transaction': []},
        'fhe_library': 'TenSEAL',
        'current_scheme': 'CKKS',
        'client': FHEClient(SERVER_URL),
        'export_mode': 'api'  # 'api' or 'csv'
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_synthetic_data(num_users: int = 100, accounts_per_user: int = 2,
                           transactions_per_account: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic financial data with country information"""
    import random

    countries = ['US', 'UK', 'DE', 'FR', 'JP', 'CA', 'AU', 'IN', 'BR', 'CN', 'RU', 'KP', 'IR', 'SY']

    # User data
    users = []
    for i in range(1, num_users + 1):
        users.append({
            'user_id': f'USR{i:05d}',
            'name': f'User {i}',
            'email': f'user{i}@example.com',
            'country': random.choice(countries),
            'address': f'{random.randint(100, 999)} Main St, City {i % 10}',
            'phone': f'+1{random.randint(1000000000, 9999999999)}',
            'age': random.randint(18, 80),
            'registration_date': (datetime.now() - timedelta(days=random.randint(0, 3650))).strftime('%Y-%m-%d')
        })
    user_data = pd.DataFrame(users)

    # Account data
    accounts = []
    account_types = ['Savings', 'Checking', 'Credit', 'Investment']
    account_id = 1
    for user_id in user_data['user_id']:
        num_accounts = random.randint(1, accounts_per_user)
        for _ in range(num_accounts):
            accounts.append({
                'account_id': f'ACC{account_id:07d}',
                'user_id': user_id,
                'account_type': random.choice(account_types),
                'account_number': f'{random.randint(1000000000, 9999999999)}',
                'balance': round(random.uniform(100, 50000), 2),
                'currency': random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CNY']),
                'opening_date': (datetime.now() - timedelta(days=random.randint(0, 2000))).strftime('%Y-%m-%d'),
                'status': random.choice(['Active', 'Active', 'Active', 'Frozen'])
            })
            account_id += 1
    account_data = pd.DataFrame(accounts)

    # Transaction data
    transactions = []
    transaction_types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Purchase']
    transaction_id = 1
    for _, account in account_data.iterrows():
        num_transactions = random.randint(10, transactions_per_account)
        for _ in range(num_transactions):
            transactions.append({
                'transaction_id': f'TXN{transaction_id:010d}',
                'user_id': account['user_id'],
                'account_id': account['account_id'],
                'transaction_type': random.choice(transaction_types),
                'amount': round(random.uniform(10, 5000), 2),
                'currency': account['currency'],
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'time': f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}',
                'description': f'Transaction {transaction_id}',
                'status': random.choice(['Completed', 'Completed', 'Completed', 'Pending', 'Failed'])
            })
            transaction_id += 1
    transaction_data = pd.DataFrame(transactions)

    return user_data, account_data, transaction_data


def identify_restricted_data(user_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify data from restricted countries"""
    if user_data is None or 'country' not in user_data.columns:
        return {}

    restricted_users = user_data[user_data['country'].isin(RESTRICTED_COUNTRIES)]

    return {
        'restricted_users': restricted_users['user_id'].tolist(),
        'countries': restricted_users['country'].unique().tolist(),
        'count': len(restricted_users),
        'details': restricted_users[['user_id', 'name', 'country']].to_dict('records')
    }


def page_data_management():
    """Page 1: Data Upload, Key Management, and Encryption"""
    st.title("📊 Data Management & Encryption")
    st.markdown("---")

    # Server connection check
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.session_state.client.check_server_health():
            st.success("🟢 Server Online")
        else:
            st.error("🔴 Server Offline")
            st.stop()

    # Data source selection
    st.subheader("1️⃣ Data Source")
    data_source = st.radio("Select Data Source",
                          ["Generate Synthetic Data", "Upload CSV Files", "Upload Denormalized CSV"],
                          horizontal=True)

    if data_source == "Generate Synthetic Data":
        handle_synthetic_data()
    elif data_source == "Upload CSV Files":
        handle_csv_upload()
    else:
        handle_denormalized_upload()

    # Display loaded data
    if st.session_state.user_data is not None:
        st.markdown("---")
        display_data_preview()

        st.markdown("---")
        st.subheader("2️⃣ Column Selection for Encryption")
        handle_column_selection()

        st.markdown("---")
        st.subheader("3️⃣ Key Management")
        handle_key_management()

        if st.session_state.keys:
            st.markdown("---")
            st.subheader("4️⃣ Data Encryption")
            handle_encryption()


def handle_synthetic_data():
    """Generate synthetic financial data"""
    col1, col2, col3 = st.columns(3)
    with col1:
        num_users = st.number_input("Number of Users", 10, 1000, 100)
    with col2:
        accounts_per_user = st.number_input("Accounts per User", 1, 5, 2)
    with col3:
        transactions_per_account = st.number_input("Transactions per Account", 10, 200, 50)

    if st.button("🎲 Generate Data", type="primary"):
        with st.spinner("Generating financial data..."):
            user_data, account_data, transaction_data = generate_synthetic_data(
                num_users, accounts_per_user, transactions_per_account
            )
            st.session_state.user_data = user_data
            st.session_state.account_data = account_data
            st.session_state.transaction_data = transaction_data

            # Identify restricted data
            st.session_state.restricted_data_flags = identify_restricted_data(user_data)

            st.success(f"✅ Generated {len(user_data)} users, {len(account_data)} accounts, {len(transaction_data)} transactions!")
            if st.session_state.restricted_data_flags.get('count', 0) > 0:
                st.warning(f"⚠️ Found {st.session_state.restricted_data_flags['count']} users from restricted countries!")
            st.rerun()


def handle_csv_upload():
    """Handle CSV file uploads"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**👤 User Data**")
        user_file = st.file_uploader("Upload User CSV", type=['csv'], key='user_csv')
        if user_file:
            st.session_state.user_data = pd.read_csv(user_file)
            st.session_state.restricted_data_flags = identify_restricted_data(st.session_state.user_data)
            st.success(f"✅ {len(st.session_state.user_data)} users loaded")

    with col2:
        st.write("**🏦 Account Data**")
        account_file = st.file_uploader("Upload Account CSV", type=['csv'], key='account_csv')
        if account_file:
            st.session_state.account_data = pd.read_csv(account_file)
            st.success(f"✅ {len(st.session_state.account_data)} accounts loaded")

    with col3:
        st.write("**💳 Transaction Data**")
        transaction_file = st.file_uploader("Upload Transaction CSV", type=['csv'], key='transaction_csv')
        if transaction_file:
            st.session_state.transaction_data = pd.read_csv(transaction_file)
            st.success(f"✅ {len(st.session_state.transaction_data)} transactions loaded")


def handle_denormalized_upload():
    """Handle denormalized CSV upload"""
    st.write("**Upload Denormalized CSV**")
    st.info("Upload a single CSV containing all user, account, and transaction data")

    denorm_file = st.file_uploader("Upload Denormalized CSV", type=['csv'], key='denorm_csv')

    if denorm_file:
        full_data = pd.read_csv(denorm_file)
        st.success(f"✅ Loaded {len(full_data)} rows")

        # Show preview
        st.dataframe(full_data.head(10))

        # Try to split into normalized tables
        if st.button("Normalize Data"):
            with st.spinner("Normalizing data..."):
                # Extract user data
                user_cols = [col for col in full_data.columns if any(kw in col.lower() for kw in ['user', 'name', 'email', 'country', 'age', 'address', 'phone'])]
                if 'user_id' in full_data.columns:
                    st.session_state.user_data = full_data[user_cols].drop_duplicates('user_id').reset_index(drop=True)

                # Extract account data
                account_cols = [col for col in full_data.columns if any(kw in col.lower() for kw in ['account', 'balance'])]
                if 'account_id' in full_data.columns:
                    st.session_state.account_data = full_data[account_cols].drop_duplicates('account_id').reset_index(drop=True)

                # Extract transaction data
                trans_cols = [col for col in full_data.columns if any(kw in col.lower() for kw in ['transaction', 'amount', 'currency'])]
                if 'transaction_id' in full_data.columns:
                    st.session_state.transaction_data = full_data[trans_cols].drop_duplicates('transaction_id').reset_index(drop=True)

                st.session_state.restricted_data_flags = identify_restricted_data(st.session_state.user_data)
                st.success("✅ Data normalized successfully!")
                st.rerun()


def display_data_preview():
    """Display loaded data tables"""
    st.subheader("📋 Data Preview")

    tab1, tab2, tab3 = st.tabs(["👤 Users", "🏦 Accounts", "💳 Transactions"])

    with tab1:
        if st.session_state.user_data is not None:
            st.dataframe(st.session_state.user_data.head(20), use_container_width=True)
            st.caption(f"Total Users: {len(st.session_state.user_data)}")

            if st.session_state.restricted_data_flags.get('count', 0) > 0:
                with st.expander("⚠️ Restricted Country Users"):
                    restricted_df = pd.DataFrame(st.session_state.restricted_data_flags['details'])
                    st.dataframe(restricted_df, use_container_width=True)

    with tab2:
        if st.session_state.account_data is not None:
            st.dataframe(st.session_state.account_data.head(20), use_container_width=True)
            st.caption(f"Total Accounts: {len(st.session_state.account_data)}")

    with tab3:
        if st.session_state.transaction_data is not None:
            st.dataframe(st.session_state.transaction_data.head(20), use_container_width=True)
            st.caption(f"Total Transactions: {len(st.session_state.transaction_data)}")


def handle_column_selection():
    """Handle column selection for encryption"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**User Columns**")
        if st.session_state.user_data is not None:
            user_cols = st.multiselect(
                "Select columns to encrypt",
                st.session_state.user_data.columns.tolist(),
                key='user_cols_select'
            )
            st.session_state.selected_columns['user'] = user_cols
            if user_cols:
                st.info(f"✓ {len(user_cols)} columns selected")

    with col2:
        st.write("**Account Columns**")
        if st.session_state.account_data is not None:
            account_cols = st.multiselect(
                "Select columns to encrypt",
                st.session_state.account_data.columns.tolist(),
                key='account_cols_select'
            )
            st.session_state.selected_columns['account'] = account_cols
            if account_cols:
                st.info(f"✓ {len(account_cols)} columns selected")

    with col3:
        st.write("**Transaction Columns**")
        if st.session_state.transaction_data is not None:
            transaction_cols = st.multiselect(
                "Select columns to encrypt",
                st.session_state.transaction_data.columns.tolist(),
                key='transaction_cols_select'
            )
            st.session_state.selected_columns['transaction'] = transaction_cols
            if transaction_cols:
                st.info(f"✓ {len(transaction_cols)} columns selected")


def handle_key_management():
    """Enhanced key management with rotation"""

    tab1, tab2, tab3 = st.tabs(["🔑 Generate Keys", "🔄 Rotate Keys", "📥 Import Keys"])

    with tab1:
        generate_keys_ui()

    with tab2:
        rotate_keys_ui()

    with tab3:
        import_keys_ui()


def generate_keys_ui():
    """UI for key generation with multiple options"""
    st.write("**Configure Key Generation Parameters**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        library = st.selectbox("Library", ["TenSEAL", "OpenFHE"], key='keygen_library')
        st.session_state.fhe_library = library

    with col2:
        if library == "OpenFHE":
            scheme = st.selectbox("Scheme", ["CKKS", "BFV", "BGV"], key='keygen_scheme')
        else:
            scheme = st.selectbox("Scheme", ["CKKS", "BFV"], key='keygen_scheme')
        st.session_state.current_scheme = scheme

    with col3:
        poly_mod = st.selectbox(
            "Polynomial Modulus",
            [4096, 8192, 16384, 32768],
            index=1,
            help="Higher values = more security but slower"
        )

    with col4:
        if scheme == "CKKS":
            scale_power = st.selectbox("Scale (2^x)", [30, 40, 50], index=1)
        else:
            plain_mod = st.number_input("Plain Modulus", value=1032193)

    # Show scheme information
    scheme_info = {
        'CKKS': {
            'description': 'Approximate arithmetic on real/complex numbers',
            'best_for': 'Machine learning, signal processing, analytics',
            'supports': 'Floating point operations',
            'limitations': 'Approximate results'
        },
        'BFV': {
            'description': 'Exact integer arithmetic',
            'best_for': 'Database operations, exact computations',
            'supports': 'Integer operations only',
            'limitations': 'No floating point'
        },
        'BGV': {
            'description': 'General purpose with modulus switching',
            'best_for': 'Leveled computations, general purpose',
            'supports': 'Integer operations, rotations',
            'limitations': 'Requires careful parameter selection'
        }
    }

    if scheme in scheme_info:
        with st.expander(f"ℹ️ About {scheme} Scheme"):
            info = scheme_info[scheme]
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Best For:** {info['best_for']}")
            st.write(f"**Supports:** {info['supports']}")
            st.write(f"**Limitations:** {info['limitations']}")

    if library == "OpenFHE":
        st.info("""
        **OpenFHE Mode Selection:**
        - 🔵 **ctypes**: Direct DLL (fastest, requires OpenFHE installed)
        - 🟢 **custom_dll**: Wrapper DLL (Windows, requires compiler)
        - 🟡 **subprocess**: C++ executable (cross-platform, requires compiler)
        - 🟠 **simulation**: Pure Python (fallback, not production-ready)
        """)

    if st.button("🔑 Generate Keys", type="primary"):
        with st.spinner(f"Generating keys using {library} - {scheme}..."):
            params = {'poly_modulus_degree': poly_mod}

            if scheme == "CKKS":
                params['scale'] = 2 ** scale_power
                if library == "TenSEAL":
                    params['coeff_mod_bit_sizes'] = [60, 40, 40, 60]
                else:
                    params['scale_mod_size'] = scale_power
            else:
                params['plain_modulus'] = plain_mod if scheme != "CKKS" else None

            start_time = time.time()
            keys = st.session_state.client.generate_keys(scheme, library, params)
            key_gen_time = time.time() - start_time

            if keys:
                st.session_state.keys = keys
                st.session_state.keys['scheme'] = scheme
                st.session_state.keys['library'] = library
                st.session_state.keys['generated_at'] = datetime.now().isoformat()

                # Add to history
                st.session_state.key_history.append({
                    'session_id': keys.get('session_id'),
                    'timestamp': datetime.now().isoformat(),
                    'scheme': scheme,
                    'library': library,
                    'action': 'generated'
                })

                st.success(f"✅ Keys generated in {key_gen_time:.2f}s!")

                if library == "OpenFHE" and 'mode' in keys:
                    mode_icons = {
                        'ctypes': '🔵',
                        'custom_dll': '🟢',
                        'subprocess': '🟡',
                        'simulation': '🟠'
                    }
                    st.info(f"{mode_icons.get(keys['mode'], '⚪')} OpenFHE Mode: **{keys['mode'].upper()}**")

                display_keys(keys)

                # Statistics
                st.session_state.statistics.append({
                    'operation': 'key_generation',
                    'library': library,
                    'scheme': scheme,
                    'time': key_gen_time,
                    'mode': keys.get('mode'),
                    'timestamp': datetime.now().isoformat()
                })


def display_keys(keys: Dict):
    """Display generated keys with download options"""
    with st.expander("📋 View Generated Keys", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["Public Key", "Private Key", "Evaluation Key", "Session Info"])

        with tab1:
            st.text_area("Public Key (truncated)", keys.get('public_key', '')[:200] + '...', height=100)
            if 'full_public_key' in keys:
                st.download_button(
                    "📥 Download Public Key",
                    keys.get('full_public_key', ''),
                    f"public_key_{keys.get('session_id', 'unknown')}.txt",
                    mime="text/plain"
                )

        with tab2:
            st.text_area("Private Key (truncated)", keys.get('private_key', '')[:200] + '...', height=100)
            st.warning("⚠️ Keep this key secure! Never share it publicly.")
            if 'full_private_key' in keys:
                st.download_button(
                    "📥 Download Private Key",
                    keys.get('full_private_key', ''),
                    f"private_key_{keys.get('session_id', 'unknown')}.txt",
                    mime="text/plain"
                )

        with tab3:
            eval_key = keys.get('full_evaluation_key', keys.get('evaluation_key', 'N/A'))
            st.text_area("Evaluation Key (truncated)", str(eval_key)[:200] + '...', height=100)
            st.info("Evaluation keys are used for homomorphic operations")
            if eval_key != 'N/A':
                st.download_button(
                    "📥 Download Evaluation Key",
                    str(eval_key),
                    f"evaluation_key_{keys.get('session_id', 'unknown')}.txt",
                    mime="text/plain"
                )

        with tab4:
            st.json({
                'session_id': keys.get('session_id'),
                'library': keys.get('library'),
                'scheme': keys.get('scheme'),
                'mode': keys.get('mode', 'N/A'),
                'platform': keys.get('platform', 'unknown'),
                'generated_at': keys.get('generated_at', datetime.now().isoformat())
            })


def rotate_keys_ui():
    """UI for key rotation"""
    if not st.session_state.keys:
        st.warning("⚠️ Generate keys first before rotating")
        return

    st.write("**Rotate Keys (with backward compatibility)**")

    current_keys = st.session_state.keys

    # Safely get values with fallback
    session_id = current_keys.get('session_id', 'N/A') if isinstance(current_keys, dict) else 'N/A'
    library = current_keys.get('library', 'N/A') if isinstance(current_keys, dict) else 'N/A'
    scheme = current_keys.get('scheme', 'N/A') if isinstance(current_keys, dict) else 'N/A'
    generated_at = current_keys.get('generated_at', 'N/A') if isinstance(current_keys, dict) else 'N/A'

    st.info(f"""
    **Current Keys:**
    - Session ID: {session_id}
    - Library: {library}
    - Scheme: {scheme}
    - Generated: {generated_at}
    """)

    st.write("Generate new keys while maintaining access to data encrypted with old keys")

    if st.button("🔄 Rotate Keys"):
        with st.spinner("Rotating keys..."):
            old_session_id = current_keys.get('session_id')
            library = current_keys.get('library')
            scheme = current_keys.get('scheme')

            # Get params from current keys
            params = st.session_state.keys.get('params', {
                'poly_modulus_degree': 8192,
                'scale': 2**40
            })

            start_time = time.time()
            new_keys = st.session_state.client.rotate_keys(old_session_id, scheme, library, params)
            rotation_time = time.time() - start_time

            if new_keys:
                # Keep old keys in history
                st.session_state.key_history.append({
                    'session_id': old_session_id,
                    'timestamp': current_keys.get('generated_at'),
                    'scheme': scheme,
                    'library': library,
                    'action': 'rotated_out',
                    'new_session_id': new_keys.get('session_id')
                })

                # Update current keys
                st.session_state.keys = new_keys
                st.session_state.keys['generated_at'] = datetime.now().isoformat()

                # Add new keys to history
                st.session_state.key_history.append({
                    'session_id': new_keys.get('session_id'),
                    'timestamp': datetime.now().isoformat(),
                    'scheme': scheme,
                    'library': library,
                    'action': 'rotated_in',
                    'old_session_id': old_session_id
                })

                st.success(f"✅ Keys rotated successfully in {rotation_time:.2f}s!")
                st.info("Old data remains accessible with backward compatibility")

                display_keys(new_keys)


def import_keys_ui():
    """UI for importing existing keys"""
    st.write("**Import Existing Keys**")

    col1, col2 = st.columns(2)

    with col1:
        public_key_file = st.file_uploader("Upload Public Key", type=['txt'], key='import_public')

    with col2:
        private_key_file = st.file_uploader("Upload Private Key", type=['txt'], key='import_private')

    session_id = st.text_input("Session ID (from key generation)")

    col1, col2 = st.columns(2)
    with col1:
        library = st.selectbox("Library", ["TenSEAL", "OpenFHE"], key='import_library')
    with col2:
        scheme = st.selectbox("Scheme", ["CKKS", "BFV", "BGV"], key='import_scheme')

    if st.button("📥 Import Keys") and public_key_file and private_key_file and session_id:
        st.session_state.keys = {
            'session_id': session_id,
            'public_key': public_key_file.read().decode()[:100] + '...',
            'private_key': private_key_file.read().decode()[:100] + '...',
            'full_public_key': public_key_file.read().decode(),
            'full_private_key': private_key_file.read().decode(),
            'library': library,
            'scheme': scheme,
            'imported': True,
            'generated_at': datetime.now().isoformat()
        }
        st.success("✅ Keys imported successfully!")
        st.info("Using imported keys for encryption operations")


def handle_encryption():
    """Handle data encryption with export options"""

    # Export mode selection
    col1, col2 = st.columns([3, 1])
    with col1:
        export_mode = st.radio(
            "Data Transfer Mode",
            ["REST API", "CSV Export"],
            horizontal=True,
            help="Choose how to send encrypted data to server"
        )
        st.session_state.export_mode = 'csv' if export_mode == "CSV Export" else 'api'

    if st.button("🔒 Encrypt Selected Data", type="primary", use_container_width=True):
        perform_encryption()


def perform_encryption():
    """Perform encryption on selected columns"""
    if not st.session_state.keys:
        st.error("❌ Please generate or import keys first!")
        return

    selected_any = any([
        st.session_state.selected_columns['user'],
        st.session_state.selected_columns['account'],
        st.session_state.selected_columns['transaction']
    ])

    if not selected_any:
        st.error("❌ Please select at least one column to encrypt!")
        return

    with st.spinner("Encrypting data..."):
        start_time = time.time()
        scheme = st.session_state.keys['scheme']
        library = st.session_state.keys['library']

        # Check scheme limitations
        scheme_limitations = get_scheme_limitations(scheme, library)

        # Separate restricted and allowed data
        restricted_users = st.session_state.restricted_data_flags.get('restricted_users', [])

        # Encrypt user data
        if st.session_state.selected_columns['user']:
            encrypt_table(
                st.session_state.user_data,
                st.session_state.selected_columns['user'],
                'user',
                scheme,
                library,
                scheme_limitations,
                restricted_users
            )

        # Encrypt account data
        if st.session_state.selected_columns['account']:
            encrypt_table(
                st.session_state.account_data,
                st.session_state.selected_columns['account'],
                'account',
                scheme,
                library,
                scheme_limitations,
                restricted_users
            )

        # Encrypt transaction data
        if st.session_state.selected_columns['transaction']:
            encrypt_table(
                st.session_state.transaction_data,
                st.session_state.selected_columns['transaction'],
                'transaction',
                scheme,
                library,
                scheme_limitations,
                restricted_users
            )

        encryption_time = time.time() - start_time

        # Store statistics
        st.session_state.statistics.append({
            'operation': 'encryption',
            'scheme': scheme,
            'library': library,
            'time': encryption_time,
            'columns': sum([len(v) for v in st.session_state.selected_columns.values()]),
            'mode': st.session_state.keys.get('mode'),
            'timestamp': datetime.now().isoformat()
        })

        st.success(f"✅ Data encrypted successfully in {encryption_time:.2f}s!")

        # Show export options if CSV mode
        if st.session_state.export_mode == 'csv':
            show_csv_export_options()

        # Show restricted data handling
        if restricted_users:
            st.info(f"ℹ️ Data from {len(restricted_users)} restricted country users will be processed on-premises")


def get_scheme_limitations(scheme: str, library: str) -> Dict:
    """Get scheme limitations"""
    limitations = {
        'CKKS': {
            'supports_text': False,
            'supports_numeric': True,
            'supports_date': True,
            'precision': 'Approximate',
            'operations': ['Add', 'Multiply', 'Subtract', 'Rotate']
        },
        'BFV': {
            'supports_text': False,
            'supports_numeric': True,
            'supports_date': True,
            'precision': 'Exact (integers only)',
            'operations': ['Add', 'Multiply', 'Subtract']
        },
        'BGV': {
            'supports_text': False,
            'supports_numeric': True,
            'supports_date': True,
            'precision': 'Exact (integers only)',
            'operations': ['Add', 'Multiply', 'Subtract', 'Rotate']
        }
    }
    return limitations.get(scheme, {})


def encrypt_table(data: pd.DataFrame, columns: List[str], table_name: str,
                  scheme: str, library: str, limitations: Dict, restricted_users: List[str]):
    """Encrypt selected columns of a table"""
    for col in columns:
        # Determine data type
        if data[col].dtype in ['int64', 'float64']:
            data_type = 'numeric'
        elif 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col]):
            data_type = 'date'
        else:
            data_type = 'text'

        # Check scheme limitations
        if data_type == 'text' and not limitations.get('supports_text'):
            st.warning(f"⚠️ Column '{col}' is text but {scheme} doesn't support text encryption. Encoding as numeric...")
            # Convert text to numeric encoding
            data_to_encrypt = [sum([ord(c) for c in str(val)]) if not pd.isna(val) else None for val in data[col]]
            data_type = 'numeric'
        else:
            data_to_encrypt = data[col].tolist()

        try:
            # Show progress
            progress_text = f"Encrypting {table_name}.{col}..."
            progress_bar = st.progress(0, text=progress_text)

            # Batch encryption
            batch_size = 100
            encrypted_results = []

            for i in range(0, len(data_to_encrypt), batch_size):
                batch = data_to_encrypt[i:i+batch_size]

                encrypted_result = st.session_state.client.encrypt_data_batch(
                    batch,
                    col,
                    data_type,
                    st.session_state.keys,
                    scheme,
                    library
                )

                if encrypted_result:
                    encrypted_results.extend(encrypted_result.get('encrypted_values', []))

                # Update progress
                progress = min((i + batch_size) / len(data_to_encrypt), 1.0)
                progress_bar.progress(progress, text=f"{progress_text} {int(progress*100)}%")

            progress_bar.empty()

            if table_name not in st.session_state.encrypted_data:
                st.session_state.encrypted_data[table_name] = {}

            st.session_state.encrypted_data[table_name][col] = {
                'encrypted': encrypted_results,
                'data_type': data_type,
                'length': len(data),
                'scheme': scheme,
                'library': library
            }

            st.success(f"✅ Encrypted {table_name}.{col} ({len(data)} rows)")

        except Exception as e:
            st.error(f"❌ Error encrypting {table_name}.{col}: {str(e)}")


def show_csv_export_options():
    """Show CSV export options for encrypted data"""
    st.subheader("📤 Export Encrypted Data as CSV")

    if not st.session_state.encrypted_data:
        st.warning("No encrypted data available to export")
        return

    for table_name, encrypted_cols in st.session_state.encrypted_data.items():
        with st.expander(f"Export {table_name.title()} Table"):
            # Get original data
            if table_name == 'user':
                original_data = st.session_state.user_data
            elif table_name == 'account':
                original_data = st.session_state.account_data
            else:
                original_data = st.session_state.transaction_data

            if original_data is not None:
                csv_data = st.session_state.client.export_encrypted_csv(
                    table_name,
                    original_data,
                    encrypted_cols
                )

                st.download_button(
                    f"📥 Download {table_name}_encrypted.csv",
                    csv_data,
                    f"{table_name}_encrypted.csv",
                    mime="text/csv",
                    key=f"download_{table_name}"
                )

                st.info(f"CSV contains {len(encrypted_cols)} encrypted columns")


def page_fhe_operations():
    """Page 2: FHE Operations and Query Execution"""
    st.title("🔒 FHE Operations on Encrypted Data")
    st.markdown("---")

    if not st.session_state.encrypted_data:
        st.warning("⚠️ No encrypted data available. Please go to Data Management page first.")
        return

    if not st.session_state.keys:
        st.error("❌ No encryption keys available. Please generate keys first.")
        return

    # Show current configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Library", st.session_state.keys.get('library'))
    with col2:
        st.metric("Scheme", st.session_state.keys.get('scheme'))
    with col3:
        if st.session_state.keys.get('library') == 'OpenFHE':
            st.metric("Mode", st.session_state.keys.get('mode', 'N/A'))

    st.markdown("---")

    # Query configuration
    st.subheader("📊 Query Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.user_data is not None:
            user_ids = st.session_state.user_data['user_id'].unique().tolist()
            selected_user = st.selectbox("Select User ID to Query", user_ids)
        else:
            st.error("User data not available")
            return

    with col2:
        operation_type = st.selectbox(
            "Operation Type",
            ["Transaction Count", "Transaction Analysis", "Account Summary"]
        )

    # Date range selection
    st.write("**Date Range Filter**")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Currency filter
    if st.session_state.transaction_data is not None and 'currency' in st.session_state.transaction_data.columns:
        currencies = st.session_state.transaction_data['currency'].unique().tolist()
        selected_currencies = st.multiselect(
            "Filter by Currency",
            currencies,
            default=currencies
        )
    else:
        selected_currencies = []

    st.markdown("---")

    # Check if user is from restricted country
    user_country = None
    is_restricted = False

    if st.session_state.user_data is not None:
        user_row = st.session_state.user_data[st.session_state.user_data['user_id'] == selected_user]
        if not user_row.empty and 'country' in user_row.columns:
            user_country = user_row.iloc[0]['country']
            is_restricted = user_country in RESTRICTED_COUNTRIES

            if is_restricted:
                st.error(f"⚠️ **RESTRICTED COUNTRY DATA**")
                st.warning(f"User from {user_country}. Processing will be done **ON-PREMISES ONLY**")
            else:
                st.success(f"✅ User from {user_country} (Allowed). Can process in cloud.")

    # Execute query button
    if st.button("🔍 Execute FHE Query", type="primary", use_container_width=True):
        execute_fhe_query(
            selected_user,
            start_date,
            end_date,
            selected_currencies,
            operation_type,
            user_country,
            is_restricted
        )

    # Display results
    if st.session_state.fhe_results:
        display_fhe_results()


def execute_fhe_query(user_id: str, start_date, end_date, currencies: List[str],
                      operation_type: str, user_country: Optional[str], is_restricted: bool):
    """Execute FHE query on encrypted data"""
    with st.spinner("Performing FHE operations on encrypted data..."):
        start_time = time.time()

        # Prepare query request
        query_request = {
            'encrypted_data': st.session_state.encrypted_data,
            'query_params': {
                'user_id': user_id,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'currencies': currencies,
                'operation_type': operation_type,
                'user_country': user_country,
                'is_restricted': is_restricted
            },
            'keys': st.session_state.keys,
            'library': st.session_state.keys.get('library')
        }

        # Execute query
        results = st.session_state.client.perform_fhe_query(query_request)

        operation_time = time.time() - start_time

        if results:
            st.session_state.fhe_results = {
                'results': results,
                'query_params': query_request['query_params'],
                'operation_time': operation_time,
                'is_restricted': is_restricted,
                'user_country': user_country,
                'timestamp': datetime.now().isoformat()
            }

            # Store statistics
            st.session_state.statistics.append({
                'operation': 'fhe_query',
                'scheme': st.session_state.keys.get('scheme'),
                'library': st.session_state.keys.get('library'),
                'time': operation_time,
                'user_id': user_id,
                'is_restricted': is_restricted,
                'operation_type': operation_type,
                'timestamp': datetime.now().isoformat()
            })

            st.success(f"✅ FHE operations completed in {operation_time:.2f}s!")

            if is_restricted:
                st.info("ℹ️ Results from restricted country - processed on-premises")


def display_fhe_results():
    """Display FHE operation results with decrypted view"""
    st.markdown("---")
    st.subheader("📊 Operation Results")

    results = st.session_state.fhe_results

    # Show restriction warning
    if results.get('is_restricted'):
        st.error(f"""
        ⚠️ **RESTRICTED COUNTRY DATA**  
        Country: **{results.get('user_country')}**  
        Processing Location: **ON-PREMISES ONLY**  
        Decryption: **Only in authorized jurisdiction (HSM/KMS)**
        """)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("User ID", results['query_params']['user_id'])
    with col2:
        st.metric("Operation Time", f"{results['operation_time']:.2f}s")
    with col3:
        days = (datetime.fromisoformat(results['query_params']['end_date']) -
                datetime.fromisoformat(results['query_params']['start_date'])).days
        st.metric("Date Range", f"{days} days")
    with col4:
        st.metric("Currencies", len(results['query_params']['currencies']))

    st.markdown("---")

    # Results display
    res_data = results['results']
    operation_type = results['query_params']['operation_type']

    if operation_type == "Transaction Count":
        display_transaction_count(res_data)
    elif operation_type == "Transaction Analysis":
        display_transaction_analysis(res_data)
    else:
        display_account_summary(res_data)


def display_transaction_count(results: Dict):
    """Display transaction count results"""
    st.subheader("Transaction Count Results")

    total_txns = results.get('total_transactions', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_txns)
    with col2:
        st.metric("Encrypted", "✅ Yes")
    with col3:
        st.metric("Processing", results.get('processing_location', 'cloud'))

    st.info("Results computed on encrypted data using homomorphic operations")


def display_transaction_analysis(results: Dict):
    """Display transaction analysis results"""
    st.subheader("Transaction Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Summary Statistics",
        "💰 Currency Analysis",
        "📅 Temporal Patterns",
        "📊 Distributions"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Amount", f"${results.get('total_amount', 0):,.2f}")
            st.metric("Average Amount", f"${results.get('avg_amount', 0):,.2f}")

        with col2:
            st.metric("Min Amount", f"${results.get('min_amount', 0):,.2f}")
            st.metric("Max Amount", f"${results.get('max_amount', 0):,.2f}")

        with col3:
            st.metric("Total Transactions", results.get('total_transactions', 0))
            st.metric("Encrypted", "✅ Yes")

    with tab2:
        st.write("**Transactions per Currency**")
        currency_data = results.get('currency_analysis', {})
        if currency_data:
            df = pd.DataFrame([
                {'Currency': k, 'Count': v['count'], 'Total Amount': v['total']}
                for k, v in currency_data.items()
            ])
            st.dataframe(df, use_container_width=True)

            # Bar chart
            fig = px.bar(df, x='Currency', y='Count',
                        title='Transaction Count by Currency',
                        color='Currency')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.write("**Monthly Transaction Pattern**")
        monthly_data = results.get('monthly_pattern', {})
        if monthly_data:
            df = pd.DataFrame([
                {'Month': k, 'Transactions': v}
                for k, v in monthly_data.items()
            ])

            fig = px.line(df, x='Month', y='Transactions',
                         title='Transaction Trends Over Time',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Transaction Type Distribution**")
            # Simulated data
            types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment']
            counts = [30, 25, 20, 25]
            fig = px.pie(values=counts, names=types,
                        title='Transaction Types')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Status Distribution**")
            statuses = ['Completed', 'Pending', 'Failed']
            status_counts = [85, 10, 5]
            fig = px.pie(values=status_counts, names=statuses,
                        title='Transaction Status')
            st.plotly_chart(fig, use_container_width=True)


def display_account_summary(results: Dict):
    """Display account summary results"""
    st.subheader("Account Summary Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Accounts", results.get('total_accounts', 0))

    with col2:
        st.metric("Active Accounts", results.get('active_accounts', 0))

    with col3:
        st.metric("Total Balance", f"${results.get('total_balance', 0):,.2f}")

    st.markdown("---")

    # Account types distribution
    st.write("**Account Types Distribution**")
    account_types = results.get('account_types', {})
    if account_types:
        df = pd.DataFrame([
            {'Account Type': k, 'Count': v}
            for k, v in account_types.items()
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            fig = px.bar(df, x='Account Type', y='Count',
                        title='Account Types',
                        color='Account Type')
            st.plotly_chart(fig, use_container_width=True)


def page_statistics():
    """Page 3: Statistics and Comparison"""
    st.title("📈 Statistics & Scheme Comparison")
    st.markdown("---")

    if not st.session_state.statistics:
        st.warning("⚠️ No statistics available yet. Please perform some operations first.")
        st.info("💡 Go to Data Management to encrypt data and FHE Operations to run queries.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Overview",
        "⚖️ Scheme Comparison",
        "🌍 Restricted Data Analysis",
        "📉 Detailed Analytics",
        "🔑 Key History"
    ])

    with tab1:
        display_performance_overview()

    with tab2:
        display_scheme_comparison()

    with tab3:
        display_restricted_data_analysis()

    with tab4:
        display_detailed_analytics()

    with tab5:
        display_key_history()


def display_performance_overview():
    """Display overall performance metrics"""
    st.subheader("Performance Overview")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Operations", len(stats_df))

    with col2:
        st.metric("Avg Time (s)", f"{stats_df['time'].mean():.3f}")

    with col3:
        st.metric("Total Time (s)", f"{stats_df['time'].sum():.2f}")

    with col4:
        if 'is_restricted' in stats_df.columns:
            restricted_count = stats_df['is_restricted'].sum()
            st.metric("Restricted Operations", int(restricted_count))

    st.markdown("---")

    # Timeline
    st.subheader("Operation Timeline")
    stats_df['op_id'] = range(1, len(stats_df) + 1)

    fig = px.line(stats_df, x='op_id', y='time', color='operation',
                  markers=True, title='Execution Time per Operation',
                  labels={'op_id': 'Operation Number', 'time': 'Time (s)'})
    st.plotly_chart(fig, use_container_width=True)

    # Operation distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operations by Type")
        op_counts = stats_df['operation'].value_counts()
        fig = px.pie(values=op_counts.values, names=op_counts.index,
                     title='Operation Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Time by Operation")
        time_by_op = stats_df.groupby('operation')['time'].sum()
        fig = px.bar(x=time_by_op.index, y=time_by_op.values,
                     labels={'x': 'Operation', 'y': 'Total Time (s)'},
                     title='Total Time by Operation Type')
        st.plotly_chart(fig, use_container_width=True)


def display_scheme_comparison():
    """Compare different FHE schemes"""
    st.subheader("Scheme Comparison")

    stats_df = pd.DataFrame(st.session_state.statistics)

    if 'scheme' not in stats_df.columns:
        st.info("No scheme data available yet")
        return

    schemes = stats_df['scheme'].unique()

    if len(schemes) < 2:
        st.info(f"Currently using only {schemes[0]} scheme. Perform operations with different schemes to compare.")
        display_scheme_info(schemes[0])
        return

    # Comparison table
    st.write("**Performance Comparison by Scheme**")

    scheme_stats = stats_df.groupby('scheme').agg({
        'time': ['mean', 'min', 'max', 'sum', 'count']
    }).round(4)

    scheme_stats.columns = ['Avg Time', 'Min Time', 'Max Time', 'Total Time', 'Operations']
    st.dataframe(scheme_stats, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        avg_times = stats_df.groupby('scheme')['time'].mean()
        fig = px.bar(x=avg_times.index, y=avg_times.values,
                     labels={'x': 'Scheme', 'y': 'Average Time (s)'},
                     title='Average Execution Time by Scheme',
                     color=avg_times.index)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(stats_df, x='scheme', y='time',
                     title='Time Distribution by Scheme',
                     labels={'scheme': 'Scheme', 'time': 'Time (s)'})
        st.plotly_chart(fig, use_container_width=True)

    # Library comparison if available
    if 'library' in stats_df.columns:
        st.markdown("---")
        st.subheader("Library Performance")

        library_scheme = stats_df.groupby(['library', 'scheme'])['time'].mean().reset_index()
        fig = px.bar(library_scheme, x='scheme', y='time', color='library',
                     barmode='group',
                     title='Average Time by Library and Scheme',
                     labels={'time': 'Average Time (s)'})
        st.plotly_chart(fig, use_container_width=True)


def display_scheme_info(scheme: str):
    """Display information about a single scheme"""
    st.markdown("---")
    st.subheader(f"Scheme Information: {scheme}")

    characteristics = {
        'CKKS': {
            'Type': 'Approximate',
            'Data Support': 'Real/Complex numbers',
            'Best For': 'Machine learning, signal processing, analytics',
            'Precision': 'Approximate (configurable)',
            'Operations': 'Addition, Multiplication, Rotation, Conjugation',
            'Security': 'Based on RLWE problem'
        },
        'BFV': {
            'Type': 'Exact',
            'Data Support': 'Integers',
            'Best For': 'Exact computations, database operations',
            'Precision': 'Exact',
            'Operations': 'Addition, Multiplication',
            'Security': 'Based on RLWE problem'
        },
        'BGV': {
            'Type': 'Exact',
            'Data Support': 'Integers',
            'Best For': 'General purpose, leveled computations',
            'Precision': 'Exact',
            'Operations': 'Addition, Multiplication, Rotation',
            'Security': 'Based on RLWE problem'
        }
    }

    if scheme in characteristics:
        info = characteristics[scheme]
        col1, col2 = st.columns(2)

        with col1:
            for key in ['Type', 'Data Support', 'Best For']:
                st.write(f"**{key}:** {info[key]}")

        with col2:
            for key in ['Precision', 'Operations', 'Security']:
                st.write(f"**{key}:** {info[key]}")


def display_restricted_data_analysis():
    """Display analysis of restricted country data handling"""
    st.subheader("🌍 Restricted Country Data Analysis")

    if not st.session_state.restricted_data_flags.get('count', 0):
        st.info("ℹ️ No restricted country data detected in current dataset.")
        return

    # Summary
    st.write("**Restricted Data Summary**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Restricted Users", st.session_state.restricted_data_flags['count'])

    with col2:
        st.metric("Restricted Countries",
                 len(st.session_state.restricted_data_flags['countries']))

    with col3:
        total_users = len(st.session_state.user_data) if st.session_state.user_data is not None else 0
        st.metric("Total Users", total_users)

    # Countries breakdown
    st.markdown("---")
    st.write("**Affected Countries**")

    restricted_users_df = st.session_state.user_data[
        st.session_state.user_data['country'].isin(RESTRICTED_COUNTRIES)
    ]

    country_counts = restricted_users_df['country'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(values=country_counts.values,
                     names=country_counts.index,
                     title='Users by Restricted Country')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=country_counts.index,
                     y=country_counts.values,
                     labels={'x': 'Country', 'y': 'User Count'},
                     title='User Count by Country')
        st.plotly_chart(fig, use_container_width=True)

    # Processing info
    st.markdown("---")
    st.write("**Data Processing Information**")

    st.info("""
    **How Restricted Country Data is Handled:**
    
    1. **Identification**: Data from restricted countries (CN, RU, KP, IR, SY) is automatically identified
    
    2. **Processing Location**: 
       - ✅ Restricted country data → Processed On-Premises only
       - ✅ Allowed country data → Can be sent to cloud (GCP regions)
    
    3. **Security Measures**:
       - Data remains encrypted throughout processing
       - Decryption only allowed in permitted jurisdictions (HSM/KMS)
       - Separate processing pipelines for restricted data
       - No cloud transfer for restricted data
    
    4. **Compliance**: Ensures adherence to data residency and privacy regulations
    """)

    # Operations on restricted data
    stats_df = pd.DataFrame(st.session_state.statistics)
    if 'is_restricted' in stats_df.columns:
        restricted_ops = stats_df[stats_df['is_restricted'] == True]

        if len(restricted_ops) > 0:
            st.markdown("---")
            st.write("**Operations on Restricted Data**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Restricted Operations", len(restricted_ops))
            with col2:
                st.metric("Avg Processing Time",
                         f"{restricted_ops['time'].mean():.2f}s")

            st.dataframe(
                restricted_ops[['operation', 'scheme', 'library', 'time', 'user_id', 'timestamp']],
                use_container_width=True
            )


def display_detailed_analytics():
    """Display detailed analytics"""
    st.subheader("Detailed Analytics")

    stats_df = pd.DataFrame(st.session_state.statistics)

    # Time analysis
    st.write("**Time Analysis**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Min Time", f"{stats_df['time'].min():.4f}s")
    with col2:
        st.metric("Max Time", f"{stats_df['time'].max():.4f}s")
    with col3:
        st.metric("Mean Time", f"{stats_df['time'].mean():.4f}s")
    with col4:
        st.metric("Std Dev", f"{stats_df['time'].std():.4f}s")

    # Distribution
    fig = px.histogram(stats_df, x='time', nbins=20,
                       title='Distribution of Execution Times',
                       labels={'time': 'Time (s)', 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)

    # Efficiency metrics
    if 'columns' in stats_df.columns:
        st.markdown("---")
        st.subheader("Efficiency Metrics")

        encryption_ops = stats_df[stats_df['operation'] == 'encryption']
        if len(encryption_ops) > 0:
            encryption_ops['cols_per_sec'] = encryption_ops['columns'] / encryption_ops['time']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Columns/Second",
                         f"{encryption_ops['cols_per_sec'].mean():.2f}")
            with col2:
                st.metric("Max Columns/Second",
                         f"{encryption_ops['cols_per_sec'].max():.2f}")

    # Raw data
    st.markdown("---")
    st.subheader("Raw Statistics Data")
    st.dataframe(stats_df, use_container_width=True)

    # Download
    csv = stats_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Statistics CSV",
        data=csv,
        file_name=f"fhe_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_key_history():
    """Display key rotation history"""
    st.subheader("🔑 Key Management History")

    if not st.session_state.key_history:
        st.info("No key history available yet.")
        return

    history_df = pd.DataFrame(st.session_state.key_history)

    st.write("**Key Operations Timeline**")
    st.dataframe(history_df, use_container_width=True)

    # Visualize key operations
    if len(history_df) > 0:
        fig = px.timeline(
            history_df,
            x_start='timestamp',
            x_end='timestamp',
            y='session_id',
            color='action',
            title='Key Operations Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Current key info
    st.markdown("---")
    st.write("**Current Active Keys**")

    if st.session_state.keys:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Session ID:**\n{st.session_state.keys.get('session_id')}")
        with col2:
            st.info(f"**Library:**\n{st.session_state.keys.get('library')}")
        with col3:
            st.info(f"**Scheme:**\n{st.session_state.keys.get('scheme')}")


def main():
    """Main application"""
    st.set_page_config(
        page_title="FHE Financial Processor - Client",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Sidebar
    st.sidebar.title("🔐 FHE Financial Processor")
    st.sidebar.markdown("**Client Application**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Management", "🔒 FHE Operations", "📈 Statistics"],
        key="navigation"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("FHE Library Selection")

    # Library selection
    library = st.sidebar.selectbox(
        "Select FHE Library",
        ["TenSEAL", "OpenFHE"],
        key="fhe_library_select"
    )

    if library != st.session_state.fhe_library:
        st.session_state.fhe_library = library
        st.rerun()

    # Show library info
    if library == "TenSEAL":
        st.sidebar.info("""
        **TenSEAL**
        - Schemes: CKKS, BFV
        - Python native
        - Fast and reliable
        - Good for ML operations
        """)
    else:
        st.sidebar.info("""
        **OpenFHE**
        - Schemes: CKKS, BFV, BGV
        - Multi-mode support
        - Windows compatible
        - Auto-mode selection
        """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Server Configuration")
    new_server_url = st.sidebar.text_input("Server URL", SERVER_URL)
    if new_server_url != SERVER_URL:
        st.session_state.client = FHEClient(new_server_url)

    # Server status
    if st.session_state.client.check_server_health():
        st.sidebar.success(f"🟢 Server Online")

        # Show library availability from server
        try:
            response = st.session_state.client.session.get(f"{SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                libraries = health_data.get('libraries', {})

                with st.sidebar.expander("Server Capabilities"):
                    for lib_name, lib_info in libraries.items():
                        if isinstance(lib_info, dict):
                            available = lib_info.get('available', False)
                            status = "✅" if available else "❌"
                            st.write(f"{status} **{lib_name.upper()}**")
                            if available and lib_name == 'openfhe':
                                mode = lib_info.get('mode', 'unknown')
                                st.caption(f"Mode: {mode}")
        except:
            pass
    else:
        st.sidebar.error("🔴 Server Offline")

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Restricted Countries:**\n{', '.join(RESTRICTED_COUNTRIES)}")

    # Display selected page
    if page == "📊 Data Management":
        page_data_management()
    elif page == "🔒 FHE Operations":
        page_fhe_operations()
    elif page == "📈 Statistics":
        page_statistics()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("FHE Financial Processor v2.1")
    st.sidebar.caption("Enhanced Client-Server Architecture")
    st.sidebar.caption(f"© {datetime.now().year}")


if __name__ == "__main__":
    main()