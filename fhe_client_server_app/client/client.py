"""
Comprehensive FHE Financial Transaction Client - FIXED VERSION
3-Screen Application with Complete Session Management
Python 3.11+
"""

import streamlit as st
import pandas as pd
import numpy as np
import httpx
import json
import time
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List

st.set_page_config(page_title="FHE Client", layout="wide", initial_sidebar_state="expanded")

# Initialize session state - FIXED with library and scheme persistence
def init_state():
    defaults = {
        'server_url': 'http://localhost:8000',
        'df_parties': None,
        'df_accounts': None,
        'df_payments': None,
        'keys_info': None,
        'encrypted_data': {},
        'analysis_results': None,
        'statistics': [],
        'selected_columns': {'parties': [], 'accounts': [], 'payments': []},
        'encryption_metrics': [],
        'query_metrics': [],
        'current_library': None,  # NEW: Store current library
        'current_scheme': None,   # NEW: Store current scheme
        'current_params': {},     # NEW: Store current parameters
        'skipped_columns': []     # NEW: Track skipped columns
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    SERVER_URL = st.text_input("Server URL", st.session_state.server_url)
    st.session_state.server_url = SERVER_URL

    try:
        health = httpx.get(f"{SERVER_URL}/health", timeout=2).json()
        st.success("✅ Server Connected")
        with st.expander("Server Info"):
            st.json(health)
    except:
        st.error("❌ Server Offline")

    # Display current session info
    if st.session_state.keys_info:
        st.markdown("---")
        st.success("✅ Active FHE Session")
        st.metric("Library", st.session_state.current_library or "N/A")
        st.metric("Scheme", st.session_state.current_scheme or "N/A")
        st.caption(f"Session: {st.session_state.keys_info.get('session_id', 'N/A')[:12]}...")

# Scheme information - UPDATED with proper support flags
SCHEME_INFO = {
    'CKKS': {
        'desc': 'Approximate arithmetic on real numbers',
        'supports': {'numeric': True, 'text': 'limited', 'date': True},
        'warnings': ['Approximate results', 'No exact integers']
    },
    'BFV': {
        'desc': 'Exact integer arithmetic',
        'supports': {'numeric': True, 'text': False, 'date': 'limited'},
        'warnings': ['Integer only', 'No floating point', 'Text NOT supported']
    },
    'BGV': {
        'desc': 'Exact integer with rotation',
        'supports': {'numeric': True, 'text': 'limited', 'date': 'limited'},
        'warnings': ['Integer only', 'Requires parameter tuning']
    }
}

# Generate synthetic data - FIXED random.randint issue
def generate_data(n_parties=100, n_accounts=150, n_payments=1000):
    import random
    countries = ['USA', 'UK', 'Germany', 'France', 'Japan', 'China', 'Turkey', 'Saudi Arabia']

    parties = [{
        'party_id': f'P{i+1:05d}',
        'name': f'Person {i+1}',
        'email': f'person{i+1}@example.com',
        'dob': (date.today() - timedelta(days=random.randint(7000, 25000))).isoformat(),
        'country': random.choice(countries),
        'region': random.choice(['North', 'South', 'East', 'West']),
        'address': f'{random.randint(1,9999)} Street {i+1}'
    } for i in range(n_parties)]

    # FIXED: Convert to int first, then to string
    accounts = [{
        'account_id': f'A{i+1:06d}',
        'party_id': random.choice(parties)['party_id'],
        'account_number': str(random.randint(1000000000, 9999999999)),  # FIXED
        'account_type': random.choice(['Checking', 'Savings', 'Credit', 'Investment']),
        'balance': round(random.uniform(100, 100000), 2),
        'currency': random.choice(['USD', 'EUR', 'GBP', 'JPY'])
    } for i in range(n_accounts)]

    payments = [{
        'transaction_id': f'T{i+1:07d}',
        'party_id': random.choice(parties)['party_id'],
        'account_id': random.choice(accounts)['account_id'],
        'amount': round(random.uniform(10, 10000), 2),
        'currency': random.choice(['USD', 'EUR', 'GBP']),
        'transaction_date': (date.today() - timedelta(days=random.randint(1, 365))).isoformat(),
        'transaction_type': random.choice(['Purchase', 'Transfer', 'Withdrawal', 'Deposit'])
    } for i in range(n_payments)]

    return pd.DataFrame(parties), pd.DataFrame(accounts), pd.DataFrame(payments)

# Detect data type
def detect_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    return 'text'

# ENHANCED: Check scheme compatibility with detailed messages
def check_compatibility(scheme, dtype):
    """Check if scheme supports data type - returns (compatible, message)"""
    supports = SCHEME_INFO[scheme]['supports']

    if dtype == 'text':
        if supports.get('text') == False:
            return False, f"❌ {scheme} does NOT support text data type"
        elif supports.get('text') == 'limited':
            return True, f"⚠️ {scheme} has limited text support (will encode as numeric)"

    if dtype == 'date':
        if supports.get('date') == 'limited':
            return True, f"⚠️ {scheme} has limited date support (will encode as timestamp)"
        elif supports.get('date') == False:
            return False, f"❌ {scheme} does NOT support date data type"

    return True, None

# Main app
st.title("🔐 FHE Financial Transaction System")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Screen 1: Data & Keys", "🔒 Screen 2: Encryption & Analysis", "📈 Screen 3: Statistics"])

# ==================== SCREEN 1 ====================
with tab1:
    st.header("Screen 1: Data Management & Key Generation")

    # Section: Data Source
    st.subheader("📁 Step 1: Load Data")
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio("Data Source", ["Generate Synthetic", "Upload CSV"])

        if data_source == "Generate Synthetic":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                n_parties = st.number_input("Parties", 10, 1000, 100)
            with col_b:
                n_accounts = st.number_input("Accounts", 10, 2000, 150)
            with col_c:
                n_payments = st.number_input("Payments", 50, 10000, 1000)

            if st.button("🎲 Generate Data", type="primary"):
                with st.spinner("Generating..."):
                    df_p, df_a, df_t = generate_data(n_parties, n_accounts, n_payments)
                    st.session_state.df_parties = df_p
                    st.session_state.df_accounts = df_a
                    st.session_state.df_payments = df_t
                    st.success("✅ Data generated!")

        else:
            uploaded = st.file_uploader("Upload CSVs", type=['csv'], accept_multiple_files=True)
            if uploaded:
                for file in uploaded:
                    df = pd.read_csv(file)
                    if 'party' in file.name.lower() or 'user' in file.name.lower():
                        st.session_state.df_parties = df
                        st.success(f"✅ Loaded {len(df)} parties")
                    elif 'account' in file.name.lower():
                        st.session_state.df_accounts = df
                        st.success(f"✅ Loaded {len(df)} accounts")
                    elif 'payment' in file.name.lower() or 'transaction' in file.name.lower():
                        st.session_state.df_payments = df
                        st.success(f"✅ Loaded {len(df)} payments")

    with col2:
        st.subheader("🔑 Step 2: FHE Configuration")

        # Library selection - STORE in session state
        library = st.selectbox("Library", ["TenSEAL", "OpenFHE"], key='lib_select')

        # Scheme selection - STORE in session state
        schemes = ["CKKS", "BFV", "BGV"] if library == "OpenFHE" else ["CKKS", "BFV"]
        scheme = st.selectbox("Scheme", schemes, key='scheme_select')

        # Display scheme info and limitations
        st.info(f"**{scheme}**: {SCHEME_INFO[scheme]['desc']}")

        with st.expander("ℹ️ Scheme Capabilities & Limitations"):
            supports = SCHEME_INFO[scheme]['supports']
            st.write("**Data Type Support:**")
            for dtype, support in supports.items():
                if support == True:
                    st.success(f"✅ {dtype.capitalize()}: Fully supported")
                elif support == 'limited':
                    st.warning(f"⚠️ {dtype.capitalize()}: Limited support")
                else:
                    st.error(f"❌ {dtype.capitalize()}: NOT supported")

            st.write("**Warnings:**")
            for warning in SCHEME_INFO[scheme]['warnings']:
                st.caption(f"⚠️ {warning}")

        with st.expander("⚙️ Parameters"):
            poly_deg = st.selectbox("Polynomial Degree", [4096, 8192, 16384, 32768], index=1)

            params = {'poly_modulus_degree': poly_deg}

            if scheme == "CKKS":
                scale = st.selectbox("Scale", [2**30, 2**40, 2**50], index=1)
                params['scale'] = scale

            if library == "OpenFHE":
                mult_depth = st.number_input("Mult Depth", 1, 20, 10)
                scale_mod = st.number_input("Scale Mod Size", 30, 60, 50)
                params['mult_depth'] = mult_depth
                params['scale_mod_size'] = scale_mod

        if st.button("🔑 Generate Keys", type="primary"):
            with st.spinner("Generating keys..."):
                try:
                    resp = httpx.post(f"{SERVER_URL}/generate_keys", json={
                        'scheme': scheme,
                        'library': library,
                        'params': params
                    }, timeout=60)

                    if resp.status_code != 200:
                        st.error(f"❌ Server error: {resp.text}")
                        st.stop()

                    keys = resp.json()

                    # CRITICAL FIX: Store library and scheme in session state
                    st.session_state.keys_info = keys
                    st.session_state.current_library = library
                    st.session_state.current_scheme = scheme
                    st.session_state.current_params = params

                    # Also ensure they're in the keys object
                    if 'library' not in keys:
                        keys['library'] = library
                    if 'scheme' not in keys:
                        keys['scheme'] = scheme

                    st.success("✅ Keys generated successfully!")

                    # Display comprehensive info
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric("Library", library)
                        st.metric("Scheme", scheme)
                    with col_y:
                        st.metric("Poly Degree", poly_deg)
                        st.metric("Session ID", keys.get('session_id', 'N/A')[:12] + "...")

                    # Download keys
                    st.download_button(
                        "📥 Download Keys",
                        json.dumps(keys, indent=2),
                        f"keys_{library}_{scheme}_{datetime.now():%Y%m%d_%H%M%S}.json",
                        "application/json"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display loaded data
    st.markdown("---")
    st.subheader("📋 Loaded Data Preview")

    if st.session_state.df_parties is not None:
        with st.expander(f"👥 Parties ({len(st.session_state.df_parties)} records)"):
            st.dataframe(st.session_state.df_parties.head(10), use_container_width=True)

    if st.session_state.df_accounts is not None:
        with st.expander(f"💳 Accounts ({len(st.session_state.df_accounts)} records)"):
            st.dataframe(st.session_state.df_accounts.head(10), use_container_width=True)

    if st.session_state.df_payments is not None:
        with st.expander(f"💰 Payments ({len(st.session_state.df_payments)} records)"):
            st.dataframe(st.session_state.df_payments.head(10), use_container_width=True)

# ==================== SCREEN 2 ====================
with tab2:
    st.header("Screen 2: Encryption & Analysis")

    if not st.session_state.keys_info:
        st.warning("⚠️ Generate keys in Screen 1 first")
    elif st.session_state.df_parties is None:
        st.warning("⚠️ Load data in Screen 1 first")
    else:
        # Verify session has library and scheme
        if not st.session_state.current_library or not st.session_state.current_scheme:
            st.error("❌ Session information incomplete. Please regenerate keys in Screen 1.")
            st.stop()

        # Display current session
        st.info(f"**Active Session:** Library={st.session_state.current_library}, Scheme={st.session_state.current_scheme}")

        # Section 1: Encryption
        st.subheader("Section 1: Data Encryption")

        st.write("**Select columns to encrypt:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Parties**")
            if st.session_state.df_parties is not None:
                party_cols = st.multiselect("Party columns", list(st.session_state.df_parties.columns), key='sel_party')
                st.session_state.selected_columns['parties'] = party_cols

        with col2:
            st.write("**Accounts**")
            if st.session_state.df_accounts is not None:
                account_cols = st.multiselect("Account columns", list(st.session_state.df_accounts.columns), key='sel_acc')
                st.session_state.selected_columns['accounts'] = account_cols

        with col3:
            st.write("**Payments**")
            if st.session_state.df_payments is not None:
                payment_cols = st.multiselect("Payment columns", list(st.session_state.df_payments.columns), key='sel_pay')
                st.session_state.selected_columns['payments'] = payment_cols

        batch_size = st.slider("Batch Size", 10, 1000, 100)

        if st.button("🔒 Encrypt Selected Columns", type="primary", use_container_width=True):
            # Get session info
            keys = st.session_state.keys_info
            session_id = keys.get('session_id')
            library = st.session_state.current_library
            scheme = st.session_state.current_scheme

            if not session_id or not library or not scheme:
                st.error("❌ Invalid session. Please regenerate keys.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            total_cols = sum(len(v) for v in st.session_state.selected_columns.values())

            if total_cols == 0:
                st.warning("⚠️ No columns selected")
                st.stop()

            current = 0
            skipped = []  # Track skipped columns

            datasets = {
                'parties': st.session_state.df_parties,
                'accounts': st.session_state.df_accounts,
                'payments': st.session_state.df_payments
            }

            for table, df in datasets.items():
                if df is None:
                    continue

                for col in st.session_state.selected_columns[table]:
                    current += 1
                    progress.progress(current / total_cols)
                    status.text(f"Processing {table}.{col}... ({current}/{total_cols})")

                    data = df[col].tolist()
                    dtype = detect_type(df[col])

                    # CRITICAL: Check scheme compatibility BEFORE sending to server
                    compatible, msg = check_compatibility(scheme, dtype)

                    if not compatible:
                        # SKIP this column - do NOT send to server
                        skipped.append({
                            'table': table,
                            'column': col,
                            'data_type': dtype,
                            'reason': msg
                        })
                        st.warning(f"⚠️ SKIPPED {table}.{col}: {msg}")
                        continue
                    elif msg:
                        # Compatible but with warning
                        st.info(msg)

                    # Proceed with encryption
                    start_time = time.time()

                    try:
                        encrypted = []
                        storage_keys = []  # Track server-side storage keys

                        for i in range(0, len(data), batch_size):
                            batch = data[i:i+batch_size]
                            resp = httpx.post(f"{SERVER_URL}/encrypt", json={
                                'data': batch,
                                'column_name': col,
                                'data_type': dtype,
                                'keys': {'session_id': session_id},
                                'scheme': scheme,
                                'library': library,
                                'batch_id': i//batch_size
                            }, timeout=120)

                            if resp.status_code != 200:
                                raise Exception(f"Server error: {resp.text}")

                            result = resp.json()
                            if result.get('success'):
                                # OPTIMIZED: Store only preview + storage key
                                if result.get('stored_server_side'):
                                    storage_keys.append(result.get('storage_key'))
                                    encrypted.extend(result['encrypted_values'])  # Only preview
                                else:
                                    encrypted.extend(result['encrypted_values'])

                        enc_time = time.time() - start_time

                        st.session_state.encrypted_data[f"{table}:{col}"] = {
                            'encrypted_values': encrypted[:5],  # Store only first 5 for preview
                            'storage_keys': storage_keys,  # Server-side storage references
                            'dtype': dtype,
                            'table': table,
                            'column': col,
                            'count': len(data),  # Original count
                            'time': enc_time,
                            'scheme': scheme,
                            'library': library,
                            'stored_server_side': True
                        }

                        st.session_state.encryption_metrics.append({
                            'table': table,
                            'column': col,
                            'dtype': dtype,
                            'count': len(data),
                            'time': enc_time,
                            'throughput': len(data)/enc_time if enc_time > 0 else 0,
                            'scheme': scheme,
                            'library': library
                        })

                    except Exception as e:
                        st.error(f"❌ Error encrypting {table}.{col}: {e}")
                        skipped.append({
                            'table': table,
                            'column': col,
                            'data_type': dtype,
                            'reason': f"Encryption error: {str(e)}"
                        })

            # Store skipped columns
            st.session_state.skipped_columns = skipped

            progress.progress(1.0)
            status.text("✅ Processing complete!")

            # Summary
            successful = len(st.session_state.encrypted_data)
            skipped_count = len(skipped)

            if successful > 0:
                st.success(f"✅ Encrypted {successful} columns successfully")

            if skipped_count > 0:
                st.warning(f"⚠️ Skipped {skipped_count} incompatible columns")
                with st.expander("📋 View Skipped Columns Details"):
                    df_skipped = pd.DataFrame(skipped)
                    st.dataframe(df_skipped, use_container_width=True)
                    st.info("💡 These columns were NOT sent to the server due to scheme limitations")

        # Display encrypted data preview
        if st.session_state.encrypted_data:
            st.markdown("---")
            st.subheader("🔐 Encrypted Data Preview (First 5 records)")

            for key, data in st.session_state.encrypted_data.items():
                with st.expander(f"🔒 {key} ({data['count']} records)", expanded=False):
                    table = data['table']
                    col = data['column']

                    # Get original data
                    if table == 'parties':
                        original = st.session_state.df_parties[col].head(5).tolist()
                    elif table == 'accounts':
                        original = st.session_state.df_accounts[col].head(5).tolist()
                    else:
                        original = st.session_state.df_payments[col].head(5).tolist()

                    encrypted = data['encrypted_values'][:5]

                    # Display comparison
                    preview_data = []
                    for i in range(min(5, len(original))):
                        enc_val = encrypted[i]
                        if enc_val and isinstance(enc_val, dict) and 'ciphertext' in enc_val:
                            enc_display = enc_val['ciphertext'][:50] + "..."
                        else:
                            enc_display = str(enc_val)[:50] + "..." if enc_val else "NULL"

                        preview_data.append({
                            'Index': i,
                            'Original': str(original[i])[:30],
                            'Encrypted': enc_display,
                            'Type': data['dtype']
                        })

                    st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
                    st.caption(f"⏱️ Encryption: {data['time']:.2f}s | Library: {data['library']} | Scheme: {data['scheme']}")

        st.markdown("---")

        # Section 2: Analysis
        st.subheader("Section 2: FHE Analysis")

        if not st.session_state.encrypted_data:
            st.info("ℹ️ Encrypt data first to run analysis")
        else:
            st.success(f"✅ {len(st.session_state.encrypted_data)} columns encrypted and ready")

            col_a, col_b = st.columns([2, 1])

            with col_a:
                analysis_type = st.selectbox("Analysis Type", [
                    "Transaction Analysis",
                    "Transaction Count",
                    "Account Summary"
                ])

                col_x, col_y = st.columns(2)
                with col_x:
                    start_date = st.date_input("Start Date", date.today() - timedelta(365))
                with col_y:
                    end_date = st.date_input("End Date", date.today())

                party_id = st.text_input("Party ID (optional)")

            with col_b:
                st.write("**Session Info**")
                st.metric("Library", st.session_state.current_library)
                st.metric("Scheme", st.session_state.current_scheme)
                st.metric("Encrypted Cols", len(st.session_state.encrypted_data))

            if st.button("▶️ Run FHE Analysis", type="primary", use_container_width=True):
                with st.spinner("Running FHE operations on encrypted data..."):
                    keys = st.session_state.keys_info
                    library = st.session_state.current_library
                    scheme = st.session_state.current_scheme

                    # OPTIMIZED: Send only metadata, not full encrypted data
                    encrypted_metadata = {}
                    for key, data in st.session_state.encrypted_data.items():
                        encrypted_metadata[key] = {
                            'count': data['count'],
                            'dtype': data['dtype'],
                            'table': data['table'],
                            'column': data['column']
                        }

                    start_time = time.time()

                    try:
                        resp = httpx.post(f"{SERVER_URL}/fhe_query", json={
                            'encrypted_metadata': encrypted_metadata,  # Send metadata instead of full data
                            'query_params': {
                                'operation_type': analysis_type,
                                'user_id': party_id or None,
                                'start_date': str(start_date),
                                'end_date': str(end_date)
                            },
                            'keys': {'session_id': keys['session_id']},
                            'library': library,
                            'scheme': scheme
                        }, timeout=60)

                        if resp.status_code != 200:
                            st.error(f"❌ Server error: {resp.text}")
                            st.stop()

                        results = resp.json()
                        query_time = time.time() - start_time

                        st.session_state.analysis_results = results
                        st.session_state.query_metrics.append({
                            'operation': analysis_type,
                            'time': query_time,
                            'timestamp': datetime.now(),
                            'library': library,
                            'scheme': scheme
                        })

                        st.success("✅ FHE Analysis complete!")

                        # Display results
                        st.subheader("📊 Analysis Results")
                        st.json(results)

                        st.download_button(
                            "📥 Download Results",
                            json.dumps(results, indent=2, default=str),
                            f"results_{datetime.now():%Y%m%d_%H%M%S}.json",
                            "application/json"
                        )

                    except httpx.ReadTimeout:
                        st.error("❌ Request timeout. Server might be processing large data.")
                    except MemoryError:
                        st.error("❌ Memory error. Try reducing the amount of data or batch size.")
                    except Exception as e:
                        st.error(f"❌ Query error: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

# ==================== SCREEN 3 ====================
with tab3:
    st.header("Screen 3: Performance Statistics & Comparisons")

    if not st.session_state.encryption_metrics:
        st.info("ℹ️ No statistics available. Perform encryption in Screen 2 first.")
    else:
        # Encryption metrics
        st.subheader("🔒 Encryption Performance")

        df_enc = pd.DataFrame(st.session_state.encryption_metrics)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Columns", len(df_enc))
        with col2:
            st.metric("Total Records", int(df_enc['count'].sum()))
        with col3:
            st.metric("Avg Throughput", f"{df_enc['throughput'].mean():.1f} rec/s")
        with col4:
            if st.session_state.skipped_columns:
                st.metric("Skipped Columns", len(st.session_state.skipped_columns),
                         delta="incompatible", delta_color="off")

        st.dataframe(df_enc, use_container_width=True)

        # Visualizations
        st.subheader("📈 Performance Visualizations")

        tab_a, tab_b, tab_c = st.tabs(["Encryption Time", "Throughput", "By Scheme"])

        with tab_a:
            fig = px.bar(df_enc, x='column', y='time', color='table',
                        title='Encryption Time by Column',
                        labels={'time': 'Time (seconds)', 'column': 'Column'})
            st.plotly_chart(fig, use_container_width=True)

        with tab_b:
            fig = px.bar(df_enc, x='column', y='throughput', color='dtype',
                        title='Encryption Throughput',
                        labels={'throughput': 'Records/Second'})
            st.plotly_chart(fig, use_container_width=True)

        with tab_c:
            if 'scheme' in df_enc.columns:
                scheme_stats = df_enc.groupby('scheme').agg({
                    'time': 'mean',
                    'throughput': 'mean',
                    'count': 'sum'
                }).reset_index()

                fig = px.bar(scheme_stats, x='scheme', y='throughput',
                            title='Average Throughput by Scheme',
                            labels={'throughput': 'Records/Second'})
                st.plotly_chart(fig, use_container_width=True)

        # Query metrics
        if st.session_state.query_metrics:
            st.markdown("---")
            st.subheader("🔍 Query Performance")

            df_query = pd.DataFrame(st.session_state.query_metrics)
            st.dataframe(df_query, use_container_width=True)

        # Scheme comparison
        st.markdown("---")
        st.subheader("🔬 Scheme Characteristics Comparison")

        comparison_data = []
        for scheme_name, info in SCHEME_INFO.items():
            comparison_data.append({
                'Scheme': scheme_name,
                'Description': info['desc'],
                'Numeric': '✅' if info['supports']['numeric'] == True else '❌',
                'Text': '✅' if info['supports']['text'] == True else '⚠️' if info['supports']['text'] == 'limited' else '❌',
                'Date': '✅' if info['supports']['date'] == True else '⚠️' if info['supports']['date'] == 'limited' else '❌',
                'Warnings': ', '.join(info['warnings'])
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

        # Skipped columns summary
        if st.session_state.skipped_columns:
            st.markdown("---")
            st.subheader("⚠️ Skipped Columns Summary")
            st.warning(f"The following {len(st.session_state.skipped_columns)} columns were skipped due to scheme incompatibility:")

            df_skipped = pd.DataFrame(st.session_state.skipped_columns)
            st.dataframe(df_skipped, use_container_width=True)

            st.info("💡 **Recommendation:** Consider using CKKS scheme for better compatibility with mixed data types")

        # Download statistics
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            if st.button("📥 Download Encryption Statistics (CSV)"):
                csv = df_enc.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"encryption_stats_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    "text/csv"
                )

        with col_d2:
            if st.button("📥 Download All Statistics (JSON)"):
                stats = {
                    'encryption_metrics': st.session_state.encryption_metrics,
                    'query_metrics': st.session_state.query_metrics,
                    'skipped_columns': st.session_state.skipped_columns,
                    'session_info': {
                        'library': st.session_state.current_library,
                        'scheme': st.session_state.current_scheme,
                        'params': st.session_state.current_params
                    }
                }
                st.download_button(
                    "Download JSON",
                    json.dumps(stats, indent=2, default=str),
                    f"statistics_{datetime.now():%Y%m%d_%H%M%S}.json",
                    "application/json"
                )

if __name__ == "__main__":
    pass