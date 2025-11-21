"""
FHE Client Application - Streamlit UI
Compatible with Python 3.11
Financial Data Encryption & Analysis with FHE
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import io
import base64
import asyncio
import concurrent.futures
from threading import Lock

# Configuration
SERVER_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="FHE Financial Data Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thread lock for session state updates
session_lock = Lock()

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
        # Fixed: Use smaller range for int32 compatibility
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
            'balance': np.round(np.random.uniform(1000, 1000000), 2),
            'payment_mode': np.random.choice(payment_modes),
            'transaction_id': np.random.randint(100000, 9999999),
            'amount_transferred': np.round(np.random.uniform(10, 50000), 2),
            'payment_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(np.random.randint(0, 365))),
            'payment_country': np.random.choice(countries)
        }
        data.append(record)

    return pd.DataFrame(data)

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
            response = requests.post(url, json=data, timeout=60)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Server timeout - operation taking longer than expected")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Server error: {str(e)}")
        return None

# ==================== SCREEN 1: DATA UPLOAD & CONFIGURATION ====================

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

    # FHE Configuration Section
    st.header("2️⃣ FHE Configuration")

    col1, col2 = st.columns(2)

    with col1:
        library = st.selectbox("Select FHE Library", ["TenSEAL", "OpenFHE"])
        scheme = st.selectbox("Select Encryption Scheme", ["CKKS", "BFV", "BGV"])

    with col2:
        poly_degree = st.selectbox("Polynomial Degree", [4096, 8192, 16384, 32768], index=1)

        if library == "TenSEAL" and scheme == "CKKS":
            scale = st.selectbox("Scale (2^n)", [30, 40, 50, 60], index=1)
            scale_value = 2 ** scale
        else:
            scale_mod_size = st.number_input("Scale Modulus Size", min_value=30, max_value=60, value=50)
            scale_value = None

    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters"):
        mult_depth = st.slider("Multiplicative Depth", min_value=1, max_value=20, value=10)

        if library == "TenSEAL":
            coeff_sizes = st.text_input("Coefficient Modulus Bit Sizes (comma-separated)", "60,40,40,60")
            coeff_list = [int(x.strip()) for x in coeff_sizes.split(',')]

        if scheme == "BFV":
            plain_modulus = st.number_input("Plain Modulus", min_value=2, value=1032193)

    # Key Generation
    st.subheader("🔑 Key Generation")

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
        with st.expander("🔐 View Generated Keys"):
            st.json({
                "public_key": st.session_state.keys_info.get('public_key', '')[:100] + "...",
                "private_key": st.session_state.keys_info.get('private_key', '')[:100] + "...",
                "scheme": st.session_state.scheme,
                "library": st.session_state.library
            })

# ==================== SCREEN 2: DATA ENCRYPTION ====================

def screen_2_encryption():
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

        for idx, col in enumerate(all_selected):
            status_text.text(f"Encrypting {col}... ({idx + 1}/{total_cols})")

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

            # Get party_ids and payment_dates for filtering
            party_ids = st.session_state.data['partyid'].tolist() if 'partyid' in st.session_state.data.columns else []
            payment_dates = st.session_state.data['payment_date'].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            ).tolist() if 'payment_date' in st.session_state.data.columns else []

            # Send encryption request
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

            start_time = time.time()
            result = call_server("/encrypt", "POST", enc_data)
            elapsed = time.time() - start_time

            if result and result.get('status') == 'success':
                # Store metadata
                st.session_state.encrypted_metadata[col] = {
                    "records": result.get('metadata_records', []),
                    "count": result.get('encrypted_count', 0),
                    "batch_id": batch_id,
                    "time": elapsed
                }

                # Store stats
                st.session_state.encryption_stats.append({
                    "column": col,
                    "type": data_type,
                    "count": len(column_data),
                    "time": elapsed,
                    "throughput": len(column_data) / elapsed if elapsed > 0 else 0
                })

            progress_bar.progress((idx + 1) / total_cols)

        status_text.text("✅ Encryption complete!")
        st.success(f"Encrypted {total_cols} columns successfully!")
        time.sleep(1)
        st.rerun()

    # Display encrypted metadata
    if st.session_state.encrypted_metadata:
        st.divider()
        st.subheader("📊 Encrypted Data Metadata")

        for col, metadata in st.session_state.encrypted_metadata.items():
            with st.expander(f"🔒 {col} - {metadata['count']} records encrypted"):
                records_df = pd.DataFrame(metadata['records'][:100])
                st.dataframe(records_df, use_container_width=True)

# ==================== SCREEN 2: FHE ANALYSIS ====================

def screen_2_analysis():
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
                        if st.button(f"Decrypt {operation.upper()}", key=f"decrypt_{column_name}_{operation}", use_container_width=True):
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
                                        if entry['Column'] == reconciliation_entry['Column'] and entry['Operation'] == reconciliation_entry['Operation']:
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

# ==================== SCREEN 3: STATISTICS ====================

def screen_3_statistics():
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

# ==================== MAIN APP ====================

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Screen",
        ["1. Data Upload & Config", "2. Data Encryption", "3. FHE Analysis", "4. Statistics"]
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

    # Main content
    if page == "1. Data Upload & Config":
        screen_1_data_upload()
    elif page == "2. Data Encryption":
        screen_2_encryption()
    elif page == "3. FHE Analysis":
        screen_2_analysis()
    elif page == "4. Statistics":
        screen_3_statistics()

if __name__ == "__main__":
    main()