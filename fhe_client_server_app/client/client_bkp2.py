"""
FHE Financial Transaction Client Application
Multi-screen GUI application with data encryption and analysis
Python 3.11+
Required packages: streamlit, pandas, numpy, httpx, plotly, altair
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


# ==================== Tab 3: Results & Statistics ====================

def render_results_statistics_tab():
    st.header("📈 Results & Statistics")

    # Analysis Results Section
    if st.session_state.analysis_results:
        st.subheader("🔍 Analysis Results")

        results = st.session_state.analysis_results

        # Check if restricted
        if results.get('is_restricted'):
            st.error("🚨 RESTRICTED COUNTRY DATA")
            st.warning(f"**Processing Location:** {results.get('processing_location', 'Unknown')}")
            st.info(f"**Compliance Note:** {results.get('compliance', 'N/A')}")
        else:
            st.success(f"✅ **Processing Location:** {results.get('processing_location', 'cloud')}")

        # Display detailed results
        with st.expander("📊 Detailed Results", expanded=True):
            # Summary metrics
            if 'total_transactions' in results:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", results.get('total_transactions', 0))
                with col2:
                    st.metric("Total Amount", f"${results.get('total_amount', 0):,.2f}")
                with col3:
                    st.metric("Average Amount", f"${results.get('avg_amount', 0):,.2f}")
                with col4:
                    st.metric("Std Deviation", f"${results.get('std_deviation', 0):,.2f}")

            # Currency breakdown
            if 'currency_breakdown' in results:
                st.markdown("### 💱 Currency Analysis")
                currency_data = results['currency_breakdown']

                df_currency = pd.DataFrame([
                    {
                        'Currency': curr,
                        'Count': data['count'],
                        'Total': data['total'],
                        'Average': data['avg']
                    }
                    for curr, data in currency_data.items()
                ])

                col_x, col_y = st.columns(2)
                with col_x:
                    st.dataframe(df_currency, use_container_width=True)

                with col_y:
                    fig = px.pie(df_currency, values='Total', names='Currency',
                                 title='Transaction Amount by Currency')
                    st.plotly_chart(fig, use_container_width=True)

            # Monthly pattern
            if 'monthly_pattern' in results:
                st.markdown("### 📅 Monthly Transaction Pattern")
                monthly_data = results['monthly_pattern']

                df_monthly = pd.DataFrame([
                    {'Month': month, 'Transactions': count}
                    for month, count in monthly_data.items()
                ])

                fig = px.line(df_monthly, x='Month', y='Transactions',
                              title='Transaction Trend Over Time',
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            # Transaction type distribution
            if 'transaction_type_distribution' in results:
                st.markdown("### 🔄 Transaction Type Distribution")
                type_data = results['transaction_type_distribution']

                df_types = pd.DataFrame([
                    {'Type': type_name, 'Count': count}
                    for type_name, count in type_data.items()
                ])

                fig = px.bar(df_types, x='Type', y='Count',
                             title='Transactions by Type',
                             color='Type')
                st.plotly_chart(fig, use_container_width=True)

            # Account summary
            if 'account_types' in results:
                st.markdown("### 💳 Account Summary")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Total Accounts", results.get('total_accounts', 0))
                    st.metric("Active Accounts", results.get('active_accounts', 0))
                    st.metric("Total Balance", f"${results.get('total_balance', 0):,.2f}")

                with col_b:
                    account_types = results['account_types']
                    df_accounts = pd.DataFrame([
                        {'Type': type_name, 'Count': count}
                        for type_name, count in account_types.items()
                    ])
                    fig = px.bar(df_accounts, x='Type', y='Count',
                                 title='Accounts by Type')
                    st.plotly_chart(fig, use_container_width=True)

        # Download results
        results_json = json.dumps(results, indent=2)
        st.download_button(
            "📥 Download Analysis Results",
            results_json,
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    else:
        st.info("ℹ️ No analysis results available. Run analysis in Tab 2.")

    st.markdown("---")

    # Statistics Section
    st.subheader("📊 Encryption & Performance Statistics")

    if st.session_state.statistics:
        df_stats = pd.DataFrame(st.session_state.statistics)

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns Encrypted", len(df_stats))
        with col2:
            st.metric("Total Records Encrypted", df_stats['encrypted_count'].sum())
        with col3:
            st.metric("Avg Encryption Rate", f"{df_stats['rate'].mean():.2f} rec/sec")

        # Detailed statistics table
        with st.expander("📋 Detailed Encryption Statistics", expanded=True):
            st.dataframe(df_stats[['table', 'column', 'data_type', 'library', 'scheme',
                                   'original_count', 'encrypted_count', 'encryption_time', 'rate']],
                         use_container_width=True)

        # Visualizations
        st.markdown("### 📈 Performance Visualizations")

        tab_a, tab_b, tab_c = st.tabs(["Encryption Time", "Scheme Comparison", "Library Comparison"])

        with tab_a:
            # Encryption time by column
            fig = px.bar(df_stats, x='column', y='encryption_time', color='table',
                         title='Encryption Time by Column',
                         labels={'encryption_time': 'Time (seconds)', 'column': 'Column'})
            st.plotly_chart(fig, use_container_width=True)

        with tab_b:
            # Scheme comparison
            if 'scheme' in df_stats.columns and len(df_stats['scheme'].unique()) > 1:
                scheme_stats = df_stats.groupby('scheme').agg({
                    'encryption_time': 'mean',
                    'rate': 'mean',
                    'encrypted_count': 'sum'
                }).reset_index()

                col_x, col_y = st.columns(2)

                with col_x:
                    fig1 = px.bar(scheme_stats, x='scheme', y='encryption_time',
                                  title='Average Encryption Time by Scheme',
                                  labels={'encryption_time': 'Avg Time (seconds)'})
                    st.plotly_chart(fig1, use_container_width=True)

                with col_y:
                    fig2 = px.bar(scheme_stats, x='scheme', y='rate',
                                  title='Average Encryption Rate by Scheme',
                                  labels={'rate': 'Records/Second'})
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Encrypt data with multiple schemes to see comparison")

        with tab_c:
            # Library comparison
            if 'library' in df_stats.columns and len(df_stats['library'].unique()) > 1:
                library_stats = df_stats.groupby('library').agg({
                    'encryption_time': 'mean',
                    'rate': 'mean',
                    'encrypted_count': 'sum'
                }).reset_index()

                col_x, col_y = st.columns(2)

                with col_x:
                    fig1 = px.bar(library_stats, x='library', y='encryption_time',
                                  title='Average Encryption Time by Library',
                                  labels={'encryption_time': 'Avg Time (seconds)'})
                    st.plotly_chart(fig1, use_container_width=True)

                with col_y:
                    fig2 = px.bar(library_stats, x='library', y='rate',
                                  title='Average Encryption Rate by Library',
                                  labels={'rate': 'Records/Second'})
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Encrypt data with multiple libraries to see comparison")

        # Download statistics
        stats_csv = df_stats.to_csv(index=False)
        st.download_button(
            "📥 Download Statistics (CSV)",
            stats_csv,
            file_name=f"encryption_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    else:
        st.info("ℹ️ No encryption statistics available. Encrypt data in Tab 2.")

    st.markdown("---")

    # Scheme Comparison Section
    st.subheader("🔬 Scheme Comparison Analysis")

    with st.expander("📖 Scheme Capabilities", expanded=False):
        for scheme_name, config in SCHEME_CONFIGS.items():
            st.markdown(f"### {scheme_name}")
            st.write(f"**Description:** {config['description']}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Supports:**")
                for data_type, support in config['supports'].items():
                    icon = "✅" if support == True else "⚠️" if support == 'limited' else "❌"
                    st.write(f"{icon} {data_type.capitalize()}: {support}")

            with col2:
                st.write("**Warnings:**")
                for warning in config['warnings']:
                    st.warning(f"⚠️ {warning}")

            st.markdown("---")

    # Restricted Data Summary
    if st.session_state.restricted_data_detected:
        st.subheader("🚨 Restricted Country Data Summary")

        with st.expander("⚠️ Restricted Data Details", expanded=True):
            if st.session_state.df_users is not None and 'is_restricted' in st.session_state.df_users.columns:
                restricted_users = st.session_state.df_users[st.session_state.df_users['is_restricted']]

                st.write(f"**Total Restricted Users:** {len(restricted_users)}")

                # Country breakdown
                country_counts = restricted_users['country'].value_counts()

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Restricted Countries:**")
                    st.dataframe(country_counts.reset_index().rename(
                        columns={'index': 'Country', 'country': 'Count'}),
                        use_container_width=True)

                with col2:
                    fig = px.pie(values=country_counts.values, names=country_counts.index,
                                 title='Restricted Users by Country')
                    st.plotly_chart(fig, use_container_width=True)

                st.error(
                    "⚠️ **Processing Requirement:** All data from these countries must be processed on-premises only")
                st.info("🔒 **Compliance:** Data sovereignty and privacy regulations enforced")

# == == == == == == == == == == Configuration == == == == == == == == == ==

st.set_page_config(
    page_title="FHE Financial Transaction Client",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state first
if 'server_url' not in st.session_state:
    st.session_state.server_url = "http://localhost:8000"

# Server configuration in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Server Configuration")
    SERVER_URL = st.text_input(
        "Server URL",
        value=st.session_state.server_url,
        key="server_url_input",
        help="URL of the FHE server (default: http://localhost:8000)"
    )
    if SERVER_URL != st.session_state.server_url:
        st.session_state.server_url = SERVER_URL

    # Use the session state value
    SERVER_URL = st.session_state.server_url

# Restricted countries
RESTRICTED_COUNTRIES = ['CN', 'TR', 'SA', 'KP', 'IR', 'RU', 'China', 'Turkey', 'Saudi Arabia']

# Scheme configurations
SCHEME_CONFIGS = {
    'CKKS': {
        'name': 'CKKS',
        'description': 'Approximate arithmetic on encrypted real numbers',
        'supports': {
            'numeric': True,
            'text': 'limited',
            'date': True
        },
        'params': ['poly_modulus_degree', 'scale', 'coeff_mod_bit_sizes'],
        'warnings': ['Approximate precision', 'No comparison operations']
    },
    'BFV': {
        'name': 'BFV',
        'description': 'Exact integer arithmetic',
        'supports': {
            'numeric': True,
            'text': False,
            'date': 'limited'
        },
        'params': ['poly_modulus_degree', 'plain_modulus'],
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
        'params': ['poly_modulus_degree', 'plain_modulus', 'mult_depth'],
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
        'restricted_data_detected': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def make_download_link(data, filename, mime_type="application/json"):
    """Create download link for data"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, indent=2)
    else:
        data_str = str(data)

    b64 = base64.b64encode(data_str.encode()).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'


def check_server_health():
    """Check if server is healthy"""
    try:
        response = httpx.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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
    # Use Python's random module to avoid numpy int32 overflow
    import random

    countries = ['USA', 'UK', 'Germany', 'France', 'Japan', 'Canada', 'Australia',
                 'China', 'Turkey', 'Saudi Arabia', 'India', 'Brazil', 'Mexico', 'Spain']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY', 'TRY', 'SAR', 'INR', 'BRL']
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


# ==================== Main Application ====================

def main():
    init_session_state()

    # Header
    st.title("🔐 FHE Financial Transaction Analysis System")
    st.markdown("### Secure homomorphic encryption for financial data processing")

    # Check server health
    if not check_server_health():
        st.error(f"⚠️ Cannot connect to server at {SERVER_URL}. Please ensure server is running.")
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

    # ==================== TAB 1: Data & Configuration ====================
    with tab1:
        render_data_configuration_tab()

    # ==================== TAB 2: Encryption & Analysis ====================
    with tab2:
        render_encryption_analysis_tab()

    # ==================== TAB 3: Results & Statistics ====================
    with tab3:
        render_results_statistics_tab()


# ==================== Tab 1: Data & Configuration ====================

def render_data_configuration_tab():
    st.header("📊 Data Management & Configuration")

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Generate Synthetic Data", "Upload CSV Files"],
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

                        # Check for restricted countries
                        restricted_count = df_users['is_restricted'].sum()
                        if restricted_count > 0:
                            st.session_state.restricted_data_detected = True
                            st.warning(f"⚠️ Detected {restricted_count} users from restricted countries")

                        st.success("✅ Synthetic data generated successfully!")

        else:  # Upload CSV
            with st.expander("📤 Upload CSV Files", expanded=True):
                st.info("Upload CSV files for users, accounts, and transactions")

                uploaded_files = st.file_uploader(
                    "Upload CSV files (users.csv, accounts.csv, transactions.csv)",
                    type=['csv'],
                    accept_multiple_files=True
                )

                if uploaded_files:
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        filename_lower = file.name.lower()

                        if 'user' in filename_lower:
                            st.session_state.df_users = df
                            st.success(f"✅ Loaded {len(df)} users")
                        elif 'account' in filename_lower:
                            st.session_state.df_accounts = df
                            st.success(f"✅ Loaded {len(df)} accounts")
                        elif 'transaction' in filename_lower or 'tx' in filename_lower:
                            st.session_state.df_transactions = df
                            st.success(f"✅ Loaded {len(df)} transactions")

    with col2:
        st.subheader("FHE Configuration")

        # Library selection
        library = st.selectbox(
            "FHE Library",
            ["TenSEAL", "OpenFHE"],
            key="library"
        )

        # Scheme selection
        available_schemes = ["CKKS", "BFV", "BGV"] if library == "OpenFHE" else ["CKKS", "BFV"]
        scheme = st.selectbox(
            "FHE Scheme",
            available_schemes,
            key="scheme"
        )

        # Display scheme info
        scheme_info = SCHEME_CONFIGS.get(scheme, {})
        st.info(f"**{scheme}**: {scheme_info.get('description', '')}")

        # Warnings
        warnings = scheme_info.get('warnings', [])
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")

        # Parameters
        with st.expander("⚙️ Advanced Parameters"):
            poly_modulus_degree = st.selectbox(
                "Polynomial Modulus Degree",
                [4096, 8192, 16384, 32768],
                index=1
            )

            if scheme == "CKKS":
                scale = st.selectbox(
                    "Scale",
                    [2 ** 30, 2 ** 40, 2 ** 50],
                    index=1
                )
                coeff_mod_bit_sizes = st.text_input(
                    "Coefficient Modulus Bit Sizes",
                    "60,40,40,60"
                )

            if library == "OpenFHE":
                mult_depth = st.number_input("Multiplicative Depth", min_value=1, max_value=20, value=10)
                scale_mod_size = st.number_input("Scaling Modulus Size", min_value=30, max_value=60, value=50)
                batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8)
                security_level = st.selectbox(
                    "Security Level",
                    ["HEStd_128_classic", "HEStd_192_classic", "HEStd_256_classic"]
                )

        # Generate keys
        if st.button("🔑 Generate Encryption Keys", type="primary"):
            with st.spinner("Generating keys on server..."):
                params = {
                    'poly_modulus_degree': poly_modulus_degree
                }

                if scheme == "CKKS":
                    params['scale'] = scale
                    params['coeff_mod_bit_sizes'] = [int(x.strip()) for x in coeff_mod_bit_sizes.split(',')]

                if library == "OpenFHE":
                    params['mult_depth'] = mult_depth
                    params['scale_mod_size'] = scale_mod_size
                    params['batch_size'] = batch_size
                    params['security_level'] = security_level

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

                    st.success("✅ Keys generated successfully!")
                    st.json({
                        'session_id': keys_info.get('session_id'),
                        'library': keys_info.get('library'),
                        'scheme': keys_info.get('scheme'),
                        'mode': keys_info.get('mode')
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

    # Display data tables
    st.markdown("---")
    st.subheader("📋 Data Tables")

    if st.session_state.df_users is not None:
        with st.expander(f"👥 Users Table ({len(st.session_state.df_users)} records)", expanded=False):
            st.dataframe(st.session_state.df_users, use_container_width=True)

            # Restricted country analysis
            if 'is_restricted' in st.session_state.df_users.columns:
                restricted_count = st.session_state.df_users['is_restricted'].sum()
                if restricted_count > 0:
                    st.warning(f"⚠️ {restricted_count} users from restricted countries detected")
                    restricted_countries = st.session_state.df_users[
                        st.session_state.df_users['is_restricted']
                    ]['country'].value_counts()
                    st.write("**Restricted Countries Distribution:**")
                    st.bar_chart(restricted_countries)

    if st.session_state.df_accounts is not None:
        with st.expander(f"💳 Accounts Table ({len(st.session_state.df_accounts)} records)", expanded=False):
            st.dataframe(st.session_state.df_accounts, use_container_width=True)

    if st.session_state.df_transactions is not None:
        with st.expander(f"💰 Transactions Table ({len(st.session_state.df_transactions)} records)", expanded=False):
            st.dataframe(st.session_state.df_transactions, use_container_width=True)


# ==================== Tab 2: Encryption & Analysis ====================

def render_encryption_analysis_tab():
    st.header("🔒 Data Encryption & Analysis")

    if st.session_state.keys_info is None:
        st.warning("⚠️ Please generate encryption keys in Tab 1 first")
        return

    if st.session_state.df_users is None:
        st.warning("⚠️ Please load or generate data in Tab 1 first")
        return

    # Column selection for encryption
    st.subheader("1️⃣ Select Columns for Encryption")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Users Table**")
        if st.session_state.df_users is not None:
            user_columns = list(st.session_state.df_users.columns)
            selected_user_cols = st.multiselect(
                "Select user columns:",
                user_columns,
                key="selected_user_cols"
            )
            st.session_state.selected_columns['users'] = selected_user_cols

    with col2:
        st.write("**Accounts Table**")
        if st.session_state.df_accounts is not None:
            account_columns = list(st.session_state.df_accounts.columns)
            selected_account_cols = st.multiselect(
                "Select account columns:",
                account_columns,
                key="selected_account_cols"
            )
            st.session_state.selected_columns['accounts'] = selected_account_cols

    with col3:
        st.write("**Transactions Table**")
        if st.session_state.df_transactions is not None:
            transaction_columns = list(st.session_state.df_transactions.columns)
            selected_transaction_cols = st.multiselect(
                "Select transaction columns:",
                transaction_columns,
                key="selected_transaction_cols"
            )
            st.session_state.selected_columns['transactions'] = selected_transaction_cols

    # Batch size configuration
    batch_size = st.slider(
        "Batch Size (rows per batch)",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

    # Encrypt button
    if st.button("🔐 Encrypt Selected Columns", type="primary"):
        encrypt_selected_columns(batch_size)

    st.markdown("---")

    # Analysis section
    st.subheader("2️⃣ Run FHE Analysis")

    if not st.session_state.encrypted_data:
        st.info("ℹ️ Encrypt data first to perform analysis")
        return

    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Query parameters
        st.write("**Analysis Parameters**")

        operation_type = st.selectbox(
            "Analysis Type",
            ["Transaction Analysis", "Transaction Count", "Account Summary", "Country Analysis"]
        )

        col_x, col_y = st.columns(2)
        with col_x:
            start_date = st.date_input(
                "Start Date",
                date.today() - timedelta(days=365)
            )
        with col_y:
            end_date = st.date_input(
                "End Date",
                date.today()
            )

        user_id_filter = st.text_input("User ID (optional)", "")

        # Currency filter
        if st.session_state.df_transactions is not None and 'currency' in st.session_state.df_transactions.columns:
            available_currencies = st.session_state.df_transactions['currency'].unique().tolist()
            selected_currencies = st.multiselect(
                "Filter by Currency",
                available_currencies,
                default=available_currencies[:3] if len(available_currencies) >= 3 else available_currencies
            )
        else:
            selected_currencies = ['USD', 'EUR', 'GBP']

        # Country filter
        country_filter = st.text_input("Country (optional)", "")

    with col_b:
        st.write("**Jurisdiction Information**")

        # Check if restricted data
        is_restricted = False
        if country_filter and country_filter in RESTRICTED_COUNTRIES:
            is_restricted = True
            st.error("🚨 RESTRICTED COUNTRY")
            st.warning("Data will be processed on-premises only")
        elif st.session_state.restricted_data_detected:
            st.warning("⚠️ Dataset contains restricted country data")
            is_restricted = st.checkbox("Process as restricted data", value=False)
        else:
            st.success("✅ Non-restricted processing")

    # Run analysis
    if st.button("▶️ Run Analysis", type="primary"):
        run_fhe_analysis(
            operation_type,
            start_date,
            end_date,
            user_id_filter,
            selected_currencies,
            country_filter,
            is_restricted
        )


def encrypt_selected_columns(batch_size):
    """Encrypt selected columns"""
    keys_info = st.session_state.keys_info
    session_id = keys_info.get('session_id')
    library = keys_info.get('library')
    scheme = keys_info.get('scheme')

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_columns = (
            len(st.session_state.selected_columns['users']) +
            len(st.session_state.selected_columns['accounts']) +
            len(st.session_state.selected_columns['transactions'])
    )

    if total_columns == 0:
        st.warning("⚠️ No columns selected for encryption")
        return

    current_column = 0
    all_encrypted_results = []  # Store results for display

    try:
        client = httpx.Client(timeout=120)

        # Encrypt each selected column
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

                # Get column data
                column_data = df[column].tolist()
                data_type = detect_data_type(df[column])

                # Check compatibility
                compatible, warning = check_scheme_compatibility(scheme, data_type)
                if not compatible:
                    st.warning(f"⚠️ Skipping {column}: {warning}")
                    continue

                # Encrypt in batches
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

                # Store encrypted data
                key = f"{table_name}:{column}"
                st.session_state.encrypted_data[key] = {
                    'encrypted_values': encrypted_values,
                    'data_type': data_type,
                    'library': library,
                    'scheme': scheme,
                    'original_count': len(column_data),
                    'encrypted_count': len(encrypted_values),
                    'encryption_time': encryption_time
                }

                # Prepare display data
                display_data = {
                    'Table': table_name,
                    'Column': column,
                    'Data Type': data_type,
                    'Original Count': len(column_data),
                    'Encrypted Count': len(encrypted_values),
                    'Time (sec)': round(encryption_time, 3),
                    'Rate (rec/sec)': round(len(column_data) / encryption_time, 2) if encryption_time > 0 else 0,
                    'Scheme': scheme,
                    'Library': library
                }
                all_encrypted_results.append(display_data)

                # Store statistics
                st.session_state.statistics.append({
                    'table': table_name,
                    'column': column,
                    'data_type': data_type,
                    'library': library,
                    'scheme': scheme,
                    'original_count': len(column_data),
                    'encrypted_count': len(encrypted_values),
                    'encryption_time': encryption_time,
                    'rate': len(column_data) / encryption_time if encryption_time > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                })

        client.close()
        progress_bar.progress(1.0)
        status_text.text("✅ Encryption complete!")

        # Display encrypted data summary
        if all_encrypted_results:
            st.success(f"✅ Successfully encrypted {len(all_encrypted_results)} columns")

            # Show summary table
            st.markdown("### 📊 Encryption Summary")
            df_summary = pd.DataFrame(all_encrypted_results)
            st.dataframe(df_summary, use_container_width=True)

            # Show detailed encrypted data for each column
            st.markdown("### 🔐 Encrypted Data Preview")

            # Add view mode selector
            view_mode = st.radio(
                "View Mode:",
                ["Side-by-Side Comparison", "Encrypted Only", "Summary Statistics"],
                horizontal=True,
                key="encrypted_view_mode"
            )

            for table_name, df_key in [('users', 'df_users'), ('accounts', 'df_accounts'),
                                       ('transactions', 'df_transactions')]:
                df = st.session_state.get(df_key)
                if df is None:
                    continue

                selected_cols = st.session_state.selected_columns[table_name]
                if not selected_cols:
                    continue

                with st.expander(f"📋 {table_name.capitalize()} - Encrypted Columns", expanded=True):
                    if view_mode == "Side-by-Side Comparison":
                        # Create display dataframe with original and encrypted
                        display_data = []

                        for idx in range(min(100, len(df))):  # Show first 100 rows
                            row_data = {'Row': idx + 1}

                            for column in selected_cols:
                                key = f"{table_name}:{column}"
                                if key in st.session_state.encrypted_data:
                                    encrypted_values = st.session_state.encrypted_data[key]['encrypted_values']
                                    if idx < len(encrypted_values):
                                        enc_val = encrypted_values[idx]

                                        # Original value
                                        original = df.iloc[idx][column]
                                        row_data[f"📄 {column}"] = str(original)[:50]

                                        # Encrypted value
                                        if enc_val is None:
                                            row_data[f"🔐 {column}"] = "NULL"
                                        elif isinstance(enc_val, dict):
                                            ciphertext = enc_val.get('ciphertext', 'N/A')
                                            if isinstance(ciphertext, str) and len(ciphertext) > 40:
                                                ciphertext = ciphertext[:40] + "..."
                                            row_data[f"🔐 {column}"] = ciphertext
                                        else:
                                            row_data[f"🔐 {column}"] = str(enc_val)[:40] + "..."

                            display_data.append(row_data)

                        if display_data:
                            df_display = pd.DataFrame(display_data)
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                height=400
                            )

                    elif view_mode == "Encrypted Only":
                        # Show only encrypted values
                        display_data = []

                        for idx in range(min(100, len(df))):
                            row_data = {'Row': idx + 1}

                            for column in selected_cols:
                                key = f"{table_name}:{column}"
                                if key in st.session_state.encrypted_data:
                                    encrypted_values = st.session_state.encrypted_data[key]['encrypted_values']
                                    if idx < len(encrypted_values):
                                        enc_val = encrypted_values[idx]

                                        if enc_val is None:
                                            row_data[f"🔐 {column}"] = "NULL"
                                        elif isinstance(enc_val, dict):
                                            # Show full ciphertext structure
                                            ciphertext = enc_val.get('ciphertext', 'N/A')
                                            enc_type = enc_val.get('type', 'N/A')
                                            row_data[f"🔐 {column}"] = f"{ciphertext[:50]}... (Type: {enc_type})"
                                        else:
                                            row_data[f"🔐 {column}"] = str(enc_val)[:50] + "..."

                            display_data.append(row_data)

                        if display_data:
                            df_display = pd.DataFrame(display_data)
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                height=400
                            )

                    else:  # Summary Statistics
                        # Show statistics per column
                        st.markdown("#### 📊 Encryption Statistics")

                        for column in selected_cols:
                            key = f"{table_name}:{column}"
                            if key in st.session_state.encrypted_data:
                                enc_info = st.session_state.encrypted_data[key]
                                encrypted_values = enc_info['encrypted_values']

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric(
                                        f"📝 {column}",
                                        f"{len(encrypted_values)} records"
                                    )

                                with col2:
                                    null_count = sum(1 for v in encrypted_values if v is None)
                                    st.metric(
                                        "Encrypted",
                                        f"{len(encrypted_values) - null_count}",
                                        delta=f"{null_count} nulls" if null_count > 0 else "0 nulls"
                                    )

                                with col3:
                                    st.metric(
                                        "Time",
                                        f"{enc_info['encryption_time']:.2f}s"
                                    )

                                with col4:
                                    rate = enc_info['original_count'] / enc_info['encryption_time'] if enc_info[
                                                                                                           'encryption_time'] > 0 else 0
                                    st.metric(
                                        "Rate",
                                        f"{rate:.1f} rec/s"
                                    )

                                # Show sample encrypted values
                                st.markdown(f"**Sample Encrypted Values for `{column}`:**")
                                sample_encrypted = []
                                for i, enc_val in enumerate(encrypted_values[:5]):
                                    if enc_val and isinstance(enc_val, dict):
                                        cipher = enc_val.get('ciphertext', 'N/A')
                                        sample_encrypted.append({
                                            'Index': i,
                                            'Ciphertext (truncated)': cipher[:60] + "..." if len(
                                                cipher) > 60 else cipher,
                                            'Type': enc_val.get('type', 'N/A'),
                                            'Library': enc_info['library'],
                                            'Scheme': enc_info['scheme']
                                        })

                                if sample_encrypted:
                                    st.dataframe(pd.DataFrame(sample_encrypted), use_container_width=True)

                                st.markdown("---")

                    st.info(f"ℹ️ Showing first 100 rows. Total encrypted: {len(df)} rows")

                    # Download encrypted data option
                    encrypted_json = {}
                    for column in selected_cols:
                        key = f"{table_name}:{column}"
                        if key in st.session_state.encrypted_data:
                            encrypted_json[column] = {
                                'encrypted_values': st.session_state.encrypted_data[key]['encrypted_values'],
                                'metadata': {
                                    'data_type': st.session_state.encrypted_data[key]['data_type'],
                                    'library': st.session_state.encrypted_data[key]['library'],
                                    'scheme': st.session_state.encrypted_data[key]['scheme'],
                                    'original_count': st.session_state.encrypted_data[key]['original_count'],
                                    'encrypted_count': st.session_state.encrypted_data[key]['encrypted_count'],
                                    'encryption_time': st.session_state.encrypted_data[key]['encryption_time']
                                }
                            }

                    if encrypted_json:
                        col1, col2 = st.columns(2)

                        with col1:
                            json_str = json.dumps(encrypted_json, indent=2)
                            st.download_button(
                                f"📥 Download {table_name.capitalize()} Encrypted Data (JSON)",
                                json_str,
                                file_name=f"encrypted_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                key=f"download_json_{table_name}"
                            )

                        with col2:
                            # Create CSV with encrypted data
                            csv_data = []
                            for idx in range(len(df)):
                                row = {'Row': idx + 1}
                                for column in selected_cols:
                                    key = f"{table_name}:{column}"
                                    if key in st.session_state.encrypted_data:
                                        encrypted_values = st.session_state.encrypted_data[key]['encrypted_values']
                                        if idx < len(encrypted_values):
                                            enc_val = encrypted_values[idx]
                                            if enc_val is None:
                                                row[f"{column}_encrypted"] = "NULL"
                                            elif isinstance(enc_val, dict):
                                                row[f"{column}_encrypted"] = enc_val.get('ciphertext', 'N/A')
                                            else:
                                                row[f"{column}_encrypted"] = str(enc_val)
                                csv_data.append(row)

                            if csv_data:
                                df_csv = pd.DataFrame(csv_data)
                                csv_str = df_csv.to_csv(index=False)
                                st.download_button(
                                    f"📥 Download {table_name.capitalize()} Encrypted Data (CSV)",
                                    csv_str,
                                    file_name=f"encrypted_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key=f"download_csv_{table_name}"
                                )

    except Exception as e:
        st.error(f"❌ Encryption error: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")


def run_fhe_analysis(operation_type, start_date, end_date, user_id_filter,
                     selected_currencies, country_filter, is_restricted):
    """Run FHE analysis on encrypted data"""
    keys_info = st.session_state.keys_info
    session_id = keys_info.get('session_id')
    library = keys_info.get('library')
    scheme = keys_info.get('scheme')

    with st.spinner("Running FHE analysis..."):
        try:
            # Prepare encrypted data
            encrypted_data_payload = {}
            for key, value in st.session_state.encrypted_data.items():
                encrypted_data_payload[key] = value['encrypted_values']

            # Query parameters
            query_params = {
                'operation_type': operation_type,
                'user_id': user_id_filter if user_id_filter else None,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'currencies': selected_currencies,
                'country': country_filter if country_filter else None,
                'is_restricted': is_restricted
            }

            # Send query to server
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

            # Display results
            st.success("✅ Analysis complete!")

            if is_restricted:
                st.error("🚨 RESTRICTED DATA - Results encrypted and processed on-premises")
                st.warning(results.get('compliance', 'Data processed per sovereignty requirements'))

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'total_transactions' in results:
                    st.metric("Total Transactions", results['total_transactions'])
            with col2:
                if 'total_amount' in results:
                    st.metric("Total Amount", f"${results['total_amount']:,.2f}")
            with col3:
                if 'avg_amount' in results:
                    st.metric("Average Amount", f"${results['avg_amount']:,.2f}")
            with col4:
                st.metric("Processing", results.get('processing_location', 'Unknown'))

        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")

# ==================== Run Application ====================

if __name__ == "__main__":
    main()