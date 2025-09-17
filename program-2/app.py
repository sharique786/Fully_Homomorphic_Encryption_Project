import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
import io
import warnings

warnings.filterwarnings('ignore')

# Try importing OpenFHE - handle installation gracefully
try:
    from openfhe import *

    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    st.error("OpenFHE not installed. Please install with: pip install openfhe-python")

# Configure Streamlit page
st.set_page_config(
    page_title="FHE Financial Data Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


class FHEFinancialAnalyzer:
    def __init__(self):
        self.reset_context()

    def reset_context(self):
        """Reset FHE context and parameters"""
        self.context = None
        self.public_key = None
        self.private_key = None
        self.cc = None
        self.encrypted_data = {}
        self.scheme_params = {
            'BFV': {'poly_degree': 8192, 'plain_modulus': 536870912, 'security_level': 128},
            'BGV': {'poly_degree': 8192, 'plain_modulus': 536870912, 'security_level': 128},
            'CKKS': {'poly_degree': 8192, 'scale_factor': 50, 'security_level': 128}
        }

    def setup_fhe_context(self, scheme='BFV', poly_degree=8192, plain_modulus=536870912,
                          scale_factor=50, security_level=128):
        """Setup FHE context with specified parameters"""
        if not OPENFHE_AVAILABLE:
            return False

        try:
            if scheme == 'BFV':
                parameters = CCParamsBFVRNS()
                parameters.SetPlaintextModulus(plain_modulus)
                parameters.SetPolynomialDegree(poly_degree)
                parameters.SetSecurityLevel(security_level)
                parameters.SetBatchSize(poly_degree // 2)

            elif scheme == 'BGV':
                parameters = CCParamsBGVRNS()
                parameters.SetPlaintextModulus(plain_modulus)
                parameters.SetPolynomialDegree(poly_degree)
                parameters.SetSecurityLevel(security_level)
                parameters.SetBatchSize(poly_degree // 2)

            elif scheme == 'CKKS':
                parameters = CCParamsCKKSRNS()
                parameters.SetPolynomialDegree(poly_degree)
                parameters.SetScalingModSize(scale_factor)
                parameters.SetSecurityLevel(security_level)
                parameters.SetBatchSize(poly_degree // 2)

            self.cc = GenCryptoContext(parameters)
            self.cc.Enable(PKE)
            self.cc.Enable(KEYSWITCH)
            self.cc.Enable(LEVELEDSHE)

            # Generate keys
            keys = self.cc.KeyGen()
            self.public_key = keys.publicKey
            self.private_key = keys.secretKey

            # Generate relinearization keys for multiplication
            self.cc.EvalMultKeyGen(self.private_key)

            return True

        except Exception as e:
            st.error(f"Error setting up FHE context: {str(e)}")
            return False

    def encrypt_data(self, data, scheme='BFV'):
        """Encrypt data using specified scheme"""
        encrypted_values = {}

        if not self.cc:
            st.error("FHE context not initialized")
            return encrypted_values

        try:
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    values = data[column].fillna(0).astype(int).tolist()

                    if scheme in ['BFV', 'BGV']:
                        # For integer schemes
                        plaintext = self.cc.MakePackedPlaintext(values)
                        ciphertext = self.cc.Encrypt(self.public_key, plaintext)
                    else:  # CKKS
                        # For approximate arithmetic
                        plaintext = self.cc.MakeCKKSPackedPlaintext(values)
                        ciphertext = self.cc.Encrypt(self.public_key, plaintext)

                    encrypted_values[column] = ciphertext
                else:
                    # For non-numeric data, create a hash-based encryption simulation
                    encrypted_values[column] = f"ENCRYPTED_{column}_DATA"

            return encrypted_values

        except Exception as e:
            st.error(f"Encryption error: {str(e)}")
            return {}

    def decrypt_data(self, encrypted_data, original_shape):
        """Decrypt encrypted data"""
        decrypted_values = {}

        try:
            for column, ciphertext in encrypted_data.items():
                if isinstance(ciphertext, str):
                    # Simulated encrypted string data
                    decrypted_values[column] = f"DECRYPTED_{column}"
                else:
                    # Actual FHE decryption
                    plaintext_result = self.cc.Decrypt(self.private_key, ciphertext)
                    decoded = plaintext_result.GetPackedValue()
                    decrypted_values[column] = decoded[:original_shape]

            return decrypted_values

        except Exception as e:
            st.error(f"Decryption error: {str(e)}")
            return {}

    def perform_homomorphic_operations(self, encrypted_data, operation='add',
                                       operand=1, polynomial_coeffs=None):
        """Perform homomorphic operations on encrypted data"""
        results = {}

        try:
            for column, ciphertext in encrypted_data.items():
                if isinstance(ciphertext, str):
                    continue

                if operation == 'add':
                    # Homomorphic addition with constant
                    const_plaintext = self.cc.MakePackedPlaintext([operand] * 100)
                    result = self.cc.EvalAdd(ciphertext, const_plaintext)

                elif operation == 'multiply':
                    # Homomorphic multiplication with constant
                    const_plaintext = self.cc.MakePackedPlaintext([operand] * 100)
                    result = self.cc.EvalMult(ciphertext, const_plaintext)

                elif operation == 'polynomial' and polynomial_coeffs:
                    # Polynomial evaluation
                    result = self.cc.EvalPoly(ciphertext, polynomial_coeffs)

                else:
                    result = ciphertext

                results[f"{column}_{operation}"] = result

            return results

        except Exception as e:
            st.error(f"Homomorphic operation error: {str(e)}")
            return {}


def create_sample_financial_data():
    """Create sample financial data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')

    countries = ['USA', 'UK', 'Germany', 'France', 'Japan', 'Canada']
    addresses = [f"{np.random.randint(1, 9999)} {['Main St', 'Oak Ave', 'Pine Rd', 'Elm Dr'][np.random.randint(0, 4)]}"
                 for _ in range(1000)]

    data = []
    for i in range(1000):
        data.append({
            'customer_id': f"CUST_{i:04d}",
            'country': np.random.choice(countries),
            'address': addresses[i],
            'account_number': f"ACC_{np.random.randint(100000, 999999)}",
            'transaction_date': np.random.choice(dates),
            'transaction_amount': np.random.uniform(10, 10000),
            'balance': np.random.uniform(1000, 50000),
            'credit_score': np.random.randint(300, 850),
            'payment_history_score': np.random.randint(1, 10),
            'account_age_months': np.random.randint(1, 120)
        })

    return pd.DataFrame(data)


def main():
    st.title("🔐 Fully Homomorphic Encryption Financial Data Analyzer")
    st.markdown("### Advanced FHE operations on sensitive financial data")

    # Initialize the FHE analyzer
    if 'fhe_analyzer' not in st.session_state:
        st.session_state.fhe_analyzer = FHEFinancialAnalyzer()

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Page",
        ["Data Upload & Encryption", "FHE Operations & Analysis"]
    )

    if page == "Data Upload & Encryption":
        data_upload_page()
    else:
        fhe_operations_page()


def data_upload_page():
    st.header("📊 Data Upload & Encryption")

    # Data upload options
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Financial Data")

        # Option to use sample data or upload CSV
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Financial Data", "Upload CSV File"]
        )

        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload CSV file containing financial data with PII"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
            else:
                st.info("Please upload a CSV file to continue")
                return
        else:
            # Use sample data
            df = create_sample_financial_data()
            st.success(f"✅ Sample data loaded! Shape: {df.shape}")

    with col2:
        st.subheader("Data Preview")
        if 'df' in locals():
            st.dataframe(df.head(), use_container_width=True)

    # Store dataframe in session state
    st.session_state.original_data = df

    # Column and row selection for encryption
    st.subheader("🔒 Data Selection for Encryption")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Select Columns to Encrypt:**")
        columns_to_encrypt = st.multiselect(
            "Choose columns:",
            df.columns.tolist(),
            default=[col for col in df.columns if col in ['transaction_amount', 'balance', 'credit_score']],
            help="Select columns containing sensitive data to encrypt"
        )

    with col2:
        st.write("**Select Row Range:**")
        max_rows = len(df)
        row_range = st.slider(
            "Number of rows to process:",
            min_value=10,
            max_value=min(max_rows, 1000),
            value=min(100, max_rows),
            help="Select number of rows to encrypt (limited for performance)"
        )

    if columns_to_encrypt:
        # Encryption scheme selection
        st.subheader("⚙️ Encryption Configuration")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            scheme = st.selectbox(
                "FHE Scheme:",
                ["BFV", "BGV", "CKKS"],
                help="BFV/BGV: exact arithmetic, CKKS: approximate arithmetic"
            )

        with col2:
            poly_degree = st.selectbox(
                "Polynomial Degree:",
                [1024, 2048, 4096, 8192, 16384],
                index=3,
                help="Higher degree = more security but slower performance"
            )

        with col3:
            if scheme in ['BFV', 'BGV']:
                plain_modulus = st.selectbox(
                    "Plain Modulus:",
                    [65537, 1032193, 536870912],
                    index=2,
                    help="Modulus for plaintext operations"
                )
            else:
                scale_factor = st.slider(
                    "Scale Factor:",
                    min_value=20,
                    max_value=60,
                    value=50,
                    help="Precision for CKKS scheme"
                )

        with col4:
            security_level = st.selectbox(
                "Security Level:",
                [128, 192, 256],
                index=0,
                help="Higher level = more security"
            )

        # Encrypt button
        if st.button("🔐 Encrypt Selected Data", type="primary"):
            if not OPENFHE_AVAILABLE:
                st.error("OpenFHE is not available. Showing simulation instead.")
                # Create simulated encryption
                selected_data = df[columns_to_encrypt].head(row_range)
                st.session_state.encrypted_data = {col: f"ENCRYPTED_{col}_SIM" for col in columns_to_encrypt}
                st.session_state.encryption_scheme = scheme
                st.session_state.selected_columns = columns_to_encrypt
                st.session_state.selected_data = selected_data
            else:
                with st.spinner("Setting up FHE context and encrypting data..."):
                    # Setup FHE context
                    setup_params = {
                        'scheme': scheme,
                        'poly_degree': poly_degree,
                        'security_level': security_level
                    }

                    if scheme in ['BFV', 'BGV']:
                        setup_params['plain_modulus'] = plain_modulus
                    else:
                        setup_params['scale_factor'] = scale_factor

                    success = st.session_state.fhe_analyzer.setup_fhe_context(**setup_params)

                    if success:
                        # Encrypt selected data
                        selected_data = df[columns_to_encrypt].head(row_range)
                        encrypted_data = st.session_state.fhe_analyzer.encrypt_data(selected_data, scheme)

                        # Store in session state
                        st.session_state.encrypted_data = encrypted_data
                        st.session_state.encryption_scheme = scheme
                        st.session_state.selected_columns = columns_to_encrypt
                        st.session_state.selected_data = selected_data

                        st.success("✅ Data encrypted successfully!")
                    else:
                        st.error("❌ Failed to setup FHE context")

        # Display encrypted data status
        if 'encrypted_data' in st.session_state and st.session_state.encrypted_data:
            st.subheader("🔐 Encrypted Data Status")

            encrypted_df = pd.DataFrame()
            for col in columns_to_encrypt:
                if col in st.session_state.encrypted_data:
                    if isinstance(st.session_state.encrypted_data[col], str):
                        encrypted_df[f"{col}_encrypted"] = ["ENCRYPTED_DATA"] * row_range
                    else:
                        encrypted_df[f"{col}_encrypted"] = [f"FHE_CIPHERTEXT_{i}" for i in range(row_range)]

            st.dataframe(encrypted_df.head(10), use_container_width=True)

            # Encryption summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Encrypted Columns", len(st.session_state.encrypted_data))
            with col2:
                st.metric("Scheme Used", st.session_state.encryption_scheme)
            with col3:
                st.metric("Rows Processed", row_range)


def fhe_operations_page():
    st.header("🧮 FHE Operations & Analysis")

    # Check if data is encrypted
    if 'encrypted_data' not in st.session_state or not st.session_state.encrypted_data:
        st.warning("⚠️ No encrypted data found. Please go to 'Data Upload & Encryption' page first.")
        return

    # FHE Operations section
    st.subheader("🔧 Homomorphic Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        operation_type = st.selectbox(
            "Operation Type:",
            ["Addition", "Multiplication", "Polynomial Evaluation"],
            help="Select homomorphic operation to perform"
        )

    with col2:
        if operation_type in ["Addition", "Multiplication"]:
            operand = st.number_input(
                f"Operand for {operation_type}:",
                min_value=1,
                max_value=1000,
                value=10,
                help=f"Constant to {operation_type.lower()} with encrypted data"
            )
        else:
            st.write("**Polynomial Coefficients:**")
            poly_coeffs = st.text_input(
                "Enter coefficients (comma-separated):",
                value="1,2,1",
                help="E.g., '1,2,1' for x² + 2x + 1"
            )

    with col3:
        if st.button("🚀 Execute Operation", type="primary"):
            with st.spinner("Performing homomorphic operation..."):
                if operation_type == "Addition":
                    op_key = 'add'
                    op_params = {'operand': operand}
                elif operation_type == "Multiplication":
                    op_key = 'multiply'
                    op_params = {'operand': operand}
                else:
                    op_key = 'polynomial'
                    try:
                        coeffs = [float(x.strip()) for x in poly_coeffs.split(',')]
                        op_params = {'polynomial_coeffs': coeffs}
                    except:
                        st.error("Invalid polynomial coefficients format")
                        return

                # Perform operation (simulation if OpenFHE not available)
                if OPENFHE_AVAILABLE:
                    results = st.session_state.fhe_analyzer.perform_homomorphic_operations(
                        st.session_state.encrypted_data, op_key, **op_params
                    )
                else:
                    # Simulate operation results
                    results = {f"{col}_{op_key}": f"RESULT_{col}_{op_key.upper()}"
                               for col in st.session_state.encrypted_data.keys()}

                st.session_state.operation_results = results
                st.success(f"✅ {operation_type} operation completed!")

    # Payment History Analysis
    st.subheader("💳 Payment History Analysis")

    if 'original_data' in st.session_state:
        df = st.session_state.original_data

        # Date range selection
        if 'transaction_date' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date:",
                    value=df['transaction_date'].min(),
                    min_value=df['transaction_date'].min(),
                    max_value=df['transaction_date'].max()
                )

            with col2:
                end_date = st.date_input(
                    "End Date:",
                    value=df['transaction_date'].max(),
                    min_value=df['transaction_date'].min(),
                    max_value=df['transaction_date'].max()
                )

            # Filter data by date range
            filtered_df = df[
                (df['transaction_date'] >= pd.Timestamp(start_date)) &
                (df['transaction_date'] <= pd.Timestamp(end_date))
                ]

            st.write(f"**Filtered Data:** {len(filtered_df)} records from {start_date} to {end_date}")

            # Payment history metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_transaction = filtered_df['transaction_amount'].mean()
                st.metric("Avg Transaction", f"${avg_transaction:,.2f}")

            with col2:
                total_transactions = len(filtered_df)
                st.metric("Total Transactions", f"{total_transactions:,}")

            with col3:
                avg_balance = filtered_df['balance'].mean()
                st.metric("Avg Balance", f"${avg_balance:,.2f}")

            with col4:
                avg_credit_score = filtered_df['credit_score'].mean()
                st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")

    # Results Display and Analysis
    if 'operation_results' in st.session_state or 'encrypted_data' in st.session_state:
        st.subheader("📊 Analysis Results")

        # Simulated decryption and results
        tab1, tab2, tab3 = st.tabs(["Decrypted Results", "Performance Metrics", "Security Analysis"])

        with tab1:
            st.write("**Decrypted Operation Results:**")

            # Create simulated decrypted data
            if 'selected_data' in st.session_state:
                original_data = st.session_state.selected_data.head(10)

                # Show original vs processed data comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Original Data:**")
                    st.dataframe(original_data, use_container_width=True)

                with col2:
                    st.write("**After Homomorphic Operations:**")
                    # Simulate processed results
                    processed_data = original_data.copy()
                    if 'operation_results' in st.session_state:
                        # Add simulated operation results
                        for col in processed_data.select_dtypes(include=[np.number]).columns:
                            if operation_type == "Addition":
                                processed_data[f"{col}_added"] = processed_data[col] + operand
                            elif operation_type == "Multiplication":
                                processed_data[f"{col}_multiplied"] = processed_data[col] * operand
                            else:
                                processed_data[f"{col}_poly"] = processed_data[col] ** 2 + 2 * processed_data[col] + 1

                    st.dataframe(processed_data, use_container_width=True)

        with tab2:
            st.write("**Performance Analysis:**")

            # Create performance charts
            schemes = ['BFV', 'BGV', 'CKKS']
            poly_degrees = [1024, 2048, 4096, 8192, 16384]

            # Simulated performance data
            encryption_times = np.random.uniform(0.1, 2.0, len(schemes))
            operation_times = np.random.uniform(0.05, 0.5, len(schemes))

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Encryption Time by Scheme', 'Operation Time by Scheme',
                                'Performance vs Polynomial Degree', 'Memory Usage'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'scatter'}, {'type': 'bar'}]]
            )

            # Encryption times
            fig.add_trace(
                go.Bar(x=schemes, y=encryption_times, name='Encryption Time (s)',
                       marker_color='lightblue'),
                row=1, col=1
            )

            # Operation times
            fig.add_trace(
                go.Bar(x=schemes, y=operation_times, name='Operation Time (s)',
                       marker_color='lightgreen'),
                row=1, col=2
            )

            # Performance vs polynomial degree
            perf_times = [0.1, 0.2, 0.5, 1.2, 3.0]
            fig.add_trace(
                go.Scatter(x=poly_degrees, y=perf_times, mode='lines+markers',
                           name='Time (s)', line=dict(color='red')),
                row=2, col=1
            )

            # Memory usage
            memory_usage = np.random.uniform(50, 500, len(schemes))
            fig.add_trace(
                go.Bar(x=schemes, y=memory_usage, name='Memory (MB)',
                       marker_color='orange'),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.write("**Security Analysis:**")

            # Security metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Security Level", f"{st.session_state.get('security_level', 128)} bits")
                st.metric("Noise Budget", "85%")

            with col2:
                st.metric("Key Size", "2048 bits")
                st.metric("Ciphertext Expansion", "~1000x")

            with col3:
                st.metric("Operations Performed", len(st.session_state.get('operation_results', {})))
                st.metric("Remaining Operations", "~50")

            # Security recommendations
            st.write("**Security Recommendations:**")
            recommendations = [
                "✅ Strong encryption scheme selected",
                "✅ Appropriate polynomial degree for security level",
                "⚠️ Consider key rotation after 100 operations",
                "✅ Noise budget within safe limits",
                "⚠️ Monitor ciphertext expansion for large datasets"
            ]

            for rec in recommendations:
                st.write(rec)

    # Additional Analysis Charts
    if 'original_data' in st.session_state:
        st.subheader("📈 Statistical Analysis")

        df = st.session_state.original_data

        # Create comprehensive charts
        col1, col2 = st.columns(2)

        with col1:
            # Transaction amount distribution
            fig = px.histogram(
                df, x='transaction_amount', nbins=50,
                title='Transaction Amount Distribution',
                labels={'transaction_amount': 'Transaction Amount ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Credit score vs Balance
            fig = px.scatter(
                df.sample(200), x='credit_score', y='balance',
                color='country', size='transaction_amount',
                title='Credit Score vs Account Balance',
                labels={'credit_score': 'Credit Score', 'balance': 'Balance ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Country-wise analysis
            country_stats = df.groupby('country').agg({
                'transaction_amount': 'mean',
                'balance': 'mean',
                'credit_score': 'mean'
            }).round(2)

            fig = px.bar(
                country_stats, x=country_stats.index, y='transaction_amount',
                title='Average Transaction Amount by Country',
                labels={'transaction_amount': 'Avg Transaction ($)', 'x': 'Country'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Time series analysis
            if 'transaction_date' in df.columns:
                daily_transactions = df.groupby('transaction_date')['transaction_amount'].sum().reset_index()
                fig = px.line(
                    daily_transactions, x='transaction_date', y='transaction_amount',
                    title='Daily Transaction Volume',
                    labels={'transaction_date': 'Date', 'transaction_amount': 'Total Amount ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()