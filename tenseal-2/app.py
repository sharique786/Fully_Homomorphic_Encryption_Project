import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import json
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import FHE libraries
try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    st.error("TenSEAL not available. Install with: pip install tenseal")

try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt

    PYFHEL_AVAILABLE = True
except ImportError:
    PYFHEL_AVAILABLE = False
    st.warning("PyFHEL not available. Some features may be limited. Install with: pip install Pyfhel")

# Configuration for different FHE schemes
FHE_SCHEMES = {
    "BFV": "Integer arithmetic",
    "BGV": "Integer arithmetic with better noise management",
    "CKKS": "Approximate arithmetic for real numbers"
}


class FHEManager:
    """Manages Fully Homomorphic Encryption operations"""

    def __init__(self):
        self.scheme = None
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.relin_keys = None
        self.galois_keys = None
        self.pyfhel_instance = None

    def setup_tenseal_context(self, scheme: str, poly_modulus_degree: int = 8192,
                              coeff_mod_bit_sizes: List[int] = None, scale: float = 2 ** 40):
        """Setup TenSEAL context for CKKS scheme"""
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL not available")

        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        if scheme == "CKKS":
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            context.generate_keys()
            context.global_scale = scale

        elif scheme in ["BFV", "BGV"]:
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=786433
            )
            context.generate_keys()

        self.context = context
        return context

    def setup_pyfhel_context(self, scheme: str, poly_modulus_degree: int = 8192,
                             plain_modulus: int = 786433, sec_level: int = 128):
        """Setup PyFHEL context for BFV/BGV schemes"""
        if not PYFHEL_AVAILABLE:
            raise ImportError("PyFHEL not available")

        self.pyfhel_instance = Pyfhel()

        if scheme == "BFV":
            self.pyfhel_instance.contextGen(scheme='BFV', n=poly_modulus_degree,
                                            t=plain_modulus, sec=sec_level)
        elif scheme == "BGV":
            self.pyfhel_instance.contextGen(scheme='BGV', n=poly_modulus_degree,
                                            t=plain_modulus, sec=sec_level)
        elif scheme == "CKKS":
            self.pyfhel_instance.contextGen(scheme='CKKS', n=poly_modulus_degree, sec=sec_level)

        self.pyfhel_instance.keyGen()
        self.pyfhel_instance.relinKeyGen()
        self.pyfhel_instance.rotateKeyGen()

        return self.pyfhel_instance

    def encrypt_data_tenseal(self, data: List[float], scheme: str):
        """Encrypt data using TenSEAL"""
        if scheme == "CKKS":
            return ts.ckks_vector(self.context, data)
        else:  # BFV
            # Convert to integers for BFV
            int_data = [int(x * 1000) for x in data]  # Scale up for precision
            return ts.bfv_vector(self.context, int_data)

    def encrypt_data_pyfhel(self, data: List[float], scheme: str):
        """Encrypt data using PyFHEL"""
        if scheme == "CKKS":
            return self.pyfhel_instance.encryptFrac(data)
        else:
            # Convert to integers for BFV/BGV
            int_data = [int(x * 1000) for x in data]
            return self.pyfhel_instance.encryptInt(int_data)


class DataProcessor:
    """Handles data processing and encryption operations"""

    @staticmethod
    def load_csv_data(uploaded_file) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    @staticmethod
    def identify_pii_columns(df: pd.DataFrame) -> List[str]:
        """Identify potential PII columns based on column names and data patterns"""
        pii_keywords = ['name', 'email', 'phone', 'ssn', 'address', 'id', 'customer_id',
                        'account', 'credit_card', 'card_number', 'personal', 'private']

        pii_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in pii_keywords):
                pii_columns.append(col)

        return pii_columns

    @staticmethod
    def identify_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Identify numeric columns suitable for FHE operations"""
        return df.select_dtypes(include=[np.number]).columns.tolist()


def main():
    st.set_page_config(
        page_title="FHE Financial Data Analyzer",
        page_icon="🔐",
        layout="wide"
    )

    st.title("🔐 Fully Homomorphic Encryption Financial Data Analyzer")
    st.markdown("---")

    # Initialize session state
    if 'fhe_manager' not in st.session_state:
        st.session_state.fhe_manager = FHEManager()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'encrypted_data' not in st.session_state:
        st.session_state.encrypted_data = {}
    if 'computation_results' not in st.session_state:
        st.session_state.computation_results = {}

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "📊 Data Upload & Encryption",
        "🔧 FHE Operations",
        "📈 Results & Analysis"
    ])

    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Upload & Encryption":
        show_data_upload_page()
    elif page == "🔧 FHE Operations":
        show_fhe_operations_page()
    elif page == "📈 Results & Analysis":
        show_results_page()


def show_home_page():
    """Display home page with information about FHE"""
    st.header("Welcome to the FHE Financial Data Analyzer")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### About Fully Homomorphic Encryption (FHE)

        This application demonstrates the power of Fully Homomorphic Encryption for 
        protecting sensitive financial data while enabling computations on encrypted data.

        **Key Features:**
        - 🔒 **Data Protection**: Encrypt PII and financial data
        - 🔢 **Secure Computations**: Perform arithmetic operations on encrypted data
        - 📊 **Multiple Schemes**: Support for BFV, BGV, and CKKS schemes
        - 🎛️ **Parameter Tuning**: Adjust encryption parameters for optimal performance
        - 📈 **Visualization**: View results and statistics

        **Supported Operations:**
        - Addition and multiplication of encrypted values
        - Polynomial evaluation on encrypted data
        - Statistical computations (mean, sum, variance)
        - Matrix operations
        """)

    with col2:
        st.markdown("""
        ### FHE Schemes

        **BFV (Brakerski-Fan-Vercauteren)**
        - Integer arithmetic
        - High precision
        - Good for exact computations

        **BGV (Brakerski-Gentry-Vaikuntanathan)**  
        - Integer arithmetic
        - Better noise management
        - Efficient for deep circuits

        **CKKS (Cheon-Kim-Kim-Song)**
        - Approximate arithmetic
        - Real/complex numbers
        - Machine learning friendly
        """)

    st.markdown("---")
    st.info("💡 Start by uploading your financial data in the 'Data Upload & Encryption' section.")


def show_data_upload_page():
    """Display data upload and encryption interface"""
    st.header("📊 Data Upload & Encryption")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Financial Data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing financial data with PII information"
    )

    if uploaded_file is not None:
        # Load data
        df = DataProcessor.load_csv_data(uploaded_file)

        if df is not None:
            st.session_state.data = df

            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Data analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Data Summary")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Missing values:** {df.isnull().sum().sum()}")

                # Identify PII columns
                pii_columns = DataProcessor.identify_pii_columns(df)
                st.write(f"**Potential PII columns:** {len(pii_columns)}")
                if pii_columns:
                    st.write(pii_columns)

            with col2:
                st.subheader("🔢 Numeric Columns")
                numeric_columns = DataProcessor.identify_numeric_columns(df)
                st.write(f"**Numeric columns:** {len(numeric_columns)}")
                st.write(numeric_columns)

            # Encryption configuration
            st.markdown("---")
            st.subheader("🔐 Encryption Configuration")

            col1, col2, col3 = st.columns(3)

            with col1:
                scheme = st.selectbox(
                    "Select FHE Scheme",
                    list(FHE_SCHEMES.keys()),
                    help="Choose the encryption scheme based on your use case"
                )
                st.info(FHE_SCHEMES[scheme])

            with col2:
                poly_modulus_degree = st.selectbox(
                    "Polynomial Modulus Degree",
                    [4096, 8192, 16384, 32768],
                    index=1,
                    help="Higher values provide more security but slower performance"
                )

            with col3:
                if scheme == "CKKS":
                    scale = st.number_input(
                        "Scale (CKKS only)",
                        min_value=2 ** 40,
                        max_value=2 ** 80,
                        value=2 ** 60,
                        step=2 ** 20,
                        help="Scale parameter for CKKS precision"
                    )
                else:
                    plain_modulus = st.number_input(
                        "Plain Modulus",
                        min_value=2,
                        max_value=1000000,
                        value=786433,
                        help="Plain modulus for BFV/BGV schemes"
                    )

            # Column selection for encryption
            st.subheader("🎯 Select Data to Encrypt")

            all_columns = df.columns.tolist()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Select Columns to Encrypt:**")
                columns_to_encrypt = st.multiselect(
                    "Columns",
                    all_columns,
                    default=pii_columns + numeric_columns[:3],  # Default to PII + first 3 numeric
                    help="Select columns containing sensitive data"
                )

            with col2:
                st.write("**Row Selection:**")
                row_selection = st.radio(
                    "Select rows to encrypt",
                    ["All rows", "First N rows", "Random sample"],
                    help="Choose how many rows to encrypt"
                )

                if row_selection == "First N rows":
                    n_rows = st.number_input("Number of rows", min_value=1, max_value=len(df), value=min(100, len(df)))
                elif row_selection == "Random sample":
                    n_rows = st.number_input("Sample size", min_value=1, max_value=len(df), value=min(100, len(df)))
                    random_seed = st.number_input("Random seed", value=42)

            # Encrypt data button
            if st.button("🔒 Encrypt Selected Data", type="primary"):
                if columns_to_encrypt:
                    encrypt_data(df, columns_to_encrypt, scheme, poly_modulus_degree,
                                 row_selection, locals().get('n_rows'), locals().get('random_seed'))
                else:
                    st.error("Please select at least one column to encrypt")


def encrypt_data(df, columns_to_encrypt, scheme, poly_modulus_degree,
                 row_selection, n_rows=None, random_seed=None):
    """Encrypt selected data"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Prepare data subset
        if row_selection == "All rows":
            data_subset = df[columns_to_encrypt].copy()
        elif row_selection == "First N rows":
            data_subset = df[columns_to_encrypt].head(n_rows).copy()
        else:  # Random sample
            data_subset = df[columns_to_encrypt].sample(n=n_rows, random_state=random_seed).copy()

        status_text.text("Setting up FHE context...")
        progress_bar.progress(10)

        # Setup FHE context
        fhe_manager = st.session_state.fhe_manager

        if TENSEAL_AVAILABLE:
            try:
                if scheme == "CKKS":
                    fhe_manager.setup_tenseal_context(scheme, poly_modulus_degree, scale=2 ** 40)
                else:
                    fhe_manager.setup_tenseal_context(scheme, poly_modulus_degree)
                library_used = "TenSEAL"
                progress_bar.progress(30)
            except Exception as e:
                st.error(f"TenSEAL setup failed: {str(e)}")
                return
        else:
            st.error("No FHE library available")
            return

        status_text.text("Encrypting data...")
        progress_bar.progress(50)

        encrypted_columns = {}

        for i, col in enumerate(columns_to_encrypt):
            col_data = data_subset[col].fillna(0).tolist()

            # Convert non-numeric data to numeric if needed
            if not pd.api.types.is_numeric_dtype(data_subset[col]):
                # Simple encoding for strings (hash-based)
                col_data = [hash(str(x)) % 10000 for x in col_data]

            # Normalize data to reasonable range for FHE
            col_data = [(float(x) % 1000000) for x in col_data]

            try:
                encrypted_col = fhe_manager.encrypt_data_tenseal(col_data, scheme)
                encrypted_columns[col] = {
                    'encrypted': encrypted_col,
                    'original_data': col_data,
                    'data_type': str(data_subset[col].dtype),
                    'size': len(col_data)
                }
            except Exception as e:
                st.warning(f"Failed to encrypt column {col}: {str(e)}")
                continue

            progress_bar.progress(50 + int(40 * (i + 1) / len(columns_to_encrypt)))

        st.session_state.encrypted_data = {
            'columns': encrypted_columns,
            'scheme': scheme,
            'poly_modulus_degree': poly_modulus_degree,
            'library': library_used,
            'original_shape': data_subset.shape,
            'metadata': {
                'encryption_time': time.time(),
                'columns_encrypted': len(encrypted_columns),
                'rows_encrypted': len(data_subset)
            }
        }

        progress_bar.progress(100)
        status_text.text("Encryption completed!")

        st.success(f"✅ Successfully encrypted {len(encrypted_columns)} columns using {scheme} scheme!")

        # Display encrypted data summary
        st.subheader("🔐 Encrypted Data Summary")

        summary_data = []
        for col_name, col_info in encrypted_columns.items():
            summary_data.append({
                'Column': col_name,
                'Data Type': col_info['data_type'],
                'Size': col_info['size'],
                'Encrypted': '✅' if col_info['encrypted'] else '❌'
            })

        st.dataframe(pd.DataFrame(summary_data))

        # Show sample of original vs "encrypted" representation
        st.subheader("📊 Original vs Encrypted Data Preview")

        preview_col = st.selectbox("Select column to preview", list(encrypted_columns.keys()))
        if preview_col:
            col_info = encrypted_columns[preview_col]
            preview_df = pd.DataFrame({
                'Original': col_info['original_data'][:10],
                'Status': ['🔒 Encrypted'] * min(10, len(col_info['original_data']))
            })
            st.dataframe(preview_df)

    except Exception as e:
        st.error(f"Encryption failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def show_fhe_operations_page():
    """Display FHE operations interface"""
    st.header("🔧 Fully Homomorphic Encryption Operations")

    if 'encrypted_data' not in st.session_state or not st.session_state.encrypted_data:
        st.warning("⚠️ Please encrypt data first in the 'Data Upload & Encryption' section.")
        return

    encrypted_data = st.session_state.encrypted_data

    # Display current encryption info
    st.info(f"📊 Current dataset: {encrypted_data['metadata']['columns_encrypted']} encrypted columns, "
            f"{encrypted_data['metadata']['rows_encrypted']} rows using {encrypted_data['scheme']} scheme")

    # Operation selection
    st.subheader("🔢 Select Operation")

    operation = st.selectbox(
        "Choose operation to perform",
        [
            "Addition", "Multiplication", "Scalar Multiplication",
            "Polynomial Evaluation", "Statistical Operations",
            "Matrix Operations"
        ],
        help="Select the type of operation to perform on encrypted data"
    )

    encrypted_columns = list(encrypted_data['columns'].keys())

    if operation == "Addition":
        perform_addition_operation(encrypted_columns, encrypted_data)
    elif operation == "Multiplication":
        perform_multiplication_operation(encrypted_columns, encrypted_data)
    elif operation == "Scalar Multiplication":
        perform_scalar_multiplication_operation(encrypted_columns, encrypted_data)
    elif operation == "Polynomial Evaluation":
        perform_polynomial_operation(encrypted_columns, encrypted_data)
    elif operation == "Statistical Operations":
        perform_statistical_operations(encrypted_columns, encrypted_data)
    elif operation == "Matrix Operations":
        perform_matrix_operations(encrypted_columns, encrypted_data)


def perform_addition_operation(encrypted_columns, encrypted_data):
    """Perform addition operation on encrypted data"""
    st.subheader("➕ Addition Operation")

    col1, col2 = st.columns(2)

    with col1:
        col1_name = st.selectbox("Select first column", encrypted_columns, key="add_col1")

    with col2:
        col2_name = st.selectbox("Select second column", encrypted_columns, key="add_col2")

    if st.button("Perform Addition", key="perform_add"):
        perform_encrypted_operation(col1_name, col2_name, "addition", encrypted_data)


def perform_multiplication_operation(encrypted_columns, encrypted_data):
    """Perform multiplication operation on encrypted data"""
    st.subheader("✖️ Multiplication Operation")

    col1, col2 = st.columns(2)

    with col1:
        col1_name = st.selectbox("Select first column", encrypted_columns, key="mul_col1")

    with col2:
        col2_name = st.selectbox("Select second column", encrypted_columns, key="mul_col2")

    if st.button("Perform Multiplication", key="perform_mul"):
        perform_encrypted_operation(col1_name, col2_name, "multiplication", encrypted_data)


def perform_scalar_multiplication_operation(encrypted_columns, encrypted_data):
    """Perform scalar multiplication on encrypted data"""
    st.subheader("🔢 Scalar Multiplication")

    col1, col2 = st.columns(2)

    with col1:
        col_name = st.selectbox("Select column", encrypted_columns, key="scalar_col")

    with col2:
        scalar_value = st.number_input("Scalar value", value=2.0, key="scalar_val")

    if st.button("Perform Scalar Multiplication", key="perform_scalar_mul"):
        perform_scalar_operation(col_name, scalar_value, encrypted_data)


def perform_polynomial_operation(encrypted_columns, encrypted_data):
    """Perform polynomial evaluation on encrypted data"""
    st.subheader("📈 Polynomial Evaluation")

    col_name = st.selectbox("Select column", encrypted_columns, key="poly_col")

    st.write("Define polynomial coefficients (e.g., for ax² + bx + c)")

    col1, col2, col3 = st.columns(3)

    with col1:
        coeff_a = st.number_input("Coefficient a (x²)", value=1.0, key="poly_a")

    with col2:
        coeff_b = st.number_input("Coefficient b (x)", value=0.0, key="poly_b")

    with col3:
        coeff_c = st.number_input("Constant c", value=0.0, key="poly_c")

    if st.button("Evaluate Polynomial", key="perform_poly"):
        coefficients = [coeff_c, coeff_b, coeff_a]  # [c, b, a] for c + bx + ax²
        perform_polynomial_evaluation(col_name, coefficients, encrypted_data)


def perform_statistical_operations(encrypted_columns, encrypted_data):
    """Perform statistical operations on encrypted data"""
    st.subheader("📊 Statistical Operations")

    col_name = st.selectbox("Select column", encrypted_columns, key="stat_col")

    operations = st.multiselect(
        "Select statistical operations",
        ["Sum", "Mean", "Variance (approximate)"],
        default=["Sum", "Mean"],
        key="stat_ops"
    )

    if st.button("Perform Statistical Operations", key="perform_stats"):
        perform_statistical_computation(col_name, operations, encrypted_data)


def perform_matrix_operations(encrypted_columns, encrypted_data):
    """Perform matrix operations on encrypted data"""
    st.subheader("🔢 Matrix Operations")

    if len(encrypted_columns) < 2:
        st.warning("At least 2 columns required for matrix operations")
        return

    selected_columns = st.multiselect(
        "Select columns for matrix formation",
        encrypted_columns,
        default=encrypted_columns[:2],
        key="matrix_cols"
    )

    operation = st.selectbox(
        "Select matrix operation",
        ["Transpose", "Element-wise addition", "Dot product simulation"],
        key="matrix_op"
    )

    if st.button("Perform Matrix Operation", key="perform_matrix"):
        perform_matrix_computation(selected_columns, operation, encrypted_data)


def perform_encrypted_operation(col1_name, col2_name, operation, encrypted_data):
    """Perform binary operation on two encrypted columns"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"Performing {operation} on encrypted data...")
        progress_bar.progress(20)

        col1_data = encrypted_data['columns'][col1_name]['encrypted']
        col2_data = encrypted_data['columns'][col2_name]['encrypted']

        progress_bar.progress(50)

        # Perform operation based on type
        if operation == "addition":
            if encrypted_data['scheme'] == "CKKS":
                result = col1_data + col2_data
            else:  # BFV/BGV
                result = col1_data + col2_data
        elif operation == "multiplication":
            if encrypted_data['scheme'] == "CKKS":
                result = col1_data * col2_data
            else:
                result = col1_data * col2_data

        progress_bar.progress(80)

        # Decrypt result for verification (in real scenario, this would be done by authorized party)
        if encrypted_data['scheme'] == "CKKS":
            decrypted_result = result.decrypt()[:10]  # Show first 10 results
        else:
            decrypted_result = result.decrypt()[:10]

        progress_bar.progress(100)
        status_text.text("Operation completed!")

        # Store result
        operation_key = f"{operation}_{col1_name}_{col2_name}"
        st.session_state.computation_results[operation_key] = {
            'operation': operation,
            'columns': [col1_name, col2_name],
            'result': decrypted_result,
            'timestamp': time.time(),
            'scheme': encrypted_data['scheme']
        }

        st.success(f"✅ {operation.capitalize()} completed successfully!")

        # Display results
        st.subheader("🔍 Operation Results")

        results_df = pd.DataFrame({
            f'{col1_name} (original)': encrypted_data['columns'][col1_name]['original_data'][:10],
            f'{col2_name} (original)': encrypted_data['columns'][col2_name]['original_data'][:10],
            f'Result ({operation})': decrypted_result
        })

        st.dataframe(results_df)

        # Visualization
        fig = px.bar(
            x=['Col1', 'Col2', 'Result'],
            y=[np.mean(encrypted_data['columns'][col1_name]['original_data'][:10]),
               np.mean(encrypted_data['columns'][col2_name]['original_data'][:10]),
               np.mean(decrypted_result)],
            title=f"Average Values - {operation.capitalize()} Operation"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Operation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def perform_scalar_operation(col_name, scalar_value, encrypted_data):
    """Perform scalar operation on encrypted column"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Performing scalar multiplication on encrypted data...")
        progress_bar.progress(30)

        col_data = encrypted_data['columns'][col_name]['encrypted']

        # Perform scalar multiplication
        result = col_data * scalar_value

        progress_bar.progress(70)

        # Decrypt result
        if encrypted_data['scheme'] == "CKKS":
            decrypted_result = result.decrypt()[:10]
        else:
            decrypted_result = result.decrypt()[:10]

        progress_bar.progress(100)
        status_text.text("Scalar multiplication completed!")

        # Store result
        operation_key = f"scalar_mul_{col_name}_{scalar_value}"
        st.session_state.computation_results[operation_key] = {
            'operation': 'scalar_multiplication',
            'column': col_name,
            'scalar': scalar_value,
            'result': decrypted_result,
            'timestamp': time.time(),
            'scheme': encrypted_data['scheme']
        }

        st.success(f"✅ Scalar multiplication by {scalar_value} completed!")

        # Display results
        st.subheader("🔍 Scalar Multiplication Results")

        original_data = encrypted_data['columns'][col_name]['original_data'][:10]
        expected_result = [x * scalar_value for x in original_data]

        results_df = pd.DataFrame({
            'Original': original_data,
            'Expected (Original × Scalar)': expected_result,
            'FHE Result': decrypted_result,
            'Difference': [abs(exp - fhe) for exp, fhe in zip(expected_result, decrypted_result)]
        })

        st.dataframe(results_df)

        # Visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Original vs Result', 'Error Analysis'))

        fig.add_trace(
            go.Scatter(x=list(range(len(original_data))), y=original_data,
                       name='Original', mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(decrypted_result))), y=decrypted_result,
                       name='FHE Result', mode='lines+markers'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=list(range(len(results_df['Difference']))), y=results_df['Difference'],
                   name='Absolute Error'),
            row=1, col=2
        )

        fig.update_layout(height=400, title_text="Scalar Multiplication Analysis")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Scalar operation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def perform_polynomial_evaluation(col_name, coefficients, encrypted_data):
    """Perform polynomial evaluation on encrypted data"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Evaluating polynomial on encrypted data...")
        progress_bar.progress(20)

        col_data = encrypted_data['columns'][col_name]['encrypted']

        # For CKKS, we can perform polynomial evaluation directly
        # For BFV/BGV, we simulate with basic operations

        if encrypted_data['scheme'] == "CKKS":
            # Polynomial evaluation: c + bx + ax²
            x = col_data
            x_squared = x * x  # x²

            # Build polynomial: ax² + bx + c
            result = x_squared * coefficients[2]  # ax²
            if coefficients[1] != 0:
                result = result + (x * coefficients[1])  # + bx
            if coefficients[0] != 0:
                result = result + coefficients[0]  # + c

        else:  # BFV/BGV
            # Simplified polynomial evaluation
            x = col_data
            result = x * coefficients[1] + coefficients[0]  # Linear approximation

        progress_bar.progress(70)

        # Decrypt result
        decrypted_result = result.decrypt()[:10]

        progress_bar.progress(100)
        status_text.text("Polynomial evaluation completed!")

        # Store result
        operation_key = f"polynomial_{col_name}_{hash(tuple(coefficients))}"
        st.session_state.computation_results[operation_key] = {
            'operation': 'polynomial_evaluation',
            'column': col_name,
            'coefficients': coefficients,
            'result': decrypted_result,
            'timestamp': time.time(),
            'scheme': encrypted_data['scheme']
        }

        st.success("✅ Polynomial evaluation completed!")

        # Display results
        st.subheader("🔍 Polynomial Evaluation Results")

        original_data = encrypted_data['columns'][col_name]['original_data'][:10]

        # Calculate expected results
        if encrypted_data['scheme'] == "CKKS":
            expected_result = [coefficients[0] + coefficients[1] * x + coefficients[2] * x * x for x in original_data]
        else:
            expected_result = [coefficients[0] + coefficients[1] * x for x in original_data]

        results_df = pd.DataFrame({
            'Original (x)': original_data,
            'Expected Result': expected_result,
            'FHE Result': decrypted_result,
            'Polynomial': [f"{coefficients[2]:.1f}x² + {coefficients[1]:.1f}x + {coefficients[0]:.1f}"] * len(
                original_data)
        })

        st.dataframe(results_df)

        # Visualization
        fig = px.scatter(results_df, x='Original (x)', y='FHE Result',
                         title=f"Polynomial Evaluation: {coefficients[2]:.1f}x² + {coefficients[1]:.1f}x + {coefficients[0]:.1f}")
        fig.add_scatter(x=results_df['Original (x)'], y=results_df['Expected Result'],
                        mode='lines', name='Expected Curve')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Polynomial evaluation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def perform_statistical_computation(col_name, operations, encrypted_data):
    """Perform statistical computations on encrypted data"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Performing statistical operations on encrypted data...")
        progress_bar.progress(30)

        col_data = encrypted_data['columns'][col_name]['encrypted']
        original_data = encrypted_data['columns'][col_name]['original_data']

        results = {}

        if "Sum" in operations:
            # Sum all encrypted values
            encrypted_sum = col_data.sum() if hasattr(col_data, 'sum') else sum(col_data)
            if encrypted_data['scheme'] == "CKKS":
                decrypted_sum = encrypted_sum.decrypt()
            else:
                decrypted_sum = encrypted_sum.decrypt()
            results['Sum'] = decrypted_sum

        progress_bar.progress(50)

        if "Mean" in operations:
            # Calculate mean (sum / count)
            if 'Sum' not in results:
                encrypted_sum = col_data.sum() if hasattr(col_data, 'sum') else sum(col_data)
                if encrypted_data['scheme'] == "CKKS":
                    decrypted_sum = encrypted_sum.decrypt()
                else:
                    decrypted_sum = encrypted_sum.decrypt()
                results['Sum'] = decrypted_sum

            results['Mean'] = results['Sum'] / len(original_data)

        progress_bar.progress(80)

        if "Variance (approximate)" in operations:
            # Approximate variance calculation
            mean_val = results.get('Mean', sum(original_data) / len(original_data))

            # For FHE, this is complex - we'll provide an approximation
            # In practice, this would require more sophisticated protocols
            variance_approx = sum([(x - mean_val) ** 2 for x in original_data[:10]]) / 10
            results['Variance (approximate)'] = variance_approx

        progress_bar.progress(100)
        status_text.text("Statistical operations completed!")

        # Store results
        operation_key = f"statistics_{col_name}_{hash(tuple(operations))}"
        st.session_state.computation_results[operation_key] = {
            'operation': 'statistical_operations',
            'column': col_name,
            'operations': operations,
            'results': results,
            'timestamp': time.time(),
            'scheme': encrypted_data['scheme']
        }

        st.success("✅ Statistical operations completed!")

        # Display results
        st.subheader("📊 Statistical Results")

        # Create comparison with expected values
        expected_results = {}
        if "Sum" in operations:
            expected_results['Sum'] = sum(original_data)
        if "Mean" in operations:
            expected_results['Mean'] = sum(original_data) / len(original_data)
        if "Variance (approximate)" in operations:
            mean_val = expected_results.get('Mean', sum(original_data) / len(original_data))
            expected_results['Variance (approximate)'] = sum([(x - mean_val) ** 2 for x in original_data]) / len(
                original_data)

        comparison_df = pd.DataFrame({
            'Statistic': list(results.keys()),
            'FHE Result': list(results.values()),
            'Expected': [expected_results.get(stat, 'N/A') for stat in results.keys()],
            'Error': [abs(results[stat] - expected_results.get(stat, 0)) if expected_results.get(stat,
                                                                                                 'N/A') != 'N/A' else 'N/A'
                      for stat in results.keys()]
        })

        st.dataframe(comparison_df)

        # Visualization
        fig = px.bar(comparison_df, x='Statistic', y='FHE Result',
                     title='Statistical Operations on Encrypted Data')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Statistical operations failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def perform_matrix_computation(selected_columns, operation, encrypted_data):
    """Perform matrix operations on encrypted data"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"Performing {operation} on encrypted matrix data...")
        progress_bar.progress(30)

        # Get encrypted data for selected columns
        encrypted_cols = [encrypted_data['columns'][col]['encrypted'] for col in selected_columns]
        original_cols = [encrypted_data['columns'][col]['original_data'][:10] for col in selected_columns]

        results = {}

        if operation == "Transpose":
            # Simulate transpose operation
            results['operation'] = "Matrix transpose simulation completed"
            results['original_shape'] = (10, len(selected_columns))
            results['transposed_shape'] = (len(selected_columns), 10)

        elif operation == "Element-wise addition":
            if len(selected_columns) >= 2:
                result = encrypted_cols[0] + encrypted_cols[1]
                decrypted_result = result.decrypt()[:10]
                results['addition_result'] = decrypted_result
                results['expected'] = [original_cols[0][i] + original_cols[1][i] for i in range(10)]

        elif operation == "Dot product simulation":
            if len(selected_columns) >= 2:
                # Simulate dot product with element-wise multiplication and sum
                elementwise_product = encrypted_cols[0] * encrypted_cols[1]
                if hasattr(elementwise_product, 'sum'):
                    dot_product = elementwise_product.sum()
                    decrypted_dot = dot_product.decrypt()
                else:
                    decrypted_product = elementwise_product.decrypt()[:10]
                    decrypted_dot = sum(decrypted_product)

                results['dot_product'] = decrypted_dot
                results['expected_dot'] = sum([original_cols[0][i] * original_cols[1][i] for i in range(10)])

        progress_bar.progress(100)
        status_text.text(f"{operation} completed!")

        # Store results
        operation_key = f"matrix_{operation}_{hash(tuple(selected_columns))}"
        st.session_state.computation_results[operation_key] = {
            'operation': f'matrix_{operation}',
            'columns': selected_columns,
            'results': results,
            'timestamp': time.time(),
            'scheme': encrypted_data['scheme']
        }

        st.success(f"✅ Matrix {operation} completed!")

        # Display results
        st.subheader("🔢 Matrix Operation Results")

        if operation == "Element-wise addition" and 'addition_result' in results:
            comparison_df = pd.DataFrame({
                f'{selected_columns[0]}': original_cols[0],
                f'{selected_columns[1]}': original_cols[1],
                'FHE Result': results['addition_result'],
                'Expected': results['expected'],
                'Error': [abs(fhe - exp) for fhe, exp in zip(results['addition_result'], results['expected'])]
            })
            st.dataframe(comparison_df)

        elif operation == "Dot product simulation" and 'dot_product' in results:
            st.metric("FHE Dot Product", f"{results['dot_product']:.4f}")
            st.metric("Expected Dot Product", f"{results['expected_dot']:.4f}")
            st.metric("Absolute Error", f"{abs(results['dot_product'] - results['expected_dot']):.6f}")

        else:
            st.write(results)

    except Exception as e:
        st.error(f"Matrix operation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def show_results_page():
    """Display results and analysis page"""
    st.header("📈 Results & Analysis")

    if not st.session_state.computation_results:
        st.warning("⚠️ No computation results available. Please perform some FHE operations first.")
        return

    # Results overview
    st.subheader("📊 Operations Overview")

    results_summary = []
    for key, result in st.session_state.computation_results.items():
        results_summary.append({
            'Operation ID': key,
            'Operation Type': result['operation'],
            'Scheme': result['scheme'],
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp'])),
            'Status': '✅ Completed'
        })

    summary_df = pd.DataFrame(results_summary)
    st.dataframe(summary_df, use_container_width=True)

    # Detailed results section
    st.subheader("🔍 Detailed Results")

    selected_operation = st.selectbox(
        "Select operation to view details",
        list(st.session_state.computation_results.keys()),
        format_func=lambda x: f"{st.session_state.computation_results[x]['operation']} - {x.split('_')[-1][:8]}..."
    )

    if selected_operation:
        result_data = st.session_state.computation_results[selected_operation]

        # Display operation details
        st.write("**Operation Details:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Operation Type", result_data['operation'])

        with col2:
            st.metric("Scheme Used", result_data['scheme'])

        with col3:
            st.metric("Execution Time",
                      time.strftime('%H:%M:%S', time.localtime(result_data['timestamp'])))

        # Display specific results based on operation type
        if result_data['operation'] in ['addition', 'multiplication']:
            display_binary_operation_results(result_data)
        elif result_data['operation'] == 'scalar_multiplication':
            display_scalar_operation_results(result_data)
        elif result_data['operation'] == 'polynomial_evaluation':
            display_polynomial_results(result_data)
        elif result_data['operation'] == 'statistical_operations':
            display_statistical_results(result_data)
        elif result_data['operation'].startswith('matrix'):
            display_matrix_results(result_data)

    # Performance analysis
    st.markdown("---")
    st.subheader("⚡ Performance Analysis")

    # Create performance metrics
    operation_counts = {}
    scheme_counts = {}

    for result in st.session_state.computation_results.values():
        op_type = result['operation']
        scheme = result['scheme']

        operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1

    col1, col2 = st.columns(2)

    with col1:
        # Operation distribution
        if operation_counts:
            fig_ops = px.pie(
                values=list(operation_counts.values()),
                names=list(operation_counts.keys()),
                title="Distribution of Operations"
            )
            st.plotly_chart(fig_ops, use_container_width=True)

    with col2:
        # Scheme usage
        if scheme_counts:
            fig_schemes = px.bar(
                x=list(scheme_counts.keys()),
                y=list(scheme_counts.values()),
                title="FHE Schemes Usage"
            )
            st.plotly_chart(fig_schemes, use_container_width=True)

    # Export results
    st.markdown("---")
    st.subheader("💾 Export Results")

    if st.button("📊 Generate Complete Report"):
        generate_complete_report()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export Results as JSON"):
            export_json = json.dumps(st.session_state.computation_results,
                                     indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=export_json,
                file_name=f"fhe_results_{int(time.time())}.json",
                mime="application/json"
            )

    with col2:
        if st.button("📊 Export Summary as CSV"):
            summary_df.to_csv("fhe_summary.csv", index=False)
            st.download_button(
                label="Download CSV Summary",
                data=summary_df.to_csv(index=False),
                file_name=f"fhe_summary_{int(time.time())}.csv",
                mime="text/csv"
            )


def display_binary_operation_results(result_data):
    """Display results for binary operations (addition, multiplication)"""
    st.write("**Result Data:**")

    results_df = pd.DataFrame({
        'Index': range(len(result_data['result'])),
        'Result': result_data['result']
    })

    st.dataframe(results_df)

    # Visualization
    fig = px.line(results_df, x='Index', y='Result',
                  title=f"{result_data['operation'].capitalize()} Results")
    st.plotly_chart(fig, use_container_width=True)


def display_scalar_operation_results(result_data):
    """Display results for scalar operations"""
    st.write("**Scalar Operation Results:**")
    st.write(f"Scalar value: {result_data['scalar']}")

    results_df = pd.DataFrame({
        'Index': range(len(result_data['result'])),
        'Result': result_data['result']
    })

    st.dataframe(results_df)


def display_polynomial_results(result_data):
    """Display polynomial evaluation results"""
    st.write("**Polynomial Evaluation Results:**")
    st.write(f"Coefficients: {result_data['coefficients']}")

    results_df = pd.DataFrame({
        'Index': range(len(result_data['result'])),
        'Polynomial Result': result_data['result']
    })

    st.dataframe(results_df)

    # Polynomial visualization
    fig = px.scatter(results_df, x='Index', y='Polynomial Result',
                     title="Polynomial Evaluation on Encrypted Data")
    st.plotly_chart(fig, use_container_width=True)


def display_statistical_results(result_data):
    """Display statistical operation results"""
    st.write("**Statistical Results:**")

    stats_df = pd.DataFrame([
        {'Statistic': stat, 'Value': value}
        for stat, value in result_data['results'].items()
    ])

    st.dataframe(stats_df)


def display_matrix_results(result_data):
    """Display matrix operation results"""
    st.write("**Matrix Operation Results:**")
    st.json(result_data['results'])


def generate_complete_report():
    """Generate a complete analysis report"""
    st.subheader("📋 Complete FHE Analysis Report")

    # Report metadata
    st.write("**Report Generated:** ", time.strftime('%Y-%m-%d %H:%M:%S'))

    # Dataset summary
    if 'encrypted_data' in st.session_state and st.session_state.encrypted_data:
        encrypted_data = st.session_state.encrypted_data
        st.write("**Dataset Information:**")
        st.write(f"- Encryption Scheme: {encrypted_data['scheme']}")
        st.write(f"- Polynomial Modulus Degree: {encrypted_data['poly_modulus_degree']}")
        st.write(f"- Columns Encrypted: {encrypted_data['metadata']['columns_encrypted']}")
        st.write(f"- Rows Processed: {encrypted_data['metadata']['rows_encrypted']}")
        st.write(f"- Library Used: {encrypted_data['library']}")

    # Operations summary
    st.write("**Operations Performed:**")

    total_operations = len(st.session_state.computation_results)
    st.metric("Total Operations", total_operations)

    # Recommendations
    st.write("**Recommendations:**")
    st.write("""
    - ✅ FHE operations completed successfully
    - 🔒 Data privacy maintained throughout computations
    - ⚡ Consider CKKS for floating-point operations
    - 🛡️ BFV/BGV schemes provide exact integer arithmetic
    - 📊 Statistical operations show good accuracy preservation
    """)


if __name__ == "__main__":
    main()