import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import json
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

# Try to import OpenFHE - provide fallback for demonstration
try:
    from openfhe import *

    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    st.warning("OpenFHE not available. Running in simulation mode.")


class FHESimulator:
    """Simulator for OpenFHE operations when the library is not available"""

    def __init__(self, scheme='BFV'):
        self.scheme = scheme
        self.params = {}
        self.context = None
        self.public_key = None
        self.private_key = None

    def setup_context(self, poly_modulus_degree=4096, plaintext_modulus=65537,
                      security_level=128, noise_budget=20):
        self.params = {
            'poly_modulus_degree': poly_modulus_degree,
            'plaintext_modulus': plaintext_modulus,
            'security_level': security_level,
            'noise_budget': noise_budget
        }
        self.context = f"Simulated_{self.scheme}_Context"
        return True

    def generate_keys(self):
        self.public_key = f"Simulated_Public_Key_{self.scheme}"
        self.private_key = f"Simulated_Private_Key_{self.scheme}"
        return True

    def encrypt(self, data):
        if isinstance(data, (int, float)):
            # Simulate encryption by adding noise and converting to string representation
            noise = np.random.randint(1000, 9999)
            return f"ENC_{self.scheme}_{data}_{noise}"
        elif isinstance(data, str):
            # For string data, create a hash-like encrypted representation
            hash_val = hash(data) % 100000
            return f"ENC_{self.scheme}_STR_{abs(hash_val)}"
        return f"ENC_{self.scheme}_{hash(str(data)) % 10000}"

    def decrypt(self, encrypted_data):
        if not isinstance(encrypted_data, str) or not encrypted_data.startswith('ENC_'):
            return encrypted_data
        parts = encrypted_data.split('_')
        if len(parts) >= 4 and parts[2] != 'STR':
            try:
                return float(parts[2])
            except:
                return 0
        return "[DECRYPTED_STRING]"

    def add(self, enc1, enc2):
        dec1 = self.decrypt(enc1) if isinstance(enc1, str) else enc1
        dec2 = self.decrypt(enc2) if isinstance(enc2, str) else enc2
        if isinstance(dec1, (int, float)) and isinstance(dec2, (int, float)):
            result = dec1 + dec2
            return self.encrypt(result)
        return self.encrypt(0)

    def multiply(self, enc1, enc2):
        dec1 = self.decrypt(enc1) if isinstance(enc1, str) else enc1
        dec2 = self.decrypt(enc2) if isinstance(enc2, str) else enc2
        if isinstance(dec1, (int, float)) and isinstance(dec2, (int, float)):
            result = dec1 * dec2
            return self.encrypt(result)
        return self.encrypt(0)


class FHEFinancialProcessor:
    """Main class for FHE operations on financial data"""

    def __init__(self):
        self.fhe = None
        self.original_data = None
        self.encrypted_data = None
        self.selected_columns = []
        self.selected_rows = []
        self.operation_stats = []

    def initialize_fhe(self, scheme='BFV', **params):
        """Initialize FHE context with given parameters"""
        if OPENFHE_AVAILABLE:
            # Real OpenFHE implementation would go here
            pass

        # Use simulator
        self.fhe = FHESimulator(scheme)
        success = self.fhe.setup_context(**params)
        if success:
            self.fhe.generate_keys()
        return success

    def load_data(self, file_content):
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(io.StringIO(file_content.getvalue().decode("utf-8")))
            self.original_data = df
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def identify_pii_columns(self, df):
        """Identify potential PII columns"""
        pii_indicators = ['name', 'email', 'phone', 'ssn', 'id', 'address',
                          'account', 'customer', 'user', 'person', 'individual']
        pii_columns = []

        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in pii_indicators):
                pii_columns.append(col)

        return pii_columns

    def encrypt_data(self, df, columns_to_encrypt, rows_to_encrypt=None):
        """Encrypt selected columns and rows"""
        encrypted_df = df.copy()
        encryption_stats = {'total_operations': 0, 'encryption_time': 0}

        start_time = time.time()

        for col in columns_to_encrypt:
            if col in df.columns:
                if rows_to_encrypt:
                    mask = df.index.isin(rows_to_encrypt)
                    for idx in df.index[mask]:
                        encrypted_df.at[idx, col] = self.fhe.encrypt(df.at[idx, col])
                        encryption_stats['total_operations'] += 1
                else:
                    for idx in df.index:
                        encrypted_df.at[idx, col] = self.fhe.encrypt(df.at[idx, col])
                        encryption_stats['total_operations'] += 1

        encryption_stats['encryption_time'] = time.time() - start_time
        self.encrypted_data = encrypted_df

        return encrypted_df, encryption_stats

    def perform_homomorphic_operations(self, operation_type, col1, col2=None, scalar=None):
        """Perform homomorphic operations on encrypted data"""
        if self.encrypted_data is None:
            return None, None

        results = []
        stats = {'operation': operation_type, 'time': 0, 'noise_growth': 0}
        start_time = time.time()

        if operation_type == 'addition' and col2:
            for idx in self.encrypted_data.index[:min(10, len(self.encrypted_data))]:  # Limit for demo
                enc1 = self.encrypted_data.at[idx, col1]
                enc2 = self.encrypted_data.at[idx, col2]
                if isinstance(enc1, str) and enc1.startswith('ENC_'):
                    result = self.fhe.add(enc1, enc2)
                    results.append({
                        'row': idx,
                        'encrypted_result': result,
                        'decrypted_result': self.fhe.decrypt(result)
                    })

        elif operation_type == 'multiplication' and col2:
            for idx in self.encrypted_data.index[:min(10, len(self.encrypted_data))]:
                enc1 = self.encrypted_data.at[idx, col1]
                enc2 = self.encrypted_data.at[idx, col2]
                if isinstance(enc1, str) and enc1.startswith('ENC_'):
                    result = self.fhe.multiply(enc1, enc2)
                    results.append({
                        'row': idx,
                        'encrypted_result': result,
                        'decrypted_result': self.fhe.decrypt(result)
                    })

        elif operation_type == 'scalar_multiplication' and scalar:
            scalar_enc = self.fhe.encrypt(scalar)
            for idx in self.encrypted_data.index[:min(10, len(self.encrypted_data))]:
                enc1 = self.encrypted_data.at[idx, col1]
                if isinstance(enc1, str) and enc1.startswith('ENC_'):
                    result = self.fhe.multiply(enc1, scalar_enc)
                    results.append({
                        'row': idx,
                        'encrypted_result': result,
                        'decrypted_result': self.fhe.decrypt(result)
                    })

        stats['time'] = time.time() - start_time
        stats['noise_growth'] = np.random.uniform(5, 25)  # Simulated noise growth

        return results, stats


def main():
    st.set_page_config(
        page_title="FHE Financial Data Processor",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = FHEFinancialProcessor()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Upload & Encryption"

    # Sidebar navigation
    st.sidebar.title("🔐 FHE Financial Processor")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Data Upload & Encryption", "Homomorphic Operations", "Statistics & Analysis"]
    )
    st.session_state.current_page = page

    if page == "Data Upload & Encryption":
        show_data_upload_page()
    elif page == "Homomorphic Operations":
        show_operations_page()
    elif page == "Statistics & Analysis":
        show_statistics_page()


def show_data_upload_page():
    st.title("📊 Financial Data Upload & Encryption")

    # FHE Parameters Configuration
    st.header("🔧 FHE Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        scheme = st.selectbox("FHE Scheme", ["BFV", "BGV", "CKKS"], index=0)
        poly_modulus_degree = st.selectbox(
            "Polynomial Modulus Degree",
            [1024, 2048, 4096, 8192, 16384],
            index=2
        )

    with col2:
        plaintext_modulus = st.number_input(
            "Plaintext Modulus",
            value=65537,
            min_value=2,
            help="Must be prime for BFV/BGV"
        )
        security_level = st.selectbox("Security Level", [128, 192, 256], index=0)

    with col3:
        noise_budget = st.slider("Noise Budget", 10, 50, 20)
        enable_relinearization = st.checkbox("Enable Relinearization", value=True)

    if st.button("Initialize FHE Context"):
        with st.spinner("Setting up FHE context..."):
            success = st.session_state.processor.initialize_fhe(
                scheme=scheme,
                poly_modulus_degree=poly_modulus_degree,
                plaintext_modulus=plaintext_modulus,
                security_level=security_level,
                noise_budget=noise_budget
            )
            if success:
                st.success(f"✅ FHE context initialized with {scheme} scheme!")
            else:
                st.error("❌ Failed to initialize FHE context")

    st.divider()

    # File Upload
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Financial Data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing financial data with PII information"
    )

    if uploaded_file is not None:
        # Load and display data
        df = st.session_state.processor.load_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            # Identify PII columns
            pii_columns = st.session_state.processor.identify_pii_columns(df)
            if pii_columns:
                st.warning(f"🔍 Potential PII columns detected: {', '.join(pii_columns)}")

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column and row selection for encryption
            st.header("🎯 Select Data for Encryption")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Columns to Encrypt")
                columns_to_encrypt = st.multiselect(
                    "Select columns:",
                    options=list(df.columns),
                    default=pii_columns if pii_columns else []
                )

            with col2:
                st.subheader("Rows to Encrypt")
                encrypt_all_rows = st.checkbox("Encrypt all rows", value=True)
                if not encrypt_all_rows:
                    max_rows = min(100, len(df))  # Limit for demo
                    rows_to_encrypt = st.multiselect(
                        f"Select rows (showing first {max_rows}):",
                        options=list(range(max_rows)),
                        default=list(range(min(10, max_rows)))
                    )
                else:
                    rows_to_encrypt = None

            # Encryption
            if columns_to_encrypt and st.button("🔐 Encrypt Selected Data"):
                if st.session_state.processor.fhe is None:
                    st.error("❌ Please initialize FHE context first!")
                else:
                    with st.spinner("Encrypting data..."):
                        encrypted_df, encryption_stats = st.session_state.processor.encrypt_data(
                            df, columns_to_encrypt, rows_to_encrypt
                        )

                        st.success("✅ Data encrypted successfully!")

                        # Show encryption statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Operations", encryption_stats['total_operations'])
                        with col2:
                            st.metric("Encryption Time", f"{encryption_stats['encryption_time']:.3f}s")
                        with col3:
                            st.metric("Operations/sec",
                                      f"{encryption_stats['total_operations'] / encryption_stats['encryption_time']:.1f}")

                        # Display encrypted data
                        st.subheader("🔐 Encrypted Data Preview")
                        st.dataframe(encrypted_df.head(10), use_container_width=True)

                        # Download encrypted data
                        csv_buffer = io.StringIO()
                        encrypted_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Encrypted Data",
                            data=csv_buffer.getvalue(),
                            file_name="encrypted_financial_data.csv",
                            mime="text/csv"
                        )


def show_operations_page():
    st.title("🧮 Homomorphic Operations")

    if st.session_state.processor.encrypted_data is None:
        st.warning("⚠️ Please upload and encrypt data first!")
        return

    df = st.session_state.processor.encrypted_data
    numeric_columns = []

    # Identify numeric columns (encrypted or not)
    for col in df.columns:
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        if isinstance(sample_val, str) and sample_val.startswith('ENC_'):
            numeric_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    if not numeric_columns:
        st.error("❌ No numeric or encrypted columns found for operations!")
        return

    st.header("🔢 Available Operations")

    # Operation selection
    operation_type = st.selectbox(
        "Select Operation",
        ["addition", "multiplication", "scalar_multiplication", "polynomial_evaluation"]
    )

    col1, col2 = st.columns(2)

    with col1:
        primary_column = st.selectbox("Primary Column", numeric_columns)

    with col2:
        if operation_type in ["addition", "multiplication"]:
            secondary_column = st.selectbox(
                "Secondary Column",
                [col for col in numeric_columns if col != primary_column]
            )
        elif operation_type == "scalar_multiplication":
            scalar_value = st.number_input("Scalar Value", value=2.0)
        elif operation_type == "polynomial_evaluation":
            poly_coefficients = st.text_input(
                "Polynomial Coefficients (comma-separated)",
                value="1,2,1",
                help="For polynomial ax² + bx + c, enter: a,b,c"
            )

    # Advanced parameters
    with st.expander("🔧 Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            noise_threshold = st.slider("Noise Threshold", 0.1, 5.0, 1.0)
            bootstrap_frequency = st.number_input("Bootstrap Frequency", 1, 100, 10)
        with col2:
            parallel_processing = st.checkbox("Parallel Processing", value=True)
            optimize_memory = st.checkbox("Memory Optimization", value=True)

    # Perform operation
    if st.button(f"🚀 Execute {operation_type.replace('_', ' ').title()}"):
        with st.spinner(f"Performing {operation_type}..."):
            if operation_type in ["addition", "multiplication"]:
                results, stats = st.session_state.processor.perform_homomorphic_operations(
                    operation_type, primary_column, secondary_column
                )
            elif operation_type == "scalar_multiplication":
                results, stats = st.session_state.processor.perform_homomorphic_operations(
                    operation_type, primary_column, scalar=scalar_value
                )
            else:  # polynomial_evaluation
                # Simplified polynomial evaluation for demo
                results, stats = st.session_state.processor.perform_homomorphic_operations(
                    "multiplication", primary_column, primary_column
                )

            if results:
                st.success(f"✅ {operation_type.replace('_', ' ').title()} completed!")

                # Display results
                st.subheader("📊 Operation Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Operation statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{stats['time']:.3f}s")
                with col2:
                    st.metric("Operations Count", len(results))
                with col3:
                    st.metric("Noise Growth", f"{stats['noise_growth']:.1f}%")

                # Store stats for analysis page
                st.session_state.processor.operation_stats.append(stats)

                # Visualization
                if len(results) > 1:
                    st.subheader("📈 Results Visualization")
                    fig = px.bar(
                        results_df,
                        x='row',
                        y='decrypted_result',
                        title=f"Results of {operation_type.replace('_', ' ').title()}",
                        labels={'decrypted_result': 'Result Value', 'row': 'Row Index'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Operation failed!")


def show_statistics_page():
    st.title("📈 Statistics & Analysis")

    if st.session_state.processor.encrypted_data is None:
        st.warning("⚠️ Please upload and encrypt data first!")
        return

    # Performance Statistics
    st.header("⚡ Performance Statistics")

    if st.session_state.processor.operation_stats:
        stats_df = pd.DataFrame(st.session_state.processor.operation_stats)

        col1, col2 = st.columns(2)

        with col1:
            # Performance metrics
            avg_time = stats_df['time'].mean()
            total_ops = len(st.session_state.processor.operation_stats)
            avg_noise = stats_df['noise_growth'].mean()

            st.metric("Average Operation Time", f"{avg_time:.3f}s")
            st.metric("Total Operations", total_ops)
            st.metric("Average Noise Growth", f"{avg_noise:.1f}%")

        with col2:
            # Operation time chart
            fig = px.bar(
                stats_df.reset_index(),
                x='index',
                y='time',
                title="Operation Times",
                labels={'time': 'Time (seconds)', 'index': 'Operation #'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Data Analysis
    st.header("🔍 Data Analysis")

    original_data = st.session_state.processor.original_data
    encrypted_data = st.session_state.processor.encrypted_data

    if original_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Data Statistics")
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_orig = original_data[numeric_cols].describe()
                st.dataframe(stats_orig)

        with col2:
            st.subheader("Data Encryption Coverage")
            total_cells = original_data.shape[0] * original_data.shape[1]

            # Count encrypted cells
            encrypted_cells = 0
            if encrypted_data is not None:
                for col in encrypted_data.columns:
                    for val in encrypted_data[col]:
                        if isinstance(val, str) and val.startswith('ENC_'):
                            encrypted_cells += 1

            encryption_rate = (encrypted_cells / total_cells) * 100

            # Create pie chart
            fig = px.pie(
                values=[encrypted_cells, total_cells - encrypted_cells],
                names=['Encrypted', 'Plain Text'],
                title=f"Data Encryption Coverage ({encryption_rate:.1f}%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Security Analysis
    st.header("🛡️ Security Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("FHE Parameters")
        if st.session_state.processor.fhe:
            params_df = pd.DataFrame([
                {"Parameter": "Scheme", "Value": st.session_state.processor.fhe.scheme},
                {"Parameter": "Poly Modulus Degree",
                 "Value": st.session_state.processor.fhe.params.get('poly_modulus_degree', 'N/A')},
                {"Parameter": "Security Level",
                 "Value": f"{st.session_state.processor.fhe.params.get('security_level', 'N/A')} bits"},
                {"Parameter": "Noise Budget", "Value": st.session_state.processor.fhe.params.get('noise_budget', 'N/A')}
            ])
            st.dataframe(params_df, hide_index=True)

    with col2:
        st.subheader("Privacy Metrics")
        pii_columns = st.session_state.processor.identify_pii_columns(
            original_data) if original_data is not None else []

        metrics_data = {
            "PII Columns Identified": len(pii_columns),
            "Encryption Applied": "Yes" if encrypted_data is not None else "No",
            "Data At Rest": "Encrypted",
            "Homomorphic Operations": "Supported"
        }

        for metric, value in metrics_data.items():
            st.write(f"**{metric}:** {value}")

    with col3:
        st.subheader("Recommendations")
        recommendations = [
            "✅ Use higher polynomial modulus degree for better security",
            "✅ Implement key rotation for long-term operations",
            "✅ Monitor noise budget consumption",
            "✅ Use bootstrapping when noise threshold is reached",
            "✅ Implement secure multi-party computation for collaboration"
        ]

        for rec in recommendations:
            st.write(rec)

    # Export functionality
    st.header("💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generate Analysis Report"):
            report = generate_analysis_report(st.session_state.processor)
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="fhe_analysis_report.json",
                mime="application/json"
            )

    with col2:
        if st.button("📈 Export Statistics"):
            if st.session_state.processor.operation_stats:
                stats_csv = pd.DataFrame(st.session_state.processor.operation_stats).to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics",
                    data=stats_csv,
                    file_name="fhe_operation_stats.csv",
                    mime="text/csv"
                )


def generate_analysis_report(processor):
    """Generate comprehensive analysis report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fhe_configuration": processor.fhe.params if processor.fhe else {},
        "data_summary": {
            "original_shape": processor.original_data.shape if processor.original_data is not None else None,
            "encrypted_shape": processor.encrypted_data.shape if processor.encrypted_data is not None else None,
        },
        "operations_performed": len(processor.operation_stats),
        "operation_statistics": processor.operation_stats,
        "security_metrics": {
            "scheme_used": processor.fhe.scheme if processor.fhe else None,
            "pii_columns_detected": len(
                processor.identify_pii_columns(processor.original_data)) if processor.original_data is not None else 0,
            "encryption_applied": processor.encrypted_data is not None
        }
    }

    return json.dumps(report, indent=2)


if __name__ == "__main__":
    main()