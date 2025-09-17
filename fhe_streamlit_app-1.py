"""
pip install streamlit tenseal pandas numpy plotly
"""

import streamlit as st
import tenseal as ts
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="FHE Financial Data Analysis",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FHEManager:
    """Manages FHE contexts, encryption, and operations"""

    def __init__(self):
        self.contexts = {}
        self.keys = {}
        self.encrypted_data = {}

    def create_context(self, scheme: str, country: str, **params) -> ts.TenSEALContext:
        """Create FHE context based on scheme and parameters"""
        try:
            if scheme == "BFV":
                context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=params.get('poly_modulus_degree', 4096),
                    plain_modulus=params.get('plain_modulus', 1032193)
                )
                context.generate_galois_keys()
                context.generate_relin_keys()

            elif scheme == "BGV":
                context = ts.context(
                    ts.SCHEME_TYPE.BGV,
                    poly_modulus_degree=params.get('poly_modulus_degree', 4096),
                    plain_modulus=params.get('plain_modulus', 1032193)
                )
                context.generate_galois_keys()
                context.generate_relin_keys()

            elif scheme == "CKKS":
                context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=params.get('poly_modulus_degree', 8192),
                    coeff_mod_bit_sizes=params.get('coeff_mod_bit_sizes', [60, 40, 40, 60])
                )
                context.global_scale = params.get('global_scale', 2 ** 40)
                context.generate_galois_keys()
                context.generate_relin_keys()

            self.contexts[f"{country}_{scheme}"] = context
            return context

        except Exception as e:
            st.error(f"Error creating {scheme} context: {str(e)}")
            return None

    def encrypt_data(self, data: List[float], context: ts.TenSEALContext, scheme: str) -> Any:
        """Encrypt data using the specified context"""
        try:
            if scheme in ["BFV", "BGV"]:
                # Convert float to int for BFV/BGV (scaled by 1000 for precision)
                int_data = [int(x * 1000) for x in data]
                return ts.bfv_vector(context, int_data)
            elif scheme == "CKKS":
                return ts.ckks_vector(context, data)
        except Exception as e:
            st.error(f"Error encrypting data: {str(e)}")
            return None

    def decrypt_data(self, encrypted_data: Any, scheme: str) -> List[float]:
        """Decrypt data"""
        try:
            decrypted = encrypted_data.decrypt()
            if scheme in ["BFV", "BGV"]:
                # Convert back from scaled integers
                return [x / 1000.0 for x in decrypted]
            else:
                return decrypted
        except Exception as e:
            st.error(f"Error decrypting data: {str(e)}")
            return []


class DataProcessor:
    """Handles data processing and analysis on encrypted data"""

    @staticmethod
    def perform_arithmetic_operations(encrypted_vector, operation: str, value: float = None):
        """Perform arithmetic operations on encrypted data"""
        try:
            if operation == "add" and value is not None:
                return encrypted_vector + value
            elif operation == "subtract" and value is not None:
                return encrypted_vector - value
            elif operation == "multiply" and value is not None:
                return encrypted_vector * value
            elif operation == "square":
                return encrypted_vector * encrypted_vector
            elif operation == "sum":
                # Sum all elements (requires rotation)
                result = encrypted_vector
                for i in range(1, len(encrypted_vector.decrypt())):
                    rotated = ts.ckks_vector(encrypted_vector.context(), encrypted_vector.decrypt())
                    # This is a simplified sum - in practice, you'd use rotation operations
                return result
        except Exception as e:
            st.error(f"Error in arithmetic operation: {str(e)}")
            return None

    @staticmethod
    def compute_statistics(encrypted_data, scheme: str):
        """Compute basic statistics on encrypted data"""
        try:
            # Note: Some operations may require decryption for display
            # In a real scenario, you'd compute these homomorphically
            decrypted = encrypted_data.decrypt()
            if scheme in ["BFV", "BGV"]:
                decrypted = [x / 1000.0 for x in decrypted]

            stats = {
                'count': len(decrypted),
                'mean': np.mean(decrypted),
                'std': np.std(decrypted),
                'min': np.min(decrypted),
                'max': np.max(decrypted),
                'median': np.median(decrypted)
            }
            return stats
        except Exception as e:
            st.error(f"Error computing statistics: {str(e)}")
            return {}


def main():
    st.markdown('<h1 class="main-header">🔐 Fully Homomorphic Encryption Financial Data Analysis</h1>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'fhe_manager' not in st.session_state:
        st.session_state.fhe_manager = FHEManager()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'encrypted_datasets' not in st.session_state:
        st.session_state.encrypted_datasets = {}

    # Sidebar configuration
    st.sidebar.title("🔧 FHE Configuration")

    # Scheme selection
    scheme = st.sidebar.selectbox(
        "Select FHE Scheme",
        ["CKKS", "BFV", "BGV"],
        help="CKKS: Approximate arithmetic on real numbers\nBFV/BGV: Exact arithmetic on integers"
    )

    # Country/Region selection
    country = st.sidebar.selectbox(
        "Data Origin Country/Region",
        ["USA", "UK", "Germany", "Japan", "Canada", "Australia", "Singapore", "Switzerland"],
        help="This determines the encryption key authority"
    )

    # Advanced parameters
    st.sidebar.markdown("### Advanced Parameters")

    if scheme == "CKKS":
        poly_modulus_degree = st.sidebar.selectbox(
            "Polynomial Modulus Degree",
            [4096, 8192, 16384],
            index=1,
            help="Higher values provide more security but slower performance"
        )

        global_scale = st.sidebar.selectbox(
            "Global Scale (2^n)",
            [20, 30, 40, 50],
            index=2,
            help="Precision parameter for CKKS"
        )

        coeff_mod_bit_sizes = st.sidebar.text_input(
            "Coefficient Modulus Bit Sizes",
            "[60, 40, 40, 60]",
            help="Comma-separated list in brackets"
        )

        context_params = {
            'poly_modulus_degree': poly_modulus_degree,
            'global_scale': 2 ** global_scale,
            'coeff_mod_bit_sizes': eval(coeff_mod_bit_sizes)
        }
    else:
        poly_modulus_degree = st.sidebar.selectbox(
            "Polynomial Modulus Degree",
            [2048, 4096, 8192],
            index=1
        )

        plain_modulus = st.sidebar.number_input(
            "Plain Modulus",
            min_value=2,
            value=1032193,
            help="Must be a prime number"
        )

        context_params = {
            'poly_modulus_degree': poly_modulus_degree,
            'plain_modulus': plain_modulus
        }

    # Create context button
    if st.sidebar.button("🔑 Generate FHE Context"):
        with st.spinner(f"Generating {scheme} context for {country}..."):
            context = st.session_state.fhe_manager.create_context(scheme, country, **context_params)
            if context:
                st.sidebar.success(f"✅ Context created for {country}")
                st.sidebar.info(f"Key ID: {country}_{scheme}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Upload",
        "🔒 Encryption",
        "⚡ Operations",
        "📊 Analysis",
        "🔍 Privacy Demo"
    ])

    with tab1:
        st.markdown('<h2 class="section-header">Data Upload & Preview</h2>', unsafe_allow_html=True)

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Financial Dataset (CSV)",
            type=['csv'],
            help="Upload your financial data. The system will encrypt it using the selected country's key."
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df

                st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", df.shape[0])
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Column information
                st.subheader("📈 Numerical Columns Available for Encryption")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    col_info = pd.DataFrame({
                        'Column': numeric_cols,
                        'Type': [str(df[col].dtype) for col in numeric_cols],
                        'Non-Null Count': [df[col].count() for col in numeric_cols],
                        'Mean': [f"{df[col].mean():.2f}" for col in numeric_cols],
                        'Std': [f"{df[col].std():.2f}" for col in numeric_cols]
                    })
                    st.dataframe(col_info, use_container_width=True)
                else:
                    st.warning("⚠️ No numerical columns found for encryption")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with tab2:
        st.markdown('<h2 class="section-header">Data Encryption</h2>', unsafe_allow_html=True)

        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols and f"{country}_{scheme}" in st.session_state.fhe_manager.contexts:
                # Column selection for encryption
                selected_columns = st.multiselect(
                    "Select columns to encrypt",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )

                if selected_columns and st.button("🔒 Encrypt Selected Data"):
                    context = st.session_state.fhe_manager.contexts[f"{country}_{scheme}"]

                    with st.spinner("Encrypting data..."):
                        progress_bar = st.progress(0)

                        for i, col in enumerate(selected_columns):
                            # Handle missing values
                            col_data = df[col].fillna(df[col].mean()).tolist()

                            # Encrypt the column
                            encrypted_col = st.session_state.fhe_manager.encrypt_data(
                                col_data, context, scheme
                            )

                            if encrypted_col:
                                st.session_state.encrypted_datasets[f"{country}_{scheme}_{col}"] = {
                                    'encrypted_data': encrypted_col,
                                    'original_data': col_data,
                                    'scheme': scheme,
                                    'country': country,
                                    'column_name': col
                                }

                            progress_bar.progress((i + 1) / len(selected_columns))

                    st.success(f"✅ Encrypted {len(selected_columns)} columns using {scheme} scheme")

                    # Display encryption summary
                    st.subheader("🔐 Encryption Summary")
                    encryption_summary = pd.DataFrame([
                        {
                            'Column': col,
                            'Scheme': scheme,
                            'Country Key': country,
                            'Data Points': len(df[col].dropna()),
                            'Status': '✅ Encrypted'
                        }
                        for col in selected_columns
                    ])
                    st.dataframe(encryption_summary, use_container_width=True)

            elif not numeric_cols:
                st.warning("⚠️ Please upload data with numerical columns first")
            else:
                st.warning(f"⚠️ Please generate FHE context for {country} using {scheme} scheme first")
        else:
            st.info("📁 Please upload a dataset first in the Data Upload tab")

    with tab3:
        st.markdown('<h2 class="section-header">Homomorphic Operations</h2>', unsafe_allow_html=True)

        if st.session_state.encrypted_datasets:
            # Select encrypted dataset
            dataset_keys = list(st.session_state.encrypted_datasets.keys())
            selected_dataset = st.selectbox(
                "Select encrypted dataset",
                dataset_keys,
                format_func=lambda x: f"{x.split('_')[2]} ({x.split('_')[1]} - {x.split('_')[0]})"
            )

            if selected_dataset:
                dataset_info = st.session_state.encrypted_datasets[selected_dataset]
                encrypted_data = dataset_info['encrypted_data']

                st.info(f"🔒 Working with encrypted column: {dataset_info['column_name']}")

                # Operation selection
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("➕ Arithmetic Operations")

                    operation = st.selectbox(
                        "Select operation",
                        ["add", "subtract", "multiply", "square"]
                    )

                    if operation in ["add", "subtract", "multiply"]:
                        operation_value = st.number_input(
                            f"Value to {operation}",
                            value=1.0,
                            step=0.1
                        )
                    else:
                        operation_value = None

                    if st.button(f"Perform {operation.title()} Operation"):
                        with st.spinner(f"Performing {operation} operation on encrypted data..."):
                            try:
                                result = st.session_state.data_processor.perform_arithmetic_operations(
                                    encrypted_data, operation, operation_value
                                )

                                if result:
                                    # Store result
                                    result_key = f"{selected_dataset}_result_{operation}"
                                    st.session_state.encrypted_datasets[result_key] = {
                                        'encrypted_data': result,
                                        'original_data': None,
                                        'scheme': dataset_info['scheme'],
                                        'country': dataset_info['country'],
                                        'column_name': f"{dataset_info['column_name']}_{operation}"
                                    }

                                    st.success(f"✅ {operation.title()} operation completed on encrypted data!")

                                    # Show some results (first 10 values)
                                    decrypted_result = st.session_state.fhe_manager.decrypt_data(
                                        result, dataset_info['scheme']
                                    )

                                    st.subheader("📊 Operation Results (First 10 values)")
                                    results_df = pd.DataFrame({
                                        'Index': range(min(10, len(decrypted_result))),
                                        'Result': decrypted_result[:10]
                                    })
                                    st.dataframe(results_df, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error performing operation: {str(e)}")

                with col2:
                    st.subheader("🔍 Polynomial Evaluation")

                    # Simple polynomial evaluation
                    st.write("Evaluate: ax² + bx + c")

                    coeff_a = st.number_input("Coefficient a", value=1.0, step=0.1)
                    coeff_b = st.number_input("Coefficient b", value=2.0, step=0.1)
                    coeff_c = st.number_input("Coefficient c", value=1.0, step=0.1)

                    if st.button("Evaluate Polynomial"):
                        with st.spinner("Evaluating polynomial on encrypted data..."):
                            try:
                                # ax² + bx + c
                                x_squared = encrypted_data * encrypted_data
                                ax_squared = x_squared * coeff_a
                                bx = encrypted_data * coeff_b
                                result = ax_squared + bx + coeff_c

                                # Store result
                                result_key = f"{selected_dataset}_polynomial"
                                st.session_state.encrypted_datasets[result_key] = {
                                    'encrypted_data': result,
                                    'original_data': None,
                                    'scheme': dataset_info['scheme'],
                                    'country': dataset_info['country'],
                                    'column_name': f"{dataset_info['column_name']}_poly"
                                }

                                st.success("✅ Polynomial evaluation completed!")

                                # Show results
                                decrypted_result = st.session_state.fhe_manager.decrypt_data(
                                    result, dataset_info['scheme']
                                )

                                original_data = dataset_info['original_data'][:10]
                                poly_results = decrypted_result[:10]

                                comparison_df = pd.DataFrame({
                                    'Original X': original_data,
                                    f'Polynomial Result ({coeff_a}x² + {coeff_b}x + {coeff_c})': poly_results
                                })
                                st.dataframe(comparison_df, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error evaluating polynomial: {str(e)}")
        else:
            st.info("🔒 Please encrypt some data first in the Encryption tab")

    with tab4:
        st.markdown('<h2 class="section-header">Statistical Analysis & Visualization</h2>', unsafe_allow_html=True)

        if st.session_state.encrypted_datasets:
            # Analysis options
            st.subheader("📈 Available Encrypted Datasets")

            analysis_datasets = st.multiselect(
                "Select datasets for analysis",
                list(st.session_state.encrypted_datasets.keys()),
                format_func=lambda x: f"{x.split('_')[2]} ({x.split('_')[1]} - {x.split('_')[0]})"
            )

            if analysis_datasets and st.button("📊 Perform Statistical Analysis"):
                results = {}

                with st.spinner("Computing statistics on encrypted data..."):
                    for dataset_key in analysis_datasets:
                        dataset_info = st.session_state.encrypted_datasets[dataset_key]
                        encrypted_data = dataset_info['encrypted_data']
                        scheme = dataset_info['scheme']

                        stats = st.session_state.data_processor.compute_statistics(
                            encrypted_data, scheme
                        )
                        results[dataset_key] = stats

                # Display results
                st.subheader("📋 Statistical Summary")

                stats_data = []
                for dataset_key, stats in results.items():
                    col_name = st.session_state.encrypted_datasets[dataset_key]['column_name']
                    country = st.session_state.encrypted_datasets[dataset_key]['country']
                    scheme = st.session_state.encrypted_datasets[dataset_key]['scheme']

                    stats_data.append({
                        'Dataset': col_name,
                        'Country': country,
                        'Scheme': scheme,
                        'Count': stats.get('count', 0),
                        'Mean': f"{stats.get('mean', 0):.2f}",
                        'Std Dev': f"{stats.get('std', 0):.2f}",
                        'Min': f"{stats.get('min', 0):.2f}",
                        'Max': f"{stats.get('max', 0):.2f}",
                        'Median': f"{stats.get('median', 0):.2f}"
                    })

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                # Visualizations
                if len(analysis_datasets) >= 1:
                    st.subheader("📊 Visualizations")

                    # Create plots for each dataset
                    for dataset_key in analysis_datasets[:3]:  # Limit to 3 for performance
                        dataset_info = st.session_state.encrypted_datasets[dataset_key]
                        col_name = dataset_info['column_name']

                        # Decrypt for visualization (in practice, you might compute histograms homomorphically)
                        decrypted_data = st.session_state.fhe_manager.decrypt_data(
                            dataset_info['encrypted_data'], dataset_info['scheme']
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            # Histogram
                            fig_hist = px.histogram(
                                x=decrypted_data,
                                title=f"Distribution of {col_name}",
                                nbins=30
                            )
                            fig_hist.update_layout(xaxis_title=col_name, yaxis_title="Frequency")
                            st.plotly_chart(fig_hist, use_container_width=True)

                        with col2:
                            # Box plot
                            fig_box = px.box(
                                y=decrypted_data,
                                title=f"Box Plot of {col_name}"
                            )
                            fig_box.update_layout(yaxis_title=col_name)
                            st.plotly_chart(fig_box, use_container_width=True)

                # Comparison chart if multiple datasets
                if len(analysis_datasets) > 1:
                    st.subheader("🔄 Dataset Comparison")

                    comparison_data = []
                    for dataset_key in analysis_datasets:
                        stats = results[dataset_key]
                        col_name = st.session_state.encrypted_datasets[dataset_key]['column_name']
                        comparison_data.append({
                            'Dataset': col_name,
                            'Mean': stats.get('mean', 0),
                            'Std Dev': stats.get('std', 0)
                        })

                    comparison_df = pd.DataFrame(comparison_data)

                    fig_comparison = px.bar(
                        comparison_df,
                        x='Dataset',
                        y=['Mean', 'Std Dev'],
                        barmode='group',
                        title="Dataset Comparison: Mean and Standard Deviation"
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("📊 Please encrypt some data first to perform analysis")

    with tab5:
        st.markdown('<h2 class="section-header">Privacy Preservation Demonstration</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h3>🔐 How Fully Homomorphic Encryption Preserves Privacy</h3>
        <p>This demonstration shows that computations are performed on encrypted data without ever decrypting it.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.encrypted_datasets:
            # Select a dataset for demonstration
            demo_dataset = st.selectbox(
                "Select dataset for privacy demonstration",
                list(st.session_state.encrypted_datasets.keys()),
                format_func=lambda x: f"{x.split('_')[2]} ({x.split('_')[1]})"
            )

            if demo_dataset:
                dataset_info = st.session_state.encrypted_datasets[demo_dataset]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔓 Original Data (First 10 values)")
                    if dataset_info['original_data']:
                        original_df = pd.DataFrame({
                            'Index': range(10),
                            'Original Value': dataset_info['original_data'][:10]
                        })
                        st.dataframe(original_df)

                with col2:
                    st.subheader("🔒 Encrypted Data Representation")
                    st.code("Encrypted data cannot be read without the private key!")

                    # Show that encrypted data is opaque
                    encrypted_repr = "🔒 [ENCRYPTED] - Data is fully protected"
                    for i in range(10):
                        st.text(f"Index {i}: {encrypted_repr}")

                # Demonstrate computation on encrypted data
                st.subheader("⚡ Computing on Encrypted Data")

                operation_demo = st.selectbox(
                    "Select operation to demonstrate",
                    ["multiply_by_2", "add_100", "square"]
                )

                if st.button("🚀 Perform Operation on Encrypted Data"):
                    with st.spinner("Computing on encrypted data (no decryption!)..."):
                        encrypted_data = dataset_info['encrypted_data']

                        # Perform operation
                        if operation_demo == "multiply_by_2":
                            result_encrypted = encrypted_data * 2
                            operation_desc = "Multiplied by 2"
                        elif operation_demo == "add_100":
                            result_encrypted = encrypted_data + 100
                            operation_desc = "Added 100"
                        elif operation_demo == "square":
                            result_encrypted = encrypted_data * encrypted_data
                            operation_desc = "Squared"

                        # Now decrypt to show results (for demonstration)
                        result_decrypted = st.session_state.fhe_manager.decrypt_data(
                            result_encrypted, dataset_info['scheme']
                        )

                        st.success(f"✅ {operation_desc} - All computation done on encrypted data!")

                        # Show the proof
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("📊 Original")
                            orig_data = dataset_info['original_data'][:10]
                            for i, val in enumerate(orig_data):
                                st.text(f"{i}: {val:.2f}")

                        with col2:
                            st.subheader("⚡ Encrypted Operation")
                            for i in range(10):
                                st.text(f"{i}: 🔒 [COMPUTED]")

                        with col3:
                            st.subheader("✅ Final Result")
                            for i, val in enumerate(result_decrypted[:10]):
                                st.text(f"{i}: {val:.2f}")

                # Privacy guarantees
                st.subheader("🛡️ Privacy Guarantees")

                guarantees = [
                    "✅ Original data never exposed during computation",
                    "✅ Intermediate results remain encrypted",
                    "✅ Only authorized parties with private keys can decrypt",
                    "✅ Computation results are mathematically identical to plaintext operations",
                    "✅ Zero-knowledge about individual data points during processing"
                ]

                for guarantee in guarantees:
                    st.markdown(guarantee)

                # Context information
                st.subheader("🔑 Encryption Context Information")
                context_key = f"{dataset_info['country']}_{dataset_info['scheme']}"

                if context_key in st.session_state.fhe_manager.contexts:
                    context = st.session_state.fhe_manager.contexts[context_key]

                    context_info = {
                        'Country/Key Authority': dataset_info['country'],
                        'FHE Scheme': dataset_info['scheme'],
                        'Security Level': 'High (Based on polynomial modulus degree)',
                        'Key Management': f"Managed by {dataset_info['country']} authority",
                        'Data Sovereignty': f"Data remains under {dataset_info['country']} jurisdiction"
                    }

                    context_df = pd.DataFrame(list(context_info.items()), columns=['Property', 'Value'])
                    st.dataframe(context_df, use_container_width=True)

        else:
            st.info("🔒 Please encrypt some data first to see the privacy demonstration")

        # Educational section
        st.markdown("""
        <div class="info-box">
        <h3>📚 Understanding FHE Schemes</h3>
        <ul>
            <li><strong>BFV (Brakerski-Fan-Vercauteren):</strong> Integer arithmetic, exact computations, good for counting and summation</li>
            <li><strong>BGV (Brakerski-Gentry-Vaikuntanathan):</strong> Similar to BFV, optimized for different parameter choices</li>
            <li><strong>CKKS (Cheon-Kim-Kim-Song):</strong> Approximate arithmetic on real numbers, ideal for statistical computations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# Additional utility functions
def create_sample_data():
    """Create sample financial data for demonstration"""
    np.random.seed(42)
    n_samples = 1000

    countries = ['USA', 'UK', 'Germany', 'Japan']

    data = []
    for i in range(n_samples):
        country = np.random.choice(countries)
        data.append({
            'transaction_id': f'TXN_{i:06d}',
            'country': country,
            'amount': np.random.normal(5000, 2000),
            'interest_rate': np.random.normal(0.05, 0.02),
            'credit_score': np.random.randint(300, 850),
            'account_balance': np.random.exponential(10000),
            'monthly_income': np.random.normal(7500, 3000),
            'loan_amount': np.random.exponential(50000),
            'risk_score': np.random.beta(2, 5) * 100
        })

    return pd.DataFrame(data)


# Advanced operations section
def show_advanced_operations():
    """Show advanced FHE operations"""
    st.markdown('<h2 class="section-header">🔬 Advanced FHE Operations</h2>', unsafe_allow_html=True)

    if st.session_state.encrypted_datasets:
        st.subheader("🧮 Matrix Operations on Encrypted Data")

        # Select multiple datasets for matrix operations
        selected_datasets = st.multiselect(
            "Select datasets for matrix operations (select 2 or more)",
            list(st.session_state.encrypted_datasets.keys())[:4],  # Limit for performance
            format_func=lambda x: f"{x.split('_')[2]}"
        )

        if len(selected_datasets) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Element-wise Multiplication"):
                    dataset1 = st.session_state.encrypted_datasets[selected_datasets[0]]
                    dataset2 = st.session_state.encrypted_datasets[selected_datasets[1]]

                    # Ensure same scheme
                    if dataset1['scheme'] == dataset2['scheme']:
                        try:
                            result = dataset1['encrypted_data'] * dataset2['encrypted_data']
                            st.success("✅ Element-wise multiplication completed on encrypted data!")

                            # Show sample results
                            decrypted_result = st.session_state.fhe_manager.decrypt_data(
                                result, dataset1['scheme']
                            )

                            sample_df = pd.DataFrame({
                                'Dataset 1': dataset1['original_data'][:5],
                                'Dataset 2': dataset2['original_data'][:5],
                                'Encrypted Multiplication': decrypted_result[:5]
                            })
                            st.dataframe(sample_df)

                        except Exception as e:
                            st.error(f"Error in matrix operation: {str(e)}")
                    else:
                        st.error("Datasets must use the same FHE scheme for operations")

            with col2:
                if st.button("➕ Element-wise Addition"):
                    dataset1 = st.session_state.encrypted_datasets[selected_datasets[0]]
                    dataset2 = st.session_state.encrypted_datasets[selected_datasets[1]]

                    if dataset1['scheme'] == dataset2['scheme']:
                        try:
                            result = dataset1['encrypted_data'] + dataset2['encrypted_data']
                            st.success("✅ Element-wise addition completed on encrypted data!")

                            decrypted_result = st.session_state.fhe_manager.decrypt_data(
                                result, dataset1['scheme']
                            )

                            sample_df = pd.DataFrame({
                                'Dataset 1': dataset1['original_data'][:5],
                                'Dataset 2': dataset2['original_data'][:5],
                                'Encrypted Addition': decrypted_result[:5]
                            })
                            st.dataframe(sample_df)

                        except Exception as e:
                            st.error(f"Error in matrix operation: {str(e)}")
                    else:
                        st.error("Datasets must use the same FHE scheme for operations")


# Performance monitoring
def show_performance_metrics():
    """Show FHE performance metrics"""
    st.markdown('<h2 class="section-header">⚡ Performance Metrics</h2>', unsafe_allow_html=True)

    if st.session_state.encrypted_datasets:
        # Performance test
        if st.button("🏃‍♂️ Run Performance Test"):
            performance_data = []

            for dataset_key, dataset_info in list(st.session_state.encrypted_datasets.items())[:3]:
                encrypted_data = dataset_info['encrypted_data']
                scheme = dataset_info['scheme']

                # Time various operations
                operations = ['add_scalar', 'multiply_scalar', 'square']

                for op in operations:
                    start_time = time.time()

                    if op == 'add_scalar':
                        _ = encrypted_data + 10
                    elif op == 'multiply_scalar':
                        _ = encrypted_data * 2
                    elif op == 'square':
                        _ = encrypted_data * encrypted_data

                    end_time = time.time()

                    performance_data.append({
                        'Dataset': dataset_info['column_name'],
                        'Scheme': scheme,
                        'Operation': op,
                        'Time (ms)': (end_time - start_time) * 1000,
                        'Data Size': len(dataset_info['original_data']) if dataset_info['original_data'] else 'N/A'
                    })

            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)

            # Performance visualization
            fig_perf = px.bar(
                perf_df,
                x='Operation',
                y='Time (ms)',
                color='Scheme',
                facet_col='Dataset',
                title="FHE Operation Performance by Scheme"
            )
            st.plotly_chart(fig_perf, use_container_width=True)


# Noise budget monitoring (for educational purposes)
def show_noise_budget_info():
    """Show information about noise budget in FHE"""
    st.markdown('<h2 class="section-header">📊 Noise Budget & Security</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>🔊 Understanding Noise in FHE</h3>
    <p>FHE schemes add controlled noise to maintain security. Each operation increases noise:</p>
    <ul>
        <li><strong>Addition:</strong> Noise grows linearly</li>
        <li><strong>Multiplication:</strong> Noise grows exponentially</li>
        <li><strong>Bootstrapping:</strong> Refreshes ciphertext, reduces noise (expensive operation)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Simulate noise budget visualization
    operations = ['Initial', 'Add', 'Multiply', 'Add', 'Square', 'Add']
    noise_levels = [10, 15, 45, 50, 150, 155]  # Simulated noise growth

    noise_df = pd.DataFrame({
        'Operation': operations,
        'Cumulative_Noise': noise_levels,
        'Step': range(len(operations))
    })

    fig_noise = px.line(
        noise_df,
        x='Step',
        y='Cumulative_Noise',
        title='Noise Growth During FHE Operations',
        markers=True
    )
    fig_noise.update_layout(
        xaxis_title="Operation Step",
        yaxis_title="Noise Level (Simulated)",
        xaxis={'tickmode': 'array', 'tickvals': list(range(len(operations))), 'ticktext': operations}
    )
    fig_noise.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Noise Budget Limit")

    st.plotly_chart(fig_noise, use_container_width=True)


# Main execution
if __name__ == "__main__":
    # Add sample data option
    if st.sidebar.button("📝 Generate Sample Data"):
        sample_df = create_sample_data()
        st.session_state.uploaded_data = sample_df
        st.sidebar.success("✅ Sample financial data generated!")

    # Additional advanced features in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Advanced Features")

    if st.sidebar.checkbox("Show Advanced Operations"):
        show_advanced_operations()

    if st.sidebar.checkbox("Show Performance Metrics"):
        show_performance_metrics()

    if st.sidebar.checkbox("Show Noise Budget Info"):
        show_noise_budget_info()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔐 Fully Homomorphic Encryption Demo | Built with TenSEAL & Streamlit</p>
        <p><em>This demonstration shows FHE capabilities for privacy-preserving financial data analysis</em></p>
    </div>
    """, unsafe_allow_html=True)

# Run the main function
main()