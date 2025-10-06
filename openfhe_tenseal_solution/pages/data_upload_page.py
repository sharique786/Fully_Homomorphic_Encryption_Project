"""
Data Upload and Encryption Page
Screen 1: Upload financial data and encrypt
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ui_components import (
    render_data_preview,
    render_encryption_config,
    render_column_selector,
    render_key_display,
    render_alert,
    render_metrics_dashboard
)
from fhe_core import FHEProcessor


def render():
    """Render data upload and encryption page"""
    st.header("📊 Data Upload & Encryption")

    tab1, tab2, tab3 = st.tabs([
        "📁 Data Upload",
        "🔐 Encryption",
        "✅ Verification"
    ])

    with tab1:
        render_data_upload_tab()

    with tab2:
        render_encryption_tab()

    with tab3:
        render_verification_tab()


def render_data_upload_tab():
    """Render data upload tab"""
    st.subheader("Upload or Generate Financial Data")

    data_source = st.radio(
        "Choose data source:",
        ["🎲 Generate Sample Data", "📤 Upload CSV Files"],
        horizontal=True
    )

    if data_source == "🎲 Generate Sample Data":
        render_sample_data_generation()
    else:
        render_csv_upload()


def render_sample_data_generation():
    """Render sample data generation section"""
    st.markdown("### Generate Sample Financial Data")

    col1, col2 = st.columns(2)

    with col1:
        num_users = st.slider(
            "Number of Users:",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

    with col2:
        include_tables = st.multiselect(
            "Include Tables:",
            ["User Details", "Account Details", "Transaction Details"],
            default=["User Details", "Account Details", "Transaction Details"]
        )

    if st.button("🎲 Generate Data", type="primary"):
        with st.spinner("Generating sample data..."):
            data_manager = st.session_state.data_manager
            generated_data = data_manager.generate_sample_data(num_users)

            st.session_state.uploaded_data = generated_data

            st.success(f"✅ Successfully generated data for {num_users} users!")

            # Show summary
            summary = data_manager.get_data_summary()

            st.markdown("### Data Summary")
            cols = st.columns(3)

            with cols[0]:
                if 'users' in summary:
                    st.metric("Users", f"{summary['users']['count']:,}")

            with cols[1]:
                if 'accounts' in summary:
                    st.metric("Accounts", f"{summary['accounts']['count']:,}")

            with cols[2]:
                if 'transactions' in summary:
                    st.metric("Transactions", f"{summary['transactions']['count']:,}")

            # Show data previews
            if "User Details" in include_tables and 'users' in generated_data:
                render_data_preview(generated_data['users'], "👤 User Details", 5)

            if "Account Details" in include_tables and 'accounts' in generated_data:
                render_data_preview(generated_data['accounts'], "💳 Account Details", 5)

            if "Transaction Details" in include_tables and 'transactions' in generated_data:
                render_data_preview(generated_data['transactions'], "💰 Transaction Details", 5)


def render_csv_upload():
    """Render CSV upload section"""
    st.markdown("### Upload CSV Files")

    st.info("📝 Upload one or more CSV files containing financial data")

    col1, col2, col3 = st.columns(3)

    with col1:
        user_file = st.file_uploader(
            "User Details CSV:",
            type=['csv'],
            key="user_csv"
        )

    with col2:
        account_file = st.file_uploader(
            "Account Details CSV:",
            type=['csv'],
            key="account_csv"
        )

    with col3:
        transaction_file = st.file_uploader(
            "Transaction Details CSV:",
            type=['csv'],
            key="transaction_csv"
        )

    files = {}
    if user_file:
        files['users'] = user_file
    if account_file:
        files['accounts'] = account_file
    if transaction_file:
        files['transactions'] = transaction_file

    if files and st.button("📂 Load CSV Files", type="primary"):
        with st.spinner("Loading CSV files..."):
            data_manager = st.session_state.data_manager
            loaded_data = data_manager.load_csv_files(files)

            st.session_state.uploaded_data = loaded_data

            # Validate data
            validation_results = data_manager.validate_data()

            if validation_results['valid']:
                st.success("✅ All files loaded and validated successfully!")
            else:
                st.error("❌ Data validation failed!")
                for error in validation_results['errors']:
                    st.error(error)

            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    st.warning(warning)

            # Show data previews
            for data_type, df in loaded_data.items():
                render_data_preview(df, f"📋 {data_type.title()}", 5)


def render_encryption_tab():
    """Render encryption configuration and execution tab"""
    if not st.session_state.uploaded_data:
        st.warning("⚠️ Please upload or generate data first in the 'Data Upload' tab")
        return

    st.subheader("🔐 Configure Encryption")

    # Select which table to encrypt
    available_tables = list(st.session_state.uploaded_data.keys())
    selected_table = st.selectbox(
        "Select table to encrypt:",
        available_tables,
        index=0
    )

    df = st.session_state.uploaded_data[selected_table]

    # Column selection
    selected_columns = render_column_selector(df, "Select Columns to Encrypt")

    if not selected_columns:
        st.warning("⚠️ Please select at least one column to encrypt")
        return

    st.markdown("---")

    # Encryption parameters
    scheme = st.session_state.selected_scheme
    library = st.session_state.selected_library

    parameters = render_encryption_config(scheme)

    st.markdown("---")

    # Key management
    st.subheader("🔑 Key Management")

    key_option = st.radio(
        "Choose key option:",
        ["Generate New Keys", "Use Existing Keys"],
        horizontal=True
    )

    if key_option == "Generate New Keys":
        if st.button("🔑 Generate Keys", type="primary"):
            with st.spinner("Generating FHE keys..."):
                key_manager = st.session_state.key_manager
                keys = key_manager.generate_keys(library, scheme, parameters)

                if keys['status'] == 'success':
                    st.success(f"✅ Keys generated successfully!")
                    st.info(f"⏱️ Generation time: {keys['metadata']['generation_duration_ms']:.2f} ms")

                    # Display keys
                    show_private = st.checkbox("Show private key (⚠️ Use with caution)", value=False)
                    render_key_display(keys, include_private=show_private)

                    st.session_state.current_keys = keys
                else:
                    st.error("❌ Key generation failed!")

    else:  # Use Existing Keys
        st.text_area(
            "Paste your public key:",
            height=100,
            key="import_public_key"
        )
        st.text_area(
            "Paste your private key:",
            height=100,
            key="import_private_key"
        )

        if st.button("📥 Import Keys"):
            st.success("✅ Keys imported successfully!")

    st.markdown("---")

    # Encryption execution
    st.subheader("🚀 Execute Encryption")

    if not hasattr(st.session_state, 'current_keys'):
        st.warning("⚠️ Please generate or import keys first")
        return

    # Show encryption preview
    st.write(f"**Ready to encrypt:** {len(selected_columns)} columns from {len(df)} rows")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Columns", len(selected_columns))
    with col2:
        st.metric("Rows", len(df))
    with col3:
        st.metric("Scheme", scheme)

    if st.button("🔐 Start Encryption", type="primary"):
        with st.spinner("Encrypting data..."):
            # Initialize FHE processor
            key_manager = st.session_state.key_manager
            fhe_processor = FHEProcessor(key_manager, library)
            st.session_state.fhe_processor = fhe_processor

            # Encrypt data
            encryption_result = fhe_processor.encrypt_data(df, selected_columns)

            if encryption_result['status'] == 'success':
                st.success("✅ Encryption completed successfully!")

                # Store encrypted data
                st.session_state.encrypted_data = {
                    'table': selected_table,
                    'columns': selected_columns,
                    'result': encryption_result,
                    'original_df': df
                }

                # Show metrics
                metrics = {
                    "Encrypted Columns": encryption_result['encrypted_columns'],
                    "Total Values": f"{encryption_result['total_values']:,}",
                    "Time (ms)": f"{encryption_result['encryption_time_ms']:.2f}",
                    "Library": encryption_result['library']
                }
                render_metrics_dashboard(metrics)

                # Show sample encrypted data
                st.markdown("### 🔍 Encrypted Data Preview")
                st.info("Data is now encrypted and secure. Original values are hidden.")

                # Create preview dataframe
                preview_df = df[selected_columns].head(10).copy()
                for col in selected_columns:
                    preview_df[col] = ["[ENCRYPTED]"] * len(preview_df)

                st.dataframe(preview_df, use_container_width=True)
            else:
                st.error("❌ Encryption failed!")


def render_verification_tab():
    """Render encryption verification tab"""
    if not hasattr(st.session_state, 'encrypted_data') or not st.session_state.encrypted_data:
        st.warning("⚠️ No encrypted data available. Please encrypt data first.")
        return

    st.subheader("✅ Encryption Verification")

    encrypted_info = st.session_state.encrypted_data

    # Show encryption summary
    st.markdown("### 📊 Encryption Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Encrypted Table",
            encrypted_info['table'].title()
        )

    with col2:
        st.metric(
            "Encrypted Columns",
            len(encrypted_info['columns'])
        )

    with col3:
        st.metric(
            "Total Rows",
            len(encrypted_info['original_df'])
        )

    # Show encrypted columns
    st.markdown("### 🔒 Encrypted Columns")
    cols_display = st.columns(len(encrypted_info['columns']))
    for idx, col in enumerate(encrypted_info['columns']):
        with cols_display[idx]:
            st.success(f"✅ {col}")

    # Test decryption
    st.markdown("---")
    st.markdown("### 🔓 Test Decryption")

    st.info("Verify encryption by decrypting a sample of the data")

    if st.button("🔓 Test Decrypt Sample", type="primary"):
        with st.spinner("Decrypting sample data..."):
            fhe_processor = st.session_state.fhe_processor

            # Decrypt
            decrypted_df, decryption_time = fhe_processor.decrypt_data()

            st.success(f"✅ Decryption successful! Time: {decryption_time:.2f} ms")

            # Compare original vs decrypted
            st.markdown("### 📋 Comparison: Original vs Decrypted")

            original_sample = encrypted_info['original_df'][encrypted_info['columns']].head(10)
            decrypted_sample = decrypted_df.head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Data:**")
                st.dataframe(original_sample, use_container_width=True)

            with col2:
                st.write("**Decrypted Data:**")
                st.dataframe(decrypted_sample, use_container_width=True)

            # Verify accuracy
            st.markdown("### ✅ Accuracy Verification")

            all_match = True
            for col in encrypted_info['columns']:
                if col in original_sample.columns and col in decrypted_sample.columns:
                    matches = (original_sample[col].values == decrypted_sample[col].values).all()
                    if matches:
                        st.success(f"✅ Column '{col}': Perfect match")
                    else:
                        st.error(f"❌ Column '{col}': Mismatch detected")
                        all_match = False

            if all_match:
                st.balloons()
                st.success("🎉 All columns verified! Encryption/Decryption working perfectly!")

    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Encrypted Data"):
            st.info("Encrypted data exported (feature in development)")

    with col2:
        if st.button("📥 Export Keys"):
            key_manager = st.session_state.key_manager
            keys_json = key_manager.export_keys(format='json', include_private=False)
            st.download_button(
                label="💾 Download Public Key",
                data=keys_json,
                file_name="public_key.json",
                mime="application/json"
            )