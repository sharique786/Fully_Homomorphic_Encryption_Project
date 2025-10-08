import streamlit as st
import pandas as pd
from utils.data_generator import FinancialDataGenerator
from fhe.tenseal_wrapper import TenSEALWrapper
from fhe.openfhe_wrapper import OpenFHEWrapper
import time
import json


def data_management_page():
    st.title("📊 Data Management & Encryption")
    st.markdown("---")

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Data Source",
            ["Generate Synthetic Data", "Upload CSV Files"],
            horizontal=True
        )

    # Handle data loading
    if data_source == "Generate Synthetic Data":
        handle_synthetic_data()
    else:
        handle_csv_upload()

    st.markdown("---")

    # Display loaded data
    if st.session_state.user_data is not None:
        display_data_tables()
        st.markdown("---")
        handle_encryption()


def handle_synthetic_data():
    """Generate synthetic financial data"""
    st.subheader("Generate Synthetic Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_users = st.number_input("Number of Users", min_value=10, max_value=1000, value=100)
    with col2:
        accounts_per_user = st.number_input("Accounts per User", min_value=1, max_value=5, value=2)
    with col3:
        transactions_per_account = st.number_input("Transactions per Account", min_value=10, max_value=200, value=50)

    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating financial data..."):
            user_data, account_data, transaction_data = FinancialDataGenerator.generate_complete_dataset(
                num_users=num_users,
                accounts_per_user=accounts_per_user,
                transactions_per_account=transactions_per_account
            )

            st.session_state.user_data = user_data
            st.session_state.account_data = account_data
            st.session_state.transaction_data = transaction_data

            st.success(
                f"✅ Generated {len(user_data)} users, {len(account_data)} accounts, and {len(transaction_data)} transactions!")
            st.rerun()


def handle_csv_upload():
    """Handle CSV file uploads"""
    st.subheader("Upload CSV Files")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**User Data**")
        user_file = st.file_uploader("Upload User CSV", type=['csv'], key='user_csv')
        if user_file:
            st.session_state.user_data = pd.read_csv(user_file)
            st.success(f"✅ {len(st.session_state.user_data)} users loaded")

    with col2:
        st.write("**Account Data**")
        account_file = st.file_uploader("Upload Account CSV", type=['csv'], key='account_csv')
        if account_file:
            st.session_state.account_data = pd.read_csv(account_file)
            st.success(f"✅ {len(st.session_state.account_data)} accounts loaded")

    with col3:
        st.write("**Transaction Data**")
        transaction_file = st.file_uploader("Upload Transaction CSV", type=['csv'], key='transaction_csv')
        if transaction_file:
            st.session_state.transaction_data = pd.read_csv(transaction_file)
            st.success(f"✅ {len(st.session_state.transaction_data)} transactions loaded")


def display_data_tables():
    """Display loaded data tables"""
    st.subheader("📋 Loaded Data")

    tab1, tab2, tab3 = st.tabs(["👤 User Data", "🏦 Account Data", "💳 Transaction Data"])

    with tab1:
        if st.session_state.user_data is not None:
            st.dataframe(st.session_state.user_data, use_container_width=True)
            st.caption(f"Total Users: {len(st.session_state.user_data)}")

    with tab2:
        if st.session_state.account_data is not None:
            st.dataframe(st.session_state.account_data, use_container_width=True)
            st.caption(f"Total Accounts: {len(st.session_state.account_data)}")

    with tab3:
        if st.session_state.transaction_data is not None:
            st.dataframe(st.session_state.transaction_data, use_container_width=True)
            st.caption(f"Total Transactions: {len(st.session_state.transaction_data)}")


def handle_encryption():
    """Handle data encryption"""
    st.subheader("🔐 Data Encryption")

    # Column selection
    st.write("**Select Columns to Encrypt**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**User Columns**")
        if st.session_state.user_data is not None:
            user_cols = st.multiselect(
                "Select user columns",
                st.session_state.user_data.columns.tolist(),
                key='user_cols_select'
            )
            st.session_state.selected_columns['user'] = user_cols

    with col2:
        st.write("**Account Columns**")
        if st.session_state.account_data is not None:
            account_cols = st.multiselect(
                "Select account columns",
                st.session_state.account_data.columns.tolist(),
                key='account_cols_select'
            )
            st.session_state.selected_columns['account'] = account_cols

    with col3:
        st.write("**Transaction Columns**")
        if st.session_state.transaction_data is not None:
            transaction_cols = st.multiselect(
                "Select transaction columns",
                st.session_state.transaction_data.columns.tolist(),
                key='transaction_cols_select'
            )
            st.session_state.selected_columns['transaction'] = transaction_cols

    # Key management
    st.markdown("---")
    configure_keys_and_encrypt()


def configure_keys_and_encrypt():
    """Configure keys and perform encryption"""
    st.subheader("🔑 Key Management & Encryption")

    col1, col2 = st.columns([2, 1])

    with col1:
        key_option = st.radio(
            "Key Management",
            ["Generate New Keys", "Use Existing Keys"],
            horizontal=True
        )

    if key_option == "Generate New Keys":
        generate_new_keys()
    else:
        use_existing_keys()

    # Encryption button
    st.markdown("---")
    if st.button("🔒 Encrypt Selected Data", type="primary", use_container_width=True):
        perform_encryption()


def generate_new_keys():
    """Generate new encryption keys"""
    st.write("**Key Generation Parameters**")

    library = st.session_state.fhe_library

    col1, col2, col3 = st.columns(3)

    with col1:
        scheme = st.selectbox(
            "Encryption Scheme",
            ["CKKS", "BFV", "BGV"] if library == "OpenFHE" else ["CKKS", "BFV"]
        )

    with col2:
        if scheme == "CKKS":
            poly_mod = st.selectbox("Polynomial Modulus Degree", [8192, 16384, 32768], index=0)
        else:
            poly_mod = st.selectbox("Polynomial Modulus Degree", [4096, 8192, 16384], index=1)

    with col3:
        if scheme == "CKKS":
            scale = st.selectbox("Scale (2^x)", [30, 40, 50], index=1)
        else:
            plain_mod = st.number_input("Plain Modulus", value=1032193)

    if st.button("Generate Keys"):
        with st.spinner("Generating encryption keys..."):
            if library == "TenSEAL":
                wrapper = TenSEALWrapper()
                if scheme == "CKKS":
                    wrapper.generate_context(
                        scheme=scheme,
                        poly_modulus_degree=poly_mod,
                        scale=2 ** scale
                    )
                else:
                    wrapper.generate_context(
                        scheme=scheme,
                        poly_modulus_degree=poly_mod
                    )
            else:
                wrapper = OpenFHEWrapper()
                wrapper.generate_context(
                    scheme=scheme,
                    ring_dim=poly_mod
                )

            st.session_state.context = wrapper
            keys_info = wrapper.get_keys_info()

            st.success("✅ Keys generated successfully!")

            # Display keys
            display_keys(keys_info)


def display_keys(keys_info):
    """Display generated keys"""
    st.write("**Generated Keys (Save these securely)**")

    tab1, tab2, tab3, tab4 = st.tabs(["Public Key", "Private Key", "Evaluation Keys", "Additional Keys"])

    with tab1:
        st.text_area("Public Key (Truncated)", keys_info['public_key'], height=100)
        st.download_button(
            "Download Full Public Key",
            keys_info['full_public_key'],
            "public_key.txt",
            "text/plain"
        )

    with tab2:
        st.text_area("Private Key (Truncated)", keys_info['private_key'], height=100)
        st.download_button(
            "Download Full Private Key",
            keys_info['full_private_key'],
            "private_key.txt",
            "text/plain"
        )

    with tab3:
        if 'evaluation_key' in keys_info:
            st.text_area("Evaluation Key (Truncated)", keys_info['evaluation_key'], height=100)
        else:
            st.info("Evaluation keys are embedded in the context")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Galois Keys", keys_info.get('galois_keys', 'N/A'), height=100)
        with col2:
            st.text_area("Relinearization Keys", keys_info.get('relin_keys', 'N/A'), height=100)

    # Key rotation option
    st.markdown("---")
    if st.button("🔄 Rotate Keys"):
        rotate_keys()


def rotate_keys():
    """Rotate encryption keys"""
    if st.session_state.context:
        with st.spinner("Rotating keys..."):
            rotation_info = st.session_state.context.rotate_keys()
            st.session_state.key_history.append(rotation_info)
            st.success("✅ Keys rotated successfully! Old data remains accessible with backward compatibility.")
            st.info(f"Rotation performed at: {rotation_info['rotation_time']}")
    else:
        st.error("No context available. Please generate keys first.")


def use_existing_keys():
    """Use existing keys"""
    st.write("**Upload Existing Keys**")

    col1, col2 = st.columns(2)

    with col1:
        public_key_file = st.file_uploader("Upload Public Key", type=['txt', 'pem'], key='pub_key_upload')

    with col2:
        private_key_file = st.file_uploader("Upload Private Key", type=['txt', 'pem'], key='priv_key_upload')

    if public_key_file and private_key_file:
        st.info("⚠️ Key import functionality requires custom implementation based on key format.")


def perform_encryption():
    """Perform encryption on selected columns"""
    if not st.session_state.context:
        st.error("❌ Please generate or upload keys first!")
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
        wrapper = st.session_state.context

        # Get scheme limitations
        limitations = wrapper.get_scheme_limitations()

        # Encrypt user data
        if st.session_state.selected_columns['user']:
            encrypt_table_columns(
                st.session_state.user_data,
                st.session_state.selected_columns['user'],
                'user',
                wrapper,
                limitations
            )

        # Encrypt account data
        if st.session_state.selected_columns['account']:
            encrypt_table_columns(
                st.session_state.account_data,
                st.session_state.selected_columns['account'],
                'account',
                wrapper,
                limitations
            )

        # Encrypt transaction data
        if st.session_state.selected_columns['transaction']:
            encrypt_table_columns(
                st.session_state.transaction_data,
                st.session_state.selected_columns['transaction'],
                'transaction',
                wrapper,
                limitations
            )

        encryption_time = time.time() - start_time

        # Store statistics
        st.session_state.statistics.append({
            'operation': 'encryption',
            'scheme': wrapper.scheme,
            'library': st.session_state.fhe_library,
            'time': encryption_time,
            'columns_encrypted': sum([
                len(st.session_state.selected_columns['user']),
                len(st.session_state.selected_columns['account']),
                len(st.session_state.selected_columns['transaction'])
            ])
        })

        st.success(f"✅ Data encrypted successfully in {encryption_time:.2f} seconds!")
        st.info(f"Scheme: {wrapper.scheme} | Library: {st.session_state.fhe_library}")


def encrypt_table_columns(data, columns, table_name, wrapper, limitations):
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
        if data_type == 'text' and limitations.get('supports_text') == 'No':
            st.warning(
                f"⚠️ Column '{col}' is text but scheme {wrapper.scheme} doesn't support text encryption well. Skipping.")
            continue

        try:
            encrypted_values = wrapper.encrypt_data(data[col].tolist(), col, data_type)

            if table_name not in st.session_state.encrypted_data:
                st.session_state.encrypted_data[table_name] = {}

            st.session_state.encrypted_data[table_name][col] = {
                'encrypted': encrypted_values,
                'data_type': data_type,
                'original_length': len(data)
            }
        except Exception as e:
            st.error(f"❌ Error encrypting column '{col}': {str(e)}")