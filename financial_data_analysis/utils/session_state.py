import streamlit as st
import pandas as pd


def initialize_session_state():
    """Initialize all session state variables"""

    # Data storage
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'account_data' not in st.session_state:
        st.session_state.account_data = None
    if 'transaction_data' not in st.session_state:
        st.session_state.transaction_data = None

    # Encrypted data
    if 'encrypted_data' not in st.session_state:
        st.session_state.encrypted_data = {}

    # Keys
    if 'public_key' not in st.session_state:
        st.session_state.public_key = None
    if 'private_key' not in st.session_state:
        st.session_state.private_key = None
    if 'evaluation_key' not in st.session_state:
        st.session_state.evaluation_key = None
    if 'galois_keys' not in st.session_state:
        st.session_state.galois_keys = None
    if 'relin_keys' not in st.session_state:
        st.session_state.relin_keys = None

    # FHE context
    if 'context' not in st.session_state:
        st.session_state.context = None

    # Selected columns for encryption
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = {
            'user': [],
            'account': [],
            'transaction': []
        }

    # Operation results
    if 'operation_results' not in st.session_state:
        st.session_state.operation_results = None

    # Statistics
    if 'statistics' not in st.session_state:
        st.session_state.statistics = []

    # Scheme parameters
    if 'scheme_params' not in st.session_state:
        st.session_state.scheme_params = {}

    # Key rotation history
    if 'key_history' not in st.session_state:
        st.session_state.key_history = []


def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()