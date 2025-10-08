"""
Helper utility functions for FHE Financial Processor
"""

import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json


def detect_column_type(series):
    """
    Detect the data type of a pandas series

    Args:
        series: pandas Series

    Returns:
        str: 'numeric', 'text', or 'date'
    """
    # Check if datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'date'

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'

    # Check column name for date keywords
    col_name = series.name.lower() if series.name else ''
    date_keywords = ['date', 'time', 'timestamp', 'created', 'updated']
    if any(keyword in col_name for keyword in date_keywords):
        return 'date'

    # Check numeric keywords
    numeric_keywords = ['amount', 'balance', 'price', 'value', 'count', 'age', 'quantity']
    if any(keyword in col_name for keyword in numeric_keywords):
        # Try to convert to numeric
        try:
            pd.to_numeric(series, errors='coerce')
            return 'numeric'
        except:
            pass

    # Default to text
    return 'text'


def validate_csv_structure(df, expected_columns=None):
    """
    Validate CSV structure

    Args:
        df: pandas DataFrame
        expected_columns: list of expected column names (optional)

    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"

    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {', '.join(missing_cols)}"

    return True, "Valid"


def format_time(seconds):
    """
    Format seconds into human-readable time

    Args:
        seconds: float

    Returns:
        str: formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_hash(data):
    """
    Calculate SHA256 hash of data

    Args:
        data: any serializable data

    Returns:
        str: hex hash
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def export_keys_to_dict(keys_info):
    """
    Export keys information to dictionary format

    Args:
        keys_info: dict containing key information

    Returns:
        dict: formatted keys dictionary
    """
    return {
        'public_key': keys_info.get('full_public_key', ''),
        'private_key': keys_info.get('full_private_key', ''),
        'evaluation_key': keys_info.get('full_evaluation_key', ''),
        'timestamp': datetime.now().isoformat(),
        'key_type': 'FHE_KEYS'
    }


def merge_dataframes(user_df, account_df, transaction_df):
    """
    Merge user, account, and transaction dataframes

    Args:
        user_df: user DataFrame
        account_df: account DataFrame
        transaction_df: transaction DataFrame

    Returns:
        DataFrame: merged DataFrame
    """
    # Merge transactions with accounts
    if account_df is not None and transaction_df is not None:
        merged = transaction_df.merge(
            account_df[['account_id', 'user_id', 'account_type']],
            on='account_id',
            how='left',
            suffixes=('', '_account')
        )
    else:
        merged = transaction_df.copy() if transaction_df is not None else None

    # Merge with users
    if user_df is not None and merged is not None:
        merged = merged.merge(
            user_df[['user_id', 'name', 'email']],
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

    return merged


def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column

    Args:
        df: pandas DataFrame
        column: column name

    Returns:
        dict: statistics dictionary
    """
    if column not in df.columns:
        return {}

    series = df[column]

    stats = {
        'count': len(series),
        'missing': series.isna().sum(),
        'unique': series.nunique()
    }

    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75)
        })

    return stats


def filter_dataframe(df, filters):
    """
    Apply filters to DataFrame

    Args:
        df: pandas DataFrame
        filters: dict of {column: filter_value} or {column: (operator, value)}

    Returns:
        DataFrame: filtered DataFrame
    """
    filtered_df = df.copy()

    for column, filter_spec in filters.items():
        if column not in filtered_df.columns:
            continue

        if isinstance(filter_spec, tuple):
            operator, value = filter_spec
            if operator == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == 'in':
                filtered_df = filtered_df[filtered_df[column].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[column] == filter_spec]

    return filtered_df


def encode_text_to_numeric(text):
    """
    Encode text to numeric value for encryption

    Args:
        text: string

    Returns:
        int: numeric encoding
    """
    if not text or pd.isna(text):
        return 0

    # Method 1: Sum of ASCII values (simple but works)
    return sum([ord(c) for c in str(text)])


def decode_numeric_to_text(numeric_value, original_mapping=None):
    """
    Decode numeric value back to text (if mapping provided)

    Args:
        numeric_value: int
        original_mapping: dict mapping numeric values to original text

    Returns:
        str: decoded text or numeric representation
    """
    if original_mapping and numeric_value in original_mapping:
        return original_mapping[numeric_value]

    return f"ENCODED_{numeric_value}"


def generate_sample_keys():
    """
    Generate sample keys for demonstration

    Returns:
        dict: sample keys
    """
    import secrets

    return {
        'public_key': secrets.token_hex(128),
        'private_key': secrets.token_hex(128),
        'evaluation_key': secrets.token_hex(64),
        'timestamp': datetime.now().isoformat()
    }


def benchmark_operation(func, *args, **kwargs):
    """
    Benchmark a function execution

    Args:
        func: function to benchmark
        *args, **kwargs: function arguments

    Returns:
        tuple: (result, execution_time)
    """
    import time

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time

    return result, execution_time


def create_backup(session_state_data):
    """
    Create backup of session state

    Args:
        session_state_data: dict of session state

    Returns:
        str: JSON backup string
    """
    backup = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'data': {}
    }

    # Only backup serializable data
    for key, value in session_state_data.items():
        try:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                backup['data'][key] = value.to_dict()
            elif isinstance(value, (dict, list, str, int, float, bool)):
                backup['data'][key] = value
        except:
            pass

    return json.dumps(backup, default=str)


def restore_backup(backup_string):
    """
    Restore session state from backup

    Args:
        backup_string: JSON backup string

    Returns:
        dict: restored session state data
    """
    backup = json.loads(backup_string)
    restored_data = {}

    for key, value in backup['data'].items():
        if isinstance(value, dict) and 'columns' in str(value):
            # Likely a DataFrame
            try:
                restored_data[key] = pd.DataFrame(value)
            except:
                restored_data[key] = value
        else:
            restored_data[key] = value

    return restored_data


def get_memory_usage(df):
    """
    Get memory usage of DataFrame

    Args:
        df: pandas DataFrame

    Returns:
        str: formatted memory usage
    """
    memory_bytes = df.memory_usage(deep=True).sum()

    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 ** 3:
        return f"{memory_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_bytes / (1024 ** 3):.2f} GB"


def validate_date_range(start_date, end_date):
    """
    Validate date range

    Args:
        start_date: start date
        end_date: end date

    Returns:
        tuple: (is_valid, error_message)
    """
    if start_date > end_date:
        return False, "Start date must be before end date"

    date_diff = (end_date - start_date).days
    if date_diff > 365 * 10:
        return False, "Date range too large (max 10 years)"

    return True, "Valid"


def optimize_dataframe(df):
    """
    Optimize DataFrame memory usage

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame: optimized DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Try to convert to category if unique values < 50%
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        elif col_type in ['int64', 'int32']:
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')

        elif col_type in ['float64', 'float32']:
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')

    return df