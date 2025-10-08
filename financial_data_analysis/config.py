"""
Configuration settings for FHE Financial Processor
"""

# OpenFHE Configuration
OPENFHE_CONFIG = {
    'install_path': r"C:\Program Files (x86)\OpenFHE",
    'build_path': r"C:\Users\alish\Workspaces\Python\openfhe-development",
    'library_name': 'libOPENFHEcore.dll'  # or .so for Linux
}

# Default Encryption Parameters
DEFAULT_PARAMS = {
    'CKKS': {
        'poly_modulus_degree': 8192,
        'coeff_mod_bit_sizes': [60, 40, 40, 60],
        'scale': 2**40,
        'security_level': 'HEStd_128_classic'
    },
    'BFV': {
        'poly_modulus_degree': 8192,
        'plain_modulus': 1032193,
        'security_level': 'HEStd_128_classic'
    },
    'BGV': {
        'poly_modulus_degree': 8192,
        'plain_modulus': 1032193,
        'security_level': 'HEStd_128_classic'
    }
}

# Data Generation Defaults
DATA_GENERATION_DEFAULTS = {
    'num_users': 100,
    'accounts_per_user': 2,
    'transactions_per_account': 50
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'FHE Financial Data Processor',
    'page_icon': '🔐',
    'layout': 'wide',
    'max_upload_size': 200  # MB
}

# Scheme Characteristics
SCHEME_INFO = {
    'CKKS': {
        'name': 'CKKS',
        'type': 'Approximate',
        'supports_float': True,
        'supports_integer': True,
        'supports_text': 'Limited',
        'operations': ['Addition', 'Multiplication', 'Rotation', 'Conjugation'],
        'use_cases': ['Machine Learning', 'Signal Processing', 'Analytics'],
        'description': 'Efficient for approximate arithmetic on real/complex numbers'
    },
    'BFV': {
        'name': 'BFV',
        'type': 'Exact',
        'supports_float': False,
        'supports_integer': True,
        'supports_text': 'Limited',
        'operations': ['Addition', 'Multiplication'],
        'use_cases': ['Database Operations', 'Exact Computations'],
        'description': 'Exact integer arithmetic with noise management'
    },
    'BGV': {
        'name': 'BGV',
        'type': 'Exact',
        'supports_float': False,
        'supports_integer': True,
        'supports_text': 'Limited',
        'operations': ['Addition', 'Multiplication', 'Rotation'],
        'use_cases': ['General Purpose', 'Leveled Computations'],
        'description': 'Versatile scheme with modulus switching'
    }
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'fast': 1.0,      # seconds
    'medium': 5.0,    # seconds
    'slow': 10.0      # seconds
}

# File Upload Settings
UPLOAD_SETTINGS = {
    'max_file_size': 200 * 1024 * 1024,  # 200 MB in bytes
    'allowed_extensions': ['csv', 'xlsx', 'xls'],
    'encoding': 'utf-8'
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    'color_scheme': 'plotly',
    'chart_height': 400,
    'chart_template': 'plotly_white'
}

# Security Settings
SECURITY_LEVELS = {
    'HEStd_128_classic': 'Standard 128-bit security',
    'HEStd_192_classic': 'High 192-bit security',
    'HEStd_256_classic': 'Very High 256-bit security'
}

# Warning Messages
WARNINGS = {
    'text_bfv': 'BFV scheme does not support text encryption directly. Consider using CKKS or encoding text as integers.',
    'large_modulus': 'Large polynomial modulus degrees will increase computation time significantly.',
    'no_keys': 'No encryption keys found. Please generate or upload keys first.',
    'no_data': 'No data available. Please load or generate data first.',
    'scheme_change': 'Changing scheme requires re-encryption of data.'
}

# Error Messages
ERRORS = {
    'encryption_failed': 'Encryption failed. Please check your data and parameters.',
    'decryption_failed': 'Decryption failed. Please check your keys.',
    'key_generation_failed': 'Key generation failed. Please try different parameters.',
    'library_not_found': 'FHE library not found. Please check installation.'
}

# Success Messages
SUCCESS = {
    'data_generated': 'Data generated successfully!',
    'data_uploaded': 'Data uploaded successfully!',
    'keys_generated': 'Keys generated successfully!',
    'encryption_complete': 'Encryption completed successfully!',
    'operation_complete': 'Operation completed successfully!'
}

# Column Type Detection
COLUMN_TYPES = {
    'numeric_keywords': ['amount', 'balance', 'price', 'value', 'count', 'quantity', 'age'],
    'text_keywords': ['name', 'email', 'address', 'description', 'type', 'status'],
    'date_keywords': ['date', 'time', 'timestamp', 'created', 'updated']
}

# Export Settings
EXPORT_CONFIG = {
    'key_format': 'txt',
    'data_format': 'csv',
    'statistics_format': 'csv'
}