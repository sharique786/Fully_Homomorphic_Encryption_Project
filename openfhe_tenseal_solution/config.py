"""
Configuration file for FHE Financial Analytics System
Contains all constants, settings, and configuration parameters
"""

# Application Configuration
APP_CONFIG = {
    'app_name': 'FHE Financial Analytics',
    'version': '1.0.0',
    'max_file_size_mb': 200,
    'default_batch_size': 100,
    'temp_dir': './temp',
    'keys_dir': './keys',
    'export_dir': './exports'
}

# FHE Library Options
LIBRARY_OPTIONS = {
    'TenSEAL': {
        'available': True,
        'schemes': ['BFV', 'CKKS'],
        'description': 'TenSEAL - Fast and user-friendly FHE library'
    },
    'OpenFHE': {
        'available': True,
        'schemes': ['BFV', 'BGV', 'CKKS'],
        'description': 'OpenFHE - Comprehensive FHE implementation with C++ backend'
    }
}

# FHE Schemes Configuration
FHE_SCHEMES = {
    'BFV': {
        'name': 'Brakerski-Fan-Vercauteren',
        'type': 'Integer Arithmetic',
        'poly_modulus_degrees': [2048, 4096, 8192, 16384],
        'default_poly_modulus': 8192,
        'plain_modulus_options': [1032193, 786433, 65537],
        'default_plain_modulus': 1032193,
        'security_levels': [128, 192, 256],
        'default_security': 128,
        'supports_batching': True,
        'precision': 'Exact',
        'use_cases': ['Integer operations', 'Financial calculations', 'Database queries']
    },
    'BGV': {
        'name': 'Brakerski-Gentry-Vaikuntanathan',
        'type': 'Integer Arithmetic',
        'poly_modulus_degrees': [2048, 4096, 8192, 16384],
        'default_poly_modulus': 8192,
        'plain_modulus_options': [1032193, 786433, 65537],
        'default_plain_modulus': 786433,
        'security_levels': [128, 192, 256],
        'default_security': 128,
        'supports_batching': True,
        'precision': 'Exact',
        'use_cases': ['SIMD operations', 'Parallel processing', 'Complex queries']
    },
    'CKKS': {
        'name': 'Cheon-Kim-Kim-Song',
        'type': 'Approximate Arithmetic',
        'poly_modulus_degrees': [2048, 4096, 8192, 16384],
        'default_poly_modulus': 8192,
        'scale_factors': [20, 30, 40, 50, 60],
        'default_scale_factor': 40,
        'security_levels': [128, 192, 256],
        'default_security': 128,
        'supports_batching': True,
        'precision': 'Approximate',
        'use_cases': ['Real numbers', 'Machine learning', 'Statistical analysis']
    }
}

# Key Generation Parameters
KEY_GENERATION_CONFIG = {
    'key_types': ['Public Key', 'Private Key', 'Evaluation Key', 'Relinearization Key', 'Galois Keys'],
    'key_formats': ['JSON', 'Binary', 'Base64'],
    'default_format': 'Base64',
    'rotation_indices': [1, 2, 3, 4, 5, -1, -2, -3, -4, -5],
    'backup_enabled': True
}

# Data Schema for Financial Transactions
FINANCIAL_DATA_SCHEMA = {
    'user_details': {
        'required_columns': ['user_id', 'name', 'address'],
        'optional_columns': ['email', 'phone', 'date_of_birth', 'country'],
        'sensitive_columns': ['user_id', 'name', 'address', 'email', 'phone']
    },
    'account_details': {
        'required_columns': ['user_id', 'account_number', 'account_type'],
        'optional_columns': ['balance', 'opening_date', 'status', 'branch'],
        'sensitive_columns': ['account_number', 'balance']
    },
    'transaction_details': {
        'required_columns': ['user_id', 'transaction_id', 'amount', 'currency', 'transaction_date'],
        'optional_columns': ['merchant', 'category', 'status', 'account_number'],
        'sensitive_columns': ['transaction_id', 'amount']
    }
}

# Operations Configuration
FHE_OPERATIONS = {
    'basic': {
        'addition': {'supported': True, 'noise_growth': 'Low'},
        'subtraction': {'supported': True, 'noise_growth': 'Low'},
        'multiplication': {'supported': True, 'noise_growth': 'High'},
        'scalar_multiplication': {'supported': True, 'noise_growth': 'Medium'}
    },
    'advanced': {
        'sum': {'supported': True, 'description': 'Sum encrypted values'},
        'mean': {'supported': True, 'description': 'Calculate mean'},
        'variance': {'supported': True, 'description': 'Calculate variance'},
        'count': {'supported': True, 'description': 'Count records'},
        'filter': {'supported': True, 'description': 'Filter by condition'},
        'group_by': {'supported': True, 'description': 'Group and aggregate'}
    }
}

# Performance Metrics
PERFORMANCE_METRICS = {
    'encryption_time': {'unit': 'milliseconds', 'target': 1000},
    'operation_time': {'unit': 'milliseconds', 'target': 500},
    'decryption_time': {'unit': 'milliseconds', 'target': 800},
    'memory_usage': {'unit': 'MB', 'target': 1000},
    'noise_budget': {'unit': 'bits', 'minimum': 20},
    'ciphertext_size': {'unit': 'KB', 'acceptable_range': [100, 10000]}
}

# UI Colors and Themes
UI_THEME = {
    'primary_color': '#1E88E5',
    'secondary_color': '#764ba2',
    'success_color': '#4CAF50',
    'warning_color': '#FF9800',
    'error_color': '#F44336',
    'info_color': '#2196F3',
    'chart_colors': ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
}

# Export Settings
EXPORT_CONFIG = {
    'formats': ['CSV', 'JSON', 'Excel'],
    'include_metadata': True,
    'include_keys': False,  # Security: Don't export keys by default
    'compress': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'fhe_analytics.log',
    'max_size_mb': 10,
    'backup_count': 5
}

# Currency Options
CURRENCY_OPTIONS = [
    'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD',
    'CNY', 'INR', 'BRL', 'MXN', 'ZAR'
]

# Transaction Categories
TRANSACTION_CATEGORIES = [
    'Groceries', 'Dining', 'Transportation', 'Entertainment',
    'Shopping', 'Bills', 'Healthcare', 'Education',
    'Travel', 'Investment', 'Other'
]

# Sample Data Generation Config
SAMPLE_DATA_CONFIG = {
    'num_users': 100,
    'num_accounts_per_user': [1, 2, 3],
    'num_transactions_per_account': [10, 50],
    'transaction_amount_range': [10, 10000],
    'date_range_days': 365
}

# Account Types
ACCOUNT_TYPES = [
    'Savings',
    'Checking',
    'Credit Card',
    'Investment',
    'Money Market',
    'Certificate of Deposit'
]

# Transaction Status Options
TRANSACTION_STATUS = [
    'Completed',
    'Pending',
    'Failed',
    'Cancelled',
    'Processing'
]

# Transaction Types
TRANSACTION_TYPES = [
    'Debit',
    'Credit',
    'Transfer',
    'Withdrawal',
    'Deposit'
]

# Merchant Categories
MERCHANT_CATEGORIES = [
    'Retail',
    'Online',
    'Gas Station',
    'Restaurant',
    'Grocery Store',
    'Pharmacy',
    'Utility',
    'Insurance',
    'Other'
]

# Date Format Options
DATE_FORMATS = {
    'display': '%Y-%m-%d',
    'export': '%Y%m%d',
    'timestamp': '%Y-%m-%d %H:%M:%S'
}

# Validation Rules
VALIDATION_RULES = {
    'user_id': {
        'pattern': r'^USR_\d{5}$',
        'required': True
    },
    'account_number': {
        'pattern': r'^\d{9,12}$',
        'required': True
    },
    'transaction_id': {
        'pattern': r'^TXN_\d{8}$',
        'required': True
    },
    'amount': {
        'min': 0.01,
        'max': 1000000,
        'required': True
    },
    'email': {
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'required': False
    }
}

# Noise Budget Thresholds
NOISE_BUDGET_THRESHOLDS = {
    'critical': 10,    # Below this, operations may fail
    'warning': 30,     # Warning level
    'safe': 50,        # Safe operating range
    'optimal': 80      # Optimal range
}

# Encryption Settings
ENCRYPTION_SETTINGS = {
    'enable_bootstrapping': False,  # Advanced feature
    'enable_key_switching': True,
    'enable_relinearization': True,
    'enable_rotation': True,
    'max_depth': 10,  # Maximum multiplicative depth
    'batch_size': 100
}

# Analytics Settings
ANALYTICS_SETTINGS = {
    'enable_caching': True,
    'cache_timeout': 3600,  # seconds
    'max_visualization_points': 1000,
    'anomaly_threshold': 3.0,  # Z-score threshold
    'correlation_threshold': 0.5
}

# Security Settings
SECURITY_SETTINGS = {
    'min_key_rotation_days': 30,
    'max_key_rotation_days': 365,
    'require_key_backup': True,
    'enable_audit_log': True,
    'max_failed_operations': 5
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    'parallel_processing': True,
    'num_workers': 4,
    'chunk_size': 1000,
    'memory_limit_mb': 4096,
    'timeout_seconds': 300
}

# UI Settings
UI_SETTINGS = {
    'page_title': 'FHE Financial Analytics',
    'page_icon': '🔐',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'max_display_rows': 100,
    'chart_height': 400
}

# API Settings (for future extension)
API_SETTINGS = {
    'enable_api': False,
    'api_version': 'v1',
    'rate_limit': 100,  # requests per minute
    'timeout': 30  # seconds
}

# Database Settings (for future extension)
DATABASE_SETTINGS = {
    'enable_database': False,
    'db_type': 'sqlite',
    'connection_pool_size': 5,
    'query_timeout': 30
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_advanced_analytics': True,
    'enable_ml_predictions': False,
    'enable_real_time_processing': False,
    'enable_multi_user': False,
    'enable_cloud_storage': False
}

# Error Messages
ERROR_MESSAGES = {
    'no_data': 'No data available. Please upload or generate data first.',
    'no_keys': 'No encryption keys found. Please generate keys first.',
    'encryption_failed': 'Encryption operation failed. Please check parameters.',
    'decryption_failed': 'Decryption operation failed. Please verify keys.',
    'invalid_data': 'Invalid data format. Please check your input.',
    'operation_timeout': 'Operation timed out. Try reducing data size.',
    'insufficient_noise': 'Insufficient noise budget. Consider key rotation.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Data loaded successfully!',
    'keys_generated': 'Keys generated successfully!',
    'encryption_complete': 'Encryption completed successfully!',
    'operation_complete': 'Operation completed successfully!',
    'export_complete': 'Export completed successfully!'
}

# Help Text
HELP_TEXT = {
    'poly_modulus_degree': 'Higher values provide more security but slower performance. Recommended: 8192',
    'plain_modulus': 'Defines the plaintext space size. Larger values allow bigger numbers.',
    'security_level': 'Security strength in bits. 128 bits is standard, 256 for high security.',
    'scale_factor': 'Precision for CKKS scheme. Higher values = more precision but more noise.',
    'noise_budget': 'Remaining computational capacity. Operations consume noise budget.',
    'relinearization': 'Required after multiplication to reduce ciphertext size.',
    'key_rotation': 'Regular key updates improve long-term security.'
}