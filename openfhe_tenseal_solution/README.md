# 🔐 FHE Financial Data Analysis System

A comprehensive modular Python application for performing Fully Homomorphic Encryption (FHE) operations on financial data using Streamlit, OpenFHE, and TenSEAL.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Screenshots](#screenshots)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## ✨ Features

### Core Functionality
- 🔐 **Multi-Library Support**: Switch between TenSEAL and OpenFHE
- 🎯 **Multiple FHE Schemes**: BFV, BGV, and CKKS implementations
- 🔑 **Advanced Key Management**: Generate, rotate, import/export keys
- 📊 **Financial Data Processing**: Handle users, accounts, and transactions
- 🧮 **Homomorphic Operations**: Perform computations on encrypted data
- 📈 **Performance Analytics**: Compare schemes and analyze metrics
- 🎨 **Interactive UI**: Beautiful Streamlit interface with visualizations

### Use Cases
1. **Encrypted Transaction Analysis**: Analyze financial patterns without exposing data
2. **Privacy-Preserving Statistics**: Calculate aggregates on encrypted data
3. **Secure Multi-Party Computation**: Share encrypted data for analysis
4. **Compliance**: GDPR and PCI-DSS compatible data processing

## 📁 Project Structure

```
fhe-financial-analytics/
│
├── app.py                          # Main application entry point
├── config.py                       # Configuration and constants
├── fhe_core.py                     # FHE operations and key management
├── data_manager.py                 # Data handling and validation
├── ui_components.py                # Reusable UI components
├── analytics.py                    # Analytics and visualizations
├── openfhe_wrapper.py             # OpenFHE C++ wrapper (optional)
├── tenseal_wrapper.py             # TenSEAL wrapper (optional)
│
├── pages/                         # Application pages
│   ├── __init__.py
│   ├── data_upload_page.py       # Screen 1: Data upload & encryption
│   ├── fhe_operations_page.py    # Screen 2: FHE operations
│   ├── statistics_page.py        # Screen 3: Performance statistics
│   └── key_management_page.py    # Key management interface
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License information
```

## 🚀 Installation

### Prerequisites

- Python 3.8 - 3.11 (3.12+ may have compatibility issues)
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/fhe-financial-analytics.git
cd fhe-financial-analytics
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv fhe_env
fhe_env\Scripts\activate

# Linux/macOS
python3 -m venv fhe_env
source fhe_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt Content

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
tenseal>=0.3.14
openfhe-python>=1.1.4
python-dateutil>=2.8.2
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import tenseal; print('Installation successful!')"
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Basic Workflow

#### 1. **Data Upload & Encryption (Screen 1)**

**Option A: Generate Sample Data**
```python
# In the UI:
1. Select "Generate Sample Data"
2. Choose number of users (10-1000)
3. Click "Generate Data"
```

**Option B: Upload CSV Files**
```python
# Prepare CSV files with these columns:

# users.csv
user_id, name, email, address, country

# accounts.csv
user_id, account_number, account_type, balance

# transactions.csv
user_id, transaction_id, amount, currency, transaction_date
```

**Encryption Steps:**
```python
1. Select table to encrypt (users/accounts/transactions)
2. Choose columns to encrypt
3. Configure FHE parameters:
   - Scheme: BFV/BGV/CKKS
   - Polynomial Degree: 8192 (recommended)
   - Security Level: 128 bits
4. Generate or import keys
5. Click "Start Encryption"
```

#### 2. **FHE Operations (Screen 2)**

**User Analysis:**
```python
1. Select user ID
2. Choose analysis type:
   - Transaction Count
   - Total Amount
   - Average Amount
   - Transaction Pattern
3. Click "Analyze on Encrypted Data"
```

**Transaction Analysis:**
```python
1. Select date range
2. Choose grouping options:
   - By Currency
   - By User ID
   - By Account Type
3. Click "Analyze Encrypted Transactions"
4. View results and visualizations
```

#### 3. **Performance Statistics (Screen 3)**

```python
1. Select schemes to compare (BFV, BGV, CKKS)
2. Run performance benchmark
3. View:
   - Time comparisons
   - Memory usage
   - Security metrics
   - Efficiency rankings
```

#### 4. **Key Management**

**Generate Keys:**
```python
1. Go to Key Management page
2. Select library and scheme
3. Configure parameters
4. Click "Generate Keys"
5. Save keys securely
```

**Rotate Keys:**
```python
1. View current key information
2. Configure new parameters
3. Click "Rotate Keys"
4. Download backup of old keys
```

## 📚 Module Documentation

### config.py

Central configuration file containing:
- FHE scheme parameters
- Application settings
- Data schemas
- UI themes

### fhe_core.py

**FHEKeyManager Class:**
```python
# Generate keys
key_manager = FHEKeyManager()
keys = key_manager.generate_keys(
    library='TenSEAL',
    scheme='BFV',
    parameters={'poly_modulus_degree': 8192}
)

# Rotate keys
new_keys = key_manager.rotate_keys(old_keys, new_parameters)

# Export keys
keys_json = key_manager.export_keys(format='json', include_private=True)
```

**FHEProcessor Class:**
```python
# Initialize processor
processor = FHEProcessor(key_manager, library='TenSEAL')

# Encrypt data
result = processor.encrypt_data(dataframe, columns=['amount', 'balance'])

# Perform operations
op_result = processor.perform_operation('add', 'amount', operand=100)

# Decrypt
decrypted_df, time = processor.decrypt_data()
```

### data_manager.py

**DataManager Class:**
```python
# Generate sample data
data_manager = DataManager()
data = data_manager.generate_sample_data(num_users=100)

# Load CSV files
loaded_data = data_manager.load_csv_files({
    'users': user_file,
    'accounts': account_file,
    'transactions': transaction_file
})

# Validate data
validation = data_manager.validate_data()

# Filter transactions
filtered = data_manager.filter_transactions_by_date(start_date, end_date)
```

### analytics.py

**AnalyticsEngine Class:**
```python
# Initialize engine
analytics = AnalyticsEngine()

# Analyze patterns
patterns = analytics.analyze_transaction_patterns(df)

# Detect anomalies
anomalies = analytics.detect_anomalies(df, column='amount', threshold=3.0)

# Calculate user metrics
metrics = analytics.calculate_user_metrics(transactions_df, 'USR_00001')

# Create visualizations
fig = analytics.create_time_series_plot(df, 'transaction_date', 'amount')
```

## 🔧 Advanced Configuration

### Custom FHE Parameters

```python
# In config.py, modify FHE_SCHEMES dictionary

FHE_SCHEMES = {
    'BFV': {
        'default_poly_modulus': 16384,  # Increase for more security
        'default_plain_modulus': 786433,
        'default_security': 192
    }
}
```

### Custom Data Schema

```python
# In config.py, modify FINANCIAL_DATA_SCHEMA

FINANCIAL_DATA_SCHEMA = {
    'custom_table': {
        'required_columns': ['id', 'value'],
        'sensitive_columns': ['value']
    }
}
```

## 🎨 UI Customization

Modify `ui_components.py` for custom styling:

```python
# Change color scheme
UI_THEME = {
    'primary_color': '#1E88E5',
    'secondary_color': '#764ba2',
    'chart_colors': ['#667eea', '#764ba2', '#f093fb']
}
```

## ⚡ Performance Optimization

### For Large Datasets

```python
# In data_manager.py
# Process in batches
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    encrypt_batch(batch)
```

### For Faster Encryption

```python
# Use lower polynomial degree for development
parameters = {
    'poly_modulus_degree': 4096,  # Instead of 8192
    'security_level': 128
}
```

## 🐛 Troubleshooting

### TenSEAL Installation Issues

```bash
# If installation fails, try:
pip install --upgrade pip setuptools wheel
pip install tenseal --no-cache-dir
```

### OpenFHE Issues

```bash
# For OpenFHE, you may need to compile from source
# See: https://github.com/openfheorg/openfhe-python

# Or use simulation mode in the application
```

### Memory Issues

```bash
# Reduce batch size or polynomial degree
# Or increase system swap space
```

## 📊 Example Outputs

### Encryption Results
```
✅ Data encrypted successfully!
- Encrypted Columns: 3
- Total Values: 10,000
- Time: 1,245.67 ms
- Library: TenSEAL
- Scheme: BFV
```

### Performance Comparison
```
Scheme Performance (ms):
BFV:  Encryption: 1200 | Operation: 300 | Decryption: 650
CKKS: Encryption: 1450 | Operation: 400 | Decryption: 720
```

## 🔒 Security Best Practices

1. **Never commit keys to version control**
2. **Use strong security levels (128+ bits)**
3. **Rotate keys regularly (every 90 days)**
4. **Store keys encrypted at rest**
5. **Use separate keys for different data categories**
6. **Implement key escrow for disaster recovery**
7. **Monitor noise budget during operations**
8. **Audit all FHE operations**

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Email: support@fhe-analytics.com
- Documentation: https://fhe-analytics.readthedocs.io

## 🙏 Acknowledgments

- OpenFHE Development Team
- TenSEAL Contributors
- Streamlit Community
- FHE Research Community

## 📈 Roadmap

- [ ] Support for additional FHE schemes (TFHE, FHEW)
- [ ] Real-time encrypted data streaming
- [ ] Multi-user collaboration features
- [ ] Cloud deployment templates
- [ ] REST API for programmatic access
- [ ] Mobile application support
- [ ] Integration with major databases
- [ ] Machine learning on encrypted data

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Production Ready