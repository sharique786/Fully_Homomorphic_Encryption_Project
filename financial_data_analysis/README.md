# FHE Financial Data Processor

A comprehensive Fully Homomorphic Encryption (FHE) application for secure financial data processing using OpenFHE and TenSEAL libraries.

## 🎯 Features

- 🔐 **Multiple FHE Libraries**: Support for both TenSEAL and OpenFHE
- 🎯 **Multiple Schemes**: CKKS, BFV, and BGV encryption schemes
- 📊 **Data Management**: Generate synthetic data or upload CSV files
- 🔑 **Key Management**: Generate, export, and rotate encryption keys
- 🔒 **FHE Operations**: Perform operations on encrypted financial data
- 📈 **Statistics**: Compare performance across different schemes and libraries
- 🎨 **Interactive UI**: Built with Streamlit for easy interaction

## 📁 Project Structure

```
fhe_financial_processor/
├── main.py                      # Main application entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── setup.sh                     # Linux/Mac setup script
├── setup.bat                    # Windows setup script
├── README.md                    # This file
├── QUICKSTART.md                # Quick start guide
├── utils/
│   ├── __init__.py
│   ├── session_state.py         # Session state management
│   ├── data_generator.py        # Financial data generator
│   └── helpers.py               # Helper functions
├── fhe/
│   ├── __init__.py
│   ├── tenseal_wrapper.py       # TenSEAL implementation
│   └── openfhe_wrapper.py       # OpenFHE wrapper
└── ui/
    ├── __init__.py
    ├── data_management.py       # Data management page
    ├── fhe_operations.py        # FHE operations page
    └── statistics.py            # Statistics page
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) OpenFHE C++ library compiled on your system

### Quick Install

#### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

#### Windows:
```cmd
setup.bat
```

### Manual Installation

1. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

2. **OpenFHE Setup (Optional)**

For full OpenFHE support, compile the library:

**Windows:**
1. Follow: https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/windows.html
2. Clone: `git clone https://github.com/openfheorg/openfhe-development.git`
3. Build using CMake and Visual Studio
4. Update paths in `fhe/openfhe_wrapper.py`

**Note**: The application includes a simulation mode if OpenFHE is not available.

## 📖 Usage

### 1. Start the Application

```bash
streamlit run main.py
```

The application will open at `http://localhost:8501`

### 2. Data Management Page

**Generate Synthetic Data:**
1. Select "Generate Synthetic Data"
2. Configure parameters (users, accounts, transactions)
3. Click "Generate Data"

**Upload CSV Files:**
1. Select "Upload CSV Files"
2. Upload user.csv, account.csv, transaction.csv
3. Expected structure:
   - **User**: user_id, name, email, address, phone, age
   - **Account**: account_id, user_id, account_type, balance, currency
   - **Transaction**: transaction_id, user_id, account_id, amount, currency, date

**Encrypt Data:**
1. Select columns to encrypt from each table
2. Choose "Generate New Keys"
3. Select encryption scheme (CKKS, BFV, or BGV)
4. Configure parameters
5. Click "Generate Keys"
6. Download and save keys
7. Click "Encrypt Selected Data"

### 3. FHE Operations Page

1. Select user ID to query
2. Choose operation type:
   - Transaction Analysis
   - Account Summary
   - Custom Query
3. Set date range and currency filters
4. Click "Execute FHE Query"
5. View results in tables and charts

### 4. Statistics & Comparison Page

View comprehensive statistics:
- Performance overview
- Scheme comparison (CKKS vs BFV vs BGV)
- Detailed analytics
- Operation breakdown
- Export statistics

## 🔐 FHE Schemes Comparison

| Scheme | Type | Data Support | Precision | Best For | Speed |
|--------|------|--------------|-----------|----------|-------|
| CKKS | Approximate | Real/Complex | ~40-60 bits | ML, Analytics | Fast |
| BFV | Exact | Integers | Exact | Database ops | Medium |
| BGV | Exact | Integers | Exact | General purpose | Medium-Fast |

## 🔑 Key Features

### Key Management

- **Generate Keys**: Create public, private, and evaluation keys
- **Export Keys**: Download keys for secure storage
- **Key Rotation**: Rotate keys with backward compatibility
- **Multiple Key Types**: Public, private, Galois, relinearization

### Scheme Limitations

The application handles scheme limitations automatically:
- **CKKS**: Approximate arithmetic, best for real numbers
- **BFV**: Exact integers, limited text support
- **BGV**: Versatile exact arithmetic with rotation support

### Security Parameters

- **Polynomial Modulus Degree**: 8192, 16384, 32768 (higher = more secure, slower)
- **Scale (CKKS)**: 2^30, 2^40, 2^50 (precision parameter)
- **Security Level**: HEStd_128_classic, HEStd_192_classic, HEStd_256_classic

## 💡 Performance Tips

1. **Choose the Right Scheme**:
   - CKKS for approximate computations (faster)
   - BFV/BGV for exact integer operations

2. **Optimize Parameters**:
   - Start with lower polynomial modulus degrees
   - Increase only when needed for security

3. **Batch Operations**:
   - Process multiple values together
   - Use SIMD operations in CKKS

4. **Monitor Statistics**:
   - Use Statistics page to identify bottlenecks
   - Compare different schemes for your use case

## 🔧 Configuration

Edit `config.py` to customize:
- OpenFHE library paths
- Default encryption parameters
- Data generation defaults
- UI settings
- Security levels

## 📊 Supported Operations

### Encryption Operations
- Numeric data encryption
- Text encoding and encryption
- Date/timestamp encryption
- Batch processing

### Homomorphic Operations
- Addition
- Multiplication
- Subtraction
- Aggregation (sum, count, average)
- Filtering
- Grouping

### Query Operations
- Transaction counting
- Amount summation
- Currency grouping
- Date range filtering
- Pattern analysis

## 🐛 Troubleshooting

### TenSEAL Installation Issues

```bash
# Try with conda
conda install -c conda-forge tenseal

# Or build from source
pip install tenseal --no-binary tenseal
```

### OpenFHE Not Found

The application runs in simulation mode if OpenFHE is unavailable. To use real OpenFHE:
1. Verify the library is compiled
2. Check paths in `fhe/openfhe_wrapper.py`
3. Ensure DLL files are in PATH (Windows) or LD_LIBRARY_PATH (Linux)

### Memory Issues

If you encounter memory errors:
1. Reduce number of users/transactions
2. Encrypt fewer columns at once
3. Use smaller polynomial modulus degrees
4. Process data in batches

## 📚 Documentation

- **README.md** - This file
- **QUICKSTART.md** - 5-minute quick start guide
- **PROJECT_SUMMARY.md** - Complete project overview
- **TESTING_GUIDE.md** - Testing instructions

## 🔗 References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [TenSEAL GitHub](https://github.com/OpenMined/TenSEAL)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is for educational and research purposes. Ensure compliance with relevant data protection regulations when handling sensitive financial data.

## 🙏 Acknowledgments

- OpenMined for TenSEAL
- OpenFHE development team
- Streamlit community
- FHE research community

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review QUICKSTART.md
3. Consult official FHE library documentation

---

**Version**: 1.0.0  
**Last Updated**: October 2024  
**Status**: Production-Ready for Educational/Research Use