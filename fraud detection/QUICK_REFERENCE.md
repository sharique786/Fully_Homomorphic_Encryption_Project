# 🚀 Quick Reference Guide

## 📦 Complete File List

```
your_project/
├── app.py                              ✅ Main entry point
├── config.py                           ✅ All configurations
├── fhe_core.py                         ✅ FHE operations (UPDATED)
├── data_manager.py                     ✅ Data handling
├── ui_components.py                    ✅ UI components (UPDATED)
├── analytics.py                        ✅ Analytics engine
├── openfhe_wrapper.py                  ✅ NEW - OpenFHE C++ wrapper
├── requirements.txt                    ✅ Dependencies
├── README.md                           ✅ Documentation
├── OPENFHE_COMPILATION_GUIDE.md       ✅ NEW - Compilation guide
├── CHANGES_SUMMARY.md                  ✅ NEW - All changes
├── run.py                              ✅ Launcher script
├── test_imports.py                     ✅ Import tester
│
└── pages/
    ├── __init__.py                     ✅ Package init
    ├── data_upload_page.py            ✅ Screen 1 (UPDATED)
    ├── fhe_operations_page.py         ✅ Screen 2
    ├── statistics_page.py             ✅ Screen 3
    └── key_management_page.py         ✅ Screen 4
```

## ⚡ Quick Commands

```bash
# Run application
streamlit run app.py

# Test imports
python test_imports.py

# Use launcher (auto-fixes imports)
python run.py

# Generate OpenFHE C++ code
python -c "from openfhe_wrapper import generate_cpp_wrapper_code; print(generate_cpp_wrapper_code())" > fhe_wrapper.cpp
```

## 🎯 4 Key Changes Made

### 1️⃣ Multi-Table Column Selection
**Location:** Encryption tab in Data Upload page
**Usage:** Select columns from Users, Accounts, AND Transactions simultaneously

### 2️⃣ OpenFHE C++ Integration  
**Location:** Sidebar - select "OpenFHE" library
**Usage:** Enter path to compiled executable, generates real C++ keys

### 3️⃣ All Keys Displayed
**Location:** After generating keys
**Keys Shown:**
- 📤 Public Key
- 🔒 Private Key
- ⚙️ Evaluation Key
- 🔄 Relinearization Key
- 🔀 Galois Keys

### 4️⃣ Icon-Only Navigation
**Location:** Sidebar
**Buttons:** 📊 🧮 📈 🔑 (no text labels)

## 🔧 OpenFHE Setup (3 Steps)

```bash
# 1. Generate C++ files
python -c "from openfhe_wrapper import generate_cpp_wrapper_code, create_cmake_file; open('fhe_wrapper.cpp','w').write(generate_cpp_wrapper_code()); open('CMakeLists.txt','w').write(create_cmake_file())"

# 2. Compile
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# 3. Copy executable
copy Release\fhe_wrapper.exe ..\
```

## 📊 Usage Flow

```
1. Launch: streamlit run app.py
2. Sidebar: Select library (TenSEAL/OpenFHE) and scheme
3. Click 📊 icon → Generate/Upload data
4. Encryption tab → Select columns from all tables
5. Generate keys → See all 5 key types
6. Encrypt → All tables encrypted
7. Click 🧮 icon → Perform FHE operations
8. Click 📈 icon → View performance stats
9. Click 🔑 icon → Manage keys
```

## 🐛 Common Issues & Quick Fixes

### Import Error
```bash
# Fix:
python run.py  # Uses auto-fix launcher
```

### OpenFHE Not Working
```bash
# Check:
1. Executable exists at specified path
2. DLLs in PATH or same directory
3. Falls back to simulation automatically
```

### No Keys Displayed
```bash
# Fix:
1. Click "Generate Keys" button
2. Wait for success message
3. Check "Show Private Key" for all keys
```

### Multi-Table Not Showing
```bash
# Fix:
1. Ensure data uploaded/generated
2. Go to "Encryption" tab (not "Data Upload")
3. Look for tabs at top of page
```

## 📝 Key Concepts

| Term | Meaning |
|------|---------|
| **BFV** | Exact integer arithmetic scheme |
| **CKKS** | Approximate real number scheme |
| **Public Key** | For encryption (shareable) |
| **Private Key** | For decryption (secret) |
| **Evaluation Key** | For operations on encrypted data |
| **Relinearization** | After multiplication |
| **Galois Keys** | For rotation operations |

## 🎨 UI Navigation

| Icon | Page | Purpose |
|------|------|---------|
| 📊 | Data Upload | Upload/generate & encrypt data |
| 🧮 | FHE Operations | Perform encrypted computations |
| 📈 | Statistics | Compare performance |
| 🔑 | Key Management | Generate/rotate/import keys |

## ⚙️ Configuration Quick Reference

### TenSEAL Mode
```python
Library: TenSEAL
Scheme: BFV or CKKS
Poly Degree: 8192
Plain Modulus: 1032193 (BFV)
Scale Factor: 40 (CKKS)
```

### OpenFHE C++ Mode
```python
Library: OpenFHE
Path: C:\openfhe-development\build\bin\Release
Scheme: BFV, BGV, or CKKS
Poly Degree: 8192
Status: Check for ✅ or ⚠️ in sidebar
```

### Simulation Mode
```python
Library: Any (if C++/TenSEAL unavailable)
Mode: Automatic fallback
Speed: Fastest
Security: Demo only
```

## 📥 Sample Data Stats

```
Users: 100 records
- user_id, name, email, address, country

Accounts: ~200 records (1-3 per user)
- account_number, account_type, balance

Transactions: ~5,000 records (10-50 per account)
- transaction_id, amount, currency, date
```

## 💾 Export Options

```
Keys: JSON format with all 5 types
Data: CSV per table
Reports: Performance statistics as CSV
Charts: Interactive Plotly (view only)
```

## 🔒 Security Tips

1. **Never commit keys** to version control
2. **Download keys immediately** after generation
3. **Use 128-bit+ security** for production
4. **Rotate keys** every 90 days
5. **Separate keys** per data category
6. **Test with simulation** before production

## 🏃 Performance Tips

1. **Start small**: Test with 100 rows
2. **Use batching**: For >1000 rows
3. **Select fewer columns**: Only encrypt sensitive data
4. **Choose right scheme**: BFV for exact, CKKS for approximate
5. **Monitor memory**: FHE is memory-intensive
6. **Use C++ when available**: 20% faster than Python

## 📞 Support

**Getting Help:**
1. Check error message carefully
2. Run `test_imports.py` to verify setup
3. Check OPENFHE_COMPILATION_GUIDE.md for C++ issues
4. Review CHANGES_SUMMARY.md for feature details
5. See README.md for full documentation

**Common Error Patterns:**
- `ImportError` → Run `python run.py`
- `OpenFHE not found` → Check path in sidebar
- `No data` → Upload/generate data first
- `No keys` → Click "Generate Keys" button

---

**Everything you need in one place! 🎉**

Quick start: `streamlit run app.py` and click the 📊 icon!