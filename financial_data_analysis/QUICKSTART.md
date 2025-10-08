# Quick Start Guide - FHE Financial Processor

Get up and running in 5 minutes!

## 🚀 Installation (2 minutes)

### Step 1: Setup Project

**Windows:**
```cmd
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual Setup:**
```bash
pip install streamlit pandas numpy tenseal plotly python-dateutil
```

### Step 2: Start Application

```bash
streamlit run main.py
```

Browser opens at `http://localhost:8501`

## 🎯 First Run (3 minutes)

### Step 1: Generate Sample Data

1. Go to **📊 Data Management** page
2. Select **"Generate Synthetic Data"**
3. Use default values:
   - Users: 100
   - Accounts per User: 2
   - Transactions per Account: 50
4. Click **"Generate Data"**
5. ✅ See: "Generated 100 users, 200 accounts, 10000 transactions!"

### Step 2: Setup Encryption

**Select Columns:**
- User: `user_id`, `name`, `age`
- Account: `account_id`, `balance`, `currency`
- Transaction: `transaction_id`, `amount`, `date`

**Generate Keys:**
1. Keep **"Generate New Keys"** selected
2. Scheme: **CKKS** (recommended)
3. Poly Modulus: **8192**
4. Scale: **40**
5. Click **"Generate Keys"**
6. Download keys (optional)

**Encrypt:**
1. Click **"🔒 Encrypt Selected Data"**
2. Wait for completion
3. ✅ Success!

### Step 3: Run FHE Query

1. Go to **🔒 FHE Operations** page
2. Select any User ID
3. Operation: **"Transaction Analysis"**
4. Keep default date range
5. Click **"🔍 Execute FHE Query"**
6. ✅ View results in tables and charts!

### Step 4: View Statistics

1. Go to **📈 Statistics & Comparison** page
2. Explore tabs:
   - Performance Overview
   - Scheme Comparison
   - Detailed Analytics
   - Operation Breakdown

## 🎓 Next Steps

### Try Different Schemes

**CKKS** (Approximate, Fast):
- Best for analytics
- Use with numeric data

**BFV** (Exact, Medium):
- Best for database operations
- Use with integer data only

**BGV** (Exact, Versatile):
- Best for general purpose
- Supports rotation operations

### Upload Your Own Data

**User CSV:**
```csv
user_id,name,email,address,age
USR001,John Doe,john@example.com,123 Main St,35
```

**Account CSV:**
```csv
account_id,user_id,account_type,balance,currency
ACC001,USR001,Savings,5000.50,USD
```

**Transaction CSV:**
```csv
transaction_id,user_id,account_id,amount,currency,date
TXN001,USR001,ACC001,250.00,USD,2024-10-01
```

### Compare Libraries

Switch between **TenSEAL** and **OpenFHE** in sidebar:
1. Perform operations with TenSEAL
2. Switch to OpenFHE
3. Repeat same operations
4. Compare in Statistics page

## 🔧 Common Tasks

### Save Keys
After generating keys, click download buttons:
- Download Full Public Key
- Download Full Private Key
- Store securely!

### Rotate Keys
1. After generating keys, click **"🔄 Rotate Keys"**
2. New keys generated
3. Old data remains accessible

### Export Statistics
1. Go to Statistics → Detailed Analytics
2. Scroll to bottom
3. Click **"📥 Download Statistics CSV"**

## ⚠️ Troubleshooting

**"No encrypted data available"**
→ Go to Data Management and encrypt columns first

**"Context not initialized"**
→ Generate encryption keys first

**Slow encryption**
→ Try smaller poly modulus (4096)
→ Reduce columns to encrypt
→ Use CKKS (faster than BFV/BGV)

**TenSEAL import error**
```bash
pip uninstall tenseal
pip install tenseal==0.3.14
```

## 📊 Understanding Results

### Transaction Analysis Shows:
- Total/Average/Min/Max amounts
- Currency breakdown
- Monthly patterns
- Type distribution

### Performance Metrics:
- Operation Time: Encryption/query duration
- Rows Processed: Amount of data
- Efficiency Score: Speed + throughput

## 🎯 Recommended Workflow

**For Testing:**
1. Generate 50-100 users
2. Use CKKS scheme
3. Encrypt 3-5 columns
4. Run Transaction Analysis
5. Check Statistics

**For Comparison:**
1. Generate 500 users
2. Try all schemes (CKKS, BFV, BGV)
3. Encrypt multiple columns
4. Run various operations
5. Compare performance

## 📚 Learn More

- **README.md** - Full documentation
- **PROJECT_SUMMARY.md** - Project overview
- **TESTING_GUIDE.md** - Testing instructions
- [TenSEAL Docs](https://github.com/OpenMined/TenSEAL)
- [OpenFHE Docs](https://openfhe-development.readthedocs.io/)

---

**🎉 You're ready to explore Fully Homomorphic Encryption!**