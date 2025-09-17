# FHE Streamlit Demo App

This package contains a Streamlit web app demonstrating Fully Homomorphic Encryption (FHE) concepts:
- Uses OpenFHE (preferred), TenSEAL or Pyfhel as available in the environment.
- For textual/categorical PII columns the app uses symmetric AES (Fernet) since FHE primarily operates over numeric data.
- Allows uploading CSV, selecting columns to encrypt, running simple homomorphic-style operations (add, sub, mul, polynomial eval), adjusting a small subset of crypto parameters.

## Files
- `app.py` — main Streamlit application
- `fhe_utils.py` — helper wrapper that tries multiple backends and provides AES helpers
- `requirements.txt` — minimal Python dependencies
- `sample_data.csv` — small example CSV (financial-like)
- `README.md` — this file

## Quick-start (local)
1. Create a Python 3.11 virtual environment and activate it.
2. `pip install -r requirements.txt`  
   Optionally install one or more FHE backends:
   - `pip install openfhe` (may require platform-specific steps; see OpenFHE docs)
   - `pip install tenseal` (TenSEAL may be tricky on Windows)
   - `pip install pyfhel`
3. Run: `streamlit run app.py`
4. Open the local URL shown in console.

## Notes & Caveats
- Installing OpenFHE Python wrapper on Windows may require building from source or using a prebuilt wheel. See OpenFHE docs: https://github.com/openfheorg/openfhe-python and https://openfhe.org
- TenSEAL binaries are not always available on all Windows Python versions; on Linux/Colab it's usually easier.
- This app contains demo / pedagogical implementations and **does NOT** replace production cryptographic engineering. Parameters used here are simplified for demonstration.

## Colab
If you want to run on Google Colab, install the packages and run a Streamlit cloud runtime or use `pyngrok` to tunnel; Colab is convenient for TenSEAL and Pyfhel experiments.

## License
MIT
