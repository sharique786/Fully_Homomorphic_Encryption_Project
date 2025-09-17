
# app.py
# Streamlit + OpenFHE demo for encrypting CSV columns and simulating FHE ops (BFV, BGV, CKKS)
# Tested against openfhe-python >= 1.3.x API docs. If OpenFHE is missing, the app will still run
# and show installation guidance.

import io
import time
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Try to import OpenFHE. If not available, set a flag so the UI can guide the user.
OPENFHE_OK = True
try:
    from openfhe import (
        CCParamsBFVRNS, CCParamsBGVRNS, CCParamsCKKSRNS,
        CryptoContext, KeyPair, PKESchemeFeature,
        GenCryptoContext, SecurityLevel,
        SERJSON, SERBINARY
    )
except Exception as e:
    OPENFHE_OK = False
    OPENFHE_IMPORT_ERROR = e

st.set_page_config(page_title="OpenFHE Streamlit Lab", layout="wide")

st.title("🔐 OpenFHE Streamlit Lab — Encrypt finance CSV, run HE ops (BFV/BGV/CKKS)")

with st.expander("ℹ️ What this app does"):
    st.markdown("""
    - Upload a **CSV** with financial/PII columns (e.g., name, country/region, account numbers, amounts).
    - Choose a **scheme** (BFV/BGV for integers, CKKS for real numbers).
    - Select **columns to encrypt**, generate keys, and run **homomorphic operations** (add, multiply, polynomial).
    - Tweak parameters: **poly/ring dimension**, **multiplicative depth**, **plaintext modulus** (BFV/BGV), **scaling modulus** (CKKS), **security level**, **relinearization**, **rotation keys**, etc.
    - See **timings and basic stats** for the operations; preview **masked PII** and **decrypted results** to verify correctness.
    """)

if not OPENFHE_OK:
    st.error("OpenFHE Python bindings are not available in this environment.")
    with st.expander("📥 Installation notes (Windows / Linux / RHEL / Conda)", expanded=True):
        st.markdown("""
        - **Conda (recommended)**: try `conda install -c conda-forge openfhe` (when available for your platform).
        - **Pip (Linux/Ubuntu)**: `pip install openfhe` (prebuilt wheels are primarily for recent Ubuntu).
        - **Windows/RHEL/GCP (custom images)**: you may need to **build from source** following the official instructions.
        - Ensure your environment uses **Python 3.12** (Anaconda 24.11.3 compatible) and the C++ toolchain required by OpenFHE.
        - If you see `ImportError: libOPENFHEpke.so.1` on Linux: set `LD_LIBRARY_PATH` to the directory containing OpenFHE shared libs.
        """)
        st.code(str(OPENFHE_IMPORT_ERROR))
    st.stop()

# -----------------------------
# Helpers: Create CryptoContext
# -----------------------------

def build_context(
    scheme: str,
    multiplicative_depth: int,
    ring_dim: int,
    security: str,
    plaintext_modulus: Optional[int] = None,
    scaling_mod_size: Optional[int] = None,
) -> CryptoContext:
    """
    Construct a CryptoContext for the chosen scheme with common parameters.
    """
    sec_map = {
        "HEStd_128_classic": SecurityLevel.HEStd_128_classic,
        "HEStd_192_classic": SecurityLevel.HEStd_192_classic,
        "HEStd_256_classic": SecurityLevel.HEStd_256_classic,
    }
    sec_level = sec_map.get(security, SecurityLevel.HEStd_128_classic)

    if scheme == "BFV":
        params = CCParamsBFVRNS()
        if plaintext_modulus:
            params.SetPlaintextModulus(int(plaintext_modulus))
        params.SetMultiplicativeDepth(int(multiplicative_depth))
        if ring_dim:
            try:
                params.SetRingDim(int(ring_dim))
            except Exception:
                pass  # some builds may not expose SetRingDim in python
        params.SetSecurityLevel(sec_level)
        cc = GenCryptoContext(params)
    elif scheme == "BGV":
        params = CCParamsBGVRNS()
        if plaintext_modulus:
            params.SetPlaintextModulus(int(plaintext_modulus))
        params.SetMultiplicativeDepth(int(multiplicative_depth))
        if ring_dim:
            try:
                params.SetRingDim(int(ring_dim))
            except Exception:
                pass
        params.SetSecurityLevel(sec_level)
        cc = GenCryptoContext(params)
    else:  # CKKS
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(int(multiplicative_depth))
        if ring_dim:
            try:
                params.SetRingDim(int(ring_dim))
            except Exception:
                pass
        if scaling_mod_size:
            try:
                params.SetScalingModSize(int(scaling_mod_size))
            except Exception:
                pass
        params.SetSecurityLevel(sec_level)
        cc = GenCryptoContext(params)

    # Enable features
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        try:
            cc.Enable(feat)
        except Exception:
            pass
    return cc

# ----------------------------------
# Plaintext helpers per scheme
# ----------------------------------

def make_plain(cc: CryptoContext, scheme: str, values: List[Any]):
    if scheme in ("BFV", "BGV"):
        # integer packing
        try:
            return cc.MakePackedPlaintext([int(x) for x in values])
        except Exception:
            # fallback: map to small ints
            return cc.MakePackedPlaintext([int(round(float(x))) for x in values])
    else:
        # CKKS uses reals
        vals = [float(x) for x in values]
        try:
            return cc.MakeCKKSPackedPlaintext(vals)
        except Exception:
            return cc.MakePackedPlaintext(vals)

# -----------------------------
# Encrypt / Decrypt helpers
# -----------------------------

def keygen_and_evalkeys(cc: CryptoContext, relinearize: bool, rotation_indices: List[int]) -> KeyPair:
    keys = cc.KeyGen()
    if relinearize:
        try:
            cc.EvalMultKeyGen(keys.secretKey)
        except Exception:
            pass
    if rotation_indices:
        try:
            cc.EvalAtIndexKeyGen(keys.secretKey, rotation_indices)
        except Exception:
            pass
    return keys

def encrypt_series(cc: CryptoContext, scheme: str, public_key, series: pd.Series, batch_size: int = 0):
    """
    Encrypt a Pandas Series by packing in chunks (batch). Returns list of ciphertexts and packing meta.
    If batch_size==0 we try to pack all values (subject to slot count).
    """
    data = series.fillna(0).tolist()
    if batch_size <= 0:
        batch_size = len(data)
    cts = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        pt = make_plain(cc, scheme, chunk)
        ct = cc.Encrypt(public_key, pt)
        cts.append(ct)
    return cts, {"batch_size": batch_size, "n_chunks": len(cts), "n_values": len(data)}

def decrypt_to_series(cc: CryptoContext, scheme: str, secret_key, cts: List, meta: Dict[str, Any], index):
    decoded = []
    for ct in cts:
        pt = cc.Decrypt(secret_key, ct)
        try:
            decoded.extend(pt.GetPackedValue())
        except Exception:
            try:
                decoded.extend(pt.GetRealPackedValue())
            except Exception:
                # last resort: try to convert to string then eval-ish parse
                decoded.append(float(str(pt)))
    # trim to original length
    decoded = decoded[:meta["n_values"]]
    return pd.Series(decoded, index=index)

# -----------------------------
# UI — Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("⚙️ Parameters")

    scheme = st.selectbox("Scheme", ["BFV", "BGV", "CKKS"], index=0,
                          help="BFV/BGV for integers; CKKS for real numbers.")

    multiplicative_depth = st.slider("Multiplicative depth", 0, 8, 2,
                                     help="Max multiplication depth without bootstrapping.")

    ring_dim = st.selectbox("poly_modulus_degree / Ring dimension (N)", [0, 1024, 2048, 4096, 8192, 16384], index=3,
                            help="Some Python builds may ignore this; OpenFHE can auto-select based on security & depth.")

    security = st.selectbox("Security level", ["HEStd_128_classic", "HEStd_192_classic", "HEStd_256_classic"], index=0)

    plaintext_modulus = None
    scaling_mod_size = None
    if scheme in ("BFV", "BGV"):
        plaintext_modulus = st.number_input("Plaintext modulus (t) — BFV/BGV", min_value=2, value=65537, step=1,
                                            help="Pick prime t suitable for your data range. For PII IDs, small t may wrap values.")
    else:
        scaling_mod_size = st.selectbox("Scaling modulus size (CKKS)", [0, 30, 40, 50, 60], index=2,
                                        help="Controls precision/levels for CKKS. 0 lets library choose.")

    relinearize = st.checkbox("Generate relinearization (EvalMult) keys", value=True)
    rotation_indices_raw = st.text_input("Rotation indices (comma-separated)", value="", help="Optional, e.g., 1,-1,2,-2 for vector rotations")
    rotation_indices = []
    if rotation_indices_raw.strip():
        try:
            rotation_indices = [int(x.strip()) for x in rotation_indices_raw.split(",") if x.strip()]
        except Exception:
            rotation_indices = []

# -----------------------------
# Data upload and column selection
# -----------------------------

st.subheader("📤 Upload CSV with financial data (PII)")
file = st.file_uploader("Upload .csv", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    # Show masked preview for potential PII columns
    preview = df.copy()
    # Mask strings and known PII-like columns
    for col in preview.columns:
        if preview[col].dtype == object or "name" in col.lower() or "email" in col.lower() or "phone" in col.lower():
            preview[col] = preview[col].astype(str).str.replace(r".", "•", regex=True).str.slice(0, 6) + "..."
    st.dataframe(preview.head(20), use_container_width=True)

    # Column selection
    st.subheader("🔎 Select columns to encrypt")
    cols = st.multiselect("Columns", options=list(df.columns))
    if cols:
        st.info("Tip: For BFV/BGV, non-numeric columns will be cast to integers (hash). For CKKS, values are cast to floats.")

        # Build context & keys
        with st.status("Building crypto context and keys...", expanded=False):
            t0 = time.perf_counter()
            cc = build_context(
                scheme=scheme,
                multiplicative_depth=multiplicative_depth,
                ring_dim=ring_dim,
                security=security,
                plaintext_modulus=plaintext_modulus,
                scaling_mod_size=scaling_mod_size,
            )
            keys = keygen_and_evalkeys(cc, relinearize=relinearize, rotation_indices=rotation_indices)
            build_ms = (time.perf_counter() - t0) * 1000.0
        st.success(f"Context ready in {build_ms:.1f} ms")

        # Encrypt selected columns
        enc_store: Dict[str, Dict[str, Any]] = {}
        timings = []
        for c in cols:
            series = df[c]
            # For non-numeric types, we map to integer via hash for BFV/BGV, or to length for CKKS default
            if scheme in ("BFV", "BGV"):
                if not np.issubdtype(series.dtype, np.number):
                    series_enc = series.fillna("").astype(str).apply(lambda x: abs(hash(x)) % (plaintext_modulus or 2**20))
                else:
                    series_enc = series.fillna(0).astype(np.int64)
            else:
                # CKKS
                if not np.issubdtype(series.dtype, np.number):
                    series_enc = series.fillna("").astype(str).apply(lambda s: float(len(s)))
                else:
                    series_enc = series.fillna(0.0).astype(float)

            t0 = time.perf_counter()
            cts, meta = encrypt_series(cc, scheme, keys.publicKey, series_enc)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            enc_store[c] = {"cts": cts, "meta": meta, "index": series.index, "dtype": series.dtype}
            timings.append({"phase": f"encrypt[{c}]", "ms": enc_ms, "chunks": meta["n_chunks"], "values": meta["n_values"]})

        st.success("Columns encrypted.")

        # -----------------------------
        # Homomorphic operations
        # -----------------------------
        st.subheader("🧮 Homomorphic operations")

        op = st.selectbox("Operation", ["None", "Add (cipher + cipher)", "Multiply (cipher * cipher)", "Add constant", "Multiply constant", "Polynomial a*x^2 + b*x + c"], index=0)

        left_col = right_col = const = None
        a = b = ccoef = 0.0

        if op in ("Add (cipher + cipher)", "Multiply (cipher * cipher)"):
            left_col = st.selectbox("Left column", cols, index=0)
            right_col = st.selectbox("Right column", cols, index=min(1, len(cols)-1))
        elif op in ("Add constant", "Multiply constant"):
            left_col = st.selectbox("Target column", cols, index=0)
            const = st.number_input("Constant", value=1.0 if scheme == "CKKS" else 1.0)
        elif op == "Polynomial a*x^2 + b*x + c":
            left_col = st.selectbox("Target column", cols, index=0)
            a = st.number_input("a", value=1.0)
            b = st.number_input("b", value=0.0)
            ccoef = st.number_input("c", value=0.0)

        op_timings = []
        result_series = None
        if op != "None" and left_col:
            left_cts = enc_store[left_col]["cts"]
            # for two-cipher ops, we assume both have same packing
            right_cts = enc_store.get(right_col, {}).get("cts") if right_col else None
            out_cts = []
            t0 = time.perf_counter()
            for i, lct in enumerate(left_cts):
                if op == "Add (cipher + cipher)":
                    rct = right_cts[i]
                    out = cc.EvalAdd(lct, rct)
                elif op == "Multiply (cipher * cipher)":
                    rct = right_cts[i]
                    out = cc.EvalMult(lct, rct)
                elif op == "Add constant":
                    if scheme == "CKKS":
                        out = cc.EvalAdd(lct, float(const))
                    else:
                        ptc = make_plain(cc, scheme, [int(const)] * enc_store[left_col]["meta"]["batch_size"])
                        out = cc.EvalAdd(lct, ptc)
                elif op == "Multiply constant":
                    if scheme == "CKKS":
                        out = cc.EvalMult(lct, float(const))
                    else:
                        ptc = make_plain(cc, scheme, [int(const)] * enc_store[left_col]["meta"]["batch_size"])
                        out = cc.EvalMult(lct, ptc)
                elif op == "Polynomial a*x^2 + b*x + c":
                    # Horner form: a*x^2 + b*x + c = x*(a*x + b) + c
                    ax = cc.EvalMult(lct, float(a) if scheme == "CKKS" else make_plain(cc, scheme, [int(a)] * enc_store[left_col]["meta"]["batch_size"]))
                    x2 = cc.EvalMult(lct, lct)
                    ax2 = cc.EvalMult(ax, lct) if scheme == "CKKS" else cc.EvalMult(lct, ax)
                    bx = cc.EvalMult(lct, float(b) if scheme == "CKKS" else make_plain(cc, scheme, [int(b)] * enc_store[left_col]["meta"]["batch_size"]))
                    tmp = cc.EvalAdd(ax2, bx)
                    out = cc.EvalAdd(tmp, float(ccoef) if scheme == "CKKS" else make_plain(cc, scheme, [int(ccoef)] * enc_store[left_col]["meta"]["batch_size"]))
                else:
                    out = lct
                out_cts.append(out)
            op_ms = (time.perf_counter() - t0) * 1000.0
            op_timings.append({"phase": f"op:{op}", "ms": op_ms, "chunks": len(out_cts)})
            st.success(f"Operation '{op}' completed in {op_ms:.1f} ms")

            # decrypt result for verification
            res = decrypt_to_series(cc, scheme, keys.secretKey, out_cts, enc_store[left_col]["meta"], enc_store[left_col]["index"])
            result_series = res

        # -----------------------------
        # Decrypt preview & statistics
        # -----------------------------
        st.subheader("📊 Results & stats")

        # Build timing table
        timing_df = pd.DataFrame(timings + op_timings)
        st.dataframe(timing_df, use_container_width=True)

        # If we have a result, show side-by-side
        if result_series is not None:
            show_cols = [left_col]
            if right_col:
                show_cols.append(right_col)
            original_view = df[show_cols].copy()
            # Don't show PII directly if it's string: mask it
            for col in original_view.columns:
                if original_view[col].dtype == object:
                    original_view[col] = original_view[col].astype(str).str.replace(r".", "•", regex=True).str.slice(0, 6) + "..."
            result_df = pd.DataFrame({f"{left_col} (decrypted result)": result_series})
            st.write("**Original (masked where needed)** vs **Decrypted result**")
            st.dataframe(pd.concat([original_view, result_df], axis=1).head(20), use_container_width=True)

            # Simple chart (numeric)
            try:
                import matplotlib.pyplot as plt
                st.write("Distribution of decrypted result (first 1,000 records)")
                vals = pd.to_numeric(result_series, errors="coerce").dropna().values[:1000]
                if len(vals) > 0:
                    fig = plt.figure()
                    plt.hist(vals, bins=30)
                    st.pyplot(fig)
            except Exception:
                pass

        with st.expander("🔧 Serialize keys/ciphertexts (advanced)"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download public key (binary)"):
                    tmp = io.BytesIO()
                    ok = cc.SerializePublicKey(tmp, keys.publicKey, SERBINARY)
                    st.download_button("Save public.key", data=tmp.getvalue(), file_name="public.key", mime="application/octet-stream")
                if st.button("Download private key (binary)"):
                    tmp = io.BytesIO()
                    ok = cc.SerializePrivateKey(tmp, keys.secretKey, SERBINARY)
                    st.download_button("Save private.key", data=tmp.getvalue(), file_name="private.key", mime="application/octet-stream")
            with col2:
                st.caption("Ciphertext serialization for the first encrypted column:")
                if cols:
                    sample_cts = enc_store[cols[0]]["cts"]
                    if sample_cts:
                        tmp = io.BytesIO()
                        ok = cc.SerializeCiphertext(tmp, sample_cts[0], SERBINARY)
                        st.download_button("Save sample ciphertext", data=tmp.getvalue(), file_name="sample.ct", mime="application/octet-stream")

else:
    st.info("Upload a CSV to begin.")
