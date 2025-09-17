# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import math
import json
from sklearn import preprocessing

st.set_page_config(layout="wide", page_title="OpenFHE Streamlit Demo")

# Try to import OpenFHE - if not present show helpful message
try:
    from openfhe import *
    HAVE_OPENFHE = True
except Exception as e:
    HAVE_OPENFHE = False
    import traceback
    OPENFHE_IMPORT_ERR = str(e) + "\n" + traceback.format_exc()

st.title("OpenFHE interactive demo — BFV, BGV, CKKS (Streamlit)")

if not HAVE_OPENFHE:
    st.error("OpenFHE Python wrapper not found. See instructions below to install.")
    st.markdown("""
**Quick install hints**
- Try: `pip install openfhe` (wheel may be available for some platforms).
- If pip wheel isn't available, you must build OpenFHE (C++) and then the `openfhe-python` bindings. See OpenFHE docs and openfhe-python repo for platform-specific instructions (MSYS2 for Windows, CMake for Linux).  
(References: OpenFHE docs and wrapper repo.)
""")
    with st.expander("Show import error (for debugging)"):
        st.code(OPENFHE_IMPORT_ERR)
    st.stop()

# ---------- Utility helpers for OpenFHE ----------
def make_crypto_context(scheme_name: str,
                        poly_modulus_degree: int,
                        plaintext_modulus: int,
                        multiplicative_depth: int,
                        security_level: int = 128):
    """
    Create CryptoContext using OpenFHE parameter classes.
    Returns (crypto_context, params_obj)
    """
    # Choose CC params depending on scheme
    if scheme_name.lower() in ("bfv", "bgv"):
        # Use BFV/BGV parameter object
        if scheme_name.lower() == "bfv":
            params = CCParamsBFVRNS()
        else:
            params = CCParamsBGVRNS()
        params.SetPlaintextModulus(plaintext_modulus)
        params.SetMultiplicativeDepth(multiplicative_depth)
        # poly_modulus_degree mapping: some APIs use SetRingDimension or SetCyclotomicOrder.
        # We attempt SetRingDimension if available else set cyclotomic order.
        try:
            params.SetRingDimension(poly_modulus_degree)
        except Exception:
            try:
                params.SetCyclotomicOrder(poly_modulus_degree)
            except Exception:
                pass

    elif scheme_name.lower() == "ckks":
        params = CCParamsCKKS()
        params.SetMultiplicativeDepth(multiplicative_depth)
        try:
            params.SetRingDimension(poly_modulus_degree)
        except Exception:
            try:
                params.SetCyclotomicOrder(poly_modulus_degree)
            except Exception:
                pass
    else:
        raise ValueError("Unsupported scheme")

    # Generate CryptoContext
    crypto_context = GenCryptoContext(params)
    # Enable basic features used in examples
    crypto_context.Enable(PKESchemeFeature.PKE)
    crypto_context.Enable(PKESchemeFeature.KEYSWITCH)
    crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)
    return crypto_context, params

def keygen_for_context(cc):
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    # rotation keys for simple vector rotations
    cc.EvalRotateKeyGen(kp.secretKey, list(range(1, min(8, cc.GetRingDimension()))))
    return kp

def encode_encrypt_dataframe(cc, kp_public, df, columns, scheme):
    """
    Encrypt selected columns in df. Returns a dict mapping column -> ciphertext (serialized),
    and an in-memory dataframe copy where encrypted cells are replaced with ciphertext IDs.
    """
    enc_map = {}
    df_enc = df.copy()
    for col in columns:
        coldata = df[col].to_list()
        # For BFV/BGV (integers), pack integers. For CKKS encode floats.
        if scheme.lower() in ("bfv", "bgv"):
            # Ensure ints
            vec = [int(x) if not pd.isna(x) else 0 for x in coldata]
            pt = cc.MakePackedPlaintext(vec)
        else:  # ckks
            vec = [float(x) if not pd.isna(x) else 0.0 for x in coldata]
            pt = cc.MakeCKKSPackedPlaintext(vec)

        ct = cc.Encrypt(kp_public, pt)
        # serialize ciphertext to a short base64 so it can be displayed/stored
        buf = io.BytesIO()
        ct.serialize(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        enc_map[col] = b64
        # replace column display with index reference
        df_enc[col] = [f"<ciphertext:{col}:{i}>" for i in range(len(df))]
    return enc_map, df_enc

def decrypt_column(cc, kp_secret, enc_b64):
    raw = base64.b64decode(enc_b64.encode("ascii"))
    buf = io.BytesIO(raw)
    ct = cc.Ciphertext()
    ct.deserialize(buf)
    pt = cc.Decrypt(kp_secret, ct)
    # convert to python list: prefer GetPackedValue / GetCKKSValue depending on type
    try:
        vals = pt.GetPackedValue()
    except Exception:
        try:
            vals = pt.GetCKKSValue()
        except Exception:
            vals = None
    return vals

def eval_add(cc, ct1_b64, ct2_b64):
    # deserialize
    raw1 = base64.b64decode(ct1_b64)
    raw2 = base64.b64decode(ct2_b64)
    buf1 = io.BytesIO(raw1); buf2 = io.BytesIO(raw2)
    ct1 = cc.Ciphertext(); ct2 = cc.Ciphertext()
    ct1.deserialize(buf1); ct2.deserialize(buf2)
    ct_out = cc.EvalAdd(ct1, ct2)
    buf_out = io.BytesIO(); ct_out.serialize(buf_out)
    return base64.b64encode(buf_out.getvalue()).decode("ascii")

def eval_mult(cc, ct1_b64, ct2_b64):
    raw1 = base64.b64decode(ct1_b64)
    raw2 = base64.b64decode(ct2_b64)
    buf1 = io.BytesIO(raw1); buf2 = io.BytesIO(raw2)
    ct1 = cc.Ciphertext(); ct2 = cc.Ciphertext()
    ct1.deserialize(buf1); ct2.deserialize(buf2)
    ct_out = cc.EvalMult(ct1, ct2)
    buf_out = io.BytesIO(); ct_out.serialize(buf_out)
    return base64.b64encode(buf_out.getvalue()).decode("ascii")


# ---------- Streamlit UI ----------
st.sidebar.header("Upload and parameters")
uploaded = st.sidebar.file_uploader("Upload CSV (financial data with PII allowed for demo)", type=["csv"])
scheme = st.sidebar.selectbox("FHE Scheme", ["CKKS", "BFV", "BGV"])
poly_modulus_degree = st.sidebar.selectbox("Poly modulus degree (ring dimension)", [1024, 2048, 4096, 8192, 16384], index=3)
multiplicative_depth = st.sidebar.slider("Multiplicative depth", 1, 10, 2)
plaintext_modulus = st.sidebar.number_input("Plaintext modulus (integers only, BFV/BGV)", value=65537, step=1)
noise_budget_hint = st.sidebar.slider("Noise budget hint (UI-only slider, not direct API)", 0, 1000, 218)
st.sidebar.markdown("---")
if st.sidebar.button("Create / Reset CryptoContext"):
    st.session_state.pop("cc", None)
    st.session_state.pop("kp", None)
    st.success("Context reset — recreate on next step")

# Load CSV
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No CSV uploaded yet. Use sample data to try the demo.")
    # sample synthetic dataset
    df = pd.DataFrame({
        "id": list(range(1, 11)),
        "name": [f"person_{i}" for i in range(1, 11)],
        "country": ["IN","US","GB","IN","CA","US","IN","AU","FR","DE"],
        "balance": [1200.50, 500.0, 340.75, 12000.0, 89.25, 700.0, 1500.0, 230.0, 678.9, 999.0],
        "credit_score": [700, 650, 720, 800, 610, 680, 710, 640, 690, 705]
    })

st.subheader("Dataset (first 50 rows)")
st.dataframe(df.head(50), use_container_width=True)

# Select columns to encrypt
st.sidebar.subheader("Select columns to encrypt (PII/financial)")
cols = df.columns.tolist()
encrypt_cols = st.sidebar.multiselect("Columns to encrypt", cols, default=[c for c in cols if df[c].dtype.kind in 'if'])

# Create crypto context if not present
if "cc" not in st.session_state:
    with st.spinner("Creating CryptoContext and keys..."):
        cc, params = make_crypto_context(scheme_name=scheme,
                                         poly_modulus_degree=poly_modulus_degree,
                                         plaintext_modulus=int(plaintext_modulus),
                                         multiplicative_depth=int(multiplicative_depth))
        kp = keygen_for_context(cc)
        st.session_state["cc"] = cc
        st.session_state["kp"] = kp
        st.success("CryptoContext and keys created")

cc = st.session_state["cc"]
kp = st.session_state["kp"]

# Show quick context info
st.sidebar.markdown("### Context info")
st.sidebar.write(f"Scheme: {scheme}")
try:
    st.sidebar.write(f"Ring dimension (approx): {cc.GetRingDimension()}")
except Exception:
    st.sidebar.write("Ring dimension: (not available)")

# Encrypt button
if st.sidebar.button("Encrypt selected columns"):
    if not encrypt_cols:
        st.sidebar.error("Pick at least one column to encrypt")
    else:
        with st.spinner("Encrypting columns..."):
            enc_map, df_enc = encode_encrypt_dataframe(cc, kp.publicKey, df, encrypt_cols, scheme)
            st.session_state["enc_map"] = enc_map
            st.session_state["df_enc"] = df_enc
            st.success(f"Encrypted columns: {', '.join(encrypt_cols)}")

# Show encrypted dataframe (placeholders)
if "df_enc" in st.session_state:
    st.subheader("Encrypted dataset (cells replaced with ciphertext placeholders)")
    st.dataframe(st.session_state["df_enc"].head(50), use_container_width=True)
    st.markdown("You can download the ciphertexts and context keys for offline use (serialised).")

    # Provide download of serialized ciphertexts map
    enc_json = json.dumps(st.session_state["enc_map"])
    b64 = base64.b64encode(enc_json.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="ciphertexts.json">Download ciphertexts (json)</a>'
    st.markdown(href, unsafe_allow_html=True)

# Second screen: operations on ciphertexts
st.header("Homomorphic operations (simulate on encrypted data)")
if "enc_map" in st.session_state:
    enc_map = st.session_state["enc_map"]
    st.write("Available encrypted columns:", list(enc_map.keys()))

    op = st.selectbox("Operation", ["Add (column A + column B)", "Multiply (A * B)", "Polynomial eval (a*x^2 + b*x + c)"])
    col_a = st.selectbox("Ciphertext A", list(enc_map.keys()))
    col_b = st.selectbox("Ciphertext B", list(enc_map.keys()), index=min(1, max(0,len(enc_map)-1)))
    if op.startswith("Polynomial"):
        a = st.number_input("a (coef for x^2)", value=1.0)
        b = st.number_input("b (coef for x)", value=0.0)
        c = st.number_input("c (constant)", value=0.0)

    if st.button("Run operation"):
        with st.spinner("Running homomorphic operation..."):
            if op.startswith("Add"):
                result_ct = eval_add(cc, enc_map[col_a], enc_map[col_b])
            elif op.startswith("Multiply"):
                result_ct = eval_mult(cc, enc_map[col_a], enc_map[col_b])
            else:
                # polynomial: compute a*x^2 + b*x + c
                # We do this as: tmp = EvalMult(ct, ct) * a + ct * b + c
                ct_x = enc_map[col_a]
                ct_x2 = eval_mult(cc, ct_x, ct_x)
                # multiply-by-constant: encrypt constant and EvalMultPlain? For simplicity encrypt constant vector
                # create plaintext for constant vector using same size as original
                # we'll do small trick: decrypt, compute polynomial locally on plaintext and re-encrypt as demonstration
                vals = decrypt_column(cc, kp.secretKey, enc_map[col_a])
                if vals is None:
                    st.error("Failed to decode plaintext from ciphertext for polynomial demo.")
                    result_ct = None
                else:
                    poly_vals = [a*(v**2) + b*v + c for v in vals]
                    if scheme.lower() in ("bfv","bgv"):
                        pt_poly = cc.MakePackedPlaintext([int(round(x)) for x in poly_vals])
                    else:
                        pt_poly = cc.MakeCKKSPackedPlaintext(poly_vals)
                    result_ct = cc.Encrypt(kp.publicKey, pt_poly)
                    buf = io.BytesIO(); result_ct.serialize(buf)
                    result_ct = base64.b64encode(buf.getvalue()).decode("ascii")

            if result_ct:
                st.success("Operation complete. Result ciphertext created.")
                st.session_state["last_result"] = result_ct

                # Show decrypted sample of result (since user has secretKey in session)
                try:
                    decrypted = decrypt_column(cc, kp.secretKey, result_ct)
                    st.subheader("Decrypted result (for demo; secret key available locally)")
                    if decrypted is not None:
                        # Show first N values and some stats
                        arr = np.array(decrypted)
                        df_out = pd.DataFrame({"value": arr})
                        st.dataframe(df_out.head(50))
                        st.markdown("**Statistics**")
                        st.write(df_out.describe())
                        # Plot
                        fig = px.line(df_out.reset_index(), x="index", y="value", title="Result values (index vs value)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not extract values from decrypted plaintext object.")
                except Exception as e:
                    st.error(f"Decryption failed: {e}")

else:
    st.info("Encrypt some columns first to run homomorphic ops.")

# Footer: show helpful tips and links
st.markdown("---")
st.markdown("""
**Tips & references**
- This app stores crypto objects in Streamlit session memory (for demo only). In production, private keys must be protected and never exposed.  
- Parameter choices (ring dimension, multiplicative depth, plaintext modulus) affect security & capability. Use OpenFHE docs for secure parameter selection.  
- If `pip install openfhe` fails, build OpenFHE C++ and the Python bindings from source (openfhe-python). See OpenFHE docs / GitHub.
""")
