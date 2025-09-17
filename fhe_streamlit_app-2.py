"""
Streamlit app demonstrating homomorphic operations (TenSEAL) on tabular CSV financial data.
Supports CKKS (approx) and BFV (integer). BGV is not provided by TenSEAL; see Pyfhel/OpenFHE for that.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tenseal as ts
import base64
import io
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

st.set_page_config(layout="wide", page_title="FHE Financial Demo (TenSEAL)")

# --- Utilities for serializing encrypted vectors / contexts ---
def serialize_ciphertexts(ciphertexts):
    """Serialize list of TenSEAL ciphertexts to bytes (base64) so they can be saved as csv cells."""
    serialized = []
    for ct in ciphertexts:
        b = ct.serialize()
        serialized.append(base64.b64encode(b).decode("utf-8"))
    return serialized

def deserialize_ciphertext(b64_bytes: str, context: ts.Context):
    b = base64.b64decode(b64_bytes.encode("utf-8"))
    return ts.ckks_vector_from(context, b) if st.session_state.active_scheme == "CKKS" else ts.bfv_vector_from(context, b)

def save_encrypted_csv(df: pd.DataFrame) -> bytes:
    """Return bytes for CSV ready to download"""
    return df.to_csv(index=False).encode("utf-8")

# --- Session-state helpers to store per-country contexts/keys ---
if "country_keys" not in st.session_state:
    # map country -> dict with keys: context (serialized), public_context (serialized), meta
    st.session_state.country_keys = {}

def context_to_dict(ctx: ts.Context):
    """Serialize context and public/secret keys as necessary."""
    # TenSEAL contexts can be serialized; for secret key keep private
    ser_ctx = ctx.serialize()
    # public context (without secret key) can be created by copying and removing secret key;
    # TenSEAL provides a method context.serialize_public() — but in some versions you must manually zero secret_key
    try:
        ser_pub = ctx.serialize_public()
    except Exception:
        # best-effort: create a copy, remove secret key attribute if present
        # fallback: store same serialized context (be careful in production)
        ser_pub = ser_ctx
    return {"context": base64.b64encode(ser_ctx).decode("utf-8"),
            "public": base64.b64encode(ser_pub).decode("utf-8"),
            "scheme": st.session_state.active_scheme}

def dict_to_context(b64_ser: str) -> ts.Context:
    raw = base64.b64decode(b64_ser.encode("utf-8"))
    return ts.Context.load(raw)

# --- UI: left column for key management and params ---
st.title("Homomorphic Financial Data Lab — TenSEAL demo")
st.markdown("Encrypt CSVs per-country, run stats and transformations on encrypted data, visualize results. Keys are stored in session only (demo).")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Key / Context Manager")

    scheme = st.selectbox("Scheme", ["CKKS", "BFV"])
    poly_modulus_degree = st.selectbox("poly_modulus_degree", [4096, 8192, 16384], index=1)
    coeff_mod_bit_sizes_str = st.text_input("coeff_mod_bit_sizes (comma sep bits)", "60, 40, 40")
    coeff_mod_bit_sizes = [int(x.strip()) for x in coeff_mod_bit_sizes_str.split(",") if x.strip()]
    global_scale = st.number_input("global scale (CKKS) (as 2**x)", min_value=2**20, value=2**40, step=1)
    plain_modulus = st.number_input("plain_modulus (BFV only)", min_value=2**8, value=2**13, step=1)
    relinearize = st.checkbox("Enable relinearization (on homomorphic mult ops)", value=True)

    st.markdown("### Create new country key")
    country = st.text_input("Country name (owner)")
    if st.button("Create key for country"):
        if not country:
            st.error("Enter a country name")
        else:
            # create context
            sc = ts.SCHEME_TYPE.CKKS if scheme == "CKKS" else ts.SCHEME_TYPE.BFV
            if sc == ts.SCHEME_TYPE.CKKS:
                ctx = ts.context(
                    sc,
                    poly_modulus_degree=poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                )
                ctx.global_scale = global_scale
            else:
                ctx = ts.context(
                    sc,
                    poly_modulus_degree=poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                    plain_modulus=plain_modulus
                )
            st.session_state.active_scheme = scheme
            print(f"Creating context for {country} with scheme {st.session_state.active_scheme}, poly_modulus_degree={poly_modulus_degree},"
                  f" coeff_mod_bit_sizes={coeff_mod_bit_sizes}, global_scale={global_scale}, plain_modulus={plain_modulus}")
            ctx.generate_galois_keys()
            ctx.generate_relin_keys()
            st.session_state.country_keys[country] = context_to_dict(ctx)
            st.success(f"Context created for {country} (scheme={scheme})")
    st.markdown("Existing country keys (session-only)")
    for c in st.session_state.country_keys.keys():
        st.write("-", c, st.session_state.country_keys[c]["scheme"])

    st.markdown("---")
    st.markdown("### Upload encrypted CSV for processing (TenSEAL serialized cells)")
    uploaded_enc = st.file_uploader("Encrypted CSV (TenSEAL-serialized cells)", type=["csv"], key="enc_upload")

    st.markdown("---")
    st.markdown("### Crypto notes")
    st.caption("CKKS = approximate (floats). BFV = integers. Parameter changes affect noise budget and capability; larger poly_modulus_degree and coeff_mod_bit_sizes increase security/noise budget but cost CPU & memory.")

with col2:
    st.header("Data upload / encryption / operations")

    tab1, tab2, tab3 = st.tabs(["Encrypt CSV", "Homomorphic Ops", "Visualization & Download"])

    with tab1:
        st.subheader("Upload plaintext CSV and encrypt it under a country's key")
        uploaded = st.file_uploader("Upload plaintext CSV (must contain numeric columns to encrypt)", type=["csv"], key="plain_upload")
        target_country = st.selectbox("Encrypt for country", options=list(st.session_state.country_keys.keys()) or ["-- create key first --"])
        encrypt_cols = st.multiselect("Columns to encrypt (numeric)", [])
        if uploaded is not None:
            df_plain = pd.read_csv(uploaded)
            # show inferred numeric cols
            numeric_cols = df_plain.select_dtypes(include=[np.number]).columns.tolist()
            if not encrypt_cols:
                encrypt_cols = st.multiselect("Columns to encrypt (numeric)", numeric_cols, default=numeric_cols)
            st.dataframe(df_plain.head(5))
            if st.button("Encrypt CSV under selected country"):
                if target_country not in st.session_state.country_keys:
                    st.error("Choose a valid target country (create one first)")
                else:
                    ctx_ser = st.session_state.country_keys[target_country]["context"]
                    ctx = dict_to_context(ctx_ser)
                    # we expect to encrypt row-wise vectors for each numeric column or each row as vector
                    df_enc = df_plain.copy()
                    # For demo: encrypt each numeric column as a vector (column-wise), serialize to base64 cell
                    for col in encrypt_cols:
                        arr = df_plain[col].astype(float).tolist()
                        if st.session_state.active_scheme == "CKKS":
                            enc_vec = ts.ckks_vector(ctx, arr)
                        else:
                            # BFV vector API expects integers (rounded)
                            enc_vec = ts.bfv_vector(ctx, [int(round(v)) for v in arr])
                        df_enc[col] = serialize_ciphertexts([enc_vec])[0]  # store single serialized vector in column cell
                    # Save a metadata column showing this is TenSEAL serialized
                    df_enc["_tenseal_encrypted"] = True
                    st.session_state["last_encrypted_df"] = df_enc
                    st.success("Encrypted dataframe created in session. Download below.")
                    st.download_button("Download Encrypted CSV", data=save_encrypted_csv(df_enc), file_name=f"encrypted_{target_country}.csv")

    with tab2:
        st.subheader("Perform homomorphic operations on uploaded encrypted CSV (or use last encrypted in session)")
        use_session_df = st.checkbox("Use last encrypted dataframe in session", value=True)
        if use_session_df and "last_encrypted_df" in st.session_state:
            df_enc = st.session_state["last_encrypted_df"]
            st.write("Using session encrypted dataframe sample:")
            st.dataframe(df_enc.head(3))
        elif uploaded_enc is not None:
            df_enc = pd.read_csv(uploaded_enc)
            st.write("Uploaded encrypted CSV sample:")
            st.dataframe(df_enc.head(3))
        else:
            st.info("Upload or encrypt a CSV first.")
            df_enc = None

        if df_enc is not None:
            # Ask user for which country key to use for public operations
            op_country = st.selectbox("Public context (country) used to interpret encrypted cells", options=list(st.session_state.country_keys.keys()))
            op_ctx = dict_to_context(st.session_state.country_keys[op_country]["context"])

            # let user select which encrypted column stores serialized vector (we used that format above)
            enc_cols = [c for c in df_enc.columns if c not in ["_tenseal_encrypted"]]
            chosen_enc_col = st.selectbox("Choose encrypted column (serialized vector cell)", enc_cols)

            st.markdown("### Operations (performed on ciphertexts, without decrypting)")
            op = st.selectbox("Operation", ["Add scalar", "Multiply by scalar", "Elementwise multiply (with plaintext vector)", "Polynomial evaluation (coeffs)", "Sum (encrypted vector)"])
            scalar = st.number_input("Scalar value (for add/mul)", value=2.0)
            poly_coeffs_text = st.text_input("Polynomial coeffs (comma-separated low->high) e.g. '1,0,2' => 1 + 0*x + 2*x^2", "1,0,2")
            plain_vector_text = st.text_input("Plaintext vector (comma sep) for elementwise multiply", "1,1,1")
            run_op = st.button("Run operation homomorphically")
            if run_op:
                # read serialized ciphertext from first row cell (assuming column stores serialized vector)
                cell = df_enc.iloc[0][chosen_enc_col]
                # deserialize
                try:
                    # detect if BFV or CKKS
                    if st.session_state.active_scheme == "CKKS":
                        enc_vec = ts.ckks_vector_from(op_ctx, base64.b64decode(cell.encode("utf-8")))
                    else:
                        enc_vec = ts.bfv_vector_from(op_ctx, base64.b64decode(cell.encode("utf-8")))
                except Exception as e:
                    st.error(f"Failed to deserialize ciphertext: {e}")
                    enc_vec = None

                if enc_vec is not None:
                    # perform ops on ciphertext
                    if op == "Add scalar":
                        res_ct = enc_vec + scalar
                    elif op == "Multiply by scalar":
                        res_ct = enc_vec * scalar
                        if relinearize:
                            try:
                                res_ct.relinearize()
                            except Exception:
                                pass
                    elif op == "Elementwise multiply (with plaintext vector)":
                        plain_vec = [float(x.strip()) for x in plain_vector_text.split(",") if x.strip()]
                        res_ct = enc_vec * plain_vec
                    elif op == "Polynomial evaluation (coeffs)":
                        coeffs = [float(x.strip()) for x in poly_coeffs_text.split(",") if x.strip()]
                        # Horner's method on ciphertexts (poly applied elementwise)
                        # Start with zero ciphertext (encode scalar 0)
                        # For CKKS we can create a plaintext scalar vector by encrypting a vector of same length
                        # Simpler: evaluate polynomial using repeated multiplication/add on ciphertext
                        # WARNING: multiplicative depth increases with degree
                        # We'll implement Horner:
                        # res = coeffs[-1] ; for c in reversed(coeffs[:-1]): res = res * x + c
                        res_ct = None
                        try:
                            for c in reversed(coeffs):
                                if res_ct is None:
                                    # start with constant c (encrypt constant)
                                    res_ct = enc_vec * 0  # zero vector, then add c
                                    res_ct += c
                                else:
                                    res_ct = res_ct * enc_vec
                                    if relinearize:
                                        try: res_ct.relinearize()
                                        except Exception: pass
                                    res_ct += c
                        except Exception as e:
                            st.error(f"Polynomial evaluation failed: {e}")
                            res_ct = None
                    elif op == "Sum (encrypted vector)":
                        # sum all slots: requires Galois rotations (TenSEAL supports .sum())
                        try:
                            res_ct = enc_vec.sum()
                        except Exception:
                            # fallback: decrypt required to sum
                            res_ct = None
                            st.warning("Sum on ciphertexts requires Galois keys; ensure they were generated on context.")
                    else:
                        res_ct = None

                    if res_ct is not None:
                        # show that the operation was done on ciphertext: we only have ciphertext object
                        st.success("Operation completed on ciphertext (data remained encrypted).")
                        st.write("Serialized resulting ciphertext (base64) preview:")
                        ser = base64.b64encode(res_ct.serialize()).decode("utf-8")
                        st.code(ser[:400] + " ...")
                        # Optionally decrypt IF user provides secret key (owner)
                        st.markdown("### Decrypt (supply secret key by selecting owner country below)")
                        dec_country = st.selectbox("Provide secret key of country (owner) to decrypt result", options=list(st.session_state.country_keys.keys()), key="dec_country")
                        if st.button("Decrypt result (temporary; session-only)"):
                            # In this demo we use the stored context which already has secret key.
                            # In production the secret key must be kept private and not uploaded to the service.
                            try:
                                dec_ctx = dict_to_context(st.session_state.country_keys[dec_country]["context"])
                                if st.session_state.active_scheme == "CKKS":
                                    res_plain = ts.ckks_vector_from(dec_ctx, res_ct.serialize()).decrypt()
                                else:
                                    res_plain = ts.bfv_vector_from(dec_ctx, res_ct.serialize()).decrypt()
                                st.write("Decrypted result (first 10 elements):")
                                st.write(np.array(res_plain)[:10].tolist())
                            except Exception as e:
                                st.error(f"Decryption failed: {e}")
                    else:
                        st.error("Operation failed or produced no ciphertext result.")

    with tab3:
        st.subheader("Visualize decrypted result (owner must provide secret key) or show encrypted stats")
        if "last_encrypted_df" in st.session_state:
            df_enc = st.session_state["last_encrypted_df"]
            st.write("Session encrypted dataset sample:")
            st.dataframe(df_enc.head(3))
            viz_country = st.selectbox("Decrypt & visualize using secret key of country", options=list(st.session_state.country_keys.keys()))
            if st.button("Decrypt & show stats/plots"):
                # decrypt numeric columns and show stats
                ctx = dict_to_context(st.session_state.country_keys[viz_country]["context"])
                numeric_columns = [c for c in df_enc.columns if c not in ["_tenseal_encrypted"]]
                data = {}
                for col in numeric_columns:
                    cell = df_enc.loc[0, col]  # we stored entire vector in single cell
                    try:
                        if st.session_state.active_scheme == "CKKS":
                            vec = ts.ckks_vector_from(ctx, base64.b64decode(cell.encode("utf-8")))
                        else:
                            vec = ts.bfv_vector_from(ctx, base64.b64decode(cell.encode("utf-8")))
                        plain = vec.decrypt()
                        data[col] = plain
                    except Exception as e:
                        st.error(f"Failed to decrypt column {col}: {e}")
                if data:
                    df_plain = pd.DataFrame(data)
                    st.write("Decrypted dataframe sample (first rows):")
                    st.dataframe(df_plain.head())
                    st.write("Descriptive statistics:")
                    st.dataframe(df_plain.describe())
                    # simple plot
                    num_cols = df_plain.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols:
                        fig, ax = plt.subplots()
                        df_plain[num_cols].hist(ax=ax)
                        st.pyplot(fig)
                    # allow download
                    st.download_button("Download decrypted CSV", data=save_encrypted_csv(df_plain), file_name="decrypted.csv")
        else:
            st.info("No encrypted dataframe found in session (use Encrypt CSV tab).")

st.markdown("---")
st.markdown("## Notes & Limitations")
st.markdown("""
- This demo keeps keys in Streamlit session state for convenience only; DO NOT do this in production.
- TenSEAL: production workloads should carefully manage contexts and secret keys. CKKS is approximate; BFV is integer only. Some homomorphic operations (e.g., complex poly evals, deep multiplicative circuits) may exhaust noise budget and require bootstrapping (not shown).
- If you need **BGV**, consider using Pyfhel or OpenFHE. This demo centers on TenSEAL (CKKS/BFV). 
""")
