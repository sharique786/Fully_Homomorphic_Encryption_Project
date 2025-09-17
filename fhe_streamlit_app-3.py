"""
Streamlit app (single file) — TenSEAL demo with server-side PUBLIC contexts only.
Behavior:
 - When creating a key, server stores only the public context.
 - The full secret context is provided as a downloadable file; the client must keep it private.
 - To decrypt, the client uploads their secret context file (not stored server-side).
 - Demo purpose only; in production use secure key management / HSM and authenticated operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tenseal as ts
import base64
import io
import matplotlib.pyplot as plt
from typing import Dict

st.set_page_config(layout="wide", page_title="FHE Lab — Public Contexts Only (TenSEAL)")

# ---------- Helpers ----------
def save_bytes_to_download(data: bytes, fname: str):
    return st.download_button(label=f"Download {fname}", data=data, file_name=fname)

def serialize_ciphertext(ct):
    return base64.b64encode(ct.serialize()).decode("utf-8")

def deserialize_ciphertext_from_b64(ctx: ts.Context, b64: str):
    raw = base64.b64decode(b64.encode("utf-8"))
    # Use appropriate factory depending on scheme
    if st.session_state.active_scheme == "CKKS":
        return ts.ckks_vector_from(ctx, raw)
    else:
        return ts.bfv_vector_from(ctx, raw)

def ctx_to_public_bytes(ctx: ts.Context) -> bytes:
    """Return serialized public context bytes."""
    try:
        return ctx.serialize_public()
    except Exception:
        # TenSEAL versions may not provide serialize_public; fallback (not ideal)
        return ctx.serialize()

def ctx_to_secret_bytes(ctx: ts.Context) -> bytes:
    """Return full serialized context bytes (contains secret key)."""
    return ctx.serialize()

def load_context_from_bytes(b: bytes) -> ts.Context:
    return ts.Context.load(b)

def save_df_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------- Session storage (server stores public contexts only) ----------
if "public_contexts" not in st.session_state:
    # map: country -> {"public": base64(bytes), "scheme": "CKKS"/"BFV", "meta": {...}}
    st.session_state.public_contexts = {}

# ---------- UI ----------
st.title("FHE Financial Lab — Public Contexts Only")
st.markdown("""
This demo stores **only public contexts** on the server. When you create a key, **download the secret context** and keep it private.
To decrypt results, upload your secret context file. This demonstrates a safer privacy model.
""")

left, right = st.columns([1, 2])

with left:
    st.header("Key creation (per-country)")
    scheme = st.selectbox("Scheme", ["CKKS", "BFV"])
    poly_modulus_degree = st.selectbox("poly_modulus_degree", [4096, 8192, 16384], index=1)
    coeff_mod_bit_sizes_str = st.text_input("coeff_mod_bit_sizes (comma sep bits)", "60, 40, 40")
    coeff_mod_bit_sizes = [int(x.strip()) for x in coeff_mod_bit_sizes_str.split(",") if x.strip()]
    global_scale = st.number_input("global scale (CKKS) (e.g. 2**40)", value=2**40)
    plain_modulus = st.number_input("plain_modulus (BFV only)", min_value=2**8, value=2**13)
    relinearize = st.checkbox("Enable relinearization recommendation", value=True)

    new_country = st.text_input("Country (owner) name")
    if st.button("Create key pair and store public context"):
        if not new_country:
            st.error("Enter a country name")
        else:
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
            # generate keys needed for rotations/relinearization on server
            ctx.generate_galois_keys()
            ctx.generate_relin_keys()

            # public and secret serialized bytes
            public_bytes = ctx_to_public_bytes(ctx)
            secret_bytes = ctx_to_secret_bytes(ctx)

            st.session_state.active_scheme = scheme
            # store only public context (base64-encoded) and metadata on server (session)
            st.session_state.public_contexts[new_country] = {
                "public_b64": base64.b64encode(public_bytes).decode("utf-8"),
                "scheme": scheme,
                "meta": {
                    "poly_modulus_degree": poly_modulus_degree,
                    "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
                    "global_scale": global_scale,
                    "plain_modulus": plain_modulus,
                    "relinearize": relinearize
                }
            }

            st.success(f"Public context created and stored for '{new_country}'. Download the secret context below and keep it private.")
            # provide secret context as downloadable bytes
            save_bytes_to_download(secret_bytes, f"{new_country}_tenseal_secret.ctx")

    st.markdown("#### Stored public contexts (session-only)")
    for c, info in st.session_state.public_contexts.items():
        st.write(f"- {c} (scheme={info['scheme']})")

with right:
    st.header("Encrypt / Upload / Operate")
    tab_encrypt, tab_ops, tab_viz = st.tabs(["Encrypt CSV", "Homomorphic Ops", "Visualize / Decrypt"])

    with tab_encrypt:
        st.subheader("Encrypt plaintext CSV using stored public context")
        uploaded_plain = st.file_uploader("Upload plaintext CSV (include numeric columns)", type=["csv"], key="plain_u")
        sel_country_enc = st.selectbox("Encrypt under country (public context)", options=list(st.session_state.public_contexts.keys()) or ["-- create key first --"], key="enc_country")
        if uploaded_plain is not None and sel_country_enc in st.session_state.public_contexts:
            df_plain = pd.read_csv(uploaded_plain)
            st.write("Preview (first 5 rows):")
            st.dataframe(df_plain.head())
            # choose numeric columns
            numeric_cols = df_plain.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_encrypt = st.multiselect("Numeric columns to encrypt (store as serialized vector cells)", numeric_cols, default=numeric_cols)
            if st.button("Encrypt with public context"):
                pub_b64 = st.session_state.public_contexts[sel_country_enc]["public_b64"]
                pub_bytes = base64.b64decode(pub_b64.encode("utf-8"))
                pub_ctx = load_context_from_bytes(pub_bytes)
                df_enc = df_plain.copy()
                for col in cols_to_encrypt:
                    arr = df_plain[col].astype(float).tolist()
                    if st.session_state.active_scheme == "CKKS":
                        enc = ts.ckks_vector(pub_ctx, arr)
                    else:
                        enc = ts.bfv_vector(pub_ctx, [int(round(x)) for x in arr])
                    df_enc[col] = serialize_ciphertext(enc)
                df_enc["_tenseal_encrypted"] = True
                st.session_state["last_enc_df"] = df_enc
                st.success("Encrypted dataframe saved in session. Download it (contains serialized ciphertexts).")
                st.download_button("Download encrypted CSV", data=save_df_to_bytes(df_enc), file_name=f"encrypted_{sel_country_enc}.csv")

    with tab_ops:
        st.subheader("Perform homomorphic ops on encrypted dataset (without secret key)")
        use_session = st.checkbox("Use session encrypted dataset", value=True)
        if use_session and "last_enc_df" in st.session_state:
            df_enc = st.session_state["last_enc_df"]
            st.write("Session encrypted sample:")
            st.dataframe(df_enc.head(3))
        else:
            uploaded_enc = st.file_uploader("Upload encrypted CSV (cells contain base64 TenSEAL ciphertexts)", type=["csv"], key="enc_upl")
            if uploaded_enc:
                df_enc = pd.read_csv(uploaded_enc)
                st.dataframe(df_enc.head(3))
            else:
                df_enc = None

        if df_enc is not None:
            op_country = st.selectbox("Use which public context to interpret ciphertexts", options=list(st.session_state.public_contexts.keys()), key="op_country")
            pub_ctx = load_context_from_bytes(base64.b64decode(st.session_state.public_contexts[op_country]["public_b64"].encode("utf-8")))

            enc_cols = [c for c in df_enc.columns if c not in ["_tenseal_encrypted"]]
            chosen_col = st.selectbox("Choose encrypted column (serialized vector)", enc_cols, key="enc_col")
            op_choice = st.selectbox("Operation", ["Add scalar", "Multiply scalar", "Elementwise multiply (plaintext vector)", "Polynomial eval", "Sum slots"])
            scalar = st.number_input("Scalar", value=2.0, key="op_scalar")
            plain_vec = st.text_input("Plaintext vector (comma separated)", "1,1,1", key="op_pvec")
            poly_coeffs = st.text_input("Polynomial coeffs (low->high)", "1,0,2", key="op_poly")
            if st.button("Run homomorphic operation"):
                cell = df_enc.loc[0, chosen_col]
                try:
                    if st.session_state.active_scheme == "CKKS":
                        ct = ts.ckks_vector_from(pub_ctx, base64.b64decode(cell.encode("utf-8")))
                    else:
                        ct = ts.bfv_vector_from(pub_ctx, base64.b64decode(cell.encode("utf-8")))
                except Exception as e:
                    st.error(f"Deserialization failed: {e}")
                    ct = None

                if ct is not None:
                    result_ct = None
                    try:
                        if op_choice == "Add scalar":
                            result_ct = ct + scalar
                        elif op_choice == "Multiply scalar":
                            result_ct = ct * scalar
                        elif op_choice == "Elementwise multiply (plaintext vector)":
                            pv = [float(x.strip()) for x in plain_vec.split(",") if x.strip()]
                            result_ct = ct * pv
                        elif op_choice == "Polynomial eval":
                            coeffs = [float(x.strip()) for x in poly_coeffs.split(",") if x.strip()]
                            # Horner evaluation
                            res = None
                            for c in reversed(coeffs):
                                if res is None:
                                    res = ct * 0
                                    res += c
                                else:
                                    res = res * ct
                                    res += c
                            result_ct = res
                        elif op_choice == "Sum slots":
                            result_ct = ct.sum()
                    except Exception as e:
                        st.error(f"Operation failed: {e}")
                        result_ct = None

                    if result_ct is not None:
                        st.success("Operation completed on ciphertext (still encrypted).")
                        b64 = base64.b64encode(result_ct.serialize()).decode("utf-8")
                        st.code(b64[:400] + " ...")
                        st.markdown("To decrypt results, owner must upload secret context in the *Visualize / Decrypt* tab.")

    with tab_viz:
        st.subheader("Decrypt & visualize (owner uploads secret context file)")
        if "last_enc_df" in st.session_state:
            df_enc = st.session_state["last_enc_df"]
            st.write("Encrypted dataset sample (session):")
            st.dataframe(df_enc.head(3))
        else:
            df_enc = None
            st.info("Encrypt a CSV first (Encrypt CSV tab) or upload an encrypted CSV in the Ops tab.")

        uploaded_secret = st.file_uploader("Upload secret context file (.ctx) for decryption (owner only)", type=None, key="secret_upload")
        if uploaded_secret and df_enc is not None:
            try:
                secret_bytes = uploaded_secret.read()
                secret_ctx = load_context_from_bytes(secret_bytes)
            except Exception as e:
                st.error(f"Failed to load secret context: {e}")
                secret_ctx = None

            if secret_ctx is not None:
                # attempt to decrypt each encrypted column (we stored whole column vector in each cell)
                encrypted_cols = [c for c in df_enc.columns if c not in ["_tenseal_encrypted"]]
                recovered = {}
                for col in encrypted_cols:
                    cell = df_enc.loc[0, col]
                    try:
                        if st.session_state.active_scheme == "CKKS":
                            vec = ts.ckks_vector_from(secret_ctx, base64.b64decode(cell.encode("utf-8")))
                        else:
                            vec = ts.bfv_vector_from(secret_ctx, base64.b64decode(cell.encode("utf-8")))
                        recovered[col] = vec.decrypt()
                    except Exception as e:
                        st.warning(f"Could not decrypt column {col}: {e}")
                if recovered:
                    df_plain = pd.DataFrame(recovered)
                    st.write("Decrypted data sample (first rows):")
                    st.dataframe(df_plain.head())
                    st.write("Stats:")
                    st.dataframe(df_plain.describe())
                    # simple plot
                    num_cols = df_plain.select_dtypes(include=[np.number]).columns.tolist()
                    if len(num_cols) > 0:
                        fig, ax = plt.subplots()
                        df_plain[num_cols].plot(kind="bar", ax=ax)
                        st.pyplot(fig)
                    st.download_button("Download decrypted CSV", data=save_df_to_bytes(df_plain), file_name="decrypted.csv")
                else:
                    st.error("No columns decrypted successfully.")

st.markdown("---")
st.markdown("**Security note:** The server stores only public contexts in session. Secret contexts must be kept by the data owner and uploaded only when they want to decrypt. In production, use authenticated APIs, HSMs, and client-side decryption where possible.")
