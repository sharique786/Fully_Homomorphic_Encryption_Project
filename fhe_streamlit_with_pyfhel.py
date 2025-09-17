import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from openfhe import (
    CryptoContextFactory,
    BFVParameters,
    BGVParameters,
    CKKSParameters,
    KeyPair,
    Ciphertext,
    Plaintext
)

st.set_page_config(layout="wide", page_title="FHE Lab — OpenFHE Streamlit")
st.title("OpenFHE Financial Lab — Server holds only public keys")

# Helpers
def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def ub64(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

# Session store for contexts per country
if "contexts" not in st.session_state:
    st.session_state.contexts = {}

with st.sidebar:
    st.header("Create Context (per country)")
    scheme = st.selectbox("Scheme", ["BFV", "BGV", "CKKS"])
    country = st.text_input("Country / Owner name")
    if st.button("Create"):
        params = {
            "BFV": BFVParameters,
            "BGV": BGVParameters,
            "CKKS": CKKSParameters
        }[scheme]()
        context = CryptoContextFactory.create_context(params)
        keypair = context.new_key_pair()
        pub = context.serialize_public_key(keypair.publicKey())
        sec = context.serialize_secret_key(keypair.privateKey())
        st.session_state.contexts[country] = {
            "scheme": scheme,
            "context": context,
            "public_key": pub
        }
        st.success(f"Public key stored for {country}. Download your secret key below:")
        st.download_button("Download Secret Key", data=sec, file_name=f"{country}_secret.key")

    st.markdown("### Public Keys Stored:")
    for c, info in st.session_state.contexts.items():
        st.write(f"- {c}  ({info['scheme']})")

st.header("Encrypt / Operate / Decrypt")

tabs = st.tabs(["Encrypt CSV", "Homomorphic Ops", "Decrypt & Visualize"])
df_enc = None

with tabs[0]:
    st.subheader("Encrypt CSV")
    uploaded = st.file_uploader("Upload plaintext CSV", type="csv")
    sel = st.selectbox("Encrypt under country", options=list(st.session_state.contexts.keys()))
    if uploaded and sel:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = st.multiselect("Numeric columns to encrypt", num_cols, default=num_cols)
        if st.button("Encrypt & Download CSV"):
            info = st.session_state.contexts[sel]
            ctx = info["context"]
            enc_df = df.copy()
            for col in cols:
                enc_df[col] = df[col].astype(float).apply(
                    lambda v: b64(ctx.serialize_ciphertext(ctx.encrypt(ctx.make_plaintext(v))))
                )
            enc_df["_openfhe_encrypted"] = True
            st.download_button("Download Encrypted CSV", data=enc_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"encrypted_{sel}.csv")

with tabs[1]:
    st.subheader("Homomorphic Ops")
    uploaded_enc = st.file_uploader("Upload encrypted CSV", type="csv")
    sel2 = st.selectbox("Use public key of", options=list(st.session_state.contexts.keys()), key="op_key")
    if uploaded_enc and sel2:
        df_e = pd.read_csv(uploaded_enc, dtype=str)
        st.dataframe(df_e.head())
        info = st.session_state.contexts[sel2]
        ctx = info["context"]
        col = st.selectbox("Encrypted column", options=[c for c in df_e.columns if not c.startswith("_")])
        op = st.selectbox("Operation", ["Add scalar", "Multiply scalar"])
        scalar = st.number_input("Scalar", value=2.0)
        if st.button("Apply (first 5 rows)"):
            out = []
            for i in range(min(5, len(df_e))):
                ctext = df_e.at[i, col]
                raw = ub64(ctext)
                ct = ctx.deserialize_ciphertext(raw)
                if op == "Add scalar":
                    pt = ctx.make_plaintext(float(scalar))
                    res = ctx.eval_add(ct, pt)
                else:
                    pt = ctx.make_plaintext(float(scalar))
                    res = ctx.eval_mult(ct, pt)
                out.append(b64(ctx.serialize_ciphertext(res)))
            st.json(out)

with tabs[2]:
    st.subheader("Decrypt")
    uploaded_enc2 = st.file_uploader("Upload encrypted CSV", type="csv", key="dec_enc")
    uploaded_sec = st.file_uploader("Upload secret key", type=None, key="sec_key")
    if uploaded_enc2 and uploaded_sec:
        df_e2 = pd.read_csv(uploaded_enc2, dtype=str)
        sec = uploaded_sec.read()
        st.session_state._last_secret = sec
        st.session_state._last_enc = df_e2
        st.success("Uploaded secret key and encrypted CSV; proceed to decrypt.")

    if hasattr(st.session_state, "_last_secret") and hasattr(st.session_state, "_last_enc"):
        df_e2 = st.session_state._last_enc
        sec = st.session_state._last_secret
        sel3 = st.selectbox("Use public context of", options=list(st.session_state.contexts.keys()), key="dec_key")
        info = st.session_state.contexts[sel3]
        ctx = info["context"]
        secret = ctx.deserialize_secret_key(sec)
        col = st.selectbox("Column to decrypt", options=[c for c in df_e2.columns if not c.startswith("_")], key="dec_col")
        if st.button("Decrypt & Show"):
            vals = []
            for v in df_e2[col].tolist():
                raw = ub64(v)
                ct = ctx.deserialize_ciphertext(raw)
                pt = ctx.decrypt(ct, secret)
                vals.append(pt.to_double())
            df_dec = pd.DataFrame({col: vals})
            st.dataframe(df_dec.head())
            st.download_button("Download decrypted CSV", data=df_dec.to_csv(index=False).encode("utf-8"),
                               file_name="decrypted.csv")

st.markdown("---")
st.markdown("**Notes:** OpenFHE (Python wrapper) requires Ubuntu 24.04 (or compatible) and Python ≥3.12 :contentReference[oaicite:2]{index=2}. The app keeps **only public keys** in server memory. Secret keys remain with data owners.")

