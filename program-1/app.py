# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import json

from fhe_utils import (
    make_context, keygen, pack_column, encrypt_vector, deserialize_ciphertext,
    serialize_ciphertext, decrypt_vector, eval_add, eval_mul, polynomial_eval
)

st.set_page_config(layout="wide", page_title="FHE Streamlit Playground")

st.title("FHE Playground — Upload → Encrypt → Operate → Decrypt")

PAGE = st.sidebar.selectbox("Page", ["Upload & Encrypt", "FHE Playground"])

if PAGE == "Upload & Encrypt":
    st.header("Upload financial CSV (PII may be present)")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=True, infer_datetime_format=True)
        st.subheader("Sample data")
        st.dataframe(df.head(50))

        # Select columns to encrypt
        st.markdown("### Choose data to encrypt")
        cols = st.multiselect("Columns to encrypt", list(df.columns), default=list(df.columns)[:2])
        rows_sel = st.multiselect("Rows to encrypt (by index)", df.index.tolist()[:200], default=df.index.tolist()[:5])

        # encryption settings
        scheme = st.selectbox("Scheme", ["CKKS", "BFV", "BGV"])
        poly_degree = st.selectbox("poly_modulus_degree", [2048, 4096, 8192, 16384], index=2)
        plaintext_modulus = st.number_input("plaintext_modulus (BFV/BGV)", value=65537)
        coeff_bits = st.text_input("coeff_mod_bit_sizes (CKKS comma-separated)", value="60,40,40,60")

        if st.button("Generate context & keys"):
            coeff_list = [int(x.strip()) for x in coeff_bits.split(",")] if coeff_bits else None
            with st.spinner("Making crypto context..."):
                cc = make_context(scheme, poly_modulus_degree=poly_degree,
                                  plaintext_modulus=int(plaintext_modulus),
                                  coeff_mod_bit_sizes=coeff_list)
                keys = keygen(cc)
                st.session_state['cc'] = cc
                st.session_state['keys'] = keys
                st.success("Context and keys created")

        if 'cc' in st.session_state:
            cc = st.session_state['cc']
            keys = st.session_state['keys']

            st.write("Encrypt selection")
            if st.button("Encrypt selection"):
                encrypted_cells = {}
                for c in cols:
                    vec = pack_column(df.loc[rows_sel, c].values, scheme)
                    ct = encrypt_vector(cc, keys, vec, scheme)
                    encrypted_cells[c] = serialize_ciphertext(ct)

                # Add an 'encrypted' column for the UI
                encrypted_preview = df.loc[rows_sel, cols].copy()
                for c in cols:
                    encrypted_preview[c + "_enc"] = encrypted_cells[c]
                st.subheader("Encrypted preview (serialized ciphertexts)")
                st.dataframe(encrypted_preview.head(50))

                # store
                st.session_state['encrypted'] = encrypted_cells
                st.success("Encrypted selected rows/columns (serialized shown)")

            if 'encrypted' in st.session_state:
                st.markdown("### Download / Serialize keys & context")
                if st.button("Serialize keys to session"):
                    st.session_state['serialized_keys'] = True
                    st.success("Keys serialized in session (not persisted to disk). For production, export keys securely.")

else:
    st.header("FHE Playground — perform homomorphic ops")

    if 'cc' not in st.session_state:
        st.warning("No crypto context in session. Please go to 'Upload & Encrypt' and create context/keys and encrypt some data first.")
        st.stop()

    cc = st.session_state['cc']
    keys = st.session_state['keys']
    scheme = st.selectbox("Scheme (must match encryption one)", ["CKKS", "BFV", "BGV"])

    # Show available encrypted items
    encrypted = st.session_state.get('encrypted', {})
    chosen_cipher_name = st.selectbox("Choose a ciphertext to operate on", ["-- none --"] + list(encrypted.keys()))
    if chosen_cipher_name != "-- none --":
        b64ct = encrypted[chosen_cipher_name]
        ct = deserialize_ciphertext(cc, b64ct)
        st.write("Selected ciphertext (serialized length):", len(b64ct))

        st.markdown("### Operations")
        op = st.selectbox("Operation", ["Add (cipher+cipher)", "Mul (cipher*cipher)", "Scalar Mul (cipher * scalar)", "Polynomial eval"])
        if op == "Add (cipher+cipher)" or op == "Mul (cipher*cipher)":
            other = st.selectbox("Second ciphertext", ["-- none --"] + list(encrypted.keys()))
            if other != "-- none --":
                other_ct = deserialize_ciphertext(cc, encrypted[other])
                if st.button("Run operation"):
                    if op.startswith("Add"):
                        res_ct = eval_add(cc, ct, other_ct)
                    else:
                        res_ct = eval_mul(cc, ct, other_ct)
                    # decrypt and show
                    res = decrypt_vector(cc, keys, res_ct, scheme)
                    st.subheader("Decrypted result (vector)")
                    st.write(res)
                    st.session_state['last_result'] = res
                    st.success("Operation complete")
        elif op == "Scalar Mul (cipher * scalar)":
            scalar = st.number_input("Scalar", value=2.0)
            if st.button("Run scalar mul"):
                res_ct = eval_mul(cc, ct, cc.MakeCKKSPackedPlaintext([scalar]) if scheme.upper()=="CKKS" else cc.MakePackedPlaintext([int(scalar)]))
                res = decrypt_vector(cc, keys, res_ct, scheme)
                st.session_state['last_result'] = res
                st.write(res)
                st.success("Done")
        else:
            st.write("Polynomial coefficients (lowest-order first). Example for 2 + 3x + 4x^2: 2,3,4")
            coeffs_text = st.text_input("coeffs comma-separated", value="0,1")  # default identity f(x)=x
            if st.button("Evaluate polynomial"):
                coeffs = [float(x.strip()) for x in coeffs_text.split(",")]
                res_ct = polynomial_eval(cc, keys, ct, coeffs, scheme)
                res = decrypt_vector(cc, keys, res_ct, scheme)
                st.session_state['last_result'] = res
                st.write(res)
                st.success("Polynomial evaluated")

    # Stats & charts for last result
    if 'last_result' in st.session_state:
        arr = np.array(st.session_state['last_result'])
        st.subheader("Result stats & charts")
        st.write("Count:", arr.size)
        st.write("Mean:", float(np.mean(arr)))
        st.write("Std:", float(np.std(arr)))
        fig = px.histogram(arr, nbins=30, title="Distribution of decrypted result")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Notes:** This Playground is a demo. In production you MUST handle keys, serialization, and PII carefully: don't store private keys in session or disk without encryption.")
