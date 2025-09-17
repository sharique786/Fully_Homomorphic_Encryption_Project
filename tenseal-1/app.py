import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fhe_utils import get_fhe_context

st.set_page_config(page_title="FHE Streamlit App", layout="wide")

st.title("🔐 Fully Homomorphic Encryption (FHE) Demo")

# Sidebar config
st.sidebar.header("Configuration")
backend = st.sidebar.selectbox("Encryption backend (try OpenFHE first)", ["pyfhel", "tenseal"])
scheme = st.sidebar.selectbox("FHE Scheme (numeric columns only)", ["CKKS", "BFV", "BGV"])
poly_modulus_degree = st.sidebar.selectbox("poly_modulus_degree", [2048, 4096, 8192, 16384])
noise = st.sidebar.slider("noise (relative)", 10, 100, 30)
relin = st.sidebar.checkbox("relinearization enabled", True)

# Initialize context
fhe = None
try:
    fhe = get_fhe_context(backend, scheme, poly_modulus_degree, noise, relin)
    if fhe:
        st.success(f"Context initialized using {backend.upper()} with {scheme}.")
    else:
        st.error("Failed to initialize context.")
except Exception as e:
    st.error(f"Failed to initialize context: {e}")

# File upload
st.header("1) Upload CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    st.header("2) Select columns to encrypt")
    cols_to_encrypt = st.multiselect("Select columns to encrypt", df.columns.tolist())

    if st.button("Encrypt selected columns"):
        if not fhe:
            st.error("FHE context not initialized!")
        elif not cols_to_encrypt:
            st.warning("Please select at least one column.")
        else:
            encrypted_cols = {}
            for col in cols_to_encrypt:
                try:
                    numeric_series = pd.to_numeric(df[col], errors="coerce")
                    encrypted = [fhe.encrypt(val) for val in numeric_series.dropna()]
                    encrypted_cols[col] = encrypted
                except Exception as e:
                    st.error(f"Encryption failed for {col}: {e}")

            st.success("Columns encrypted.")

            # Choose operation
            st.header("3) Perform homomorphic operation")
            operation = st.selectbox("Operation", ["add", "mul", "poly_eval"])

            if st.button("Run operation"):
                try:
                    encrypted_results = []
                    if operation == "add":
                        for i in range(len(list(encrypted_cols.values())[0])):
                            res = encrypted_cols[cols_to_encrypt[0]][i]
                            for col in cols_to_encrypt[1:]:
                                res += encrypted_cols[col][i]
                            encrypted_results.append(res)
                    elif operation == "mul":
                        for i in range(len(list(encrypted_cols.values())[0])):
                            res = encrypted_cols[cols_to_encrypt[0]][i]
                            for col in cols_to_encrypt[1:]:
                                res *= encrypted_cols[col][i]
                            encrypted_results.append(res)
                    elif operation == "poly_eval":
                        for val in encrypted_cols[cols_to_encrypt[0]]:
                            # Example: evaluate polynomial f(x) = x^2 + 2x + 1
                            res = (val * val) + (val * 2) + 1
                            encrypted_results.append(res)

                    # Decrypt results before storing
                    results = []
                    for enc_val in encrypted_results:
                        try:
                            dec_val = fhe.decrypt(enc_val)
                            results.append(dec_val)
                        except Exception as e:
                            results.append(f"<error: {str(e)}>")

                    res_df = pd.DataFrame({f"result_{operation}": results})
                    st.write("Statistics of result", res_df.describe(include="all"))

                    # Plot only numeric results
                    numeric_res = pd.to_numeric(res_df[f"result_{operation}"], errors="coerce")
                    if numeric_res.notnull().any():
                        fig, ax = plt.subplots()
                        numeric_res.plot(kind="line", title=f"Result: {operation}", ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("No numeric data to plot (all results invalid).")

                except Exception as e:
                    st.error(f"Operation failed: {e}")
