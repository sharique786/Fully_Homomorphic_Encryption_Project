import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openfhe

st.set_page_config(page_title="OpenFHE Streamlit Demo", layout="wide")

st.title("🔐 OpenFHE Streamlit Application")

uploaded_file = st.file_uploader("Upload CSV file with financial + PII data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df)

    col_choice = st.multiselect("Select columns to encrypt", df.columns.tolist())
    row_choice = st.slider("Select rows to encrypt", 0, len(df), (0, len(df)))

    scheme = st.selectbox("Encryption Scheme", ["BFV", "BGV", "CKKS"])

    if st.button("Encrypt Selected Data"):
        st.success(f"Simulating {scheme} encryption on selected data...")
        enc_data = df.loc[row_choice[0]:row_choice[1], col_choice]
        st.dataframe(enc_data.style.highlight_max(color="lightblue"))

        st.subheader("Encrypted View (mock-up)")
        st.write(enc_data.applymap(lambda x: f"enc({x})"))

    st.markdown("---")
    st.subheader("Homomorphic Operations Playground")

    noise = st.slider("Noise Budget", 10, 200, 50)
    poly_mod = st.selectbox("Poly Modulus Degree", [1024, 2048, 4096, 8192])
    relin = st.checkbox("Enable Relinearization", value=True)

    st.write(f"Configured Parameters → Noise: {noise}, PolyModulus: {poly_mod}, Relinearization: {relin}")

    st.subheader("Simulated Statistics")
    stats = pd.DataFrame({
        "Operation": ["Addition", "Multiplication", "Polynomial Eval"],
        "Latency(ms)": [5, 12, 20],
        "NoiseBudgetUsed": [5, 10, 15]
    })
    st.dataframe(stats)

    fig, ax = plt.subplots()
    stats.plot(kind="bar", x="Operation", y="Latency(ms)", ax=ax, legend=False)
    plt.ylabel("Latency (ms)")
    st.pyplot(fig)
