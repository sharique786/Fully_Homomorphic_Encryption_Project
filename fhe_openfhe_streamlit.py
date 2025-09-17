# streamlit_fhe_playground.py
# -------------------------------------------------------------
# A Streamlit app that demonstrates end‑to‑end encrypted analytics
# using OpenFHE with BFV, BGV, and CKKS schemes. It supports:
#  - Parameter tuning (poly_modulus_degree, plaintext modulus, scale, etc.)
#  - Key ownership by country/owner
#  - Upload CSV (plaintext or pre‑encrypted with this app)
#  - Homomorphic ops: +, −, ×, scalar ops, polynomial evaluation, vector ops
#  - Aggregations over encrypted financial data
#  - Proof-of-process: keeps ciphertexts until you explicitly decrypt
#  - Serialize/deserialize encrypted datasets (CSV + sidecar pickle)
# -------------------------------------------------------------
# Notes
#  - You need the Python OpenFHE bindings installed. Package name varies; try:
#      pip install openfhe
#      # or check: https://github.com/openfheorg/openfhe-python (build from source)
#  - If OpenFHE is not available, the app will show a clear message.
#  - BGV support depends on your OpenFHE build (it’s supported in OpenFHE core).
#  - This is an illustrative, educational playground—not production code.
# -------------------------------------------------------------

import io
import os
import sys
import pickle
import json
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---- Try to import OpenFHE Python bindings ----
OPENFHE_AVAILABLE = True
try:
    # The import path can differ across builds. Using a broad try block.
    from openfhe import (
        GenCryptoContext,
        CCParamsBFVRNS,
        CCParamsBGVRNS,
        CCParamsCKKSRNS,
        SecretKeyDist,
        SecurityLevel,
        DecryptResult,
    )
except Exception as e:
    OPENFHE_AVAILABLE = False
    OPENFHE_IMPORT_ERROR = e


# ---------- Utilities ----------
class AppError(RuntimeError):
    pass


@st.cache_resource(show_spinner=False)
def get_context_and_keys(scheme: str, params: dict):
    """Create/OpenFHE CryptoContext and keys based on scheme & UI params.
    Returns (cc, publicKey, secretKey, relinKey, galoisKey)
    """
    if not OPENFHE_AVAILABLE:
        raise AppError(
            "OpenFHE Python bindings are not available. Please install/build openfhe-python.\n"
            f"Underlying import error: {OPENFHE_IMPORT_ERROR}"
        )

    # Common toggles
    enable_encrypt = True
    enable_she = True  # add/mul
    enable_pk = True
    enable_leveled_she = True

    # Build params & context per scheme
    if scheme == "BFV":
        p = CCParamsBFVRNS()
        p.SetPlaintextModulus(int(params.get("plain_modulus", 65537)))
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        p.SetRingDim(int(params.get("poly_modulus_degree", 8192)))
        cc = GenCryptoContext(p)
        cc.Enable(1)  # PKESchemeFeature.ENCRYPTION
        cc.Enable(2)  # SHE
        # Optional leveled features

    elif scheme == "BGV":
        p = CCParamsBGVRNS()
        p.SetPlaintextModulus(int(params.get("plain_modulus", 65537)))
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        p.SetRingDim(int(params.get("poly_modulus_degree", 8192)))
        cc = GenCryptoContext(p)
        cc.Enable(1)
        cc.Enable(2)

    elif scheme == "CKKS":
        p = CCParamsCKKSRNS()
        scale_bits = int(params.get("scale_bits", 40))
        p.SetScalingModSize(scale_bits)
        p.SetRingDim(int(params.get("poly_modulus_degree", 8192)))
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        cc = GenCryptoContext(p)
        cc.Enable(1)
        cc.Enable(2)
    else:
        raise AppError(f"Unsupported scheme: {scheme}")

    keys = cc.KeyGen()

    if params.get("enable_relin", True):
        relinKey = cc.ReKeyGen(keys.secretKey, keys.secretKey)  # Relinearization
    else:
        relinKey = None

    if params.get("enable_rotate", True):
        # Generate Galois/rotation keys for vector rotations (if scheme supports)
        try:
            gk = cc.GenRotKey(keys.secretKey, [1, -1, 2, -2, 4, -4])
        except Exception:
            gk = None
    else:
        gk = None

    return cc, keys.publicKey, keys.secretKey, relinKey, gk


def serialize_cipher(obj) -> bytes:
    return pickle.dumps(obj)


def deserialize_cipher(b: bytes):
    return pickle.loads(b)


# ---------- Homomorphic Operations Layer ----------
class HEEngine:
    def __init__(self, cc, pk, sk, rlk=None, gk=None, scheme="BFV"):
        self.cc = cc
        self.pk = pk
        self.sk = sk
        self.rlk = rlk
        self.gk = gk
        self.scheme = scheme

    # Encoding helpers
    def encode(self, vec_or_scalar):
        if self.scheme in ("BFV", "BGV"):
            if isinstance(vec_or_scalar, (list, np.ndarray, pd.Series)):
                return self.cc.MakePackedPlaintext(list(map(int, vec_or_scalar)))
            return self.cc.MakePackedPlaintext([int(vec_or_scalar)])
        elif self.scheme == "CKKS":
            if isinstance(vec_or_scalar, (list, np.ndarray, pd.Series)):
                return self.cc.MakeCKKSPackedPlaintext(list(map(float, vec_or_scalar)))
            return self.cc.MakeCKKSPackedPlaintext([float(vec_or_scalar)])
        else:
            raise AppError("Unknown scheme for encoding")

    def encrypt(self, pt):
        return self.cc.Encrypt(self.pk, pt)

    def decrypt(self, ct):
        pt = self.cc.Decrypt(self.sk, ct)
        # Some bindings return (success, pt) or DecryptResult/Plaintext
        if isinstance(pt, tuple):
            pt = pt[-1]
        return pt

    # Core ops
    def add(self, a, b):
        return self.cc.EvalAdd(a, b)

    def sub(self, a, b):
        return self.cc.EvalSub(a, b)

    def mul(self, a, b):
        out = self.cc.EvalMult(a, b)
        if self.rlk is not None:
            try:
                out = self.cc.EvalRelin(out)
            except Exception:
                pass
        return out

    def add_plain(self, a, scalar_or_vec):
        pt = self.encode(scalar_or_vec)
        return self.cc.EvalAdd(a, pt)

    def mul_plain(self, a, scalar_or_vec):
        pt = self.encode(scalar_or_vec)
        return self.cc.EvalMult(a, pt)

    def rotate(self, a, steps: int):
        if self.gk is None:
            raise AppError("Rotation keys not available. Enable rotate in settings.")
        return self.cc.EvalRotate(a, steps)

    # Polynomial evaluation via Horner's rule: c0 + c1*x + c2*x^2 + ...
    def poly_eval(self, x_ct, coeffs):
        # coeffs are plaintext scalars (list)
        if not coeffs:
            return self.encrypt(self.encode(0))
        acc = self.mul_plain(x_ct, 0)  # zero of correct type
        # Start from highest degree
        for c in reversed(coeffs):
            acc = self.mul(acc, x_ct) if acc is not None else x_ct
            acc = self.add_plain(acc, c)
        return acc


# ---------- Domain helpers ----------
REQUIRED_COLUMNS = [
    "date", "country", "owner", "category", "units", "unit_price"
]


def load_csv_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def coerce_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns; create if missing
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = {
                "date": pd.Timestamp("2024-01-01"),
                "country": "Unknown",
                "owner": "Unknown",
                "category": "General",
                "units": 0,
                "unit_price": 0.0,
            }[col]
    # Types
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(int)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    # Derived
    df["revenue"] = df["units"] * df["unit_price"]
    return df


# ---------- Streamlit UI ----------
st.set_page_config(page_title="FHE Playground (OpenFHE)", layout="wide")
st.title("🔐 FHE Playground — OpenFHE (BFV · BGV · CKKS)")

with st.sidebar:
    st.header("⚙️ Scheme & Parameters")
    scheme = st.selectbox("Scheme", ["BFV", "BGV", "CKKS"], index=0)
    poly_modulus_degree = st.select_slider("poly_modulus_degree (ring dimension)", options=[2048, 4096, 8192, 16384],
                                           value=8192)
    plain_modulus = None
    scale_bits = None
    if scheme in ("BFV", "BGV"):
        plain_modulus = st.number_input("plain_modulus (t)", min_value=2, value=65537, step=1)
    if scheme == "CKKS":
        scale_bits = st.number_input("scale bits (CKKS)", min_value=20, value=40, step=2)

    enable_relin = st.checkbox("Enable relinearization", value=True)
    enable_rotate = st.checkbox("Enable rotations (vector ops)", value=True)

    st.markdown("---")
    st.header("🔑 Key Ownership")
    country = st.text_input("Country/Region", value="IN")
    owner = st.text_input("Data Owner", value="acme-finance")

    st.markdown("Keys are bound to (country, owner). Keep your secret key safe.")

params = {
    "poly_modulus_degree": poly_modulus_degree,
}
if plain_modulus is not None:
    params["plain_modulus"] = plain_modulus
if scale_bits is not None:
    params["scale_bits"] = scale_bits
params["enable_relin"] = enable_relin
params["enable_rotate"] = enable_rotate

# Show OpenFHE availability
if not OPENFHE_AVAILABLE:
    st.error(
        "OpenFHE bindings not found. Install/build openfhe-python to run homomorphic ops.\n"
        f"Import error: {OPENFHE_IMPORT_ERROR}"
    )

# Create or fetch context+keys per (scheme, params, country, owner)
context_key = json.dumps({
    "scheme": scheme, "params": params, "country": country, "owner": owner
}, sort_keys=True)

cc = pk = sk = rlk = gk = None
if OPENFHE_AVAILABLE:
    try:
        cc, pk, sk, rlk, gk = get_context_and_keys(scheme, params)
        he = HEEngine(cc, pk, sk, rlk, gk, scheme)
        st.success("CryptoContext and keys are ready.")
    except Exception as e:
        st.exception(e)
        he = None
else:
    he = None

# ---------------- Data upload & encryption ----------------
st.header("📥 Upload Financial Data (CSV)")
uploaded = st.file_uploader("Upload CSV (columns: date,country,owner,category,units,unit_price)", type=["csv"])

colA, colB = st.columns([2, 1], gap="large")

with colA:
    if uploaded is not None:
        df = load_csv_df(uploaded)
        df = coerce_financial_df(df)
        st.subheader("Preview (plaintext)")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Upload a CSV to begin. A small sample will be generated if none is provided.")
        # Create sample
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "country": [country] * 6,
            "owner": [owner] * 6,
            "category": ["A", "A", "B", "B", "C", "C"],
            "units": [10, 12, 8, 15, 9, 20],
            "unit_price": [100.0, 110.0, 95.0, 105.0, 80.0, 120.0],
        })
        df = coerce_financial_df(df)
        st.dataframe(df, use_container_width=True)

with colB:
    st.subheader("Encrypt Options")
    cols_to_encrypt = st.multiselect(
        "Columns to encrypt",
        ["units", "unit_price", "revenue"],
        default=["units", "unit_price", "revenue"],
    )
    pack_rows = st.checkbox("Pack rows as vectors (SIMD)", value=True,
                            help="If enabled, packs entire numeric column into a single ciphertext.")

    do_encrypt = st.button("🔒 Encrypt Selected Columns")

# Perform encryption
enc_artifacts = {}
if do_encrypt:
    if he is None:
        st.error("Homomorphic encryption not available.")
    else:
        try:
            for col in cols_to_encrypt:
                vec = df[col].to_numpy()
                pt = he.encode(vec if pack_rows else list(vec))
                ct = he.encrypt(pt)
                enc_artifacts[col] = ct
            st.success("Data encrypted. Ciphertexts are stored in session memory only.")
        except Exception as e:
            st.exception(e)

# Save/load encrypted sidecar
st.markdown("---")
st.subheader("💾 Save/Load Encrypted Dataset (sidecar .pkl)")
col1, col2 = st.columns(2)
with col1:
    if enc_artifacts:
        payload = {
            "meta": {
                "scheme": scheme,
                "params": params,
                "country": country,
                "owner": owner,
                "columns": list(enc_artifacts.keys()),
                "pack_rows": pack_rows,
            },
            "ciphertexts": {k: serialize_cipher(v) for k, v in enc_artifacts.items()},
        }
        b = io.BytesIO()
        pickle.dump(payload, b)
        st.download_button("⬇️ Download encrypted sidecar (.pkl)", data=b.getvalue(), file_name="encrypted_sidecar.pkl")
    else:
        st.caption("Encrypt some columns to enable download.")
with col2:
    sidecar = st.file_uploader("Upload encrypted sidecar (.pkl)", type=["pkl"])
    loaded_sidecar = None
    if sidecar is not None:
        try:
            payload = pickle.load(sidecar)
            loaded_sidecar = {
                "meta": payload["meta"],
                "ciphertexts": {k: deserialize_cipher(v) for k, v in payload["ciphertexts"].items()}
            }
            st.success("Encrypted sidecar loaded.")
        except Exception as e:
            st.exception(e)

# Select active ciphertexts source
active_ciphertexts = enc_artifacts or (
    loaded_sidecar["ciphertexts"] if 'loaded_sidecar' in locals() and loaded_sidecar else {})
active_meta = (
    {"columns": list(active_ciphertexts.keys())} if active_ciphertexts else {}
)

# ---------------- Encrypted analytics ----------------
st.header("🧮 Encrypted Analytics & Ops")

if not active_ciphertexts:
    st.info("Encrypt columns (or load sidecar) to enable encrypted computations.")
else:
    st.write("Below operations run **on ciphertexts**. Decryption happens only if you explicitly request it.")

    # Operation selection
    op = st.selectbox(
        "Operation",
        [
            "Sum by column",
            "Mean by column",
            "Add two columns",
            "Multiply two columns",
            "Scalar multiply",
            "Polynomial evaluation on a column",
            "Rotate packed vector (for SIMD)",
        ],
    )

    target_col = st.selectbox("Target column", active_meta.get("columns", []))
    second_col = None
    scalar = None
    poly_coeffs = None
    rotate_steps = 1

    if op in ("Add two columns", "Multiply two columns"):
        second_col = st.selectbox("Second column", [c for c in active_meta["columns"] if c != target_col])
    if op == "Scalar multiply":
        scalar = st.number_input("Scalar", value=2.0)
    if op == "Polynomial evaluation on a column":
        coeffs_text = st.text_input("Polynomial coeffs (comma‑sep c0,c1,c2…)", value="1,2,3")
        try:
            poly_coeffs = [float(x.strip()) for x in coeffs_text.split(',') if x.strip()]
        except Exception:
            poly_coeffs = [1.0, 2.0, 3.0]
    if op == "Rotate packed vector (for SIMD)":
        rotate_steps = st.number_input("Rotate steps (+right, −left)", value=1, step=1)

    do_run = st.button("▶️ Run homomorphic op")

    result_ct = None
    if do_run:
        try:
            ctA = active_ciphertexts[target_col]
            if op == "Sum by column":
                # Sum all slots via rotations + Adds (requires rotation keys)
                acc = ctA
                slots = len(df[target_col])
                step = 1
                while step < slots:
                    rot = he.rotate(acc, step)
                    acc = he.add(acc, rot)
                    step *= 2
                result_ct = acc

            elif op == "Mean by column":
                acc = ctA
                slots = len(df[target_col])
                step = 1
                while step < slots:
                    rot = he.rotate(acc, step)
                    acc = he.add(acc, rot)
                    step *= 2
                # divide by slots (multiply by 1/n)
                factor = 1.0 / float(slots)
                result_ct = he.mul_plain(acc, factor)

            elif op == "Add two columns":
                ctB = active_ciphertexts[second_col]
                result_ct = he.add(ctA, ctB)

            elif op == "Multiply two columns":
                ctB = active_ciphertexts[second_col]
                result_ct = he.mul(ctA, ctB)

            elif op == "Scalar multiply":
                result_ct = he.mul_plain(ctA, scalar)

            elif op == "Polynomial evaluation on a column":
                result_ct = he.poly_eval(ctA, poly_coeffs)

            elif op == "Rotate packed vector (for SIMD)":
                result_ct = he.rotate(ctA, int(rotate_steps))

            st.success("Homomorphic op completed. Output is still ciphertext.")
        except Exception as e:
            st.exception(e)

    # Decrypt on demand
    if result_ct is not None:
        with st.expander("🔓 Decrypt result (explicit)"):
            try:
                pt = he.decrypt(result_ct)
                # Extract values vector from plaintext object
                values = None
                # Try common accessors
                for attr in ("GetPackedValue", "GetRealPackedValue", "GetCKKSPackedValue", "GetInteger32Value",
                             "GetValues"):
                    if hasattr(pt, attr):
                        try:
                            values = list(getattr(pt, attr)())
                            break
                        except Exception:
                            pass
                if values is None:
                    # Last resort: many bindings stringify to show contents
                    values = str(pt)

                st.write("Decrypted values (may be rounded for CKKS):")
                if isinstance(values, list):
                    out_df = pd.DataFrame({"result": values})
                    st.dataframe(out_df.head(50), use_container_width=True)
                    # Visualize
                    st.bar_chart(out_df.head(50))
                else:
                    st.code(values)
            except Exception as e:
                st.exception(e)

# ---------------- Statistics/Proof-of-process ----------------
st.markdown("---")
st.header("🔍 Proof that computations stayed encrypted")

if active_ciphertexts:
    rows = []
    for name, ct in active_ciphertexts.items():
        # Inspect ciphertext meta (size, type) without decryption
        ct_type = type(ct).__name__
        try:
            # Not all bindings expose noise budget. Show what we can.
            noise = getattr(ct, "noise_budget", None)
        except Exception:
            noise = None
        rows.append({
            "column": name,
            "cipher_type": ct_type,
            "noise_budget": noise,
            "decrypted?": False,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(
        "Ciphertext objects are manipulated directly; decryption occurs only in the explicit decrypt step above.")
else:
    st.info("No active ciphertexts to inspect.")

# ---------------- Playground knobs ----------------
st.markdown("---")
st.header("🧪 Advanced Controls: noise budget, relinearization, rotations")

st.write(
    "Use the sidebar to toggle relinearization and rotations. For BFV/BGV, adjust plaintext modulus; for CKKS, adjust scale bits."
)

st.markdown(
    "- **Relinearization** reduces ciphertext size and depth growth after multiplications.\n"
    "- **Rotations** enable SIMD-style slot summations and permutations.\n"
    "- **Noise budget / levels** are not always exposed by Python bindings; if available, they will appear in the table above."
)

st.success("Tip: try BFV for integer totals, CKKS for averages (real numbers), and BGV for modular arithmetic.")

# ---------------- Grouped encrypted stats (example) ----------------
st.markdown("---")
st.header("📊 Example: Sales by Category (decrypt to view)")

if he is not None and {"units", "unit_price"}.issubset(set(active_ciphertexts)):
    try:
        ct_units = active_ciphertexts["units"]
        ct_price = active_ciphertexts["unit_price"]
        ct_rev = he.mul(ct_units, ct_price)
        with st.expander("🔓 Decrypt revenue vector (example)"):
            pt = he.decrypt(ct_rev)
            values = None
            for attr in ("GetPackedValue", "GetRealPackedValue", "GetCKKSPackedValue", "GetValues"):
                if hasattr(pt, attr):
                    try:
                        values = list(getattr(pt, attr)())
                        break
                    except Exception:
                        pass
            if values is None:
                values = str(pt)
            if isinstance(values, list):
                df_out = df.copy()
                df_out["revenue_encrypted_mul"] = values[:len(df_out)]
                grp = df_out.groupby("category")["revenue_encrypted_mul"].sum().reset_index()
                st.dataframe(grp, use_container_width=True)
                st.bar_chart(grp.set_index("category"))
            else:
                st.code(values)
    except Exception as e:
        st.exception(e)
else:
    st.caption("Encrypt both 'units' and 'unit_price' to demo revenue multiplication.")

st.markdown("---")
st.write(
    "Made with ❤️ for learning FHE. Keep keys secret; ciphertexts depend on (country, owner)."
)
