# app_extended.py
import streamlit as st
import pandas as pd
import numpy as np
import io, base64, json
import plotly.express as px

st.set_page_config(layout='wide', page_title='OpenFHE Streamlit Extended')

# Try to import openfhe
try:
    from openfhe import *
    HAVE_OPENFHE = True
except Exception as e:
    HAVE_OPENFHE = False
    IMPORT_ERR = str(e)

st.title('OpenFHE — Extended Streamlit Demo')

if not HAVE_OPENFHE:
    st.error('OpenFHE not available in the environment. Install the openfhe wheel or build from source.')
    st.write('Import error:')
    st.code(IMPORT_ERR)
    st.stop()

# --- utilities for serialization ---
def serialize_key(obj):
    buf = io.BytesIO()
    obj.serialize(buf)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def deserialize_public_key(cc, b64):
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    pk = cc.PublicKey()
    pk.deserialize(buf)
    return pk

def deserialize_secret_key(cc, b64):
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    sk = cc.SecretKey()
    sk.deserialize(buf)
    return sk

# --- CKKS presets ---
CKKS_PRESETS = {
    'small': {'poly_modulus_degree': 4096, 'multiplicative_depth': 3, 'scale_exp': 30},
    'medium': {'poly_modulus_degree': 8192, 'multiplicative_depth': 5, 'scale_exp': 40},
    'large': {'poly_modulus_degree': 16384, 'multiplicative_depth': 8, 'scale_exp': 50}
}

# --- UI: mode selection ---
mode = st.sidebar.radio('Mode', ['Server (secret key present)', 'Public-only (multi-user)'])

# --- Upload CSV ---
uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.DataFrame({'id':list(range(1,11)),'balance':[100.0*i for i in range(1,11)],'country':['IN','US','GB','IN','CA','US','IN','AU','FR','DE']})

st.subheader('Dataset')
st.dataframe(df.head(50))

# --- param selection ---
scheme = st.sidebar.selectbox('Scheme', ['CKKS','BFV','BGV'])
if scheme == 'CKKS':
    preset = st.sidebar.selectbox('CKKS Preset', list(CKKS_PRESETS.keys()), index=1)
    preset_vals = CKKS_PRESETS[preset]
    poly_modulus_degree = st.sidebar.selectbox('Poly modulus degree', [4096,8192,16384], index=[4096,8192,16384].index(preset_vals['poly_modulus_degree']))
    multiplicative_depth = st.sidebar.slider('Multiplicative depth', 1, 12, value=preset_vals['multiplicative_depth'])
    scale_exp = st.sidebar.number_input('Scale exponent (base-2)', value=preset_vals['scale_exp'])
else:
    poly_modulus_degree = st.sidebar.selectbox('Poly modulus degree', [2048,4096,8192,16384], index=2)
    multiplicative_depth = st.sidebar.slider('Multiplicative depth', 1, 10, 2)
    plaintext_modulus = st.sidebar.number_input('Plaintext modulus (BFV/BGV)', value=65537)

# create or load context
if 'cc' not in st.session_state:
    # Basic CC creation wrapper (depends on openfhe-python API)
    if scheme in ('BFV','BGV'):
        params = CCParamsBFVRNS() if scheme=='BFV' else CCParamsBGVRNS()
        params.SetPlaintextModulus(int(plaintext_modulus))
        params.SetMultiplicativeDepth(int(multiplicative_depth))
        try:
            params.SetRingDimension(int(poly_modulus_degree))
        except Exception:
            pass
    else:
        params = CCParamsCKKS()
        params.SetMultiplicativeDepth(int(multiplicative_depth))
        try:
            params.SetRingDimension(int(poly_modulus_degree))
        except Exception:
            pass
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    st.session_state['cc'] = cc
else:
    cc = st.session_state['cc']

# key management
if mode == 'Server (secret key present)':
    if 'kp' not in st.session_state:
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        st.session_state['kp'] = kp
    kp = st.session_state['kp']
    # allow download of serialized keys
    if st.sidebar.button('Download keys'):
        pk_b64 = serialize_key(kp.publicKey)
        sk_b64 = serialize_key(kp.secretKey)
        payload = json.dumps({'public_key': pk_b64, 'secret_key': sk_b64})
        b64 = base64.b64encode(payload.encode()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="openfhe_keys.json">Download keys (public+secret)</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    # Public-only: user may upload a public key (created elsewhere)
    uploaded_pk = st.sidebar.file_uploader('Upload public key JSON', type=['json'])
    if uploaded_pk:
        raw = json.load(uploaded_pk)
        if 'public_key' in raw:
            try:
                pk = deserialize_public_key(cc, raw['public_key'])
                st.session_state['external_pk'] = pk
                st.success('Public key loaded')
            except Exception as e:
                st.error(f'Failed to load public key: {e}')

# encrypt selected columns
cols = df.columns.tolist()
encrypt_cols = st.multiselect('Columns to encrypt', cols, default=[c for c in cols if df[c].dtype.kind in 'if'])

def encrypt_columns_use_pk(public_key):
    enc_map = {}
    for col in encrypt_cols:
        data = df[col].fillna(0).tolist()
        if scheme in ('BFV','BGV'):
            pt = cc.MakePackedPlaintext([int(x) for x in data])
        else:
            pt = cc.MakeCKKSPackedPlaintext([float(x) for x in data])
        ct = cc.Encrypt(public_key, pt)
        buf = io.BytesIO(); ct.serialize(buf)
        enc_map[col] = base64.b64encode(buf.getvalue()).decode()
    return enc_map

if st.button('Encrypt now'):
    if mode == 'Server (secret key present)':
        enc_map = encrypt_columns_use_pk(st.session_state['kp'].publicKey)
        st.session_state['enc_map'] = enc_map
        st.success('Encrypted with server keypair')
    else:
        if 'external_pk' not in st.session_state:
            st.error('Upload a public key first for public-only mode')
        else:
            enc_map = encrypt_columns_use_pk(st.session_state['external_pk'])
            st.session_state['enc_map'] = enc_map
            st.success('Encrypted with provided public key')

# show encrypted placeholders and allow download
if 'enc_map' in st.session_state:
    st.subheader('Encrypted columns (placeholders shown)')
    df_enc = df.copy()
    for col in encrypt_cols:
        df_enc[col] = [f'<ct:{col}:{i}>' for i in range(len(df))]
    st.dataframe(df_enc.head(50))

    if st.button('Download ciphertexts JSON'):
        j = json.dumps(st.session_state['enc_map'])
        b64 = base64.b64encode(j.encode()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="ciphertexts.json">Download ciphertexts</a>'
        st.markdown(href, unsafe_allow_html=True)

# operations (similar to earlier app)
st.header('Operations')
if 'enc_map' in st.session_state:
    cols_available = list(st.session_state['enc_map'].keys())
    op = st.selectbox('Operation', ['Add', 'Multiply', 'Polynomial'])
    a = st.selectbox('A', cols_available)
    b = st.selectbox('B', cols_available)

    if op == 'Polynomial':
        coef2 = st.number_input('a (x^2)', value=1.0)
        coef1 = st.number_input('b (x)', value=0.0)
        coef0 = st.number_input('c', value=0.0)

    if st.button('Run op'):
        enc_map = st.session_state['enc_map']
        # helper to deserialize
        def ct_from_b64(b64):
            raw = base64.b64decode(b64)
            buf = io.BytesIO(raw)
            ct = cc.Ciphertext(); ct.deserialize(buf); return ct

        if op == 'Add':
            ct_res = cc.EvalAdd(ct_from_b64(enc_map[a]), ct_from_b64(enc_map[b]))
        elif op == 'Multiply':
            ct_res = cc.EvalMult(ct_from_b64(enc_map[a]), ct_from_b64(enc_map[b]))
        else:
            # polynomial: decrypt to evaluate and re-encrypt to show example
            # (production: you would use EvalMult/EvalAdd chain instead)
            if mode == 'Public-only':
                st.warning('Polynomial example requires secret key on server to evaluate/encode constants; falling back to client-side evaluation of decrypted plaintext is not possible without secret key')
                ct_res = None
            else:
                vals = cc.Decrypt(st.session_state['kp'].secretKey, ct_from_b64(enc_map[a])).GetCKKSValue() if scheme=='CKKS' else cc.Decrypt(st.session_state['kp'].secretKey, ct_from_b64(enc_map[a])).GetPackedValue()
                poly_vals = [coef2*(v**2) + coef1*v + coef0 for v in vals]
                if scheme == 'CKKS':
                    pt = cc.MakeCKKSPackedPlaintext(poly_vals)
                else:
                    pt = cc.MakePackedPlaintext([int(round(x)) for x in poly_vals])
                ct_res = cc.Encrypt(st.session_state['kp'].publicKey, pt)

        if ct_res is not None:
            buf = io.BytesIO(); ct_res.serialize(buf)
            st.session_state['last_result'] = base64.b64encode(buf.getvalue()).decode()
            st.success('Operation done; result ciphertext stored as last_result')

            # If secret key present, show decrypted preview
            if mode == 'Server (secret key present)':
                try:
                    raw = base64.b64decode(st.session_state['last_result']); buf = io.BytesIO(raw); ct = cc.Ciphertext(); ct.deserialize(buf)
                    pt = cc.Decrypt(st.session_state['kp'].secretKey, ct)
                    try:
                        vals = pt.GetCKKSValue()
                    except Exception:
                        vals = pt.GetPackedValue()
                    df_out = pd.DataFrame({'value': vals})
                    st.dataframe(df_out.head(50))
                    st.write(df_out.describe())
                    fig = px.line(df_out.reset_index(), x='index', y='value')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f'Decrypt/display failed: {e}')

# Noise monitor (UI estimation)
st.sidebar.header('Noise & health')
if 'last_result' in st.session_state and mode=='Server (secret key present)':
    try:
        raw = base64.b64decode(st.session_state['last_result']); buf = io.BytesIO(raw); ct = cc.Ciphertext(); ct.deserialize(buf)
        # Some OpenFHE APIs expose GetNoiseBudget or similar; try safe call
        try:
            nb = cc.GetNoiseBudget(ct)
            st.sidebar.metric('Noise budget (bits)', nb)
        except Exception:
            st.sidebar.info('Noise budget API not available in this wrapper build; consider using OpenFHE C++ examples or check wrapper version')
    except Exception:
        pass

st.sidebar.markdown('---')
st.sidebar.markdown('CKKS presets: small/medium/large tuned for short demos. For production, run lattice estimator and pick secure params.')

# End of app_extended.py