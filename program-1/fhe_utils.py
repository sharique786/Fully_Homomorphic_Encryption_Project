# fhe_utils.py
import base64
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple

# Try to import openfhe with tolerant messages
try:
    import openfhe as of  # typical alias; some installs expose `openfhe` module
except Exception as e:
    # If import fails, raise a helpful message
    raise ImportError(
        "Could not import OpenFHE Python bindings. "
        "Install openfhe (pip install openfhe) or build openfhe/openfhe-python from source. "
        f"Original error: {e}"
    )

# Helper wrappers
def make_context(scheme: str, poly_modulus_degree: int = 8192, plaintext_modulus: int = 65537, coeff_mod_bit_sizes: List[int] = None):
    """
    Create a crypto context for a chosen scheme.
    scheme: 'BFV' | 'BGV' | 'CKKS'
    poly_modulus_degree: N
    plaintext_modulus: used for BFV/BGV
    coeff_mod_bit_sizes: list of bit sizes for CKKS modulus chain (e.g., [60, 40, 40, 60])
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60] if scheme.upper() == "CKKS" else None

    if scheme.upper() == "CKKS":
        # CKKS: use GenCryptoContextCKKS (OpenFHE naming may vary slightly)
        cc = of.GenCryptoContextCKKS(poly_modulus_degree=poly_modulus_degree,
                                     coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                                     security_level=of.SecurityLevel.HEStd_128_classic)
    elif scheme.upper() == "BFV":
        cc = of.GenCryptoContextBFV(poly_modulus_degree=poly_modulus_degree,
                                    plaintext_modulus=plaintext_modulus,
                                    security_level=of.SecurityLevel.HEStd_128_classic)
    elif scheme.upper() == "BGV":
        cc = of.GenCryptoContextBGV(poly_modulus_degree=poly_modulus_degree,
                                    plaintext_modulus=plaintext_modulus,
                                    security_level=of.SecurityLevel.HEStd_128_classic)
    else:
        raise ValueError("Unsupported scheme: choose BFV, BGV or CKKS")

    cc.SetBootstrap(0) if hasattr(cc, "SetBootstrap") else None
    cc.Enable(of.PKE)
    cc.Enable(of.KEYSWITCH)
    cc.Enable(of.LEVELEDSHE)  # enable essential features; some wrappers use different enums
    return cc

def keygen(cc):
    kp = {}
    kp['keypair'] = cc.KeyGen()
    # evaluation keys for multiplication / rotation
    try:
        cc.EvalMultKeyGen(kp['keypair'].secretKey)
    except Exception:
        # some wrappers return keys differently; try alternative
        pass
    try:
        cc.EvalAtIndexKeyGen(kp['keypair'].secretKey, list(range(1, 8)))  # a few rotations
    except Exception:
        pass
    return kp

def encode_plain(cc, values: List[float], scheme: str):
    if scheme.upper() == "CKKS":
        return cc.MakeCKKSPackedPlaintext(values)
    else:
        # BFV/BGV
        return cc.MakePackedPlaintext([int(v) for v in values])

def encrypt_vector(cc, keypair, values: List[float], scheme: str):
    pt = encode_plain(cc, values, scheme)
    ct = cc.Encrypt(keypair['keypair'].publicKey, pt)
    return ct

def decrypt_vector(cc, keypair, ct, scheme: str):
    pt = cc.Decrypt(keypair['keypair'].secretKey, ct)
    # decode packed plaintext to python list
    if scheme.upper() == "CKKS":
        return pt.GetValues()
    else:
        return list(map(int, pt.GetValues()))

def eval_add(cc, ct1, ct2):
    return cc.EvalAdd(ct1, ct2)

def eval_sub(cc, ct1, ct2):
    return cc.EvalSub(ct1, ct2)

def eval_mul(cc, ct1, ct2):
    return cc.EvalMult(ct1, ct2)

def eval_scalar_mul(cc, ct, scalar, scheme):
    # multiply plaintext scalar
    if scheme.upper() == "CKKS":
        pt = cc.MakeCKKSPackedPlaintext([scalar])
    else:
        pt = cc.MakePackedPlaintext([int(scalar)])
    return cc.EvalMult(ct, pt)

def serialize_ciphertext(ct) -> str:
    """
    Serialize ciphertext to base64 so it can be displayed in UI.
    """
    data = pickle.dumps(ct)
    return base64.b64encode(data).decode()

def deserialize_ciphertext(cc, b64: str):
    data = base64.b64decode(b64.encode())
    return pickle.loads(data)

def polynomial_eval(cc, keypair, ct, coeffs: List[float], scheme: str):
    """
    Evaluate polynomial with given coefficients (lowest-first) on ciphertext ct.
    Simple Horner's method using EvalMult and EvalAdd.
    NOTE: this is a naive implementation (not optimized).
    """
    # result = coeffs[-1]
    # iterate from highest to lowest
    # create ciphertext for constant coeffs by encrypting a plaintext vector with that scalar
    current = None
    for a in reversed(coeffs):
        if current is None:
            # current = a (as ciphertext)
            pt = encode_plain(cc, [a], scheme)
            current = cc.Encrypt(keypair['keypair'].publicKey, pt)
        else:
            # current = current * x + a where x is ct
            current = eval_mul(cc, current, ct)
            pt = encode_plain(cc, [a], scheme)
            a_ct = cc.Encrypt(keypair['keypair'].publicKey, pt)
            current = eval_add(cc, current, a_ct)
    return current

# convenience: pack a pandas column to vector-of-values for CKKS; for BFV/BGV show integer mapping
def pack_column(col_values, scheme: str):
    arr = list(col_values)
    if scheme.upper() != "CKKS":
        return [int(round(x)) if x is not None and str(x).strip() != "" else 0 for x in arr]
    return [float(x) if x is not None and str(x).strip() != "" else 0.0 for x in arr]
