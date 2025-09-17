import numpy as np

# Try imports dynamically
try:
    from Pyfhel import Pyfhel
except ImportError:
    Pyfhel = None

try:
    import tenseal as ts
except ImportError:
    ts = None

# OpenFHE stub (if installed)
try:
    import openfhe
except ImportError:
    openfhe = None


class FHEWrapper:
    def __init__(self, backend, scheme, poly_modulus_degree, noise, relin):
        self.backend = backend
        self.scheme = scheme
        self.poly_modulus_degree = poly_modulus_degree
        self.noise = noise
        self.relin = relin
        self.context = None
        self.he = None

    def initialize(self):
        if self.backend == "pyfhel" and Pyfhel:
            self.he = Pyfhel()
            if self.scheme == "CKKS":
                self.he.contextGen(
                    scheme="CKKS",
                    n=self.poly_modulus_degree,
                    scale=2 ** 30,
                    qi_sizes=[60, 30, 30, 30, 60],  # default param sizes
                )
            elif self.scheme == "BFV":
                self.he.contextGen(
                    scheme="BFV",
                    n=self.poly_modulus_degree,
                    t_bits=20,
                )
            elif self.scheme == "BGV":
                self.he.contextGen(
                    scheme="BGV",
                    n=self.poly_modulus_degree,
                    t_bits=20,
                )
            else:
                raise ValueError(f"Unsupported scheme {self.scheme} for Pyfhel")

            self.he.keyGen()
            if self.relin:
                self.he.relinKeyGen()
            return True

        elif self.backend == "tenseal" and ts:
            if self.scheme == "CKKS":
                init_tenseal_context()
                # ctx = ts.context(
                #     scheme=ts.SCHEME_TYPE.CKKS,
                #     poly_modulus_degree=self.poly_modulus_degree,
                #     coeff_mod_bit_sizes=[60, 40, 40, 60]
                # )
                # ctx.global_scale = 2 ** 40  # ✅ Set scale for CKKS
                # ctx.generate_galois_keys()
                #
                # self.context.generate_galois_keys()
                # if self.relin:
                #     self.context.generate_relin_keys()
                # return True
            elif self.scheme == "BFV":
                self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=self.poly_modulus_degree,
                    plain_modulus=1032193,
                )
                return True
            else:
                raise ValueError(f"Unsupported scheme {self.scheme} for TenSEAL")

        elif self.backend == "openfhe" and openfhe:
            # NOTE: adjust depending on available OpenFHE Python bindings
            # For now, we’ll just return False if not fully integrated
            return False

        else:
            return False

    def encrypt(self, value):
        if self.backend == "pyfhel" and self.he:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if self.scheme == "CKKS":
                return self.he.encryptFrac(float(value))
            else:  # BFV/BGV → integers only
                return self.he.encryptInt(int(value))
        elif self.backend == "tenseal" and self.context:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if self.scheme == "CKKS":
                return ts.ckks_vector(self.context, [float(value)])
            else:  # BFV
                return ts.bfv_vector(self.context, [int(value)])
        else:
            raise RuntimeError("No encryption context initialized")

    def decrypt(self, ctxt):
        if ctxt is None:
            return None
        if self.backend == "pyfhel" and self.he:
            try:
                if self.scheme == "CKKS":
                    return self.he.decryptFrac(ctxt)
                else:
                    return self.he.decryptInt(ctxt)
            except Exception as e:
                return f"<decrypt error: {e}>"
        elif self.backend == "tenseal" and self.context:
            try:
                return ctxt.decrypt()[0]
            except Exception as e:
                return f"<decrypt error: {e}>"
        else:
            return "<no backend>"

    def add(self, ctxt1, ctxt2):
        return ctxt1 + ctxt2

    def mul(self, ctxt1, ctxt2):
        return ctxt1 * ctxt2


def get_fhe_context(backend, scheme, poly_modulus_degree, noise, relin):
    fhe = FHEWrapper(backend, scheme, poly_modulus_degree, noise, relin)
    ok = fhe.initialize()
    return fhe if ok else None


import tenseal as ts

def init_tenseal_context(scheme="CKKS", poly_modulus_degree=8192, coeff_mod_bit_sizes=None):
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]

    try:
        if scheme.upper() == "CKKS":
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            ctx.global_scale = 2**40
            ctx.generate_galois_keys()
            return ctx

        elif scheme.upper() == "BFV":
            ctx = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=1032193,
            )
            return ctx

        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize TenSEAL context: {e}")

