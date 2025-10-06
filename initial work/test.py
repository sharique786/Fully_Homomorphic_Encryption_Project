import tenseal as ts

# User selects scheme
scheme = "CKKS"  # or "BFV", "BGV"

if scheme == "CKKS":
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

elif scheme == "BFV":
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        plain_modulus=1032193
    )
    context.generate_galois_keys()

# Keep track of the scheme separately
active_scheme = scheme

print(f"Context created with scheme: {2.0 ** 40}")
