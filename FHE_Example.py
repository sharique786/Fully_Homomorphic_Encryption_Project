"""
TenSEAL Fully Homomorphic Encryption Example

TenSEAL is a Python library built on Microsoft SEAL that provides
easy-to-use FHE operations with excellent Python integration.

Installation:
pip install tenseal

This example demonstrates:
1. BFV scheme for integer operations
2. CKKS scheme for floating-point operations
3. Homomorphic operations on encrypted data
4. Practical use cases
"""

try:
    import tenseal as ts
    import numpy as np
except ImportError:
    print("Error: TenSEAL not found. Install with: pip install tenseal")
    exit(1)


class TenSEALFHE:
    def __init__(self, scheme='BFV'):
        """Initialize TenSEAL context with specified scheme"""
        self.scheme = scheme.upper()
        self.setup_context()

    def setup_context(self):
        """Setup encryption context based on scheme"""
        if self.scheme == 'BFV':
            # BFV scheme for integer arithmetic
            self.context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=4096,
                plain_modulus=1032193  # Prime number for BFV
            )
            print("Initialized BFV context for integer operations")

        elif self.scheme == 'CKKS':
            # CKKS scheme for floating-point arithmetic
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = 2 ** 40
            print("Initialized CKKS context for floating-point operations")

        # Generate galois keys for rotations and other operations
        self.context.generate_galois_keys()

        # Make context public (removes secret key from context)
        # Useful for sending context to untrusted parties
        self.public_context = self.context.copy()
        self.public_context.make_context_public()


def demonstrate_bfv_operations():
    """Demonstrate BFV scheme with integer operations"""
    print("=" * 60)
    print("BFV SCHEME - INTEGER OPERATIONS")
    print("=" * 60)

    # Initialize FHE with BFV scheme
    fhe = TenSEALFHE('BFV')

    # Test data
    x = 15
    y = 25
    vector_data = [1, 2, 3, 4, 5]

    print(f"\nOriginal values:")
    print(f"x = {x}, y = {y}")
    print(f"vector = {vector_data}")

    # Encrypt single values
    encrypted_x = ts.bfv_vector(fhe.context, [x])
    encrypted_y = ts.bfv_vector(fhe.context, [y])

    # Encrypt vector
    encrypted_vector = ts.bfv_vector(fhe.context, vector_data)

    print(f"\nHomomorphic Operations:")

    # Addition
    encrypted_sum = encrypted_x + encrypted_y
    print(f"x + y = {encrypted_sum.decrypt()[0]} (expected: {x + y})")

    # Multiplication
    encrypted_product = encrypted_x * encrypted_y
    print(f"x * y = {encrypted_product.decrypt()[0]} (expected: {x * y})")

    # Scalar operations
    scalar = 3
    encrypted_scalar_mult = encrypted_x * scalar
    print(f"x * {scalar} = {encrypted_scalar_mult.decrypt()[0]} (expected: {x * scalar})")

    # Vector operations
    encrypted_vector_sum = encrypted_vector + encrypted_vector
    decrypted_vector_sum = encrypted_vector_sum.decrypt()
    expected_vector_sum = [2 * v for v in vector_data]
    print(f"vector + vector = {decrypted_vector_sum} (expected: {expected_vector_sum})")

    # Element-wise multiplication with scalar
    encrypted_vector_scaled = encrypted_vector * 2
    decrypted_vector_scaled = encrypted_vector_scaled.decrypt()
    expected_vector_scaled = [2 * v for v in vector_data]
    print(f"vector * 2 = {decrypted_vector_scaled} (expected: {expected_vector_scaled})")


def demonstrate_ckks_operations():
    """Demonstrate CKKS scheme with floating-point operations"""
    print(f"\n" + "=" * 60)
    print("CKKS SCHEME - FLOATING-POINT OPERATIONS")
    print("=" * 60)

    # Initialize FHE with CKKS scheme
    fhe = TenSEALFHE('CKKS')

    # Test data (floating-point)
    x = 3.14
    y = 2.71
    vector_data = [1.1, 2.2, 3.3, 4.4, 5.5]

    print(f"\nOriginal values:")
    print(f"x = {x}, y = {y}")
    print(f"vector = {vector_data}")

    # Encrypt values
    encrypted_x = ts.ckks_vector(fhe.context, [x])
    encrypted_y = ts.ckks_vector(fhe.context, [y])
    encrypted_vector = ts.ckks_vector(fhe.context, vector_data)

    print(f"\nHomomorphic Operations:")

    # Addition
    encrypted_sum = encrypted_x + encrypted_y
    result_sum = encrypted_sum.decrypt()[0]
    print(f"x + y = {result_sum:.4f} (expected: {x + y:.4f})")

    # Multiplication
    encrypted_product = encrypted_x * encrypted_y
    result_product = encrypted_product.decrypt()[0]
    print(f"x * y = {result_product:.4f} (expected: {x * y:.4f})")

    # Power operations (approximate)
    encrypted_squared = encrypted_x * encrypted_x
    result_squared = encrypted_squared.decrypt()[0]
    print(f"x² = {result_squared:.4f} (expected: {x ** 2:.4f})")

    # Vector operations
    encrypted_vector_sum = encrypted_vector + encrypted_vector
    decrypted_vector_sum = encrypted_vector_sum.decrypt()
    expected_vector_sum = [2 * v for v in vector_data]
    print(f"vector + vector = {[round(v, 2) for v in decrypted_vector_sum[:5]]}")
    print(f"expected: {[round(v, 2) for v in expected_vector_sum]}")


def demonstrate_polynomial_evaluation():
    """Demonstrate polynomial evaluation: f(x) = x² + 2x + 1"""
    print(f"\n" + "=" * 60)
    print("POLYNOMIAL EVALUATION EXAMPLE")
    print("Evaluating f(x) = x² + 2x + 1 homomorphically")
    print("=" * 60)

    fhe = TenSEALFHE('CKKS')

    # Input value
    x = 5.0
    print(f"Input: x = {x}")
    print(f"Expected result: f({x}) = {x}² + 2×{x} + 1 = {x ** 2 + 2 * x + 1}")

    # Encrypt input
    encrypted_x = ts.ckks_vector(fhe.context, [x])

    # Compute x²
    encrypted_x_squared = encrypted_x * encrypted_x

    # Compute 2x
    encrypted_2x = encrypted_x * 2

    # Compute x² + 2x
    encrypted_partial = encrypted_x_squared + encrypted_2x

    # Add constant 1
    encrypted_result = encrypted_partial + 1

    # Decrypt result
    result = encrypted_result.decrypt()[0]
    print(f"Homomorphic result: f({x}) = {result:.4f}")
    print(f"Verification: {abs(result - (x ** 2 + 2 * x + 1)) < 0.01}")


def demonstrate_privacy_preserving_average():
    """Demonstrate privacy-preserving computation of average"""
    print(f"\n" + "=" * 60)
    print("PRIVACY-PRESERVING AVERAGE CALCULATION")
    print("Multiple parties contribute encrypted data")
    print("=" * 60)

    fhe = TenSEALFHE('CKKS')

    # Simulate multiple parties with private data
    party1_data = [10.5, 20.3, 15.7]
    party2_data = [25.1, 18.9, 22.4]
    party3_data = [12.8, 30.2, 19.6]

    all_data = party1_data + party2_data + party3_data
    expected_average = sum(all_data) / len(all_data)

    print(f"Party 1 data: {party1_data}")
    print(f"Party 2 data: {party2_data}")
    print(f"Party 3 data: {party3_data}")
    print(f"Expected average: {expected_average:.4f}")

    # Each party encrypts their data
    encrypted_party1 = ts.ckks_vector(fhe.context, party1_data)
    encrypted_party2 = ts.ckks_vector(fhe.context, party2_data)
    encrypted_party3 = ts.ckks_vector(fhe.context, party3_data)

    # Combine encrypted data
    encrypted_combined = encrypted_party1 + encrypted_party2 + encrypted_party3

    # Compute average homomorphically
    total_count = len(all_data)
    encrypted_average = encrypted_combined * (1.0 / total_count)

    # Extract the sum by decrypting (in practice, only authorized party can decrypt)
    decrypted_values = encrypted_average.decrypt()
    computed_average = sum(decrypted_values[:total_count]) / total_count

    print(f"Homomorphic average: {computed_average:.4f}")
    print(f"Accuracy: {abs(computed_average - expected_average) < 0.01}")


def demonstrate_context_serialization():
    """Demonstrate context and ciphertext serialization"""
    print(f"\n" + "=" * 60)
    print("CONTEXT & CIPHERTEXT SERIALIZATION")
    print("For distributed computation scenarios")
    print("=" * 60)

    fhe = TenSEALFHE('BFV')

    # Original data
    data = [42, 84, 126]
    print(f"Original data: {data}")

    # Encrypt data
    encrypted_data = ts.bfv_vector(fhe.context, data)

    # Serialize context (to send to computing party)
    serialized_context = fhe.public_context.serialize()
    print(f"Serialized context size: {len(serialized_context)} bytes")

    # Serialize encrypted data
    serialized_ciphertext = encrypted_data.serialize()
    print(f"Serialized ciphertext size: {len(serialized_ciphertext)} bytes")

    # Simulate sending to another party...
    # Deserialize context
    received_context = ts.context_from(serialized_context)

    # Deserialize ciphertext
    received_ciphertext = ts.bfv_vector_from(received_context, serialized_ciphertext)

    # Perform computation on received data
    computed_result = received_ciphertext * 2

    # Serialize result to send back
    serialized_result = computed_result.serialize()

    # Original party deserializes and decrypts result
    final_result = ts.bfv_vector_from(fhe.context, serialized_result)
    decrypted_result = final_result.decrypt()

    print(f"Result after remote computation: {decrypted_result}")
    print(f"Expected: {[x * 2 for x in data]}")


if __name__ == "__main__":
    try:
        print("TenSEAL Fully Homomorphic Encryption Demonstration")
        print("Using Microsoft SEAL backend with Python interface")

        # Run all demonstrations
        demonstrate_bfv_operations()
        demonstrate_ckks_operations()
        demonstrate_polynomial_evaluation()
        demonstrate_privacy_preserving_average()
        demonstrate_context_serialization()

        print(f"\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("TenSEAL provides a robust FHE solution for Python")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure TenSEAL is installed: pip install tenseal")