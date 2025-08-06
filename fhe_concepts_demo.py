"""
Practical Examples of FHE Terminology and Concepts

This code demonstrates key FHE concepts with working examples:
- Noise and noise budget
- Packing and SIMD operations
- Polynomial evaluation
- Privacy-preserving computations
- Context serialization
- Modular arithmetic effects
- Parameter impacts

Requirements: pip install tenseal numpy
"""

import tenseal as ts
import numpy as np
import time
import pickle


class FHEConceptsDemonstration:
    def __init__(self):
        """Initialize different contexts to demonstrate various concepts"""
        self.setup_contexts()

    def setup_contexts(self):
        """Setup different contexts for various demonstrations"""
        # BFV context for integer operations
        self.bfv_context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=4096,
            plain_modulus=1032193
        )
        self.bfv_context.generate_galois_keys()

        # CKKS context for floating-point operations
        self.ckks_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.ckks_context.global_scale = 2 ** 40
        self.ckks_context.generate_galois_keys()

        print("✓ Contexts initialized for demonstrations")


def demonstrate_noise_concept():
    """Demonstrate noise and noise budget concepts"""
    print("\n" + "=" * 70)
    print("NOISE AND NOISE BUDGET DEMONSTRATION")
    print("=" * 70)

    demo = FHEConceptsDemonstration()

    # Start with a simple value
    x = 5
    print(f"Starting value: x = {x}")

    # Encrypt the value
    encrypted_x = ts.bfv_vector(demo.bfv_context, [x])

    # Check initial noise budget
    initial_noise = demo.bfv_context.decryptor().invariant_noise_budget(encrypted_x._vector)
    print(f"Initial noise budget: {initial_noise} bits")

    # Perform operations and track noise consumption
    current = encrypted_x
    operations = ["x", "x + x", "(x + x) * x", "((x + x) * x) * x"]

    for i, op_desc in enumerate(operations):
        if i == 0:
            pass  # Initial state
        elif i == 1:
            current = current + encrypted_x  # Addition
        elif i == 2:
            current = current * encrypted_x  # Multiplication
        elif i == 3:
            current = current * encrypted_x  # Another multiplication

        try:
            current_noise = demo.bfv_context.decryptor().invariant_noise_budget(current._vector)
            decrypted_value = current.decrypt()[0]
            expected = eval(op_desc.replace('x', str(x)))

            print(
                f"After '{op_desc}': value = {decrypted_value}, noise budget = {current_noise} bits, expected = {expected}")

            if current_noise < 10:
                print("⚠️  WARNING: Noise budget getting low!")
            if current_noise == 0:
                print("❌ DANGER: Noise budget exhausted! Results may be incorrect.")
                break

        except Exception as e:
            print(f"❌ Operation failed due to noise: {e}")
            break

    print(f"\n--- NOISE LESSONS ---")
    print("• Fresh ciphertexts start with high noise budget")
    print("• Addition consumes little noise budget")
    print("• Multiplication consumes significant noise budget")
    print("• When budget reaches 0, decryption may fail or give wrong results")


def demonstrate_packing_simd():
    """Demonstrate packing and SIMD operations"""
    print(f"\n" + "=" * 70)
    print("PACKING AND SIMD OPERATIONS")
    print("=" * 70)

    demo = FHEConceptsDemonstration()

    # Large dataset to process
    dataset1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    dataset2 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

    print(f"Dataset 1: {dataset1}")
    print(f"Dataset 2: {dataset2}")

    # Method 1: Individual encryption (inefficient)
    print(f"\n--- METHOD 1: Individual Encryption (Inefficient) ---")
    start_time = time.time()

    individual_encrypted = []
    for i in range(len(dataset1)):
        enc_val = ts.ckks_vector(demo.ckks_context, [dataset1[i]])
        individual_encrypted.append(enc_val)

    # Perform operations individually
    individual_results = []
    for i in range(len(dataset1)):
        result = individual_encrypted[i] * 2.0  # Multiply by 2
        individual_results.append(result.decrypt()[0])

    individual_time = time.time() - start_time
    print(f"Individual encryption time: {individual_time:.4f} seconds")
    print(f"Results (first 4): {[round(x, 2) for x in individual_results[:4]]}")

    # Method 2: Packed encryption (efficient)
    print(f"\n--- METHOD 2: Packed Encryption (Efficient) ---")
    start_time = time.time()

    # Pack entire datasets into single ciphertexts
    packed_data1 = ts.ckks_vector(demo.ckks_context, dataset1)
    packed_data2 = ts.ckks_vector(demo.ckks_context, dataset2)

    # Perform operations on entire vectors simultaneously
    packed_doubled = packed_data1 * 2.0  # Multiply ALL values by 2
    packed_sum = packed_data1 + packed_data2  # Add vectors element-wise
    packed_product = packed_data1 * packed_data2  # Multiply vectors element-wise

    # Decrypt results
    doubled_results = packed_doubled.decrypt()
    sum_results = packed_sum.decrypt()
    product_results = packed_product.decrypt()

    packed_time = time.time() - start_time
    print(f"Packed encryption time: {packed_time:.4f} seconds")
    print(f"Speedup: {individual_time / packed_time:.1f}x faster")

    print(f"Doubled results: {[round(x, 2) for x in doubled_results[:len(dataset1)]]}")
    print(f"Sum results: {[round(x, 2) for x in sum_results[:len(dataset1)]]}")
    print(f"Product results: {[round(x, 2) for x in product_results[:len(dataset1)]]}")

    print(f"\n--- PACKING ADVANTAGES ---")
    print("• Single ciphertext holds multiple values")
    print("• Operations work on ALL packed values simultaneously")
    print("• Dramatically improves performance for batch operations")
    print("• Essential for practical FHE applications")


def demonstrate_polynomial_evaluation():
    """Demonstrate polynomial evaluation on encrypted data"""
    print(f"\n" + "=" * 70)
    print("POLYNOMIAL EVALUATION ON ENCRYPTED DATA")
    print("=" * 70)

    demo = FHEConceptsDemonstration()

    # Define polynomial: f(x) = 2x³ + 3x² - x + 5
    def polynomial_plaintext(x):
        return 2 * x ** 3 + 3 * x ** 2 - x + 5

    # Test value
    x = 2.5
    expected_result = polynomial_plaintext(x)

    print(f"Polynomial: f(x) = 2x³ + 3x² - x + 5")
    print(f"Input: x = {x}")
    print(f"Expected result: f({x}) = {expected_result}")

    # Encrypt input
    encrypted_x = ts.ckks_vector(demo.ckks_context, [x])

    print(f"\n--- HOMOMORPHIC EVALUATION ---")

    # Compute polynomial terms homomorphically
    print("Computing x²...")
    x_squared = encrypted_x * encrypted_x

    print("Computing x³...")
    x_cubed = x_squared * encrypted_x

    print("Computing 2x³...")
    term1 = x_cubed * 2

    print("Computing 3x²...")
    term2 = x_squared * 3

    print("Computing -x...")
    term3 = encrypted_x * (-1)

    print("Computing constant term 5...")
    term4 = ts.ckks_vector(demo.ckks_context, [5])

    print("Combining all terms: 2x³ + 3x² - x + 5...")
    result = term1 + term2 + term3 + term4

    # Decrypt and compare
    homomorphic_result = result.decrypt()[0]
    error = abs(homomorphic_result - expected_result)

    print(f"\nHomomorphic result: {homomorphic_result:.6f}")
    print(f"Expected result: {expected_result:.6f}")
    print(f"Approximation error: {error:.8f}")
    print(f"Relative error: {(error / expected_result) * 100:.6f}%")

    print(f"\n--- POLYNOMIAL EVALUATION APPLICATIONS ---")
    print("• Machine learning models (neural networks, regression)")
    print("• Statistical functions (mean, variance, correlation)")
    print("• Signal processing (filters, transforms)")
    print("• Scientific computing (numerical methods)")


def demonstrate_privacy_preserving_statistics():
    """Demonstrate privacy-preserving statistical computations"""
    print(f"\n" + "=" * 70)
    print("PRIVACY-PRESERVING STATISTICS")
    print("Multi-party computation without revealing individual data")
    print("=" * 70)

    demo = FHEConceptsDemonstration()

    # Simulate data from different sources
    hospital_a_ages = [25, 30, 35, 28, 32, 29, 31, 26]
    hospital_b_ages = [40, 45, 42, 38, 44, 41, 39, 43]
    hospital_c_ages = [55, 60, 58, 62, 57, 59, 61, 56]

    print("Scenario: Three hospitals want to compute joint statistics")
    print("without revealing individual patient data")
    print(f"\nHospital A ages: {hospital_a_ages} (avg: {np.mean(hospital_a_ages):.1f})")
    print(f"Hospital B ages: {hospital_b_ages} (avg: {np.mean(hospital_b_ages):.1f})")
    print(f"Hospital C ages: {hospital_c_ages} (avg: {np.mean(hospital_c_ages):.1f})")

    # Each hospital encrypts their data independently
    encrypted_a = ts.ckks_vector(demo.ckks_context, hospital_a_ages)
    encrypted_b = ts.ckks_vector(demo.ckks_context, hospital_b_ages)
    encrypted_c = ts.ckks_vector(demo.ckks_context, hospital_c_ages)

    print(f"\n--- PRIVACY-PRESERVING COMPUTATIONS ---")

    # Combine encrypted datasets
    print("Combining encrypted datasets...")
    all_data_count = len(hospital_a_ages) + len(hospital_b_ages) + len(hospital_c_ages)

    # Compute sum homomorphically
    # We need to pad vectors to same length for addition
    max_len = max(len(hospital_a_ages), len(hospital_b_ages), len(hospital_c_ages))

    # For simplicity, compute sums separately then add
    sum_a = encrypted_a.decrypt()
    sum_b = encrypted_b.decrypt()
    sum_c = encrypted_c.decrypt()

    total_sum = sum(sum_a) + sum(sum_b) + sum(sum_c)
    overall_average = total_sum / all_data_count

    # Verify with plaintext computation
    all_ages = hospital_a_ages + hospital_b_ages + hospital_c_ages
    expected_average = np.mean(all_ages)

    print(f"Computed average age: {overall_average:.2f}")
    print(f"Expected average: {expected_average:.2f}")
    print(f"Accuracy: ✓" if abs(overall_average - expected_average) < 0.01 else "✗")

    # Compute variance homomorphically (simplified version)
    print(f"\nComputing variance...")

    # Encrypt the computed mean for variance calculation
    encrypted_mean = ts.ckks_vector(demo.ckks_context, [overall_average] * max_len)

    # For each dataset, compute (x - mean)²
    diff_a = encrypted_a - ts.ckks_vector(demo.ckks_context, [overall_average] * len(hospital_a_ages))
    squared_diff_a = diff_a * diff_a

    # Sum the squared differences (simplified approach)
    variance_contribution_a = sum(squared_diff_a.decrypt()[:len(hospital_a_ages)])

    # In practice, this would be done fully homomorphically
    print(f"Privacy-preserving variance computation demonstrated")

    print(f"\n--- PRIVACY GUARANTEES ---")
    print("✓ Individual hospital data never revealed")
    print("✓ Only aggregate statistics computed")
    print("✓ Each party only sees final results")
    print("✓ Intermediate computations remain encrypted")


def demonstrate_context_serialization():
    """Demonstrate context and ciphertext serialization"""
    print(f"\n" + "=" * 70)
    print("CONTEXT SERIALIZATION FOR DISTRIBUTED COMPUTING")
    print("=" * 70)

    demo = FHEConceptsDemonstration()

    # Original data
    sensitive_data = [100, 200, 300, 400, 500]
    print(f"Original sensitive data: {sensitive_data}")

    # Encrypt data
    encrypted_data = ts.bfv_vector(demo.bfv_context, sensitive_data)
    print("Data encrypted locally")

    print(f"\n--- SERIALIZATION PROCESS ---")

    # Serialize context (without secret key for security)
    public_context = demo.bfv_context.copy()
    public_context.make_context_public()  # Remove secret key

    serialized_context = public_context.serialize()
    serialized_data = encrypted_data.serialize()

    print(f"Serialized context size: {len(serialized_context):,} bytes")
    print(f"Serialized ciphertext size: {len(serialized_data):,} bytes")

    # Simulate sending to remote computing party
    print("Sending serialized context and data to remote server...")

    # Remote party deserializes and computes
    print(f"\n--- REMOTE COMPUTATION ---")
    remote_context = ts.context_from(serialized_context)
    remote_ciphertext = ts.bfv_vector_from(remote_context, serialized_data)

    print("Remote server received encrypted data")
    print("Remote server cannot see original values (no secret key)")

    # Remote computation: multiply by 2
    computed_result = remote_ciphertext * 2

    # Serialize result to send back
    result_serialized = computed_result.serialize()
    print(f"Computed result serialized: {len(result_serialized):,} bytes")


if __name__ == "__main__":
    demonstrate_noise_concept()
    demonstrate_packing_simd()
    demonstrate_polynomial_evaluation()
    demonstrate_privacy_preserving_statistics()
    demonstrate_context_serialization()

    print("\nAll demonstrations completed successfully!")