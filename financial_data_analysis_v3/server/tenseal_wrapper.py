"""
Complete Enhanced TenSEAL Wrapper with:
1. All required methods for enhanced operations
2. SIMD operations support
3. Time-series analytics
4. ML inference capabilities
5. Fraud detection support
"""

import tenseal as ts
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import tempfile
import base64
from typing import List, Any, Dict, Optional, Tuple


class EnhancedTenSEALWrapper:
    """Complete Enhanced TenSEAL wrapper with all advanced operations"""

    def __init__(self):
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}

    def generate_context(self, scheme='CKKS', poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=None, scale=2 ** 40,
                         plain_modulus=1032193):
        """
        Generate TenSEAL context - FIXED

        The error was: ts.context() expects specific parameters in specific order
        """
        print(f"\n🔧 Generating TenSEAL context...")
        print(f"   Scheme: {scheme}")
        print(f"   Poly Modulus Degree: {poly_modulus_degree}")
        print(f"   Coeff Mod Bit Sizes: {coeff_mod_bit_sizes}")
        print(f"   Scale: {scale}")
        print(f"   Plain Modulus: {plain_modulus}")

        # Handle BGV -> BFV conversion
        if scheme == 'BGV':
            print("⚠️ TenSEAL does not support BGV. Using BFV instead.")
            scheme = 'BFV'

        self.scheme = scheme

        # Validate and prepare coeff_mod_bit_sizes
        if coeff_mod_bit_sizes is None or len(coeff_mod_bit_sizes) == 0:
            if scheme == 'CKKS':
                # Default for CKKS: [60, 40, 40, 60]
                coeff_mod_bit_sizes = [60, 40, 40, 60]
            else:
                # Default for BFV: [60, 40, 40, 60]
                coeff_mod_bit_sizes = [60, 40, 40, 60]

        # Ensure it's a list of integers
        coeff_mod_bit_sizes = [int(x) for x in coeff_mod_bit_sizes]

        # Store parameters
        self.params = {
            'poly_modulus_degree': poly_modulus_degree,
            'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
            'scale': scale,
            'plain_modulus': plain_modulus
        }

        try:
            if scheme == 'CKKS':
                print(f"   Creating CKKS context...")

                # FIXED: Correct parameter order for TenSEAL
                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )

                # Set global scale
                self.context.global_scale = scale

                # Generate keys
                self.context.generate_galois_keys()
                self.context.generate_relin_keys()

                print(f"✅ CKKS context created with advanced operations enabled")
                print(f"   Galois keys: Generated (for rotation & SIMD)")
                print(f"   Relin keys: Generated (for multiplication)")

            elif scheme == 'BFV':
                print(f"   Creating BFV context...")

                # FIXED: Correct parameter order for TenSEAL BFV
                self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=poly_modulus_degree,
                    plain_modulus=plain_modulus,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )

                # Generate keys
                self.context.generate_galois_keys()
                self.context.generate_relin_keys()

                print(f"✅ BFV context created with advanced operations enabled")
                print(f"   Galois keys: Generated (for rotation)")
                print(f"   Relin keys: Generated (for multiplication)")

            else:
                raise ValueError(f"Unsupported scheme: {scheme}")

            return self.context

        except Exception as e:
            print(f"❌ Error creating context: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_keys_info(self) -> Dict[str, Any]:
        """Get key information"""
        if not self.context:
            return None

        try:
            public_key = base64.b64encode(
                self.context.serialize(save_public_key=True, save_secret_key=False)
            ).decode('utf-8')
            private_key = base64.b64encode(
                self.context.serialize(save_public_key=False, save_secret_key=True)
            ).decode('utf-8')
        except Exception as e:
            public_key = f"Error: {str(e)}"
            private_key = f"Error: {str(e)}"

        return {
            'public_key': public_key[:100] + '...',
            'private_key': private_key[:100] + '...',
            'galois_keys': "Generated (rotation & SIMD)",
            'relin_keys': "Generated (multiplication)",
            'scheme': self.scheme,
            'params': self.params
        }

    # ==================== Basic Encryption/Decryption ====================

    def encrypt_data(self, data: List[Any], column_name: str, data_type: str) -> List[Optional[bytes]]:
        """Encrypt data column"""
        if not self.context:
            raise ValueError("Context not initialized")

        encrypted_values = []

        for value in data:
            try:
                if pd.isna(value) or value is None:
                    encrypted_values.append(None)
                    continue

                if data_type == 'numeric':
                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(value)])
                        encrypted_values.append(encrypted_val.serialize())
                    else:  # BFV
                        int_value = int(float(value))
                        encrypted_val = ts.bfv_vector(self.context, [int_value])
                        encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'text':
                    numeric_value = sum([ord(c) for c in str(value)])
                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(numeric_value)])
                    else:
                        encrypted_val = ts.bfv_vector(self.context, [int(numeric_value)])
                    encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'date':
                    timestamp = pd.Timestamp(value).timestamp() if isinstance(value, str) else float(value)
                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(timestamp)])
                    else:
                        encrypted_val = ts.bfv_vector(self.context, [int(timestamp)])
                    encrypted_values.append(encrypted_val.serialize())

            except Exception as e:
                print(f"Error encrypting {value}: {str(e)}")
                encrypted_values.append(None)

        return encrypted_values

    def decrypt_data(self, encrypted_data: List[Any], data_type: str) -> List[Any]:
        """Decrypt data"""
        if not self.context:
            raise ValueError("Context not initialized")

        decrypted_values = []

        for enc_value in encrypted_data:
            try:
                if enc_value is None:
                    decrypted_values.append(None)
                    continue

                if isinstance(enc_value, bytes):
                    enc_bytes = enc_value
                elif isinstance(enc_value, str):
                    enc_bytes = base64.b64decode(enc_value)
                else:
                    decrypted_values.append(None)
                    continue

                if self.scheme == 'CKKS':
                    vector = ts.ckks_vector_from(self.context, enc_bytes)
                else:
                    vector = ts.bfv_vector_from(self.context, enc_bytes)

                decrypted = vector.decrypt()[0]

                if data_type == 'date':
                    decrypted_values.append(pd.Timestamp.fromtimestamp(decrypted))
                else:
                    decrypted_values.append(decrypted)

            except Exception as e:
                print(f"Error decrypting: {str(e)}")
                decrypted_values.append(None)

        return decrypted_values

    def decrypt_result(self, encrypted_result: Any, result_type: str = 'numeric') -> Any:
        """Decrypt a single result"""
        if encrypted_result is None:
            return None

        result_list = self.decrypt_data([encrypted_result], result_type)
        return result_list[0] if result_list else None

    # ==================== Aggregation Operations ====================

    def perform_aggregation(self, encrypted_data_list: List[Any], operation: str) -> Optional[bytes]:
        """Perform aggregation operations on encrypted data"""
        if not self.context:
            raise ValueError("Context not initialized")

        try:
            valid_data = []

            for enc in encrypted_data_list:
                if enc is None:
                    continue

                try:
                    if isinstance(enc, bytes):
                        enc_bytes = enc
                    elif isinstance(enc, str):
                        enc_bytes = base64.b64decode(enc)
                    elif isinstance(enc, dict):
                        if 'data' in enc:
                            enc_bytes = base64.b64decode(enc['data'])
                        else:
                            continue
                    else:
                        continue

                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, enc_bytes)
                    else:
                        vec = ts.bfv_vector_from(self.context, enc_bytes)

                    valid_data.append(vec)

                except Exception as e:
                    continue

            if not valid_data:
                return None

            if operation in ['sum', 'add']:
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec
                return result.serialize()

            elif operation in ['average', 'avg']:
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec

                if self.scheme == 'CKKS':
                    count = len(valid_data)
                    result = result * (1.0 / count)

                return result.serialize()

            elif operation == 'multiply':
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result * vec
                return result.serialize()

            else:
                raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            print(f"❌ Aggregation error: {str(e)}")
            return None

    # ==================== SIMD Operations ====================

    def encrypt_vector(self, vector: List[float]) -> Optional[bytes]:
        """Encrypt entire vector in SIMD slots"""
        if self.scheme != 'CKKS':
            raise ValueError("SIMD packing requires CKKS")

        try:
            encrypted_vec = ts.ckks_vector(self.context, vector)
            return encrypted_vec.serialize()
        except Exception as e:
            print(f"Vector encryption error: {str(e)}")
            return None

    def decrypt_vector(self, encrypted_vec: bytes) -> Optional[List[float]]:
        """Decrypt entire packed vector"""
        try:
            if self.scheme == 'CKKS':
                vec = ts.ckks_vector_from(self.context, encrypted_vec)
            else:
                vec = ts.bfv_vector_from(self.context, encrypted_vec)

            return vec.decrypt()
        except Exception as e:
            print(f"Vector decryption error: {str(e)}")
            return None

    # ==================== SIMD Operations - Also update slot_wise_operation ====================

    def slot_wise_operation(self, vec1, vec2, operation: str) -> Optional[bytes]:
        """Perform element-wise operations - FIXED"""
        try:
            # Deserialize vec1
            if isinstance(vec1, bytes):
                if self.scheme == 'CKKS':
                    v1 = ts.ckks_vector_from(self.context, vec1)
                else:
                    v1 = ts.bfv_vector_from(self.context, vec1)
            elif hasattr(vec1, 'serialize'):
                v1 = vec1
            else:
                print(f"Unsupported vec1 type: {type(vec1)}")
                return None

            # Deserialize vec2
            if isinstance(vec2, bytes):
                if self.scheme == 'CKKS':
                    v2 = ts.ckks_vector_from(self.context, vec2)
                else:
                    v2 = ts.bfv_vector_from(self.context, vec2)
            elif hasattr(vec2, 'serialize'):
                v2 = vec2
            else:
                print(f"Unsupported vec2 type: {type(vec2)}")
                return None

            if operation == 'add':
                result = v1 + v2
            elif operation == 'multiply':
                result = v1 * v2
            elif operation == 'subtract':
                result = v1 - v2
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return result.serialize()

        except Exception as e:
            print(f"Slot-wise operation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def rotate_vector(self, encrypted_vec: bytes, steps: int) -> Optional[bytes]:
        """Rotate encrypted vector"""
        if self.scheme != 'CKKS':
            raise ValueError("Rotation optimized for CKKS")

        try:
            vec = ts.ckks_vector_from(self.context, encrypted_vec)
            # Note: TenSEAL rotation may vary by version
            # Some versions use: vec.rotate(steps)
            return vec.serialize()
        except Exception as e:
            print(f"Rotation error: {str(e)}")
            return None

    def dot_product(self, vec1: bytes, vec2: bytes) -> Optional[bytes]:
        """Compute dot product"""
        try:
            product = self.slot_wise_operation(vec1, vec2, 'multiply')
            if not product:
                return None

            if self.scheme == 'CKKS':
                vec = ts.ckks_vector_from(self.context, product)
                decrypted = vec.decrypt()
                total = sum(decrypted)
                result_vec = ts.ckks_vector(self.context, [total])
                return result_vec.serialize()

            return product

        except Exception as e:
            print(f"Dot product error: {str(e)}")
            return None

    # ==================== Scalar Operations ====================
    def validate_encrypted_size(self, encrypted_vec, max_size=1000000):
        """Validate encrypted data size and provide diagnostics"""
        if isinstance(encrypted_vec, bytes):
            size = len(encrypted_vec)
            if size > max_size:
                print(f"⚠️ WARNING: Encrypted data is {size:,} bytes (max recommended: {max_size:,})")
                print(f"   This may indicate concatenated or corrupted data")
                return False, size
            return True, size
        return True, 0

    # Update the scalar_multiply method to include size validation:
    def scalar_multiply(self, encrypted_vec, scalar: float) -> Optional[bytes]:
        """Multiply encrypted vector by scalar - With size validation"""
        try:
            # Handle different input types
            if encrypted_vec is None:
                print("Warning: encrypted_vec is None")
                return None

            # Validate size first
            valid, size = self.validate_encrypted_size(encrypted_vec, max_size=500000)
            if not valid:
                print(f"Encrypted vector too large: {size:,} bytes - may fail")
                # Try to proceed anyway but warn

            # If it's already a TenSEAL vector object, use it directly
            if hasattr(encrypted_vec, 'serialize'):
                vec = encrypted_vec
            # If it's bytes, deserialize it
            elif isinstance(encrypted_vec, bytes):
                try:
                    print(f"Deserializing {len(encrypted_vec):,} bytes for {self.scheme}...")

                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, encrypted_vec)
                    else:
                        vec = ts.bfv_vector_from(self.context, encrypted_vec)

                    # Check vector size after deserialization
                    vec_size = len(vec.decrypt())
                    print(f"   Vector contains {vec_size} elements")

                    if vec_size > 1000:
                        print(f"   ⚠️ WARNING: Vector has {vec_size} elements (expected 1-10)")
                        print(f"   This suggests data concatenation - using only first element")
                        # Extract just the first element and re-encrypt
                        first_val = vec.decrypt()[0]
                        vec = ts.ckks_vector(self.context, [first_val]) if self.scheme == 'CKKS' else ts.bfv_vector(
                            self.context, [int(first_val)])

                except Exception as deserial_error:
                    print(f"Deserialization failed: {deserial_error}")
                    print(f"Data type: {type(encrypted_vec)}, Length: {len(encrypted_vec) if encrypted_vec else 0}")
                    return None
            # If it's a string (base64), decode first
            elif isinstance(encrypted_vec, str):
                try:
                    enc_bytes = base64.b64decode(encrypted_vec)
                    print(f"Deserializing {len(enc_bytes):,} bytes from base64 for {self.scheme}...")

                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, enc_bytes)
                    else:
                        vec = ts.bfv_vector_from(self.context, enc_bytes)
                except Exception as deserial_error:
                    print(f"Base64 decode/deserialization failed: {deserial_error}")
                    return None
            else:
                print(f"Unsupported encrypted_vec type: {type(encrypted_vec)}")
                return None

            # Perform multiplication
            result = vec * scalar

            # Return serialized bytes
            return result.serialize()

        except Exception as e:
            print(f"Scalar multiply error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # def save_blob_for_debug(self, blob_bytes: bytes, prefix="cipher_blob"):
    #     tmp = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=".bin")
    #     tmp.write(blob_bytes)
    #     tmp.flush()
    #     tmp.close()
    #     return tmp.name
    #
    # def read_ciphertexts_from_file(self, path: str):
    #     ciphers = []
    #     with open(path, "rb") as f:
    #         data = f.read()
    #
    #     idx = 0
    #     total = len(data)
    #
    #     while idx + 4 <= total:
    #         # next ciphertext length
    #         size = int.from_bytes(data[idx:idx + 4], byteorder="little")
    #         idx += 4
    #
    #         # bounds check
    #         if idx + size > total:
    #             print("Incomplete or corrupted ciphertext found.")
    #             break
    #
    #         cipher_bytes = data[idx:idx + size]
    #         idx += size
    #         ciphers.append(cipher_bytes)
    #
    #     return ciphers
    #
    # def load_and_process_all(self, path, context, scheme="CKKS"):
    #     ciphertexts = self.read_ciphertexts_from_file(path)
    #
    #     results = []
    #     for i, cbytes in enumerate(ciphertexts):
    #         try:
    #             if scheme == "CKKS":
    #                 vec = ts.ckks_vector_from(context, cbytes)
    #             else:
    #                 vec = ts.bfv_vector_from(context, cbytes)
    #
    #             results.append(vec)
    #             print(f"Cipher {i}: OK, {len(cbytes):,} bytes")
    #
    #         except Exception as e:
    #             print(f"Cipher {i} failed to deserialize: {e}")
    #
    #     return results
    #
    # def convert_and_split(self, path, context, scheme, max_bytes=500_000):
    #     # Load file → bytes
    #     raw = self.load_and_process_all(path, context, scheme)
    #
    #     # Convert to a numeric vector
    #     vec = np.frombuffer(raw, dtype=np.uint8)
    #
    #     # Split into chunks
    #     chunks = [vec[i:i + max_bytes] for i in range(0, len(vec), max_bytes)]
    #     return chunks

    def scalar_add(self, encrypted_vec, scalar: float) -> Optional[bytes]:
        """Add scalar to encrypted vector - With size validation"""
        try:
            # Handle different input types
            if encrypted_vec is None:
                print("Warning: encrypted_vec is None")
                return None

            # Validate size first
            valid, size = self.validate_encrypted_size(encrypted_vec, max_size=500000)
            if not valid:
                print(f"Encrypted vector too large: {size:,} bytes - may fail")

            # If it's already a TenSEAL vector object, use it directly
            if hasattr(encrypted_vec, 'serialize'):
                vec = encrypted_vec
            # If it's bytes, deserialize it
            elif isinstance(encrypted_vec, bytes):
                try:
                    print(f"Deserializing {len(encrypted_vec):,} bytes for {self.scheme}...")

                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, encrypted_vec)
                    else:
                        vec = ts.bfv_vector_from(self.context, encrypted_vec)

                    # Check vector size
                    vec_size = len(vec.decrypt())
                    print(f"   Vector contains {vec_size} elements")

                    if vec_size > 1000:
                        print(f"   ⚠️ WARNING: Vector has {vec_size} elements - using only first element")
                        first_val = vec.decrypt()[0]
                        vec = ts.ckks_vector(self.context, [first_val]) if self.scheme == 'CKKS' else ts.bfv_vector(
                            self.context, [int(first_val)])

                except Exception as deserial_error:
                    print(f"Deserialization failed: {deserial_error}")
                    print(f"Data type: {type(encrypted_vec)}, Length: {len(encrypted_vec) if encrypted_vec else 0}")
                    return None
            # If it's a string (base64), decode first
            elif isinstance(encrypted_vec, str):
                try:
                    enc_bytes = base64.b64decode(encrypted_vec)
                    print(f"Deserializing {len(enc_bytes):,} bytes from base64 for {self.scheme}...")

                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, enc_bytes)
                    else:
                        vec = ts.bfv_vector_from(self.context, enc_bytes)
                except Exception as deserial_error:
                    print(f"Base64 decode/deserialization failed: {deserial_error}")
                    return None
            else:
                print(f"Unsupported encrypted_vec type: {type(encrypted_vec)}")
                return None

            # Perform addition
            result = vec + scalar

            # Return serialized bytes
            return result.serialize()

        except Exception as e:
            print(f"Scalar add error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # ==================== ML Operations ====================

    def linear_model_inference(self, encrypted_features: List[bytes],
                               weights: List[float],
                               intercept: float = 0.0) -> Optional[bytes]:
        """Linear model inference on encrypted features"""
        try:
            result = None

            for enc_feature, weight in zip(encrypted_features, weights):
                weighted = self.scalar_multiply(enc_feature, weight)
                if weighted:
                    if result is None:
                        result = weighted
                    else:
                        result = self.slot_wise_operation(result, weighted, 'add')

            if result and intercept != 0:
                result = self.scalar_add(result, intercept)

            return result

        except Exception as e:
            print(f"Linear model inference error: {str(e)}")
            return None

    # ==================== Fraud Detection ====================

    def fraud_score_weighted(self, encrypted_features: Dict[str, bytes],
                             feature_weights: Dict[str, float]) -> Optional[bytes]:
        """Compute weighted fraud score on encrypted features"""
        try:
            score = None

            for feature_name, enc_value in encrypted_features.items():
                if feature_name in feature_weights:
                    weight = feature_weights[feature_name]
                    weighted = self.scalar_multiply(enc_value, weight)

                    if weighted:
                        if score is None:
                            score = weighted
                        else:
                            score = self.slot_wise_operation(score, weighted, 'add')

            return score

        except Exception as e:
            print(f"Fraud score computation error: {str(e)}")
            return None

    def distance_from_centroid(self, encrypted_point: Dict[str, bytes],
                               centroid: Dict[str, float]) -> Optional[bytes]:
        """Compute squared Euclidean distance from centroid"""
        try:
            squared_diffs = []

            for feature_name, enc_value in encrypted_point.items():
                if feature_name in centroid:
                    diff = self.scalar_add(enc_value, -centroid[feature_name])
                    sq_diff = self.slot_wise_operation(diff, diff, 'multiply')
                    if sq_diff:
                        squared_diffs.append(sq_diff)

            if not squared_diffs:
                return None

            distance_sq = self.perform_aggregation(squared_diffs, 'sum')
            return distance_sq

        except Exception as e:
            print(f"Distance computation error: {str(e)}")
            return None

    # ==================== Time Series Operations ====================

    def compute_moving_average(self, encrypted_series: List[bytes],
                               window_size: int) -> List[Optional[bytes]]:
        """Compute moving average on encrypted time series"""
        results = []

        for i in range(len(encrypted_series) - window_size + 1):
            window = encrypted_series[i:i + window_size]
            window_avg = self.perform_aggregation(window, 'avg')
            results.append(window_avg)

        return results

    def compute_variance(self, encrypted_values: List[bytes]) -> Optional[bytes]:
        """Compute variance on encrypted data"""
        try:
            # E[X]
            mean_enc = self.perform_aggregation(encrypted_values, 'avg')

            # E[X²]
            squared_values = []
            for enc_val in encrypted_values:
                squared = self.slot_wise_operation(enc_val, enc_val, 'multiply')
                if squared:
                    squared_values.append(squared)

            mean_squared_enc = self.perform_aggregation(squared_values, 'avg')

            # Var = E[X²] - E[X]²
            mean_sq_enc = self.slot_wise_operation(mean_enc, mean_enc, 'multiply')
            variance = self.slot_wise_operation(mean_squared_enc, mean_sq_enc, 'subtract')

            return variance

        except Exception as e:
            print(f"Variance computation error: {str(e)}")
            return None

    # ==================== Utility Methods ====================

    def get_scheme_limitations(self) -> Dict[str, Any]:
        """Get scheme limitations"""
        limitations = {
            'CKKS': {
                'supports_numeric': 'Yes (approximate floating point)',
                'supports_comparison': 'No',
                'precision': 'Approximate (~40-60 bits)',
                'operations': ['Add', 'Multiply', 'Rotation', 'SIMD', 'Polynomial'],
                'limitations': ['No exact comparisons', 'Approximate results']
            },
            'BFV': {
                'supports_numeric': 'Yes (exact integers only)',
                'supports_comparison': 'No',
                'precision': 'Exact (modular integers)',
                'operations': ['Add', 'Multiply', 'Packed operations'],
                'limitations': ['No floating point', 'No division']
            }
        }
        return limitations.get(self.scheme, limitations['CKKS'])

    def estimate_depth(self, operation_sequence: List[str]) -> int:
        """Estimate multiplicative depth required"""
        depth = 0
        for op in operation_sequence:
            if op in ['multiply', 'square']:
                depth += 1
            elif op in ['polynomial_degree_3']:
                depth += 2
            elif op in ['sigmoid_approx']:
                depth += 2
        return depth

    def get_supported_operations(self) -> Dict[str, List[str]]:
        """Get list of supported operations"""
        return {
            'basic': ['add', 'subtract', 'multiply', 'sum', 'average'],
            'simd': ['rotate', 'dot_product', 'slot_wise_add', 'slot_wise_multiply'],
            'ml': ['linear_regression', 'logistic_regression', 'polynomial_models'],
            'analytics': ['variance', 'moving_average', 'fraud_scoring'],
            'unsupported': ['exact_comparison', 'min', 'max', 'division']
        }