"""
Enhanced TenSEAL Wrapper for FHE Operations
Supports CKKS, BFV schemes with proper parameter handling
Compatible with server application
Note: TenSEAL does not support BGV - will use BFV as fallback
"""

import tenseal as ts
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import base64
from typing import List, Any, Dict, Optional


class TenSEALWrapper:
    """Enhanced wrapper for TenSEAL FHE operations"""

    def __init__(self):
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}

    def generate_context(self, scheme='CKKS', poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=None, scale=2 ** 40,
                         plain_modulus=1032193):
        """Generate TenSEAL context with specified scheme and parameters"""

        # Note: TenSEAL only supports CKKS and BFV, not BGV
        if scheme == 'BGV':
            print("⚠️ TenSEAL does not support BGV scheme. Using BFV instead.")
            scheme = 'BFV'

        self.scheme = scheme

        if coeff_mod_bit_sizes is None:
            # Default coefficient modulus bit sizes
            if scheme == 'CKKS':
                coeff_mod_bit_sizes = [60, 40, 40, 60]
            else:  # BFV
                coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.params = {
            'poly_modulus_degree': poly_modulus_degree,
            'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
            'scale': scale,
            'plain_modulus': plain_modulus
        }

        try:
            if scheme == 'CKKS':
                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )
                self.context.global_scale = scale
                self.context.generate_galois_keys()
                self.context.generate_relin_keys()
                print(f"✅ CKKS context created with scale: {scale}")

            elif scheme == 'BFV':
                self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=poly_modulus_degree,
                    plain_modulus=plain_modulus,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )
                self.context.generate_galois_keys()
                self.context.generate_relin_keys()
                print(f"✅ BFV context created with plain_modulus: {plain_modulus}")
            else:
                raise ValueError(f"Unsupported scheme: {scheme}. TenSEAL supports CKKS and BFV only.")

            return self.context

        except Exception as e:
            print(f"❌ Error creating context: {str(e)}")
            raise

    def get_keys_info(self) -> Dict[str, Any]:
        """Get information about generated keys"""
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
            public_key = f"Error serializing: {str(e)}"
            private_key = f"Error serializing: {str(e)}"

        return {
            'public_key': public_key[:100] + '...' if len(public_key) > 100 else public_key,
            'private_key': private_key[:100] + '...' if len(private_key) > 100 else private_key,
            'galois_keys': "Generated",
            'relin_keys': "Generated",
            'full_public_key': public_key,
            'full_private_key': private_key,
            'scheme': self.scheme,
            'params': self.params
        }

    def encrypt_data(self, data: List[Any], column_name: str, data_type: str) -> List[Optional[bytes]]:
        """Encrypt data based on type"""
        if not self.context:
            raise ValueError("Context not initialized")

        # Scheme limitations
        if self.scheme == 'BFV':
            if data_type in ['text', 'date']:
                print(f"⚠️ BFV scheme does not support {data_type} directly. Converting to integer encoding.")

        encrypted_values = []

        for value in data:
            try:
                if pd.isna(value) or value is None:
                    encrypted_values.append(None)
                    continue

                # Handle different data types
                if data_type == 'numeric':
                    if self.scheme == 'CKKS':
                        # CKKS supports floating point
                        try:
                            encrypted_val = ts.ckks_vector(self.context, [float(value)])
                            encrypted_values.append(encrypted_val.serialize())
                        except Exception as e:
                            print(f"CKKS encryption error for {value}: {str(e)}")
                            encrypted_values.append(None)
                    else:  # BFV
                        # BFV requires integers - convert and round
                        try:
                            int_value = int(float(value))
                            encrypted_val = ts.bfv_vector(self.context, [int_value])
                            encrypted_values.append(encrypted_val.serialize())
                        except Exception as e:
                            print(f"BFV encryption error for {value}: {str(e)}")
                            encrypted_values.append(None)

                elif data_type == 'text':
                    # Encode text as sum of ASCII values
                    numeric_value = sum([ord(c) for c in str(value)])

                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(numeric_value)])
                    else:  # BFV
                        encrypted_val = ts.bfv_vector(self.context, [int(numeric_value)])

                    encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'date':
                    # Convert date to timestamp
                    try:
                        if isinstance(value, (pd.Timestamp, datetime)):
                            timestamp = value.timestamp()
                        else:
                            timestamp = pd.Timestamp(value).timestamp()

                        if self.scheme == 'CKKS':
                            encrypted_val = ts.ckks_vector(self.context, [float(timestamp)])
                        else:  # BFV - use integer timestamp
                            encrypted_val = ts.bfv_vector(self.context, [int(timestamp)])

                        encrypted_values.append(encrypted_val.serialize())
                    except Exception as e:
                        print(f"Date encryption error for {value}: {str(e)}")
                        encrypted_values.append(None)
                else:
                    # Default: treat as numeric
                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(value)])
                    else:
                        encrypted_val = ts.bfv_vector(self.context, [int(float(value))])
                    encrypted_values.append(encrypted_val.serialize())

            except Exception as e:
                print(f"Error encrypting value {value}: {str(e)}")
                encrypted_values.append(None)

        print(f"✅ Encrypted {len([x for x in encrypted_values if x is not None])}/{len(data)} values")
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

                # Handle different input formats
                if isinstance(enc_value, dict):
                    if 'ciphertext' in enc_value:
                        enc_bytes = bytes.fromhex(enc_value['ciphertext'])
                    elif 'data' in enc_value and enc_value.get('type') == 'bytes':
                        enc_bytes = base64.b64decode(enc_value['data'])
                    else:
                        decrypted_values.append(None)
                        continue
                elif isinstance(enc_value, bytes):
                    enc_bytes = enc_value
                elif isinstance(enc_value, str):
                    try:
                        enc_bytes = bytes.fromhex(enc_value)
                    except:
                        enc_bytes = base64.b64decode(enc_value)
                else:
                    decrypted_values.append(None)
                    continue

                # Decrypt based on scheme
                if self.scheme == 'CKKS':
                    vector = ts.ckks_vector_from(self.context, enc_bytes)
                else:  # BFV
                    vector = ts.bfv_vector_from(self.context, enc_bytes)

                decrypted = vector.decrypt()[0]

                # Convert back to appropriate type
                if data_type == 'date':
                    decrypted_values.append(pd.Timestamp.fromtimestamp(decrypted))
                else:
                    decrypted_values.append(decrypted)

            except Exception as e:
                print(f"Error decrypting value: {str(e)}")
                decrypted_values.append(None)

        return decrypted_values

    def perform_operation(self, encrypted_data1: List[Any], encrypted_data2: List[Any],
                          operation: str) -> List[Optional[bytes]]:
        """Perform homomorphic operations on two lists"""
        if not self.context:
            raise ValueError("Context not initialized")

        results = []

        for enc1, enc2 in zip(encrypted_data1, encrypted_data2):
            try:
                if enc1 is None or enc2 is None:
                    results.append(None)
                    continue

                # Convert to bytes
                if isinstance(enc1, dict) and 'ciphertext' in enc1:
                    enc1 = bytes.fromhex(enc1['ciphertext'])
                if isinstance(enc2, dict) and 'ciphertext' in enc2:
                    enc2 = bytes.fromhex(enc2['ciphertext'])

                # Deserialize based on scheme
                if self.scheme == 'CKKS':
                    vec1 = ts.ckks_vector_from(self.context, enc1)
                    vec2 = ts.ckks_vector_from(self.context, enc2)
                else:  # BFV
                    vec1 = ts.bfv_vector_from(self.context, enc1)
                    vec2 = ts.bfv_vector_from(self.context, enc2)

                # Perform operation
                if operation == 'add':
                    result = vec1 + vec2
                elif operation == 'multiply':
                    result = vec1 * vec2
                elif operation == 'subtract':
                    result = vec1 - vec2
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                results.append(result.serialize())

            except Exception as e:
                print(f"Error performing operation: {str(e)}")
                results.append(None)

        return results

    def perform_aggregation(self, encrypted_data_list: List[Any], operation: str) -> Optional[bytes]:
        """
        Perform aggregation operations on multiple encrypted values
        Supports: sum, average
        Note: min/max require comparison circuits not available in basic FHE
        """
        print(f"TenSEAL wrapper: Performing aggregation: {operation}")
        if not self.context:
            raise ValueError("Context not initialized")

        try:
            # Filter out None values and prepare ciphertexts
            valid_data = []
            for enc in encrypted_data_list:
                if enc is None:
                    continue

                # Handle different input formats
                if isinstance(enc, dict):
                    if 'ciphertext' in enc:
                        enc_bytes = bytes.fromhex(enc['ciphertext'])
                    elif 'data' in enc and enc.get('type') == 'bytes':
                        enc_bytes = base64.b64decode(enc['data'])
                    else:
                        continue
                elif isinstance(enc, bytes):
                    enc_bytes = enc
                else:
                    continue

                # Deserialize based on scheme
                try:
                    if self.scheme == 'CKKS':
                        vec = ts.ckks_vector_from(self.context, enc_bytes)
                    else:  # BFV
                        vec = ts.bfv_vector_from(self.context, enc_bytes)
                    valid_data.append(vec)
                except Exception as e:
                    print(f"Error deserializing ciphertext: {str(e)}")
                    continue

            if not valid_data:
                print("⚠️ No valid encrypted data for aggregation")
                return None

            print(f"   Processing {len(valid_data)} encrypted values...")

            # Perform aggregation
            if operation in ['sum', 'add']:
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec
                print(f"✅ Sum aggregation complete")
                return result.serialize()

            elif operation in ['average', 'avg']:
                # Sum all values first
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec

                # For average, we can only do this in CKKS (supports floating point division)
                if self.scheme == 'CKKS':
                    count = len(valid_data)
                    result = result * (1.0 / count)
                    print(f"✅ Average aggregation complete")
                    return result.serialize()
                else:
                    # BFV doesn't support division - return sum instead
                    # Client would need to decrypt and divide by count
                    print(f"⚠️ BFV doesn't support division. Returning sum (divide by {len(valid_data)} after decryption)")
                    return result.serialize()

            elif operation == 'min':
                raise ValueError(f"MIN operation requires comparison circuits not available in basic CKKS/BFV")

            elif operation == 'max':
                raise ValueError(f"MAX operation requires comparison circuits not available in basic CKKS/BFV")

            else:
                raise ValueError(f"Unsupported aggregation: {operation}")

        except Exception as e:
            print(f"❌ Error in aggregation '{operation}': {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_supported_operations(self) -> Dict[str, Dict[str, List[str]]]:
        """Get list of supported operations per scheme"""
        return {
            'CKKS': {
                'binary': ['add', 'subtract', 'multiply'],
                'aggregation': ['sum', 'average'],
                'unsupported': ['min', 'max', 'comparison', 'division'],
                'data_types': ['numeric (float)', 'text (encoded)', 'date (encoded)']
            },
            'BFV': {
                'binary': ['add', 'subtract', 'multiply'],
                'aggregation': ['sum'],
                'unsupported': ['average (requires division)', 'min', 'max', 'comparison'],
                'data_types': ['integer only', 'text (encoded as int)', 'date (encoded as int)']
            }
        }

    def get_scheme_limitations(self) -> Dict[str, Any]:
        """Get limitations of the current scheme"""
        limitations = {
            'CKKS': {
                'supports_text': 'Limited (requires encoding to numeric)',
                'supports_numeric': 'Yes (approximate floating point)',
                'supports_comparison': 'No',
                'precision': 'Approximate (floating point)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Sum', 'Average'],
                'advantages': ['Supports floating point', 'Efficient for financial calculations', 'Can do division (avg)'],
                'disadvantages': ['Approximate results', 'No exact integers', 'No min/max', 'No comparisons']
            },
            'BFV': {
                'supports_text': 'Limited (encoded as integer)',
                'supports_numeric': 'Yes (exact integers only)',
                'supports_comparison': 'No',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Sum'],
                'advantages': ['Exact arithmetic', 'Faster for integers', 'No rounding errors'],
                'disadvantages': ['No floating point', 'No division (no average)', 'No min/max', 'Must convert floats to ints']
            },
            'BGV': {
                'note': 'TenSEAL does not support BGV. Using BFV instead.',
                'supports_text': 'Limited (encoded as integer)',
                'supports_numeric': 'Yes (exact integers only)',
                'supports_comparison': 'No',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Sum'],
                'advantages': ['Exact arithmetic', 'Similar to BFV'],
                'disadvantages': ['Not implemented in TenSEAL - BFV used as fallback']
            }
        }
        return limitations.get(self.scheme, limitations.get('CKKS', {}))

    def rotate_keys(self) -> Dict[str, Any]:
        """Rotate encryption keys"""
        if not self.context:
            raise ValueError("Context not initialized")

        old_context_serial = self.context.serialize()
        new_context = self.generate_context(
            scheme=self.scheme,
            poly_modulus_degree=self.params.get('poly_modulus_degree', 8192),
            coeff_mod_bit_sizes=self.params.get('coeff_mod_bit_sizes', [60, 40, 40, 60]),
            scale=self.params.get('scale', 2 ** 40),
            plain_modulus=self.params.get('plain_modulus', 1032193)
        )

        return {
            'old_context': base64.b64encode(old_context_serial).decode('utf-8'),
            'new_context': base64.b64encode(new_context.serialize()).decode('utf-8'),
            'rotation_time': datetime.now().isoformat(),
            'scheme': self.scheme,
            'params': self.params
        }

    def serialize_context(self) -> str:
        """Serialize context for storage"""
        if not self.context:
            raise ValueError("Context not initialized")
        return base64.b64encode(self.context.serialize()).decode('utf-8')

    def load_context(self, serialized_context: str):
        """Load context from serialized string"""
        context_bytes = base64.b64decode(serialized_context)
        self.context = ts.context_from(context_bytes)