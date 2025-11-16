"""
Enhanced TenSEAL Wrapper for FHE Operations
Supports multiple operations on multiple encrypted values
Compatible with server application
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
        self.scheme = scheme

        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.params = {
            'poly_modulus_degree': poly_modulus_degree,
            'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
            'scale': scale,
            'plain_modulus': plain_modulus
        }

        if scheme == 'CKKS':
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()

        elif scheme == 'BFV':
            self.context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus
            )
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        return self.context

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

        if self.scheme == 'BFV' and data_type == 'text':
            raise ValueError("BFV scheme does not support text encryption directly")

        encrypted_values = []

        for value in data:
            try:
                if pd.isna(value) or value is None:
                    encrypted_values.append(None)
                    continue

                if data_type == 'numeric':
                    if self.scheme == 'CKKS':
                        encrypted_val = ts.ckks_vector(self.context, [float(value)])
                    else:  # BFV
                        encrypted_val = ts.bfv_vector(self.context, [int(value)])
                    encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'text':
                    # For text, convert to numeric encoding
                    numeric_value = sum([ord(c) for c in str(value)])
                    encrypted_val = ts.ckks_vector(self.context, [float(numeric_value)])
                    encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'date':
                    timestamp = pd.Timestamp(value).timestamp()
                    encrypted_val = ts.ckks_vector(self.context, [float(timestamp)])
                    encrypted_values.append(encrypted_val.serialize())
                else:
                    encrypted_val = ts.ckks_vector(self.context, [float(value)])
                    encrypted_values.append(encrypted_val.serialize())

            except Exception as e:
                print(f"Error encrypting value {value}: {str(e)}")
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

                if isinstance(enc_value, dict) and 'ciphertext' in enc_value:
                    enc_bytes = bytes.fromhex(enc_value['ciphertext'])
                elif isinstance(enc_value, bytes):
                    enc_bytes = enc_value
                elif isinstance(enc_value, str):
                    enc_bytes = bytes.fromhex(enc_value)
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

                # Deserialize
                if self.scheme == 'CKKS':
                    vec1 = ts.ckks_vector_from(self.context, enc1)
                    vec2 = ts.ckks_vector_from(self.context, enc2)
                else:
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
        Supports: sum, average, min, max
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

                if isinstance(enc, dict) and 'ciphertext' in enc:
                    enc_bytes = bytes.fromhex(enc['ciphertext'])
                elif isinstance(enc, bytes):
                    enc_bytes = enc
                else:
                    continue

                if self.scheme == 'CKKS':
                    vec = ts.ckks_vector_from(self.context, enc_bytes)
                else:
                    vec = ts.bfv_vector_from(self.context, enc_bytes)

                valid_data.append(vec)

            if not valid_data:
                return None

            # Perform aggregation
            if operation == 'sum' or operation == 'add':
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec
                return result.serialize()

            elif operation == 'average' or operation == 'avg':
                # Sum all values
                result = valid_data[0]
                for vec in valid_data[1:]:
                    result = result + vec

                # Divide by count (not fully homomorphic, but demonstrates approach)
                # In production, would return encrypted sum and count separately
                count = len(valid_data)
                result = result * (1.0 / count)  # Note: This is not secure in BFV
                # For demo, return sum (client would decrypt and divide)
                return result.serialize()

            elif operation == 'min':
                # MIN/MAX not directly supported in FHE
                # Would require comparison circuits or approximation
                raise ValueError(f"MIN operation requires comparison circuits not available in basic CKKS/BFV")

            elif operation == 'max':
                raise ValueError(f"MAX operation requires comparison circuits not available in basic CKKS/BFV")

            else:
                raise ValueError(f"Unsupported aggregation: {operation}")

        except Exception as e:
            print(f"Error in aggregation '{operation}': {str(e)}")
            return None


    def get_supported_operations(self) -> dict[str, dict[str, list[str]]]:
        """Get list of supported operations per scheme"""
        return {
            'CKKS': {
                'binary': ['add', 'subtract', 'multiply'],
                'aggregation': ['sum', 'average'],
                'unsupported': ['min', 'max', 'comparison'],
                'data_types': ['numeric', 'text (encoded)', 'date (encoded)']
            },
            'BFV': {
                'binary': ['add', 'subtract', 'multiply'],
                'aggregation': ['sum'],
                'unsupported': ['average (requires division)', 'min', 'max', 'comparison'],
                'data_types': ['integer only']
            }
        }

    def get_scheme_limitations(self) -> Dict[str, Any]:
        """Get limitations of the current scheme"""
        limitations = {
            'CKKS': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (approximate)',
                'supports_comparison': 'No',
                'precision': 'Approximate (floating point)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Sum', 'Average'],
                'advantages': ['Supports floating point', 'Efficient for ML'],
                'disadvantages': ['Approximate results', 'No exact integers', 'No min/max']
            },
            'BFV': {
                'supports_text': 'No (integers only)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'No',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Sum'],
                'advantages': ['Exact arithmetic', 'Faster for integers'],
                'disadvantages': ['No floating point', 'No average', 'No min/max']
            }
        }
        return limitations.get(self.scheme, {})

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