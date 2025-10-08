import tenseal as ts
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import base64


class TenSEALWrapper:
    """Wrapper for TenSEAL FHE operations"""

    def __init__(self):
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}

    def generate_context(self, scheme='CKKS', poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2 ** 40):
        """Generate TenSEAL context with specified scheme and parameters"""
        self.scheme = scheme
        self.params = {
            'poly_modulus_degree': poly_modulus_degree,
            'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
            'scale': scale
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
                plain_modulus=1032193
            )
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()

        return self.context

    def get_keys_info(self):
        """Get information about generated keys"""
        if not self.context:
            return None

        # Serialize keys to base64 for display
        public_key = base64.b64encode(self.context.serialize(save_public_key=True, save_secret_key=False)).decode(
            'utf-8')
        private_key = base64.b64encode(self.context.serialize(save_public_key=False, save_secret_key=True)).decode(
            'utf-8')
        galois_keys = "Generated (embedded in context)"
        relin_keys = "Generated (embedded in context)"

        return {
            'public_key': public_key[:100] + '...',  # Truncate for display
            'private_key': private_key[:100] + '...',
            'galois_keys': galois_keys,
            'relin_keys': relin_keys,
            'full_public_key': public_key,
            'full_private_key': private_key
        }

    def encrypt_data(self, data, column_name, data_type):
        """Encrypt data based on type"""
        if self.scheme == 'BFV' and data_type == 'text':
            raise ValueError(
                "BFV scheme does not support text encryption directly. Use CKKS or encode text as integers.")

        encrypted_values = []

        for value in data:
            try:
                if data_type == 'numeric':
                    # Encrypt numeric data
                    if pd.isna(value):
                        encrypted_values.append(None)
                    else:
                        encrypted_val = ts.ckks_vector(self.context, [float(value)])
                        encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'text':
                    # For text, convert to numeric encoding (ASCII sum or hash)
                    if pd.isna(value):
                        encrypted_values.append(None)
                    else:
                        # Simple encoding: sum of ASCII values
                        numeric_value = sum([ord(c) for c in str(value)])
                        encrypted_val = ts.ckks_vector(self.context, [float(numeric_value)])
                        encrypted_values.append(encrypted_val.serialize())

                elif data_type == 'date':
                    # Convert date to timestamp
                    if pd.isna(value):
                        encrypted_values.append(None)
                    else:
                        timestamp = pd.Timestamp(value).timestamp()
                        encrypted_val = ts.ckks_vector(self.context, [float(timestamp)])
                        encrypted_values.append(encrypted_val.serialize())

            except Exception as e:
                print(f"Error encrypting value {value}: {str(e)}")
                encrypted_values.append(None)

        return encrypted_values

    def decrypt_data(self, encrypted_data, data_type):
        """Decrypt data"""
        if not self.context:
            raise ValueError("Context not initialized")

        decrypted_values = []

        for enc_value in encrypted_data:
            try:
                if enc_value is None:
                    decrypted_values.append(None)
                else:
                    # Deserialize and decrypt
                    if self.scheme == 'CKKS':
                        vector = ts.ckks_vector_from(self.context, enc_value)
                    else:
                        vector = ts.bfv_vector_from(self.context, enc_value)

                    decrypted = vector.decrypt()[0]

                    if data_type == 'date':
                        # Convert timestamp back to date
                        decrypted_values.append(pd.Timestamp.fromtimestamp(decrypted))
                    else:
                        decrypted_values.append(decrypted)

            except Exception as e:
                print(f"Error decrypting value: {str(e)}")
                decrypted_values.append(None)

        return decrypted_values

    def perform_operation(self, encrypted_data1, encrypted_data2, operation):
        """Perform homomorphic operations"""
        results = []

        for enc1, enc2 in zip(encrypted_data1, encrypted_data2):
            try:
                if enc1 is None or enc2 is None:
                    results.append(None)
                    continue

                if self.scheme == 'CKKS':
                    vec1 = ts.ckks_vector_from(self.context, enc1)
                    vec2 = ts.ckks_vector_from(self.context, enc2)
                else:
                    vec1 = ts.bfv_vector_from(self.context, enc1)
                    vec2 = ts.bfv_vector_from(self.context, enc2)

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

    def get_scheme_limitations(self):
        """Get limitations of the current scheme"""
        limitations = {
            'CKKS': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (approximate)',
                'supports_comparison': 'No',
                'precision': 'Approximate (floating point)',
                'operations': ['Addition', 'Multiplication', 'Subtraction']
            },
            'BFV': {
                'supports_text': 'No (integers only)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'Limited',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction']
            }
        }
        return limitations.get(self.scheme, {})

    def rotate_keys(self):
        """Rotate encryption keys while maintaining backward compatibility"""
        if not self.context:
            raise ValueError("Context not initialized")

        # Save old context
        old_context_serial = self.context.serialize()

        # Generate new context with same parameters
        new_context = self.generate_context(
            scheme=self.scheme,
            **self.params
        )

        return {
            'old_context': old_context_serial,
            'new_context': new_context.serialize(),
            'rotation_time': datetime.now().isoformat()
        }