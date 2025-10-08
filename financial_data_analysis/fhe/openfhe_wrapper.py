import ctypes
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json


class OpenFHEWrapper:
    """Python wrapper for OpenFHE C++ library"""

    def __init__(self):
        self.openfhe_path = r"C:\Program Files (x86)\OpenFHE"
        self.build_path = r"C:\Users\alish\Workspaces\Python\openfhe-development"
        self.lib = None
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}

        # Try to load the library
        self._load_library()

    def _load_library(self):
        """Load OpenFHE shared library"""
        try:
            # Try different possible library locations
            possible_paths = [
                os.path.join(self.openfhe_path, "lib", "libOPENFHEcore.dll"),
                os.path.join(self.build_path, "lib", "Release", "libOPENFHEcore.dll"),
                os.path.join(self.build_path, "lib", "Debug", "libOPENFHEcore.dll"),
                os.path.join(self.openfhe_path, "lib", "libOPENFHEcore.so"),  # Linux
                os.path.join(self.build_path, "lib", "libOPENFHEcore.so"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.lib = ctypes.CDLL(path)
                    print(f"✅ OpenFHE library loaded from: {path}")
                    return True

            print("⚠️ Warning: OpenFHE library not found. Using simulation mode.")
            return False

        except Exception as e:
            print(f"⚠️ Error loading OpenFHE library: {str(e)}")
            print("Running in simulation mode.")
            return False

    def generate_context(self, scheme='CKKS', mult_depth=10, scale_mod_size=50,
                         batch_size=8, security_level='HEStd_128_classic',
                         ring_dim=16384):
        """Generate OpenFHE context with specified scheme and parameters"""
        self.scheme = scheme
        self.params = {
            'mult_depth': mult_depth,
            'scale_mod_size': scale_mod_size,
            'batch_size': batch_size,
            'security_level': security_level,
            'ring_dim': ring_dim
        }

        # Simulation mode - generate mock context
        self.context = {
            'scheme': scheme,
            'params': self.params,
            'initialized': True,
            'timestamp': datetime.now().isoformat()
        }

        # Generate mock keys
        self._generate_keys()

        print(f"✅ Context generated for {scheme} scheme")
        return self.context

    def _generate_keys(self):
        """Generate encryption keys (simulated)"""
        import secrets

        # Generate mock keys for demonstration
        self.public_key = {
            'key_data': secrets.token_hex(128),
            'scheme': self.scheme,
            'params': self.params
        }

        self.private_key = {
            'key_data': secrets.token_hex(128),
            'scheme': self.scheme,
            'params': self.params
        }

        self.evaluation_key = {
            'mult_key': secrets.token_hex(64),
            'rotation_keys': secrets.token_hex(64),
            'scheme': self.scheme
        }

        print("✅ Keys generated successfully")

    def get_keys_info(self):
        """Get information about generated keys"""
        if not self.context:
            return None

        return {
            'public_key': self.public_key['key_data'][:100] + '...',
            'private_key': self.private_key['key_data'][:100] + '...',
            'evaluation_key': self.evaluation_key['mult_key'][:100] + '...',
            'rotation_keys': self.evaluation_key['rotation_keys'][:100] + '...',
            'full_public_key': self.public_key['key_data'],
            'full_private_key': self.private_key['key_data'],
            'full_evaluation_key': json.dumps(self.evaluation_key)
        }

    def encrypt_data(self, data, column_name, data_type):
        """Encrypt data based on type (simulated)"""
        if not self.context:
            raise ValueError("Context not initialized")

        encrypted_values = []

        for value in data:
            try:
                if pd.isna(value):
                    encrypted_values.append(None)
                    continue

                if data_type == 'numeric':
                    # Simulate encryption
                    encrypted_val = {
                        'ciphertext': f"ENC_{hash(str(value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'scheme': self.scheme,
                        'type': 'numeric'
                    }

                elif data_type == 'text':
                    if self.scheme == 'CKKS':
                        # CKKS works better with numeric data
                        numeric_value = sum([ord(c) for c in str(value)])
                        encrypted_val = {
                            'ciphertext': f"ENC_{hash(str(numeric_value)) % 10000000}",
                            'original_hash': hash(str(value)),
                            'encoded_value': numeric_value,
                            'scheme': self.scheme,
                            'type': 'text'
                        }
                    else:
                        encrypted_val = {
                            'ciphertext': f"ENC_{hash(str(value)) % 10000000}",
                            'original_hash': hash(str(value)),
                            'scheme': self.scheme,
                            'type': 'text'
                        }

                elif data_type == 'date':
                    timestamp = pd.Timestamp(value).timestamp()
                    encrypted_val = {
                        'ciphertext': f"ENC_{hash(str(timestamp)) % 10000000}",
                        'original_hash': hash(str(timestamp)),
                        'timestamp': timestamp,
                        'scheme': self.scheme,
                        'type': 'date'
                    }

                encrypted_values.append(encrypted_val)

            except Exception as e:
                print(f"Error encrypting value {value}: {str(e)}")
                encrypted_values.append(None)

        return encrypted_values

    def decrypt_data(self, encrypted_data, data_type):
        """Decrypt data (simulated)"""
        if not self.context:
            raise ValueError("Context not initialized")

        decrypted_values = []

        for enc_value in encrypted_data:
            try:
                if enc_value is None:
                    decrypted_values.append(None)
                    continue

                # In simulation, we'll use the original hash to retrieve stored values
                # In real implementation, this would decrypt the ciphertext
                if data_type == 'date':
                    if 'timestamp' in enc_value:
                        decrypted_values.append(pd.Timestamp.fromtimestamp(enc_value['timestamp']))
                    else:
                        decrypted_values.append(None)
                elif data_type == 'text' and 'encoded_value' in enc_value:
                    decrypted_values.append(enc_value['encoded_value'])
                else:
                    # Return a placeholder value
                    decrypted_values.append(f"DECRYPTED_{enc_value['original_hash'] % 1000}")

            except Exception as e:
                print(f"Error decrypting value: {str(e)}")
                decrypted_values.append(None)

        return decrypted_values

    def perform_operation(self, encrypted_data1, encrypted_data2, operation):
        """Perform homomorphic operations (simulated)"""
        results = []

        for enc1, enc2 in zip(encrypted_data1, encrypted_data2):
            try:
                if enc1 is None or enc2 is None:
                    results.append(None)
                    continue

                # Simulate homomorphic operation
                result = {
                    'ciphertext': f"RESULT_{hash(str(enc1) + str(enc2) + operation) % 10000000}",
                    'operation': operation,
                    'scheme': self.scheme,
                    'operands': [enc1.get('original_hash'), enc2.get('original_hash')]
                }

                results.append(result)

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
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Rotation']
            },
            'BFV': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'Limited',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction']
            },
            'BGV': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'Limited',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Rotation']
            }
        }
        return limitations.get(self.scheme, {})

    def rotate_keys(self):
        """Rotate encryption keys while maintaining backward compatibility"""
        if not self.context:
            raise ValueError("Context not initialized")

        # Save old keys
        old_keys = {
            'public_key': self.public_key,
            'private_key': self.private_key,
            'evaluation_key': self.evaluation_key
        }

        # Generate new keys
        self._generate_keys()

        return {
            'old_keys': old_keys,
            'new_keys': {
                'public_key': self.public_key,
                'private_key': self.private_key,
                'evaluation_key': self.evaluation_key
            },
            'rotation_time': datetime.now().isoformat()
        }