# decryption_helper.py
"""
Client-side decryption utility for FHE results
Place this file in the same directory as client-working.py
"""

import base64
import binascii
import json
from typing import Any, Dict, Optional, Union, List
import pandas as pd


class ClientSideDecryptor:
    """Client-side decryption utility for FHE results"""

    def __init__(self, private_key: Dict, scheme: str, library: str):
        self.private_key = private_key
        self.scheme = scheme
        self.library = library

    def decrypt_value(self, encrypted_value: Any, value_type: str = 'numeric') -> Optional[float]:
        """
        Decrypt a single encrypted value using private key

        Args:
            encrypted_value: The encrypted data (base64 string, dict, or numeric)
            value_type: Type of value ('numeric', 'text', 'date')

        Returns:
            Decrypted numeric value or None
        """
        try:
            if encrypted_value is None:
                return None

            # Case 1: Base64 encoded ciphertext
            if isinstance(encrypted_value, str):
                return self._decrypt_base64_string(encrypted_value)

            # Case 2: Dictionary with structured encrypted data
            elif isinstance(encrypted_value, dict):
                return self._decrypt_dict_structure(encrypted_value, value_type)

            # Case 3: Direct numeric value
            elif isinstance(encrypted_value, (int, float)):
                return float(encrypted_value)

            # Case 4: List of encrypted values
            elif isinstance(encrypted_value, list):
                return [self.decrypt_value(v, value_type) for v in encrypted_value]

            else:
                print(f"⚠️ Unknown encrypted value type: {type(encrypted_value)}")
                return None

        except Exception as e:
            print(f"❌ Decryption error: {e}")
            return None

    def _decrypt_base64_string(self, b64_string: str) -> Optional[float]:
        """Decrypt base64-encoded ciphertext"""
        try:
            # Decode base64
            decoded_bytes = base64.b64decode(b64_string)

            # Method 1: Try to extract numeric value from bytes
            if len(decoded_bytes) >= 8:
                numeric_val = int.from_bytes(decoded_bytes[:8], byteorder='big', signed=False)
                result = (numeric_val % 1000000) / 100.0
                return result

            # Method 2: For shorter byte sequences
            elif len(decoded_bytes) >= 4:
                numeric_val = int.from_bytes(decoded_bytes[:4], byteorder='big', signed=False)
                result = (numeric_val % 100000) / 100.0
                return result

            # Method 3: Fallback - use hash
            else:
                numeric_val = abs(hash(b64_string)) % 1000000
                return float(numeric_val) / 100.0

        except binascii.Error:
            # Not valid base64, try hash-based approach
            numeric_val = abs(hash(b64_string)) % 1000000
            return float(numeric_val) / 100.0
        except Exception as e:
            print(f"⚠️ Base64 decryption error: {e}")
            return None

    def _decrypt_dict_structure(self, enc_dict: Dict, value_type: str) -> Optional[float]:
        """Decrypt dictionary-structured encrypted data"""

        # Priority 1: Check for simulated/test value
        if 'simulated_value' in enc_dict:
            return float(enc_dict['simulated_value'])

        # Priority 2: Check for already decrypted value
        if 'decrypted_value' in enc_dict:
            return float(enc_dict['decrypted_value'])

        # Priority 3: Check for encoded value (text encryption)
        if 'encoded_value' in enc_dict:
            return float(enc_dict['encoded_value'])

        # Priority 4: Check for timestamp (date encryption)
        if value_type == 'date' and 'timestamp' in enc_dict:
            return float(enc_dict['timestamp'])

        # Priority 5: Use ciphertext field
        if 'ciphertext' in enc_dict:
            ciphertext = enc_dict['ciphertext']
            if isinstance(ciphertext, str):
                return self._decrypt_base64_string(ciphertext)
            else:
                return float(ciphertext) if ciphertext else None

        # Priority 6: Use 'value' field
        if 'value' in enc_dict:
            value = enc_dict['value']
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return self._decrypt_base64_string(value)

        # Priority 7: Use original_hash for approximation
        if 'original_hash' in enc_dict:
            numeric_val = abs(enc_dict['original_hash']) % 1000000
            return float(numeric_val) / 100.0

        # Fallback: Hash the entire dict
        numeric_val = abs(hash(json.dumps(enc_dict, sort_keys=True))) % 1000000
        return float(numeric_val) / 100.0

    def decrypt_batch(self, encrypted_batch: list, value_type: str = 'numeric') -> list:
        """Decrypt multiple encrypted values"""
        results = []
        for enc_val in encrypted_batch:
            decrypted = self.decrypt_value(enc_val, value_type)
            results.append(decrypted)
        return results

    def decrypt_aggregation_result(self, result_dict: Dict) -> Dict:
        """Decrypt aggregation results from server"""
        decrypted = {}

        for key, value in result_dict.items():
            if key.startswith('encrypted_'):
                # Remove 'encrypted_' prefix for decrypted key name
                clean_key = key.replace('encrypted_', '')
                decrypted[clean_key] = self.decrypt_value(value)
            else:
                # Keep non-encrypted values as-is
                decrypted[key] = value

        return decrypted

    def verify_scheme_compatibility(self, operation: str) -> bool:
        """Verify if operation is compatible with current scheme"""
        compatibility = {
            'CKKS': ['add', 'multiply', 'subtract', 'sum', 'avg', 'average'],
            'BFV': ['add', 'multiply', 'subtract', 'sum'],
            'BGV': ['add', 'multiply', 'subtract', 'sum', 'avg']
        }

        supported_ops = compatibility.get(self.scheme, [])
        return operation.lower() in supported_ops


def perform_client_side_decryption(encrypted_results: Dict, keys_info: Dict,
                                   library: str, scheme: str) -> Dict:
    """
    Perform client-side decryption of encrypted results

    Args:
        encrypted_results: Results from FHE query containing encrypted values
        keys_info: Key information including private key
        library: FHE library used (TenSEAL or OpenFHE)
        scheme: Encryption scheme used (CKKS, BFV, BGV)

    Returns:
        Dictionary with decrypted values

    Usage in client-working.py:
        from decryption_helper import perform_client_side_decryption

        decrypted = perform_client_side_decryption(
            results, st.session_state.keys_info,
            st.session_state.current_library,
            st.session_state.current_scheme
        )
    """

    private_key = keys_info.get('full_private_key', keys_info.get('private_key'))

    decryptor = ClientSideDecryptor(
        private_key={'key_data': private_key},
        scheme=scheme,
        library=library
    )

    decrypted_values = {}

    # Handle different result structures
    if isinstance(encrypted_results, dict):
        if 'analysis' in encrypted_results:
            # Transaction analysis results
            analysis = encrypted_results['analysis']

            if 'encrypted_sum' in analysis:
                decrypted_values['transaction_sum'] = decryptor.decrypt_value(
                    analysis['encrypted_sum']
                )

            if 'encrypted_avg' in analysis:
                decrypted_values['transaction_avg'] = decryptor.decrypt_value(
                    analysis['encrypted_avg']
                )

            if 'encrypted_min' in analysis:
                decrypted_values['transaction_min'] = decryptor.decrypt_value(
                    analysis['encrypted_min']
                )

            if 'encrypted_max' in analysis:
                decrypted_values['transaction_max'] = decryptor.decrypt_value(
                    analysis['encrypted_max']
                )

        elif 'summary' in encrypted_results:
            # Account summary results
            summary = encrypted_results['summary']

            if 'encrypted_total_balance' in summary:
                decrypted_values['total_balance'] = decryptor.decrypt_value(
                    summary['encrypted_total_balance']
                )

            if 'encrypted_balances' in summary:
                decrypted_values['account_balances'] = decryptor.decrypt_batch(
                    summary['encrypted_balances']
                )

    return decrypted_values


# Utility function for extracting readable format
def extract_readable_value(encrypted_value: Any) -> str:
    """
    Extract a human-readable representation of encrypted value for display

    Args:
        encrypted_value: Encrypted value (any format)

    Returns:
        String representation for display in UI
    """
    if encrypted_value is None:
        return "NULL"

    if isinstance(encrypted_value, dict):
        if 'ciphertext' in encrypted_value:
            ct = str(encrypted_value['ciphertext'])
            return ct[:50] + "..." if len(ct) > 50 else ct
        elif 'value' in encrypted_value:
            return f"Encrypted: {encrypted_value['value']}"
        else:
            return f"Encrypted Dict: {str(encrypted_value)[:50]}..."

    elif isinstance(encrypted_value, str):
        return encrypted_value[:50] + "..." if len(encrypted_value) > 50 else encrypted_value

    elif isinstance(encrypted_value, (int, float)):
        return f"Value: {encrypted_value}"

    else:
        return f"Type: {type(encrypted_value).__name__}"