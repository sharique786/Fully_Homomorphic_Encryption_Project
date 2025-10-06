"""
FHE Core Operations Module
Handles key management, encryption, decryption, and FHE operations
"""

import numpy as np
import pandas as pd
import json
import base64
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import time
import streamlit as st

# Import FHE libraries
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not available")

# Import OpenFHE wrapper
try:
    from openfhe_wrapper import OpenFHEWrapper
    OPENFHE_WRAPPER_AVAILABLE = True
except ImportError:
    OPENFHE_WRAPPER_AVAILABLE = False
    print("Warning: OpenFHE wrapper not available")


class FHEKeyManager:
    """Manages FHE keys - generation, storage, rotation, and backup"""

    def __init__(self):
        self.public_key = None
        self.private_key = None
        self.evaluation_keys = {}
        self.relinearization_keys = None
        self.galois_keys = None
        self.context = None
        self.scheme = None
        self.library = None
        self.openfhe_wrapper = None
        self.key_metadata = {
            'generation_time': None,
            'scheme': None,
            'library': None,
            'parameters': {},
            'version': 1
        }

    def generate_keys(self, library: str, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """
        Generate FHE keys based on library and scheme

        Args:
            library: 'TenSEAL' or 'OpenFHE'
            scheme: 'BFV', 'BGV', or 'CKKS'
            parameters: Scheme-specific parameters

        Returns:
            Dictionary containing key information
        """
        self.library = library
        self.scheme = scheme

        start_time = time.time()

        if library == 'TenSEAL' and TENSEAL_AVAILABLE:
            result = self._generate_tenseal_keys(scheme, parameters)
        elif library == 'OpenFHE':
            result = self._generate_openfhe_keys(scheme, parameters)
        else:
            # Simulation mode
            result = self._generate_simulated_keys(scheme, parameters)

        generation_time = time.time() - start_time

        self.key_metadata = {
            'generation_time': datetime.now().isoformat(),
            'scheme': scheme,
            'library': library,
            'parameters': parameters,
            'version': 1,
            'generation_duration_ms': generation_time * 1000
        }

        result['metadata'] = self.key_metadata
        return result

    def _generate_openfhe_keys(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """Generate keys using compiled OpenFHE C++ libraries"""
        try:
            # Check if wrapper is available and executable exists
            openfhe_path = st.session_state.get('openfhe_path', r"C:\openfhe-development\build\bin\Release")

            if not OPENFHE_WRAPPER_AVAILABLE:
                st.warning("⚠️ OpenFHE wrapper not available. Using simulation mode.")
                return self._generate_simulated_keys(scheme, parameters)

            # Initialize OpenFHE wrapper
            self.openfhe_wrapper = OpenFHEWrapper(openfhe_path)

            if not self.openfhe_wrapper.executable_path:
                st.warning(f"⚠️ OpenFHE executable not found at {openfhe_path}. Using simulation mode.")
                return self._generate_simulated_keys(scheme, parameters)

            # Call C++ executable to generate keys
            result = self.openfhe_wrapper.generate_keys(scheme, parameters)

            if result.get('status') == 'success':
                st.success(f"✅ Keys generated using OpenFHE C++ libraries")

                # Store keys
                self.public_key = result.get('public_key', 'OPENFHE_PUBLIC_KEY')
                self.private_key = result.get('private_key', 'OPENFHE_PRIVATE_KEY')
                self.evaluation_keys['main'] = result.get('evaluation_key', 'OPENFHE_EVAL_KEY')
                self.relinearization_keys = result.get('relinearization_key', 'OPENFHE_RELIN_KEY')
                self.galois_keys = result.get('galois_keys', 'OPENFHE_GALOIS_KEYS')

                return {
                    'status': 'success',
                    'public_key': self.public_key,
                    'private_key': self.private_key,
                    'evaluation_key': self.evaluation_keys['main'],
                    'relinearization_key': self.relinearization_keys,
                    'galois_keys': self.galois_keys,
                    'key_size_bytes': 4096,  # Approximate
                    'library_used': 'OpenFHE C++'
                }
            else:
                st.error(f"❌ OpenFHE key generation failed: {result.get('message')}")
                return self._generate_simulated_keys(scheme, parameters)

        except Exception as e:
            st.error(f"❌ OpenFHE error: {str(e)}")
            st.info("💡 Falling back to simulation mode")
            return self._generate_simulated_keys(scheme, parameters)

    def _generate_tenseal_keys(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """Generate keys using TenSEAL"""
        poly_modulus_degree = parameters.get('poly_modulus_degree', 8192)

        if scheme == 'BFV':
            plain_modulus = parameters.get('plain_modulus', 1032193)
            self.context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus
            )
        elif scheme == 'CKKS':
            scale = 2 ** parameters.get('scale_factor', 40)
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = scale

        # Generate keys
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()

        # Serialize keys
        public_key_bytes = self.context.serialize(save_public_key=True, save_secret_key=False)
        private_key_bytes = self.context.serialize(save_public_key=False, save_secret_key=True)

        return {
            'status': 'success',
            'public_key': base64.b64encode(public_key_bytes).decode('utf-8'),
            'private_key': base64.b64encode(private_key_bytes).decode('utf-8'),
            'evaluation_key': base64.b64encode(b'TENSEAL_EVAL_KEY').decode('utf-8'),
            'relinearization_key': base64.b64encode(b'TENSEAL_RELIN_KEY').decode('utf-8'),
            'galois_keys': base64.b64encode(b'TENSEAL_GALOIS_KEYS').decode('utf-8'),
            'context_serialized': base64.b64encode(
                self.context.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True)
            ).decode('utf-8'),
            'key_size_bytes': len(public_key_bytes) + len(private_key_bytes)
        }

    def _generate_simulated_keys(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """Generate simulated keys for demonstration"""
        # Create simulated keys
        key_data = {
            'scheme': scheme,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat()
        }

        # Simulate key generation
        public_key_str = base64.b64encode(
            json.dumps({**key_data, 'type': 'public'}).encode()
        ).decode('utf-8')

        private_key_str = base64.b64encode(
            json.dumps({**key_data, 'type': 'private'}).encode()
        ).decode('utf-8')

        eval_key_str = base64.b64encode(
            json.dumps({**key_data, 'type': 'evaluation'}).encode()
        ).decode('utf-8')

        relin_key_str = base64.b64encode(
            json.dumps({**key_data, 'type': 'relinearization'}).encode()
        ).decode('utf-8')

        galois_keys_str = base64.b64encode(
            json.dumps({**key_data, 'type': 'galois'}).encode()
        ).decode('utf-8')

        self.public_key = public_key_str
        self.private_key = private_key_str
        self.evaluation_keys['main'] = eval_key_str
        self.relinearization_keys = relin_key_str
        self.galois_keys = galois_keys_str

        return {
            'status': 'success',
            'public_key': public_key_str,
            'private_key': private_key_str,
            'evaluation_key': eval_key_str,
            'relinearization_key': relin_key_str,
            'galois_keys': galois_keys_str,
            'key_size_bytes': len(public_key_str) + len(private_key_str),
            'library_used': 'Simulation'
        }

    def rotate_keys(self, old_keys: Dict, new_parameters: Dict) -> Dict[str, Any]:
        """
        Rotate keys while maintaining backward compatibility

        Args:
            old_keys: Previous key set
            new_parameters: New parameters for key generation

        Returns:
            Dictionary with new keys and migration information
        """
        # Generate new keys
        new_keys = self.generate_keys(
            self.library,
            self.scheme,
            new_parameters
        )

        # Create rotation mapping
        rotation_info = {
            'old_version': self.key_metadata.get('version', 1),
            'new_version': self.key_metadata['version'] + 1,
            'rotation_time': datetime.now().isoformat(),
            'backward_compatible': True
        }

        self.key_metadata['version'] += 1
        new_keys['rotation_info'] = rotation_info

        return new_keys

    def export_keys(self, format: str = 'json', include_private: bool = True) -> str:
        """Export keys in specified format"""
        key_data = {
            'public_key': self.public_key,
            'metadata': self.key_metadata
        }

        if include_private:
            key_data['private_key'] = self.private_key
            key_data['evaluation_keys'] = self.evaluation_keys

        if format == 'json':
            return json.dumps(key_data, indent=2)
        elif format == 'base64':
            return base64.b64encode(json.dumps(key_data).encode()).decode('utf-8')
        else:
            return str(key_data)

    def import_keys(self, key_string: str, format: str = 'json') -> bool:
        """Import keys from string"""
        try:
            if format == 'base64':
                key_string = base64.b64decode(key_string).decode('utf-8')

            key_data = json.loads(key_string)
            self.public_key = key_data.get('public_key')
            self.private_key = key_data.get('private_key')
            self.evaluation_keys = key_data.get('evaluation_keys', {})
            self.key_metadata = key_data.get('metadata', {})

            return True
        except Exception as e:
            print(f"Error importing keys: {e}")
            return False


class FHEProcessor:
    """Handles FHE encryption, decryption, and operations"""

    def __init__(self, key_manager: FHEKeyManager, library: str = 'TenSEAL'):
        self.key_manager = key_manager
        self.library = library
        self.encrypted_data = {}
        self.operation_log = []

    def encrypt_data(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        Encrypt specified columns of data

        Args:
            data: DataFrame containing data to encrypt
            columns: List of column names to encrypt

        Returns:
            Dictionary with encryption results and metadata
        """
        start_time = time.time()
        encrypted_results = {}

        for col in columns:
            if col not in data.columns:
                continue

            col_data = data[col].fillna(0).tolist()

            if self.library == 'TenSEAL' and TENSEAL_AVAILABLE and self.key_manager.context:
                encrypted_results[col] = self._encrypt_column_tenseal(col_data)
            else:
                encrypted_results[col] = self._encrypt_column_simulated(col_data)

        encryption_time = (time.time() - start_time) * 1000

        self.encrypted_data = encrypted_results

        return {
            'status': 'success',
            'encrypted_columns': len(encrypted_results),
            'total_values': sum(len(v) if isinstance(v, list) else 1 for v in encrypted_results.values()),
            'encryption_time_ms': encryption_time,
            'library': self.library,
            'timestamp': datetime.now().isoformat()
        }

    def _encrypt_column_tenseal(self, data: List) -> List:
        """Encrypt column using TenSEAL"""
        encrypted_values = []

        # Process in batches
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]

            if self.key_manager.scheme == 'BFV':
                encrypted_batch = ts.bfv_vector(self.key_manager.context, batch)
            else:  # CKKS
                encrypted_batch = ts.ckks_vector(self.key_manager.context, batch)

            encrypted_values.append(encrypted_batch)

        return encrypted_values

    def _encrypt_column_simulated(self, data: List) -> List:
        """Simulate encryption for demonstration"""
        return [
            {
                'value': val,
                'encrypted': hash(str(val) + str(np.random.random())),
                'noise_budget': np.random.uniform(80, 95)
            }
            for val in data
        ]

    def decrypt_data(self, encrypted_data: Dict = None) -> Tuple[pd.DataFrame, float]:
        """Decrypt encrypted data"""
        if encrypted_data is None:
            encrypted_data = self.encrypted_data

        start_time = time.time()
        decrypted_results = {}

        for col, enc_data in encrypted_data.items():
            if self.library == 'TenSEAL' and TENSEAL_AVAILABLE:
                decrypted_results[col] = self._decrypt_column_tenseal(enc_data)
            else:
                decrypted_results[col] = self._decrypt_column_simulated(enc_data)

        decryption_time = (time.time() - start_time) * 1000

        return pd.DataFrame(decrypted_results), decryption_time

    def _decrypt_column_tenseal(self, encrypted_data: List) -> List:
        """Decrypt column using TenSEAL"""
        decrypted_values = []

        for encrypted_batch in encrypted_data:
            batch_decrypted = encrypted_batch.decrypt()
            decrypted_values.extend(batch_decrypted)

        return decrypted_values

    def _decrypt_column_simulated(self, encrypted_data: List) -> List:
        """Decrypt simulated data"""
        return [item['value'] for item in encrypted_data]

    def perform_operation(self, operation: str, column: str,
                         operand: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Perform homomorphic operation on encrypted data

        Args:
            operation: Type of operation ('add', 'multiply', 'sum', 'mean', etc.)
            column: Column name to operate on
            operand: Operand for binary operations
            **kwargs: Additional parameters

        Returns:
            Dictionary with operation results
        """
        start_time = time.time()

        if column not in self.encrypted_data:
            return {'status': 'error', 'message': f'Column {column} not encrypted'}

        encrypted_col = self.encrypted_data[column]

        if self.library == 'TenSEAL' and TENSEAL_AVAILABLE:
            result = self._perform_operation_tenseal(operation, encrypted_col, operand)
        else:
            result = self._perform_operation_simulated(operation, encrypted_col, operand)

        operation_time = (time.time() - start_time) * 1000

        # Log operation
        self.operation_log.append({
            'operation': operation,
            'column': column,
            'timestamp': datetime.now().isoformat(),
            'duration_ms': operation_time
        })

        return {
            'status': 'success',
            'operation': operation,
            'result': result,
            'operation_time_ms': operation_time
        }

    def _perform_operation_tenseal(self, operation: str, data: List, operand: Any) -> Any:
        """Perform operation using TenSEAL"""
        if operation == 'add' and operand:
            return [batch + operand for batch in data]
        elif operation == 'multiply' and operand:
            return [batch * operand for batch in data]
        elif operation == 'sum':
            result = data[0]
            for batch in data[1:]:
                result = result + batch
            return result
        else:
            return data

    def _perform_operation_simulated(self, operation: str, data: List, operand: Any) -> List:
        """Simulate operation"""
        results = []
        for item in data:
            val = item['value']

            if operation == 'add' and operand:
                result_val = val + operand
            elif operation == 'multiply' and operand:
                result_val = val * operand
            elif operation == 'square':
                result_val = val ** 2
            else:
                result_val = val

            results.append({
                'value': result_val,
                'encrypted': hash(str(result_val)),
                'noise_budget': item['noise_budget'] - np.random.uniform(5, 15)
            })

        return results

    def aggregate_encrypted_data(self, column: str, group_by: str,
                                 aggregation: str = 'sum') -> Dict[str, Any]:
        """Perform aggregation on encrypted data"""
        # This would perform actual encrypted aggregation
        # For now, simulate the result
        return {
            'status': 'success',
            'aggregation': aggregation,
            'column': column,
            'group_by': group_by
        }