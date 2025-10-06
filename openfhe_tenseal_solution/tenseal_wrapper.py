"""
TenSEAL Wrapper Module
Provides a clean interface for TenSEAL FHE operations
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import base64
import pickle
import time

try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not installed. Install with: pip install tenseal")


class TenSEALWrapper:
    """
    Wrapper class for TenSEAL FHE library
    Provides simplified interface for encryption operations
    """

    def __init__(self, scheme: str = 'BFV', poly_modulus_degree: int = 8192):
        """
        Initialize TenSEAL wrapper

        Args:
            scheme: 'BFV' or 'CKKS'
            poly_modulus_degree: Polynomial modulus degree (power of 2)
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is not installed. Please install it first.")

        self.scheme = scheme
        self.poly_modulus_degree = poly_modulus_degree
        self.context = None
        self.public_context = None
        self.encrypted_vectors = []

    def setup_context(self, **kwargs) -> Dict[str, Any]:
        """
        Setup TenSEAL context with encryption parameters

        Args:
            **kwargs: Additional parameters based on scheme
                For BFV: plain_modulus (int)
                For CKKS: scale (float), coeff_mod_bit_sizes (list)

        Returns:
            Dictionary with setup status
        """
        try:
            if self.scheme == 'BFV':
                plain_modulus = kwargs.get('plain_modulus', 1032193)

                self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=self.poly_modulus_degree,
                    plain_modulus=plain_modulus
                )

            elif self.scheme == 'CKKS':
                scale_factor = kwargs.get('scale_factor', 40)
                scale = 2 ** scale_factor
                coeff_mod_bit_sizes = kwargs.get('coeff_mod_bit_sizes', [60, 40, 40, 60])

                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=self.poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )
                self.context.global_scale = scale

            else:
                return {'status': 'error', 'message': f'Unknown scheme: {self.scheme}'}

            # Generate keys
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()

            # Create public context (without secret key)
            self.public_context = self.context.copy()
            self.public_context.make_context_public()

            return {
                'status': 'success',
                'scheme': self.scheme,
                'poly_modulus_degree': self.poly_modulus_degree,
                'message': 'Context setup successfully'
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def encrypt_vector(self, data: List[Union[int, float]]) -> Optional[Any]:
        """
        Encrypt a vector of numbers

        Args:
            data: List of numbers to encrypt

        Returns:
            Encrypted vector or None on error
        """
        if not self.context:
            raise RuntimeError("Context not initialized. Call setup_context() first.")

        try:
            if self.scheme == 'BFV':
                # BFV for integer arithmetic
                encrypted = ts.bfv_vector(self.context, data)
            else:  # CKKS
                # CKKS for approximate arithmetic
                encrypted = ts.ckks_vector(self.context, data)

            self.encrypted_vectors.append(encrypted)
            return encrypted

        except Exception as e:
            print(f"Encryption error: {e}")
            return None

    def decrypt_vector(self, encrypted_vector: Any) -> List[Union[int, float]]:
        """
        Decrypt an encrypted vector

        Args:
            encrypted_vector: Encrypted vector

        Returns:
            List of decrypted values
        """
        if not self.context:
            raise RuntimeError("Context not initialized.")

        try:
            decrypted = encrypted_vector.decrypt()
            return decrypted
        except Exception as e:
            print(f"Decryption error: {e}")
            return []

    def add(self, enc_vector1: Any, enc_vector2: Any) -> Any:
        """
        Homomorphic addition of two encrypted vectors

        Args:
            enc_vector1: First encrypted vector
            enc_vector2: Second encrypted vector

        Returns:
            Encrypted result of addition
        """
        try:
            return enc_vector1 + enc_vector2
        except Exception as e:
            print(f"Addition error: {e}")
            return None

    def add_plain(self, enc_vector: Any, plain_value: Union[int, float, List]) -> Any:
        """
        Add a plaintext value/vector to encrypted vector

        Args:
            enc_vector: Encrypted vector
            plain_value: Plain value or list to add

        Returns:
            Encrypted result
        """
        try:
            return enc_vector + plain_value
        except Exception as e:
            print(f"Plain addition error: {e}")
            return None

    def multiply(self, enc_vector1: Any, enc_vector2: Any) -> Any:
        """
        Homomorphic multiplication of two encrypted vectors

        Args:
            enc_vector1: First encrypted vector
            enc_vector2: Second encrypted vector

        Returns:
            Encrypted result of multiplication
        """
        try:
            return enc_vector1 * enc_vector2
        except Exception as e:
            print(f"Multiplication error: {e}")
            return None

    def multiply_plain(self, enc_vector: Any, plain_value: Union[int, float, List]) -> Any:
        """
        Multiply encrypted vector by plaintext value/vector

        Args:
            enc_vector: Encrypted vector
            plain_value: Plain value or list to multiply

        Returns:
            Encrypted result
        """
        try:
            return enc_vector * plain_value
        except Exception as e:
            print(f"Plain multiplication error: {e}")
            return None

    def dot_product(self, enc_vector1: Any, enc_vector2: Any) -> Any:
        """
        Compute dot product of two encrypted vectors

        Args:
            enc_vector1: First encrypted vector
            enc_vector2: Second encrypted vector

        Returns:
            Encrypted dot product result
        """
        try:
            return enc_vector1.dot(enc_vector2)
        except Exception as e:
            print(f"Dot product error: {e}")
            return None

    def polynomial_evaluation(self, enc_vector: Any, coefficients: List[float]) -> Any:
        """
        Evaluate polynomial on encrypted data

        Args:
            enc_vector: Encrypted vector
            coefficients: Polynomial coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...

        Returns:
            Encrypted result of polynomial evaluation
        """
        try:
            return enc_vector.polyval(coefficients)
        except Exception as e:
            print(f"Polynomial evaluation error: {e}")
            return None

    def negate(self, enc_vector: Any) -> Any:
        """
        Negate an encrypted vector

        Args:
            enc_vector: Encrypted vector

        Returns:
            Negated encrypted vector
        """
        try:
            return -enc_vector
        except Exception as e:
            print(f"Negation error: {e}")
            return None

    def square(self, enc_vector: Any) -> Any:
        """
        Square an encrypted vector (element-wise)

        Args:
            enc_vector: Encrypted vector

        Returns:
            Encrypted squared vector
        """
        try:
            return enc_vector * enc_vector
        except Exception as e:
            print(f"Square error: {e}")
            return None

    def power(self, enc_vector: Any, exponent: int) -> Any:
        """
        Raise encrypted vector to a power

        Args:
            enc_vector: Encrypted vector
            exponent: Integer exponent

        Returns:
            Encrypted result
        """
        try:
            result = enc_vector
            for _ in range(exponent - 1):
                result = result * enc_vector
            return result
        except Exception as e:
            print(f"Power error: {e}")
            return None

    def serialize_context(self, include_secret: bool = False) -> str:
        """
        Serialize context to string

        Args:
            include_secret: Whether to include secret key

        Returns:
            Base64 encoded serialized context
        """
        if not self.context:
            return ""

        try:
            if include_secret:
                serialized = self.context.serialize(
                    save_public_key=True,
                    save_secret_key=True,
                    save_galois_keys=True,
                    save_relin_keys=True
                )
            else:
                serialized = self.public_context.serialize()

            return base64.b64encode(serialized).decode('utf-8')

        except Exception as e:
            print(f"Serialization error: {e}")
            return ""

    def deserialize_context(self, serialized_context: str) -> bool:
        """
        Deserialize context from string

        Args:
            serialized_context: Base64 encoded serialized context

        Returns:
            True if successful, False otherwise
        """
        try:
            context_bytes = base64.b64decode(serialized_context.encode('utf-8'))
            self.context = ts.context_from(context_bytes)
            return True
        except Exception as e:
            print(f"Deserialization error: {e}")
            return False

    def serialize_vector(self, enc_vector: Any) -> str:
        """
        Serialize encrypted vector to string

        Args:
            enc_vector: Encrypted vector

        Returns:
            Base64 encoded serialized vector
        """
        try:
            serialized = enc_vector.serialize()
            return base64.b64encode(serialized).decode('utf-8')
        except Exception as e:
            print(f"Vector serialization error: {e}")
            return ""

    def deserialize_vector(self, serialized_vector: str) -> Optional[Any]:
        """
        Deserialize encrypted vector from string

        Args:
            serialized_vector: Base64 encoded serialized vector

        Returns:
            Encrypted vector or None
        """
        try:
            vector_bytes = base64.b64decode(serialized_vector.encode('utf-8'))

            if self.scheme == 'BFV':
                return ts.bfv_vector_from(self.context, vector_bytes)
            else:
                return ts.ckks_vector_from(self.context, vector_bytes)

        except Exception as e:
            print(f"Vector deserialization error: {e}")
            return None

    def get_context_info(self) -> Dict[str, Any]:
        """
        Get information about current context

        Returns:
            Dictionary with context information
        """
        if not self.context:
            return {'status': 'error', 'message': 'Context not initialized'}

        return {
            'scheme': self.scheme,
            'poly_modulus_degree': self.poly_modulus_degree,
            'is_private': self.context.is_private(),
            'is_public': self.context.is_public(),
            'has_galois_keys': self.context.has_galois_key(),
            'has_relin_keys': self.context.has_relin_key(),
        }

    def batch_encrypt(self, data_list: List[List[Union[int, float]]]) -> List[Any]:
        """
        Encrypt multiple vectors in batch

        Args:
            data_list: List of data vectors to encrypt

        Returns:
            List of encrypted vectors
        """
        encrypted_list = []

        for data in data_list:
            enc = self.encrypt_vector(data)
            if enc is not None:
                encrypted_list.append(enc)

        return encrypted_list

    def batch_decrypt(self, encrypted_list: List[Any]) -> List[List[Union[int, float]]]:
        """
        Decrypt multiple vectors in batch

        Args:
            encrypted_list: List of encrypted vectors

        Returns:
            List of decrypted data vectors
        """
        decrypted_list = []

        for enc in encrypted_list:
            dec = self.decrypt_vector(enc)
            if dec:
                decrypted_list.append(dec)

        return decrypted_list

    def clear_encrypted_vectors(self):
        """Clear stored encrypted vectors to free memory"""
        self.encrypted_vectors.clear()


# Utility functions
def create_tenseal_wrapper(scheme: str = 'BFV', **kwargs) -> TenSEALWrapper:
    """
    Factory function to create and setup TenSEAL wrapper

    Args:
        scheme: 'BFV' or 'CKKS'
        **kwargs: Additional parameters

    Returns:
        Initialized TenSEALWrapper instance
    """
    wrapper = TenSEALWrapper(
        scheme=scheme,
        poly_modulus_degree=kwargs.get('poly_modulus_degree', 8192)
    )

    result = wrapper.setup_context(**kwargs)

    if result['status'] == 'success':
        return wrapper
    else:
        raise RuntimeError(f"Failed to setup TenSEAL wrapper: {result['message']}")


def benchmark_operations(wrapper: TenSEALWrapper, data_size: int = 100) -> Dict[str, float]:
    """
    Benchmark TenSEAL operations

    Args:
        wrapper: Initialized TenSEALWrapper
        data_size: Size of test data

    Returns:
        Dictionary with timing results (in milliseconds)
    """
    import time

    # Generate test data
    data1 = list(range(data_size))
    data2 = list(range(data_size, data_size * 2))

    results = {}

    # Encryption benchmark
    start = time.time()
    enc1 = wrapper.encrypt_vector(data1)
    results['encryption'] = (time.time() - start) * 1000

    # Second encryption for operations
    enc2 = wrapper.encrypt_vector(data2)

    # Addition benchmark
    start = time.time()
    result_add = wrapper.add(enc1, enc2)
    results['addition'] = (time.time() - start) * 1000

    # Multiplication benchmark
    start = time.time()
    result_mul = wrapper.multiply(enc1, enc2)
    results['multiplication'] = (time.time() - start) * 1000

    # Decryption benchmark
    start = time.time()
    wrapper.decrypt_vector(enc1)
    results['decryption'] = (time.time() - start) * 1000

    return results