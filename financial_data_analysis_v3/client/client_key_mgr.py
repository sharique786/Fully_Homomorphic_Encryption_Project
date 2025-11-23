import streamlit as st
import tenseal as ts
import pandas as pd
import base64
import pickle
import json

class ClientKeyManager:
    """Manages FHE keys on client side"""
    
    def __init__(self):
        self.context = None
        self.private_key = None
        self.public_context = None  # Context without private key
        self.library = None
        self.scheme = None
    
    def generate_context_locally(self, library: str, scheme: str, context_params: dict):
        """
        Generate TenSEAL context locally using same params as server
        """
        self.library = library
        self.scheme = scheme
        
        if library != "TenSEAL":
            st.error("Client-side key generation only supported for TenSEAL")
            return False
        
        try:
            poly_degree = context_params.get('poly_modulus_degree', 8192)
            coeff_bits = context_params.get('coeff_mod_bit_sizes', [60, 40, 40, 60])
            scale = context_params.get('scale', 2**40)
            plain_modulus = context_params.get('plain_modulus', 1032193)
            
            st.info(f"🔧 Creating local TenSEAL context: {scheme}")
            
            if scheme == 'CKKS':
                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_degree,
                    coeff_mod_bit_sizes=coeff_bits
                )
                self.context.global_scale = scale
                
            elif scheme == 'BFV':
                self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=poly_degree,
                    plain_modulus=plain_modulus,
                    coeff_mod_bit_sizes=coeff_bits
                )
            else:
                st.error(f"Unsupported scheme: {scheme}")
                return False
            
            # Generate ALL keys locally
            st.info("🔑 Generating Galois keys (for rotation & SIMD)...")
            self.context.generate_galois_keys()
            
            st.info("🔑 Generating Relinearization keys (for multiplication)...")
            self.context.generate_relin_keys()
            
            st.success("✅ Local context and keys generated!")
            
            # Create public context (without private key)
            self.public_context = self.context.copy()
            self.public_context.make_context_public()
            
            # Store private key separately for safety
            self.private_key = self.context.secret_key()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Local context generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_public_keys_for_server(self):
        """
        Extract ONLY public keys to send to server
        Private key stays on client
        """
        if not self.public_context:
            return None
        
        try:
            # Serialize public context (includes public key, galois keys, relin keys)
            public_context_bytes = self.public_context.serialize()
            
            return {
                'public_context': base64.b64encode(public_context_bytes).decode('utf-8'),
                'library': self.library,
                'scheme': self.scheme,
                'has_galois_keys': True,
                'has_relin_keys': True,
                'has_private_key': False  # Explicitly mark no private key
            }
            
        except Exception as e:
            st.error(f"Failed to extract public keys: {e}")
            return None
    
    def decrypt_locally(self, encrypted_data: bytes, data_type: str = 'numeric'):
        """
        Decrypt data locally using private key
        NEVER sends private key to server
        """
        if not self.context:
            st.error("No context available for decryption")
            return None
        
        try:
            # Deserialize encrypted data
            if isinstance(encrypted_data, str):
                encrypted_data = base64.b64decode(encrypted_data)
            
            # Decrypt based on scheme
            if self.scheme == 'CKKS':
                vec = ts.ckks_vector_from(self.context, encrypted_data)
                decrypted = vec.decrypt()[0]
                
            elif self.scheme == 'BFV':
                vec = ts.bfv_vector_from(self.context, encrypted_data)
                decrypted = vec.decrypt()[0]
            else:
                st.error(f"Unsupported scheme: {self.scheme}")
                return None
            
            return decrypted
            
        except Exception as e:
            st.error(f"Local decryption failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def decrypt_list_locally(self, encrypted_list: list, data_type: str = 'numeric'):
        """Decrypt multiple values locally"""
        return [self.decrypt_locally(enc, data_type) for enc in encrypted_list]
    
    def save_private_key(self, filename: str = "private_key.bin"):
        """Save private key to file for backup"""
        if not self.context:
            return False
        
        try:
            # Serialize full context (with private key)
            context_with_secret = self.context.serialize()
            
            with open(filename, 'wb') as f:
                f.write(context_with_secret)
            
            return True
        except Exception as e:
            st.error(f"Failed to save private key: {e}")
            return False
    
    def load_private_key(self, filename: str = "private_key.bin"):
        """Load private key from file"""
        try:
            with open(filename, 'rb') as f:
                context_bytes = f.read()
            
            if self.scheme == 'CKKS':
                self.context = ts.context_from(context_bytes)
            elif self.scheme == 'BFV':
                self.context = ts.context_from(context_bytes)
            
            return True
        except Exception as e:
            st.error(f"Failed to load private key: {e}")
            return False
        
    # ==================== Client-Side Encryption Methods ====================
    
    def encrypt_column_locally(self, column_data: list, column_name: str, 
                               data_type: str, simd_mode: str = "individual"):
        """
        Encrypt entire column locally on client
        
        Args:
            column_data: List of values to encrypt
            column_name: Name of column
            data_type: 'numeric', 'text', or 'date'
            simd_mode: 'individual', 'packed_vector', or 'batch_processing'
        
        Returns:
            List of encrypted values (as bytes)
        """
        if not self.context:
            st.error("Context not initialized. Generate keys first.")
            return None
        
        try:
            encrypted_results = []
            
            # SIMD Mode: Individual Encryption
            if simd_mode == "individual":
                st.info(f"   🔒 Encrypting {len(column_data)} values individually on client...")
                
                for idx, value in enumerate(column_data):
                    if value is None or pd.isna(value):
                        encrypted_results.append(None)
                        continue
                    
                    try:
                        # Convert value based on data type
                        if data_type == 'numeric':
                            num_value = float(value)
                        elif data_type == 'text':
                            # Encode text as numeric (sum of ASCII values)
                            num_value = float(sum([ord(c) for c in str(value)]))
                        elif data_type == 'date':
                            # Convert date to timestamp
                            if isinstance(value, str):
                                num_value = pd.Timestamp(value).timestamp()
                            elif isinstance(value, pd.Timestamp):
                                num_value = value.timestamp()
                            else:
                                num_value = float(value)
                        else:
                            num_value = float(value)
                        
                        # Encrypt based on scheme
                        if self.scheme == 'CKKS':
                            enc_vec = ts.ckks_vector(self.context, [num_value])
                        elif self.scheme == 'BFV':
                            enc_vec = ts.bfv_vector(self.context, [int(num_value)])
                        else:
                            st.error(f"Unsupported scheme: {self.scheme}")
                            return None
                        
                        # Serialize to bytes
                        encrypted_results.append(enc_vec.serialize())
                        
                        # Progress indicator every 100 items
                        if (idx + 1) % 100 == 0:
                            st.info(f"      Encrypted {idx + 1}/{len(column_data)} values...")
                    
                    except Exception as e:
                        st.warning(f"   ⚠️ Failed to encrypt value {idx}: {e}")
                        encrypted_results.append(None)
            
            # SIMD Mode: Packed Vector
            elif simd_mode == "packed_vector":
                if self.scheme != 'CKKS':
                    st.error("Packed vector mode requires CKKS scheme")
                    return None
                
                if data_type != 'numeric':
                    st.error("Packed vector mode only supports numeric data")
                    return None
                
                st.info(f"   🔒 Encrypting using SIMD packed vectors on client...")
                
                batch_size = 128  # SIMD slot size
                
                for batch_start in range(0, len(column_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(column_data))
                    batch_data = column_data[batch_start:batch_end]
                    
                    # Convert to float and filter None values
                    valid_data = []
                    for v in batch_data:
                        if v is not None and not pd.isna(v):
                            try:
                                valid_data.append(float(v))
                            except (ValueError, TypeError):
                                continue
                    
                    if valid_data:
                        # Encrypt entire batch in single ciphertext
                        enc_vec = ts.ckks_vector(self.context, valid_data)
                        
                        # Store with metadata about batch
                        encrypted_results.append({
                            'encrypted_bytes': enc_vec.serialize(),
                            'batch_start': batch_start,
                            'batch_end': batch_end,
                            'batch_size': len(valid_data),
                            'mode': 'packed_vector'
                        })
                        
                        st.info(f"      Encrypted batch {batch_start}-{batch_end} ({len(valid_data)} values)")
            
            # SIMD Mode: Batch Processing
            elif simd_mode == "batch_processing":
                st.info(f"   🔒 Encrypting with optimized batching on client...")
                
                batch_size = 256
                
                for batch_start in range(0, len(column_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(column_data))
                    batch_data = column_data[batch_start:batch_end]
                    
                    # Encrypt batch
                    for idx, value in enumerate(batch_data):
                        if value is None or pd.isna(value):
                            encrypted_results.append(None)
                            continue
                        
                        try:
                            # Convert based on data type
                            if data_type == 'numeric':
                                num_value = float(value)
                            elif data_type == 'text':
                                num_value = float(sum([ord(c) for c in str(value)]))
                            elif data_type == 'date':
                                if isinstance(value, str):
                                    num_value = pd.Timestamp(value).timestamp()
                                else:
                                    num_value = float(value)
                            else:
                                num_value = float(value)
                            
                            # Encrypt
                            if self.scheme == 'CKKS':
                                enc_vec = ts.ckks_vector(self.context, [num_value])
                            elif self.scheme == 'BFV':
                                enc_vec = ts.bfv_vector(self.context, [int(num_value)])
                            
                            encrypted_results.append(enc_vec.serialize())
                        
                        except Exception as e:
                            encrypted_results.append(None)
                    
                    st.info(f"      Encrypted batch {batch_start}-{batch_end}")
            
            else:
                st.error(f"Unsupported SIMD mode: {simd_mode}")
                return None
            
            st.success(f"   ✅ Encrypted {len(encrypted_results)} values for {column_name}")
            return encrypted_results
        
        except Exception as e:
            st.error(f"Column encryption failed: {e}")
            import traceback
            traceback.print_exc()
            return None