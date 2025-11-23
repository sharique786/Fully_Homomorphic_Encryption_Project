import streamlit as st
import tenseal as ts
import pandas as pd
import base64
import pickle
import json
import concurrent.futures
import threading

class ClientKeyManager:
    """Manages FHE keys on client side"""
    
    def __init__(self):
        self.context = None
        self.private_key = None
        self.public_context = None
        self.library = None
        self.scheme = None
        # self.encryption_lock = threading.Lock()
    
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
    
    def decrypt_locally(self, encrypted_data, data_type: str = 'numeric'):
        """
        Enhanced decrypt with better error handling and format support
        
        Supports:
        - Raw bytes
        - Base64 string
        - Compressed bytes
        - Dict with various keys
        - Simulation mode values
        """
        if not self.context:
            st.error("No context available for decryption")
            return None
        
        try:
            # Step 1: Extract raw encrypted bytes
            encrypted_bytes = self._extract_encrypted_bytes(encrypted_data)
            
            if encrypted_bytes is None:
                st.error("Failed to extract encrypted bytes")
                return None
            
            # Step 2: Check if it's already a plain number (simulation mode)
            if isinstance(encrypted_bytes, (int, float)):
                return float(encrypted_bytes)
            
            # Step 3: Try decompression if it looks compressed
            if isinstance(encrypted_bytes, bytes):
                # Try gzip decompression
                try:
                    decompressed = gzip.decompress(encrypted_bytes)
                    encrypted_bytes = decompressed
                except:
                    # Not compressed, use as-is
                    pass
            
            # Step 4: Decrypt based on scheme
            if self.scheme == 'CKKS':
                vec = ts.ckks_vector_from(self.context, encrypted_bytes)
                decrypted = vec.decrypt()[0]
                
            elif self.scheme == 'BFV':
                vec = ts.bfv_vector_from(self.context, encrypted_bytes)
                decrypted = vec.decrypt()[0]
            else:
                st.error(f"Unsupported scheme: {self.scheme}")
                return None
            
            return decrypted
            
        except Exception as e:
            st.error(f"Local decryption failed: {e}")
            st.write(f"**Debug Info:**")
            st.write(f"- Data type received: {type(encrypted_data)}")
            st.write(f"- Scheme: {self.scheme}")
            
            if isinstance(encrypted_data, dict):
                st.write(f"- Dict keys: {list(encrypted_data.keys())}")
            elif isinstance(encrypted_data, bytes):
                st.write(f"- Bytes length: {len(encrypted_data)}")
            elif isinstance(encrypted_data, str):
                st.write(f"- String length: {len(encrypted_data)}")
            
            import traceback
            st.code(traceback.format_exc())
            return None
    
    
    def _extract_encrypted_bytes(self, encrypted_data):
        """
        Helper to extract bytes from various encrypted data formats
        """
        if encrypted_data is None:
            return None
        
        # Already bytes
        if isinstance(encrypted_data, bytes):
            return encrypted_data
        
        # Already a number (simulation mode)
        if isinstance(encrypted_data, (int, float)):
            return encrypted_data
        
        # Base64 string
        if isinstance(encrypted_data, str):
            try:
                return base64.b64decode(encrypted_data)
            except Exception as e:
                st.error(f"Failed to decode base64 string: {e}")
                return None
        
        # Dictionary format
        if isinstance(encrypted_data, dict):
            # Try common keys in order of preference
            for key in ['encrypted_value', 'ciphertext', 'data', 'value', 'encrypted_data']:
                if key in encrypted_data:
                    # Recursively extract from the value
                    return self._extract_encrypted_bytes(encrypted_data[key])
            
            # Special case: simulation mode
            if 'simulated_value' in encrypted_data:
                return encrypted_data['simulated_value']
            
            # If mode is specified
            if 'mode' in encrypted_data:
                mode = encrypted_data['mode']
                if mode == 'simulation':
                    # Look for simulated value
                    if 'simulated_value' in encrypted_data:
                        return encrypted_data['simulated_value']
            
            st.error(f"Dict doesn't contain expected keys. Available: {list(encrypted_data.keys())}")
            return None
        
        # List (take first element)
        if isinstance(encrypted_data, list):
            if len(encrypted_data) > 0:
                return self._extract_encrypted_bytes(encrypted_data[0])
            return None
        
        st.error(f"Unsupported encrypted data type: {type(encrypted_data)}")
        return None
    
    
    def decrypt_list_locally(self, encrypted_list: list, data_type: str = 'numeric'):
        """Decrypt multiple values locally with error handling"""
        results = []
        
        for idx, enc_data in enumerate(encrypted_list):
            try:
                decrypted = self.decrypt_locally(enc_data, data_type)
                results.append(decrypted)
            except Exception as e:
                st.warning(f"Failed to decrypt item {idx}: {e}")
                results.append(None)
        
        return results
    
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
    
    def encrypt_single_value(self, value, data_type: str):
        """
        Simple single value encryption - NO THREADING
        """
        if value is None or pd.isna(value):
            return None
        
        try:
            # Convert value based on data type
            if data_type == 'numeric':
                num_value = float(value)
            elif data_type == 'text':
                num_value = float(sum([ord(c) for c in str(value)]))
            elif data_type == 'date':
                if isinstance(value, str):
                    num_value = pd.Timestamp(value).timestamp()
                elif isinstance(value, pd.Timestamp):
                    num_value = value.timestamp()
                else:
                    num_value = float(value)
            else:
                num_value = float(value)
            
            # Simple encryption - no locks needed
            if self.scheme == 'CKKS':
                enc_vec = ts.ckks_vector(self.context, [num_value])
            elif self.scheme == 'BFV':
                enc_vec = ts.bfv_vector(self.context, [int(num_value)])
            else:
                return None
            
            return enc_vec.serialize()
        
        except Exception as e:
            return None
        
    
    def encrypt_batch_parallel(self, values: list, data_type: str, max_workers: int = 4):
        """
        Encrypt batch of values in parallel using thread pool
        """
        encrypted_results = [None] * len(values)
        
        def encrypt_item(idx, value):
            encrypted = self.encrypt_single_value(value, data_type)
            return idx, encrypted
        
        # Use ThreadPoolExecutor for parallel encryption
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(encrypt_item, idx, value): idx 
                for idx, value in enumerate(values)
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, encrypted = future.result()
                    encrypted_results[idx] = encrypted
                except Exception as e:
                    pass
        
        return encrypted_results
    
    def encrypt_column_locally(self, column_data: list, column_name: str, 
                               data_type: str, simd_mode: str = "individual",
                               progress_callback=None):
        """
        SIMPLIFIED: Sequential encryption with progress updates
        No threading - just optimized loops
        """
        if not self.context:
            st.error("Context not initialized. Generate keys first.")
            return None
        
        try:
            encrypted_results = []
            total = len(column_data)
            
            # SIMD Mode: Individual Encryption (Sequential)
            if simd_mode == "individual":
                st.info(f"   🔒 Encrypting {total} values for {column_name}...")
                
                # Process in display chunks for progress updates
                chunk_size = 100
                
                for i, value in enumerate(column_data):
                    # Encrypt single value
                    encrypted = self.encrypt_single_value(value, data_type)
                    encrypted_results.append(encrypted)
                    
                    # Update progress every chunk
                    if (i + 1) % chunk_size == 0 or (i + 1) == total:
                        progress = ((i + 1) / total) * 100
                        if progress_callback:
                            progress_callback(i + 1, total, progress)
                        else:
                            st.info(f"      Progress: {progress:.1f}% ({i + 1}/{total})")
            
            # SIMD Mode: Packed Vector (Sequential)
            elif simd_mode == "packed_vector":
                if self.scheme != 'CKKS':
                    st.error("Packed vector mode requires CKKS scheme")
                    return None
                
                if data_type != 'numeric':
                    st.error("Packed vector mode only supports numeric data")
                    return None
                
                st.info(f"   🔒 Encrypting using SIMD packed vectors...")
                
                batch_size = 128
                num_batches = (total + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total)
                    batch_data = column_data[batch_start:batch_end]
                    
                    # Convert to float and filter None
                    valid_data = []
                    for v in batch_data:
                        if v is not None and not pd.isna(v):
                            try:
                                valid_data.append(float(v))
                            except (ValueError, TypeError):
                                continue
                    
                    if valid_data:
                        # Encrypt entire batch
                        enc_vec = ts.ckks_vector(self.context, valid_data)
                        
                        encrypted_results.append({
                            'encrypted_bytes': enc_vec.serialize(),
                            'batch_start': batch_start,
                            'batch_end': batch_end,
                            'batch_size': len(valid_data),
                            'mode': 'packed_vector'
                        })
                        
                        progress = ((batch_idx + 1) / num_batches) * 100
                        st.info(f"      Batch {batch_idx + 1}/{num_batches}: {progress:.1f}%")
            
            # SIMD Mode: Batch Processing (Sequential)
            elif simd_mode == "batch_processing":
                st.info(f"   🔒 Encrypting with optimized batching...")
                
                batch_size = 256
                num_batches = (total + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total)
                    batch_data = column_data[batch_start:batch_end]
                    
                    # Encrypt batch sequentially
                    for value in batch_data:
                        encrypted = self.encrypt_single_value(value, data_type)
                        encrypted_results.append(encrypted)
                    
                    progress = ((batch_idx + 1) / num_batches) * 100
                    st.info(f"      Batch {batch_idx + 1}/{num_batches}: {progress:.1f}%")
            
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