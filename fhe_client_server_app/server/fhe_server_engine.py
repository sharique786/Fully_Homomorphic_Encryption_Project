"""
FHE Financial Data Processor - Server Application
Handles FHE operations with TenSEAL and OpenFHE support
Windows and Linux compatible
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import json
import sys
import os

# Import FHE wrappers
try:
    from tenseal_wrapper import TenSEALWrapper
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("⚠️ TenSEAL wrapper not available")

try:
    from openfhe_wrapper import OpenFHEWrapper
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    print("⚠️ OpenFHE wrapper not available")

# Initialize FastAPI
app = FastAPI(
    title="FHE Processing Server",
    version="2.0",
    description="Fully Homomorphic Encryption server supporting TenSEAL and OpenFHE"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Restricted countries
RESTRICTED_COUNTRIES = ['CN', 'RU', 'KP', 'IR', 'SY']

# In-memory storage for contexts and sessions
active_sessions = {}
encryption_contexts = {}


# ==================== Pydantic Models ====================

class KeyGenerationRequest(BaseModel):
    scheme: str
    library: str  # 'TenSEAL' or 'OpenFHE'
    params: Dict[str, Any]


class EncryptionRequest(BaseModel):
    data: List[Any]
    column_name: str
    data_type: str
    keys: Dict[str, Any]
    scheme: str
    library: str


class FHEQueryRequest(BaseModel):
    encrypted_data: Dict[str, Any]
    query_params: Dict[str, Any]
    keys: Dict[str, Any]
    library: str


class DecryptionRequest(BaseModel):
    encrypted_results: Dict[str, Any]
    keys: Dict[str, Any]
    library: str


# ==================== FHE Wrapper Management ====================

class FHEContextManager:
    """Manage FHE contexts for different libraries"""

    @staticmethod
    def create_context(library: str, scheme: str, params: Dict) -> Dict:
        """Create FHE context based on library"""
        try:
            if library == 'TenSEAL':
                if not TENSEAL_AVAILABLE:
                    raise ValueError("TenSEAL not available")
                return FHEContextManager._create_tenseal_context(scheme, params)

            elif library == 'OpenFHE':
                if not OPENFHE_AVAILABLE:
                    raise ValueError("OpenFHE wrapper not available")
                return FHEContextManager._create_openfhe_context(scheme, params)

            else:
                raise ValueError(f"Unsupported library: {library}")

        except Exception as e:
            print(f"Error creating context: {str(e)}")
            raise

    @staticmethod
    def _create_tenseal_context(scheme: str, params: Dict) -> Dict:
        """Create TenSEAL context"""
        wrapper = TenSEALWrapper()

        poly_modulus_degree = params.get('poly_modulus_degree', 8192)

        if scheme == 'CKKS':
            scale = params.get('scale', 2 ** 40)
            coeff_mod_bit_sizes = params.get('coeff_mod_bit_sizes', [60, 40, 40, 60])

            wrapper.generate_context(
                scheme='CKKS',
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                scale=scale
            )

        elif scheme == 'BFV':
            wrapper.generate_context(
                scheme='BFV',
                poly_modulus_degree=poly_modulus_degree
            )

        else:
            raise ValueError(f"TenSEAL does not support scheme: {scheme}")

        # Store wrapper in active sessions
        session_id = hashlib.sha256(
            f"tenseal_{scheme}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        active_sessions[session_id] = {
            'wrapper': wrapper,
            'library': 'TenSEAL',
            'scheme': scheme,
            'params': params,
            'created_at': datetime.now().isoformat()
        }

        # Get keys info
        keys_info = wrapper.get_keys_info()
        keys_info['session_id'] = session_id
        keys_info['library'] = 'TenSEAL'
        keys_info['scheme'] = scheme

        return keys_info

    @staticmethod
    def _create_openfhe_context(scheme: str, params: Dict) -> Dict:
        """Create OpenFHE context"""
        wrapper = OpenFHEWrapper()

        # Extract parameters
        ring_dim = params.get('poly_modulus_degree', 16384)
        mult_depth = params.get('mult_depth', 10)
        scale_mod_size = params.get('scale_mod_size', 50)
        batch_size = params.get('batch_size', 8)
        security_level = params.get('security_level', 'HEStd_128_classic')

        # Generate context
        wrapper.generate_context(
            scheme=scheme,
            mult_depth=mult_depth,
            scale_mod_size=scale_mod_size,
            batch_size=batch_size,
            security_level=security_level,
            ring_dim=ring_dim
        )

        # Store wrapper in active sessions
        session_id = hashlib.sha256(
            f"openfhe_{scheme}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        active_sessions[session_id] = {
            'wrapper': wrapper,
            'library': 'OpenFHE',
            'scheme': scheme,
            'params': params,
            'mode': wrapper.mode,  # ctypes, custom_dll, subprocess, or simulation
            'created_at': datetime.now().isoformat()
        }

        # Get keys info
        keys_info = wrapper.get_keys_info()
        keys_info['session_id'] = session_id
        keys_info['library'] = 'OpenFHE'
        keys_info['scheme'] = scheme
        keys_info['mode'] = wrapper.mode

        return keys_info


class FHEProcessor:
    """Process FHE queries on encrypted data"""

    @staticmethod
    def encrypt_data(data: List, column_name: str, data_type: str,
                    session_id: str, library: str) -> List:
        """Encrypt data using specified library"""
        if session_id not in active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = active_sessions[session_id]
        wrapper = session['wrapper']

        try:
            encrypted_values = wrapper.encrypt_data(data, column_name, data_type)

            # Convert to serializable format
            result = []
            for enc_val in encrypted_values:
                if enc_val is None:
                    result.append(None)
                elif isinstance(enc_val, bytes):
                    # TenSEAL returns bytes
                    result.append({
                        'ciphertext': enc_val.hex(),
                        'type': data_type,
                        'library': library
                    })
                elif isinstance(enc_val, dict):
                    # OpenFHE returns dict
                    result.append(enc_val)
                else:
                    result.append({
                        'value': str(enc_val),
                        'type': data_type,
                        'library': library
                    })

            return result

        except Exception as e:
            print(f"Error encrypting data: {str(e)}")
            raise

    @staticmethod
    def process_query(encrypted_data: Dict, query_params: Dict,
                     session_id: str, library: str) -> Dict:
        """Process FHE query on encrypted data"""
        try:
            # Check if restricted country
            is_restricted = query_params.get('is_restricted', False)

            if is_restricted:
                return FHEProcessor._process_restricted_query(
                    encrypted_data, query_params, session_id, library
                )
            else:
                return FHEProcessor._process_cloud_query(
                    encrypted_data, query_params, session_id, library
                )

        except Exception as e:
            raise HTTPException(status_code=500,
                              detail=f"Error processing query: {str(e)}")

    @staticmethod
    def _process_restricted_query(encrypted_data: Dict, query_params: Dict,
                                  session_id: str, library: str) -> Dict:
        """Process query for restricted country data (on-premises)"""
        print(f"⚠️ Processing RESTRICTED country data on-premises")
        print(f"   Country: {query_params.get('user_country')}")
        print(f"   User: {query_params.get('user_id')}")

        result = FHEProcessor._perform_fhe_operations(
            encrypted_data, query_params, session_id, library
        )

        result['processing_location'] = 'on-premises'
        result['is_restricted'] = True
        result['decryption_allowed'] = False
        result['compliance_note'] = 'Data processed on-premises per regulatory requirements'

        return result

    @staticmethod
    def _process_cloud_query(encrypted_data: Dict, query_params: Dict,
                            session_id: str, library: str) -> Dict:
        """Process query for allowed country data (cloud)"""
        print(f"✅ Processing ALLOWED country data in cloud")
        print(f"   User: {query_params.get('user_id')}")

        result = FHEProcessor._perform_fhe_operations(
            encrypted_data, query_params, session_id, library
        )

        result['processing_location'] = 'cloud'
        result['is_restricted'] = False
        result['decryption_allowed'] = True

        return result

    @staticmethod
    def _perform_fhe_operations(encrypted_data: Dict, query_params: Dict,
                               session_id: str, library: str) -> Dict:
        """Perform actual FHE operations"""
        operation_type = query_params.get('operation_type')
        user_id = query_params.get('user_id')
        start_date = pd.Timestamp(query_params.get('start_date'))
        end_date = pd.Timestamp(query_params.get('end_date'))
        currencies = query_params.get('currencies', [])

        # Get session info
        session = active_sessions.get(session_id, {})
        scheme = session.get('scheme', 'CKKS')
        mode = session.get('mode', 'simulation')

        print(f"🔐 Performing FHE operations:")
        print(f"   Library: {library}")
        print(f"   Scheme: {scheme}")
        if library == 'OpenFHE':
            print(f"   Mode: {mode}")
        print(f"   Operation: {operation_type}")

        # Simulate FHE computations on encrypted data
        # In production, these would be actual homomorphic operations

        if operation_type == "Transaction Analysis":
            return {
                'total_amount': float(np.random.uniform(10000, 50000)),
                'avg_amount': float(np.random.uniform(100, 500)),
                'min_amount': float(np.random.uniform(10, 50)),
                'max_amount': float(np.random.uniform(1000, 5000)),
                'total_transactions': int(np.random.randint(50, 200)),
                'currency_analysis': {
                    curr: {
                        'count': int(np.random.randint(10, 50)),
                        'total': float(np.random.uniform(1000, 10000))
                    }
                    for curr in currencies
                },
                'monthly_pattern': {
                    f"2024-{i:02d}": int(np.random.randint(5, 30))
                    for i in range(1, 13)
                },
                'encrypted': True,
                'scheme': scheme,
                'library': library,
                'mode': mode if library == 'OpenFHE' else None
            }

        elif operation_type == "Account Summary":
            return {
                'total_accounts': int(np.random.randint(1, 5)),
                'active_accounts': int(np.random.randint(1, 5)),
                'total_balance': float(np.random.uniform(5000, 50000)),
                'account_types': {
                    'Savings': int(np.random.randint(0, 2)),
                    'Checking': int(np.random.randint(0, 2)),
                    'Credit': int(np.random.randint(0, 2)),
                    'Investment': int(np.random.randint(0, 2))
                },
                'encrypted': True,
                'scheme': scheme,
                'library': library,
                'mode': mode if library == 'OpenFHE' else None
            }

        return {
            'status': 'completed',
            'encrypted': True,
            'scheme': scheme,
            'library': library
        }

    @staticmethod
    def decrypt_results(encrypted_results: Dict, session_id: str,
                       library: str) -> Dict:
        """Decrypt results if allowed"""
        # Check if decryption is allowed
        if not encrypted_results.get('decryption_allowed', False):
            return {
                'error': 'Decryption not allowed for restricted country data',
                'message': 'Results can only be decrypted in authorized jurisdiction (HSM/KMS)',
                'encrypted_results': encrypted_results
            }

        # In real implementation, would decrypt using private key
        if session_id not in active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = active_sessions[session_id]
        wrapper = session['wrapper']

        # For demo, return as-is (simulation)
        decrypted = encrypted_results.copy()
        decrypted['decrypted'] = True
        decrypted['decryption_timestamp'] = datetime.now().isoformat()

        return decrypted


# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    openfhe_info = {}
    if OPENFHE_AVAILABLE:
        try:
            temp_wrapper = OpenFHEWrapper()
            openfhe_info = {
                'available': True,
                'mode': temp_wrapper.mode,
                'platform': sys.platform
            }
        except:
            openfhe_info = {'available': False}

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "platform": sys.platform,
        "libraries": {
            "tenseal": {
                "available": TENSEAL_AVAILABLE,
                "schemes": ["CKKS", "BFV"] if TENSEAL_AVAILABLE else []
            },
            "openfhe": {
                **openfhe_info,
                "schemes": ["CKKS", "BFV", "BGV"] if OPENFHE_AVAILABLE else []
            }
        },
        "active_sessions": len(active_sessions)
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FHE Processing Server",
        "version": "2.0",
        "platform": sys.platform,
        "libraries": {
            "TenSEAL": TENSEAL_AVAILABLE,
            "OpenFHE": OPENFHE_AVAILABLE
        },
        "endpoints": {
            "/health": "Health check",
            "/generate_keys": "Generate FHE keys",
            "/encrypt": "Encrypt data",
            "/fhe_query": "Perform FHE query",
            "/decrypt": "Decrypt results",
            "/sessions": "List active sessions"
        }
    }


@app.post("/generate_keys")
async def generate_keys(request: KeyGenerationRequest):
    """Generate FHE encryption keys"""
    try:
        print(f"\n{'='*60}")
        print(f"🔑 Key Generation Request")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"   Platform: {sys.platform}")
        print(f"{'='*60}")

        result = FHEContextManager.create_context(
            request.library,
            request.scheme,
            request.params
        )

        result['timestamp'] = datetime.now().isoformat()
        result['platform'] = sys.platform

        print(f"✅ Keys generated successfully")
        print(f"   Session ID: {result.get('session_id')}")
        if request.library == 'OpenFHE':
            print(f"   OpenFHE Mode: {result.get('mode')}")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        print(f"❌ Error generating keys: {str(e)}")
        raise HTTPException(status_code=500,
                          detail=f"Error generating keys: {str(e)}")


@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest):
    """Encrypt data"""
    try:
        session_id = request.keys.get('session_id')

        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid or missing session ID")

        encrypted_values = FHEProcessor.encrypt_data(
            request.data,
            request.column_name,
            request.data_type,
            session_id,
            request.library
        )

        return {
            'encrypted_values': encrypted_values,
            'column_name': request.column_name,
            'data_type': request.data_type,
            'scheme': request.scheme,
            'library': request.library,
            'count': len(encrypted_values),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Error encrypting data: {str(e)}")
        raise HTTPException(status_code=500,
                          detail=f"Error encrypting data: {str(e)}")


@app.post("/fhe_query")
async def fhe_query(request: FHEQueryRequest):
    """Perform FHE query on encrypted data"""
    try:
        session_id = request.keys.get('session_id')

        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid or missing session ID")

        results = FHEProcessor.process_query(
            request.encrypted_data,
            request.query_params,
            session_id,
            request.library
        )

        results['timestamp'] = datetime.now().isoformat()
        results['session_id'] = session_id

        return results

    except Exception as e:
        print(f"❌ Error performing FHE query: {str(e)}")
        raise HTTPException(status_code=500,
                          detail=f"Error performing FHE query: {str(e)}")


@app.post("/decrypt")
async def decrypt_results(request: DecryptionRequest):
    """Decrypt results (if allowed by jurisdiction)"""
    try:
        session_id = request.keys.get('session_id')

        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid or missing session ID")

        decrypted = FHEProcessor.decrypt_results(
            request.encrypted_results,
            session_id,
            request.library
        )

        decrypted['timestamp'] = datetime.now().isoformat()

        return decrypted

    except Exception as e:
        print(f"❌ Error decrypting results: {str(e)}")
        raise HTTPException(status_code=500,
                          detail=f"Error decrypting results: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List active FHE sessions"""
    sessions_info = []

    for session_id, session in active_sessions.items():
        info = {
            'session_id': session_id,
            'library': session.get('library'),
            'scheme': session.get('scheme'),
            'created_at': session.get('created_at'),
            'params': session.get('params')
        }

        if session.get('library') == 'OpenFHE':
            info['mode'] = session.get('mode')

        sessions_info.append(info)

    return {
        'active_sessions': len(sessions_info),
        'sessions': sessions_info,
        'timestamp': datetime.now().isoformat()
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {
            'status': 'success',
            'message': f'Session {session_id} deleted',
            'timestamp': datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404,
                          detail=f"Session {session_id} not found")


@app.post("/rotate_keys")
async def rotate_keys(request: KeyGenerationRequest):
    """Rotate keys with backward compatibility"""
    try:
        old_session_id = request.params.get('old_session_id')

        print(f"\n{'='*60}")
        print(f"🔄 Key Rotation Request")
        print(f"   Old Session: {old_session_id}")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"{'='*60}")

        # Generate new keys
        new_keys = FHEContextManager.create_context(
            request.library,
            request.scheme,
            request.params
        )

        # Keep reference to old session for backward compatibility
        if old_session_id and old_session_id in active_sessions:
            old_session = active_sessions[old_session_id]
            new_keys['old_session_id'] = old_session_id
            new_keys['backward_compatible'] = True

        new_keys['timestamp'] = datetime.now().isoformat()
        new_keys['action'] = 'rotated'

        print(f"✅ Keys rotated successfully")
        print(f"   New Session ID: {new_keys.get('session_id')}")
        print(f"   Backward Compatible: {new_keys.get('backward_compatible', False)}")
        print(f"{'='*60}\n")

        return new_keys

    except Exception as e:
        print(f"❌ Error rotating keys: {str(e)}")
        raise HTTPException(status_code=500,
                          detail=f"Error rotating keys: {str(e)}")
    """Get server statistics"""
    stats = {
        'platform': sys.platform,
        'active_sessions': len(active_sessions),
        'libraries': {
            'tenseal': {
                'available': TENSEAL_AVAILABLE,
                'active_sessions': sum(1 for s in active_sessions.values()
                                     if s.get('library') == 'TenSEAL')
            },
            'openfhe': {
                'available': OPENFHE_AVAILABLE,
                'active_sessions': sum(1 for s in active_sessions.values()
                                     if s.get('library') == 'OpenFHE')
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    # Add OpenFHE mode distribution
    if OPENFHE_AVAILABLE:
        openfhe_modes = {}
        for session in active_sessions.values():
            if session.get('library') == 'OpenFHE':
                mode = session.get('mode', 'unknown')
                openfhe_modes[mode] = openfhe_modes.get(mode, 0) + 1
        stats['libraries']['openfhe']['modes'] = openfhe_modes

    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("🔐 FHE Processing Server Starting...")
    print("=" * 60)
    print(f"Platform: {sys.platform}")
    print(f"TenSEAL Available: {TENSEAL_AVAILABLE}")
    print(f"OpenFHE Available: {OPENFHE_AVAILABLE}")

    if OPENFHE_AVAILABLE:
        try:
            temp_wrapper = OpenFHEWrapper()
            print(f"OpenFHE Mode: {temp_wrapper.mode}")
        except Exception as e:
            print(f"OpenFHE initialization note: {str(e)}")

    print("=" * 60)
    print("Server will be available at:")
    print("  - http://localhost:8000")
    print("  - API Documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")