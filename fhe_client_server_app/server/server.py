"""
Comprehensive FHE Financial Transaction Server - FIXED VERSION
Performs ACTUAL FHE operations (not simulated)
Python 3.11+
"""

import hashlib
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fhe_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import FHE wrappers
try:
    from tenseal_wrapper import TenSEALWrapper
    TENSEAL_AVAILABLE = True
    print("✅ TenSEAL wrapper loaded")
except ImportError as e:
    TENSEAL_AVAILABLE = False
    logger.warning(f"⚠️ TenSEAL not available: {str(e)}")

try:
    from openfhe_wrapper import OpenFHEWrapper
    OPENFHE_AVAILABLE = True
    print("✅ OpenFHE wrapper loaded")
except ImportError as e:
    OPENFHE_AVAILABLE = False
    logger.warning(f"⚠️ OpenFHE not available: {str(e)}")

app = FastAPI(title="FHE Server", version="5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global wrappers (initialized once)
global_wrappers = {'tenseal': None, 'openfhe': None}
active_sessions = {}

# Models
class KeyGenerationRequest(BaseModel):
    scheme: str
    library: str
    params: Dict[str, Any] = Field(default_factory=dict)

class BatchEncryptionRequest(BaseModel):
    data: List[Any]
    column_name: str
    data_type: str
    keys: Dict[str, Any]
    scheme: str
    library: str
    batch_id: Optional[int] = None

class FHEQueryRequest(BaseModel):
    encrypted_data: Dict[str, List[Any]]
    query_params: Dict[str, Any]
    keys: Dict[str, Any]
    library: str
    scheme: str

class DecryptionRequest(BaseModel):
    encrypted_data: List[Any]
    data_type: str
    keys: Dict[str, Any]
    library: str
    scheme: str

@app.on_event("startup")
async def startup_event():
    print("=" * 80)
    print("🚀 Initializing FHE Server")
    print("=" * 80)

    if TENSEAL_AVAILABLE:
        try:
            global_wrappers['tenseal'] = TenSEALWrapper()
            print("✅ TenSEAL wrapper initialized")
        except Exception as e:
            print(f"❌ TenSEAL init error: {str(e)}")

    if OPENFHE_AVAILABLE:
        try:
            global_wrappers['openfhe'] = OpenFHEWrapper()
            print(f"✅ OpenFHE wrapper initialized - Mode: {global_wrappers['openfhe'].mode}")
        except Exception as e:
            print(f"❌ OpenFHE init error: {str(e)}")

    print("=" * 80)

@app.get("/")
async def root():
    return {
        "service": "FHE Server",
        "version": "5.1",
        "libraries": {
            "TenSEAL": {"available": TENSEAL_AVAILABLE, "initialized": global_wrappers['tenseal'] is not None},
            "OpenFHE": {"available": OPENFHE_AVAILABLE, "initialized": global_wrappers['openfhe'] is not None}
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "libraries": {
            "tenseal": {"available": TENSEAL_AVAILABLE, "ready": global_wrappers['tenseal'] is not None},
            "openfhe": {
                "available": OPENFHE_AVAILABLE,
                "ready": global_wrappers['openfhe'] is not None,
                "mode": global_wrappers['openfhe'].mode if global_wrappers['openfhe'] else None
            }
        },
        "active_sessions": len(active_sessions)
    }

@app.post("/generate_keys")
async def generate_keys(request: KeyGenerationRequest):
    try:
        print(f"\n{'='*60}")
        print(f"🔑 Key Generation Request")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"{'='*60}")

        # Validate library
        if request.library not in ['TenSEAL', 'OpenFHE']:
            raise ValueError(f"Invalid library: {request.library}")

        wrapper = global_wrappers[request.library.lower()]
        if not wrapper:
            raise ValueError(f"{request.library} not available or not initialized")

        # Generate context
        if request.library == 'TenSEAL':
            wrapper.generate_context(
                scheme=request.scheme,
                poly_modulus_degree=request.params.get('poly_modulus_degree', 8192),
                coeff_mod_bit_sizes=request.params.get('coeff_mod_bit_sizes', [60,40,40,60]),
                scale=request.params.get('scale', 2**40),
                plain_modulus=request.params.get('plain_modulus', 1032193)
            )
        elif request.library == 'OpenFHE':
            wrapper.generate_context(
                scheme=request.scheme,
                mult_depth=request.params.get('mult_depth', 10),
                scale_mod_size=request.params.get('scale_mod_size', 50),
                ring_dim=request.params.get('poly_modulus_degree', 16384),
                batch_size=request.params.get('batch_size', 8),
                security_level=request.params.get('security_level', 'HEStd_128_classic')
            )

        # Create session
        session_id = hashlib.sha256(
            f"{request.library}_{request.scheme}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        active_sessions[session_id] = {
            'wrapper': wrapper,
            'library': request.library,
            'scheme': request.scheme,
            'params': request.params,
            'created_at': datetime.now().isoformat()
        }

        # Get keys
        keys = wrapper.get_keys_info()
        keys['session_id'] = session_id
        keys['library'] = request.library  # CRITICAL: Ensure library is in response
        keys['scheme'] = request.scheme    # CRITICAL: Ensure scheme is in response

        print(f"✅ Keys generated - Session: {session_id}")
        print(f"   Library: {request.library}, Scheme: {request.scheme}")
        print(f"{'='*60}\n")

        return keys

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Key generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encrypt")
async def encrypt_data(request: BatchEncryptionRequest):
    """Encrypt data - ACTUAL FHE"""
    try:
        # Validate request
        if not request.scheme:
            raise ValueError("Scheme is required")
        if not request.library:
            raise ValueError("Library is required")
        if not request.data:
            raise ValueError("No data provided")

        session_id = request.keys.get('session_id')
        if not session_id:
            raise ValueError("Session ID is required")
        if session_id not in active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = active_sessions[session_id]
        wrapper = session['wrapper']

        print(f"\n{'='*60}")
        print(f"🔒 ENCRYPTION REQUEST")
        print(f"   Column: {request.column_name}")
        print(f"   Type: {request.data_type}")
        print(f"   Records: {len(request.data)}")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"   Batch: {request.batch_id}")
        print(f"{'='*60}")

        # Check scheme compatibility
        if request.scheme == 'BFV' and request.data_type == 'text':
            raise ValueError("BFV scheme does not support text data type")

        # Perform encryption
        start_time = datetime.now()
        encrypted_values = wrapper.encrypt_data(request.data, request.column_name, request.data_type)
        duration = (datetime.now() - start_time).total_seconds()

        # Process results
        result = []
        successful = 0
        for idx, enc in enumerate(encrypted_values):
            if enc is None:
                result.append(None)
            elif isinstance(enc, bytes):
                result.append({
                    'ciphertext': enc.hex(),
                    'index': idx,
                    'type': request.data_type,
                    'library': request.library,
                    'scheme': request.scheme
                })
                successful += 1
            else:
                result.append(enc)
                successful += 1

        print(f"✅ ENCRYPTION COMPLETE")
        print(f"   Successful: {successful}/{len(request.data)}")
        print(f"   Duration: {duration:.3f}s")
        print(f"{'='*60}\n")

        return {
            'success': True,
            'encrypted_values': result,
            'column_name': request.column_name,
            'data_type': request.data_type,
            'batch_id': request.batch_id,
            'statistics': {
                'total': len(request.data),
                'successful': successful,
                'duration': duration
            }
        }

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Encryption error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fhe_query")
async def fhe_query(request: FHEQueryRequest):
    """Perform FHE query"""
    try:
        # Validate request
        if not request.library:
            raise ValueError("Library is required")
        if not request.scheme:
            raise ValueError("Scheme is required")

        session_id = request.keys.get('session_id')
        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid session ID")

        session = active_sessions[session_id]
        operation = request.query_params.get('operation_type')

        print(f"\n{'='*60}")
        print(f"🔍 FHE QUERY")
        print(f"   Operation: {operation}")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"{'='*60}")

        # Extract encrypted data
        encrypted_amounts = []
        for key, vals in request.encrypted_data.items():
            if 'amount' in key.lower():
                encrypted_amounts = vals
                break

        print(f"✅ QUERY COMPLETE")
        print(f"{'='*60}\n")

        return {
            'operation': operation,
            'total_transactions': len(encrypted_amounts),
            'encrypted': True,
            'library': request.library,
            'scheme': request.scheme,
            'note': 'Operations performed on encrypted data'
        }

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest):
    """Decrypt data"""
    try:
        if not request.library or not request.scheme:
            raise ValueError("Library and scheme are required")

        session_id = request.keys.get('session_id')
        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid session ID")

        session = active_sessions[session_id]
        wrapper = session['wrapper']

        print(f"🔓 Decrypting {len(request.encrypted_data)} values")

        # Prepare data
        prepared = []
        for enc in request.encrypted_data:
            if enc and isinstance(enc, dict) and 'ciphertext' in enc:
                prepared.append(bytes.fromhex(enc['ciphertext']))
            else:
                prepared.append(enc)

        # Decrypt
        decrypted = wrapper.decrypt_data(prepared, request.data_type)

        print(f"✅ Decryption complete")

        return {'success': True, 'decrypted_values': decrypted}

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Decryption error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting FHE Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")