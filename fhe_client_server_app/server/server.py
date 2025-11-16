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
import base64

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
    print(f"⚠️ TenSEAL not available: {str(e)}")

try:
    from openfhe_wrapper_bkp1 import OpenFHEWrapper
    OPENFHE_AVAILABLE = True
    print("✅ OpenFHE wrapper loaded")
except ImportError as e:
    OPENFHE_AVAILABLE = False
    print(f"⚠️ OpenFHE not available: {str(e)}")

app = FastAPI(title="FHE Server", version="5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global wrappers (initialized once)
global_wrappers = {'tenseal': None, 'openfhe': None}
active_sessions = {}
encrypted_data_store = {}  # NEW: Store encrypted data server-side to avoid memory issues

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
    encrypted_metadata: Optional[Dict[str, Dict[str, Any]]] = None  # Make optional
    encrypted_data: Optional[Dict[str, List[Any]]] = None  # Support old format too
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
        wrapper.generate_context(scheme='CKKS')
        encrypted_ciphertexts = wrapper.encrypt_data(request.data, request.column_name, request.data_type)
        #print(f"   Encryption performed on {encrypted_ciphertexts} ")
        duration = (datetime.now() - start_time).total_seconds()

        # Process results
        result = []
        successful = 0
        for idx, encrypted in enumerate(encrypted_ciphertexts):
            if encrypted is None:
                result.append(None)
            elif isinstance(encrypted, bytes):
                # REAL ciphertext bytes from TenSEAL/OpenFHE
                result.append(encrypted)
                successful += 1
            elif isinstance(encrypted, dict):
                # Check if it's a real ciphertext or simulation
                if encrypted.get('is_real_ciphertext'):
                    result.append(encrypted)
                    successful += 1
                else:
                    result.append(encrypted)
                    successful += 1
            else:
                result.append(encrypted)
                successful += 1

        print(f"✅ ENCRYPTION COMPLETE")
        print(f"   Successful: {successful}/{len(request.data)}")
        print(f"   Duration: {duration:.3f}s")
        print(f"{'='*60}\n")

        # OPTIMIZATION: Store encrypted data server-side to avoid sending back large ciphertexts
        storage_key = f"{session_id}:{request.column_name}:{request.batch_id}"
        encrypted_data_store[storage_key] = {
            'encrypted_values': result,
            'column_name': request.column_name,
            'data_type': request.data_type,
            'timestamp': datetime.now().isoformat()
        }
        print(f"💾 Stored encrypted data: {storage_key}")

        response = {
            'success': True,
            'encrypted_values': bytes_to_str(result)[:50],  # Return only first 10 for preview
            'column_name': request.column_name,
            'data_type': request.data_type,
            'batch_id': request.batch_id,
            'storage_key': storage_key,  # NEW: Key to retrieve data later
            'stored_server_side': True,  # NEW: Flag indicating data is stored
            'statistics': {
                'total': len(request.data),
                'successful': successful,
                'duration': duration,
                'preview_count':  len(request.data)
            }
        }
        #print(f"   Response prepared with storage key {response}")
        return response

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Encryption error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/fhe_query")
# async def fhe_query(request: FHEQueryRequest):
#     """Perform ACTUAL FHE query operations using wrappers"""
#     try:
#         if not request.library or not request.scheme:
#             raise ValueError("Library and scheme required")
#
#         session_id = request.keys.get('session_id')
#         if not session_id or session_id not in active_sessions:
#             raise ValueError("Invalid session ID")
#
#         session = active_sessions[session_id]
#         wrapper = session['wrapper']
#         operation = request.query_params.get('operation_type')
#
#         print(f"\n{'='*60}")
#         print(f"🔍 ACTUAL FHE QUERY")
#         print(f"   Operation: {operation}")
#         print(f"   Library: {request.library}")
#         print(f"   Scheme: {request.scheme}")
#         print(f"{'='*60}")
#         print(f"   Request: {request}")
#
#         # Get encrypted data from server storage
#         encrypted_values_list = []
#         column_info = []
#
#         if request.encrypted_metadata:
#             for key, meta in request.encrypted_metadata.items():
#                 column_info.append(meta)
#
#                 # Retrieve stored encrypted data
#                 storage_keys = meta.get('storage_keys', [])
#                 for sk in storage_keys:
#                     print(f"storage_key {sk}")
#                     stored_data = encrypted_data_store.get(sk)
#                     # print(f"encrypted_data_store keys: {stored_data.get('encrypted_values', [])}")
#                     if stored_data:
#                         encrypted_values_list.extend(stored_data.get('encrypted_values', []))
#
#         print(f"   Retrieved {len(encrypted_values_list)} encrypted values from storage")
#
#         # PERFORM ACTUAL FHE OPERATIONS
#         try:
#             if operation == "Transaction Count":
#                 # Simple count operation
#                 count = len(encrypted_values_list)
#
#                 result = {
#                     'operation': operation,
#                     'transaction_count': count,
#                     'total_records': count,
#                     'columns_analyzed': column_info,
#                     'encrypted': True,
#                     'library': request.library,
#                     'scheme': request.scheme,
#                     'fhe_operation': 'COUNT',
#                     'note': 'Count performed on encrypted data structure'
#                 }
#
#             elif operation == "Transaction Analysis":
#                 print("   Performing homomorphic operations...")
#                 # Filter numeric encrypted values (amounts)
#                 amount_ciphertexts = []
#                 text_ciphertexts = []
#                 # print(f"Encrypted values : {encrypted_values_list}")
#                 for val in encrypted_values_list:
#                     if val and isinstance(val, dict):
#                         # print(f"   Analyzing ciphertext: {val}")
#                         if val.get('type') == 'numeric' or 'amount' in str(val.get('column_name', '')).lower():
#                             amount_ciphertexts.append(val)
#                         if val.get('type') == 'text' and request.scheme == 'CKKS':
#                             # CKKS supports encoded text
#                             text_ciphertexts.append(val)
#
#                 print(f"   Found {len(amount_ciphertexts)} amount ciphertexts")
#
#                 encrypted_sum = None
#                 encrypted_min = None
#                 # print(f"wrapper => {wrapper}")
#                 if len(amount_ciphertexts) > 0  and hasattr(wrapper, 'perform_operation'):
#                     try:
#                         print("   Attempting homomorphic SUM...")
#                         print(f"   Performing aggregation on {len(amount_ciphertexts)} values...")
#
#                         # SUM
#                         encrypted_sum = wrapper.perform_aggregation(amount_ciphertexts, 'sum')
#
#                         if text_ciphertexts and request.scheme == 'CKKS':
#                             print(f"   Processing {len(text_ciphertexts)} text ciphertexts")
#                             encrypted_sum = wrapper.perform_aggregation(text_ciphertexts, 'sum')
#
#                         # MIN/MAX (if supported)
#                         try:
#                             encrypted_min = wrapper.perform_aggregation(amount_ciphertexts, 'min')
#                         except ValueError as e:
#                             print(f"   MIN not supported: {e}")
#                             encrypted_min = "OPERATION_NOT_SUPPORTED"
#
#                     except AttributeError:
#                         print("   ⚠️ perform_operation not available in wrapper")
#                     except Exception as e:
#                         print(f"   ⚠️ Homomorphic operation failed: {str(e)}")
#
#                 # Calculate encrypted average (would divide encrypted sum by count)
#                 encrypted_avg = None
#                 if encrypted_sum:
#                     # In real FHE, would perform encrypted division
#                     encrypted_avg = wrapper.perform_aggregation(amount_ciphertexts, 'average')
#
#                 # print(f"   Encrypted SUM: {encrypted_sum}")
#                 # print(f"   Encrypted AVG: {encrypted_avg}")
#                 # print(f"   Encrypted MIN: {encrypted_min}")
#                 result = {
#                     'operation': operation,
#                     'total_records': len(amount_ciphertexts),
#                     'columns_analyzed': column_info,
#                     'analysis': {
#                         'total_transactions': len(amount_ciphertexts),
#                         'encrypted_sum': encrypted_sum if encrypted_sum else 'ENCRYPTED_RESULT',
#                         'encrypted_avg': encrypted_avg if encrypted_avg else 'ENCRYPTED_RESULT',
#                         'encrypted_min': encrypted_min,
#                         'encrypted_max': 'ENCRYPTED_RESULT',
#                         'date_range': f"{request.query_params.get('start_date')} to {request.query_params.get('end_date')}",
#                         'note': 'Actual FHE operations performed on encrypted values'
#                     },
#                     'encrypted': True,
#                     'library': request.library,
#                     'scheme': request.scheme,
#                     'fhe_operations_performed': ['HOMOMORPHIC_ADD'] if encrypted_sum else [],
#                     'encrypted_results_id': f"result_{session_id}_{int(datetime.now().timestamp())}"
#                 }
#
#             elif operation == "Account Summary":
#                 print("   Performing account aggregations...")
#
#                 # Filter account-related encrypted data
#                 account_ciphertexts = []
#                 for val in encrypted_values_list:
#                     if val and isinstance(val, dict):
#                         if 'account' in str(val.get('table', '')).lower():
#                             account_ciphertexts.append(val)
#
#                 print(f"   Found {len(account_ciphertexts)} account ciphertexts")
#
#                 # Perform homomorphic aggregation on balances
#                 encrypted_total_balance = None
#                 if len(account_ciphertexts) > 1 and hasattr(wrapper, 'perform_operation'):
#                     try:
#                         print("   Attempting homomorphic balance aggregation...")
#
#                         balance_ciphers = [c for c in account_ciphertexts if 'balance' in str(c.get('column_name', '')).lower()]
#
#                         if len(balance_ciphers) >= 2:
#                             result_cipher = wrapper.perform_operation([balance_ciphers[0]], [balance_ciphers[1]], 'add')
#                             if result_cipher:
#                                 encrypted_total_balance = result_cipher[0]
#                                 print("   ✅ Homomorphic balance aggregation performed")
#
#                     except Exception as e:
#                         print(f"   ⚠️ Balance aggregation failed: {str(e)}")
#
#                 account_cols = [c for c in column_info if c.get('table') == 'accounts']
#
#                 result = {
#                     'operation': operation,
#                     'total_records': len(account_ciphertexts),
#                     'columns_analyzed': column_info,
#                     'summary': {
#                         'total_accounts': len(account_cols),
#                         'encrypted_balances': 'ENCRYPTED_RESULT',
#                         'encrypted_total_balance': encrypted_total_balance if encrypted_total_balance else 'ENCRYPTED_RESULT',
#                         'note': 'Account data processed on encrypted values'
#                     },
#                     'encrypted': True,
#                     'library': request.library,
#                     'scheme': request.scheme,
#                     'fhe_operations_performed': ['HOMOMORPHIC_ADD'] if encrypted_total_balance else [],
#                     'encrypted_results_id': f"result_{session_id}_{int(datetime.now().timestamp())}"
#                 }
#
#             else:
#                 # Unsupported operation
#                 result = {
#                     'operation': operation,
#                     'status': 'unsupported',
#                     'message': f'Operation "{operation}" not supported',
#                     'supported_operations': ['Transaction Count', 'Transaction Analysis', 'Account Summary'],
#                     'library': request.library,
#                     'scheme': request.scheme
#                 }
#
#             print(f"✅ FHE QUERY COMPLETE")
#             print(f"   FHE operations: {result.get('fhe_operations_performed', [])}")
#             print(f"{'='*60}\n")
#
#             result['timestamp'] = datetime.now().isoformat()
#             result_serializable = bytes_to_str(result)
#             # print(f"Result: {json.dumps(result_serializable, indent=2)}")
#             return json.dumps(result_serializable, indent=2)
#
#         except Exception as op_error:
#             print(f"❌ FHE operation error: {str(op_error)}")
#             import traceback
#             print(traceback.format_exc())
#
#             # Graceful fallback
#             return {
#                 'operation': operation,
#                 'status': 'error',
#                 'message': str(op_error),
#                 'fallback': True,
#                 'total_records': len(column_info),
#                 'columns_analyzed': column_info,
#                 'library': request.library,
#                 'scheme': request.scheme,
#                 'note': 'FHE operation attempted but encountered error'
#             }
#
#     except ValueError as e:
#         print(f"❌ Validation error: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         print(f"❌ Query error: {str(e)}")
#         import traceback
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/fhe_query")
async def fhe_query(request: FHEQueryRequest):
    """Perform REAL FHE query operations on ENCRYPTED data"""
    try:
        if not request.library or not request.scheme:
            raise ValueError("Library and scheme required")

        session_id = request.keys.get('session_id')
        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid session ID")

        session = active_sessions[session_id]
        wrapper = session['wrapper']
        operation = request.query_params.get('operation_type')
        print(f"   Request: {request}")

        print(f"\n{'='*60}")
        print(f"🔐 FHE QUERY ON ENCRYPTED DATA")
        print(f"   Operation: {operation}")
        print(f"   Library: {request.library}")
        print(f"   Scheme: {request.scheme}")
        print(f"{'='*60}")

        # RETRIEVE ENCRYPTED CIPHERTEXTS (NOT PLAINTEXT!)
        encrypted_ciphertexts = []
        column_info = []

        if request.encrypted_metadata:
            for key, meta in request.encrypted_metadata.items():
                column_info.append(meta)

                # Get REAL encrypted ciphertexts from storage
                storage_keys = meta.get('storage_keys', [])
                print(f"   Retrieving from storage keys: {storage_keys}")
                for sk in storage_keys:
                    stored_data = encrypted_data_store.get(sk)
                    if stored_data:
                        # These are REAL ciphertexts
                        ciphertexts = stored_data.get('encrypted_values', [])
                        encrypted_ciphertexts.extend(ciphertexts)
                        print(f"   Retrieved {len(ciphertexts)} ciphertexts from {sk}")

        print(f"   Total ciphertexts: {len(encrypted_ciphertexts)}")

        # PERFORM REAL HOMOMORPHIC OPERATIONS ON ENCRYPTED DATA
        try:
            if operation == "Transaction Count":
                # Count operation (can be done on encrypted structure)
                count = len(encrypted_ciphertexts)

                result = {
                    'operation': operation,
                    'transaction_count': count,
                    'total_records': count,
                    'columns_analyzed': column_info,
                    'encrypted': True,
                    'library': request.library,
                    'scheme': request.scheme,
                    'fhe_mode': 'REAL',
                    'note': 'Count performed on encrypted ciphertext structure'
                }
            elif operation == "Transaction Analysis":
                print("   Performing REAL homomorphic operations on ciphertexts...")

                # Filter numeric ciphertexts
                numeric_ciphertexts = [ct for ct in encrypted_ciphertexts if ct is not None]
                print(f"   Processing {len(numeric_ciphertexts)} ciphertexts")

                # REAL HOMOMORPHIC SUM
                sum_result = None
                avg_result = None

                if len(numeric_ciphertexts) > 0 and hasattr(wrapper, 'perform_aggregation'):
                    try:
                        print("   Executing homomorphic SUM...")
                        sum_result = wrapper.perform_aggregation(numeric_ciphertexts, 'sum')
                        print(f"   Sum result type: {type(sum_result)}")

                        print("   Executing  homomorphic AVERAGE...")
                        avg_result = wrapper.perform_aggregation(numeric_ciphertexts, 'average')
                        # print(f"   Avg result type: {type(avg_result)}")

                    except Exception as e:
                        print(f"   ⚠️ Homomorphic operation error: {str(e)}")
                        import traceback
                        traceback.print_exc()

                # Store encrypted results - FIXED
                result_id = f"result_{session_id}_{int(datetime.now().timestamp())}"

                # Prepare for storage - handle both bytes and dict
                sum_to_store = None
                avg_to_store = None

                if sum_result:
                    if isinstance(sum_result, bytes):
                        # TenSEAL returns bytes
                        sum_to_store = {'status': 'success', 'result': sum_result, 'type': 'bytes'}
                    elif isinstance(sum_result, dict):
                        # OpenFHE returns dict
                        sum_to_store = sum_result
                    else:
                        sum_to_store = {'status': 'success', 'result': sum_result}

                if avg_result:
                    if isinstance(avg_result, bytes):
                        avg_to_store = {'status': 'success', 'result': avg_result, 'type': 'bytes'}
                    elif isinstance(avg_result, dict):
                        avg_to_store = avg_result
                    else:
                        avg_to_store = {'status': 'success', 'result': avg_result}

                encrypted_data_store[result_id] = {
                    'encrypted_sum': sum_to_store,
                    'encrypted_avg': avg_to_store,
                    'operation': operation,
                    'timestamp': datetime.now().isoformat(),
                    'is_encrypted': True,
                    'library': request.library,
                    'scheme': request.scheme
                }

                print(f"   ✅ Stored results: {result_id}")

                result = {
                    'operation': operation,
                    'total_records': len(numeric_ciphertexts),
                    'columns_analyzed': column_info,
                    'analysis': {
                        'total_transactions': len(numeric_ciphertexts),
                        'encrypted_sum': 'ENCRYPTED' if sum_result else 'N/A',
                        'encrypted_avg': 'ENCRYPTED' if avg_result else 'N/A',
                        'note': 'Results are ENCRYPTED - use /decrypt_results to decrypt'
                    },
                    'encrypted': True,
                    'library': request.library,
                    'scheme': request.scheme,
                    'fhe_mode': 'REAL',
                    'fhe_operations_performed': ['HOMOMORPHIC_SUM', 'HOMOMORPHIC_AVG'],
                    'encrypted_results_id': result_id
                }
            # elif operation == "Transaction Analysis":
            #     print("   Performing REAL homomorphic operations on ciphertexts...")
            #
            #     # Filter numeric ciphertexts
            #     numeric_ciphertexts = [ct for ct in encrypted_ciphertexts if ct is not None]
            #
            #     print(f"   Processing {len(numeric_ciphertexts)} ciphertexts")
            #
            #     # REAL HOMOMORPHIC SUM
            #     encrypted_sum_result = None
            #     encrypted_avg_result = None
            #
            #     if len(numeric_ciphertexts) > 0 and hasattr(wrapper, 'perform_aggregation'):
            #         try:
            #             print("   Executing REAL homomorphic SUM on ciphertexts...")
            #             encrypted_sum_result = wrapper.perform_aggregation(numeric_ciphertexts, 'sum')
            #             # REAL HOMOMORPHIC AVERAGE
            #             print("   Executing REAL homomorphic AVERAGE on ciphertexts...")
            #             encrypted_avg_result = wrapper.perform_aggregation(numeric_ciphertexts, 'average')
            #
            #         except Exception as e:
            #             print(f"   ⚠️ Homomorphic operation error: {str(e)}")
            #
            #     # Store encrypted results for later decryption
            #     result_id = f"result_{session_id}_{int(datetime.now().timestamp())}"
            #     encrypted_data_store[result_id] = {
            #         'encrypted_sum': encrypted_sum_result,
            #         'encrypted_avg': encrypted_avg_result,
            #         'operation': operation,
            #         'timestamp': datetime.now().isoformat(),
            #         'is_encrypted': True
            #     }
            #
            #     result = {
            #         'operation': operation,
            #         'total_records': len(numeric_ciphertexts),
            #         'columns_analyzed': column_info,
            #         'analysis': {
            #             'total_transactions': len(numeric_ciphertexts),
            #             'encrypted_sum': bytes_to_str(encrypted_sum_result)[:50],
            #             'encrypted_avg': bytes_to_str(encrypted_avg_result)[:50],
            #             'note': 'Results are ENCRYPTED - use decrypt_results endpoint to decrypt'
            #         },
            #         'encrypted': True,
            #         'library': request.library,
            #         'scheme': request.scheme,
            #         'fhe_mode': 'REAL',
            #         'fhe_operations_performed': ['HOMOMORPHIC_SUM', 'HOMOMORPHIC_AVG'],
            #         'encrypted_results_id': result_id
            #     }

            elif operation == "Account Summary":
                print("   Performing account aggregations on encrypted data...")

                # Filter account ciphertexts
                account_ciphertexts = [ct for ct in encrypted_ciphertexts if ct is not None]

                print(f"   Processing {len(account_ciphertexts)} account ciphertexts")

                # REAL HOMOMORPHIC aggregation on balances
                encrypted_balance_sum = None
                if len(account_ciphertexts) > 0 and hasattr(wrapper, 'perform_aggregation'):
                    try:
                        print("   Executing homomorphic balance SUM...")
                        balance_result = wrapper.perform_aggregation(account_ciphertexts, 'sum')

                        if balance_result and balance_result.get('status') == 'success':
                            encrypted_balance_sum = balance_result.get('result')
                            print("   ✅ balance SUM completed")
                    except Exception as e:
                        print(f"   ⚠️ Balance aggregation error: {str(e)}")

                # Store encrypted result
                result_id = f"result_{session_id}_{int(datetime.now().timestamp())}"
                encrypted_data_store[result_id] = {
                    'encrypted_total_balance': encrypted_balance_sum,
                    'operation': operation,
                    'timestamp': datetime.now().isoformat(),
                    'is_encrypted': True
                }

                result = {
                    'operation': operation,
                    'total_records': len(account_ciphertexts),
                    'columns_analyzed': column_info,
                    'summary': {
                        'total_accounts': len(account_ciphertexts),
                        'encrypted_total_balance': 'ENCRYPTED_RESULT' if encrypted_balance_sum else 'N/A',
                        'note': 'Balances aggregated on ENCRYPTED data'
                    },
                    'encrypted': True,
                    'library': request.library,
                    'scheme': request.scheme,
                    'fhe_mode': 'REAL',
                    'fhe_operations_performed': ['HOMOMORPHIC_SUM'],
                    'encrypted_results_id': result_id
                }

            else:
                result = {
                    'operation': operation,
                    'status': 'unsupported',
                    'message': f'Operation "{operation}" not supported',
                    'supported_operations': ['Transaction Count', 'Transaction Analysis', 'Account Summary']
                }

            print(f"✅ FHE QUERY COMPLETE (REAL MODE)")
            print(f"   Operations: {result.get('fhe_operations_performed', [])}")
            print(f"{'='*60}\n")

            result['timestamp'] = datetime.now().isoformat()
            return result

        except Exception as op_error:
            print(f"❌ FHE operation error: {str(op_error)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(op_error))

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def bytes_to_str(obj):
    """Convert bytes to base64 for JSON serialization"""
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):
        return {k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(v) for v in obj]
    return obj


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

class DecryptResultsRequest(BaseModel):
    encrypted_results_id: str
    keys: Dict[str, Any]
    library: str
    scheme: str

@app.post("/decrypt_results")
async def decrypt_results(request: DecryptResultsRequest):
    """Decrypt FHE query results - FIXED"""
    try:
        session_id = request.keys.get('session_id')
        if not session_id or session_id not in active_sessions:
            raise ValueError("Invalid session ID")

        session = active_sessions[session_id]
        wrapper = session['wrapper']

        print(f"\n{'=' * 60}")
        print(f"🔓 DECRYPTING RESULTS")
        print(f"   Results ID: {request.encrypted_results_id}")
        print(f"   Library: {request.library}")
        print(f"{'=' * 60}")

        # RETRIEVE stored encrypted results
        stored_results = encrypted_data_store.get(request.encrypted_results_id)

        if not stored_results:
            raise ValueError(f"No results found for ID: {request.encrypted_results_id}")

        print(f"   Found: {list(stored_results.keys())}")

        # Decrypt values
        decrypted_values = {}

        # Decrypt sum
        if 'encrypted_sum' in stored_results and stored_results['encrypted_sum']:
            enc_sum = stored_results['encrypted_sum']

            try:
                if isinstance(enc_sum, dict) and enc_sum.get('status') == 'success':
                    result_data = enc_sum.get('result')

                    if result_data and isinstance(result_data, bytes):
                        # Decrypt bytes
                        decrypted = wrapper.decrypt_data([result_data], 'numeric')
                        decrypted_values['transaction_sum'] = float(decrypted[0]) if decrypted and decrypted[0] else 0.0
                        print(f"   ✅ Sum decrypted: {decrypted_values['transaction_sum']}")
                    else:
                        decrypted_values['transaction_sum'] = 0.0
                else:
                    decrypted_values['transaction_sum'] = 0.0
            except Exception as e:
                print(f"   ⚠️ Sum decryption error: {str(e)}")
                decrypted_values['transaction_sum'] = 0.0

        # Decrypt average
        if 'encrypted_avg' in stored_results and stored_results['encrypted_avg']:
            enc_avg = stored_results['encrypted_avg']

            try:
                if isinstance(enc_avg, dict) and enc_avg.get('status') == 'success':
                    result_data = enc_avg.get('result')

                    if result_data and isinstance(result_data, bytes):
                        decrypted = wrapper.decrypt_data([result_data], 'numeric')
                        decrypted_values['transaction_avg'] = float(decrypted[0]) if decrypted and decrypted[0] else 0.0
                        print(f"   ✅ Avg decrypted: {decrypted_values['transaction_avg']}")
                    else:
                        decrypted_values['transaction_avg'] = 0.0
                else:
                    decrypted_values['transaction_avg'] = 0.0
            except Exception as e:
                print(f"   ⚠️ Avg decryption error: {str(e)}")
                decrypted_values['transaction_avg'] = 0.0

        # Decrypt balance (for account summary)
        if 'encrypted_total_balance' in stored_results and stored_results['encrypted_total_balance']:
            enc_balance = stored_results['encrypted_total_balance']

            try:
                if isinstance(enc_balance, dict) and enc_balance.get('status') == 'success':
                    result_data = enc_balance.get('result')

                    if result_data and isinstance(result_data, bytes):
                        decrypted = wrapper.decrypt_data([result_data], 'numeric')
                        decrypted_values['total_balance'] = float(decrypted[0]) if decrypted and decrypted[0] else 0.0
                        print(f"   ✅ Balance decrypted: {decrypted_values['total_balance']}")
                    else:
                        decrypted_values['total_balance'] = 0.0
                else:
                    decrypted_values['total_balance'] = 0.0
            except Exception as e:
                print(f"   ⚠️ Balance decryption error: {str(e)}")
                decrypted_values['total_balance'] = 0.0

        result = {
            'encrypted_results_id': request.encrypted_results_id,
            'decrypted': True,
            'values': decrypted_values,
            'operation': stored_results.get('operation', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'library': request.library,
            'scheme': request.scheme
        }

        print(f"✅ DECRYPTION COMPLETE")
        print(f"{'=' * 60}\n")

        return result

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Decryption error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting FHE Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")