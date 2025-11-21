"""
FHE Server Application - FastAPI Backend
Performs real homomorphic encryption operations using OpenFHE and TenSEAL
Compatible with Python 3.11
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
import base64
from pathlib import Path
import time

# Import FHE wrappers
try:
    from openfhe_wrapper import OpenFHEWrapper
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False
    print("⚠️ OpenFHE wrapper not available")

try:
    from tenseal_wrapper import TenSEALWrapper
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("⚠️ TenSEAL wrapper not available")

# Initialize FastAPI
app = FastAPI(title="FHE Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FHE instances (initialized once)
openfhe_instance = None
tenseal_instance = None
encrypted_storage = {}  # Store encrypted data with metadata
metadata_storage = {}  # Store column metadata
transaction_metadata = {}  # Store transaction-level metadata

# Pydantic models
class ContextConfig(BaseModel):
    library: str
    scheme: str
    poly_modulus_degree: int = 8192
    scale: Optional[float] = None
    coeff_mod_bit_sizes: Optional[List[int]] = None
    plain_modulus: Optional[int] = None
    mult_depth: int = 10
    scale_mod_size: int = 50

class KeyGenerationRequest(BaseModel):
    library: str
    scheme: str
    params: Dict[str, Any]

class EncryptionRequest(BaseModel):
    library: str
    scheme: str
    column_name: str
    data_type: str
    data: List[Any]
    batch_id: str
    party_ids: Optional[List[str]] = None  # Store party IDs for filtering
    payment_dates: Optional[List[str]] = None  # Store dates for filtering

class AggregationRequest(BaseModel):
    library: str
    operation: str
    batch_ids: List[str]
    column_name: str
    party_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    currency: Optional[str] = None

class DecryptionRequest(BaseModel):
    library: str
    result_data: Any
    data_type: str = "numeric"

class TransactionQueryRequest(BaseModel):
    library: str
    party_id: str
    start_date: str
    end_date: str
    currency: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize FHE wrappers on server startup"""
    global openfhe_instance, tenseal_instance

    print("\n" + "="*60)
    print("🚀 FHE SERVER STARTING")
    print("="*60)

    if OPENFHE_AVAILABLE:
        try:
            openfhe_instance = OpenFHEWrapper()
            print("✅ OpenFHE wrapper initialized")
        except Exception as e:
            print(f"⚠️ OpenFHE initialization failed: {e}")

    if TENSEAL_AVAILABLE:
        try:
            tenseal_instance = TenSEALWrapper()
            print("✅ TenSEAL wrapper initialized")
        except Exception as e:
            print(f"⚠️ TenSEAL initialization failed: {e}")

    print("="*60)
    print("⚡ Server performs REAL FHE operations on encrypted data")
    print("="*60 + "\n")

@app.get("/")
async def root():
    return {
        "message": "FHE Server API",
        "version": "1.0.0",
        "libraries": {
            "OpenFHE": OPENFHE_AVAILABLE,
            "TenSEAL": TENSEAL_AVAILABLE
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openfhe_ready": openfhe_instance is not None,
        "tenseal_ready": tenseal_instance is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate_context")
async def generate_context(config: ContextConfig):
    """Generate FHE context with specified parameters"""
    print(f"\n🔧 Generating context: {config.library} - {config.scheme}")

    try:
        if config.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            context = openfhe_instance.generate_context(
                scheme=config.scheme,
                mult_depth=config.mult_depth,
                scale_mod_size=config.scale_mod_size,
                ring_dim=config.poly_modulus_degree
            )

        elif config.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            context = tenseal_instance.generate_context(
                scheme=config.scheme,
                poly_modulus_degree=config.poly_modulus_degree,
                coeff_mod_bit_sizes=config.coeff_mod_bit_sizes or [60, 40, 40, 60],
                scale=config.scale or 2**40,
                plain_modulus=config.plain_modulus or 1032193
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        print("✅ Context generated successfully")
        return {
            "status": "success",
            "library": config.library,
            "scheme": config.scheme,
            "message": "Context generated"
        }

    except Exception as e:
        print(f"❌ Context generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_keys")
async def generate_keys(request: KeyGenerationRequest):
    """Generate encryption keys"""
    print(f"\n🔑 Generating keys: {request.library} - {request.scheme}")

    try:
        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            openfhe_instance.generate_context(
                scheme=request.scheme,
                mult_depth=request.params.get('mult_depth', 10),
                scale_mod_size=request.params.get('scale_mod_size', 50),
                ring_dim=request.params.get('poly_modulus_degree', 8192)
            )

            keys_info = openfhe_instance.get_keys_info()

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            tenseal_instance.generate_context(
                scheme=request.scheme,
                poly_modulus_degree=request.params.get('poly_modulus_degree', 8192),
                scale=request.params.get('scale', 2**40)
            )

            keys_info = tenseal_instance.get_keys_info()
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        print("✅ Keys generated successfully")
        return {
            "status": "success",
            "keys": keys_info
        }

    except Exception as e:
        print(f"❌ Key generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest):
    """Encrypt data column - REAL FHE OPERATION"""
    print(f"\n🔒 Encrypting {len(request.data)} values ({request.column_name})")
    print(f"   Library: {request.library}, Scheme: {request.scheme}")
    print("   ⚡ PERFORMING REAL FHE ENCRYPTION ON ENCRYPTED DATA")

    start_time = time.time()

    try:
        # Convert date strings to timestamps if needed
        processed_data = []
        for item in request.data:
            if request.data_type == "date" and item is not None:
                try:
                    timestamp = pd.Timestamp(item).timestamp()
                    processed_data.append(timestamp)
                except:
                    processed_data.append(item)
            else:
                processed_data.append(item)

        if request.library == "OpenFHE":
            if not openfhe_instance or not openfhe_instance.context:
                raise HTTPException(status_code=400, detail="Context not initialized")

            encrypted_data = openfhe_instance.encrypt_data(
                processed_data,
                request.column_name,
                request.data_type
            )

        elif request.library == "TenSEAL":
            if not tenseal_instance or not tenseal_instance.context:
                raise HTTPException(status_code=400, detail="Context not initialized")

            encrypted_data = tenseal_instance.encrypt_data(
                processed_data,
                request.column_name,
                request.data_type
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        # Store encrypted data with transaction metadata
        storage_key = f"{request.batch_id}_{request.column_name}"

        # Create storage with metadata for filtering
        encrypted_storage[storage_key] = {
            'encrypted_values': encrypted_data,
            'party_ids': request.party_ids if request.party_ids else [],
            'payment_dates': request.payment_dates if request.payment_dates else [],
            'original_values': request.data  # Store originals for reconciliation
        }

        # Generate metadata (return max 100 records)
        metadata_records = []
        for i, (original, encrypted) in enumerate(zip(request.data[:100], encrypted_data[:100])):
            if encrypted is not None:
                metadata_records.append({
                    "index": i,
                    "original_value": str(original),
                    "encrypted": True,
                    "ciphertext_size": len(str(encrypted)) if encrypted else 0,
                    "scheme": request.scheme
                })

        # Store metadata
        metadata_storage[storage_key] = {
            "column_name": request.column_name,
            "data_type": request.data_type,
            "count": len(encrypted_data),
            "batch_id": request.batch_id,
            "library": request.library,
            "scheme": request.scheme,
            "timestamp": datetime.now().isoformat()
        }

        elapsed_time = time.time() - start_time
        print(f"✅ Encryption complete: {len(encrypted_data)} values in {elapsed_time:.2f}s")

        return {
            "status": "success",
            "batch_id": request.batch_id,
            "column_name": request.column_name,
            "encrypted_count": len(encrypted_data),
            "metadata_records": metadata_records,
            "encryption_time": elapsed_time
        }

    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/aggregate")
async def perform_aggregation(request: AggregationRequest):
    """Perform FHE aggregation operations - REAL HOMOMORPHIC COMPUTATION"""
    print(f"\n📊 Performing FHE aggregation: {request.operation}")
    print(f"   Library: {request.library}, Column: {request.column_name}")
    print(f"   Filters: party_id={request.party_id}, dates={request.start_date} to {request.end_date}")
    print("   ⚡ PERFORMING REAL FHE OPERATIONS ON ENCRYPTED DATA")

    start_time = time.time()

    try:
        # Collect encrypted data from all batches with filtering
        filtered_encrypted = []
        filtered_original = []  # For reconciliation

        for batch_id in request.batch_ids:
            storage_key = f"{batch_id}_{request.column_name}"
            if storage_key not in encrypted_storage:
                continue

            storage = encrypted_storage[storage_key]
            encrypted_values = storage['encrypted_values']
            party_ids = storage.get('party_ids', [])
            payment_dates = storage.get('payment_dates', [])
            original_values = storage.get('original_values', [])

            # Apply filters
            for i, enc_val in enumerate(encrypted_values):
                if enc_val is None:
                    continue

                # Check party_id filter
                if request.party_id and i < len(party_ids):
                    if party_ids[i] != request.party_id:
                        continue

                # Check date range filter
                if request.start_date and request.end_date and i < len(payment_dates):
                    try:
                        payment_date = pd.Timestamp(payment_dates[i])
                        start = pd.Timestamp(request.start_date)
                        end = pd.Timestamp(request.end_date)

                        if not (start <= payment_date <= end):
                            continue
                    except:
                        pass

                # Add to filtered list
                filtered_encrypted.append(enc_val)
                if i < len(original_values):
                    filtered_original.append(original_values[i])

        if not filtered_encrypted:
            print("⚠️ No data matched the filters")
            return {
                "status": "success",
                "message": "No data found matching filters",
                "result": None,
                "count": 0,
                "expected_value": None,
                "filtered_count": 0
            }

        print(f"   Filtered to {len(filtered_encrypted)} values")

        # Calculate expected value for reconciliation
        expected_value = None
        if filtered_original:
            try:
                numeric_values = [float(v) for v in filtered_original if v is not None]
                if request.operation in ['sum', 'add']:
                    expected_value = sum(numeric_values)
                elif request.operation in ['average', 'avg']:
                    expected_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                elif request.operation == 'min':
                    expected_value = min(numeric_values) if numeric_values else None
                elif request.operation == 'max':
                    expected_value = max(numeric_values) if numeric_values else None
            except Exception as e:
                print(f"⚠️ Could not calculate expected value: {e}")

        # Perform aggregation
        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            result = openfhe_instance.perform_aggregation(filtered_encrypted, request.operation)

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            result = tenseal_instance.perform_aggregation(filtered_encrypted, request.operation)
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        # Serialize result
        def serialize_result(result):
            if result is None:
                return None
            if isinstance(result, bytes):
                return {
                    "data": base64.b64encode(result).decode('utf-8'),
                    "type": "bytes"
                }
            elif isinstance(result, dict):
                serialized = {}
                for key, value in result.items():
                    if isinstance(value, bytes):
                        serialized[key] = {
                            "data": base64.b64encode(value).decode('utf-8'),
                            "type": "bytes"
                        }
                    else:
                        serialized[key] = value
                return serialized
            return result

        elapsed_time = time.time() - start_time
        print(f"✅ Aggregation complete in {elapsed_time:.2f}s")

        return {
            "status": "success",
            "operation": request.operation,
            "result": serialize_result(result),
            "count": len(filtered_encrypted),
            "computation_time": elapsed_time,
            "encrypted": True,
            "expected_value": expected_value,  # For reconciliation
            "filtered_count": len(filtered_encrypted)
        }

    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decrypt")
async def decrypt_result(request: DecryptionRequest):
    """Decrypt encrypted result"""
    print(f"\n🔓 Decrypting result")

    try:
        # Handle base64-encoded data
        result_data = request.result_data
        if isinstance(result_data, dict) and result_data.get('type') == 'bytes':
            result_data = base64.b64decode(result_data['data'])
        elif isinstance(result_data, dict) and 'data' in result_data:
            if isinstance(result_data['data'], dict) and result_data['data'].get('type') == 'bytes':
                result_data = base64.b64decode(result_data['data']['data'])

        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            decrypted = openfhe_instance.decrypt_result(
                result_data,
                request.data_type
            )

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            if isinstance(result_data, bytes):
                decrypted_list = tenseal_instance.decrypt_data([result_data], request.data_type)
                decrypted = decrypted_list[0] if decrypted_list else None
            elif isinstance(result_data, dict):
                if 'simulated_value' in result_data:
                    decrypted = result_data['simulated_value']
                else:
                    decrypted = result_data
            else:
                decrypted = result_data
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        print(f"✅ Decryption complete: {decrypted}")

        return {
            "status": "success",
            "decrypted_value": float(decrypted) if decrypted is not None else None
        }

    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_transactions")
async def query_transactions(request: TransactionQueryRequest):
    """Query transactions by party_id and date range - FHE operations"""
    print(f"\n🔍 Querying transactions for party: {request.party_id}")
    print(f"   Date range: {request.start_date} to {request.end_date}")
    print("   ⚡ PERFORMING FHE OPERATIONS ON ENCRYPTED DATA")

    start_time = time.time()

    try:
        # Collect all batch IDs
        batch_ids = list(set(m.get('batch_id') for m in metadata_storage.values()))

        if not batch_ids:
            return {
                "status": "success",
                "message": "No encrypted data found",
                "results": {},
                "party_id": request.party_id,
                "date_range": f"{request.start_date} to {request.end_date}"
            }

        # Perform aggregations for different columns
        results = {}
        columns_to_aggregate = ['balance', 'amount_transferred']
        operations = ['sum', 'avg']

        for column in columns_to_aggregate:
            # Check if column exists in encrypted storage
            column_exists = any(
                column in storage_key
                for storage_key in encrypted_storage.keys()
            )

            if not column_exists:
                continue

            for operation in operations:
                # Filter encrypted data
                filtered_encrypted = []
                filtered_original = []

                for batch_id in batch_ids:
                    storage_key = f"{batch_id}_{column}"
                    if storage_key not in encrypted_storage:
                        continue

                    storage = encrypted_storage[storage_key]
                    encrypted_values = storage['encrypted_values']
                    party_ids = storage.get('party_ids', [])
                    payment_dates = storage.get('payment_dates', [])
                    original_values = storage.get('original_values', [])

                    # Apply filters
                    for i, enc_val in enumerate(encrypted_values):
                        if enc_val is None:
                            continue

                        # Check party_id filter
                        if request.party_id and i < len(party_ids):
                            if party_ids[i] != request.party_id:
                                continue

                        # Check date range filter
                        if request.start_date and request.end_date and i < len(payment_dates):
                            try:
                                payment_date = pd.Timestamp(payment_dates[i])
                                start = pd.Timestamp(request.start_date)
                                end = pd.Timestamp(request.end_date)

                                if not (start <= payment_date <= end):
                                    continue
                            except:
                                pass

                        # Add to filtered list
                        filtered_encrypted.append(enc_val)
                        if i < len(original_values):
                            filtered_original.append(original_values[i])

                if not filtered_encrypted:
                    continue

                # Calculate expected value
                expected_value = None
                if filtered_original:
                    try:
                        numeric_values = [float(v) for v in filtered_original if v is not None]
                        if operation in ['sum', 'add']:
                            expected_value = sum(numeric_values)
                        elif operation in ['average', 'avg']:
                            expected_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                    except Exception as e:
                        print(f"⚠️ Could not calculate expected value: {e}")

                # Perform aggregation
                print(f"   Aggregating {column} ({operation}): {len(filtered_encrypted)} values")

                if request.library == "OpenFHE":
                    if not openfhe_instance:
                        continue
                    result = openfhe_instance.perform_aggregation(filtered_encrypted, operation)
                elif request.library == "TenSEAL":
                    if not tenseal_instance:
                        continue
                    result = tenseal_instance.perform_aggregation(filtered_encrypted, operation)
                else:
                    continue

                # Serialize result
                def serialize_result(result):
                    if result is None:
                        return None
                    if isinstance(result, bytes):
                        return {
                            "data": base64.b64encode(result).decode('utf-8'),
                            "type": "bytes"
                        }
                    elif isinstance(result, dict):
                        serialized = {}
                        for key, value in result.items():
                            if isinstance(value, bytes):
                                serialized[key] = {
                                    "data": base64.b64encode(value).decode('utf-8'),
                                    "type": "bytes"
                                }
                            else:
                                serialized[key] = value
                        return serialized
                    return result

                # Store result
                result_key = f"{column}_{operation}"
                results[result_key] = {
                    "result": serialize_result(result),
                    "expected_value": expected_value,
                    "count": len(filtered_encrypted),
                    "encrypted": True
                }

        elapsed_time = time.time() - start_time
        print(f"✅ Query complete in {elapsed_time:.2f}s")

        return {
            "status": "success",
            "party_id": request.party_id,
            "date_range": f"{request.start_date} to {request.end_date}",
            "results": results,
            "computation_time": elapsed_time
        }

    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheme_limitations/{library}/{scheme}")
async def get_scheme_limitations(library: str, scheme: str):
    """Get scheme limitations"""
    try:
        if library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")
            openfhe_instance.scheme = scheme
            limitations = openfhe_instance.get_scheme_limitations()

        elif library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")
            tenseal_instance.scheme = scheme
            limitations = tenseal_instance.get_scheme_limitations()
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        return {
            "status": "success",
            "library": library,
            "scheme": scheme,
            "limitations": limitations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get server statistics"""
    return {
        "status": "success",
        "stats": {
            "total_batches": len(set(m.get('batch_id') for m in metadata_storage.values())),
            "total_columns_encrypted": len(encrypted_storage),
            "total_encrypted_values": sum(
                m.get('count', 0) for m in metadata_storage.values()
            ),
            "libraries": {
                "OpenFHE": openfhe_instance is not None,
                "TenSEAL": tenseal_instance is not None
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )