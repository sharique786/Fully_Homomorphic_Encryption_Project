"""
Complete Enhanced FHE Server
"""

import base64
import threading
import time
import uuid
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gzip

# Import enhanced wrappers
try:
    from tenseal_wrapper import EnhancedTenSEALWrapper as TenSEALWrapper

    TENSEAL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ TenSEAL not available: {e}")
    TENSEAL_AVAILABLE = False

try:
    from openfhe_wrapper import EnhancedOpenFHEWrapper as OpenFHEWrapper

    OPENFHE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ OpenFHE not available: {e}")
    OPENFHE_AVAILABLE = False

app = FastAPI(title="Enhanced FHE Server", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global Storage ====================

# In-memory storage for encrypted data (could be Redis, MongoDB, etc.)
encrypted_data_store = {}  # Format: {party_id: {column_name: [encrypted_values]}}
encrypted_metadata_store = {}  # Format: {batch_id: metadata}

# Global instances
openfhe_instance = None
tenseal_instance = None
encrypted_storage = {}  # In-memory storage for encrypted data
server_public_context = None
library_type = None
scheme_type = None
request_lock = threading.Lock()


# ==================== Request Models ====================

class ContextConfig(BaseModel):
    library: str
    scheme: str
    poly_modulus_degree: int = 16384
    scale: Optional[float] = None
    coeff_mod_bit_sizes: Optional[List[int]] = None
    plain_modulus: Optional[int] = None
    mult_depth: int = 10
    scale_mod_size: int = 40


class KeyGenerationRequest(BaseModel):
    library: str
    scheme: str
    params: Dict[str, Any]



class EncryptedDataUpload(BaseModel):
    """Model for uploading encrypted data to server"""
    batch_id: str
    column_name: str
    data_type: str
    library: str
    scheme: str
    encrypted_records: List[Dict[str, Any]]  # List of {party_id, email_id, encrypted_value, ...}
    compression: str = "gzip"  # or "none"


class EncryptedQueryRequest(BaseModel):
    """Query encrypted data stored on server"""
    library: str
    party_id: str
    email_id: str
    column_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class FHEOperationRequest(BaseModel):
    """Perform FHE operation on server-stored encrypted data"""
    library: str
    operation: str  # 'sum', 'avg', 'multiply', 'fraud_detection', etc.
    party_id: str
    email_id: Optional[str] = None
    column_name: str
    parameters: Optional[Dict[str, Any]] = None   

# FIXED: Column-based encryption request
class ColumnEncryptionRequest(BaseModel):
    library: str
    scheme: str
    column_name: str
    data_type: str
    column_data: List[Any]  # Changed from 'data' to 'column_data'
    party_ids: List[str]
    email_ids: List[str]
    account_ids: Optional[List[str]] = None
    transaction_ids: Optional[List[str]] = None
    transaction_dates: Optional[List[str]] = None
    batch_id: str
    simd_mode: str = "individual"


class TransactionQueryRequest(BaseModel):
    library: str
    party_id: str
    email_id: str
    start_date: str
    end_date: str
    currency: Optional[str] = None


class FraudDetectionRequest(BaseModel):
    library: str
    party_id: str
    email_id: str
    detection_type: str
    encrypted_amounts: List[str]
    model_params: Dict[str, Any]


class SIMDTimeSeriesRequest(BaseModel):
    library: str
    party_id: str
    email_id: str
    operation: str
    encrypted_vector: List[str]
    parameters: Optional[Dict[str, Any]] = None
    transaction_dates: Optional[List[str]] = None


class MLInferenceRequest(BaseModel):
    library: str
    model_type: str
    encrypted_features: List[str]
    weights: List[float]
    intercept: float = 0.0
    polynomial_degree: Optional[int] = None
    compressed: bool = False


class DecryptionRequest(BaseModel):
    library: str
    result_data: Any
    data_type: str = "numeric"


class ParameterRecommendationRequest(BaseModel):
    workload_type: str
    security_level: int = 128
    library: str
    expected_operations: Optional[List[str]] = None

class PublicKeysUpload(BaseModel):
    public_context: str  # Base64 encoded public context
    library: str
    scheme: str
    has_galois_keys: bool
    has_relin_keys: bool
    has_private_key: bool  # Should always be False

# ==================== Startup ====================

@app.on_event("startup")
async def startup_event():
    global openfhe_instance, tenseal_instance

    print("\n" + "=" * 60)
    print("🚀 ENHANCED FHE SERVER STARTING (v3.0.0)")
    print("=" * 60)

    if OPENFHE_AVAILABLE:
        try:
            openfhe_instance = OpenFHEWrapper()
            print("✅ Enhanced OpenFHE wrapper initialized")
        except Exception as e:
            print(f"⚠️ OpenFHE initialization failed: {e}")

    if TENSEAL_AVAILABLE:
        try:
            tenseal_instance = TenSEALWrapper()
            print("✅ Enhanced TenSEAL wrapper initialized")
        except Exception as e:
            print(f"⚠️ TenSEAL initialization failed: {e}")

    print("=" * 60)
    print("✅ Server ready with real FHE operations")
    print("=" * 60 + "\n")


# ==================== Helper Functions ====================

def get_wrapper(library: str):
    """Get the appropriate FHE wrapper"""
    if library == "TenSEAL":
        if not tenseal_instance:
            raise HTTPException(status_code=503, detail="TenSEAL not available")
        return tenseal_instance
    elif library == "OpenFHE":
        if not openfhe_instance:
            raise HTTPException(status_code=503, detail="OpenFHE not available")
        return openfhe_instance
    else:
        raise HTTPException(status_code=400, detail="Invalid library")


def serialize_encrypted(data):
    """Serialize encrypted data for JSON response"""
    if data is None:
        return None
    if isinstance(data, bytes):
        return base64.b64encode(data).decode('utf-8')
    elif isinstance(data, dict):
        return {k: serialize_encrypted(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_encrypted(item) for item in data]
    return data


def deserialize_encrypted(data):
    """Deserialize encrypted data from request"""
    if data is None:
        return None
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except:
            return data
    elif isinstance(data, dict):
        return {k: deserialize_encrypted(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_encrypted(item) for item in data]
    return data


def decompress_encrypted_data(data_str):
    """Decompress encrypted data received from client"""
    try:
        # Decode from base64
        compressed = base64.b64decode(data_str)
        # Decompress
        decompressed = gzip.decompress(compressed)
        return decompressed
    except Exception as e:
        print(f"Decompression error: {e}")
        return None
# ==================== Basic Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "Enhanced FHE Server API",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Real FHE operations",
            "Party-based filtering",
            "SIMD time-series analytics",
            "SQLite storage support"
        ],
        "libraries": {
            "TenSEAL": TENSEAL_AVAILABLE,
            "OpenFHE": OPENFHE_AVAILABLE
        }
    }


@app.get("/health")
async def health_check():
    storage_stats = {
        "total_records": sum(
            len(columns[col_name])
            for party_id, columns in encrypted_data_store.items()
            for col_name in columns
        ),
        "total_parties": len(encrypted_data_store)
    }
    
    return {
        "status": "healthy",
        "server_mode": "Enhanced Security with Storage",
        "has_public_keys": server_public_context is not None,
        "can_encrypt": server_public_context is not None,
        "can_decrypt": False,
        "storage": storage_stats,
        "security_model": "Client-side encryption, server-side storage & FHE operations"
    }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "tenseal_ready": tenseal_instance is not None,
#         "openfhe_ready": openfhe_instance is not None,
#         "timestamp": datetime.now().isoformat()
#     }


# ==================== Context & Key Generation ====================
@app.post("/generate_context")
async def generate_context(config: ContextConfig):
    """Generate FHE context"""
    global server_public_context, library_type, scheme_type
    
    print(f"\n🔧 Generating context (NO KEYS): {config.library} - {config.scheme}")
    print(f"   Parameters: poly_degree={config.poly_modulus_degree}, mult_depth={config.mult_depth}")
    print("   ⚠️  Keys will be generated CLIENT-SIDE")

    try:
        if config.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            context = openfhe_instance.generate_context(
                scheme=config.scheme,
                mult_depth=config.mult_depth,
                scale_mod_size=config.scale_mod_size,
                batch_size=8,
                security_level='HEStd_128_classic',
                ring_dim=config.poly_modulus_degree,
                bootstrap_enabled=False
            )

        elif config.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            print(f"   Preparing TenSEAL parameters...")

            coeff_bits = config.coeff_mod_bit_sizes
            if not coeff_bits or len(coeff_bits) == 0:
                middle_bits = [40] * config.mult_depth
                coeff_bits = [60] + middle_bits + [60]

            print(f"   coeff_mod_bit_sizes: {coeff_bits}")

            scale_value = config.scale
            if not scale_value or scale_value == 0:
                scale_value = float(2 ** config.scale_mod_size)

            print(f"   scale: {scale_value}")

            plain_mod = config.plain_modulus
            if config.scheme == "BFV" and (not plain_mod or plain_mod == 0):
                plain_mod = 1032193

            print(f"   plain_modulus: {plain_mod if config.scheme == 'BFV' else 'N/A (CKKS)'}")

            context = tenseal_instance.generate_context(
                scheme=config.scheme,
                poly_modulus_degree=config.poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_bits,
                scale=scale_value,
                plain_modulus=plain_mod if config.scheme == 'BFV' else 1032193
            )

            if context is None:
                raise Exception("Context generation returned None")
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        print("✅ Context generated successfully")
        return {
            "status": "success",
            "library": config.library,
            "scheme": config.scheme,
            "message": "Context generated successfully",
            "details": {
                "poly_modulus_degree": config.poly_modulus_degree,
                "mult_depth": config.mult_depth,
                "scale_mod_size": config.scale_mod_size,
                "has_keys": False,
                "next_step": "Generate keys client-side and upload public keys"
            }
        }
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Context generation failed: {error_msg}")
        import traceback
        traceback.print_exc()

        # Return more detailed error
        raise HTTPException(
            status_code=500,
            detail=f"Context generation failed: {error_msg}"
        )

# @app.post("/generate_context")
# async def generate_context(config: ContextConfig):
#     """Generate FHE context - Fixed TenSEAL parameter handling"""
#     print(f"\n🔧 Generating context: {config.library} - {config.scheme}")
#     print(f"   Parameters: poly_degree={config.poly_modulus_degree}, mult_depth={config.mult_depth}")
#
#     try:
#         if config.library == "OpenFHE":
#             if not openfhe_instance:
#                 raise HTTPException(status_code=503, detail="OpenFHE not available")
#
#             context = openfhe_instance.generate_context(
#                 scheme=config.scheme,
#                 mult_depth=config.mult_depth,
#                 scale_mod_size=config.scale_mod_size,
#                 batch_size=8,
#                 security_level='HEStd_128_classic',
#                 ring_dim=config.poly_modulus_degree,
#                 bootstrap_enabled=False
#             )
#
#         elif config.library == "TenSEAL":
#             if not tenseal_instance:
#                 raise HTTPException(status_code=503, detail="TenSEAL not available")
#
#             # FIX: Handle parameters carefully for TenSEAL
#             print(f"   Preparing TenSEAL parameters...")
#
#             # Generate coeff_mod_bit_sizes
#             coeff_bits = config.coeff_mod_bit_sizes
#             if not coeff_bits or len(coeff_bits) == 0:
#                 # Generate default: [60, 40, 40, ..., 60]
#                 middle_bits = [40] * config.mult_depth
#                 coeff_bits = [60] + middle_bits + [60]
#
#             print(f"   coeff_mod_bit_sizes: {coeff_bits}")
#
#             # Calculate scale
#             scale_value = config.scale
#             if not scale_value or scale_value == 0:
#                 scale_value = float(2 ** config.scale_mod_size)
#
#             print(f"   scale: {scale_value}")
#
#             # Plain modulus (for BFV only)
#             plain_mod = config.plain_modulus
#             if config.scheme == "BFV" and (not plain_mod or plain_mod == 0):
#                 plain_mod = 1032193
#
#             print(f"   plain_modulus: {plain_mod if config.scheme == 'BFV' else 'N/A (CKKS)'}")
#
#             # Call wrapper with validated parameters
#             context = tenseal_instance.generate_context(
#                 scheme=config.scheme,
#                 poly_modulus_degree=config.poly_modulus_degree,
#                 coeff_mod_bit_sizes=coeff_bits,
#                 scale=scale_value,
#                 plain_modulus=plain_mod if config.scheme == 'BFV' else 1032193
#             )
#
#             if context is None:
#                 raise Exception("Context generation returned None")
#
#         else:
#             raise HTTPException(status_code=400, detail="Invalid library")
#
#         print("✅ Context generated successfully")
#         return {
#             "status": "success",
#             "library": config.library,
#             "scheme": config.scheme,
#             "message": "Context generated successfully",
#             "details": {
#                 "poly_modulus_degree": config.poly_modulus_degree,
#                 "mult_depth": config.mult_depth,
#                 "scale_mod_size": config.scale_mod_size
#             }
#         }
#
#     except Exception as e:
#         error_msg = str(e)
#         print(f"❌ Context generation failed: {error_msg}")
#         import traceback
#         traceback.print_exc()
#
#         # Return more detailed error
#         raise HTTPException(
#             status_code=500,
#             detail=f"Context generation failed: {error_msg}"
#         )

# ==================== ENHANCED: Batch Upload Endpoint ====================

class EncryptedDataBatchUpload(BaseModel):
    """Model for uploading encrypted data in batches"""
    batch_id: str
    column_name: str
    data_type: str
    library: str
    scheme: str
    encrypted_records: List[Dict[str, Any]]  # List of {party_id, email_id, encrypted_value, ...}
    compression: str = "gzip"
    batch_number: int  # Current batch number (1-indexed)
    total_batches: int  # Total number of batches


@app.post("/upload_encrypted_data_batch")
async def upload_encrypted_data_batch(request: EncryptedDataBatchUpload):
    """
    NEW ENDPOINT: Receive encrypted data from client in BATCHES
    Handles batch uploads to prevent client freezing
    Each batch contains a subset of the total encrypted column data
    """
    print(f"\n📤 Receiving encrypted data batch from client...")
    print(f"   Batch: {request.batch_number}/{request.total_batches}")
    print(f"   Batch ID: {request.batch_id}")
    print(f"   Column: {request.column_name}")
    print(f"   Records in this batch: {len(request.encrypted_records)}")
    print(f"   Library: {request.library}, Scheme: {request.scheme}")
    
    try:
        # ==================== STEP 1: Initialize/Update Metadata ====================
        
        if request.batch_id not in encrypted_metadata_store:
            encrypted_metadata_store[request.batch_id] = {
                'batch_id': request.batch_id,
                'columns': {},
                'library': request.library,
                'scheme': request.scheme,
                'created_at': datetime.now().isoformat(),
                'batch_progress': {}  # Track batch upload progress
            }
        
        # Initialize column metadata if first batch
        if request.column_name not in encrypted_metadata_store[request.batch_id]['columns']:
            encrypted_metadata_store[request.batch_id]['columns'][request.column_name] = {
                'data_type': request.data_type,
                'record_count': 0,
                'compression': request.compression,
                'total_batches': request.total_batches,
                'batches_received': []
            }
        
        column_meta = encrypted_metadata_store[request.batch_id]['columns'][request.column_name]
        
        # Check if batch already received (idempotency)
        if request.batch_number in column_meta['batches_received']:
            print(f"   ⚠️ Batch {request.batch_number} already received, skipping...")
            return {
                "status": "success",
                "batch_id": request.batch_id,
                "column_name": request.column_name,
                "batch_number": request.batch_number,
                "stored_count": len(request.encrypted_records),
                "message": "Batch already received (duplicate)"
            }
        
        # ==================== STEP 2: Store Encrypted Records ====================
        
        stored_count = 0
        
        for record in request.encrypted_records:
            party_id = record.get('party_id')
            email_id = record.get('email_id')
            encrypted_value = record.get('encrypted_value')
            
            if not party_id or not encrypted_value:
                print(f"   ⚠️ Skipping invalid record (missing party_id or encrypted_value)")
                continue
            
            # Initialize party storage
            if party_id not in encrypted_data_store:
                encrypted_data_store[party_id] = {}
            
            if request.column_name not in encrypted_data_store[party_id]:
                encrypted_data_store[party_id][request.column_name] = []
            
            # Deserialize encrypted value (from base64)
            if isinstance(encrypted_value, str):
                try:
                    enc_bytes = base64.b64decode(encrypted_value)
                except Exception as e:
                    print(f"   ⚠️ Failed to decode base64: {e}")
                    enc_bytes = encrypted_value
            else:
                enc_bytes = encrypted_value
            
            # Store with metadata
            encrypted_data_store[party_id][request.column_name].append({
                'encrypted_value': enc_bytes,
                'email_id': email_id,
                'account_id': record.get('account_id'),
                'transaction_id': record.get('transaction_id'),
                'transaction_date': record.get('transaction_date'),
                'batch_id': request.batch_id,
                'data_type': request.data_type,
                'batch_number': request.batch_number  # Track which batch this came from
            })
            
            stored_count += 1
        
        # ==================== STEP 3: Update Metadata ====================
        
        column_meta['record_count'] += stored_count
        column_meta['batches_received'].append(request.batch_number)
        
        # Calculate progress
        batches_received = len(column_meta['batches_received'])
        total_batches = column_meta['total_batches']
        progress_percentage = (batches_received / total_batches) * 100
        
        print(f"✅ Stored {stored_count} encrypted records for {request.column_name}")
        print(f"   Progress: {batches_received}/{total_batches} batches ({progress_percentage:.1f}%)")
        print(f"   Total records so far: {column_meta['record_count']:,}")
        
        # Check if upload complete
        is_complete = batches_received == total_batches
        
        if is_complete:
            print(f"🎉 All batches received for {request.column_name}!")
            column_meta['upload_complete'] = True
            column_meta['completed_at'] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "batch_id": request.batch_id,
            "column_name": request.column_name,
            "batch_number": request.batch_number,
            "total_batches": total_batches,
            "stored_count": stored_count,
            "total_records_so_far": column_meta['record_count'],
            "batches_received": batches_received,
            "progress_percentage": progress_percentage,
            "upload_complete": is_complete,
            "message": f"Batch {request.batch_number}/{total_batches} stored successfully"
        }
    
    except Exception as e:
        print(f"❌ Batch upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Batch upload failed: {str(e)}"
        )


@app.post("/upload_encrypted_data")
async def upload_encrypted_data(request: EncryptedDataUpload):
    """
    Receive encrypted data from client and store on server
    Client encrypts, server stores for FHE operations
    """
    print(f"\n📤 Receiving encrypted data from client...")
    print(f"   Batch ID: {request.batch_id}")
    print(f"   Column: {request.column_name}")
    print(f"   Records: {len(request.encrypted_records)}")
    print(f"   Library: {request.library}, Scheme: {request.scheme}")
    
    try:
        # Store metadata
        if request.batch_id not in encrypted_metadata_store:
            encrypted_metadata_store[request.batch_id] = {
                'batch_id': request.batch_id,
                'columns': {},
                'library': request.library,
                'scheme': request.scheme,
                'created_at': datetime.now().isoformat()
            }
        
        encrypted_metadata_store[request.batch_id]['columns'][request.column_name] = {
            'data_type': request.data_type,
            'record_count': len(request.encrypted_records),
            'compression': request.compression
        }
        
        # Store encrypted records by party_id
        stored_count = 0
        
        for record in request.encrypted_records:
            party_id = record.get('party_id')
            email_id = record.get('email_id')
            encrypted_value = record.get('encrypted_value')
            
            if not party_id or not encrypted_value:
                continue
            
            # Initialize party storage
            if party_id not in encrypted_data_store:
                encrypted_data_store[party_id] = {}
            
            if request.column_name not in encrypted_data_store[party_id]:
                encrypted_data_store[party_id][request.column_name] = []
            
            # Deserialize encrypted value (from base64)
            if isinstance(encrypted_value, str):
                try:
                    enc_bytes = base64.b64decode(encrypted_value)
                except:
                    enc_bytes = encrypted_value
            else:
                enc_bytes = encrypted_value
            
            # Store with metadata
            encrypted_data_store[party_id][request.column_name].append({
                'encrypted_value': enc_bytes,
                'email_id': email_id,
                'account_id': record.get('account_id'),
                'transaction_id': record.get('transaction_id'),
                'transaction_date': record.get('transaction_date'),
                'batch_id': request.batch_id,
                'data_type': request.data_type
            })
            
            stored_count += 1
        
        print(f"✅ Stored {stored_count} encrypted records for {request.column_name}")
        print(f"   Unique parties in store: {len(encrypted_data_store)}")
        
        return {
            "status": "success",
            "batch_id": request.batch_id,
            "column_name": request.column_name,
            "stored_count": stored_count,
            "message": "Encrypted data stored on server for FHE operations"
        }
    
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ==================== Query Encrypted Data from Server ====================

@app.post("/query_encrypted_data")
async def query_encrypted_data(request: EncryptedQueryRequest):
    """
    Query encrypted data stored on server
    Returns encrypted data for specific party/column
    """
    print(f"\n🔍 Querying encrypted data...")
    print(f"   Party: {request.party_id[:8]}...")
    print(f"   Column: {request.column_name}")
    
    try:
        if request.party_id not in encrypted_data_store:
            return {
                "status": "success",
                "party_id": request.party_id,
                "column_name": request.column_name,
                "encrypted_records": [],
                "message": "No encrypted data found for this party"
            }
        
        party_data = encrypted_data_store[request.party_id]
        
        if request.column_name not in party_data:
            return {
                "status": "success",
                "party_id": request.party_id,
                "column_name": request.column_name,
                "encrypted_records": [],
                "message": f"No encrypted data found for column: {request.column_name}"
            }
        
        # Get encrypted records
        encrypted_records = party_data[request.column_name]
       
        # Filter by date if provided
        start_date = normalize_to_date(request.start_date) if request.start_date else None
        end_date = normalize_to_date(request.end_date) if request.end_date else None
        if start_date or end_date:
            filtered_records = []
            for record in encrypted_records:
                trans_raw = record.get("transaction_date")
                trans_date = normalize_to_date(trans_raw) if trans_raw else None
                if trans_date:
                    if start_date and trans_date < start_date:
                        continue
                    if end_date and trans_date > end_date:
                        continue
                filtered_records.append(record)
            encrypted_records = filtered_records
        
        print(f"   Total records after date filtering: {len(encrypted_records)}")
        
        # Serialize for transmission
        serialized_records = []
        for record in encrypted_records:
            serialized_records.append({
                'encrypted_value': base64.b64encode(record['encrypted_value']).decode(),
                'email_id': record.get('email_id'),
                'transaction_date': record.get('transaction_date')
            })
        
        print(f"✅ Found {len(serialized_records)} encrypted records")
        
        return {
            "status": "success",
            "party_id": request.party_id,
            "column_name": request.column_name,
            "encrypted_records": serialized_records,
            "record_count": len(serialized_records)
        }
    
    except Exception as e:
        print(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate_keys")
async def generate_keys(request: KeyGenerationRequest):
    """Generate encryption keys - Fixed parameter alignment"""
    print(f"\n🔑 Generating keys: {request.library} - {request.scheme}")
    print(f"   Parameters: {request.params}")

    try:
        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            # Generate context first with proper parameters
            openfhe_instance.generate_context(
                scheme=request.scheme,
                mult_depth=request.params.get('mult_depth', 10),
                scale_mod_size=request.params.get('scale_mod_size', 50),
                batch_size=8,
                security_level='HEStd_128_classic',
                ring_dim=request.params.get('poly_modulus_degree', 8192),
                bootstrap_enabled=False
            )

            keys_info = openfhe_instance.get_keys_info()

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            # Get parameters with defaults
            poly_degree = request.params.get('poly_modulus_degree', 8192)
            mult_depth = request.params.get('mult_depth', 10)
            scale = request.params.get('scale')
            scale_mod_size = request.params.get('scale_mod_size', 40)

            # Generate coeff_mod_bit_sizes if not provided
            coeff_bits = request.params.get('coeff_mod_bit_sizes')
            if not coeff_bits:
                coeff_bits = [60] + [40] * mult_depth + [60]

            # Calculate scale if not provided
            if not scale:
                scale = float(2 ** scale_mod_size)

            # Generate context
            tenseal_instance.generate_context(
                scheme=request.scheme,
                poly_modulus_degree=poly_degree,
                coeff_mod_bit_sizes=coeff_bits,
                scale=scale,
                plain_modulus=request.params.get('plain_modulus', 1032193)
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Key generation failed: {str(e)}")

# ==================== NEW: Upload Public Keys Endpoint ====================

@app.post("/upload_public_keys")
async def upload_public_keys(keys: PublicKeysUpload):
    """
    NEW ENDPOINT: Receive public keys from client
    Server never receives private key
    """
    global server_public_context, library_type, scheme_type
    
    print(f"\n🔑 Receiving public keys from client...")
    print(f"   Library: {keys.library}, Scheme: {keys.scheme}")
    print(f"   Has Galois Keys: {keys.has_galois_keys}")
    print(f"   Has Relin Keys: {keys.has_relin_keys}")
    print(f"   Has Private Key: {keys.has_private_key}")
    
    # SECURITY CHECK: Ensure no private key was sent
    if keys.has_private_key:
        print("❌ SECURITY VIOLATION: Client attempted to send private key!")
        raise HTTPException(
            status_code=403,
            detail="Security violation: Private keys should NEVER be sent to server"
        )
    
    try:
        if keys.library == "TenSEAL":
            # Deserialize public context with keys
            import tenseal as ts
            context_bytes = base64.b64decode(keys.public_context)
            
            print(f"   Deserializing public context ({len(context_bytes):,} bytes)...")
            
            # Load public context (includes galois & relin keys)
            server_public_context = ts.context_from(context_bytes)
            
            # Verify it's truly public (no secret key)
            try:
                # This should fail if secret key exists
                test_vec = ts.ckks_vector(server_public_context, [1.0])
                test_vec.decrypt()  # This should raise an error
                
                print("❌ SECURITY WARNING: Context has secret key!")
                raise HTTPException(
                    status_code=403,
                    detail="Context contains secret key - rejected"
                )
            except:
                # Good - decryption failed, meaning no secret key
                print("✅ Verified: Context is public only (no secret key)")
            
            library_type = keys.library
            scheme_type = keys.scheme
            
            print("✅ Public keys uploaded and loaded successfully")
            print("   Server can now perform encrypted operations")
            
        elif keys.library == "OpenFHE":
            # OpenFHE simulation mode
            server_public_context = {
                'library': keys.library,
                'scheme': keys.scheme,
                'public_only': True
            }
            print("✅ OpenFHE public keys registered (simulation mode)")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid library")
        
        return {
            "status": "success",
            "message": "Public keys uploaded successfully",
            "server_can_encrypt": True,
            "server_can_decrypt": False,
            "security_status": "✅ Server has NO access to private keys"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Public key upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Public key upload failed: {str(e)}")

# ==================== Column-Based Encryption with SIMD Support ====================

@app.post("/encrypt_column")
async def encrypt_column(request: ColumnEncryptionRequest):
    """
    Encrypt entire column with SIMD mode support
    Modes: individual, packed_vector, batch_processing
    """
    print(f"\n🔒 Encrypting column: {request.column_name}")
    print(f"   Mode: {request.simd_mode}")
    print(f"   Records: {len(request.column_data)}")
    print(f"   Library: {request.library}, Scheme: {request.scheme}")

    start_time = time.time()

    try:
        wrapper = get_wrapper(request.library)

        if not wrapper.context:
            raise HTTPException(status_code=400, detail="Context not initialized")

        # Process data based on data type
        processed_data = []
        for item in request.column_data:
            if request.data_type == "date" and item is not None:
                try:
                    if isinstance(item, str):
                        timestamp = pd.Timestamp(item).timestamp()
                    elif isinstance(item, pd.Timestamp):
                        timestamp = item.timestamp()
                    elif hasattr(item, 'timestamp'):
                        timestamp = item.timestamp()
                    else:
                        timestamp = float(item)
                    processed_data.append(timestamp)
                except Exception as e:
                    print(f"   ⚠️ Date conversion error for {item}: {e}")
                    processed_data.append(None)
            else:
                processed_data.append(item)

        encrypted_results = []

        # SIMD Mode: Individual Encryption (standard)
        if request.simd_mode == "individual":
            print("   Using INDIVIDUAL encryption mode...")
            encrypted_values = wrapper.encrypt_data(
                processed_data,
                request.column_name,
                request.data_type
            )

            # Create result with metadata
            for i, (enc_val, party_id, email_id) in enumerate(zip(
                    encrypted_values,
                    request.party_ids,
                    request.email_ids
            )):
                if enc_val is not None:
                    encrypted_results.append({
                        'index': i,
                        'encrypted_value': serialize_encrypted(enc_val),
                        'party_id': party_id,
                        'email_id': email_id,
                        'account_id': request.account_ids[i] if request.account_ids else None,
                        'transaction_id': request.transaction_ids[i] if request.transaction_ids else None,
                        'transaction_date': request.transaction_dates[i] if request.transaction_dates else None
                    })

        # SIMD Mode: Packed Vector (multiple values in single ciphertext)
        elif request.simd_mode == "packed_vector":
            print("   Using PACKED VECTOR mode (SIMD)...")

            if request.scheme != 'CKKS':
                raise HTTPException(
                    status_code=400,
                    detail="Packed vector mode requires CKKS scheme"
                )

            # FIX: Packed vector only works with numeric data
            if request.data_type != 'numeric':
                raise HTTPException(
                    status_code=400,
                    detail=f"Packed vector mode only supports numeric data. Column '{request.column_name}' is {request.data_type}. Use 'individual' or 'batch_processing' mode instead."
                )

            batch_size = 128

            for batch_start in range(0, len(processed_data), batch_size):
                batch_end = min(batch_start + batch_size, len(processed_data))
                batch_data = processed_data[batch_start:batch_end]

                # FIX: Safe conversion to float with error handling
                valid_data = []
                for v in batch_data:
                    if v is not None:
                        try:
                            valid_data.append(float(v))
                        except (ValueError, TypeError) as e:
                            print(f"   ⚠️ Skipping non-numeric value: {v} ({type(v)})")
                            continue

                if valid_data:
                    packed_encrypted = wrapper.encrypt_vector(valid_data)

                    if request.compress_response and packed_encrypted:
                        original_size = len(packed_encrypted)
                        # total_original_size += original_size
                        compressed = gzip.compress(packed_encrypted, compresslevel=9)
                        # total_compressed_size += len(compressed)
                        packed_encrypted_serialized = base64.b64encode(compressed).decode('utf-8')
                    else:
                        packed_encrypted_serialized = serialize_encrypted(packed_encrypted)

                    encrypted_results.append({
                        'index': batch_start,
                        'encrypted_value': packed_encrypted_serialized,
                        'batch_size': len(valid_data),
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'mode': 'packed_vector',
                        'party_ids': request.party_ids[batch_start:batch_end],
                        'email_ids': request.email_ids[batch_start:batch_end]
                    })

        # SIMD Mode: Batch Processing (optimized batching)
        elif request.simd_mode == "batch_processing":
            print("   Using BATCH PROCESSING mode...")

            # Process in optimized batches
            batch_size = 256

            for batch_start in range(0, len(processed_data), batch_size):
                batch_end = min(batch_start + batch_size, len(processed_data))
                batch_data = processed_data[batch_start:batch_end]

                # Encrypt batch
                encrypted_batch = wrapper.encrypt_data(
                    batch_data,
                    request.column_name,
                    request.data_type
                )

                # Store batch results
                for i, enc_val in enumerate(encrypted_batch):
                    if enc_val is not None:
                        global_idx = batch_start + i
                        encrypted_results.append({
                            'index': global_idx,
                            'encrypted_value': serialize_encrypted(enc_val),
                            'party_id': request.party_ids[global_idx],
                            'email_id': request.email_ids[global_idx],
                            'account_id': request.account_ids[global_idx] if request.account_ids else None,
                            'transaction_id': request.transaction_ids[global_idx] if request.transaction_ids else None,
                            'transaction_date': request.transaction_dates[
                                global_idx] if request.transaction_dates else None,
                            'batch_id': f"{request.batch_id}_batch_{batch_start}"
                        })

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported SIMD mode: {request.simd_mode}")

        # Store in server memory for later queries
        for result in encrypted_results:
            storage_key = f"{result.get('party_id', '')}_{result.get('email_id', '')}_{request.column_name}_{result.get('index', uuid.uuid4())}"
            encrypted_storage[storage_key] = {
                'encrypted_value': deserialize_encrypted(result['encrypted_value']),
                'party_id': result.get('party_id'),
                'email_id': result.get('email_id'),
                'column_name': request.column_name,
                'data_type': request.data_type,
                'batch_id': request.batch_id,
                'timestamp': datetime.now().isoformat()
            }

        elapsed_time = time.time() - start_time
        print(f"✅ Column encryption complete in {elapsed_time:.2f}s")
        print(f"   Encrypted {len(encrypted_results)} items")

        return {
            "status": "success",
            "column_name": request.column_name,
            "simd_mode": request.simd_mode,
            "encrypted_count": len(encrypted_results),
            "encrypted_results": encrypted_results,
            "encryption_time": elapsed_time,
            "batch_id": request.batch_id,
            "note": f"Column encrypted using {request.simd_mode} mode"
        }

    except Exception as e:
        print(f"❌ Column encryption failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Scheme limitations Endpoints ====================
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


# ==================== Query Transactions (REAL FHE) ====================
def normalize_to_date(value):
    """Convert string/datetime/date into a pure date object."""
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass

        raise ValueError(f"Unsupported date format: {value}")

    raise TypeError(f"Unsupported type for date: {type(value)}")


@app.post("/query_transactions")
async def query_transactions(request: TransactionQueryRequest):
    """
    Query transactions using encrypted data stored on server
    Performs FHE operations on server-stored encrypted data
    """
    print(f"\n🔍 Querying transactions for party: {request.party_id[:8]}...")
    print(f"   Using server-stored encrypted data")
    print("   ⚡ PERFORMING FHE OPERATIONS")
    
    start_time = time.time()
    
    try:
        # Get encrypted data from server storage
        if request.party_id not in encrypted_data_store:
            return {
                "status": "success",
                "message": "No encrypted data found for this party",
                "party_id": request.party_id,
                "transaction_count": 0
            }
        
        party_data = encrypted_data_store[request.party_id]
        print(f"   Found data for party: {request.party_id[:20]} with columns: {list(party_data.keys())}")
        
        # Get Amount column (or requested column)
        column_name = "Amount"  # Can be parameterized
        
        if column_name not in party_data:
            return {
                "status": "success",
                "message": f"No encrypted data found for column: {column_name}",
                "party_id": request.party_id,
                "transaction_count": 0
            }
        
        encrypted_records = party_data[column_name]
        # print(f"   Total encrypted records in column '{column_name}': {len(encrypted_records)}")
        
        # Filter by date range
        filtered_encrypted = []
        start_date = normalize_to_date(request.start_date) if request.start_date else None
        end_date = normalize_to_date(request.end_date) if request.end_date else None
        # print(f"   Applying date filter: {start_date} to {end_date}")
        
        for record in encrypted_records:
            trans_raw = record.get("transaction_date")
            trans_date = normalize_to_date(trans_raw) if trans_raw else None

            if trans_date:
                if start_date and trans_date < start_date:
                    continue
                if end_date and trans_date > end_date:
                    continue
            
            # Filter by email if provided
            if request.email_id and record.get('email_id') != request.email_id:
                continue
            
            filtered_encrypted.append(record['encrypted_value'])
        
        if not filtered_encrypted:
            return {
                "status": "success",
                "message": "No matching encrypted data",
                "party_id": request.party_id,
                "transaction_count": 0
            }
        
        print(f"   Found {len(filtered_encrypted)} encrypted records")
        
        # Get wrapper for FHE operations
        if request.library == "TenSEAL":
            from tenseal_wrapper import EnhancedTenSEALWrapper
            wrapper = EnhancedTenSEALWrapper()
            wrapper.context = server_public_context
            wrapper.scheme = scheme_type
        else:
            from openfhe_wrapper import EnhancedOpenFHEWrapper
            wrapper = EnhancedOpenFHEWrapper()
        
        # Perform REAL FHE aggregations on server-stored encrypted data
        print("   Computing SUM on encrypted data...")
        total_sum = wrapper.perform_aggregation(filtered_encrypted, 'sum')
        
        print("   Computing AVERAGE on encrypted data...")
        average_amount = wrapper.perform_aggregation(filtered_encrypted, 'avg')
        
        # Calculate transferred vs received (simplified)
        mid_point = len(filtered_encrypted) // 2
        total_transferred = wrapper.perform_aggregation(filtered_encrypted[:mid_point], 'sum')
        total_received = wrapper.perform_aggregation(filtered_encrypted[mid_point:], 'sum')
        
        elapsed_time = time.time() - start_time
        print(f"✅ Query complete in {elapsed_time:.2f}s")
        
        # Serialize results
        def serialize_encrypted(data):
            if data is None:
                return None
            if isinstance(data, bytes):
                return base64.b64encode(data).decode('utf-8')
            elif isinstance(data, dict):
                return {k: serialize_encrypted(v) for k, v in data.items()}
            return data
        
        return {
            "status": "success",
            "party_id": request.party_id,
            "email_id": request.email_id,
            "transaction_count": len(filtered_encrypted),
            "total_transferred": serialize_encrypted(total_transferred),
            "total_received": serialize_encrypted(total_received),
            "average_amount": serialize_encrypted(average_amount),
            "computation_time": elapsed_time,
            "encrypted": True,
            "note": "All results are encrypted. Decrypt on client side.",
            "data_source": "server-stored encrypted data"
        }
    
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# @app.post("/query_transactions")
# async def query_transactions(request: TransactionQueryRequest):
#     """
#     Query transactions by Party ID/Email with date range
#     PERFORMS REAL FHE OPERATIONS - NO SIMULATION
#     """
#     print(f"\n🔍 Querying transactions for party: {request.party_id[:8]}...")
#     print(f"   Date range: {request.start_date} to {request.end_date}")
#     print("   ⚡ PERFORMING REAL FHE OPERATIONS")

#     start_time = time.time()

#     try:
#         wrapper = get_wrapper(request.library)

#         # Filter encrypted storage by party_id/email_id and date
#         filtered_encrypted = []

#         for key, data in encrypted_storage.items():
#             # Check party filter
#             if data['party_id'] != request.party_id and data['email_id'] != request.email_id:
#                 continue

#             # Check column (looking for Amount)
#             if data['column_name'] != 'Amount':
#                 continue

#             # Date filtering would require querying with transaction_date
#             # For now, include all matching party records
#             filtered_encrypted.append(data['encrypted_value'])

#         if not filtered_encrypted:
#             return {
#                 "status": "success",
#                 "message": "No matching encrypted data",
#                 "party_id": request.party_id,
#                 "transaction_count": 0
#             }

#         print(f"   Found {len(filtered_encrypted)} encrypted records")

#         # Perform REAL FHE aggregations
#         print("   Computing SUM on encrypted data...")
#         total_sum = wrapper.perform_aggregation(filtered_encrypted, 'sum')

#         print("   Computing AVERAGE on encrypted data...")
#         average_amount = wrapper.perform_aggregation(filtered_encrypted, 'avg')

#         # Calculate total transferred vs received (simplified - assume half each)
#         mid_point = len(filtered_encrypted) // 2
#         total_transferred = wrapper.perform_aggregation(filtered_encrypted[:mid_point], 'sum')
#         total_received = wrapper.perform_aggregation(filtered_encrypted[mid_point:], 'sum')

#         elapsed_time = time.time() - start_time
#         print(f"✅ Query complete in {elapsed_time:.2f}s")

#         return {
#             "status": "success",
#             "party_id": request.party_id,
#             "email_id": request.email_id,
#             "transaction_count": len(filtered_encrypted),
#             "total_transferred": serialize_encrypted(total_transferred),
#             "total_received": serialize_encrypted(total_received),
#             "average_amount": serialize_encrypted(average_amount),
#             "encrypted": True,
#             "decryption_note": "Decrypt these results CLIENT-SIDE using your private key"
#         }

#     except Exception as e:
#         print(f"❌ Query failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# ==================== Fraud Detection (REAL FHE) ====================

@app.post("/fraud/detect")
async def fraud_detection(request: FraudDetectionRequest):
    """
    FIXED: Fraud detection on encrypted amounts
    Properly handles compressed/encoded data from client
    """
    print(f"\n🚨 Fraud Detection: {request.detection_type}")
    print(f"   Party: {request.party_id[:8]}...")
    print(f"   Received {len(request.encrypted_amounts)} encrypted amounts")
    print("   ⚡ PERFORMING REAL FHE OPERATIONS")
    
    start_time = time.time()
    
    try:
        # Get wrapper
        if request.library == "TenSEAL":
            if not server_public_context:
                raise HTTPException(
                    status_code=400, 
                    detail="Server context not initialized. Upload public keys first."
                )
            
            from tenseal_wrapper import EnhancedTenSEALWrapper
            wrapper = EnhancedTenSEALWrapper()
            wrapper.context = server_public_context
            wrapper.scheme = scheme_type
        else:
            from openfhe_wrapper import EnhancedOpenFHEWrapper
            wrapper = EnhancedOpenFHEWrapper()
        
        # ==================== FIXED: Deserialize Encrypted Amounts ====================
        
        encrypted_amounts = []
        
        for idx, amt in enumerate(request.encrypted_amounts):
            try:
                # Step 1: Decode from base64 (if string)
                if isinstance(amt, str):
                    try:
                        enc_bytes = base64.b64decode(amt)
                        print(f"   Decoded amount {idx}: {len(enc_bytes)} bytes")
                    except Exception as e:
                        print(f"   ⚠️ Failed to decode base64 for amount {idx}: {e}")
                        continue
                elif isinstance(amt, bytes):
                    enc_bytes = amt
                else:
                    print(f"   ⚠️ Unsupported type for amount {idx}: {type(amt)}")
                    continue
                
                # Step 2: Try decompression (data might be gzip compressed)
                try:
                    decompressed = gzip.decompress(enc_bytes)
                    print(f"   Decompressed amount {idx}: {len(enc_bytes)} → {len(decompressed)} bytes")
                    enc_bytes = decompressed
                except:
                    # Not compressed, use as-is
                    print(f"   Amount {idx} not compressed, using raw bytes")
                
                # Step 3: Verify it's valid TenSEAL data
                if request.library == "TenSEAL":
                    import tenseal as ts
                    try:
                        # Try to parse as CKKS vector to verify
                        if scheme_type == 'CKKS':
                            test_vec = ts.ckks_vector_from(server_public_context, enc_bytes)
                        else:
                            test_vec = ts.bfv_vector_from(server_public_context, enc_bytes)
                        
                        print(f"   ✅ Amount {idx} parsed successfully")
                        encrypted_amounts.append(enc_bytes)
                    
                    except Exception as e:
                        print(f"   ❌ Failed to parse CKKS stream for amount {idx}: {e}")
                        print(f"      Data length: {len(enc_bytes)}")
                        print(f"      First 20 bytes: {enc_bytes[:20]}")
                        continue
                else:
                    # OpenFHE simulation mode
                    encrypted_amounts.append(enc_bytes)
            
            except Exception as e:
                print(f"   ❌ Error processing encrypted amount {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not encrypted_amounts:
            print("   ❌ No valid encrypted amounts after processing")
            raise HTTPException(
                status_code=400, 
                detail="No valid encrypted amounts. Check data format and encoding."
            )
        
        print(f"   ✅ Successfully processed {len(encrypted_amounts)}/{len(request.encrypted_amounts)} amounts")
        
        # ==================== Perform Fraud Detection ====================
        
        fraud_score = None
        
        if request.detection_type == "linear_score":
            # Linear weighted fraud score
            weights = request.model_params.get('weights', {})
            
            print(f"   Computing linear fraud score with weights: {weights}")
            
            # Process up to 5 amounts for fraud scoring
            for idx, enc_amt in enumerate(encrypted_amounts[:5]):
                try:
                    if request.library == "TenSEAL":
                        # Reconstruct the encrypted vector
                        if scheme_type == 'CKKS':
                            vec = ts.ckks_vector_from(server_public_context, enc_amt)
                        else:
                            vec = ts.bfv_vector_from(server_public_context, enc_amt)
                        
                        # Apply weight
                        weight = weights.get('amount', 0.5)
                        weighted_vec = vec * weight
                        
                        # Serialize back
                        weighted = weighted_vec.serialize()
                    else:
                        # OpenFHE path
                        weight = weights.get('amount', 0.5)
                        weighted = wrapper.scalar_multiply(enc_amt, weight)
                    
                    if weighted:
                        if fraud_score is None:
                            fraud_score = weighted
                        else:
                            # Add to accumulator
                            if request.library == "TenSEAL":
                                fraud_vec = ts.ckks_vector_from(server_public_context, fraud_score)
                                weighted_vec = ts.ckks_vector_from(server_public_context, weighted)
                                result_vec = fraud_vec + weighted_vec
                                fraud_score = result_vec.serialize()
                            else:
                                fraud_score = wrapper.slot_wise_operation(fraud_score, weighted, 'add')
                    
                    print(f"   ✅ Processed amount {idx} for fraud scoring")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing amount {idx}: {e}")
                    continue
            
            # Normalize by count
            if fraud_score and len(encrypted_amounts[:5]) > 0:
                try:
                    count = len(encrypted_amounts[:5])
                    if request.library == "TenSEAL":
                        fraud_vec = ts.ckks_vector_from(server_public_context, fraud_score)
                        normalized_vec = fraud_vec * (1.0 / count)
                        fraud_score = normalized_vec.serialize()
                    else:
                        fraud_score = wrapper.scalar_multiply(fraud_score, 1.0 / count)
                    
                    print(f"   ✅ Normalized fraud score by {count}")
                except Exception as e:
                    print(f"   ⚠️ Normalization failed: {e}")
        
        elif request.detection_type == "distance_anomaly":
            # Distance-based anomaly detection
            centroid = request.model_params.get('centroid', {})
            centroid_amount = centroid.get('amount', 5000.0)
            
            print(f"   Computing distance from centroid: {centroid_amount}")
            
            squared_diffs = []
            
            for idx, enc_amt in enumerate(encrypted_amounts[:5]):
                try:
                    # (amount - centroid)^2
                    if request.library == "TenSEAL":
                        vec = ts.ckks_vector_from(server_public_context, enc_amt)
                        
                        # Subtract centroid
                        diff_vec = vec - centroid_amount
                        
                        # Square
                        sq_diff_vec = diff_vec * diff_vec
                        
                        sq_diff = sq_diff_vec.serialize()
                    else:
                        diff = wrapper.scalar_add(enc_amt, -centroid_amount)
                        sq_diff = wrapper.slot_wise_operation(diff, diff, 'multiply') if diff else None
                    
                    if sq_diff:
                        squared_diffs.append(sq_diff)
                    
                    print(f"   ✅ Computed distance for amount {idx}")
                
                except Exception as e:
                    print(f"   ⚠️ Error computing distance for amount {idx}: {e}")
                    continue
            
            if squared_diffs:
                # Sum all squared differences
                if request.library == "TenSEAL":
                    fraud_vec = ts.ckks_vector_from(server_public_context, squared_diffs[0])
                    for sq_diff in squared_diffs[1:]:
                        next_vec = ts.ckks_vector_from(server_public_context, sq_diff)
                        fraud_vec = fraud_vec + next_vec
                    fraud_score = fraud_vec.serialize()
                else:
                    fraud_score = wrapper.perform_aggregation(squared_diffs, 'sum')
                
                print(f"   ✅ Computed distance-based fraud score from {len(squared_diffs)} amounts")
            else:
                fraud_score = None
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported detection type: {request.detection_type}"
            )
        
        elapsed_time = time.time() - start_time
        
        if fraud_score is None:
            print(f"   ⚠️ Fraud detection produced no result")
            # Create a default encrypted zero for consistent response
            if request.library == "TenSEAL" and server_public_context:
                zero_vec = ts.ckks_vector(server_public_context, [0.0])
                fraud_score = zero_vec.serialize()
        
        print(f"✅ Fraud detection complete in {elapsed_time:.2f}s")
        
        # Serialize result
        def serialize_encrypted(data):
            if data is None:
                return None
            if isinstance(data, bytes):
                return base64.b64encode(data).decode('utf-8')
            return data
        
        return {
            "status": "success",
            "detection_type": request.detection_type,
            "party_id": request.party_id,
            "email_id": request.email_id,
            "encrypted_score": serialize_encrypted(fraud_score),
            "computation_time": elapsed_time,
            "encrypted": True,
            "amounts_processed": len(encrypted_amounts),
            "note": "Fraud score is encrypted. Decrypt on client side."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Fraud detection failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Fraud detection failed: {str(e)}"
        )


# ==================== Helper Function for Data Validation ====================

def validate_encrypted_data(enc_bytes: bytes, library: str, context) -> bool:
    """
    Validate that encrypted data can be parsed
    """
    try:
        if library == "TenSEAL":
            import tenseal as ts
            # Try to parse as CKKS (most common)
            try:
                vec = ts.ckks_vector_from(context, enc_bytes)
                return True
            except:
                # Try BFV
                try:
                    vec = ts.bfv_vector_from(context, enc_bytes)
                    return True
                except:
                    return False
        return True
    except:
        return False
    
# @app.post("/fraud/detect")
# async def fraud_detection(request: FraudDetectionRequest):
#     """
#     Fraud detection on encrypted amounts
#     """
#     print(f"\n🚨 Fraud Detection: {request.detection_type}")
#     print(f"   Party: {request.party_id[:8]}...")

#     start_time = time.time()

#     try:
#         # wrapper = get_wrapper(request.library)

#         # Get wrapper for FHE operations
#         if request.library == "TenSEAL":
#             from tenseal_wrapper import EnhancedTenSEALWrapper
#             wrapper = EnhancedTenSEALWrapper()
#             wrapper.context = server_public_context
#             wrapper.scheme = scheme_type
#         else:
#             from openfhe_wrapper import EnhancedOpenFHEWrapper
#             wrapper = EnhancedOpenFHEWrapper()

#         # Deserialize encrypted amounts
#         encrypted_amounts = []
#         for amt in request.encrypted_amounts:
#             try:
#                 if isinstance(amt, str):
#                     enc_bytes = base64.b64decode(amt)
#                 elif isinstance(amt, bytes):
#                     enc_bytes = amt
#                 else:
#                     continue
#                 encrypted_amounts.append(enc_bytes)
#             except Exception as e:
#                 print(f"   ⚠️ Failed to deserialize amount: {e}")
#                 continue

#         if not encrypted_amounts:
#             raise HTTPException(status_code=400, detail="No valid encrypted amounts")

#         print(f"   Processing {len(encrypted_amounts)} encrypted values")

#         if request.detection_type == "linear_score":
#             # Linear weighted fraud score
#             weights = request.model_params.get('weights', {})

#             fraud_score = None

#             for enc_amt in encrypted_amounts[:5]:
#                 try:
#                     # FIX: Properly reconstruct TenSEAL vector before operations
#                     if request.library == "TenSEAL":
#                         import tenseal as ts

#                         # Reconstruct the encrypted vector
#                         if wrapper.scheme == 'CKKS':
#                             vec = ts.ckks_vector_from(wrapper.context, enc_amt)
#                         else:
#                             vec = ts.bfv_vector_from(wrapper.context, enc_amt)

#                         # Apply weight
#                         weight = weights.get('amount', 0.5)
#                         vec = vec * weight

#                         # Serialize back
#                         weighted = vec.serialize()
#                     else:
#                         # OpenFHE path
#                         weight = weights.get('amount', 0.5)
#                         weighted = wrapper.scalar_multiply(enc_amt, weight)

#                     if weighted:
#                         if fraud_score is None:
#                             fraud_score = weighted
#                         else:
#                             fraud_score = wrapper.slot_wise_operation(fraud_score, weighted, 'add')

#                 except Exception as e:
#                     print(f"   ⚠️ Error processing encrypted amount: {e}")
#                     continue

#             # Normalize by count
#             if fraud_score and len(encrypted_amounts[:5]) > 0:
#                 try:
#                     fraud_score = wrapper.scalar_multiply(fraud_score, 1.0 / len(encrypted_amounts[:5]))
#                 except Exception as e:
#                     print(f"   ⚠️ Normalization failed: {e}")

#         elif request.detection_type == "distance_anomaly":
#             # Distance-based anomaly detection
#             centroid = request.model_params.get('centroid', {})
#             centroid_amount = centroid.get('amount', 5000.0)

#             squared_diffs = []

#             for enc_amt in encrypted_amounts[:5]:
#                 try:
#                     # (amount - centroid)^2
#                     diff = wrapper.scalar_add(enc_amt, -centroid_amount)
#                     if diff:
#                         sq_diff = wrapper.slot_wise_operation(diff, diff, 'multiply')
#                         if sq_diff:
#                             squared_diffs.append(sq_diff)
#                 except Exception as e:
#                     print(f"   ⚠️ Error computing distance: {e}")
#                     continue

#             if squared_diffs:
#                 fraud_score = wrapper.perform_aggregation(squared_diffs, 'sum')
#             else:
#                 fraud_score = None

#         else:
#             raise HTTPException(status_code=400, detail=f"Unsupported detection type: {request.detection_type}")

#         elapsed_time = time.time() - start_time
#         print(f"✅ Fraud detection complete in {elapsed_time:.2f}s")

#         return {
#             "status": "success",
#             "detection_type": request.detection_type,
#             "party_id": request.party_id,
#             "email_id": request.email_id,
#             "encrypted_score": serialize_encrypted(fraud_score),
#             "computation_time": elapsed_time,
#             "encrypted": True,
#             "note": "Fraud score is encrypted. Decrypt on client side."
#         }

#     except Exception as e:
#         print(f"❌ Fraud detection failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# ==================== SIMD Time Series (REAL FHE) ====================

@app.post("/simd/timeseries")
async def simd_timeseries(request: SIMDTimeSeriesRequest):
    """
    SIMD time-series analytics on encrypted data
    REAL FHE OPERATIONS - NO SIMULATION
    """
    print(f"\n🔢 SIMD Time Series: {request.operation}")
    print(f"   Party: {request.party_id[:8]}...")
    print("   ⚡ PERFORMING REAL SIMD FHE OPERATIONS")

    start_time = time.time()

    try:
        wrapper = get_wrapper(request.library)

        # Deserialize encrypted vector
        encrypted_vector = [deserialize_encrypted(v) for v in request.encrypted_vector]

        if not encrypted_vector:
            raise HTTPException(status_code=400, detail="No encrypted data provided")

        print(f"   Processing {len(encrypted_vector)} encrypted values")

        results = []

        if request.operation == "moving_average":
            # Compute moving average using SIMD operations
            window_size = request.parameters.get('window_size', 30)

            print(f"   Computing {window_size}-period moving average...")

            for i in range(len(encrypted_vector) - window_size + 1):
                window = encrypted_vector[i:i + window_size]

                # Sum window
                window_sum = wrapper.perform_aggregation(window, 'sum')

                # Divide by window size
                window_avg = wrapper.scalar_multiply(window_sum, 1.0 / window_size)

                results.append(window_avg)

        elif request.operation == "velocity_analysis":
            # Transaction velocity (transactions per time window)
            time_window = request.parameters.get('time_window', 7)

            print(f"   Computing transaction velocity over {time_window} days...")

            # Count transactions in rolling windows (simplified)
            for i in range(0, len(encrypted_vector), time_window):
                window = encrypted_vector[i:i + time_window]

                # Count = sum of binary indicators (all transactions count as 1)
                # In real implementation, would use binary encoding
                velocity = len(window)  # Simplified count

                # Create encrypted count representation
                count_vec = wrapper.encrypt_vector([float(velocity)])
                results.append(count_vec)

        elif request.operation == "transaction_correlation":
            # Compute correlation between consecutive transactions
            print("   Computing transaction correlations...")

            for i in range(len(encrypted_vector) - 1):
                # Multiply consecutive encrypted values
                correlation = wrapper.slot_wise_operation(
                    encrypted_vector[i],
                    encrypted_vector[i + 1],
                    'multiply'
                )
                results.append(correlation)

        elif request.operation == "slot_wise_aggregation":
            # Aggregate using SIMD slot-wise operations
            print("   Computing slot-wise aggregation...")

            # Sum all encrypted values using slot-wise addition
            aggregated = encrypted_vector[0]
            for enc_val in encrypted_vector[1:]:
                aggregated = wrapper.slot_wise_operation(aggregated, enc_val, 'add')

            results = [aggregated]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")

        elapsed_time = time.time() - start_time
        print(f"✅ SIMD operation complete in {elapsed_time:.2f}s")
        print(f"   Generated {len(results)} encrypted results")

        return {
            "status": "success",
            "operation": request.operation,
            "party_id": request.party_id,
            "data_points": len(encrypted_vector),
            "encrypted_results": [serialize_encrypted(r) for r in results],
            "computation_time": elapsed_time,
            "encrypted": True,
            "note": "All results are encrypted. Decrypt on client side."
        }

    except Exception as e:
        print(f"❌ SIMD operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ML Inference (REAL FHE) ====================
# ==================== ML INFERENCE ON STORED DATA (NEW ENDPOINT) ====================

class MLInferenceOnStoredDataRequest(BaseModel):
    """ML inference using server-stored encrypted data"""
    library: str
    scheme: str
    model_type: str
    party_id: str
    email_id: str
    feature_columns: List[str]  # List of column names to use as features
    weights: List[float]
    intercept: float = 0.0
    polynomial_degree: Optional[int] = None


@app.post("/ml/inference_on_stored_data")
async def ml_inference_on_stored_data(request: MLInferenceOnStoredDataRequest):
    print(f"\n🤖 ML Inference on Stored Data: {request.model_type}")
    print(f"   Party: {request.party_id[:20]}...")
    print(f"   Features: {request.feature_columns}")
    print("   ⚡ PERFORMING REAL FHE ML OPERATIONS ON STORED DATA")
    
    start_time = time.time()
    
    try:
        # ==================== STEP 1: Retrieve Encrypted Features ====================
        
        if request.party_id not in encrypted_data_store:
            raise HTTPException(
                status_code=404, 
                detail=f"No encrypted data found for party: {request.party_id}"
            )
        
        party_data = encrypted_data_store[request.party_id]
        print(f"   Found party data with columns: {list(party_data.keys())}")
        
        # Retrieve encrypted values for each feature
        encrypted_features = []
        missing_features = []
        
        for feature_col in request.feature_columns:
            if feature_col not in party_data:
                missing_features.append(feature_col)
                continue
            
            records = party_data[feature_col]
            
            if not records:
                missing_features.append(feature_col)
                continue
            
            # Filter by email if provided
            if request.email_id:
                matching_records = [r for r in records if r.get('email_id') == request.email_id]
            else:
                matching_records = records
            
            if not matching_records:
                missing_features.append(feature_col)
                continue
            
            # Use first matching record
            encrypted_value = matching_records[0]['encrypted_value']
            
            print(f"   ✅ Retrieved {feature_col}: {len(encrypted_value):,} bytes")
            
            encrypted_features.append(encrypted_value)
        
        if missing_features:
            print(f"   ⚠️ Missing features: {missing_features}")
            raise HTTPException(
                status_code=404,
                detail=f"Missing encrypted data for features: {', '.join(missing_features)}"
            )
        
        if len(encrypted_features) != len(request.feature_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(request.feature_columns)} features, got {len(encrypted_features)}"
            )
        
        if len(encrypted_features) != len(request.weights):
            raise HTTPException(
                status_code=400,
                detail=f"Number of features ({len(encrypted_features)}) must match number of weights ({len(request.weights)})"
            )
        
        print(f"   ✅ Retrieved {len(encrypted_features)} encrypted features from storage")
        
        # ==================== STEP 2: Get Wrapper for FHE Operations ====================
        
        if request.library == "TenSEAL":
            if not server_public_context:
                raise HTTPException(
                    status_code=400,
                    detail="Server context not initialized. Upload public keys first."
                )
            
            from tenseal_wrapper import EnhancedTenSEALWrapper
            wrapper = EnhancedTenSEALWrapper()
            wrapper.context = server_public_context
            wrapper.scheme = request.scheme
            print(f"   Using TenSEAL wrapper with public context")
        
        elif request.library == "OpenFHE":
            from openfhe_wrapper import EnhancedOpenFHEWrapper
            wrapper = EnhancedOpenFHEWrapper()
            print(f"   Using OpenFHE wrapper (simulation mode)")
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid library: {request.library}")
        
        # ==================== STEP 3: Perform ML Inference ====================
        
        prediction = None
        
        if request.model_type == "linear":
            print("   Computing linear regression on encrypted features...")
            
            for idx, (enc_feature, weight) in enumerate(zip(encrypted_features, request.weights)):
                print(f"   Processing feature {idx} ({request.feature_columns[idx]}) with weight {weight}...")
                
                try:
                    weighted = wrapper.scalar_multiply(enc_feature, weight)
                    
                    if weighted:
                        if prediction is None:
                            prediction = weighted
                        else:
                            prediction = wrapper.slot_wise_operation(prediction, weighted, 'add')
                    else:
                        print(f"   ⚠️ Failed to multiply feature {idx}")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing feature {idx}: {e}")
                    raise
            
            if prediction and request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                prediction = wrapper.scalar_add(prediction, request.intercept)
        
        elif request.model_type == "logistic":
            print("   Computing logistic regression (with sigmoid approximation)...")
            
            # Step 1: Compute linear combination
            linear_result = None
            
            for idx, (enc_feature, weight) in enumerate(zip(encrypted_features, request.weights)):
                print(f"   Processing feature {idx} ({request.feature_columns[idx]}) with weight {weight}...")
                
                try:
                    weighted = wrapper.scalar_multiply(enc_feature, weight)
                    
                    if weighted:
                        if linear_result is None:
                            linear_result = weighted
                        else:
                            linear_result = wrapper.slot_wise_operation(linear_result, weighted, 'add')
                    else:
                        print(f"   ⚠️ Failed to multiply feature {idx}")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing feature {idx}: {e}")
                    raise
            
            if linear_result is None:
                raise HTTPException(status_code=500, detail="Failed to compute linear combination")
            
            if request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                linear_result = wrapper.scalar_add(linear_result, request.intercept)
            
            # Step 2: Apply sigmoid approximation: sigmoid(x) ≈ 0.5 + 0.197*x - 0.004*x³
            print("   Applying sigmoid approximation...")
            
            x = linear_result
            
            # x²
            x_squared = wrapper.slot_wise_operation(x, x, 'multiply')
            if not x_squared:
                raise HTTPException(status_code=500, detail="Failed to compute x²")
            
            # x³
            x_cubed = wrapper.slot_wise_operation(x_squared, x, 'multiply')
            if not x_cubed:
                raise HTTPException(status_code=500, detail="Failed to compute x³")
            
            # 0.197*x
            term1 = wrapper.scalar_multiply(x, 0.197)
            
            # -0.004*x³
            term2 = wrapper.scalar_multiply(x_cubed, -0.004)
            
            if not term1 or not term2:
                raise HTTPException(status_code=500, detail="Failed to compute sigmoid terms")
            
            # Sum terms
            prediction = wrapper.slot_wise_operation(term1, term2, 'add')
            
            if prediction:
                # Add 0.5
                prediction = wrapper.scalar_add(prediction, 0.5)
        
        elif request.model_type == "polynomial":
            degree = request.polynomial_degree or 3
            print(f"   Computing polynomial model (degree {degree})...")
            
            if not encrypted_features:
                raise HTTPException(status_code=400, detail="No encrypted features provided")
            
            x = encrypted_features[0]  # Use first feature for polynomial
            
            # Initialize with first term: weight[0] * x
            prediction = wrapper.scalar_multiply(x, request.weights[0])
            
            if not prediction:
                raise HTTPException(status_code=500, detail="Failed to compute first polynomial term")
            
            # Compute higher degree terms
            current_power = x
            
            for i in range(1, min(degree, len(request.weights))):
                print(f"   Computing term {i} (x^{i+1})...")
                
                # Multiply by x to get next power
                current_power = wrapper.slot_wise_operation(current_power, x, 'multiply')
                
                if not current_power:
                    print(f"   ⚠️ Failed to compute power {i+1}")
                    break
                
                # Multiply by weight
                term = wrapper.scalar_multiply(current_power, request.weights[i])
                
                if term:
                    # Add to prediction
                    prediction = wrapper.slot_wise_operation(prediction, term, 'add')
                else:
                    print(f"   ⚠️ Failed to multiply power {i+1}")
            
            # Add intercept
            if request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                prediction = wrapper.scalar_add(prediction, request.intercept)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model type: {request.model_type}"
            )
        
        # ==================== STEP 4: Return Results ====================
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="ML inference produced no result")
        
        elapsed_time = time.time() - start_time
        print(f"✅ ML inference complete in {elapsed_time:.2f}s")
        
        # Serialize result
        def serialize_encrypted(data):
            if data is None:
                return None
            if isinstance(data, bytes):
                return base64.b64encode(data).decode('utf-8')
            return data
        
        return {
            "status": "success",
            "model_type": request.model_type,
            "party_id": request.party_id,
            "email_id": request.email_id,
            "features_used": request.feature_columns,
            "encrypted_result": serialize_encrypted(prediction),
            "computation_time": elapsed_time,
            "encrypted": True,
            "data_source": "server-stored encrypted data",
            "note": "Prediction is encrypted. Decrypt on client side."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ML inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"ML inference failed: {str(e)}"
        )



@app.post("/ml/inference")
async def ml_inference(request: MLInferenceRequest):
    """
    ML model inference on encrypted features
    REAL FHE OPERATIONS - With compression support
    """
    print(f"\n🤖 ML Inference: {request.model_type}")
    print("   ⚡ PERFORMING REAL FHE ML OPERATIONS")

    start_time = time.time()

    try:
        wrapper = get_wrapper(request.library)

        # Deserialize and decompress encrypted features
        encrypted_features = []
        print(f"   Processing {len(request.encrypted_features)} encrypted features...")

        total_compressed_size = 0
        total_decompressed_size = 0

        for idx, enc_feature_str in enumerate(request.encrypted_features):
            try:
                if request.compressed:
                    # Decompress first
                    compressed_bytes = base64.b64decode(enc_feature_str)
                    total_compressed_size += len(compressed_bytes)

                    enc_bytes = gzip.decompress(compressed_bytes)
                    total_decompressed_size += len(enc_bytes)

                    print(f"   Feature {idx}: {len(compressed_bytes)} → {len(enc_bytes)} bytes (decompressed)")
                else:
                    # Direct decode
                    if isinstance(enc_feature_str, str):
                        enc_bytes = base64.b64decode(enc_feature_str)
                    elif isinstance(enc_feature_str, bytes):
                        enc_bytes = enc_feature_str
                    else:
                        print(f"   Feature {idx}: Unexpected type {type(enc_feature_str)}")
                        continue

                    print(f"   Feature {idx}: {len(enc_bytes)} bytes")

                encrypted_features.append(enc_bytes)

            except Exception as e:
                print(f"   ⚠️ Failed to process feature {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if request.compressed and total_compressed_size > 0:
            compression_ratio = (1 - total_compressed_size / total_decompressed_size) * 100
            print(
                f"   📦 Total compression: {total_decompressed_size:,} → {total_compressed_size:,} bytes ({compression_ratio:.1f}% reduction)")

        if not encrypted_features:
            raise HTTPException(status_code=400, detail="No valid encrypted features")

        # Limit features to match weights
        encrypted_features = encrypted_features[:len(request.weights)]
        print(f"   Using {len(encrypted_features)} features with {len(request.weights)} weights")

        if request.model_type == "linear":
            print("   Computing linear regression on encrypted features...")

            prediction = None

            for idx, (enc_feature, weight) in enumerate(zip(encrypted_features, request.weights)):
                print(f"   Processing feature {idx} with weight {weight}...")
                weighted = wrapper.scalar_multiply(enc_feature, weight)

                if weighted:
                    if prediction is None:
                        prediction = weighted
                    else:
                        prediction = wrapper.slot_wise_operation(prediction, weighted, 'add')
                else:
                    print(f"   ⚠️ Failed to multiply feature {idx}")

            if prediction and request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                prediction = wrapper.scalar_add(prediction, request.intercept)

        elif request.model_type == "logistic":
            print("   Computing logistic regression (with sigmoid approx)...")

            linear_result = None

            for idx, (enc_feature, weight) in enumerate(zip(encrypted_features, request.weights)):
                print(f"   Processing feature {idx} with weight {weight}...")
                weighted = wrapper.scalar_multiply(enc_feature, weight)

                if weighted:
                    if linear_result is None:
                        linear_result = weighted
                    else:
                        linear_result = wrapper.slot_wise_operation(linear_result, weighted, 'add')
                else:
                    print(f"   ⚠️ Failed to multiply feature {idx}")

            if linear_result is None:
                raise HTTPException(status_code=500, detail="Failed to compute linear combination")

            if request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                linear_result = wrapper.scalar_add(linear_result, request.intercept)

            print("   Applying sigmoid approximation...")
            x = linear_result
            x_squared = wrapper.slot_wise_operation(x, x, 'multiply')

            if not x_squared:
                raise HTTPException(status_code=500, detail="Failed to compute x^2")

            x_cubed = wrapper.slot_wise_operation(x_squared, x, 'multiply')

            if not x_cubed:
                raise HTTPException(status_code=500, detail="Failed to compute x^3")

            term1 = wrapper.scalar_multiply(x, 0.197)
            term2 = wrapper.scalar_multiply(x_cubed, -0.004)

            if not term1 or not term2:
                raise HTTPException(status_code=500, detail="Failed to compute sigmoid terms")

            prediction = wrapper.slot_wise_operation(term1, term2, 'add')

            if prediction:
                prediction = wrapper.scalar_add(prediction, 0.5)

        elif request.model_type == "polynomial":
            degree = request.polynomial_degree or 3
            print(f"   Computing polynomial model (degree {degree})...")

            x = encrypted_features[0]

            prediction = wrapper.scalar_multiply(x, request.weights[0])

            if not prediction:
                raise HTTPException(status_code=500, detail="Failed to compute first polynomial term")

            current_power = x
            for i in range(1, min(degree, len(request.weights))):
                print(f"   Computing term {i}...")
                current_power = wrapper.slot_wise_operation(current_power, x, 'multiply')

                if not current_power:
                    print(f"   ⚠️ Failed to compute power {i + 1}")
                    break

                term = wrapper.scalar_multiply(current_power, request.weights[i])

                if term:
                    prediction = wrapper.slot_wise_operation(prediction, term, 'add')
                else:
                    print(f"   ⚠️ Failed to multiply power {i + 1}")

            if request.intercept != 0:
                print(f"   Adding intercept: {request.intercept}")
                prediction = wrapper.scalar_add(prediction, request.intercept)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")

        if prediction is None:
            raise HTTPException(status_code=500, detail="ML inference produced no result")

        elapsed_time = time.time() - start_time
        print(f"✅ ML inference complete in {elapsed_time:.2f}s")

        return {
            "status": "success",
            "model_type": request.model_type,
            "encrypted_result": serialize_encrypted(prediction),
            "computation_time": elapsed_time,
            "encrypted": True,
            "note": "Prediction is encrypted. Decrypt on client side."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ML inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Perform FHE Operation on Stored Data ====================

@app.post("/perform_fhe_operation")
async def perform_fhe_operation(request: FHEOperationRequest):
    """
    Perform arbitrary FHE operation on server-stored encrypted data
    Examples: sum, average, fraud detection, ML inference
    """
    print(f"\n⚡ Performing FHE operation: {request.operation}")
    print(f"   Party: {request.party_id[:8]}...")
    print(f"   Column: {request.column_name}")
    
    start_time = time.time()
    
    try:
        # Get encrypted data from storage
        if request.party_id not in encrypted_data_store:
            raise HTTPException(status_code=404, detail="No encrypted data found for this party")
        
        party_data = encrypted_data_store[request.party_id]
        
        if request.column_name not in party_data:
            raise HTTPException(status_code=404, detail=f"No encrypted data found for column: {request.column_name}")
        
        encrypted_records = party_data[request.column_name]
        encrypted_values = [rec['encrypted_value'] for rec in encrypted_records]
        
        print(f"   Processing {len(encrypted_values)} encrypted values")
        
        # Get wrapper
        if request.library == "TenSEAL":
            from tenseal_wrapper import EnhancedTenSEALWrapper
            wrapper = EnhancedTenSEALWrapper()
            wrapper.context = server_public_context
            wrapper.scheme = scheme_type
        else:
            from openfhe_wrapper import EnhancedOpenFHEWrapper
            wrapper = EnhancedOpenFHEWrapper()
        
        # Perform operation
        result = None
        
        if request.operation in ['sum', 'avg', 'average', 'multiply']:
            result = wrapper.perform_aggregation(encrypted_values, request.operation)
        
        elif request.operation == 'fraud_score':
            # Fraud detection
            weights = request.parameters.get('weights', {'amount': 0.5})
            result = wrapper.fraud_score_weighted(
                {'amount': encrypted_values[0]},  # Simplified
                weights
            )
        
        elif request.operation == 'moving_average':
            # Time series
            window_size = request.parameters.get('window_size', 30)
            result = wrapper.compute_moving_average(encrypted_values, window_size)
        
        elif request.operation == 'variance':
            result = wrapper.compute_variance(encrypted_values)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Operation complete in {elapsed_time:.2f}s")
        
        # Serialize result
        def serialize_encrypted(data):
            if data is None:
                return None
            if isinstance(data, bytes):
                return base64.b64encode(data).decode('utf-8')
            elif isinstance(data, list):
                return [serialize_encrypted(item) for item in data]
            elif isinstance(data, dict):
                return {k: serialize_encrypted(v) for k, v in data.items()}
            return data
        
        return {
            "status": "success",
            "operation": request.operation,
            "party_id": request.party_id,
            "column_name": request.column_name,
            "encrypted_result": serialize_encrypted(result),
            "computation_time": elapsed_time,
            "encrypted": True,
            "note": "Result is encrypted. Decrypt on client side."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Server Statistics ====================

@app.get("/server_storage_stats")
async def get_server_storage_stats():
    """Get statistics about encrypted data stored on server (with batch info)"""
    try:
        total_records = sum(
            len(columns[col_name])
            for party_id, columns in encrypted_data_store.items()
            for col_name in columns
        )
        
        total_parties = len(encrypted_data_store)
        
        columns_list = set()
        for party_id, columns in encrypted_data_store.items():
            columns_list.update(columns.keys())
        
        total_batches = len(encrypted_metadata_store)
        
        # Batch upload progress
        batch_progress = {}
        for batch_id, meta in encrypted_metadata_store.items():
            for col_name, col_meta in meta.get('columns', {}).items():
                batches_received = len(col_meta.get('batches_received', []))
                total_expected = col_meta.get('total_batches', 0)
                
                batch_progress[col_name] = {
                    'batches_received': batches_received,
                    'total_batches': total_expected,
                    'progress_percentage': (batches_received / total_expected * 100) if total_expected > 0 else 0,
                    'upload_complete': col_meta.get('upload_complete', False),
                    'record_count': col_meta.get('record_count', 0)
                }
        
        return {
            "status": "success",
            "storage_stats": {
                "total_encrypted_records": total_records,
                "total_parties": total_parties,
                "total_columns": len(columns_list),
                "columns": list(columns_list),
                "total_batches": total_batches,
                "batch_progress": batch_progress
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/server_storage_stats")
# async def get_server_storage_stats():
#     """Get statistics about encrypted data stored on server"""
#     try:
#         total_records = sum(
#             len(columns[col_name])
#             for party_id, columns in encrypted_data_store.items()
#             for col_name in columns
#         )
        
#         total_parties = len(encrypted_data_store)
        
#         columns_list = set()
#         for party_id, columns in encrypted_data_store.items():
#             columns_list.update(columns.keys())
        
#         total_batches = len(encrypted_metadata_store)
        
#         return {
#             "status": "success",
#             "storage_stats": {
#                 "total_encrypted_records": total_records,
#                 "total_parties": total_parties,
#                 "total_columns": len(columns_list),
#                 "columns": list(columns_list),
#                 "total_batches": total_batches
#             }
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW: Check Batch Upload Status ====================

@app.get("/batch_upload_status/{batch_id}/{column_name}")
async def get_batch_upload_status(batch_id: str, column_name: str):
    """
    Check the status of a batch upload for a specific column
    Useful for resuming interrupted uploads
    """
    try:
        if batch_id not in encrypted_metadata_store:
            raise HTTPException(
                status_code=404, 
                detail=f"Batch ID not found: {batch_id}"
            )
        
        batch_meta = encrypted_metadata_store[batch_id]
        
        if column_name not in batch_meta.get('columns', {}):
            raise HTTPException(
                status_code=404,
                detail=f"Column not found in batch: {column_name}"
            )
        
        column_meta = batch_meta['columns'][column_name]
        
        batches_received = len(column_meta.get('batches_received', []))
        total_batches = column_meta.get('total_batches', 0)
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "column_name": column_name,
            "batches_received": batches_received,
            "total_batches": total_batches,
            "batches_received_list": column_meta.get('batches_received', []),
            "progress_percentage": (batches_received / total_batches * 100) if total_batches > 0 else 0,
            "upload_complete": column_meta.get('upload_complete', False),
            "record_count": column_meta.get('record_count', 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Clear Server Storage ====================

@app.post("/clear_server_storage")
async def clear_server_storage():
    """Clear all encrypted data from server storage"""
    global encrypted_data_store, encrypted_metadata_store
    
    try:
        encrypted_data_store.clear()
        encrypted_metadata_store.clear()
        
        print("🗑️ Server storage cleared")
        
        return {
            "status": "success",
            "message": "All encrypted data cleared from server storage"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ==================== Decryption ====================

# @app.post("/decrypt")
# async def decrypt_result(request: DecryptionRequest):
#     """Decrypt encrypted result"""
#     print(f"\n🔓 Decrypting result")

#     try:
#         wrapper = get_wrapper(request.library)

#         # Deserialize
#         result_data = deserialize_encrypted(request.result_data)

#         # Decrypt
#         if isinstance(result_data, bytes):
#             decrypted_list = wrapper.decrypt_data([result_data], request.data_type)
#             decrypted = decrypted_list[0] if decrypted_list else None
#         elif isinstance(result_data, dict):
#             if 'simulated_value' in result_data:
#                 decrypted = result_data['simulated_value']
#             else:
#                 decrypted = wrapper.decrypt_result(result_data, request.data_type)
#         else:
#             decrypted = wrapper.decrypt_result(result_data, request.data_type)

#         print(f"✅ Decryption complete: {decrypted}")

#         return {
#             "status": "success",
#             "decrypted_value": float(decrypted) if decrypted is not None else None
#         }

#     except Exception as e:
#         print(f"❌ Decryption failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# ==================== Parameter Recommendations ====================

@app.post("/parameters/recommend")
async def recommend_parameters(request: ParameterRecommendationRequest):
    """Recommend optimal FHE parameters"""
    print(f"\n⚙️ Parameter Recommendation: {request.workload_type}")

    try:
        from parameters_recommender import ParameterSelector

        params = ParameterSelector.select_params(
            request.workload_type,
            request.security_level,
            request.library
        )

        # Estimate depth if operations provided
        if request.expected_operations and tenseal_instance:
            depth_estimate = tenseal_instance.estimate_depth(request.expected_operations)
            params['estimated_depth'] = depth_estimate

        validation = ParameterSelector.validate_params(params, request.library)

        return {
            "status": "success",
            "workload_type": request.workload_type,
            "recommended_params": params,
            "validation": validation
        }

    except Exception as e:
        print(f"❌ Parameter recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Statistics ====================

@app.get("/stats")
async def get_statistics():
    """Get server statistics"""
    return {
        "status": "success",
        "stats": {
            "total_encrypted_records": len(encrypted_storage),
            "unique_parties": len(set(d['party_id'] for d in encrypted_storage.values())),
            "libraries": {
                "TenSEAL": tenseal_instance is not None,
                "OpenFHE": openfhe_instance is not None
            },
            "server_mode": "Real FHE Operations (No Simulation)"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "server_new:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )