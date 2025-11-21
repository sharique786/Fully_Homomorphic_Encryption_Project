"""
Enhanced FHE Server with Advanced Operations
New endpoints: ML inference, fraud detection, SIMD operations, parameter selection
Preserves all existing functionality
"""

import base64
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import enhanced wrappers (will fall back to existing if not available)
try:
    from openfhe_wrapper_enhanced import EnhancedOpenFHEWrapper as OpenFHEWrapper

    OPENFHE_ENHANCED = True
except:
    try:
        from openfhe_wrapper import OpenFHEWrapper

        OPENFHE_ENHANCED = False
    except:
        OPENFHE_ENHANCED = False

try:
    from tenseal_wrapper_enhanced import EnhancedTenSEALWrapper as TenSEALWrapper

    TENSEAL_ENHANCED = True
except:
    try:
        from tenseal_wrapper import TenSEALWrapper

        TENSEAL_ENHANCED = False
    except:
        TENSEAL_ENHANCED = False

# Initialize FastAPI
app = FastAPI(title="Enhanced FHE Server", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
openfhe_instance = None
tenseal_instance = None
encrypted_storage = {}
metadata_storage = {}
transaction_metadata = {}
request_lock = threading.Lock()
storage_lock = threading.Lock()
active_requests = {}


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


# ==================== NEW PYDANTIC MODELS ====================

class MLInferenceRequest(BaseModel):
    library: str
    model_type: str  # 'linear', 'logistic', 'polynomial'
    encrypted_features: List[str]  # Base64 encoded
    weights: List[float]
    intercept: float = 0.0
    polynomial_degree: Optional[int] = None


class FraudDetectionRequest(BaseModel):
    library: str
    detection_type: str  # 'linear_score', 'distance_anomaly'
    encrypted_transaction: Dict[str, str]  # feature_name: base64_encrypted_value
    model_params: Dict[str, Any]  # weights for linear, centroid for distance


class SIMDOperationRequest(BaseModel):
    library: str
    operation: str  # 'pack', 'rotate', 'dot_product', 'slot_wise_add'
    encrypted_vectors: List[str]  # Base64 encoded
    parameters: Optional[Dict[str, Any]] = None


class ParameterRecommendationRequest(BaseModel):
    workload_type: str  # 'transaction_analytics', 'fraud_scoring', 'ml_inference'
    security_level: int = 128
    expected_operations: Optional[List[str]] = None


class AdvancedAggregationRequest(BaseModel):
    library: str
    operation: str  # 'variance', 'moving_average', 'rolling_sum'
    encrypted_data: List[str]
    parameters: Optional[Dict[str, Any]] = None


class SearchFilterRequest(BaseModel):
    library: str
    filter_type: str  # 'range', 'threshold', 'equality'
    encrypted_data: List[str]
    filter_params: Dict[str, Any]


# ==================== EXISTING ENDPOINTS (PRESERVED) ====================

@app.on_event("startup")
async def startup_event():
    """Initialize FHE wrappers"""
    global openfhe_instance, tenseal_instance

    print("\n" + "=" * 60)
    print("🚀 ENHANCED FHE SERVER STARTING")
    print("=" * 60)

    if OPENFHE_ENHANCED:
        try:
            openfhe_instance = OpenFHEWrapper()
            print("✅ Enhanced OpenFHE wrapper initialized")
        except Exception as e:
            print(f"⚠️ OpenFHE initialization failed: {e}")

    if TENSEAL_ENHANCED:
        try:
            tenseal_instance = TenSEALWrapper()
            print("✅ Enhanced TenSEAL wrapper initialized")
        except Exception as e:
            print(f"⚠️ TenSEAL initialization failed: {e}")

    print("=" * 60)
    print("⚡ Server supports ADVANCED FHE operations")
    print("   - ML Inference (Linear, Logistic, Polynomial)")
    print("   - Fraud Detection (Weighted scoring, Distance-based)")
    print("   - SIMD Operations (Packing, Rotation, Dot products)")
    print("   - Financial Analytics (Variance, Moving averages)")
    print("   - Parameter Auto-selection")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    return {
        "message": "Enhanced FHE Server API",
        "version": "2.0.0",
        "libraries": {
            "OpenFHE": openfhe_instance is not None,
            "TenSEAL": tenseal_instance is not None,
            "Enhanced": OPENFHE_ENHANCED or TENSEAL_ENHANCED
        },
        "new_features": [
            "ML Inference",
            "Fraud Detection",
            "SIMD Operations",
            "Advanced Analytics",
            "Parameter Selection"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openfhe_ready": openfhe_instance is not None,
        "tenseal_ready": tenseal_instance is not None,
        "enhanced_features": OPENFHE_ENHANCED or TENSEAL_ENHANCED,
        "timestamp": datetime.now().isoformat()
    }


# Keep all existing endpoints from original server.py
# (generate_context, generate_keys, encrypt, aggregate, decrypt, etc.)
# These are preserved as-is

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
                scale=config.scale or 2 ** 40,
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
                scale=request.params.get('scale', 2 ** 40)
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
    """Encrypt data column - REAL FHE OPERATION (Thread-safe for concurrent requests)"""

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]

    with request_lock:
        active_requests[request_id] = {
            'column': request.column_name,
            'started': datetime.now().isoformat(),
            'status': 'processing'
        }

    print(f"\n🔒 [{request_id}] Encrypting {len(request.data)} values ({request.column_name})")
    print(f"   Library: {request.library}, Scheme: {request.scheme}")
    print(f"   Active requests: {len(active_requests)}")
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

        # Thread-safe storage operations
        storage_key = f"{request.batch_id}_{request.column_name}"

        with storage_lock:
            # Create storage with metadata for filtering
            encrypted_storage[storage_key] = {
                'encrypted_values': encrypted_data,
                'party_ids': request.party_ids if request.party_ids else [],
                'payment_dates': request.payment_dates if request.payment_dates else [],
                'original_values': request.data  # Store originals for reconciliation
            }

        # Helper function to convert encrypted data to preview string
        def get_ciphertext_preview(encrypted_value, max_length=100):
            """Convert encrypted value to preview string with truncation"""
            if encrypted_value is None:
                return None

            try:
                if isinstance(encrypted_value, bytes):
                    # Convert bytes to hex string
                    hex_str = encrypted_value.hex()
                    if len(hex_str) > max_length:
                        return hex_str[:max_length] + f"... ({len(hex_str)} chars total)"
                    return hex_str
                elif isinstance(encrypted_value, dict):
                    # Handle dict format (simulation mode)
                    ciphertext = str(encrypted_value.get('ciphertext', ''))
                    if len(ciphertext) > max_length:
                        return ciphertext[:max_length] + f"... ({len(ciphertext)} chars total)"
                    return ciphertext
                else:
                    # Convert to string
                    str_val = str(encrypted_value)
                    if len(str_val) > max_length:
                        return str_val[:max_length] + f"... ({len(str_val)} chars total)"
                    return str_val
            except Exception as e:
                return f"[Preview error: {str(e)}]"

        # Generate metadata with ciphertext preview (return max 100 records)
        metadata_records = []
        for i, (original, encrypted) in enumerate(zip(request.data[:100], encrypted_data[:100])):
            if encrypted is not None:
                # Get ciphertext preview
                ciphertext_preview = get_ciphertext_preview(encrypted, max_length=100)

                # Calculate actual ciphertext size
                if isinstance(encrypted, bytes):
                    actual_size = len(encrypted)
                elif isinstance(encrypted, dict):
                    actual_size = len(str(encrypted))
                else:
                    actual_size = len(str(encrypted))

                metadata_records.append({
                    "index": i,
                    "original_value": str(original)[:50],  # Trim original too
                    "ciphertext_preview": ciphertext_preview,
                    "ciphertext_full_size": actual_size,
                    "encrypted": True,
                    "scheme": request.scheme,
                    "library": request.library
                })

        # Thread-safe metadata storage
        with storage_lock:
            metadata_storage[storage_key] = {
                "column_name": request.column_name,
                "data_type": request.data_type,
                "count": len(encrypted_data),
                "batch_id": request.batch_id,
                "library": request.library,
                "scheme": request.scheme,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }

        elapsed_time = time.time() - start_time
        print(f"✅ [{request_id}] Encryption complete: {len(encrypted_data)} values in {elapsed_time:.2f}s")

        # Mark request as completed
        with request_lock:
            if request_id in active_requests:
                active_requests[request_id]['status'] = 'completed'
                active_requests[request_id]['duration'] = elapsed_time

        return {
            "status": "success",
            "batch_id": request.batch_id,
            "column_name": request.column_name,
            "encrypted_count": len(encrypted_data),
            "metadata_records": metadata_records,
            "encryption_time": elapsed_time,
            "request_id": request_id
        }

    except Exception as e:
        print(f"❌ [{request_id}] Encryption failed: {e}")
        import traceback
        traceback.print_exc()

        # Mark request as failed
        with request_lock:
            if request_id in active_requests:
                active_requests[request_id]['status'] = 'failed'
                active_requests[request_id]['error'] = str(e)

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


# ==================== NEW ADVANCED ENDPOINTS ====================

@app.post("/ml/inference")
async def ml_inference(request: MLInferenceRequest):
    """
    Perform ML model inference on encrypted data
    Supports: Linear regression, Logistic regression, Polynomial models
    """
    print(f"\n🤖 ML Inference: {request.model_type}")
    start_time = time.time()

    try:
        # Decode encrypted features
        encrypted_features = [base64.b64decode(enc) for enc in request.encrypted_features]

        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            if request.model_type == 'linear':
                result = openfhe_instance.credit_score_inference(
                    encrypted_features,
                    request.weights,
                    request.intercept
                )
            elif request.model_type == 'logistic':
                # Requires polynomial approximation for sigmoid
                result = openfhe_instance.credit_score_inference(
                    encrypted_features,
                    request.weights,
                    request.intercept
                )
                # TODO: Apply sigmoid approximation
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_type}")

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            if request.model_type == 'linear':
                result = tenseal_instance.linear_model_inference(
                    encrypted_features,
                    request.weights,
                    request.intercept
                )
            elif request.model_type == 'polynomial':
                if not request.polynomial_degree:
                    raise HTTPException(status_code=400, detail="polynomial_degree required")
                result = tenseal_instance.polynomial_approximation(
                    encrypted_features[0],
                    request.weights
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_type}")
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        elapsed = time.time() - start_time

        return {
            "status": "success",
            "model_type": request.model_type,
            "encrypted_result": base64.b64encode(result).decode() if result else None,
            "computation_time": elapsed,
            "encrypted": True
        }

    except Exception as e:
        print(f"❌ ML inference failed: {e}")
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
    with request_lock:
        active_count = len([r for r in active_requests.values() if r.get('status') == 'processing'])
        completed_count = len([r for r in active_requests.values() if r.get('status') == 'completed'])

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
            },
            "concurrent_requests": {
                "active": active_count,
                "completed": completed_count,
                "total_tracked": len(active_requests)
            }
        }
    }


@app.get("/active_requests")
async def get_active_requests():
    """Get currently active encryption requests"""
    with request_lock:
        return {
            "status": "success",
            "active_requests": dict(active_requests),
            "count": len(active_requests)
        }


@app.post("/fraud/detect")
async def fraud_detection(request: FraudDetectionRequest):
    """
    Perform fraud detection on encrypted transaction
    Methods: Linear scoring, Distance-based anomaly detection
    """
    print(f"\n🚨 Fraud Detection: {request.detection_type}")
    start_time = time.time()

    try:
        # Decode encrypted transaction features
        encrypted_features = {
            k: base64.b64decode(v)
            for k, v in request.encrypted_transaction.items()
        }

        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            if request.detection_type == 'linear_score':
                weights = request.model_params.get('weights', {})
                result = openfhe_instance.detect_fraud_linear(
                    encrypted_features,
                    weights
                )
            elif request.detection_type == 'distance_anomaly':
                centroid = request.model_params.get('centroid', {})
                result = openfhe_instance.detect_fraud_distance(
                    encrypted_features,
                    centroid
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported detection: {request.detection_type}")

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            if request.detection_type == 'linear_score':
                weights = request.model_params.get('weights', {})
                result = tenseal_instance.fraud_score_weighted(
                    encrypted_features,
                    weights
                )
            elif request.detection_type == 'distance_anomaly':
                centroid = request.model_params.get('centroid', {})
                result = tenseal_instance.distance_from_centroid(
                    encrypted_features,
                    centroid
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported detection: {request.detection_type}")
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        elapsed = time.time() - start_time

        return {
            "status": "success",
            "detection_type": request.detection_type,
            "encrypted_score": base64.b64encode(result).decode() if result else None,
            "computation_time": elapsed,
            "encrypted": True,
            "note": "Decrypt to get fraud score. Higher score = higher fraud risk"
        }

    except Exception as e:
        print(f"❌ Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simd/operation")
async def simd_operation(request: SIMDOperationRequest):
    """
    Perform SIMD operations on packed ciphertexts
    Operations: pack, rotate, dot_product, slot_wise operations
    """
    print(f"\n🔢 SIMD Operation: {request.operation}")
    start_time = time.time()

    try:
        # Decode encrypted vectors
        encrypted_vecs = [base64.b64decode(enc) for enc in request.encrypted_vectors]

        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            if request.operation == 'rotate':
                steps = request.parameters.get('steps', 1)
                result = openfhe_instance.rotate_ciphertext(encrypted_vecs[0], steps)
            elif request.operation == 'dot_product':
                result = openfhe_instance.inner_product(encrypted_vecs[0], encrypted_vecs[1])
            elif request.operation == 'slot_wise_multiply':
                result = openfhe_instance.slot_wise_multiply(encrypted_vecs[0], encrypted_vecs[1])
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            if request.operation == 'rotate':
                steps = request.parameters.get('steps', 1)
                result = tenseal_instance.rotate_vector(encrypted_vecs[0], steps)
            elif request.operation == 'dot_product':
                result = tenseal_instance.dot_product(encrypted_vecs[0], encrypted_vecs[1])
            elif request.operation == 'slot_wise_add':
                result = tenseal_instance.slot_wise_operation(
                    encrypted_vecs[0], encrypted_vecs[1], 'add'
                )
            elif request.operation == 'slot_wise_multiply':
                result = tenseal_instance.slot_wise_operation(
                    encrypted_vecs[0], encrypted_vecs[1], 'multiply'
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        elapsed = time.time() - start_time

        return {
            "status": "success",
            "operation": request.operation,
            "encrypted_result": base64.b64encode(result).decode() if result else None,
            "computation_time": elapsed,
            "encrypted": True
        }

    except Exception as e:
        print(f"❌ SIMD operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/advanced")
async def advanced_analytics(request: AdvancedAggregationRequest):
    """
    Perform advanced analytics on encrypted data
    Operations: variance, standard deviation, moving average, rolling sums
    """
    print(f"\n📈 Advanced Analytics: {request.operation}")
    start_time = time.time()

    try:
        # Decode encrypted data
        encrypted_data = [base64.b64decode(enc) for enc in request.encrypted_data]

        if request.library == "OpenFHE":
            if not openfhe_instance:
                raise HTTPException(status_code=503, detail="OpenFHE not available")

            if request.operation == 'variance':
                result = openfhe_instance.compute_variance(encrypted_data)
            elif request.operation == 'moving_average':
                window_size = request.parameters.get('window_size', 7)
                results = openfhe_instance.compute_rolling_average(encrypted_data, window_size)
                # Return multiple results
                return {
                    "status": "success",
                    "operation": request.operation,
                    "encrypted_results": [
                        base64.b64encode(r).decode() if r else None
                        for r in results
                    ],
                    "window_size": window_size,
                    "computation_time": time.time() - start_time
                }
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")

        elif request.library == "TenSEAL":
            if not tenseal_instance:
                raise HTTPException(status_code=503, detail="TenSEAL not available")

            if request.operation == 'variance':
                result = tenseal_instance.compute_variance(encrypted_data)
            elif request.operation == 'moving_average':
                window_size = request.parameters.get('window_size', 7)
                results = tenseal_instance.compute_moving_average(encrypted_data, window_size)
                return {
                    "status": "success",
                    "operation": request.operation,
                    "encrypted_results": [
                        base64.b64encode(r).decode() if r else None
                        for r in results
                    ],
                    "window_size": window_size,
                    "computation_time": time.time() - start_time
                }
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")
        else:
            raise HTTPException(status_code=400, detail="Invalid library")

        elapsed = time.time() - start_time

        return {
            "status": "success",
            "operation": request.operation,
            "encrypted_result": base64.b64encode(result).decode() if result else None,
            "computation_time": elapsed,
            "encrypted": True
        }

    except Exception as e:
        print(f"❌ Advanced analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parameters/recommend")
async def recommend_parameters(request: ParameterRecommendationRequest):
    """
    Recommend optimal FHE parameters based on workload
    Workload types: transaction_analytics, fraud_scoring, ml_inference, high_precision
    """
    print(f"\n⚙️ Parameter Recommendation: {request.workload_type}")

    try:
        # Use parameter selector
        from openfhe_wrapper_enhanced import ParameterSelector

        params = ParameterSelector.select_params(
            request.workload_type,
            request.security_level
        )

        # Estimate depth if operations provided
        depth_estimate = None
        if request.expected_operations:
            if tenseal_instance:
                depth_estimate = tenseal_instance.estimate_depth(request.expected_operations)
            params['estimated_depth'] = depth_estimate

        return {
            "status": "success",
            "workload_type": request.workload_type,
            "recommended_params": params,
            "usage_notes": [
                "Use these parameters when calling /generate_context",
                "Higher poly_modulus_degree = more security but slower",
                "More mult_depth = support deeper circuits but larger ciphertexts",
                "Scale affects precision vs noise trade-off"
            ]
        }

    except Exception as e:
        print(f"❌ Parameter recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/operations/supported")
async def get_supported_operations(library: str, scheme: str):
    """
    Get list of all supported operations for library and scheme
    """
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
            operations = tenseal_instance.get_supported_operations()
            limitations = tenseal_instance.get_scheme_limitations()

            return {
                "status": "success",
                "library": library,
                "scheme": scheme,
                "supported_operations": operations.get(scheme, {}),
                "limitations": limitations
            }
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


@app.get("/stats/advanced")
async def get_advanced_statistics():
    """Get advanced server statistics including operation counts"""
    with request_lock:
        active_count = len([r for r in active_requests.values() if r.get('status') == 'processing'])
        completed_count = len([r for r in active_requests.values() if r.get('status') == 'completed'])

    return {
        "status": "success",
        "stats": {
            "total_batches": len(set(m.get('batch_id') for m in metadata_storage.values())),
            "total_columns_encrypted": len(encrypted_storage),
            "total_encrypted_values": sum(m.get('count', 0) for m in metadata_storage.values()),
            "libraries": {
                "OpenFHE": {
                    "available": openfhe_instance is not None,
                    "enhanced": OPENFHE_ENHANCED
                },
                "TenSEAL": {
                    "available": tenseal_instance is not None,
                    "enhanced": TENSEAL_ENHANCED
                }
            },
            "concurrent_requests": {
                "active": active_count,
                "completed": completed_count,
                "total_tracked": len(active_requests)
            },
            "advanced_features": {
                "ml_inference": OPENFHE_ENHANCED or TENSEAL_ENHANCED,
                "fraud_detection": OPENFHE_ENHANCED or TENSEAL_ENHANCED,
                "simd_operations": OPENFHE_ENHANCED or TENSEAL_ENHANCED,
                "parameter_selection": True
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
