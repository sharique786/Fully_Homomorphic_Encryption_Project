"""
OpenFHE C++ Wrapper
Interfaces with compiled OpenFHE C++ libraries on Windows using ctypes
"""

import ctypes
import json
import os
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64
import pickle


class OpenFHEWrapper:
    """Wrapper to call compiled OpenFHE C++ libraries via DLL"""

    def __init__(self, openfhe_path: str = None, dll_name: str = "openfhe_python_wrapper.dll"):
        """
        Initialize OpenFHE wrapper

        Args:
            openfhe_path: Path to compiled OpenFHE installation
            dll_name: Name of the wrapper DLL
        """
        self.openfhe_path = openfhe_path or self.find_openfhe_path()
        self.dll_path = None
        self.lib = None
        self.temp_dir = tempfile.mkdtemp()
        self.current_scheme = None
        self.keys_generated = False

        # Try to load the DLL
        self.load_dll(dll_name)

    def find_openfhe_path(self) -> Optional[str]:
        """Find OpenFHE installation path"""
        possible_paths = [
            r"C:\openfhe-development\build\lib\Release",
            r"C:\openfhe-development\build\bin\Release",
            r"C:\Program Files\openfhe",
            r"C:\openfhe-development",
            os.path.expanduser(r"~\openfhe-development\build\lib\Release"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return os.getcwd()  # Fallback to current directory

    def load_dll(self, dll_name: str) -> bool:
        """
        Load the OpenFHE wrapper DLL

        Args:
            dll_name: Name of the DLL file

        Returns:
            True if loaded successfully, False otherwise
        """
        # Search for DLL in multiple locations
        search_paths = [
            os.path.join(self.openfhe_path, dll_name),
            os.path.join(os.getcwd(), dll_name),
            dll_name,  # Current directory or PATH
        ]

        for path in search_paths:
            if os.path.exists(path):
                try:
                    self.dll_path = path
                    self.lib = ctypes.CDLL(path)
                    self._setup_function_signatures()
                    return True
                except Exception as e:
                    print(f"Failed to load DLL from {path}: {e}")
                    continue

        print(f"Warning: Could not load {dll_name}. Using simulation mode.")
        return False

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the DLL"""
        if not self.lib:
            return

        try:
            # Initialize FHE
            self.lib.InitializeFHE.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64]
            self.lib.InitializeFHE.restype = ctypes.c_bool

            # Encrypt integer
            self.lib.EncryptInteger.argtypes = [ctypes.c_int64]
            self.lib.EncryptInteger.restype = ctypes.c_void_p

            # Encrypt double
            self.lib.EncryptDouble.argtypes = [ctypes.c_double]
            self.lib.EncryptDouble.restype = ctypes.c_void_p

            # Decrypt integer
            self.lib.DecryptInteger.argtypes = [ctypes.c_void_p]
            self.lib.DecryptInteger.restype = ctypes.c_int64

            # Decrypt double
            self.lib.DecryptDouble.argtypes = [ctypes.c_void_p]
            self.lib.DecryptDouble.restype = ctypes.c_double

            # Add encrypted values
            self.lib.AddEncrypted.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.lib.AddEncrypted.restype = ctypes.c_void_p

            # Multiply encrypted values
            self.lib.MultiplyEncrypted.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.lib.MultiplyEncrypted.restype = ctypes.c_void_p

            # Subtract encrypted values
            self.lib.SubtractEncrypted.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.lib.SubtractEncrypted.restype = ctypes.c_void_p

            # Get public key
            self.lib.GetPublicKey.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self.lib.GetPublicKey.restype = ctypes.POINTER(ctypes.c_char)

            # Get scheme info
            self.lib.GetSchemeInfo.argtypes = []
            self.lib.GetSchemeInfo.restype = ctypes.c_char_p

            # Cleanup
            self.lib.Cleanup.argtypes = []
            self.lib.Cleanup.restype = None

        except AttributeError as e:
            print(f"Warning: Some functions not available in DLL: {e}")

    def generate_keys(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """
        Generate FHE keys using OpenFHE

        Args:
            scheme: BFV, BGV, or CKKS
            parameters: Scheme parameters

        Returns:
            Dictionary with keys and status
        """
        if not self.lib:
            return self._simulate_key_generation(scheme, parameters)

        try:
            scheme_bytes = scheme.encode('utf-8')
            poly_degree = parameters.get('poly_modulus_degree', 8192)

            if scheme in ['BFV', 'BGV']:
                modulus = parameters.get('plain_modulus', 65537)
            else:  # CKKS
                modulus = parameters.get('scale_factor', 40)

            # Call C++ function
            success = self.lib.InitializeFHE(scheme_bytes, poly_degree, modulus)

            if success:
                self.current_scheme = scheme
                self.keys_generated = True

                # Get key info (simplified)
                key_size = ctypes.c_int()

                return {
                    'status': 'success',
                    'scheme': scheme,
                    'poly_modulus_degree': poly_degree,
                    'public_key': base64.b64encode(b'PUBLIC_KEY_DATA').decode('utf-8'),
                    'private_key': base64.b64encode(b'PRIVATE_KEY_DATA').decode('utf-8'),
                    'evaluation_key': base64.b64encode(b'EVAL_KEY_DATA').decode('utf-8'),
                    'relinearization_key': base64.b64encode(b'RELIN_KEY_DATA').decode('utf-8'),
                    'galois_keys': base64.b64encode(b'GALOIS_KEYS_DATA').decode('utf-8'),
                    'message': 'Keys generated successfully using OpenFHE'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to initialize FHE scheme'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating keys: {str(e)}'
            }

    def encrypt_data(self, data: List, scheme: str) -> Dict[str, Any]:
        """
        Encrypt data using OpenFHE

        Args:
            data: List of values to encrypt
            scheme: Encryption scheme

        Returns:
            Dictionary with encrypted data
        """
        if not self.lib or not self.keys_generated:
            return self._simulate_encryption(data, scheme)

        try:
            encrypted_values = []

            for value in data:
                if scheme == 'CKKS':
                    # Encrypt as double
                    ct_ptr = self.lib.EncryptDouble(float(value))
                else:
                    # Encrypt as integer
                    ct_ptr = self.lib.EncryptInteger(int(value))

                if ct_ptr:
                    encrypted_values.append(ct_ptr)
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to encrypt value: {value}'
                    }

            # Store ciphertext pointers for later operations
            self.encrypted_data = encrypted_values

            return {
                'status': 'success',
                'encrypted_count': len(encrypted_values),
                'ciphertexts': encrypted_values,
                'message': f'Successfully encrypted {len(data)} values'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Encryption error: {str(e)}'
            }

    def perform_operation(self, operation: str, operand: Any = None,
                         ciphertext1: Any = None, ciphertext2: Any = None) -> Dict[str, Any]:
        """
        Perform homomorphic operation

        Args:
            operation: Operation type (add, multiply, subtract, etc.)
            operand: Operand value (for scalar operations)
            ciphertext1: First ciphertext
            ciphertext2: Second ciphertext

        Returns:
            Dictionary with results
        """
        if not self.lib or not self.keys_generated:
            return self._simulate_operation(operation, operand)

        try:
            if operation == 'add' and ciphertext1 and ciphertext2:
                result_ptr = self.lib.AddEncrypted(ciphertext1, ciphertext2)
            elif operation == 'multiply' and ciphertext1 and ciphertext2:
                result_ptr = self.lib.MultiplyEncrypted(ciphertext1, ciphertext2)
            elif operation == 'subtract' and ciphertext1 and ciphertext2:
                result_ptr = self.lib.SubtractEncrypted(ciphertext1, ciphertext2)
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported operation: {operation}'
                }

            if result_ptr:
                return {
                    'status': 'success',
                    'operation': operation,
                    'result_ciphertext': result_ptr,
                    'message': f'Operation {operation} completed successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Operation {operation} failed'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Operation error: {str(e)}'
            }

    def decrypt_data(self, ciphertexts: List = None) -> Dict[str, Any]:
        """
        Decrypt data

        Args:
            ciphertexts: List of ciphertext pointers to decrypt

        Returns:
            Dictionary with decrypted data
        """
        if not self.lib or not self.keys_generated:
            return self._simulate_decryption(ciphertexts)

        try:
            if ciphertexts is None:
                ciphertexts = getattr(self, 'encrypted_data', [])

            decrypted_values = []

            for ct in ciphertexts:
                if self.current_scheme == 'CKKS':
                    value = self.lib.DecryptDouble(ct)
                else:
                    value = self.lib.DecryptInteger(ct)
                decrypted_values.append(value)

            return {
                'status': 'success',
                'results': decrypted_values,
                'count': len(decrypted_values),
                'message': f'Successfully decrypted {len(decrypted_values)} values'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Decryption error: {str(e)}'
            }

    def cleanup(self):
        """Cleanup resources"""
        if self.lib:
            try:
                self.lib.Cleanup()
            except:
                pass

    # Simulation methods for when DLL is not available

    def _simulate_key_generation(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """Simulate key generation when DLL is not available"""
        import time
        time.sleep(0.5)  # Simulate processing time

        self.current_scheme = scheme
        self.keys_generated = True

        return {
            'status': 'success',
            'scheme': scheme,
            'poly_modulus_degree': parameters.get('poly_modulus_degree', 8192),
            'public_key': base64.b64encode(b'SIMULATED_PUBLIC_KEY').decode('utf-8'),
            'private_key': base64.b64encode(b'SIMULATED_PRIVATE_KEY').decode('utf-8'),
            'evaluation_key': base64.b64encode(b'SIMULATED_EVAL_KEY').decode('utf-8'),
            'relinearization_key': base64.b64encode(b'SIMULATED_RELIN_KEY').decode('utf-8'),
            'galois_keys': base64.b64encode(b'SIMULATED_GALOIS_KEYS').decode('utf-8'),
            'message': '⚠️ SIMULATION MODE - Keys generated (not real FHE)',
            'simulation': True
        }

    def _simulate_encryption(self, data: List, scheme: str) -> Dict[str, Any]:
        """Simulate encryption when DLL is not available"""
        import time
        time.sleep(0.3)

        # Store encrypted data as pickled objects for simulation
        self.encrypted_data = [pickle.dumps(val) for val in data]

        return {
            'status': 'success',
            'encrypted_count': len(data),
            'ciphertexts': self.encrypted_data,
            'message': f'⚠️ SIMULATION MODE - Encrypted {len(data)} values (not real FHE)',
            'simulation': True
        }

    def _simulate_operation(self, operation: str, operand: Any) -> Dict[str, Any]:
        """Simulate operation when DLL is not available"""
        import time
        time.sleep(0.2)

        return {
            'status': 'success',
            'operation': operation,
            'message': f'⚠️ SIMULATION MODE - Operation {operation} completed (not real FHE)',
            'simulation': True
        }

    def _simulate_decryption(self, ciphertexts: List) -> Dict[str, Any]:
        """Simulate decryption when DLL is not available"""
        import time
        time.sleep(0.2)

        if ciphertexts is None:
            ciphertexts = getattr(self, 'encrypted_data', [])

        # Unpickle simulated encrypted data
        decrypted_values = []
        for ct in ciphertexts:
            try:
                val = pickle.loads(ct)
                decrypted_values.append(val)
            except:
                decrypted_values.append(0)

        return {
            'status': 'success',
            'results': decrypted_values,
            'count': len(decrypted_values),
            'message': f'⚠️ SIMULATION MODE - Decrypted {len(decrypted_values)} values (not real FHE)',
            'simulation': True
        }


def generate_cpp_wrapper_code() -> str:
    """
    Generate C++ wrapper code for OpenFHE that compiles to a DLL
    This code should be compiled with the provided CMakeLists.txt

    Returns:
        C++ source code as string
    """
    cpp_code = '''
#include "openfhe.h"
#include <iostream>
#include <vector>
#include <string>

using namespace lbcrypto;

// Global variables
CryptoContext<DCRTPoly> cryptoContext;
KeyPair<DCRTPoly> keyPair;
std::string currentScheme;

extern "C" {

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// Initialize FHE scheme
EXPORT bool InitializeFHE(const char* scheme, int polyDegree, uint64_t modulus) {
    try {
        currentScheme = std::string(scheme);
        
        if (currentScheme == "BFV") {
            CCParams<CryptoContextBFVRNS> parameters;
            parameters.SetPlaintextModulus(modulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        else if (currentScheme == "BGV") {
            CCParams<CryptoContextBGVRNS> parameters;
            parameters.SetPlaintextModulus(modulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        else if (currentScheme == "CKKS") {
            CCParams<CryptoContextCKKSRNS> parameters;
            parameters.SetMultiplicativeDepth(10);
            parameters.SetScalingModSize(modulus);
            parameters.SetRingDim(polyDegree);
            parameters.SetBatchSize(polyDegree / 2);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        else {
            return false;
        }
        
        cryptoContext->Enable(PKE);
        cryptoContext->Enable(KEYSWITCH);
        cryptoContext->Enable(LEVELEDSHE);
        
        keyPair = cryptoContext->KeyGen();
        cryptoContext->EvalMultKeyGen(keyPair.secretKey);
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

// Encrypt integer
EXPORT void* EncryptInteger(int64_t value) {
    try {
        std::vector<int64_t> vec = {value};
        Plaintext plaintext = cryptoContext->MakePackedPlaintext(vec);
        auto* ciphertext = new Ciphertext<DCRTPoly>(
            cryptoContext->Encrypt(keyPair.publicKey, plaintext)
        );
        return ciphertext;
    }
    catch (const std::exception& e) {
        std::cerr << "Encryption error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Encrypt double (for CKKS)
EXPORT void* EncryptDouble(double value) {
    try {
        std::vector<double> vec = {value};
        Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(vec);
        auto* ciphertext = new Ciphertext<DCRTPoly>(
            cryptoContext->Encrypt(keyPair.publicKey, plaintext)
        );
        return ciphertext;
    }
    catch (const std::exception& e) {
        std::cerr << "Encryption error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Decrypt integer
EXPORT int64_t DecryptInteger(void* ctPtr) {
    try {
        auto* ct = static_cast<Ciphertext<DCRTPoly>*>(ctPtr);
        Plaintext result;
        cryptoContext->Decrypt(keyPair.secretKey, *ct, &result);
        return result->GetPackedValue()[0];
    }
    catch (const std::exception& e) {
        std::cerr << "Decryption error: " << e.what() << std::endl;
        return 0;
    }
}

// Decrypt double
EXPORT double DecryptDouble(void* ctPtr) {
    try {
        auto* ct = static_cast<Ciphertext<DCRTPoly>*>(ctPtr);
        Plaintext result;
        cryptoContext->Decrypt(keyPair.secretKey, *ct, &result);
        return result->GetRealPackedValue()[0];
    }
    catch (const std::exception& e) {
        std::cerr << "Decryption error: " << e.what() << std::endl;
        return 0.0;
    }
}

// Add encrypted values
EXPORT void* AddEncrypted(void* ct1Ptr, void* ct2Ptr) {
    try {
        auto* ct1 = static_cast<Ciphertext<DCRTPoly>*>(ct1Ptr);
        auto* ct2 = static_cast<Ciphertext<DCRTPoly>*>(ct2Ptr);
        auto* result = new Ciphertext<DCRTPoly>(
            cryptoContext->EvalAdd(*ct1, *ct2)
        );
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Addition error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Multiply encrypted values
EXPORT void* MultiplyEncrypted(void* ct1Ptr, void* ct2Ptr) {
    try {
        auto* ct1 = static_cast<Ciphertext<DCRTPoly>*>(ct1Ptr);
        auto* ct2 = static_cast<Ciphertext<DCRTPoly>*>(ct2Ptr);
        auto* result = new Ciphertext<DCRTPoly>(
            cryptoContext->EvalMult(*ct1, *ct2)
        );
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Multiplication error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Subtract encrypted values
EXPORT void* SubtractEncrypted(void* ct1Ptr, void* ct2Ptr) {
    try {
        auto* ct1 = static_cast<Ciphertext<DCRTPoly>*>(ct1Ptr);
        auto* ct2 = static_cast<Ciphertext<DCRTPoly>*>(ct2Ptr);
        auto* result = new Ciphertext<DCRTPoly>(
            cryptoContext->EvalSub(*ct1, *ct2)
        );
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Subtraction error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Get scheme info
EXPORT const char* GetSchemeInfo() {
    static std::string info = currentScheme;
    return info.c_str();
}

// Cleanup
EXPORT void Cleanup() {
    // Cleanup is handled by C++ destructors
}

} // extern "C"
'''
    return cpp_code


def create_cmake_file() -> str:
    """
    Generate CMakeLists.txt for compiling the wrapper DLL

    Returns:
        CMake configuration as string
    """
    cmake_content = '''cmake_minimum_required(VERSION 3.12)
project(OpenFHEPythonWrapper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE
find_package(OpenFHE REQUIRED)

# Include directories
include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Link directories
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

# Compiler flags
set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

# Create shared library (DLL on Windows)
add_library(openfhe_python_wrapper SHARED openfhe_python_wrapper.cpp)

# Link OpenFHE libraries
target_link_libraries(openfhe_python_wrapper PUBLIC 
    OPENFHEpke 
    OPENFHEbinfhe 
    OPENFHEcore
)

# Windows specific settings
if(WIN32)
    set_target_properties(openfhe_python_wrapper PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )
    
    # Copy OpenFHE DLLs to output directory
    add_custom_command(TARGET openfhe_python_wrapper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:OPENFHEpke> $<TARGET_FILE_DIR:openfhe_python_wrapper>
    )
    add_custom_command(TARGET openfhe_python_wrapper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:OPENFHEbinfhe> $<TARGET_FILE_DIR:openfhe_python_wrapper>
    )
    add_custom_command(TARGET openfhe_python_wrapper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:OPENFHEcore> $<TARGET_FILE_DIR:openfhe_python_wrapper>
    )
endif()

# Set output name
set_target_properties(openfhe_python_wrapper PROPERTIES
    OUTPUT_NAME "openfhe_python_wrapper"
    PREFIX ""
)
'''
    return cmake_content


# Compilation instructions
COMPILATION_INSTRUCTIONS = """
# Compilation Instructions for OpenFHE Python Wrapper DLL

## Prerequisites
1. OpenFHE compiled and installed
2. CMake 3.12 or higher
3. Visual Studio 2019/2022 with C++ support

## Steps to Compile:

### 1. Save Files
Save the generated C++ code as: `openfhe_python_wrapper.cpp`
Save the CMake configuration as: `CMakeLists.txt`

### 2. Set Environment Variables (Important!)
```cmd
set OpenFHE_DIR=C:\\openfhe-development\\build\\lib\\cmake\\openfhe
set PATH=%PATH%;C:\\openfhe-development\\build\\lib\\Release
```

### 3. Create Build Directory
```cmd
mkdir build
cd build
```

### 4. Configure with CMake
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:/openfhe-development/build"
```

OR for Visual Studio 2019:
```cmd
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="C:/openfhe-development/build"
```

### 5. Build the DLL
```cmd
cmake --build . --config Release
```

### 6. Verify DLL Creation
The DLL will be at: `build/Release/openfhe_python_wrapper.dll`

### 7. Copy to Project Directory
```cmd
copy Release\\openfhe_python_wrapper.dll ..\\
```

## Testing the Wrapper

### Test in Python:
```python
from openfhe_wrapper import OpenFHEWrapper

# Initialize wrapper (will auto-detect DLL)
wrapper = OpenFHEWrapper()

# Generate keys
result = wrapper.generate_keys('BFV', {'poly_modulus_degree': 8192, 'plain_modulus': 65537})
print(result)

# Encrypt data
encrypted = wrapper.encrypt_data([10, 20, 30], 'BFV')
print(encrypted)

# Decrypt data
decrypted = wrapper.decrypt_data()
print(decrypted)
```

## Troubleshooting

### Error: "DLL not found"
1. Ensure OpenFHE DLLs are in PATH
2. Copy all OpenFHE DLLs to the same directory as openfhe_python_wrapper.dll

### Error: "Cannot find OpenFHE"
Solution: Set CMAKE_PREFIX_PATH to your OpenFHE build directory

### Error: "Unresolved external symbol"
Solution: Verify OpenFHE was built with the same compiler (VS2019/VS2022)

### Simulation Mode Warning
If you see "SIMULATION MODE" in outputs, the DLL wasn't loaded. Check:
1. DLL exists in correct location
2. All dependencies (OpenFHE DLLs) are in PATH
3. DLL architecture matches Python (both 64-bit)
"""


if __name__ == "__main__":
    print("OpenFHE Wrapper Module")
    print("=" * 50)
    print("\nTo build the C++ wrapper DLL:")
    print("1. Run: print(generate_cpp_wrapper_code())")
    print("2. Save output as: openfhe_python_wrapper.cpp")
    print("3. Run: print(create_cmake_file())")
    print("4. Save output as: CMakeLists.txt")
    print("5. Follow compilation instructions")
    print("\n" + COMPILATION_INSTRUCTIONS)