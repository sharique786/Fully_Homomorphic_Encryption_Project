"""
OpenFHE C++ Wrapper
Interfaces with compiled OpenFHE C++ libraries on Windows
"""

import subprocess
import json
import tempfile
import os
from typing import Dict, List, Any
from pathlib import Path


class OpenFHEWrapper:
    """Wrapper to call compiled OpenFHE C++ executables"""

    def __init__(self, openfhe_path: str = None):
        """
        Initialize OpenFHE wrapper

        Args:
            openfhe_path: Path to compiled OpenFHE installation
        """
        self.openfhe_path = openfhe_path or self.find_openfhe_path()
        self.executable_path = None
        self.temp_dir = tempfile.mkdtemp()

        # Try to find the executable
        self.find_executable()

    def find_openfhe_path(self) -> str:
        """Find OpenFHE installation path"""
        possible_paths = [
            r"C:\openfhe-development",
            r"C:\openfhe-development\build\bin\Release",
            r"C:\Program Files\openfhe",
            os.path.expanduser("~\openfhe-development"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def find_executable(self):
        """Find OpenFHE executable"""
        if not self.openfhe_path:
            return None

        # Look for our custom wrapper executable
        possible_names = [
            "fhe_wrapper.exe",
            "openfhe_wrapper.exe",
            "fhe_financial.exe"
        ]

        for name in possible_names:
            exe_path = os.path.join(self.openfhe_path, name)
            if os.path.exists(exe_path):
                self.executable_path = exe_path
                return exe_path

        return None

    def generate_keys(self, scheme: str, parameters: Dict) -> Dict[str, Any]:
        """
        Generate FHE keys using OpenFHE C++

        Args:
            scheme: BFV, BGV, or CKKS
            parameters: Scheme parameters

        Returns:
            Dictionary with keys
        """
        if not self.executable_path:
            return {'status': 'error', 'message': 'OpenFHE executable not found'}

        # Prepare command
        cmd = [
            self.executable_path,
            "generate_keys",
            scheme,
            str(parameters.get('poly_modulus_degree', 8192)),
            str(parameters.get('plain_modulus', 65537) if scheme in ['BFV', 'BGV']
                else parameters.get('scale_factor', 40))
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse JSON output
                return json.loads(result.stdout)
            else:
                return {'status': 'error', 'message': result.stderr}

        except subprocess.TimeoutExpired:
            return {'status': 'error', 'message': 'Operation timed out'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def encrypt_data(self, data: List, scheme: str) -> Dict[str, Any]:
        """
        Encrypt data using OpenFHE C++

        Args:
            data: List of values to encrypt
            scheme: Encryption scheme

        Returns:
            Dictionary with encrypted data
        """
        if not self.executable_path:
            return {'status': 'error', 'message': 'OpenFHE executable not found'}

        # Write data to temp file
        data_file = os.path.join(self.temp_dir, "input_data.txt")
        with open(data_file, 'w') as f:
            for value in data:
                f.write(f"{value}\n")

        # Prepare command
        cmd = [
            self.executable_path,
            "encrypt",
            data_file,
            scheme
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'status': 'error', 'message': result.stderr}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def perform_operation(self, operation: str, operand: Any = None) -> Dict[str, Any]:
        """
        Perform homomorphic operation using OpenFHE C++

        Args:
            operation: Operation type (add, multiply, etc.)
            operand: Operand value

        Returns:
            Dictionary with results
        """
        if not self.executable_path:
            return {'status': 'error', 'message': 'OpenFHE executable not found'}

        # Prepare command
        cmd = [
            self.executable_path,
            "operation",
            operation,
            str(operand) if operand is not None else "0"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'status': 'error', 'message': result.stderr}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def decrypt_data(self) -> Dict[str, Any]:
        """
        Decrypt data using OpenFHE C++

        Returns:
            Dictionary with decrypted data
        """
        if not self.executable_path:
            return {'status': 'error', 'message': 'OpenFHE executable not found'}

        # Prepare command
        cmd = [
            self.executable_path,
            "decrypt"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'status': 'error', 'message': result.stderr}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


def generate_cpp_wrapper_code() -> str:
    """
    Generate C++ wrapper code for OpenFHE
    This code should be compiled into an executable

    Returns:
        C++ source code as string
    """
    cpp_code = '''
#include "openfhe.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace lbcrypto;
using namespace std;

// Global variables for context and keys
CryptoContext<DCRTPoly> cryptoContext;
KeyPair<DCRTPoly> keyPair;
vector<Ciphertext<DCRTPoly>> encryptedData;

// Output JSON
void outputJSON(const string& status, const string& message = "") {
    cout << "{\\"status\\": \\"" << status << "\\"";
    if (!message.empty()) {
        cout << ", \\"message\\": \\"" << message << "\\"";
    }
    cout << "}" << endl;
}

// Generate keys
bool generateKeys(const string& scheme, uint32_t polyDegree, uint64_t modulus) {
    try {
        if (scheme == "BFV") {
            CCParams<CryptoContextBFVRNS> parameters;
            parameters.SetPlaintextModulus(modulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        else if (scheme == "BGV") {
            CCParams<CryptoContextBGVRNS> parameters;
            parameters.SetPlaintextModulus(modulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        else if (scheme == "CKKS") {
            CCParams<CryptoContextCKKSRNS> parameters;
            parameters.SetMultiplicativeDepth(10);
            parameters.SetScalingModSize(modulus);
            parameters.SetRingDim(polyDegree);
            parameters.SetBatchSize(polyDegree / 2);
            
            cryptoContext = GenCryptoContext(parameters);
        }
        
        cryptoContext->Enable(PKE);
        cryptoContext->Enable(KEYSWITCH);
        cryptoContext->Enable(LEVELEDSHE);
        
        keyPair = cryptoContext->KeyGen();
        cryptoContext->EvalMultKeyGen(keyPair.secretKey);
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});
        
        cout << "{\\"status\\": \\"success\\", ";
        cout << "\\"scheme\\": \\"" << scheme << "\\", ";
        cout << "\\"poly_degree\\": " << polyDegree << ", ";
        cout << "\\"public_key\\": \\"GENERATED\\", ";
        cout << "\\"private_key\\": \\"GENERATED\\", ";
        cout << "\\"evaluation_key\\": \\"GENERATED\\", ";
        cout << "\\"relinearization_key\\": \\"GENERATED\\", ";
        cout << "\\"galois_keys\\": \\"GENERATED\\"}";
        
        return true;
    }
    catch (const exception& e) {
        outputJSON("error", e.what());
        return false;
    }
}

// Encrypt data
bool encryptData(const string& dataFile, const string& scheme) {
    try {
        ifstream file(dataFile);
        if (!file.is_open()) {
            outputJSON("error", "Cannot open data file");
            return false;
        }
        
        vector<int64_t> values;
        string line;
        while (getline(file, line)) {
            values.push_back(stoll(line));
        }
        file.close();
        
        // Encrypt
        Plaintext plaintext = cryptoContext->MakePackedPlaintext(values);
        auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);
        encryptedData.push_back(ciphertext);
        
        cout << "{\\"status\\": \\"success\\", ";
        cout << "\\"encrypted_count\\": " << values.size() << ", ";
        cout << "\\"message\\": \\"Data encrypted successfully\\"}";
        
        return true;
    }
    catch (const exception& e) {
        outputJSON("error", e.what());
        return false;
    }
}

// Perform operation
bool performOperation(const string& operation, int64_t operand) {
    try {
        if (encryptedData.empty()) {
            outputJSON("error", "No encrypted data available");
            return false;
        }
        
        auto result = encryptedData[0];
        
        if (operation == "add") {
            auto operandPlaintext = cryptoContext->MakePackedPlaintext(vector<int64_t>(100, operand));
            result = cryptoContext->EvalAdd(result, operandPlaintext);
        }
        else if (operation == "multiply") {
            auto operandPlaintext = cryptoContext->MakePackedPlaintext(vector<int64_t>(100, operand));
            result = cryptoContext->EvalMult(result, operandPlaintext);
        }
        else if (operation == "square") {
            result = cryptoContext->EvalMult(result, result);
        }
        
        encryptedData[0] = result;
        
        cout << "{\\"status\\": \\"success\\", ";
        cout << "\\"operation\\": \\"" << operation << "\\", ";
        cout << "\\"message\\": \\"Operation completed\\"}";
        
        return true;
    }
    catch (const exception& e) {
        outputJSON("error", e.what());
        return false;
    }
}

// Decrypt data
bool decryptData() {
    try {
        if (encryptedData.empty()) {
            outputJSON("error", "No encrypted data available");
            return false;
        }
        
        Plaintext result;
        cryptoContext->Decrypt(keyPair.secretKey, encryptedData[0], &result);
        
        auto values = result->GetPackedValue();
        
        cout << "{\\"status\\": \\"success\\", ";
        cout << "\\"results\\": [";
        for (size_t i = 0; i < min((size_t)10, values.size()); i++) {
            cout << values[i];
            if (i < min((size_t)10, values.size()) - 1) cout << ", ";
        }
        cout << "]}";
        
        return true;
    }
    catch (const exception& e) {
        outputJSON("error", e.what());
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        outputJSON("error", "Usage: fhe_wrapper <command> [args...]");
        return 1;
    }
    
    string command = argv[1];
    
    if (command == "generate_keys") {
        if (argc < 5) {
            outputJSON("error", "Usage: generate_keys <scheme> <poly_degree> <modulus>");
            return 1;
        }
        
        string scheme = argv[2];
        uint32_t polyDegree = stoi(argv[3]);
        uint64_t modulus = stoull(argv[4]);
        
        return generateKeys(scheme, polyDegree, modulus) ? 0 : 1;
    }
    else if (command == "encrypt") {
        if (argc < 4) {
            outputJSON("error", "Usage: encrypt <data_file> <scheme>");
            return 1;
        }
        
        string dataFile = argv[2];
        string scheme = argv[3];
        
        return encryptData(dataFile, scheme) ? 0 : 1;
    }
    else if (command == "operation") {
        if (argc < 4) {
            outputJSON("error", "Usage: operation <op_type> <operand>");
            return 1;
        }
        
        string operation = argv[2];
        int64_t operand = stoll(argv[3]);
        
        return performOperation(operation, operand) ? 0 : 1;
    }
    else if (command == "decrypt") {
        return decryptData() ? 0 : 1;
    }
    else {
        outputJSON("error", "Unknown command: " + command);
        return 1;
    }
    
    return 0;
}
'''
    return cpp_code


def create_cmake_file() -> str:
    """
    Generate CMakeLists.txt for compiling the wrapper

    Returns:
        CMake configuration as string
    """
    cmake_content = '''
cmake_minimum_required(VERSION 3.12)
project(FHEWrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenFHE path
set(OPENFHE_ROOT "C:/openfhe-development" CACHE PATH "Path to OpenFHE")

# Add OpenFHE directories
include_directories(
    ${OPENFHE_ROOT}/src/pke/include
    ${OPENFHE_ROOT}/src/core/include
    ${OPENFHE_ROOT}/src/binfhe/include
    ${OPENFHE_ROOT}/third-party/include
    ${OPENFHE_ROOT}/third-party/cereal/include
)

link_directories(${OPENFHE_ROOT}/build/lib/Release)

# Add executable
add_executable(fhe_wrapper fhe_wrapper.cpp)

# Link OpenFHE libraries
target_link_libraries(fhe_wrapper 
    OPENFHEpke
    OPENFHEcore
    OPENFHEbinfhe
)

# Windows specific
if(WIN32)
    target_compile_definitions(fhe_wrapper PRIVATE _USE_MATH_DEFINES)
endif()
'''
    return cmake_content


# Instructions for compiling
COMPILATION_INSTRUCTIONS = """
# Compilation Instructions for OpenFHE C++ Wrapper

## Prerequisites
1. OpenFHE compiled and installed at: C:\\openfhe-development
2. CMake 3.12 or higher
3. Visual Studio 2019+ or MinGW-w64

## Steps to Compile:

### 1. Save the C++ Code
Save the generated C++ code as: fhe_wrapper.cpp

### 2. Save CMakeLists.txt
Save the CMake configuration as: CMakeLists.txt

### 3. Create Build Directory
```cmd
mkdir build
cd build
```

### 4. Configure with CMake
```cmd
cmake .. -DOPENFHE_ROOT="C:/openfhe-development"
```

### 5. Build
```cmd
cmake --build . --config Release
```

### 6. The executable will be at:
```
build/Release/fhe_wrapper.exe
```

### 7. Copy to Project Directory
```cmd
copy Release\\fhe_wrapper.exe ..\\fhe_wrapper.exe
```

## Testing the Executable

### Test Key Generation:
```cmd
fhe_wrapper.exe generate_keys BFV 8192 65537
```

### Expected Output:
```json
{"status": "success", "scheme": "BFV", "poly_degree": 8192, ...}
```

## Troubleshooting

### Error: Cannot find OpenFHE DLLs
Solution: Add OpenFHE lib directory to PATH:
```cmd
set PATH=%PATH%;C:\\openfhe-development\\build\\lib\\Release
```

### Error: LNK2019 (Unresolved external symbol)
Solution: Ensure OpenFHE libraries are in the correct location:
- OPENFHEpke.lib
- OPENFHEcore.lib
- OPENFHEbinfhe.lib

### Error: Cannot open include file
Solution: Verify OPENFHE_ROOT path in CMakeLists.txt
"""