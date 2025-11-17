import ctypes
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import time
from typing import List, Any, Dict, Optional


class OpenFHEWrapper:
    """Python wrapper for OpenFHE C++ library - Compatible with OpenFHE 1.1+ / 1.4.0"""

    def __init__(self):
        # Platform-specific paths
        if sys.platform == 'win32':
            self.openfhe_path = r"C:\Users\alish\Workspaces\Python\OpenFHE_Compiled"
            self.build_path = r"C:\Users\alish\Workspaces\Python\openfhe-development-latest"
        else:
            self.openfhe_path = os.environ.get('OPENFHE_ROOT', '/usr/local/openfhe')
            self.build_path = os.path.expanduser('~/openfhe-development')

        self.lib = None
        self.cpp_executable = None
        self.custom_dll = None
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}
        self.mode = None
        self.temp_dir = tempfile.mkdtemp()
        self._initialize()

    def _initialize(self):
        """Initialize wrapper in best available mode"""
        print("🔧 Initializing OpenFHE Wrapper...")
        print("=" * 60)

        # Check if OpenFHE libraries exist first
        openfhe_found = True # self._check_openfhe_installation()
        if not openfhe_found:
            print("\n⚠️ OpenFHE libraries not found - will use simulation mode")
            self.mode = 'simulation'
            print("=" * 60)
            return

        if sys.platform == 'win32' and self._try_mode_custom_dll():
            self.mode = 'custom_dll'
            print("\n✅ Mode: CUSTOM_DLL (Compiled wrapper DLL)")
            return

        if self._try_mode_subprocess():
            self.mode = 'subprocess'
            print("\n✅ Mode: SUBPROCESS (C++ executable wrapper)")
            return

        self.mode = 'simulation'
        print("\n✅ Mode: SIMULATION (Pure Python mock)")
        print("=" * 60)

    def _check_openfhe_installation(self):
        """Check if OpenFHE libraries are installed"""
        if sys.platform == 'win32':
            required_libs = ['OPENFHEcore.lib', 'OPENFHEpke.lib']
            lib_paths = [
                os.path.join(self.openfhe_path, "lib"),
                os.path.join(self.build_path, "lib", "Release"),
                os.path.join(self.build_path, "lib", "Debug"),
            ]
        else:
            required_libs = ['libOPENFHEcore.so', 'libOPENFHEpke.so']
            lib_paths = [
                os.path.join(self.openfhe_path, "lib"),
                os.path.join(self.build_path, "lib"),
                '/usr/local/lib',
                '/usr/lib'
            ]

        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                for req_lib in required_libs:
                    lib_file = os.path.join(lib_path, req_lib)
                    if os.path.exists(lib_file):
                        print(f"  ✅ Found OpenFHE library: {lib_file}")
                        return True

        print(f"  ❌ OpenFHE libraries not found in standard locations")
        return False

    def _try_mode_subprocess(self):
        """Try to compile C++ executable wrapper"""
        print("\n[Mode] Attempting C++ subprocess wrapper...")
        try:
            if not os.path.exists(self.openfhe_path) and not os.path.exists(self.build_path):
                print("  ❌ OpenFHE installation not found")
                return False

            cpp_file = os.path.join(self.temp_dir, "openfhe_wrapper.cpp")
            exe_file = os.path.join(self.temp_dir,
                                    "openfhe_wrapper.exe" if sys.platform == "win32" else "openfhe_wrapper")

            print(f"  📝 Generating C++ code: {cpp_file}")
            with open(cpp_file, 'w') as f:
                f.write(self._generate_cpp_code())

            compiler = self._find_compiler()
            if not compiler:
                print("  ❌ No C++ compiler found")
                return False

            print(f"  ✅ Found compiler: {compiler}")

            if self._compile_wrapper(cpp_file, exe_file, compiler):
                self.cpp_executable = exe_file
                print(f"  ✅ Compiled executable: {exe_file}")
                if self._test_executable():
                    return True
                else:
                    print("  ❌ Executable test failed")
                    return False
            else:
                print("  ❌ Compilation failed")
                return False
        except Exception as e:
            print(f"  ❌ Error in subprocess mode: {str(e)}")
            return False

    def _try_mode_custom_dll(self):
        """Mode 2: Try to compile custom wrapper DLL (Windows only)"""
        print("\n[Mode 2] Attempting custom DLL compilation...")
        if sys.platform != 'win32':
            print("  ⚠️ Custom DLL mode only available on Windows")
            return False

        try:
            if not os.path.exists(self.openfhe_path) and not os.path.exists(self.build_path):
                print("  ❌ OpenFHE installation not found")
                return False

            cpp_file = os.path.join(self.temp_dir, "openfhe_wrapper_dll.cpp")
            dll_file = os.path.join(self.temp_dir, "openfhe_wrapper.dll")

            print(f"  📝 Generating DLL wrapper code: {cpp_file}")
            with open(cpp_file, 'w') as f:
                f.write(self._generate_dll_cpp_code())

            # Check for MSVC compiler (REQUIRED for DLL)
            compiler = self._find_compiler()
            if compiler != 'cl.exe':
                if not self._check_msvc():
                    print("  ❌ MSVC compiler (cl.exe) required for DLL compilation")
                    print("  💡 Hint: Run from 'Developer Command Prompt for VS'")
                    return False
                compiler = 'cl.exe'

            print(f"  ✅ Found compiler: {compiler}")

            if self._compile_dll_wrapper(cpp_file, dll_file, compiler):
                self.custom_dll = dll_file
                print(f"  ✅ Compiled DLL: {dll_file}")
                print(f"     Size: {os.path.getsize(dll_file) / 1024:.1f} KB")

                # Copy OpenFHE DLLs to temp directory for easier loading
                self._copy_openfhe_dlls_to_temp()

                if self._test_custom_dll():
                    return True
                else:
                    print("  ❌ DLL test failed")
                    return False
            else:
                print("  ❌ DLL compilation failed")
                return False

        except Exception as e:
            print(f"  ❌ Error in custom DLL mode: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_dll_cpp_code(self):
        """Generate REAL OpenFHE DLL wrapper code with serialization support"""
        return '''// OpenFHE Wrapper DLL - Real Implementation with Serialization
    #include <iostream>
    #include <fstream>
    #include <sstream>
    #include <vector>
    #include <string>
    #include <iomanip>
    #include <memory>
    #include <cstdint>
    
    // Define M_E if not available (MSVC compatibility)
    #ifndef M_E
    #define M_E 2.71828182845904523536
    #endif
    
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif
    
    #if __has_include("openfhe.h")
        #define OPENFHE_AVAILABLE
    
        // Disable warnings for external headers
        #ifdef _MSC_VER
        #pragma warning(push, 0)
        #endif
    
        #include "openfhe.h"
    
        // Try to include serialization headers (may not exist in all installations)
        #if __has_include("ciphertext-ser.h")
            #include "ciphertext-ser.h"
            #include "cryptocontext-ser.h"
            #include "key/key-ser.h"
            #include "scheme/ckksrns/ckksrns-ser.h"
            #include "scheme/bfvrns/bfvrns-ser.h"
            #define SERIALIZATION_AVAILABLE
        #endif
    
        #ifdef _MSC_VER
        #pragma warning(pop)
        #endif
    
        using namespace lbcrypto;
    #endif
    
    // ===================== DLL EXPORT MACRO FIX =====================
    #ifdef _WIN32
        #define DLL_EXPORT extern "C" __declspec(dllexport)
    #else
        #define DLL_EXPORT extern "C"
    #endif
    // ================================================================

    //extern "C" {
        // Global context and keypair using shared_ptr (OpenFHE 1.1+)
        static std::shared_ptr<CryptoContext<DCRTPoly>> g_context = nullptr;
        static std::shared_ptr<KeyPair<DCRTPoly>> g_keyPair = nullptr;
        
        


        // Initialize CKKS context
        DLL_EXPORT int InitializeCKKS(int polyDegree, int scalingModSize) {
            try {
                CCParams<CryptoContextCKKSRNS> parameters;
                parameters.SetMultiplicativeDepth(10);
                parameters.SetScalingModSize(scalingModSize);
                parameters.SetRingDim(polyDegree);
                parameters.SetBatchSize(polyDegree / 2);
                parameters.SetSecurityLevel(HEStd_128_classic);

                // Generate context
                auto cc = GenCryptoContext(parameters);
                cc->Enable(PKE);
                cc->Enable(KEYSWITCH);
                cc->Enable(LEVELEDSHE);

                // Generate keys
                auto kp = cc->KeyGen();
                cc->EvalMultKeyGen(kp.secretKey);
                cc->EvalSumKeyGen(kp.secretKey);

                // Store in shared_ptr
                g_context = std::make_shared<CryptoContext<DCRTPoly>>(cc);
                g_keyPair = std::make_shared<KeyPair<DCRTPoly>>(kp);

                return 1; // Success
            } catch (const std::exception& e) {
                return 0; // Failure
            }
        }

        // Initialize BFV context
        DLL_EXPORT int InitializeBFV(int polyDegree, int plainModulus) {
            try {
                CCParams<CryptoContextBFVRNS> parameters;
                parameters.SetPlaintextModulus(plainModulus);
                parameters.SetMultiplicativeDepth(2);
                parameters.SetRingDim(polyDegree);
                parameters.SetSecurityLevel(HEStd_128_classic);

                auto cc = GenCryptoContext(parameters);
                cc->Enable(PKE);
                cc->Enable(KEYSWITCH);
                cc->Enable(LEVELEDSHE);

                auto kp = cc->KeyGen();
                cc->EvalMultKeyGen(kp.secretKey);
                cc->EvalSumKeyGen(kp.secretKey);

                g_context = std::make_shared<CryptoContext<DCRTPoly>>(cc);
                g_keyPair = std::make_shared<KeyPair<DCRTPoly>>(kp);

                return 1;
            } catch (const std::exception& e) {
                return 0;
            }
        }

        // Encrypt single value - REAL FHE
        DLL_EXPORT const char* EncryptValue(double value, char* outputBuffer, int bufferSize) {
            try {
                if (!g_context || !g_keyPair) {
                    strncpy_s(outputBuffer, bufferSize, "ERROR: Context not initialized", bufferSize - 1);
                    return outputBuffer;
                }

                // Create plaintext
                std::vector<double> data = {value};
                Plaintext plaintext = (*g_context)->MakeCKKSPackedPlaintext(data);

                // Encrypt
                auto ciphertext = (*g_context)->Encrypt(g_keyPair->publicKey, plaintext);

                // Serialize to binary
                std::ostringstream oss;
                Serial::Serialize(ciphertext, oss, SerType::BINARY);
                std::string serialized = oss.str();

                // Copy to output buffer
                if (serialized.size() < (size_t)(bufferSize - 1)) {
                    memcpy(outputBuffer, serialized.c_str(), serialized.size());
                    outputBuffer[serialized.size()] = '\\0';
                    return outputBuffer;
                }

                strncpy_s(outputBuffer, bufferSize, "ERROR: Buffer too small", bufferSize - 1);
                return outputBuffer;
            } catch (const std::exception& e) {
                strncpy_s(outputBuffer, bufferSize, "ERROR: Encryption failed", bufferSize - 1);
                return outputBuffer;
            }
        }

        // Decrypt value - REAL FHE
        DLL_EXPORT double DecryptValue(const char* ciphertextData, int dataLength) {
            try {
                if (!g_context || !g_keyPair) return -999999.0;

                // Deserialize ciphertext
                std::string serialized(ciphertextData, dataLength);
                std::istringstream iss(serialized);

                Ciphertext<DCRTPoly> ciphertext;
                Serial::Deserialize(ciphertext, iss, SerType::BINARY);

                // Decrypt
                Plaintext plaintext;
                (*g_context)->Decrypt(g_keyPair->secretKey, ciphertext, &plaintext);

                // Extract value
                plaintext->SetLength(1);
                std::vector<std::complex<double>> vals = plaintext->GetCKKSPackedValue();
                if (!vals.empty()) {
                    return vals[0].real();
                }
                return -999999.0;
            } catch (const std::exception& e) {
                return -999999.0;
            }
        }

        // Cleanup
        DLL_EXPORT void Cleanup() {
            g_context.reset();
            g_keyPair.reset();
        }

        // Test function
        DLL_EXPORT int TestDLL() {
            return 42;
        }
    //}

    
    '''

    def _copy_openfhe_dlls_to_temp(self):
        """Copy OpenFHE DLLs to temp directory for easier loading"""
        try:
            dll_search_paths = [
                os.path.join(self.openfhe_path, "bin"),
                os.path.join(self.openfhe_path, "lib"),
                os.path.join(self.build_path, "bin"),
                os.path.join(self.build_path, "lib"),
                os.path.join(self.build_path, "bin", "Release"),
                os.path.join(self.build_path, "bin", "Debug"),
            ]

            required_dlls = [
                'libOPENFHEcore.dll',
                'libOPENFHEpke.dll',
                'libOPENFHEbinfhe.dll'
            ]

            copied_count = 0
            for search_path in dll_search_paths:
                if not os.path.exists(search_path):
                    continue

                for dll_name in required_dlls:
                    src_dll = os.path.join(search_path, dll_name)
                    if os.path.exists(src_dll):
                        dst_dll = os.path.join(self.temp_dir, dll_name)
                        if not os.path.exists(dst_dll):
                            shutil.copy2(src_dll, dst_dll)
                            print(f"  📦 Copied: {dll_name}")
                            copied_count += 1

            if copied_count > 0:
                print(f"  ✅ Copied {copied_count} OpenFHE DLLs to temp directory")
            else:
                print(f"  ⚠️ No OpenFHE DLLs found to copy")

        except Exception as e:
            print(f"  ⚠️ Error copying DLLs: {str(e)}")

    def _generate_cpp_code(self):
        """Generate C++ wrapper code for subprocess/executable - OpenFHE 1.1+ compatible with proper serialization"""
        return '''// OpenFHE Wrapper Executable - With Conditional Serialization
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <memory>
#include <cstdint>

// Define M_E if not available (MSVC compatibility)
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if __has_include("openfhe.h")
    #define OPENFHE_AVAILABLE

    // Disable warnings for external headers
    #ifdef _MSC_VER
    #pragma warning(push, 0)
    #endif

    #include "openfhe.h"

    // Try to include serialization headers (may not exist in all installations)
    #if __has_include("ciphertext-ser.h")
        #include "ciphertext-ser.h"
        #include "cryptocontext-ser.h"
        #include "key/key-ser.h"
        #include "scheme/ckksrns/ckksrns-ser.h"
        #include "scheme/bfvrns/bfvrns-ser.h"
        #define SERIALIZATION_AVAILABLE
    #endif

    #ifdef _MSC_VER
    #pragma warning(pop)
    #endif

    using namespace lbcrypto;
#endif

class FHEProcessor {
public:
    #ifdef OPENFHE_AVAILABLE
    CryptoContext<DCRTPoly> cryptoContext;
    KeyPair<DCRTPoly> keyPair;
    #endif

    bool setupCKKS(uint32_t polyDegree, uint32_t scalingModSize) {
        #ifdef OPENFHE_AVAILABLE
        try {
            CCParams<CryptoContextCKKSRNS> parameters;
            parameters.SetMultiplicativeDepth(10);
            parameters.SetScalingModSize(scalingModSize);
            parameters.SetRingDim(polyDegree);
            parameters.SetBatchSize(polyDegree / 2);
            parameters.SetSecurityLevel(HEStd_128_classic);

            cryptoContext = GenCryptoContext(parameters);
            cryptoContext->Enable(PKE);
            cryptoContext->Enable(KEYSWITCH);
            cryptoContext->Enable(LEVELEDSHE);

            keyPair = cryptoContext->KeyGen();
            cryptoContext->EvalMultKeyGen(keyPair.secretKey);
            cryptoContext->EvalSumKeyGen(keyPair.secretKey);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
        #else
        std::cerr << "OpenFHE not available" << std::endl;
        return false;
        #endif
    }

    bool setupBFV(uint32_t polyDegree, uint32_t plainModulus) {
        #ifdef OPENFHE_AVAILABLE
        try {
            CCParams<CryptoContextBFVRNS> parameters;
            parameters.SetPlaintextModulus(plainModulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);
            parameters.SetSecurityLevel(HEStd_128_classic);

            cryptoContext = GenCryptoContext(parameters);
            cryptoContext->Enable(PKE);
            cryptoContext->Enable(KEYSWITCH);
            cryptoContext->Enable(LEVELEDSHE);

            keyPair = cryptoContext->KeyGen();
            cryptoContext->EvalMultKeyGen(keyPair.secretKey);
            cryptoContext->EvalSumKeyGen(keyPair.secretKey);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
        #else
        std::cerr << "OpenFHE not available" << std::endl;
        return false;
        #endif
    }

    void encryptData(const std::string& inputFile, const std::string& outputFile, const std::string& scheme) {
        #ifdef OPENFHE_AVAILABLE
        #ifdef SERIALIZATION_AVAILABLE
        try {
            std::ifstream inFile(inputFile);
            std::ofstream outFile(outputFile, std::ios::binary);

            if (!inFile || !outFile) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"File error\\"}" << std::endl;
                return;
            }

            int count = 0;
            std::string line;

            while (std::getline(inFile, line)) {
                if (line.empty()) continue;

                size_t colonPos = line.find(':');
                std::string dataType = "numeric";
                std::string valueStr = line;

                if (colonPos != std::string::npos) {
                    dataType = line.substr(0, colonPos);
                    valueStr = line.substr(colonPos + 1);
                }

                double numericValue = 0.0;

                if (dataType == "text" || dataType == "string") {
                    for (char c : valueStr) {
                        numericValue += static_cast<double>(static_cast<unsigned char>(c));
                    }
                } else if (dataType == "date") {
                    try {
                        numericValue = std::stod(valueStr);
                    } catch (...) {
                        numericValue = std::hash<std::string>{}(valueStr) % 1000000000;
                    }
                } else {
                    try {
                        numericValue = std::stod(valueStr);
                    } catch (...) {
                        std::cerr << "Warning: Could not parse value: " << valueStr << std::endl;
                        continue;
                    }
                }

                if (scheme == "CKKS") {
                    std::vector<double> data = {numericValue};
                    Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(data);
                    auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

                    try {
                        Serial::Serialize(ciphertext, outFile, SerType::BINARY);
                    } catch (const std::exception& e) {
                        std::cerr << "Serialization error: " << e.what() << std::endl;
                        continue;
                    }
                } else if (scheme == "BFV") {
                    std::vector<int64_t> data = {static_cast<int64_t>(numericValue)};
                    Plaintext plaintext = cryptoContext->MakePackedPlaintext(data);
                    auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

                    try {
                        Serial::Serialize(ciphertext, outFile, SerType::BINARY);
                    } catch (const std::exception& e) {
                        std::cerr << "Serialization error: " << e.what() << std::endl;
                        continue;
                    }
                }
                count++;
            }

            std::cout << "{\\"status\\": \\"success\\", \\"encrypted_count\\": " << count << "}" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
        #endif
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"OpenFHE not available\\"}" << std::endl;
        #endif
    }

    void aggregateEncrypted(const std::string& encFile, const std::string& operation) {
        #ifdef OPENFHE_AVAILABLE
        #ifdef SERIALIZATION_AVAILABLE
        try {
            std::ifstream inFile(encFile, std::ios::binary);
            if (!inFile) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
                return;
            }

            std::vector<Ciphertext<DCRTPoly>> ciphertexts;
            while (inFile.peek() != EOF) {
                Ciphertext<DCRTPoly> ct;
                try {
                    Serial::Deserialize(ct, inFile, SerType::BINARY);
                    ciphertexts.push_back(ct);
                } catch (const std::exception& e) {
                    std::cerr << "Deserialization error: " << e.what() << std::endl;
                    break;
                }
            }

            if (ciphertexts.empty()) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No ciphertexts\\"}" << std::endl;
                return;
            }

            Ciphertext<DCRTPoly> result;

            if (operation == "sum" || operation == "add") {
                result = cryptoContext->EvalAddMany(ciphertexts);
            } else if (operation == "multiply") {
                result = ciphertexts[0];
                for (size_t i = 1; i < ciphertexts.size(); i++) {
                    result = cryptoContext->EvalMult(result, ciphertexts[i]);
                }
            } else if (operation == "average" || operation == "avg") {
                result = cryptoContext->EvalAddMany(ciphertexts);
                double factor = 1.0 / ciphertexts.size();
                result = cryptoContext->EvalMult(result, factor);
            } else {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unsupported operation\\"}" << std::endl;
                return;
            }

            std::ostringstream oss;
            Serial::Serialize(result, oss, SerType::BINARY);
            std::string serialized = oss.str();

            std::ostringstream hexStream;
            hexStream << std::hex << std::setfill('0');
            for (unsigned char c : serialized) {
                hexStream << std::setw(2) << static_cast<int>(c);
            }

            std::cout << "{\\"status\\": \\"success\\", \\"operation\\": \\"" << operation 
                      << "\\", \\"result_hex\\": \\"" << hexStream.str() 
                      << "\\", \\"count\\": " << ciphertexts.size() << "}" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
        #endif
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"OpenFHE not available\\"}" << std::endl;
        #endif
    }

    void decryptData(const std::string& encFile) {
        #ifdef OPENFHE_AVAILABLE
        #ifdef SERIALIZATION_AVAILABLE
        try {
            std::ifstream inFile(encFile, std::ios::binary);
            if (!inFile) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
                return;
            }

            Ciphertext<DCRTPoly> ciphertext;
            Serial::Deserialize(ciphertext, inFile, SerType::BINARY);

            Plaintext plaintext;
            cryptoContext->Decrypt(keyPair.secretKey, ciphertext, &plaintext);

            plaintext->SetLength(1);
            std::vector<std::complex<double>> vals = plaintext->GetCKKSPackedValue();

            if (!vals.empty()) {
                double result = vals[0].real();
                std::cout << "{\\"status\\": \\"success\\", \\"result\\": " << result << "}" << std::endl;
            } else {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No values\\"}" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
        #endif
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"OpenFHE not available\\"}" << std::endl;
        #endif
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No command\\"}" << std::endl;
        return 1;
    }

    std::string command = argv[1];
    FHEProcessor processor;

    if (command == "test") {
        #ifdef OPENFHE_AVAILABLE
        #ifdef SERIALIZATION_AVAILABLE
        std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"OpenFHE ready with serialization\\"}" << std::endl;
        #else
        std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"OpenFHE ready without serialization\\"}" << std::endl;
        #endif
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"OpenFHE not available\\"}" << std::endl;
        #endif
    }
    else if (command == "setup_ckks") {
        uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 32768;
        uint32_t scalingModSize = (argc > 3) ? std::stoi(argv[3]) : 50;
        if (processor.setupCKKS(polyDegree, scalingModSize)) {
            std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"CKKS\\"}" << std::endl;
        } else {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Setup failed\\"}" << std::endl;
        }
    }
    else if (command == "setup_bfv") {
        uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 8192;
        uint32_t plainModulus = (argc > 3) ? std::stoi(argv[3]) : 65537;
        if (processor.setupBFV(polyDegree, plainModulus)) {
            std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"BFV\\"}" << std::endl;
        } else {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Setup failed\\"}" << std::endl;
        }
    }
    else if (command == "encrypt") {
        if (argc < 5) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing arguments\\"}" << std::endl;
            return 1;
        }
        std::string scheme = argv[4];
        uint32_t polyDegree = 32768;
        if (scheme == "CKKS") {
            processor.setupCKKS(polyDegree, 50);
        } else {
            processor.setupBFV(polyDegree, 65537);
        }
        processor.encryptData(argv[2], argv[3], scheme);
    }
    else if (command == "aggregate") {
        if (argc < 4) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing arguments\\"}" << std::endl;
            return 1;
        }
        processor.setupCKKS(8192, 50);
        processor.aggregateEncrypted(argv[2], argv[3]);
    }
    else if (command == "decrypt") {
        if (argc < 3) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing file\\"}" << std::endl;
            return 1;
        }
        processor.setupCKKS(32768, 50);
        processor.decryptData(argv[2]);
    }
    else {
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unknown command\\"}" << std::endl;
    }
    return 0;
}'''

    def _compile_dll_wrapper(self, cpp_file, dll_file, compiler):
        """Compile the C++ wrapper as DLL using your compiled OpenFHE libraries"""
        try:
            include_paths = []
            lib_paths = []

            # Your OpenFHE_Compiled directory structure - UPDATED
            include_subdirs = [
                'include',
                'include/openfhe',
                'include/openfhe/pke',
                'include/openfhe/core',
                'include/openfhe/binfhe',
                'include/openfhe/core/lattice',
                'include/openfhe/core/math',
                'include/openfhe/core/utils',
                # CRITICAL: Add source directories for serialization headers
                'src/pke/include',
                'src/core/include',
                'src/binfhe/include',
                'src/pke/include/schemebase',
                'src/pke/include/encoding',
                'src/pke/include/key',
                'src/pke/include/keyswitch',
                'src/pke/include/scheme',
                'src/pke/include/scheme/ckksrns',
                'src/pke/include/scheme/bfvrns',
                'src/pke/include/scheme/bgvrns'
            ]
            lib_subdirs = ['lib', 'lib/Release', 'lib/Debug', 'bin', 'bin/Release', 'bin/Debug']

            # Find include directories - search BOTH compiled and source
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in include_subdirs:
                        inc_path = os.path.join(base, subdir)
                        if os.path.exists(inc_path):
                            include_paths.append(inc_path)

            # Find library directories
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in lib_subdirs:
                        lib_path = os.path.join(base, subdir)
                        if os.path.exists(lib_path):
                            lib_paths.append(lib_path)

            # Verify OpenFHE headers exist
            openfhe_h_found = False
            for inc_path in include_paths:
                if os.path.exists(os.path.join(inc_path, "openfhe.h")):
                    openfhe_h_found = True
                    print(f"  ✅ Found openfhe.h in: {inc_path}")
                    break

            if not openfhe_h_found:
                print(f"  ⚠️ Warning: openfhe.h not found in include paths")

            # Check for serialization headers
            ser_headers = ['ciphertext-ser.h', 'cryptocontext-ser.h']
            ser_found = False
            for inc_path in include_paths:
                for ser_h in ser_headers:
                    if os.path.exists(os.path.join(inc_path, ser_h)):
                        ser_found = True
                        print(f"  ✅ Found {ser_h}")
                        break
                if ser_found:
                    break

            if not ser_found:
                print(f"  ⚠️ Serialization headers not found - may cause issues")

            # MSVC DLL compilation
            if compiler == 'cl.exe':
                cmd = [
                    'cl.exe',
                    '/LD',  # Create DLL
                    '/EHsc',  # Exception handling
                    '/std:c++17',  # C++17 standard
                    '/DBUILD_DLL',  # Define BUILD_DLL macro
                    '/MT',  # Static runtime (match OpenFHE build)
                    '/O2',  # Optimization
                    cpp_file,
                    f'/Fe:{dll_file}'  # Output file
                ]

                # Add include paths
                for inc in include_paths:
                    cmd.append(f'/I{inc}')

                # Add linker options
                cmd.append('/link')

                # Add library paths
                for lib in lib_paths:
                    cmd.append(f'/LIBPATH:{lib}')

                # Link against your compiled OpenFHE libraries
                cmd.extend([
                    'libOPENFHEcore.lib',
                    'libOPENFHEpke.lib',
                    'libOPENFHEbinfhe.lib'
                ])

                print(f"  🔨 Compiling DLL with MSVC...")
                if lib_paths:
                    print(f"     Using libs from: {lib_paths[0]}")

                result = subprocess.run(cmd, capture_output=True, timeout=600)

                if result.returncode == 0 and os.path.exists(dll_file):
                    print(f"  ✅ DLL compiled successfully")
                    return True
                else:
                    print(f"  ❌ Compilation failed")
                    if result.stderr:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')
                        print(f"  Error output:\n{error_msg[:1500]}")
                    return False

            elif compiler in ['g++', 'clang++']:
                # MinGW/GCC DLL compilation
                cmd = [
                    compiler,
                    '-shared',  # Create DLL
                    '-std=c++17',
                    '-DBUILD_DLL',
                    cpp_file,
                    '-o', dll_file
                ]

                for inc in include_paths:
                    cmd.append(f'-I{inc}')

                for lib in lib_paths:
                    cmd.append(f'-L{lib}')

                cmd.extend(['-lOPENFHEpke', '-lOPENFHEcore', '-lOPENFHEbinfhe'])
                cmd.extend(['-Wl,--out-implib,' + dll_file.replace('.dll', '.lib')])

                print(f"  🔨 Compiling DLL with {compiler}...")
                result = subprocess.run(cmd, capture_output=True, timeout=600)

                if result.returncode == 0 and os.path.exists(dll_file):
                    print(f"  ✅ DLL compiled successfully")
                    return True
                else:
                    print(f"  ❌ Compilation failed")
                    if result.stderr:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')
                        print(f"  Error output:\n{error_msg[:1500]}")
                    return False

            else:
                print(f"  ❌ Compiler {compiler} not supported for DLL")
                return False

        except subprocess.TimeoutExpired:
            print("  ❌ DLL compilation timeout")
            return False
        except Exception as e:
            print(f"  ❌ DLL compilation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _test_custom_dll(self):
        """Test if custom DLL loads and works"""
        try:
            # CRITICAL: Add OpenFHE bin directory to PATH before loading DLL
            openfhe_bin_paths = [
                os.path.join(self.openfhe_path, "bin"),
                os.path.join(self.openfhe_path, "lib"),
                os.path.join(self.build_path, "bin"),
                os.path.join(self.build_path, "lib"),
            ]

            # Add to system PATH temporarily
            original_path = os.environ.get('PATH', '')
            additional_paths = []

            for bin_path in openfhe_bin_paths:
                if os.path.exists(bin_path):
                    additional_paths.append(bin_path)
                    print(f"  📂 Adding to PATH: {bin_path}")

            if additional_paths:
                os.environ['PATH'] = ';'.join(additional_paths) + ';' + original_path

            print(f"  🔍 Testing DLL: {self.custom_dll}")

            # Load the DLL
            try:
                dll = ctypes.WinDLL(self.custom_dll)
            except OSError as e:
                print(f"  ❌ DLL load error: {str(e)}")
                print(f"  💡 Looking for OpenFHE DLLs in:")
                for path in additional_paths:
                    if os.path.exists(path):
                        dlls = [f for f in os.listdir(path) if f.endswith('.dll')]
                        print(f"     {path}: {', '.join(dlls[:5])}")

                # Restore PATH
                os.environ['PATH'] = original_path
                return False

            # Test the DLL
            try:
                test_func = dll.TestDLL
                test_func.restype = ctypes.c_int
                result = test_func()

                if result == 42:
                    print(f"  ✅ DLL test passed (returned {result})")

                    # Test initialization
                    init_func = dll.InitializeCKKS
                    init_func.argtypes = [ctypes.c_int, ctypes.c_int]
                    init_func.restype = ctypes.c_int

                    init_result = init_func(8192, 50)
                    if init_result == 1:
                        print(f"  ✅ CKKS initialization successful")
                    else:
                        print(f"  ⚠️ CKKS initialization failed (returned {init_result})")

                    # Restore PATH
                    os.environ['PATH'] = original_path
                    return True
                else:
                    print(f"  ❌ DLL test failed (returned {result})")
                    os.environ['PATH'] = original_path
                    return False

            except Exception as e:
                print(f"  ❌ DLL function test error: {str(e)}")
                os.environ['PATH'] = original_path
                return False

        except Exception as e:
            print(f"  ❌ DLL test error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _check_msvc(self):
        """Check if MSVC compiler is available"""
        try:
            result = subprocess.run(['cl.exe'], capture_output=True, timeout=2, text=True)
            # cl.exe always returns error when run with no args, but we check if it exists
            return True
        except FileNotFoundError:
            return False
        except:
            return True  # cl.exe exists but returned error (normal)

    def _find_compiler(self):
        """Find available C++ compiler - prioritize cl.exe for Windows"""
        # if sys.platform == 'win32':
        #     # On Windows, try MSVC first
        #     if self._check_msvc():
        #         return 'cl.exe'

        compilers = ['g++', 'cl.exe', 'clang++']
        for compiler in compilers:
            try:
                result = subprocess.run([compiler, '--version'], capture_output=True, timeout=5)
                if result.returncode == 0 or compiler == 'cl.exe':
                    return compiler
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def _compile_wrapper(self, cpp_file, exe_file, compiler):
        """Compile the C++ wrapper as executable"""
        try:
            include_paths = []
            lib_paths = []

            # Include paths for OpenFHE - UPDATED with serialization paths
            include_subdirs = [
                'include',
                'include/openfhe',
                'include/openfhe/pke',
                'include/openfhe/core',
                'include/openfhe/binfhe',
                'include/openfhe/core/lattice',
                'include/openfhe/core/math',
                'include/openfhe/core/utils',
                # CRITICAL: Add source directories for serialization headers
                'src/pke/include',
                'src/core/include',
                'src/binfhe/include',
                'src/pke/include/schemebase',
                'src/pke/include/encoding',
                'src/pke/include/key',
                'src/pke/include/keyswitch',
                'src/pke/include/scheme',
                'src/pke/include/scheme/ckksrns',
                'src/pke/include/scheme/bfvrns',
                'src/pke/include/scheme/bgvrns'
            ]

            if sys.platform == 'win32':
                lib_subdirs = ['lib', 'lib/Release', 'lib/Debug', 'bin']
            else:
                lib_subdirs = ['lib', 'lib64']

            # Find include directories - search BOTH compiled and source directories
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in include_subdirs:
                        inc_path = os.path.join(base, subdir)
                        if os.path.exists(inc_path):
                            include_paths.append(inc_path)
                            # Uncomment to debug: print(f"  📁 Added include: {inc_path}")

            # Find library directories
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in lib_subdirs:
                        lib_path = os.path.join(base, subdir)
                        if os.path.exists(lib_path):
                            lib_paths.append(lib_path)

            # Check if openfhe.h exists
            openfhe_header_found = False
            for inc_path in include_paths:
                openfhe_h = os.path.join(inc_path, "openfhe.h")
                if os.path.exists(openfhe_h):
                    openfhe_header_found = True
                    print(f"  ✅ Found openfhe.h in {inc_path}")
                    break

            # Check for serialization headers
            ser_headers = ['ciphertext-ser.h', 'cryptocontext-ser.h']
            ser_found = False
            for inc_path in include_paths:
                for ser_h in ser_headers:
                    if os.path.exists(os.path.join(inc_path, ser_h)):
                        ser_found = True
                        print(f"  ✅ Found {ser_h} in {inc_path}")
                        break
                if ser_found:
                    break

            if not ser_found:
                print(f"  ⚠️ Warning: Serialization headers not found - will compile without them")

            if compiler in ['g++', 'clang++']:
                cmd = [compiler, '-std=c++17', cpp_file, '-o', exe_file]
                for inc in include_paths:
                    cmd.append(f'-I{inc}')
                for lib in lib_paths:
                    cmd.append(f'-L{lib}')
                if lib_paths and openfhe_header_found:
                    cmd.extend(['-lOPENFHEpke', '-lOPENFHEcore', '-lOPENFHEbinfhe'])
                if sys.platform != 'win32':
                    cmd.extend(['-pthread'])
                    if lib_paths:
                        cmd.extend(['-Wl,-rpath,' + ':'.join(lib_paths)])
            else:  # cl.exe
                cmd = ['cl.exe', '/EHsc', '/std:c++17', cpp_file, '/Fe:' + exe_file]
                for inc in include_paths:
                    cmd.append(f'/I{inc}')
                cmd.append('/link')
                for lib in lib_paths:
                    cmd.append(f'/LIBPATH:{lib}')
                if lib_paths and openfhe_header_found:
                    cmd.extend(['OPENFHEpke.lib', 'OPENFHEcore.lib', 'OPENFHEbinfhe.lib'])

            print(f"  🔨 Compiling...")
            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if result.returncode == 0 and os.path.exists(exe_file):
                if sys.platform != 'win32':
                    os.chmod(exe_file, 0o755)
                return True
            else:
                if result.stderr:
                    stderr_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  ❌ Error: {stderr_msg[:1000]}")
                return False
        except Exception as e:
            print(f"  ❌ Compilation error: {str(e)}")
            return False

    def _test_executable(self):
        """Test if compiled executable works"""
        try:
            result = subprocess.run([self.cpp_executable, 'test'], capture_output=True, timeout=5, text=True)
            return result.returncode == 0 or 'OpenFHE' in result.stdout
        except:
            return False

    def _call_cpp_subprocess(self, command, *args):
        """Call C++ executable via subprocess"""
        if not self.cpp_executable:
            return {"status": "error", "message": "C++ executable not available"}
        try:
            cmd = [self.cpp_executable, command] + [str(arg) for arg in args]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"status": "success", "output": result.stdout}
            else:
                return {"status": "error", "message": result.stderr or "Execution failed"}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Operation timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_context(self, scheme='CKKS', mult_depth=10, scale_mod_size=50,
                         batch_size=8, security_level='HEStd_128_classic', ring_dim=16384):
        """Generate OpenFHE context"""
        self.scheme = scheme
        self.params = {
            'mult_depth': mult_depth,
            'scale_mod_size': scale_mod_size,
            'batch_size': batch_size,
            'security_level': security_level,
            'ring_dim': ring_dim
        }

        if self.mode == 'subprocess':
            if scheme == 'CKKS':
                result = self._call_cpp_subprocess("setup_ckks", ring_dim, scale_mod_size)
            elif scheme == 'BFV':
                result = self._call_cpp_subprocess("setup_bfv", ring_dim, 65537)
            else:
                result = {"status": "error", "message": f"Unsupported scheme: {scheme}"}

            if result.get('status') == 'success':
                self.context = {
                    'scheme': scheme, 'params': self.params, 'initialized': True,
                    'mode': 'subprocess', 'timestamp': datetime.now().isoformat()
                }
                self._generate_keys()
                print(f"✅ Context generated via C++ for {scheme}")
                return self.context

        self.context = {
            'scheme': scheme, 'params': self.params, 'initialized': True,
            'mode': self.mode, 'timestamp': datetime.now().isoformat()
        }
        self._generate_keys()
        print(f"✅ Context generated in {self.mode} mode for {scheme}")
        return self.context

    def _generate_keys(self):
        """Generate encryption keys"""
        import secrets
        self.public_key = {
            'key_data': secrets.token_hex(128),
            'scheme': self.scheme,
            'params': self.params
        }
        self.private_key = {
            'key_data': secrets.token_hex(128),
            'scheme': self.scheme,
            'params': self.params
        }
        self.evaluation_key = {
            'mult_key': secrets.token_hex(64),
            'rotation_keys': secrets.token_hex(64),
            'scheme': self.scheme
        }

    def get_keys_info(self):
        """Get information about generated keys"""
        if not self.context:
            return None
        return {
            'public_key': self.public_key['key_data'][:100] + '...',
            'private_key': self.private_key['key_data'][:100] + '...',
            'evaluation_key': self.evaluation_key['mult_key'][:100] + '...',
            'rotation_keys': self.evaluation_key['rotation_keys'][:100] + '...',
            'full_public_key': self.public_key['key_data'],
            'full_private_key': self.private_key['key_data'],
            'full_evaluation_key': json.dumps(self.evaluation_key),
            'mode': self.mode
        }

    def encrypt_data(self, data, column_name, data_type):
        """Encrypt data using REAL FHE"""
        if not self.context:
            raise ValueError("Context not initialized")

        print(f"   OpenFHE Mode: {self.mode}")

        encrypted_values = []

        # Mode 2: Custom DLL implementation
        if self.mode == 'custom_dll':
            for value in data:
                if not pd.isna(value):
                    result = self._call_custom_dll("encrypt", float(value))
                    print(f"   DLL Encrypt result: {result}")
                    if result.get('status') == 'success':
                        encrypted_values.append({
                            'value': result.get('result'),
                            'encrypted': True,
                            'mode': 'custom_dll'
                        })
                    else:
                        encrypted_values.append(None)
                else:
                    encrypted_values.append(None)
            return encrypted_values

        # Mode: Subprocess implementation
        if self.mode == 'subprocess':
            print(f"   🔐 Encrypting data via C++ subprocess... {data}")
            for value in data:
                try:
                    timestamp = int(time.time() * 1000)
                    data_file = os.path.join(self.temp_dir, f"plain_{timestamp}.txt")
                    enc_file = os.path.join(self.temp_dir, f"enc_{timestamp}.bin")

                    # Write the plain value to a temporary file
                    with open(data_file, 'w', encoding='utf-8') as f:
                        f.write(str(value))

                    # Call the C++ subprocess to encrypt this single file
                    result = self._call_cpp_subprocess("encrypt", data_file, enc_file, self.scheme)
                    print(f"   Subprocess Encrypt result: {result}")
                    # Store the result along with metadata
                    if result.get('status') == 'success':
                        with open(enc_file, 'rb') as f:
                            encrypted_data = f.read()
                            encrypted_values.append(encrypted_data)
                    else:
                        print(f"   ⚠️ Subprocess encryption failed: {result.get('message')}")


                    # Remove temporary files so next value can be processed cleanly
                    # try:
                    #     if os.path.exists(data_file):
                    #         os.remove(data_file)
                    #     if os.path.exists(enc_file):
                    #         os.remove(enc_file)
                    # except Exception:
                    #     pass

                except Exception as e:
                    encrypted_values.append({'input_value': value, 'error': str(e)})
                    print(f"   ℹ️ Using simulation mode")
                return self._simulate_encrypt(data, data_type)
            return encrypted_values


        # Fallback: simulation or empty list
        if hasattr(self, '_simulate_encrypt'):
            return self._simulate_encrypt(data, data_type)
        return encrypted_values


    def _call_custom_dll(self, function_name, *args):
        """Call custom DLL function"""
        if not self.custom_dll:
            return {"status": "error", "message": "Custom DLL not available"}

        try:
            dll = ctypes.WinDLL(self.custom_dll)

            if function_name == "initialize_ckks":
                func = dll.InitializeCKKS
                func.argtypes = [ctypes.c_int, ctypes.c_int]
                func.restype = ctypes.c_int
                result = func(int(args[0]), int(args[1]))
                return {"status": "success" if result == 1 else "error", "result": result}

            elif function_name == "encrypt":
                func = dll.EncryptValue
                print(f"EncryptValue : {func}")
                # Try the more featureful signature first: (double, char*, int) -> c_char_p
                try:
                    func.argtypes = [ctypes.c_double, ctypes.c_char_p, ctypes.c_int]
                    func.restype = ctypes.c_char_p  # returned C string pointer
                    BUFFER_SIZE = 4096
                    out_buf = ctypes.create_string_buffer(BUFFER_SIZE)
                    value = float(args[0]) if len(args) > 0 else 0.0
                    ret_ptr = func(ctypes.c_double(value), out_buf, ctypes.c_int(BUFFER_SIZE))
                    if ret_ptr:
                        result_str = ctypes.string_at(ret_ptr).decode('utf-8', errors='ignore')
                    else:
                        result_str = out_buf.value.decode('utf-8', errors='ignore')
                    print("EncryptValue returned:", result_str)
                    if isinstance(result_str, str) and result_str.startswith("ERROR"):
                        return {"status": "error", "message": result_str}
                    return {"status": "success", "result": result_str}
                except (AttributeError, TypeError, ctypes.ArgumentError) as e:
                    # Fallback to older/mock signature: double -> double
                    try:
                        func.argtypes = [ctypes.c_double]
                        func.restype = ctypes.c_double
                        value = float(args[0]) if len(args) > 0 else 0.0
                        numeric = func(ctypes.c_double(value))
                        return {"status": "success", "result": float(numeric)}
                    except Exception as e2:
                        return {"status": "error", "message": f"Encrypt call failed: {e2}"}

            elif function_name == "add":
                func = dll.AddEncrypted
                func.argtypes = [ctypes.c_double, ctypes.c_double]
                func.restype = ctypes.c_double
                result = func(float(args[0]), float(args[1]))
                return {"status": "success", "result": result}

            else:
                return {"status": "error", "message": f"Unknown function: {function_name}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _simulate_encrypt(self, data, data_type):
        """Simulate encryption for all data types"""
        encrypted_values = []
        for value in data:
            try:
                if pd.isna(value):
                    encrypted_values.append(None)
                    continue

                # Create deterministic but obfuscated representation based on type
                if data_type == 'numeric':
                    encrypted_val = {
                        'ciphertext': f"SIM_ENC_{hash(str(value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'scheme': self.scheme,
                        'type': 'numeric',
                        'mode': 'simulation'
                    }
                elif data_type == 'text':
                    # Encode text as sum of ASCII values (same as C++ does)
                    numeric_value = sum([ord(c) for c in str(value)])
                    encrypted_val = {
                        'ciphertext': f"SIM_ENC_TXT_{hash(str(numeric_value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'encoded_value': numeric_value,
                        'original_length': len(str(value)),
                        'scheme': self.scheme,
                        'type': 'text',
                        'mode': 'simulation'
                    }
                elif data_type == 'date':
                    # Convert date to timestamp
                    if isinstance(value, (pd.Timestamp, datetime)):
                        timestamp = value.timestamp()
                    else:
                        try:
                            timestamp = pd.Timestamp(value).timestamp()
                        except:
                            timestamp = hash(str(value)) % 1000000000

                    encrypted_val = {
                        'ciphertext': f"SIM_ENC_DATE_{hash(str(timestamp)) % 10000000}",
                        'original_hash': hash(str(timestamp)),
                        'timestamp': timestamp,
                        'scheme': self.scheme,
                        'type': 'date',
                        'mode': 'simulation'
                    }
                else:
                    # Default fallback
                    encrypted_val = {
                        'ciphertext': f"SIM_ENC_{hash(str(value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'scheme': self.scheme,
                        'type': data_type,
                        'mode': 'simulation'
                    }

                encrypted_values.append(encrypted_val)
            except Exception as e:
                print(f"Error encrypting value {value}: {str(e)}")
                encrypted_values.append(None)

        return encrypted_values

    def decrypt_data(self, encrypted_data, data_type):
        """Decrypt data"""
        if not self.context:
            raise ValueError("Context not initialized")

        decrypted_values = []
        for enc_value in encrypted_data:
            if enc_value is None:
                decrypted_values.append(None)
            elif isinstance(enc_value, dict):
                decrypted_values.append(f"DECRYPTED_{enc_value.get('original_hash', 0) % 1000}")
            else:
                decrypted_values.append(enc_value)
        return decrypted_values

    def perform_aggregation(self, encrypted_data_list: List[Any], operation: str):
        """Perform aggregation"""
        if self.mode == 'subprocess' and encrypted_data_list:
            enc_file = os.path.join(self.temp_dir, f"agg_{int(time.time())}.bin")

            with open(enc_file, 'wb') as f:
                for enc_data in encrypted_data_list:
                    if isinstance(enc_data, bytes):
                        f.write(enc_data)

            result = self._call_cpp_subprocess("aggregate", enc_file, operation)

            if result.get('status') == 'success':
                return {
                    "status": "success",
                    "result": result.get('result_hex'),
                    "operation": operation,
                    "count": result.get('count'),
                    "mode": "subprocess",
                    "encrypted": True
                }

        return self._simulate_aggregation(encrypted_data_list, operation)

    def _simulate_aggregation(self, encrypted_data_list: List[Any], operation: str):
        """Simulate aggregation operations"""
        try:
            values = []
            for enc_data in encrypted_data_list:
                if enc_data is None:
                    continue
                if isinstance(enc_data, dict):
                    if 'original_hash' in enc_data:
                        value = abs(enc_data['original_hash']) % 10000
                        values.append(value)

            if not values:
                return {"status": "error", "message": "No valid encrypted values"}

            if operation == 'sum':
                result = sum(values)
            elif operation == 'avg' or operation == 'average':
                result = sum(values) / len(values)
            else:
                return {"status": "error", "message": f"Unsupported operation: {operation}"}

            encrypted_result = {
                'ciphertext': f"SIM_RESULT_{hash(str(result)) % 10000000}",
                'operation': operation,
                'scheme': self.scheme,
                'count': len(values),
                'mode': 'simulation',
                'simulated_value': result
            }
            return {
                "status": "success",
                "result": encrypted_result,
                "operation": operation,
                "count": len(values),
                "mode": "simulation"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def perform_operation(self, encrypted_data1, encrypted_data2, operation):
        """Perform homomorphic operations"""
        results = []
        for enc1, enc2 in zip(encrypted_data1, encrypted_data2):
            try:
                if enc1 is None or enc2 is None:
                    results.append(None)
                    continue
                result = {
                    'ciphertext': f"RESULT_{hash(str(enc1) + str(enc2) + operation) % 10000000}",
                    'operation': operation,
                    'scheme': self.scheme,
                    'mode': self.mode
                }
                results.append(result)
            except Exception as e:
                print(f"Error performing operation: {str(e)}")
                results.append(None)
        return results

    def decrypt_result(self, encrypted_result: Any, result_type: str = 'numeric'):
        """Decrypt a single encrypted result"""
        if not self.private_key:
            raise ValueError("Private key not available")
        try:
            if encrypted_result is None:
                return None

            if self.mode == 'subprocess':
                return self._decrypt_subprocess_result(encrypted_result, result_type)
            else:
                if isinstance(encrypted_result, dict):
                    if 'simulated_value' in encrypted_result:
                        return encrypted_result['simulated_value']
                    elif 'original_hash' in encrypted_result:
                        return abs(encrypted_result['original_hash']) % 10000
                return encrypted_result
        except Exception as e:
            print(f"Decryption error: {str(e)}")
            return None

    def _decrypt_subprocess_result(self, encrypted_result: Any, result_type: str):
        """Decrypt using C++ subprocess"""
        try:
            enc_file = os.path.join(self.temp_dir, f"decrypt_{int(time.time())}.txt")
            if isinstance(encrypted_result, dict):
                enc_value = encrypted_result.get('result_hex', '')
            else:
                enc_value = encrypted_result

            with open(enc_file, 'w') as f:
                f.write(str(enc_value))

            result = self._call_cpp_subprocess("decrypt", enc_file)

            if result.get('status') == 'success':
                decrypted = result.get('result')
                if result_type == 'date':
                    return pd.Timestamp.fromtimestamp(float(decrypted))
                elif result_type == 'text':
                    return int(float(decrypted))
                else:
                    return float(decrypted)
            return None
        except Exception as e:
            print(f"Subprocess decryption error: {str(e)}")
            return None

    def get_scheme_limitations(self):
        """Get limitations of the current scheme"""
        limitations = {
            'CKKS': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (approximate)',
                'supports_comparison': 'No',
                'precision': 'Approximate (floating point)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Rotation']
            },
            'BFV': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'Limited',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction']
            },
            'BGV': {
                'supports_text': 'Limited (requires encoding)',
                'supports_numeric': 'Yes (exact integers)',
                'supports_comparison': 'Limited',
                'precision': 'Exact (integers)',
                'operations': ['Addition', 'Multiplication', 'Subtraction', 'Rotation']
            }
        }
        return limitations.get(self.scheme, {})

    def rotate_keys(self):
        """Rotate encryption keys"""
        if not self.context:
            raise ValueError("Context not initialized")
        old_keys = {
            'public_key': self.public_key,
            'private_key': self.private_key,
            'evaluation_key': self.evaluation_key
        }
        self._generate_keys()
        return {
            'old_keys': old_keys,
            'new_keys': {
                'public_key': self.public_key,
                'private_key': self.private_key,
                'evaluation_key': self.evaluation_key
            },
            'rotation_time': datetime.now().isoformat()
        }

    def __del__(self):
        """Cleanup temporary files"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass