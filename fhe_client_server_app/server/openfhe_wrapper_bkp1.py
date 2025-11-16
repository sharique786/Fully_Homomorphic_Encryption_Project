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
    """Python wrapper for OpenFHE C++ library with four operation modes"""

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

        # if self._try_mode_ctypes():
        #     self.mode = 'ctypes'
        #     print("\n✅ Mode: CTYPES (Direct DLL loading)")
        #     return
        # if sys.platform == 'win32' and self._try_mode_custom_dll():
        #     self.mode = 'custom_dll'
        #     print("\n✅ Mode: CUSTOM_DLL (Compiled wrapper DLL)")
        #     return
        if self._try_mode_subprocess():
            self.mode = 'subprocess'
            print("\n✅ Mode: SUBPROCESS (C++ executable wrapper)")
            return

        self.mode = 'simulation'
        print("\n✅ Mode: SIMULATION (Pure Python mock)")
        print("=" * 60)

    def _try_mode_ctypes(self):
        """Mode 1: Try to load existing OpenFHE DLLs via ctypes"""
        print("\n[Mode 1] Attempting ctypes DLL loading...")
        try:
            if sys.platform == 'win32':
                lib_extension = '.dll'
                lib_prefix = 'lib'
                lib_paths = [
                    os.path.join(self.openfhe_path, "lib"),
                    os.path.join(self.openfhe_path, "bin"),
                    os.path.join(self.build_path, "lib", "Release"),
                    os.path.join(self.build_path, "lib", "Debug"),
                ]
            else:
                lib_extension = '.so'
                lib_prefix = 'lib'
                lib_paths = [
                    os.path.join(self.openfhe_path, "lib"),
                    os.path.join(self.openfhe_path, "lib64"),
                    os.path.join(self.build_path, "lib"),
                    '/usr/local/lib', '/usr/lib', '/usr/lib64'
                ]
                for lib_path in lib_paths:
                    if os.path.exists(lib_path):
                        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
                        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld}"

            if sys.platform == 'win32':
                for lib_path in lib_paths:
                    if os.path.exists(lib_path):
                        os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                        if hasattr(os, 'add_dll_directory'):
                            try:
                                os.add_dll_directory(lib_path)
                            except:
                                pass

            dependency_order = [
                f"{lib_prefix}OPENFHEcore{lib_extension}",
                f"{lib_prefix}OPENFHEpke{lib_extension}",
                f"{lib_prefix}OPENFHEbinfhe{lib_extension}"
            ]

            lib_dir = None
            for path in lib_paths:
                if os.path.exists(path):
                    lib_dir = path
                    break

            if not lib_dir:
                print("  ❌ OpenFHE library directory not found")
                return False

            print(f"  📂 Checking: {lib_dir}")

            for dep in dependency_order:
                dep_path = os.path.join(lib_dir, dep)
                if os.path.exists(dep_path):
                    try:
                        if sys.platform == 'win32':
                            loaded_lib = ctypes.WinDLL(dep_path)
                        else:
                            loaded_lib = ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                        print(f"  ✅ Loaded: {dep}")
                        if "core" in dep.lower():
                            self.lib = loaded_lib
                            return True
                    except Exception as e:
                        print(f"  ❌ Failed to load {dep}: {str(e)}")
                        return False
                else:
                    print(f"  ❌ Not found: {dep}")
            return False
        except Exception as e:
            print(f"  ❌ Error in ctypes mode: {str(e)}")
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

            compiler = self._find_compiler()
            if not compiler or compiler == 'cl.exe':
                if not self._check_msvc():
                    print("  ❌ MSVC compiler required for DLL compilation")
                    return False
                compiler = 'cl.exe'
            else:
                print("  ⚠️ DLL compilation works best with MSVC")

            print(f"  ✅ Found compiler: {compiler}")

            if self._compile_dll_wrapper(cpp_file, dll_file, compiler):
                self.custom_dll = dll_file
                print(f"  ✅ Compiled DLL: {dll_file}")
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
            return False

    def _try_mode_subprocess(self):
        """Mode 3: Try to compile C++ executable wrapper"""
        print("\n[Mode 3] Attempting C++ subprocess wrapper...")
        try:
            if not os.path.exists(self.openfhe_path) and not os.path.exists(self.build_path):
                print("  ❌ OpenFHE installation not found")
                return False

            cpp_file = os.path.join(self.temp_dir, "openfhe_wrapper.cpp")
            exe_file = os.path.join(self.temp_dir,
                                    "openfhe_wrapper.exe" if sys.platform == "win32" else "openfhe_wrapper")

            print(f"  📝 Generating C++ code: {cpp_file}")
            with open(cpp_file, 'w') as f:
                f.write(self._generate_cpp_code_enhanced())

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

    def _check_msvc(self):
        """Check if MSVC compiler is available"""
        try:
            result = subprocess.run(['cl.exe'], capture_output=True, timeout=2)
            return True
        except:
            return False

    def _find_compiler(self):
        """Find available C++ compiler"""
        compilers = ['g++', 'clang++', 'cl.exe']
        for compiler in compilers:
            try:
                result = subprocess.run([compiler, '--version'], capture_output=True, timeout=5)
                if result.returncode == 0 or compiler == 'cl.exe':
                    return compiler
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    # def _generate_cpp_code_enhanced(self):
    #     """Enhanced C++ subprocess code with all operations"""
    #     return '''#include <iostream>
    #     #include <fstream>
    #     #include <vector>
    #     #include <string>
    #     #include <sstream>
    #     #include <algorithm>
    #     #include <cmath>
    #     #include <limits>
    #     #include <cstdint>
    #
    #     #ifndef M_E
    #     #define M_E 2.71828182845904523536
    #     #endif
    #
    #     #ifndef M_PI
    #     #define M_PI 3.14159265358979323846
    #     #endif
    #
    #     #if __has_include("openfhe.h")
    #         #define OPENFHE_AVAILABLE
    #         // Disable warnings for external headers
    #         #ifdef _MSC_VER
    #         #pragma warning(push, 0)
    #         #endif
    #
    #         #include "openfhe.h"
    #
    #         // Try to include serialization headers (may not exist in all installations)
    #         #if __has_include("ciphertext-ser.h")
    #             #include "ciphertext-ser.h"
    #             #include "cryptocontext-ser.h"
    #             #include "key/key-ser.h"
    #             #include "scheme/ckksrns/ckksrns-ser.h"
    #             #include "scheme/bfvrns/bfvrns-ser.h"
    #             #define SERIALIZATION_AVAILABLE
    #         #endif
    #
    #         #ifdef _MSC_VER
    #         #pragma warning(pop)
    #         #endif
    #
    #         using namespace lbcrypto;
    #     #endif
    #
    #     class FHEProcessor {
    #     public:
    #         CryptoContext<DCRTPoly> cryptoContext;
    #         KeyPair<DCRTPoly> keyPair;
    #
    #         bool setupCKKS(uint32_t polyDegree, uint32_t scalingModSize) {
    #             try {
    #                 CCParams<CryptoContextCKKSRNS> parameters;
    #                 parameters.SetMultiplicativeDepth(10);
    #                 parameters.SetScalingModSize(scalingModSize);
    #                 parameters.SetRingDim(polyDegree);
    #                 parameters.SetBatchSize(polyDegree / 2);
    #                 parameters.SetSecurityLevel(HEStd_128_classic);
    #
    #                 cryptoContext = GenCryptoContext(parameters);
    #                 cryptoContext->Enable(PKE);
    #                 cryptoContext->Enable(KEYSWITCH);
    #                 cryptoContext->Enable(LEVELEDSHE);
    #
    #                 keyPair = cryptoContext->KeyGen();
    #                 cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    #                 cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    #
    #                 return true;
    #             } catch (const std::exception& e) {
    #                 std::cerr << "Error: " << e.what() << std::endl;
    #                 return false;
    #             }
    #         }
    #
    #         bool setupBFV(uint32_t polyDegree, uint32_t plainModulus) {
    #             try {
    #                 CCParams<CryptoContextBFVRNS> parameters;
    #                 parameters.SetPlaintextModulus(plainModulus);
    #                 parameters.SetMultiplicativeDepth(2);
    #                 parameters.SetRingDim(polyDegree);
    #                 parameters.SetSecurityLevel(HEStd_128_classic);
    #
    #                 cryptoContext = GenCryptoContext(parameters);
    #                 cryptoContext->Enable(PKE);
    #                 cryptoContext->Enable(KEYSWITCH);
    #                 cryptoContext->Enable(LEVELEDSHE);
    #
    #                 keyPair = cryptoContext->KeyGen();
    #                 cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    #                 cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    #
    #                 return true;
    #             } catch (const std::exception& e) {
    #                 std::cerr << "Error: " << e.what() << std::endl;
    #                 return false;
    #             }
    #         }
    #
    #         void encryptData(const std::string& inputFile, const std::string& outputFile, const std::string& scheme) {
    #             #ifdef SERIALIZATION_AVAILABLE
    #             try {
    #                 std::ifstream inFile(inputFile);
    #                 std::ofstream outFile(outputFile, std::ios::binary);
    #
    #                 if (!inFile || !outFile) {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"File error\\"}" << std::endl;
    #                     return;
    #                 }
    #
    #                 int count = 0;
    #                 std::string line;
    #
    #                 while (std::getline(inFile, line)) {
    #                     if (line.empty()) continue;
    #
    #                     size_t colonPos = line.find(':');
    #                     std::string dataType = "numeric";
    #                     std::string valueStr = line;
    #
    #                     if (colonPos != std::string::npos) {
    #                         dataType = line.substr(0, colonPos);
    #                         valueStr = line.substr(colonPos + 1);
    #                     }
    #
    #                     double numericValue = 0.0;
    #
    #                     if (dataType == "text" || dataType == "string") {
    #                         for (char c : valueStr) {
    #                             numericValue += static_cast<double>(static_cast<unsigned char>(c));
    #                         }
    #                     } else if (dataType == "date") {
    #                         try {
    #                             numericValue = std::stod(valueStr);
    #                         } catch (...) {
    #                             numericValue = std::hash<std::string>{}(valueStr) % 1000000000;
    #                         }
    #                     } else {
    #                         try {
    #                             numericValue = std::stod(valueStr);
    #                         } catch (...) {
    #                             std::cerr << "Warning: Could not parse value: " << valueStr << std::endl;
    #                             continue;
    #                         }
    #                     }
    #
    #                     if (scheme == "CKKS") {
    #                         std::vector<double> data = {numericValue};
    #                         Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(data);
    #                         auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);
    #
    #                         try {
    #                             Serial::Serialize(ciphertext, outFile, SerType::BINARY);
    #                         } catch (const std::exception& e) {
    #                             std::cerr << "Serialization error: " << e.what() << std::endl;
    #                             continue;
    #                         }
    #                     } else if (scheme == "BFV") {
    #                         std::vector<int64_t> data = {static_cast<int64_t>(numericValue)};
    #                         Plaintext plaintext = cryptoContext->MakePackedPlaintext(data);
    #                         auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);
    #
    #                         try {
    #                             Serial::Serialize(ciphertext, outFile, SerType::BINARY);
    #                         } catch (const std::exception& e) {
    #                             std::cerr << "Serialization error: " << e.what() << std::endl;
    #                             continue;
    #                         }
    #                     }
    #                     count++;
    #                 }
    #
    #                 std::cout << "{\\"status\\": \\"success\\", \\"encrypted_count\\": " << count << "}" << std::endl;
    #             } catch (const std::exception& e) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
    #             }
    #             #else
    #             std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
    #             #endif
    #         }
    #
    #         void aggregateEncrypted(const std::string& encFile, const std::string& operation) {
    #             #ifdef SERIALIZATION_AVAILABLE
    #             try {
    #                 std::ifstream inFile(encFile, std::ios::binary);
    #                 if (!inFile) {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
    #                     return;
    #                 }
    #
    #                 std::vector<Ciphertext<DCRTPoly>> ciphertexts;
    #                 while (inFile.peek() != EOF) {
    #                     Ciphertext<DCRTPoly> ct;
    #                     try {
    #                         Serial::Deserialize(ct, inFile, SerType::BINARY);
    #                         ciphertexts.push_back(ct);
    #                     } catch (const std::exception& e) {
    #                         std::cerr << "Deserialization error: " << e.what() << std::endl;
    #                         break;
    #                     }
    #                 }
    #
    #                 if (ciphertexts.empty()) {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No ciphertexts\\"}" << std::endl;
    #                     return;
    #                 }
    #
    #                 Ciphertext<DCRTPoly> result;
    #
    #                 if (operation == "sum" || operation == "add") {
    #                     result = cryptoContext->EvalAddMany(ciphertexts);
    #                 } else if (operation == "multiply") {
    #                     result = ciphertexts[0];
    #                     for (size_t i = 1; i < ciphertexts.size(); i++) {
    #                         result = cryptoContext->EvalMult(result, ciphertexts[i]);
    #                     }
    #                 } else if (operation == "average" || operation == "avg") {
    #                     result = cryptoContext->EvalAddMany(ciphertexts);
    #                     double factor = 1.0 / ciphertexts.size();
    #                     result = cryptoContext->EvalMult(result, factor);
    #                 } else {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unsupported operation\\"}" << std::endl;
    #                     return;
    #                 }
    #
    #                 std::ostringstream oss;
    #                 Serial::Serialize(result, oss, SerType::BINARY);
    #                 std::string serialized = oss.str();
    #
    #                 std::ostringstream hexStream;
    #                 hexStream << std::hex << std::setfill('0');
    #                 for (unsigned char c : serialized) {
    #                     hexStream << std::setw(2) << static_cast<int>(c);
    #                 }
    #
    #                 std::cout << "{\\"status\\": \\"success\\", \\"operation\\": \\"" << operation
    #                           << "\\", \\"result_hex\\": \\"" << hexStream.str()
    #                           << "\\", \\"count\\": " << ciphertexts.size() << "}" << std::endl;
    #             } catch (const std::exception& e) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
    #             }
    #             #else
    #             std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
    #             #endif
    #         }
    #
    #         void decryptData(const std::string& encFile) {
    #             #ifdef SERIALIZATION_AVAILABLE
    #             try {
    #                 std::ifstream inFile(encFile, std::ios::binary);
    #                 if (!inFile) {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
    #                     return;
    #                 }
    #
    #                 Ciphertext<DCRTPoly> ciphertext;
    #                 Serial::Deserialize(ciphertext, inFile, SerType::BINARY);
    #
    #                 Plaintext plaintext;
    #                 cryptoContext->Decrypt(keyPair.secretKey, ciphertext, &plaintext);
    #
    #                 plaintext->SetLength(1);
    #                 std::vector<std::complex<double>> vals = plaintext->GetCKKSPackedValue();
    #
    #                 if (!vals.empty()) {
    #                     double result = vals[0].real();
    #                     std::cout << "{\\"status\\": \\"success\\", \\"result\\": " << result << "}" << std::endl;
    #                 } else {
    #                     std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No values\\"}" << std::endl;
    #                 }
    #             } catch (const std::exception& e) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
    #             }
    #             #else
    #             std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Serialization not available\\"}" << std::endl;
    #             #endif
    #         }
    #     };
    #
    #     int main(int argc, char* argv[]) {
    #         if (argc < 2) {
    #             std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No command\\"}" << std::endl;
    #             return 1;
    #         }
    #
    #         std::string command = argv[1];
    #         FHEProcessor processor;
    #
    #         if (command == "test") {
    #             #ifdef SERIALIZATION_AVAILABLE
    #             std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"OpenFHE ready with serialization\\"}" << std::endl;
    #             #else
    #             std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"OpenFHE ready without serialization\\"}" << std::endl;
    #             #endif
    #         }
    #         else if (command == "setup_ckks") {
    #             uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 32768;
    #             uint32_t scalingModSize = (argc > 3) ? std::stoi(argv[3]) : 50;
    #             if (processor.setupCKKS(polyDegree, scalingModSize)) {
    #                 std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"CKKS\\"}" << std::endl;
    #             } else {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Setup failed\\"}" << std::endl;
    #             }
    #         }
    #         else if (command == "setup_bfv") {
    #             uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 8192;
    #             uint32_t plainModulus = (argc > 3) ? std::stoi(argv[3]) : 65537;
    #             if (processor.setupBFV(polyDegree, plainModulus)) {
    #                 std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"BFV\\"}" << std::endl;
    #             } else {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Setup failed\\"}" << std::endl;
    #             }
    #         }
    #         else if (command == "encrypt") {
    #             if (argc < 5) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing arguments\\"}" << std::endl;
    #                 return 1;
    #             }
    #             std::string scheme = argv[4];
    #             uint32_t polyDegree = 32768;
    #             if (scheme == "CKKS") {
    #                 processor.setupCKKS(polyDegree, 50);
    #             } else {
    #                 processor.setupBFV(polyDegree, 65537);
    #             }
    #             processor.encryptData(argv[2], argv[3], scheme);
    #         }
    #         else if (command == "aggregate") {
    #             if (argc < 4) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing arguments\\"}" << std::endl;
    #                 return 1;
    #             }
    #             processor.setupCKKS(8192, 50);
    #             processor.aggregateEncrypted(argv[2], argv[3]);
    #         }
    #         else if (command == "decrypt") {
    #             if (argc < 3) {
    #                 std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing file\\"}" << std::endl;
    #                 return 1;
    #             }
    #             processor.setupCKKS(32768, 50);
    #             processor.decryptData(argv[2]);
    #         }
    #         else {
    #             std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unknown command\\"}" << std::endl;
    #         }
    #         return 0;
    #     }'''

    def _generate_cpp_code_enhanced(self):
        """Generate simplified C++ code without serialization dependency"""
        return '''
    #include <iostream>
    #include <fstream>
    #include <vector>
    #include <string>
    #include <sstream>
    #include <cmath>
    #include <cstdint>
    #include <iomanip>

    #ifndef M_E
    #define M_E 2.71828182845904523536
    #endif

    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // Simple in-memory encryption without serialization
    class SimpleFHEProcessor {
    private:
        struct SimpleCiphertext {
            std::vector<double> data;
            bool isEncrypted;
        };

        std::vector<SimpleCiphertext> storage;
        bool initialized;

    public:
        SimpleFHEProcessor() : initialized(false) {}

        bool setupCKKS(uint32_t polyDegree, uint32_t scalingModSize) {
            initialized = true;
            std::cerr << "Initialized simple CKKS (no OpenFHE)" << std::endl;
            return true;
        }

        bool setupBFV(uint32_t polyDegree, uint32_t plainModulus) {
            initialized = true;
            std::cerr << "Initialized simple BFV (no OpenFHE)" << std::endl;
            return true;
        }

        void encryptData(const std::string& inputFile, const std::string& outputFile, const std::string& scheme) {
            try {
                std::ifstream inFile(inputFile);
                std::ofstream outFile(outputFile);

                if (!inFile || !outFile) {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"File error\\"}" << std::endl;
                    return;
                }

                int count = 0;
                std::string line;

                while (std::getline(inFile, line)) {
                    if (line.empty()) continue;

                    size_t colonPos = line.find(':');
                    std::string valueStr = line;

                    if (colonPos != std::string::npos) {
                        valueStr = line.substr(colonPos + 1);
                    }

                    double numericValue = 0.0;
                    try {
                        numericValue = std::stod(valueStr);
                    } catch (...) {
                        // For text, sum ASCII values
                        for (char c : valueStr) {
                            numericValue += static_cast<double>(static_cast<unsigned char>(c));
                        }
                    }

                    // Simple "encryption": just write as hex with marker
                    outFile << std::hex << std::setfill('0') << std::setw(16);
                    uint64_t enc = *reinterpret_cast<uint64_t*>(&numericValue);
                    outFile << enc << std::endl;
                    count++;
                }

                std::cout << "{\\"status\\": \\"success\\", \\"encrypted_count\\": " << count << "}" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
            }
        }

        void aggregateEncrypted(const std::string& encFile, const std::string& operation) {
            try {
                std::ifstream inFile(encFile);
                if (!inFile) {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
                    return;
                }

                std::vector<double> values;
                std::string line;

                while (std::getline(inFile, line)) {
                    if (line.empty()) continue;

                    uint64_t enc;
                    std::istringstream iss(line);
                    iss >> std::hex >> enc;

                    double value = *reinterpret_cast<double*>(&enc);
                    values.push_back(value);
                }

                if (values.empty()) {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No values\\"}" << std::endl;
                    return;
                }

                double result = 0.0;

                if (operation == "sum" || operation == "add") {
                    for (double v : values) result += v;
                } else if (operation == "average" || operation == "avg") {
                    for (double v : values) result += v;
                    result /= values.size();
                } else if (operation == "multiply") {
                    result = 1.0;
                    for (double v : values) result *= v;
                } else {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unsupported operation\\"}" << std::endl;
                    return;
                }

                // Return result as hex
                uint64_t enc_result = *reinterpret_cast<uint64_t*>(&result);
                std::ostringstream hexStream;
                hexStream << std::hex << std::setfill('0') << std::setw(16) << enc_result;

                std::cout << "{\\"status\\": \\"success\\", \\"operation\\": \\"" << operation 
                          << "\\", \\"result_hex\\": \\"" << hexStream.str() 
                          << "\\", \\"count\\": " << values.size() << "}" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
            }
        }

        void decryptData(const std::string& encFile) {
            try {
                std::ifstream inFile(encFile);
                if (!inFile) {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Cannot open file\\"}" << std::endl;
                    return;
                }

                std::string line;
                if (std::getline(inFile, line)) {
                    uint64_t enc;
                    std::istringstream iss(line);
                    iss >> std::hex >> enc;

                    double result = *reinterpret_cast<double*>(&enc);
                    std::cout << "{\\"status\\": \\"success\\", \\"result\\": " << result << "}" << std::endl;
                } else {
                    std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No data\\"}" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
            }
        }
    };

    int main(int argc, char* argv[]) {
        if (argc < 2) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No command\\"}" << std::endl;
            return 1;
        }

        std::string command = argv[1];
        SimpleFHEProcessor processor;

        if (command == "test") {
            std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"Simple FHE ready\\"}" << std::endl;
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
            processor.setupCKKS(32768, 50);
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
    }
    '''

    def _generate_dll_cpp_code(self):
            """REAL FHE C++ DLL code with actual OpenFHE operations"""
            return '''// OpenFHE Wrapper DLL - REAL FHE Implementation
        #include <windows.h>
        #include <string>
        #include <vector>
        #include <sstream>
        #include <fstream>
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
    
        #ifdef BUILD_DLL
            #define DLL_EXPORT __declspec(dllexport)
        #else
            #define DLL_EXPORT __declspec(dllimport)
        #endif
    
        extern "C" {
        #ifdef OPENFHE_AVAILABLE
            static CryptoContext<DCRTPoly> g_context = nullptr;
            static KeyPair<DCRTPoly> g_keyPair;
        #endif
    
            DLL_EXPORT int InitializeCKKS(int polyDegree, int scalingModSize) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    CCParams<CryptoContextCKKSRNS> parameters;
                    parameters.SetMultiplicativeDepth(10);
                    parameters.SetScalingModSize(scalingModSize);
                    parameters.SetRingDim(polyDegree);
                    parameters.SetBatchSize(polyDegree / 2);
    
                    g_context = GenCryptoContext(parameters);
                    g_context->Enable(PKE);
                    g_context->Enable(KEYSWITCH);
                    g_context->Enable(LEVELEDSHE);
    
                    g_keyPair = g_context->KeyGen();
                    g_context->EvalMultKeyGen(g_keyPair.secretKey);
                    g_context->EvalSumKeyGen(g_keyPair.secretKey);
    
                    return 1;
                } catch (const std::exception& e) {
                    return 0;
                }
                #else
                return -1;
                #endif
            }
    
            DLL_EXPORT int InitializeBFV(int polyDegree, int plainModulus) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    CCParams<CryptoContextBFVRNS> parameters;
                    parameters.SetPlaintextModulus(plainModulus);
                    parameters.SetMultiplicativeDepth(2);
                    parameters.SetRingDim(polyDegree);
    
                    g_context = GenCryptoContext(parameters);
                    g_context->Enable(PKE);
                    g_context->Enable(KEYSWITCH);
                    g_context->Enable(LEVELEDSHE);
    
                    g_keyPair = g_context->KeyGen();
                    g_context->EvalMultKeyGen(g_keyPair.secretKey);
                    g_context->EvalSumKeyGen(g_keyPair.secretKey);
    
                    return 1;
                } catch (const std::exception& e) {
                    return 0;
                }
                #else
                return -1;
                #endif
            }
    
            DLL_EXPORT const char* EncryptValue(double value, char* outputBuffer, int bufferSize) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    if (!g_context) return "ERROR: Context not initialized";
    
                    std::vector<double> data = {value};
                    Plaintext plaintext = g_context->MakeCKKSPackedPlaintext(data);
                    auto ciphertext = g_context->Encrypt(g_keyPair.publicKey, plaintext);
    
                    std::ostringstream oss;
                    Serial::Serialize(ciphertext, oss, SerType::BINARY);
                    std::string serialized = oss.str();
    
                    if (serialized.size() < bufferSize) {
                        memcpy(outputBuffer, serialized.c_str(), serialized.size() + 1);
                        return outputBuffer;
                    }
                    return "ERROR: Buffer too small";
                } catch (const std::exception& e) {
                    return "ERROR: Encryption failed";
                }
                #else
                return "ERROR: OpenFHE not available";
                #endif
            }
    
            DLL_EXPORT double DecryptValue(const char* ciphertextData, int dataLength) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    if (!g_context) return -999999.0;
    
                    std::string serialized(ciphertextData, dataLength);
                    std::istringstream iss(serialized);
    
                    Ciphertext<DCRTPoly> ciphertext;
                    Serial::Deserialize(ciphertext, iss, SerType::BINARY);
    
                    Plaintext plaintext;
                    g_context->Decrypt(g_keyPair.secretKey, ciphertext, &plaintext);
    
                    plaintext->SetLength(1);
                    return plaintext->GetCKKSPackedValue()[0].real();
                } catch (const std::exception& e) {
                    return -999999.0;
                }
                #else
                return -999999.0;
                #endif
            }
    
            DLL_EXPORT const char* AddEncrypted(const char* ct1Data, int ct1Len, 
                                                const char* ct2Data, int ct2Len,
                                                char* outputBuffer, int bufferSize) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    if (!g_context) return "ERROR: Context not initialized";
    
                    std::string s1(ct1Data, ct1Len);
                    std::string s2(ct2Data, ct2Len);
                    std::istringstream iss1(s1), iss2(s2);
    
                    Ciphertext<DCRTPoly> ct1, ct2;
                    Serial::Deserialize(ct1, iss1, SerType::BINARY);
                    Serial::Deserialize(ct2, iss2, SerType::BINARY);
    
                    auto result = g_context->EvalAdd(ct1, ct2);
    
                    std::ostringstream oss;
                    Serial::Serialize(result, oss, SerType::BINARY);
                    std::string serialized = oss.str();
    
                    if (serialized.size() < bufferSize) {
                        memcpy(outputBuffer, serialized.c_str(), serialized.size() + 1);
                        return outputBuffer;
                    }
                    return "ERROR: Buffer too small";
                } catch (const std::exception& e) {
                    return "ERROR: Addition failed";
                }
                #else
                return "ERROR: OpenFHE not available";
                #endif
            }
    
            DLL_EXPORT const char* MultiplyEncrypted(const char* ct1Data, int ct1Len, 
                                                      const char* ct2Data, int ct2Len,
                                                      char* outputBuffer, int bufferSize) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    if (!g_context) return "ERROR: Context not initialized";
    
                    std::string s1(ct1Data, ct1Len);
                    std::string s2(ct2Data, ct2Len);
                    std::istringstream iss1(s1), iss2(s2);
    
                    Ciphertext<DCRTPoly> ct1, ct2;
                    Serial::Deserialize(ct1, iss1, SerType::BINARY);
                    Serial::Deserialize(ct2, iss2, SerType::BINARY);
    
                    auto result = g_context->EvalMult(ct1, ct2);
    
                    std::ostringstream oss;
                    Serial::Serialize(result, oss, SerType::BINARY);
                    std::string serialized = oss.str();
    
                    if (serialized.size() < bufferSize) {
                        memcpy(outputBuffer, serialized.c_str(), serialized.size() + 1);
                        return outputBuffer;
                    }
                    return "ERROR: Buffer too small";
                } catch (const std::exception& e) {
                    return "ERROR: Multiplication failed";
                }
                #else
                return "ERROR: OpenFHE not available";
                #endif
            }
    
            DLL_EXPORT const char* SumEncrypted(const char** ctDataArray, int* ctLenArray, 
                                                 int count, char* outputBuffer, int bufferSize) {
                #ifdef OPENFHE_AVAILABLE
                try {
                    if (!g_context || count == 0) return "ERROR: Invalid input";
    
                    std::vector<Ciphertext<DCRTPoly>> ciphertexts;
                    for (int i = 0; i < count; i++) {
                        std::string s(ctDataArray[i], ctLenArray[i]);
                        std::istringstream iss(s);
                        Ciphertext<DCRTPoly> ct;
                        Serial::Deserialize(ct, iss, SerType::BINARY);
                        ciphertexts.push_back(ct);
                    }
    
                    auto result = g_context->EvalAddMany(ciphertexts);
    
                    std::ostringstream oss;
                    Serial::Serialize(result, oss, SerType::BINARY);
                    std::string serialized = oss.str();
    
                    if (serialized.size() < bufferSize) {
                        memcpy(outputBuffer, serialized.c_str(), serialized.size() + 1);
                        return outputBuffer;
                    }
                    return "ERROR: Buffer too small";
                } catch (const std::exception& e) {
                    return "ERROR: Sum failed";
                }
                #else
                return "ERROR: OpenFHE not available";
                #endif
            }
    
            DLL_EXPORT void Cleanup() {
                #ifdef OPENFHE_AVAILABLE
                g_context = nullptr;
                g_keyPair = KeyPair<DCRTPoly>();
                #endif
            }
    
            DLL_EXPORT int TestDLL() {
                return 42;
            }
        }
    
        BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
            switch (fdwReason) {
                case DLL_PROCESS_DETACH:
                    Cleanup();
                    break;
            }
            return TRUE;
        }
    '''

    # def _compile_dll_wrapper(self, cpp_file, dll_file, compiler):
    #     """Compile the C++ wrapper as DLL"""
    #     try:
    #         include_paths = []
    #         lib_paths = []
    #
    #         # Platform-specific include and library paths
    #         include_subdirs = ['include', 'src/pke/include', 'src/core/include', 'src/binfhe/include']
    #         lib_subdirs = ['lib', 'lib/Release', 'lib/Debug']
    #
    #         # Find include directories
    #         for base in [self.openfhe_path, self.build_path]:
    #             if os.path.exists(base):
    #                 for subdir in include_subdirs:
    #                     inc_path = os.path.join(base, subdir)
    #                     if os.path.exists(inc_path):
    #                         include_paths.append(inc_path)
    #
    #         # Find library directories
    #         for base in [self.openfhe_path, self.build_path]:
    #             if os.path.exists(base):
    #                 for subdir in lib_subdirs:
    #                     lib_path = os.path.join(base, subdir)
    #                     if os.path.exists(lib_path):
    #                         lib_paths.append(lib_path)
    #
    #         if compiler == 'cl.exe':
    #             # MSVC DLL compilation
    #             cmd = [
    #                 'cl.exe',
    #                 '/LD',  # Create DLL
    #                 '/EHsc',
    #                 '/std:c++17',
    #                 '/DBUILD_DLL',
    #                 cpp_file,
    #                 f'/Fe:{dll_file}'
    #             ]
    #
    #             for inc in include_paths:
    #                 cmd.append(f'/I{inc}')
    #
    #             cmd.append('/link')
    #             for lib in lib_paths:
    #                 cmd.append(f'/LIBPATH:{lib}')
    #
    #             cmd.extend(['OPENFHEpke.lib', 'OPENFHEcore.lib', 'OPENFHEbinfhe.lib'])
    #
    #         elif compiler in ['g++', 'clang++']:
    #             # MinGW/GCC DLL compilation
    #             cmd = [
    #                 compiler,
    #                 '-shared',  # Create DLL
    #                 '-std=c++17',
    #                 '-DBUILD_DLL',
    #                 cpp_file,
    #                 '-o', dll_file
    #             ]
    #
    #             for inc in include_paths:
    #                 cmd.append(f'-I{inc}')
    #
    #             for lib in lib_paths:
    #                 cmd.append(f'-L{lib}')
    #
    #             cmd.extend(['-lOPENFHEpke', '-lOPENFHEcore', '-lOPENFHEbinfhe'])
    #             cmd.extend(['-Wl,--out-implib,' + dll_file.replace('.dll', '.lib')])
    #
    #         print(f"  🔨 Compiling DLL: {' '.join(cmd[:4])}...")
    #         result = subprocess.run(cmd, capture_output=True, timeout=120)
    #
    #         if result.returncode == 0 and os.path.exists(dll_file):
    #             print(f"  ✅ DLL compiled successfully")
    #             return True
    #         else:
    #             if result.stderr:
    #                 error_msg = result.stderr.decode()[:500]
    #                 print(f"  ⚠️ Compilation errors: {error_msg}")
    #             return False
    #
    #     except subprocess.TimeoutExpired:
    #         print("  ❌ DLL compilation timeout")
    #         return False
    #     except Exception as e:
    #         print(f"  ❌ DLL compilation error: {str(e)}")
    #         return False

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
                'include/openfhe/cereal'
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
            dll = ctypes.WinDLL(self.custom_dll)
            test_func = dll.TestDLL
            test_func.restype = ctypes.c_int
            result = test_func()
            if result == 42:
                print(f"  ✅ DLL test passed (returned {result})")
                return True
            else:
                print(f"  ❌ DLL test failed (returned {result})")
                return False
        except Exception as e:
            print(f"  ❌ DLL test error: {str(e)}")
            return False

    # def _compile_wrapper(self, cpp_file, exe_file, compiler):
    #     """Compile the C++ wrapper as executable"""
    #     try:
    #         include_paths = []
    #         lib_paths = []
    #
    #         include_subdirs = [
    #             'include',
    #             'include/openfhe',
    #             'include/openfhe/pke',
    #             'include/openfhe/core',
    #             'include/openfhe/binfhe',
    #             'include/openfhe/core/lattice',
    #             'include/openfhe/core/math',
    #             'include/openfhe/core/utils',
    #             'include/openfhe/cereal',
    #             # CRITICAL: Add source directories for serialization headers
    #             'src/pke/include',
    #             'src/core/include',
    #             'src/binfhe/include',
    #             'src/pke/include/schemebase',
    #             'src/pke/include/encoding',
    #             'src/pke/include/key',
    #             'src/pke/include/keyswitch',
    #             'src/pke/include/scheme',
    #             'src/pke/include/scheme/ckksrns',
    #             'src/pke/include/scheme/bfvrns',
    #             'src/pke/include/scheme/bgvrns'
    #         ]
    #
    #         if sys.platform == 'win32':
    #             lib_subdirs = ['lib', 'lib/Release', 'lib/Debug', 'bin']
    #         else:
    #             lib_subdirs = ['lib', 'lib64']
    #
    #         for base in [self.openfhe_path, self.build_path]:
    #             if os.path.exists(base):
    #                 for subdir in include_subdirs:
    #                     inc_path = os.path.join(base, subdir)
    #                     if os.path.exists(inc_path):
    #                         include_paths.append(inc_path)
    #
    #         for base in [self.openfhe_path, self.build_path]:
    #             if os.path.exists(base):
    #                 for subdir in lib_subdirs:
    #                     lib_path = os.path.join(base, subdir)
    #                     if os.path.exists(lib_path):
    #                         lib_paths.append(lib_path)
    #
    #         if compiler in ['g++', 'clang++']:
    #             cmd = [compiler, '-std=c++17', cpp_file, '-o', exe_file]
    #             for inc in include_paths:
    #                 cmd.append(f'-I{inc}')
    #             for lib in lib_paths:
    #                 cmd.append(f'-L{lib}')
    #             cmd.extend(['-lOPENFHEpke', '-lOPENFHEcore', '-lOPENFHEbinfhe'])
    #             if sys.platform != 'win32':
    #                 cmd.extend(['-pthread', '-Wl,-rpath,' + ':'.join(lib_paths)])
    #         else:
    #             cmd = ['cl.exe', '/EHsc', '/std:c++17', cpp_file, '/Fe:' + exe_file]
    #             for inc in include_paths:
    #                 cmd.append(f'/I{inc}')
    #             cmd.append('/link')
    #             for lib in lib_paths:
    #                 cmd.append(f'/LIBPATH:{lib}')
    #             cmd.extend(['OPENFHEpke.lib', 'OPENFHEcore.lib', 'OPENFHEbinfhe.lib'])
    #
    #         print(f"  🔨 Compiling: {' '.join(cmd[:3])}...")
    #         result = subprocess.run(cmd, capture_output=True, timeout=300)
    #
    #         if result.returncode == 0 and os.path.exists(exe_file):
    #             if sys.platform != 'win32':
    #                 os.chmod(exe_file, 0o755)
    #             return True
    #         else:
    #             if result.stderr:
    #                 error_msg = result.stderr.decode()[:500]
    #                 print(f"  ⚠️ Compilation errors: {error_msg}")
    #             return False
    #     except subprocess.TimeoutExpired:
    #         print("  ❌ Compilation timeout")
    #         return False
    #     except Exception as e:
    #         print(f"  ❌ Compilation error: {str(e)}")
    #         return False

    def _compile_wrapper(self, cpp_file, exe_file, compiler):
        """Simplified compilation - no OpenFHE dependencies"""
        try:
            if compiler in ['g++', 'clang++']:
                cmd = [compiler, '-std=c++17', cpp_file, '-o', exe_file]
            else:  # cl.exe
                cmd = ['cl.exe', '/EHsc', '/std:c++17', cpp_file, '/Fe:' + exe_file]

            print(f"  🔨 Compiling standalone executable...")
            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(exe_file):
                if sys.platform != 'win32':
                    os.chmod(exe_file, 0o755)
                print(f"  ✅ Compilation successful")
                return True
            else:
                if result.stderr:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')[:500]
                    print(f"  ⚠️ Errors: {error_msg}")
                return False

        except subprocess.TimeoutExpired:
            print("  ❌ Compilation timeout")
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

        if self.mode == 'custom_dll':
            result = self._call_custom_dll("initialize_ckks", ring_dim, scale_mod_size)
            if result.get('status') == 'success':
                self.context = {
                    'scheme': scheme, 'params': self.params, 'initialized': True,
                    'mode': 'custom_dll', 'timestamp': datetime.now().isoformat()
                }
                self._generate_keys()
                print(f"✅ Context generated via custom DLL for {scheme}")
                return self.context
        elif self.mode == 'subprocess':
            result = self._call_cpp_subprocess(f"setup_{scheme.lower()}", ring_dim, scale_mod_size)
            if result.get('status') == 'success':
                self.context = {
                    'scheme': scheme, 'params': self.params, 'initialized': True,
                    'mode': 'subprocess', 'timestamp': datetime.now().isoformat()
                }
                self._generate_keys()
                print(f"✅ Context generated via C++ subprocess for {scheme}")
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
        """Encrypt data"""
        if not self.context:
            raise ValueError("Context not initialized")

        encrypted_results = []
        if self.mode == 'custom_dll':
            for value in data:
                if not pd.isna(value):
                    result = self._call_custom_dll("encrypt", float(value))
                    if result.get('status') == 'success':
                        encrypted_results.append({
                            'value': result.get('result'),
                            'encrypted': True,
                            'mode': 'custom_dll'
                        })
                    else:
                        encrypted_results.append(None)
                else:
                    encrypted_results.append(None)
            return encrypted_results

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
                    # Store the result along with metadata
                    if result.get('status') == 'success':
                        with open(enc_file, 'rb') as f:
                            encrypted_data = f.read()
                            print(f"   Subprocess Encrypt result: {encrypted_data}")
                            encrypted_results.append(encrypted_data)
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
                    encrypted_results.append({'input_value': value, 'error': str(e)})
                    print(f"   ℹ️ Using simulation mode")
                    return self._simulate_encrypt(data, data_type)
            return encrypted_results

        # Fallback: simulation or empty list
        if hasattr(self, '_simulate_encrypt'):
            return self._simulate_encrypt(data, data_type)
        return encrypted_results

    def _simulate_encrypt(self, data, data_type):
        """Simulate encryption"""
        encrypted_values = []
        for value in data:
            try:
                if pd.isna(value):
                    encrypted_values.append(None)
                    continue
                if data_type == 'numeric':
                    encrypted_val = {
                        'ciphertext': f"ENC_{hash(str(value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'scheme': self.scheme,
                        'type': 'numeric',
                        'mode': self.mode
                    }
                elif data_type == 'text':
                    numeric_value = sum([ord(c) for c in str(value)])
                    encrypted_val = {
                        'ciphertext': f"ENC_{hash(str(numeric_value)) % 10000000}",
                        'original_hash': hash(str(value)),
                        'encoded_value': numeric_value,
                        'scheme': self.scheme,
                        'type': 'text',
                        'mode': self.mode
                    }
                elif data_type == 'date':
                    timestamp = pd.Timestamp(value).timestamp()
                    encrypted_val = {
                        'ciphertext': f"ENC_{hash(str(timestamp)) % 10000000}",
                        'original_hash': hash(str(timestamp)),
                        'timestamp': timestamp,
                        'scheme': self.scheme,
                        'type': 'date',
                        'mode': self.mode
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
            try:
                if enc_value is None:
                    decrypted_values.append(None)
                    continue
                if data_type == 'date' and 'timestamp' in enc_value:
                    decrypted_values.append(pd.Timestamp.fromtimestamp(enc_value['timestamp']))
                elif data_type == 'text' and 'encoded_value' in enc_value:
                    decrypted_values.append(enc_value['encoded_value'])
                else:
                    decrypted_values.append(f"DECRYPTED_{enc_value.get('original_hash', 0) % 1000}")
            except Exception as e:
                print(f"Error decrypting value: {str(e)}")
                decrypted_values.append(None)
        return decrypted_values

    def perform_operation(self, encrypted_data1, encrypted_data2, operation):
        """Perform homomorphic operations"""
        if self.mode == 'custom_dll' and operation == 'add':
            results = []
            for enc1, enc2 in zip(encrypted_data1, encrypted_data2):
                if enc1 is not None and enc2 is not None:
                    val1 = enc1.get('value', 0)
                    val2 = enc2.get('value', 0)
                    result = self._call_custom_dll("add", val1, val2)
                    if result.get('status') == 'success':
                        results.append({
                            'value': result.get('result'),
                            'operation': operation,
                            'mode': 'custom_dll'
                        })
                    else:
                        results.append(None)
                else:
                    results.append(None)
            return results

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
                    'operands': [enc1.get('original_hash'), enc2.get('original_hash')],
                    'mode': self.mode
                }
                results.append(result)
            except Exception as e:
                print(f"Error performing operation: {str(e)}")
                results.append(None)
        return results

    def _call_dll_aggregation(self, encrypted_data_list: List[Any], operation: str):
        """Perform aggregation using custom DLL"""
        if not self.custom_dll:
            return {"status": "error", "message": "Custom DLL not available"}
        try:
            dll = ctypes.WinDLL(self.custom_dll)
            values = []
            for enc_data in encrypted_data_list:
                if enc_data and isinstance(enc_data, dict):
                    values.append(float(enc_data.get('value', 0)))
                elif enc_data is not None:
                    values.append(float(enc_data))
            if not values:
                return {"status": "error", "message": "No valid encrypted values"}

            values_array = (ctypes.c_double * len(values))(*values)

            if operation == 'sum':
                func = dll.SumEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            elif operation == 'avg' or operation == 'average':
                func = dll.AverageEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            elif operation == 'min':
                func = dll.MinEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            elif operation == 'max':
                func = dll.MaxEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            elif operation == 'multiply':
                func = dll.MultiplyEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            elif operation == 'subtract':
                func = dll.SubtractEncrypted
                func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
                func.restype = ctypes.c_double
                result = func(values_array, len(values))
            else:
                return {"status": "error", "message": f"Unsupported operation: {operation}"}

            return {
                "status": "success",
                "result": result,
                "operation": operation,
                "count": len(values),
                "mode": "custom_dll"
            }
        except AttributeError as e:
            return {"status": "error", "message": f"DLL function not found: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"DLL aggregation error: {str(e)}"}

    def _call_cpp_aggregation(self, encrypted_data_list: List[Any], operation: str):
        """Perform aggregation using C++ subprocess"""
        if not self.cpp_executable:
            return {"status": "error", "message": "C++ executable not available"}
        try:
            data_file = os.path.join(self.temp_dir, f"agg_data_{int(time.time())}.txt")
            with open(data_file, 'w') as f:
                for enc_data in encrypted_data_list:
                    if enc_data and isinstance(enc_data, dict):
                        value = enc_data.get('value', 0)
                    elif enc_data is not None:
                        value = enc_data
                    else:
                        continue
                    f.write(f"{float(value)}\n")

            result = self._call_cpp_subprocess("aggregate", data_file, operation)
            if result.get('status') == 'success':
                return {
                    "status": "success",
                    "result": result.get('result'),
                    "operation": operation,
                    "count": len(encrypted_data_list),
                    "mode": "subprocess"
                }
            else:
                return result
        except Exception as e:
            return {"status": "error", "message": f"C++ aggregation error: {str(e)}"}

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
                    elif 'value' in enc_data:
                        value = float(enc_data['value'])
                    elif 'encoded_value' in enc_data:
                        value = float(enc_data['encoded_value'])
                    elif 'timestamp' in enc_data:
                        value = float(enc_data['timestamp'])
                    else:
                        continue
                    values.append(value)
                elif isinstance(enc_data, (int, float)):
                    values.append(float(enc_data))

            if not values:
                return {"status": "error", "message": "No valid encrypted values to aggregate"}

            if operation == 'sum':
                result = sum(values)
            elif operation == 'avg' or operation == 'average':
                result = sum(values) / len(values)
            elif operation == 'min':
                result = min(values)
            elif operation == 'max':
                result = max(values)
            elif operation == 'multiply':
                result = np.prod(values)
            elif operation == 'subtract':
                result = values[0]
                for v in values[1:]:
                    result -= v
            else:
                return {"status": "error", "message": f"Unsupported operation: {operation}"}

            encrypted_result = {
                'ciphertext': f"ENC_RESULT_{hash(str(result) + operation) % 10000000}",
                'operation': operation,
                'scheme': self.scheme,
                'count': len(values),
                'mode': self.mode,
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
            return {"status": "error", "message": f"Simulation aggregation error: {str(e)}"}

    def check_scheme_operation_support(self, operation: str) -> tuple:
        """Check if current scheme supports the requested operation"""
        scheme_capabilities = {
            'CKKS': {
                'supported': ['add', 'multiply', 'subtract', 'sum', 'avg', 'average'],
                'limited': ['min', 'max'],
                'unsupported': []
            },
            'BFV': {
                'supported': ['add', 'multiply', 'subtract', 'sum'],
                'limited': ['avg', 'average', 'min', 'max'],
                'unsupported': []
            },
            'BGV': {
                'supported': ['add', 'multiply', 'subtract', 'sum', 'avg', 'average'],
                'limited': ['min', 'max'],
                'unsupported': []
            }
        }
        capabilities = scheme_capabilities.get(self.scheme, {})
        if operation in capabilities.get('supported', []):
            return True, f"✅ {operation} is fully supported by {self.scheme}"
        elif operation in capabilities.get('limited', []):
            return True, f"⚠️ {operation} has limited support in {self.scheme} (approximate results)"
        elif operation in capabilities.get('unsupported', []):
            return False, f"❌ {operation} is NOT supported by {self.scheme}"
        else:
            return True, f"⚠️ {operation} support unknown for {self.scheme}, attempting anyway"

    def encrypt_text_data(self, text_values: List[str]) -> List[Any]:
        """Encrypt text data for CKKS scheme"""
        if self.scheme != 'CKKS':
            raise ValueError(f"Text encryption only supported for CKKS, current scheme: {self.scheme}")
        encrypted_values = []
        for text in text_values:
            if pd.isna(text) or text is None:
                encrypted_values.append(None)
                continue
            try:
                numeric_value = sum([ord(c) for c in str(text)])
                if self.mode == 'custom_dll':
                    result = self._call_custom_dll("encrypt", float(numeric_value))
                    if result.get('status') == 'success':
                        encrypted_values.append({
                            'value': result.get('result'),
                            'original_text': text[:20] + '...' if len(text) > 20 else text,
                            'encoded_value': numeric_value,
                            'type': 'text',
                            'encrypted': True,
                            'mode': 'custom_dll'
                        })
                    else:
                        encrypted_values.append(None)
                else:
                    encrypted_val = {
                        'ciphertext': f"ENC_TEXT_{hash(str(numeric_value)) % 10000000}",
                        'original_hash': hash(str(text)),
                        'encoded_value': numeric_value,
                        'original_text': text[:20] + '...' if len(text) > 20 else text,
                        'scheme': self.scheme,
                        'type': 'text',
                        'mode': self.mode
                    }
                    encrypted_values.append(encrypted_val)
            except Exception as e:
                print(f"Error encrypting text '{text}': {str(e)}")
                encrypted_values.append(None)
        return encrypted_values

    def decrypt_result(self, encrypted_result: Any, result_type: str = 'numeric'):
        """Decrypt a single encrypted result using private key"""
        if not self.private_key:
            raise ValueError("Private key not available")
        try:
            if encrypted_result is None:
                return None
            if self.mode == 'custom_dll':
                return self._decrypt_dll_result(encrypted_result, result_type)
            elif self.mode == 'subprocess':
                return self._decrypt_subprocess_result(encrypted_result, result_type)
            else:
                if isinstance(encrypted_result, dict):
                    if 'simulated_value' in encrypted_result:
                        return encrypted_result['simulated_value']
                    elif 'encoded_value' in encrypted_result:
                        return encrypted_result['encoded_value']
                    elif 'timestamp' in encrypted_result:
                        return pd.Timestamp.fromtimestamp(encrypted_result['timestamp'])
                    elif 'original_hash' in encrypted_result:
                        return abs(encrypted_result['original_hash']) % 10000
                return encrypted_result
        except Exception as e:
            print(f"Decryption error: {str(e)}")
            return None

    def _decrypt_dll_result(self, encrypted_result: Any, result_type: str):
        """Decrypt using custom DLL"""
        try:
            dll = ctypes.WinDLL(self.custom_dll)
            decrypt_func = dll.DecryptValue
            decrypt_func.argtypes = [ctypes.c_double]
            decrypt_func.restype = ctypes.c_double
            if isinstance(encrypted_result, dict):
                enc_value = encrypted_result.get('value', 0)
            else:
                enc_value = encrypted_result
            decrypted = decrypt_func(float(enc_value))
            if result_type == 'date':
                return pd.Timestamp.fromtimestamp(decrypted)
            elif result_type == 'text':
                return int(decrypted)
            else:
                return decrypted
        except Exception as e:
            print(f"DLL decryption error: {str(e)}")
            return None

    def _decrypt_subprocess_result(self, encrypted_result: Any, result_type: str):
        """Decrypt using C++ subprocess"""
        try:
            enc_file = os.path.join(self.temp_dir, f"decrypt_{int(time.time())}.txt")
            if isinstance(encrypted_result, dict):
                enc_value = encrypted_result.get('value', 0)
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

    def perform_aggregation(self, encrypted_data_list: List[Any], operation: str):
        """Perform aggregation on multiple encrypted values"""
        if self.mode == 'custom_dll':
            result = self._call_dll_aggregation(encrypted_data_list, operation)
        elif self.mode == 'subprocess':
            result = self._call_cpp_aggregation(encrypted_data_list, operation)
        else:
            result = self._simulate_aggregation(encrypted_data_list, operation)
        return result

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
        """Cleanup temporary files and DLL"""
        try:
            if self.mode == 'custom_dll' and self.custom_dll:
                try:
                    dll = ctypes.WinDLL(self.custom_dll)
                    cleanup = dll.Cleanup
                    cleanup()
                except:
                    pass
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass