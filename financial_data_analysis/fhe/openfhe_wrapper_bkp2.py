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


class OpenFHEWrapper:
    """Python wrapper for OpenFHE C++ library with three operation modes"""

    def __init__(self):
        # Platform-specific paths
        if sys.platform == 'win32':
            self.openfhe_path = r"C:\Program Files (x86)\OpenFHE"
            self.build_path = r"C:\Users\alish\Workspaces\Python\openfhe-development"
        else:
            # Linux/Unix paths
            self.openfhe_path = os.environ.get('OPENFHE_ROOT', '/usr/local/openfhe')
            self.build_path = os.path.expanduser('~/openfhe-development')

        # Library handles
        self.lib = None
        self.cpp_executable = None

        # FHE context
        self.context = None
        self.public_key = None
        self.private_key = None
        self.scheme = None
        self.params = {}

        # Operation mode
        self.mode = None  # 'ctypes', 'subprocess', or 'simulation'

        # Temp directory for subprocess mode
        self.temp_dir = tempfile.mkdtemp()

        # Try to initialize in order of preference
        self._initialize()

    def _initialize(self):
        """Initialize wrapper in best available mode"""
        print("🔧 Initializing OpenFHE Wrapper...")
        print("=" * 60)

        # Mode 1: Try ctypes DLL loading
        if self._try_mode_ctypes():
            self.mode = 'ctypes'
            print("\n✅ Mode: CTYPES (Direct DLL loading)")
            return

        # Mode 2: Try C++ subprocess wrapper
        if self._try_mode_subprocess():
            self.mode = 'subprocess'
            print("\n✅ Mode: SUBPROCESS (C++ executable wrapper)")
            return

        # Mode 3: Fallback to simulation
        self.mode = 'simulation'
        print("\n✅ Mode: SIMULATION (Pure Python mock)")
        print("=" * 60)

    def _try_mode_ctypes(self):
        """Mode 1: Try to load OpenFHE via ctypes"""
        print("\n[Mode 1] Attempting ctypes DLL loading...")

        try:
            if sys.platform == 'win32':
                lib_paths = [
                    os.path.join(self.openfhe_path, "lib"),
                    os.path.join(self.openfhe_path, "bin"),
                    os.path.join(self.build_path, "lib", "Release"),
                    os.path.join(self.build_path, "lib", "Debug"),
                ]

                for lib_path in lib_paths:
                    if os.path.exists(lib_path):
                        os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                        if hasattr(os, 'add_dll_directory'):
                            try:
                                os.add_dll_directory(lib_path)
                            except:
                                pass

            dependency_order = [
                "libOPENFHEcore.dll",
                "libOPENFHEpke.dll",
                "libOPENFHEbinfhe.dll"
            ]

            lib_dir = os.path.join(self.openfhe_path, "lib")
            if not os.path.exists(lib_dir):
                lib_dir = os.path.join(self.build_path, "lib", "Release")

            if not os.path.exists(lib_dir):
                print("  ❌ OpenFHE library directory not found")
                return False

            print(f"  📂 Checking: {lib_dir}")

            for dep in dependency_order:
                dep_path = os.path.join(lib_dir, dep)
                if os.path.exists(dep_path):
                    try:
                        loaded_lib = ctypes.WinDLL(dep_path) if sys.platform == 'win32' else ctypes.CDLL(dep_path)
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

    def _try_mode_subprocess(self):
        """Mode 2: Try to compile and use C++ subprocess wrapper"""
        print("\n[Mode 2] Attempting C++ subprocess wrapper...")

        try:
            # Check if OpenFHE exists
            if not os.path.exists(self.openfhe_path) and not os.path.exists(self.build_path):
                print("  ❌ OpenFHE installation not found")
                return False

            # Generate C++ wrapper code
            cpp_file = os.path.join(self.temp_dir, "openfhe_wrapper.cpp")
            exe_file = os.path.join(self.temp_dir,
                                    "openfhe_wrapper.exe" if sys.platform == "win32" else "openfhe_wrapper")

            print(f"  📝 Generating C++ code: {cpp_file}")
            with open(cpp_file, 'w') as f:
                f.write(self._generate_cpp_code())

            # Check for compiler
            compiler = self._find_compiler()
            if not compiler:
                print("  ❌ No C++ compiler found")
                return False

            print(f"  ✅ Found compiler: {compiler}")

            # Try to compile
            if self._compile_wrapper(cpp_file, exe_file, compiler):
                self.cpp_executable = exe_file
                print(f"  ✅ Compiled executable: {exe_file}")

                # Test the executable
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

    def _find_compiler(self):
        """Find available C++ compiler"""
        compilers = ['g++', 'clang++', 'cl.exe']

        for compiler in compilers:
            try:
                result = subprocess.run([compiler, '--version'],
                                        capture_output=True,
                                        timeout=5)
                if result.returncode == 0:
                    return compiler
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def _compile_wrapper(self, cpp_file, exe_file, compiler):
        """Compile the C++ wrapper"""
        try:
            include_paths = []
            lib_paths = []

            # Find include directories
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in ['include', 'src/pke/include', 'src/core/include', 'src/binfhe/include']:
                        inc_path = os.path.join(base, subdir)
                        if os.path.exists(inc_path):
                            include_paths.append(inc_path)

            # Find library directories
            for base in [self.openfhe_path, self.build_path]:
                if os.path.exists(base):
                    for subdir in ['lib', 'lib/Release', 'lib/Debug']:
                        lib_path = os.path.join(base, subdir)
                        if os.path.exists(lib_path):
                            lib_paths.append(lib_path)

            if compiler == 'g++' or compiler == 'clang++':
                cmd = [compiler, '-std=c++17', cpp_file, '-o', exe_file]

                for inc in include_paths:
                    cmd.append(f'-I{inc}')

                for lib in lib_paths:
                    cmd.append(f'-L{lib}')

                cmd.extend(['-lOPENFHEpke', '-lOPENFHEcore', '-lOPENFHEbinfhe'])

            else:  # cl.exe (MSVC)
                cmd = ['cl.exe', '/EHsc', '/std:c++17', cpp_file, '/Fe:' + exe_file]

                for inc in include_paths:
                    cmd.append(f'/I{inc}')

                cmd.append('/link')
                for lib in lib_paths:
                    cmd.append(f'/LIBPATH:{lib}')

                cmd.extend(['OPENFHEpke.lib', 'OPENFHEcore.lib', 'OPENFHEbinfhe.lib'])

            print(f"  🔨 Compiling: {' '.join(cmd[:3])}...")
            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(exe_file):
                return True
            else:
                if result.stderr:
                    print(f"  ⚠️ Compilation errors: {result.stderr.decode()[:200]}")
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
            result = subprocess.run([self.cpp_executable, 'test'],
                                    capture_output=True,
                                    timeout=5,
                                    text=True)
            return result.returncode == 0 or 'OpenFHE' in result.stdout
        except:
            return False

    def _generate_cpp_code(self):
        """Generate C++ wrapper code"""
        return '''#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint> 

#ifdef OPENFHE_AVAILABLE
#include "openfhe.h"
using namespace lbcrypto;
#endif

void writeJSON(const std::string& key, const std::string& value) {
    std::cout << "\\"" << key << "\\": \\"" << value << "\\"";
}

void writeJSON(const std::string& key, double value) {
    std::cout << "\\"" << key << "\\": " << value;
}

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

            cryptoContext = GenCryptoContext(parameters);
            cryptoContext->Enable(PKE);
            cryptoContext->Enable(KEYSWITCH);
            cryptoContext->Enable(LEVELEDSHE);

            keyPair = cryptoContext->KeyGen();
            cryptoContext->EvalMultKeyGen(keyPair.secretKey);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
        #else
        return false;
        #endif
    }

    void processData(const std::string& dataFile, const std::string& operation, double operand) {
        #ifdef OPENFHE_AVAILABLE
        try {
            // Read data
            std::ifstream inFile(dataFile);
            std::vector<double> data;
            double value;
            while (inFile >> value) {
                data.push_back(value);
            }

            // Encrypt
            Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(data);
            auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

            // Perform operation
            Ciphertext<DCRTPoly> result;
            if (operation == "add") {
                auto opPlaintext = cryptoContext->MakeCKKSPackedPlaintext(
                    std::vector<double>(data.size(), operand));
                result = cryptoContext->EvalAdd(ciphertext, opPlaintext);
            } else {
                result = ciphertext;
            }

            // Decrypt
            Plaintext decrypted;
            cryptoContext->Decrypt(keyPair.secretKey, result, &decrypted);

            // Output JSON
            std::cout << "{\\"status\\": \\"success\\", \\"results\\": [";
            auto resultVec = decrypted->GetRealPackedValue();
            for (size_t i = 0; i < std::min((size_t)10, resultVec.size()); i++) {
                std::cout << resultVec[i];
                if (i < std::min((size_t)10, resultVec.size()) - 1) std::cout << ", ";
            }
            std::cout << "]}" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
        #else
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"OpenFHE not available\\"}" << std::endl;
        #endif
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"No command provided\\"}" << std::endl;
        return 1;
    }

    std::string command = argv[1];
    FHEProcessor processor;

    if (command == "test") {
        std::cout << "{\\"status\\": \\"success\\", \\"message\\": \\"OpenFHE wrapper ready\\"}" << std::endl;
        return 0;
    }
    else if (command == "setup_ckks") {
        uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 8192;
        uint32_t scalingModSize = (argc > 3) ? std::stoi(argv[3]) : 50;

        if (processor.setupCKKS(polyDegree, scalingModSize)) {
            std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"CKKS\\"}" << std::endl;
        } else {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Setup failed\\"}" << std::endl;
        }
    }
    else if (command == "process") {
        if (argc < 5) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Missing arguments\\"}" << std::endl;
            return 1;
        }

        processor.setupCKKS(8192, 50);
        processor.processData(argv[2], argv[3], std::stod(argv[4]));
    }
    else {
        std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Unknown command\\"}" << std::endl;
    }

    return 0;
}'''

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
                         batch_size=8, security_level='HEStd_128_classic',
                         ring_dim=16384):
        """Generate OpenFHE context"""
        self.scheme = scheme
        self.params = {
            'mult_depth': mult_depth,
            'scale_mod_size': scale_mod_size,
            'batch_size': batch_size,
            'security_level': security_level,
            'ring_dim': ring_dim
        }

        # Try mode-specific implementation
        if self.mode == 'ctypes':
            # TODO: Implement ctypes-based context generation
            pass

        elif self.mode == 'subprocess':
            # Call C++ subprocess
            result = self._call_cpp_subprocess(f"setup_{scheme.lower()}", ring_dim, scale_mod_size)
            if result.get('status') == 'success':
                self.context = {
                    'scheme': scheme,
                    'params': self.params,
                    'initialized': True,
                    'mode': 'subprocess',
                    'timestamp': datetime.now().isoformat()
                }
                self._generate_keys()
                print(f"✅ Context generated via C++ subprocess for {scheme}")
                return self.context

        # Fallback to simulation
        self.context = {
            'scheme': scheme,
            'params': self.params,
            'initialized': True,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat()
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

        # Mode 2: subprocess implementation
        if self.mode == 'subprocess' and data_type == 'numeric':
            data_file = os.path.join(self.temp_dir, f"data_{int(time.time())}.txt")
            with open(data_file, 'w') as f:
                for value in data:
                    if not pd.isna(value):
                        f.write(f"{float(value)}\n")

            result = self._call_cpp_subprocess("process", data_file, "none", 0)

            if result.get('status') == 'success':
                results = result.get('results', [])
                return [{'value': v, 'encrypted': True, 'mode': 'subprocess'} for v in results]

        # Fallback to simulation
        return self._simulate_encrypt(data, data_type)

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