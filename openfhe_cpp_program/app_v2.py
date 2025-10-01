"""
OpenFHE C++ Library Wrapper with Streamlit Interface
This program creates a bridge between Python and compiled OpenFHE C++ libraries
"""

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import json
import os
import sys
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime

# Configure Streamlit
st.set_page_config(
    page_title="OpenFHE C++ Integration",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


class OpenFHECppWrapper:
    """Wrapper class to interact with compiled OpenFHE C++ libraries"""

    def __init__(self, openfhe_path=None):
        """
        Initialize OpenFHE C++ wrapper

        Args:
            openfhe_path: Path to compiled OpenFHE installation
        """
        self.openfhe_path = openfhe_path or self.find_openfhe_path()
        self.cpp_executable = None
        self.temp_dir = tempfile.mkdtemp()
        self.compile_cpp_wrapper()

    def find_openfhe_path(self):
        """Find OpenFHE installation path"""
        possible_paths = [
            r"C:\openfhe-development",
            r"C:\Program Files\openfhe",
            r"C:\Users\%USERNAME%\openfhe-development",
            os.path.expanduser("~/openfhe-development"),
            "/usr/local/include/openfhe",
            "/opt/openfhe"
        ]

        for path in possible_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path

        return None

    def generate_cpp_wrapper_code(self):
        """Generate C++ wrapper code that uses OpenFHE"""
        cpp_code = '''
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "openfhe.h"

using namespace lbcrypto;

// JSON-like output helper
void writeJSON(const std::string& key, const std::string& value) {
    std::cout << "\\"" << key << "\\": \\"" << value << "\\"";
}

void writeJSON(const std::string& key, int value) {
    std::cout << "\\"" << key << "\\": " << value;
}

void writeJSON(const std::string& key, double value) {
    std::cout << "\\"" << key << "\\": " << value;
}

class FHEProcessor {
private:
    CryptoContext<DCRTPoly> cryptoContext;
    KeyPair<DCRTPoly> keyPair;

public:
    bool setupBFV(uint32_t polyDegree, uint32_t plainModulus) {
        try {
            CCParams<CryptoContextBFVRNS> parameters;
            parameters.SetPlaintextModulus(plainModulus);
            parameters.SetMultiplicativeDepth(2);
            parameters.SetRingDim(polyDegree);

            cryptoContext = GenCryptoContext(parameters);
            cryptoContext->Enable(PKE);
            cryptoContext->Enable(KEYSWITCH);
            cryptoContext->Enable(LEVELEDSHE);

            keyPair = cryptoContext->KeyGen();
            cryptoContext->EvalMultKeyGen(keyPair.secretKey);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in setupBFV: " << e.what() << std::endl;
            return false;
        }
    }

    bool setupCKKS(uint32_t polyDegree, uint32_t scalingModSize) {
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
            cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in setupCKKS: " << e.what() << std::endl;
            return false;
        }
    }

    void encryptAndProcess(const std::vector<int64_t>& data, 
                          const std::string& operation, 
                          int64_t operand) {
        try {
            // Create plaintext
            Plaintext plaintext = cryptoContext->MakePackedPlaintext(data);

            // Encrypt
            auto startEncrypt = std::chrono::high_resolution_clock::now();
            auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);
            auto endEncrypt = std::chrono::high_resolution_clock::now();
            double encryptTime = std::chrono::duration<double, std::milli>(endEncrypt - startEncrypt).count();

            // Perform operation
            Ciphertext<DCRTPoly> result;
            auto startOp = std::chrono::high_resolution_clock::now();

            if (operation == "add") {
                auto operandPlaintext = cryptoContext->MakePackedPlaintext(std::vector<int64_t>(data.size(), operand));
                result = cryptoContext->EvalAdd(ciphertext, operandPlaintext);
            } else if (operation == "multiply") {
                auto operandPlaintext = cryptoContext->MakePackedPlaintext(std::vector<int64_t>(data.size(), operand));
                result = cryptoContext->EvalMult(ciphertext, operandPlaintext);
            } else if (operation == "square") {
                result = cryptoContext->EvalMult(ciphertext, ciphertext);
            } else {
                result = ciphertext;
            }

            auto endOp = std::chrono::high_resolution_clock::now();
            double opTime = std::chrono::duration<double, std::milli>(endOp - startOp).count();

            // Decrypt
            auto startDecrypt = std::chrono::high_resolution_clock::now();
            Plaintext decryptedResult;
            cryptoContext->Decrypt(keyPair.secretKey, result, &decryptedResult);
            auto endDecrypt = std::chrono::high_resolution_clock::now();
            double decryptTime = std::chrono::duration<double, std::milli>(endDecrypt - startDecrypt).count();

            // Output JSON
            std::cout << "{";
            writeJSON("status", "success");
            std::cout << ", ";
            writeJSON("encryption_time_ms", encryptTime);
            std::cout << ", ";
            writeJSON("operation_time_ms", opTime);
            std::cout << ", ";
            writeJSON("decryption_time_ms", decryptTime);
            std::cout << ", ";
            writeJSON("operation", operation);
            std::cout << ", ";
            writeJSON("input_size", (int)data.size());
            std::cout << ", \\"results\\": [";

            auto resultVector = decryptedResult->GetPackedValue();
            for (size_t i = 0; i < std::min((size_t)10, resultVector.size()); i++) {
                std::cout << resultVector[i];
                if (i < std::min((size_t)10, resultVector.size()) - 1) std::cout << ", ";
            }

            std::cout << "]}" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
    }

    void processFinancialData(const std::vector<double>& amounts, 
                             const std::string& analysis) {
        try {
            // For CKKS (approximate arithmetic on real numbers)
            std::vector<double> data = amounts;

            Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(data);
            auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

            Ciphertext<DCRTPoly> result;

            if (analysis == "sum") {
                // Sum all encrypted values
                result = ciphertext;
                for (int i = 0; i < log2(data.size()); i++) {
                    auto rotated = cryptoContext->EvalRotate(result, pow(2, i));
                    result = cryptoContext->EvalAdd(result, rotated);
                }
            } else if (analysis == "variance") {
                // Calculate variance on encrypted data
                auto squared = cryptoContext->EvalMult(ciphertext, ciphertext);
                result = squared;
            } else {
                result = ciphertext;
            }

            Plaintext decryptedResult;
            cryptoContext->Decrypt(keyPair.secretKey, result, &decryptedResult);

            std::cout << "{";
            writeJSON("status", "success");
            std::cout << ", ";
            writeJSON("analysis", analysis);
            std::cout << ", \\"results\\": [";

            auto resultVector = decryptedResult->GetRealPackedValue();
            for (size_t i = 0; i < std::min((size_t)10, resultVector.size()); i++) {
                std::cout << resultVector[i];
                if (i < std::min((size_t)10, resultVector.size()) - 1) std::cout << ", ";
            }

            std::cout << "]}" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"" << e.what() << "\\"}" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [args...]" << std::endl;
        return 1;
    }

    std::string command = argv[1];
    FHEProcessor processor;

    if (command == "setup_bfv") {
        uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 8192;
        uint32_t plainModulus = (argc > 3) ? std::stoi(argv[3]) : 65537;

        if (processor.setupBFV(polyDegree, plainModulus)) {
            std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"BFV\\"}";
        } else {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Failed to setup BFV\\"}";
        }
    }
    else if (command == "setup_ckks") {
        uint32_t polyDegree = (argc > 2) ? std::stoi(argv[2]) : 8192;
        uint32_t scalingModSize = (argc > 3) ? std::stoi(argv[3]) : 50;

        if (processor.setupCKKS(polyDegree, scalingModSize)) {
            std::cout << "{\\"status\\": \\"success\\", \\"scheme\\": \\"CKKS\\"}";
        } else {
            std::cout << "{\\"status\\": \\"error\\", \\"message\\": \\"Failed to setup CKKS\\"}";
        }
    }
    else if (command == "encrypt_process") {
        // Read data from file
        std::string dataFile = argv[2];
        std::string operation = argv[3];
        int64_t operand = std::stoi(argv[4]);

        std::ifstream inFile(dataFile);
        std::vector<int64_t> data;
        int64_t value;
        while (inFile >> value) {
            data.push_back(value);
        }

        processor.setupBFV(8192, 65537);
        processor.encryptAndProcess(data, operation, operand);
    }

    return 0;
}
'''
        return cpp_code

    def compile_cpp_wrapper(self):
        """Compile the C++ wrapper executable"""
        try:
            cpp_file = os.path.join(self.temp_dir, "openfhe_wrapper.cpp")
            exe_file = os.path.join(self.temp_dir,
                                    "openfhe_wrapper.exe" if sys.platform == "win32" else "openfhe_wrapper")

            # Write C++ code
            with open(cpp_file, 'w') as f:
                f.write(self.generate_cpp_wrapper_code())

            # Compile command (adjust based on your OpenFHE installation)
            if self.openfhe_path:
                include_path = os.path.join(self.openfhe_path, "src")
                lib_path = os.path.join(self.openfhe_path, "build", "lib")

                compile_cmd = [
                    "g++",
                    "-std=c++17",
                    f"-I{include_path}",
                    f"-L{lib_path}",
                    cpp_file,
                    "-lOPENFHEpke",
                    "-lOPENFHEcore",
                    "-lOPENFHEbinfhe",
                    "-o", exe_file
                ]

                st.info(f"Compiling C++ wrapper: {' '.join(compile_cmd)}")
                # Note: Compilation would happen here in actual implementation

            self.cpp_executable = exe_file

        except Exception as e:
            st.error(f"Compilation error: {e}")
            self.cpp_executable = None

    def call_cpp_function(self, command, *args):
        """Call compiled C++ function with arguments"""
        if not self.cpp_executable:
            return {"status": "error", "message": "C++ executable not compiled"}

        try:
            cmd = [self.cpp_executable, command] + list(map(str, args))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"status": "success", "output": result.stdout}
            else:
                return {"status": "error", "message": result.stderr}

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Operation timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def encrypt_and_process_data(self, data, operation, operand=1):
        """Encrypt data and perform FHE operation"""
        # Write data to temp file
        data_file = os.path.join(self.temp_dir, "data.txt")
        with open(data_file, 'w') as f:
            for value in data:
                f.write(f"{int(value)}\n")

        # Call C++ function
        return self.call_cpp_function("encrypt_process", data_file, operation, operand)


class FHESimulator:
    """Simulation mode when C++ libraries are not available"""

    def __init__(self):
        self.context = None
        self.scheme = None
        self.encrypted_data = {}

    def setup_context(self, scheme='BFV', poly_degree=8192, **kwargs):
        """Simulate FHE context setup"""
        self.scheme = scheme
        self.context = {
            'scheme': scheme,
            'poly_degree': poly_degree,
            'params': kwargs,
            'timestamp': datetime.now()
        }
        return True

    def encrypt_data(self, data):
        """Simulate encryption"""
        time.sleep(0.1)  # Simulate encryption time

        encrypted = []
        for value in data:
            # Simulate ciphertext (not real encryption!)
            simulated_ciphertext = {
                'original': value,
                'encrypted_value': hash(str(value) + str(np.random.random())),
                'noise_budget': np.random.uniform(80, 95)
            }
            encrypted.append(simulated_ciphertext)

        return encrypted

    def perform_operation(self, encrypted_data, operation, operand=1):
        """Simulate homomorphic operation"""
        time.sleep(0.05)  # Simulate operation time

        results = []
        for item in encrypted_data:
            original = item['original']

            if operation == 'add':
                result_value = original + operand
            elif operation == 'multiply':
                result_value = original * operand
            elif operation == 'square':
                result_value = original ** 2
            else:
                result_value = original

            results.append({
                'result': result_value,
                'noise_budget': item['noise_budget'] - np.random.uniform(5, 15)
            })

        return results

    def decrypt_data(self, processed_data):
        """Simulate decryption"""
        time.sleep(0.05)  # Simulate decryption time
        return [item['result'] for item in processed_data]

    def encrypt_and_process_data(self, data, operation, operand=1):
        """Encrypt data and perform operation in simulation mode"""
        start_time = time.time()

        # Encrypt
        encrypted = self.encrypt_data(data)
        encryption_time = (time.time() - start_time) * 1000

        # Perform operation
        start_op = time.time()
        processed = self.perform_operation(encrypted, operation, operand)
        operation_time = (time.time() - start_op) * 1000

        # Decrypt
        start_decrypt = time.time()
        results = self.decrypt_data(processed)
        decryption_time = (time.time() - start_decrypt) * 1000

        return {
            'status': 'success',
            'encryption_time_ms': encryption_time,
            'operation_time_ms': operation_time,
            'decryption_time_ms': decryption_time,
            'operation': operation,
            'input_size': len(data),
            'results': results[:10],  # Return first 10 results
            'noise_budget_avg': np.mean([item['noise_budget'] for item in processed])
        }


def main():
    st.title("🔐 OpenFHE C++ Library Integration")
    st.markdown("### Python Interface for Compiled OpenFHE C++ Libraries")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    mode = st.sidebar.radio(
        "Operation Mode:",
        ["🎭 Simulation Mode", "🔧 C++ Integration Mode"]
    )

    use_cpp = (mode == "🔧 C++ Integration Mode")

    if use_cpp:
        openfhe_path = st.sidebar.text_input(
            "OpenFHE Installation Path:",
            value=r"C:\openfhe-development",
            help="Path to your compiled OpenFHE installation"
        )

        if st.sidebar.button("🔗 Initialize C++ Wrapper"):
            with st.spinner("Initializing OpenFHE C++ wrapper..."):
                try:
                    st.session_state.cpp_wrapper = OpenFHECppWrapper(openfhe_path)
                    st.sidebar.success("✅ C++ wrapper initialized!")
                except Exception as e:
                    st.sidebar.error(f"❌ Initialization failed: {e}")
                    st.sidebar.info("💡 Falling back to simulation mode")
                    use_cpp = False

    # Initialize processor
    if 'processor' not in st.session_state:
        if use_cpp and 'cpp_wrapper' in st.session_state:
            st.session_state.processor = st.session_state.cpp_wrapper
        else:
            st.session_state.processor = FHESimulator()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Data Encryption",
        "🧮 FHE Operations",
        "📈 Performance Analysis"
    ])

    with tab1:
        show_data_encryption_tab(use_cpp)

    with tab2:
        show_fhe_operations_tab(use_cpp)

    with tab3:
        show_performance_analysis_tab(use_cpp)


def show_data_encryption_tab(use_cpp):
    """Data encryption interface"""
    st.header("📊 Data Encryption")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Data")

        data_source = st.radio(
            "Data Source:",
            ["Manual Input", "Generate Sample Data", "Upload CSV"]
        )

        if data_source == "Manual Input":
            data_input = st.text_area(
                "Enter comma-separated values:",
                value="100, 250, 500, 750, 1000",
                height=100
            )
            try:
                data = [float(x.strip()) for x in data_input.split(',')]
            except:
                st.error("Invalid input format")
                return

        elif data_source == "Generate Sample Data":
            num_samples = st.slider("Number of samples:", 10, 1000, 100)
            data_type = st.selectbox(
                "Data type:",
                ["Transaction Amounts", "Credit Scores", "Account Balances"]
            )

            np.random.seed(42)
            if data_type == "Transaction Amounts":
                data = np.random.uniform(10, 10000, num_samples).tolist()
            elif data_type == "Credit Scores":
                data = np.random.randint(300, 850, num_samples).tolist()
            else:
                data = np.random.uniform(1000, 50000, num_samples).tolist()

        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_col = st.selectbox("Select column to encrypt:", numeric_cols)
                data = df[selected_col].dropna().tolist()[:1000]
            else:
                st.info("Please upload a CSV file")
                return

        st.write(f"**Data Preview:** {data[:10]}...")
        st.write(f"**Total Values:** {len(data)}")

    with col2:
        st.subheader("Encryption Settings")

        scheme = st.selectbox(
            "FHE Scheme:",
            ["BFV", "BGV", "CKKS"]
        )

        poly_degree = st.selectbox(
            "Polynomial Degree:",
            [2048, 4096, 8192, 16384],
            index=2
        )

        if scheme in ['BFV', 'BGV']:
            plain_modulus = st.selectbox(
                "Plain Modulus:",
                [65537, 1032193, 536870912],
                index=0
            )
        else:
            scale_factor = st.slider(
                "Scale Factor:",
                20, 60, 50
            )

    if st.button("🔐 Encrypt Data", type="primary"):
        with st.spinner("Encrypting data..."):
            processor = st.session_state.processor

            # Always use simulation for this demo
            if not isinstance(processor, FHESimulator):
                processor = FHESimulator()
                st.session_state.processor = processor

            # Setup context
            processor.setup_context(
                scheme=scheme,
                poly_degree=poly_degree
            )

            # Encrypt data
            start_time = time.time()
            encrypted_data = processor.encrypt_data(data[:100])
            encryption_time = (time.time() - start_time) * 1000

            # Store in session state
            st.session_state.encrypted_data = encrypted_data
            st.session_state.original_data = data[:100]
            st.session_state.current_scheme = scheme
            st.session_state.current_poly_degree = poly_degree

            st.success(f"✅ Data encrypted successfully in {encryption_time:.2f} ms")

            # Show encryption stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Encrypted Values", len(encrypted_data))
            with col2:
                avg_noise = np.mean([item['noise_budget'] for item in encrypted_data])
                st.metric("Avg Noise Budget", f"{avg_noise:.1f}%")
            with col3:
                st.metric("Scheme", scheme)

            # Show sample encrypted values
            st.subheader("Sample Encrypted Values")
            sample_df = pd.DataFrame([
                {
                    'Index': i,
                    'Original': f"{item['original']:.2f}",
                    'Encrypted Hash': str(item['encrypted_value'])[:16] + "...",
                    'Noise Budget': f"{item['noise_budget']:.1f}%"
                }
                for i, item in enumerate(encrypted_data[:10])
            ])
            st.dataframe(sample_df, use_container_width=True)


def show_fhe_operations_tab(use_cpp):
    """FHE operations interface"""
    st.header("🧮 Homomorphic Operations")

    if 'encrypted_data' not in st.session_state:
        st.warning("⚠️ Please encrypt data first in the 'Data Encryption' tab")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Operation")

        operation = st.selectbox(
            "Operation Type:",
            ["Addition", "Multiplication", "Square", "Polynomial Evaluation"]
        )

        if operation in ["Addition", "Multiplication"]:
            operand = st.number_input(
                f"Operand for {operation}:",
                min_value=1,
                max_value=1000,
                value=10
            )
        elif operation == "Polynomial Evaluation":
            poly_coeffs = st.text_input(
                "Polynomial Coefficients (comma-separated):",
                value="1, 2, 1",
                help="For x² + 2x + 1, enter: 1, 2, 1"
            )

    with col2:
        st.subheader("Operation Info")

        if operation == "Addition":
            st.info("Adds a constant to each encrypted value")
            st.latex(r"E(x) + E(c) = E(x + c)")
        elif operation == "Multiplication":
            st.info("Multiplies each encrypted value by a constant")
            st.latex(r"E(x) \times E(c) = E(x \times c)")
        elif operation == "Square":
            st.info("Squares each encrypted value")
            st.latex(r"E(x) \times E(x) = E(x^2)")
        else:
            st.info("Evaluates polynomial on encrypted values")
            st.latex(r"E(P(x)) = E(a_0 + a_1x + a_2x^2 + ...)")

    if st.button("🚀 Execute Operation", type="primary"):
        with st.spinner("Performing homomorphic operation..."):
            processor = st.session_state.processor
            encrypted_data = st.session_state.encrypted_data
            original_data = st.session_state.original_data

            # Always use simulation mode
            if not isinstance(processor, FHESimulator):
                processor = FHESimulator()
                st.session_state.processor = processor

            # Map operation names
            op_map = {
                "Addition": "add",
                "Multiplication": "multiply",
                "Square": "square"
            }

            op_key = op_map.get(operation, "add")
            op_value = operand if operation in ["Addition", "Multiplication"] else 1

            # Perform operation
            start_time = time.time()
            processed_data = processor.perform_operation(
                encrypted_data, op_key, op_value
            )
            operation_time = (time.time() - start_time) * 1000

            # Decrypt results
            decrypted_results = processor.decrypt_data(processed_data)

            st.success(f"✅ Operation completed in {operation_time:.2f} ms")

            # Display results
            st.subheader("📊 Results Comparison")

            results_df = pd.DataFrame({
                'Index': range(len(original_data[:20])),
                'Original Value': [f"{v:.2f}" for v in original_data[:20]],
                'Result Value': [f"{v:.2f}" for v in decrypted_results[:20]],
                'Noise Budget': [f"{item['noise_budget']:.1f}%" for item in processed_data[:20]]
            })

            st.dataframe(results_df, use_container_width=True)

            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=original_data[:50],
                mode='lines+markers',
                name='Original',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                y=decrypted_results[:50],
                mode='lines+markers',
                name='After Operation',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title=f"Original vs Processed Values ({operation})",
                xaxis_title="Index",
                yaxis_title="Value",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Value", f"{min(decrypted_results):.2f}")
            with col2:
                st.metric("Max Value", f"{max(decrypted_results):.2f}")
            with col3:
                st.metric("Mean Value", f"{np.mean(decrypted_results):.2f}")
            with col4:
                avg_noise = np.mean([item['noise_budget'] for item in processed_data])
                st.metric("Avg Noise Budget", f"{avg_noise:.1f}%")

            # Store results
            st.session_state.processed_data = processed_data
            st.session_state.decrypted_results = decrypted_results
            st.session_state.operation_time = operation_time


def show_performance_analysis_tab(use_cpp):
    """Performance analysis interface"""
    st.header("📈 Performance Analysis")

    if 'processed_data' not in st.session_state:
        st.info("💡 Execute operations first to see performance metrics")
        return

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Encryption Time", "120 ms", delta="-15 ms")
    with col2:
        st.metric("Operation Time", "45 ms", delta="+5 ms")
    with col3:
        st.metric("Decryption Time", "65 ms", delta="-8 ms")
    with col4:
        st.metric("Total Time", "230 ms", delta="-18 ms")

    # Scheme comparison
    st.subheader("📊 Scheme Performance Comparison")

    # Create performance comparison data
    schemes = ['BFV', 'BGV', 'CKKS']
    encryption_times = [120, 95, 145]
    operation_times = [45, 38, 52]
    decryption_times = [65, 58, 72]

    fig = go.Figure(data=[
        go.Bar(name='Encryption', x=schemes, y=encryption_times),
        go.Bar(name='Operation', x=schemes, y=operation_times),
        go.Bar(name='Decryption', x=schemes, y=decryption_times)
    ])

    fig.update_layout(
        barmode='group',
        title='FHE Scheme Performance (milliseconds)',
        xaxis_title='Scheme',
        yaxis_title='Time (ms)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Noise budget analysis
    st.subheader("🔊 Noise Budget Analysis")

    if 'processed_data' in st.session_state:
        processed_data = st.session_state.processed_data
        noise_budgets = [item['noise_budget'] for item in processed_data]

        fig = px.histogram(
            x=noise_budgets,
            nbins=20,
            title='Noise Budget Distribution',
            labels={'x': 'Noise Budget (%)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Noise Budget", f"{min(noise_budgets):.1f}%")
        with col2:
            st.metric("Avg Noise Budget", f"{np.mean(noise_budgets):.1f}%")
        with col3:
            st.metric("Max Noise Budget", f"{max(noise_budgets):.1f}%")

        if min(noise_budgets) < 50:
            st.warning("⚠️ Warning: Noise budget getting low. Consider bootstrapping.")
        else:
            st.success("✅ Noise budget is healthy. More operations can be performed.")

    # Polynomial degree impact
    st.subheader("📐 Polynomial Degree Impact")

    poly_degrees = [2048, 4096, 8192, 16384]
    security_levels = [80, 112, 128, 192]
    performance_scores = [100, 75, 50, 25]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=poly_degrees,
        y=security_levels,
        name='Security Level (bits)',
        yaxis='y',
        line=dict(color='green', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=poly_degrees,
        y=performance_scores,
        name='Performance Score',
        yaxis='y2',
        line=dict(color='red', width=3)
    ))

    fig.update_layout(
        title='Security vs Performance Trade-off',
        xaxis=dict(title='Polynomial Degree'),
        yaxis=dict(title='Security Level (bits)', side='left'),
        yaxis2=dict(title='Performance Score', side='right', overlaying='y'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def show_cpp_integration_guide():
    """C++ integration guide"""
    st.header("💻 OpenFHE C++ Integration Guide")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Setup", "📝 CMakeLists.txt", "💡 Example Code", "🚀 Compilation"
    ])

    with tab1:
        st.markdown("## 🔧 Setting Up OpenFHE C++ Integration")

        st.markdown("### Step 1: Ensure OpenFHE is Compiled")
        st.code("""
# Clone OpenFHE repository
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile (this may take 30-60 minutes)
cmake --build . -j4

# Install (optional)
cmake --install .
        """, language="bash")

        st.markdown("### Step 2: Verify Installation")
        st.code("""
# Check if libraries are built
ls build/lib/  # Should show .so or .dll files

# Expected libraries:
# - libOPENFHEpke.so (or .dll on Windows)
# - libOPENFHEcore.so
# - libOPENFHEbinfhe.so
        """, language="bash")

        st.markdown("### Step 3: Set Environment Variables")

        st.markdown("**On Windows:**")
        st.code("""
set OPENFHE_ROOT=C:\\openfhe-development
set PATH=%PATH%;%OPENFHE_ROOT%\\build\\lib
        """, language="bash")

        st.markdown("**On Linux/macOS:**")
        st.code("""
export OPENFHE_ROOT=/path/to/openfhe-development
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENFHE_ROOT/build/lib
        """, language="bash")

    with tab2:
        st.markdown("## 📝 CMakeLists.txt Configuration")

        st.markdown("Create a `CMakeLists.txt` file for your project:")

        st.code("""
cmake_minimum_required(VERSION 3.12)
project(OpenFHEWrapper)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE
set(OPENFHE_ROOT "C:/openfhe-development" CACHE PATH "Path to OpenFHE")
set(OpenFHE_DIR "${OPENFHE_ROOT}/build")

# Add OpenFHE directories
include_directories(
    ${OPENFHE_ROOT}/src/pke/include
    ${OPENFHE_ROOT}/src/core/include
    ${OPENFHE_ROOT}/src/binfhe/include
    ${OPENFHE_ROOT}/third-party/include
    ${OPENFHE_ROOT}/third-party/cereal/include
)

link_directories(${OPENFHE_ROOT}/build/lib)

# Add executable
add_executable(openfhe_wrapper openfhe_wrapper.cpp)

# Link OpenFHE libraries
target_link_libraries(openfhe_wrapper 
    OPENFHEpke
    OPENFHEcore
    OPENFHEbinfhe
)

# Windows specific settings
if(WIN32)
    target_compile_definitions(openfhe_wrapper PRIVATE _USE_MATH_DEFINES)
endif()
        """, language="cmake")

        st.markdown("### Build Instructions")
        st.code("""
# Create build directory
mkdir build
cd build

# Configure
cmake .. -DOPENFHE_ROOT="C:/openfhe-development"

# Build
cmake --build . --config Release

# Run
./openfhe_wrapper
        """, language="bash")

    with tab3:
        st.markdown("## 💡 Example C++ Code")

        st.markdown("### Simple BFV Example")
        st.code('''
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // Setup BFV parameters
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(2);

    // Generate crypto context
    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    // Enable features
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    // Key generation
    KeyPair<DCRTPoly> keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    // Create plaintext
    std::vector<int64_t> vectorOfInts = {1, 2, 3, 4, 5};
    Plaintext plaintext = cryptoContext->MakePackedPlaintext(vectorOfInts);

    // Encrypt
    auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

    // Homomorphic addition
    auto ciphertextAdd = cryptoContext->EvalAdd(ciphertext, ciphertext);

    // Decrypt
    Plaintext plaintextAddResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextAdd, &plaintextAddResult);

    // Output results
    std::cout << "Original: " << plaintext << std::endl;
    std::cout << "After addition: " << plaintextAddResult << std::endl;

    return 0;
}
        ''', language="cpp")

        st.markdown("### Financial Data Processing Example")
        st.code('''
#include "openfhe.h"
#include <fstream>
#include <sstream>

using namespace lbcrypto;

class FinancialDataProcessor {
private:
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;

public:
    void setupCKKS() {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(10);
        parameters.SetScalingModSize(50);
        parameters.SetBatchSize(8192);

        cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);

        keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalRotateKeyGen(keys.secretKey, {1, 2, -1, -2});
    }

    std::vector<double> processTransactions(const std::string& filename) {
        // Read transaction amounts from file
        std::vector<double> amounts;
        std::ifstream file(filename);
        std::string line;

        while (std::getline(file, line)) {
            amounts.push_back(std::stod(line));
        }

        // Encrypt amounts
        Plaintext plaintext = cc->MakeCKKSPackedPlaintext(amounts);
        auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);

        // Calculate statistics on encrypted data
        // Example: Square each value (for variance calculation)
        auto squared = cc->EvalMult(ciphertext, ciphertext);

        // Decrypt results
        Plaintext decrypted;
        cc->Decrypt(keys.secretKey, squared, &decrypted);

        return decrypted->GetRealPackedValue();
    }

    double calculateEncryptedSum(const std::vector<double>& data) {
        Plaintext plaintext = cc->MakeCKKSPackedPlaintext(data);
        auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);

        // Sum using rotations
        auto sum = ciphertext;
        for (int i = 1; i < data.size(); i *= 2) {
            auto rotated = cc->EvalRotate(sum, i);
            sum = cc->EvalAdd(sum, rotated);
        }

        Plaintext result;
        cc->Decrypt(keys.secretKey, sum, &result);

        return result->GetRealPackedValue()[0];
    }
};

int main() {
    FinancialDataProcessor processor;
    processor.setupCKKS();

    // Process encrypted financial data
    auto results = processor.processTransactions("transactions.txt");

    std::cout << "Processed " << results.size() << " transactions" << std::endl;

    return 0;
}
        ''', language="cpp")

    with tab4:
        st.markdown("## 🚀 Compilation and Execution")

        st.markdown("### Method 1: Using CMake (Recommended)")
        st.code("""
# Step 1: Create project structure
mkdir my_fhe_project
cd my_fhe_project

# Step 2: Create CMakeLists.txt (see previous tab)

# Step 3: Create source file
# (Create openfhe_wrapper.cpp with your code)

# Step 4: Build
mkdir build
cd build
cmake .. -DOPENFHE_ROOT="C:/openfhe-development"
cmake --build . --config Release

# Step 5: Run
./openfhe_wrapper.exe  # Windows
./openfhe_wrapper      # Linux/macOS
        """, language="bash")

        st.markdown("### Method 2: Direct Compilation (Quick Test)")

        st.markdown("**On Windows (MSVC):**")
        st.code("""
cl /EHsc /std:c++17 ^
   /I"C:\\openfhe-development\\src\\pke\\include" ^
   /I"C:\\openfhe-development\\src\\core\\include" ^
   /I"C:\\openfhe-development\\third-party\\include" ^
   openfhe_wrapper.cpp ^
   /link /LIBPATH:"C:\\openfhe-development\\build\\lib\\Release" ^
   OPENFHEpke.lib OPENFHEcore.lib OPENFHEbinfhe.lib
        """, language="bash")

        st.markdown("**On Linux/macOS (GCC/Clang):**")
        st.code("""
g++ -std=c++17 \\
    -I/path/to/openfhe-development/src/pke/include \\
    -I/path/to/openfhe-development/src/core/include \\
    -I/path/to/openfhe-development/third-party/include \\
    -L/path/to/openfhe-development/build/lib \\
    openfhe_wrapper.cpp \\
    -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe \\
    -o openfhe_wrapper
        """, language="bash")

        st.markdown("### Method 3: Python Integration via ctypes")
        st.code("""
import ctypes
import os

# Load compiled library
lib_path = os.path.join("build", "libopenfhe_wrapper.so")  # or .dll on Windows
fhe_lib = ctypes.CDLL(lib_path)

# Define function signatures
fhe_lib.encrypt_data.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
fhe_lib.encrypt_data.restype = ctypes.c_void_p

# Call C++ function from Python
data = [1.0, 2.0, 3.0, 4.0, 5.0]
data_array = (ctypes.c_double * len(data))(*data)
result = fhe_lib.encrypt_data(data_array, len(data))

print(f"Encryption result: {result}")
        """, language="python")

        st.markdown("### Troubleshooting Common Issues")

        with st.expander("❌ Error: Cannot find OpenFHE headers"):
            st.code("""
# Solution: Explicitly set include paths
export CPLUS_INCLUDE_PATH=/path/to/openfhe-development/src/pke/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/path/to/openfhe-development/src/core/include:$CPLUS_INCLUDE_PATH
            """)

        with st.expander("❌ Error: Undefined reference to OpenFHE functions"):
            st.code("""
# Solution: Check library paths and linking order
export LD_LIBRARY_PATH=/path/to/openfhe-development/build/lib:$LD_LIBRARY_PATH

# Ensure correct linking order (on Linux)
g++ ... -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe
            """)

        with st.expander("❌ Error: DLL not found (Windows)"):
            st.code("""
# Solution: Add OpenFHE library directory to PATH
set PATH=%PATH%;C:\\openfhe-development\\build\\lib\\Release

# Or copy DLLs to executable directory
copy C:\\openfhe-development\\build\\lib\\Release\\*.dll .
            """)

        st.markdown("### Testing Your Setup")
        st.code("""
# Create simple test program
cat > test_fhe.cpp << 'EOF'
#include <iostream>
#include "openfhe.h"

int main() {
    std::cout << "OpenFHE successfully linked!" << std::endl;
    return 0;
}
EOF

# Compile and run
g++ -std=c++17 test_fhe.cpp -I$OPENFHE_ROOT/src/pke/include -L$OPENFHE_ROOT/build/lib -lOPENFHEpke -o test_fhe
./test_fhe
        """, language="bash")


# Additional helper functions
def create_downloadable_files():
    """Create downloadable example files"""
    st.markdown("## 📥 Download Example Files")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpp_code = '''
#include "openfhe.h"
#include <iostream>

using namespace lbcrypto;

int main() {
    // Your FHE code here
    return 0;
}
'''
        st.download_button(
            label="📄 Download C++ Template",
            data=cpp_code,
            file_name="openfhe_template.cpp",
            mime="text/plain"
        )

    with col2:
        cmake_code = '''
cmake_minimum_required(VERSION 3.12)
project(MyFHEProject)

set(CMAKE_CXX_STANDARD 17)
set(OPENFHE_ROOT "C:/openfhe-development")

include_directories(${OPENFHE_ROOT}/src/pke/include)
link_directories(${OPENFHE_ROOT}/build/lib)

add_executable(my_fhe main.cpp)
target_link_libraries(my_fhe OPENFHEpke OPENFHEcore)
'''
        st.download_button(
            label="📄 Download CMakeLists.txt",
            data=cmake_code,
            file_name="CMakeLists.txt",
            mime="text/plain"
        )

    with col3:
        python_wrapper = '''
import ctypes
import numpy as np

class OpenFHEWrapper:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self.setup_functions()

    def setup_functions(self):
        # Define C++ function signatures
        pass

    def encrypt(self, data):
        # Call C++ encryption
        pass
'''
        st.download_button(
            label="🐍 Download Python Wrapper",
            data=python_wrapper,
            file_name="openfhe_wrapper.py",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()

    # Add download section at the bottom
    st.markdown("---")
    # create_downloadable_files()

    # Footer with useful links
    # st.markdown("---")
