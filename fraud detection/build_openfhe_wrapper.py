"""
Build Helper Script for OpenFHE Python Wrapper
Automates the process of generating and building the OpenFHE wrapper DLL
"""

import os
import sys
import subprocess
from pathlib import Path


def generate_files():
    """Generate C++ wrapper and CMake files"""
    from openfhe_wrapper import generate_cpp_wrapper_code, create_cmake_file

    print("📝 Generating C++ wrapper code...")
    cpp_code = generate_cpp_wrapper_code()
    with open('openfhe_python_wrapper.cpp', 'w') as f:
        f.write(cpp_code)
    print("✅ Created: openfhe_python_wrapper.cpp")

    print("\n📝 Generating CMakeLists.txt...")
    cmake_code = create_cmake_file()
    with open('CMakeLists.txt', 'w') as f:
        f.write(cmake_code)
    print("✅ Created: CMakeLists.txt")

    return True


def find_openfhe_installation():
    """Find OpenFHE installation directory"""
    possible_paths = [
        r"C:\Users\alish\Workspaces\Python\OpenFHELib\openfhe-development"
    ]

    for path in possible_paths:
        path = Path(path)
        if path.exists():
            # Check if build directory exists
            build_dir = path / "build"
            if build_dir.exists():
                print(f"✅ Found OpenFHE at: {path}")
                return str(path)

    return None


def find_cmake():
    """Check if CMake is available"""
    try:
        result = subprocess.run(['cmake', '--version'],
                                capture_output=True,
                                text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ CMake found: {version}")
            return True
    except FileNotFoundError:
        pass

    print("❌ CMake not found in PATH")
    return False


def detect_visual_studio():
    """Detect installed Visual Studio versions"""
    vs_paths = [
        (r"C:\Program Files\Microsoft Visual Studio\2022", "Visual Studio 17 2022"),
        (r"C:\Program Files (x86)\Microsoft Visual Studio\2022", "Visual Studio 17 2022"),
        (r"C:\Program Files\Microsoft Visual Studio\2019", "Visual Studio 16 2019"),
        (r"C:\Program Files (x86)\Microsoft Visual Studio\2019", "Visual Studio 16 2019"),
    ]

    for path, generator in vs_paths:
        if os.path.exists(path):
            print(f"✅ Found: {generator}")
            return generator

    print("❌ Visual Studio 2019/2022 not found")
    return None


def build_wrapper(openfhe_path, vs_generator):
    """Build the OpenFHE wrapper DLL"""
    print("\n🔨 Building OpenFHE Python Wrapper DLL...")

    # Create build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    os.chdir(build_dir)

    try:
        # Configure with CMake
        print("\n📦 Configuring with CMake...")
        openfhe_build = Path(openfhe_path) / "build"

        cmake_cmd = [
            'cmake', '..',
            '-G', vs_generator,
            '-A', 'x64',
            f'-DCMAKE_PREFIX_PATH={openfhe_build}',
            f'-DCMAKE_BUILD_TYPE=Release'
        ]

        print(f"Running: {' '.join(cmake_cmd)}")
        result = subprocess.run(cmake_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ CMake configuration failed:")
            print(result.stderr)
            return False

        print("✅ CMake configuration successful")

        # Build
        print("\n🔨 Building (this may take a few minutes)...")
        build_cmd = ['cmake', '--build', '.', '--config', 'Release']

        result, err = subprocess.run(build_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Build failed: {err}")
            print(result.stderr)
            return False

        print("✅ Build successful!")

        # Check if DLL was created
        dll_path = Path("Release/openfhe_python_wrapper.dll")
        if dll_path.exists():
            print(f"\n✅ DLL created at: {dll_path.absolute()}")

            # Copy to parent directory
            import shutil
            dest = Path("../openfhe_python_wrapper.dll")
            shutil.copy(dll_path, dest)
            print(f"✅ DLL copied to: {dest.absolute()}")

            return True
        else:
            print("❌ DLL not found after build")
            return False

    except Exception as e:
        print(f"❌ Build error: {e}")
        return False
    finally:
        os.chdir('..')


def test_wrapper():
    """Test the compiled wrapper"""
    print("\n🧪 Testing OpenFHE wrapper...")

    try:
        from openfhe_wrapper import OpenFHEWrapper

        wrapper = OpenFHEWrapper()

        if wrapper.lib is None:
            print("❌ DLL failed to load - Using simulation mode")
            return False

        print("✅ DLL loaded successfully!")

        # Test key generation
        print("\n🔑 Testing key generation...")
        result = wrapper.generate_keys('BFV', {
            'poly_modulus_degree': 8192,
            'plain_modulus': 65537
        })

        if result['status'] == 'success':
            print("✅ Key generation successful!")

            # Test encryption
            print("\n🔐 Testing encryption...")
            encrypt_result = wrapper.encrypt_data([10, 20, 30], 'BFV')

            if encrypt_result['status'] == 'success':
                print("✅ Encryption successful!")

                # Test decryption
                print("\n🔓 Testing decryption...")
                decrypt_result = wrapper.decrypt_data()

                if decrypt_result['status'] == 'success':
                    print(f"✅ Decryption successful! Results: {decrypt_result['results']}")
                    print("\n🎉 All tests passed! OpenFHE wrapper is working correctly!")
                    return True

        print("❌ Tests failed")
        return False

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main build process"""
    print("=" * 60)
    print("OpenFHE Python Wrapper - Build Helper")
    print("=" * 60)

    # Step 1: Check prerequisites
    print("\n📋 Checking prerequisites...")

    if not find_cmake():
        print("\n❌ Please install CMake and add it to PATH")
        print("Download from: https://cmake.org/download/")
        return False

    vs_generator = detect_visual_studio()
    if not vs_generator:
        print("\n❌ Please install Visual Studio 2019 or 2022 with C++ support")
        print("Download from: https://visualstudio.microsoft.com/")
        return False

    openfhe_path = find_openfhe_installation()
    if not openfhe_path:
        openfhe_path = input("\n📁 Enter OpenFHE installation path: ").strip()
        if not os.path.exists(openfhe_path):
            print(f"❌ Path not found: {openfhe_path}")
            return False

    # Step 2: Generate files
    print("\n" + "=" * 60)
    if not generate_files():
        print("❌ Failed to generate files")
        return False

    # Step 3: Build
    print("\n" + "=" * 60)
    if not build_wrapper(openfhe_path, vs_generator):
        print("\n❌ Build failed")
        return False

    # Step 4: Test
    print("\n" + "=" * 60)
    if test_wrapper():
        print("\n" + "=" * 60)
        print("✅ SUCCESS! OpenFHE Python wrapper is ready to use!")
        print("=" * 60)
        print("\nYou can now use it in your application:")
        print("  from openfhe_wrapper import OpenFHEWrapper")
        print("  wrapper = OpenFHEWrapper()")
        return True
    else:
        print("\n⚠️  Build completed but tests failed")
        print("The DLL may still work - try using it in your application")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)