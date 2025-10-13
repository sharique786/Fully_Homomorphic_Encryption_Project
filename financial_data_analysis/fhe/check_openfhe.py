"""
OpenFHE Installation Checker
Run this script to diagnose OpenFHE installation issues
"""

import os
import sys


def check_openfhe_installation():
    print("=" * 60)
    print("OpenFHE Installation Checker")
    print("=" * 60)
    print()

    # Paths to check
    openfhe_path = r"C:\Program Files (x86)\OpenFHE"
    build_path = r"C:\Users\alish\Workspaces\Python\openfhe-development"

    paths_to_check = [
        openfhe_path,
        build_path
    ]

    print("📂 Checking OpenFHE directories...")
    print()

    found_libs = []

    for base_path in paths_to_check:
        if os.path.exists(base_path):
            print(f"✅ Found: {base_path}")

            # Check lib directory
            lib_dirs = ["lib", "bin", os.path.join("lib", "Release"),
                        os.path.join("lib", "Debug"), os.path.join("bin", "Release")]

            for lib_dir in lib_dirs:
                full_lib_path = os.path.join(base_path, lib_dir)
                if os.path.exists(full_lib_path):
                    print(f"  📁 {lib_dir}/ exists")

                    # List DLL files
                    try:
                        files = os.listdir(full_lib_path)
                        dll_files = [f for f in files if f.endswith('.dll') or f.endswith('.so')]

                        if dll_files:
                            print(f"     Found {len(dll_files)} library files:")
                            for dll in dll_files:
                                print(f"       • {dll}")
                                if 'OPENFHE' in dll.upper():
                                    found_libs.append(os.path.join(full_lib_path, dll))
                    except Exception as e:
                        print(f"     ⚠️ Error reading directory: {e}")
        else:
            print(f"❌ Not found: {base_path}")

    print()
    print("=" * 60)

    # Check for required DLLs
    print("🔍 Checking for required OpenFHE libraries...")
    print()

    required_libs = [
        "OPENFHEcore.dll",
        "OPENFHEpke.dll",
        "OPENFHEbinfhe.dll"
    ]

    missing_libs = []

    for lib in required_libs:
        found = False
        for found_lib in found_libs:
            if lib.lower() in found_lib.lower():
                print(f"✅ {lib}: Found at {found_lib}")
                found = True
                break

        if not found:
            print(f"❌ {lib}: NOT FOUND")
            missing_libs.append(lib)

    print()
    print("=" * 60)

    # Check system dependencies
    print("🔧 Checking system dependencies...")
    print()

    try:
        import ctypes
        # Try to load common dependencies
        deps = {
            "msvcr120.dll": "Visual C++ 2013 Redistributable",
            "msvcp140.dll": "Visual C++ 2015-2019 Redistributable",
            "vcruntime140.dll": "Visual C++ 2015-2019 Redistributable"
        }

        for dll, name in deps.items():
            try:
                ctypes.CDLL(dll)
                print(f"✅ {dll} ({name}): Available")
            except:
                print(f"⚠️ {dll} ({name}): Not found or cannot load")
    except Exception as e:
        print(f"⚠️ Error checking dependencies: {e}")

    print()
    print("=" * 60)

    # Recommendations
    print("💡 Recommendations:")
    print()

    if missing_libs:
        print("❌ MISSING LIBRARIES DETECTED")
        print()
        print("The following libraries are missing:")
        for lib in missing_libs:
            print(f"   • {lib}")
        print()
        print("Solutions:")
        print("   1. Rebuild OpenFHE from source")
        print("   2. Ensure all build configurations completed successfully")
        print("   3. Check CMake build logs for errors")
        print("   4. Verify all dependencies are installed")
        print()
    else:
        print("✅ All required libraries found!")
        print()
        print("If you're still having issues:")
        print("   1. Install Visual C++ Redistributable:")
        print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   2. Add OpenFHE paths to system PATH:")
        for found_lib in found_libs:
            lib_dir = os.path.dirname(found_lib)
            print(f"      {lib_dir}")
        print("   3. Restart your terminal/IDE after setting PATH")
        print()

    print("📚 For more help, visit:")
    print("   https://openfhe-development.readthedocs.io/")
    print()
    print("🔄 The application will run in SIMULATION MODE if libraries cannot load.")
    print("   Simulation mode provides the same interface but uses mock encryption.")
    print()


if __name__ == "__main__":
    check_openfhe_installation()
    input("\nPress Enter to exit...")