"""
Alternative OpenFHE DLL Loading Script
Run this to test different loading methods and identify the issue
"""

import os
import sys
import ctypes
from pathlib import Path


def test_dll_loading():
    print("=" * 70)
    print("OpenFHE DLL Loading Tester")
    print("=" * 70)
    print()

    openfhe_path = r"C:\Program Files (x86)\OpenFHE\lib"

    # List of DLLs to try loading
    dlls = [
        "libOPENFHEcore.dll",
        "libOPENFHEpke.dll",
        "libOPENFHEbinfhe.dll"
    ]

    print(f"OpenFHE Path: {openfhe_path}")
    print()

    # Method 1: Add to PATH and use CDLL
    print("=" * 70)
    print("METHOD 1: Add to PATH + ctypes.CDLL")
    print("=" * 70)

    os.environ['PATH'] = openfhe_path + os.pathsep + os.environ.get('PATH', '')

    for dll in dlls:
        dll_path = os.path.join(openfhe_path, dll)
        if os.path.exists(dll_path):
            print(f"\n📦 Testing: {dll}")
            print(f"   Path: {dll_path}")
            print(f"   Size: {os.path.getsize(dll_path)} bytes")

            try:
                lib = ctypes.CDLL(dll_path)
                print(f"   ✅ SUCCESS with CDLL!")
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}")

    # Method 2: Use WinDLL (Windows-specific)
    print("\n" + "=" * 70)
    print("METHOD 2: ctypes.WinDLL (Windows calling convention)")
    print("=" * 70)

    for dll in dlls:
        dll_path = os.path.join(openfhe_path, dll)
        if os.path.exists(dll_path):
            print(f"\n📦 Testing: {dll}")

            try:
                lib = ctypes.WinDLL(dll_path)
                print(f"   ✅ SUCCESS with WinDLL!")
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}")

    # Method 3: Use add_dll_directory (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        print("\n" + "=" * 70)
        print("METHOD 3: os.add_dll_directory + CDLL")
        print("=" * 70)

        try:
            os.add_dll_directory(openfhe_path)
            print(f"✅ Added DLL directory: {openfhe_path}")

            for dll in dlls:
                dll_path = os.path.join(openfhe_path, dll)
                if os.path.exists(dll_path):
                    print(f"\n📦 Testing: {dll}")

                    try:
                        lib = ctypes.CDLL(dll_path)
                        print(f"   ✅ SUCCESS!")
                    except Exception as e:
                        print(f"   ❌ FAILED: {str(e)}")
        except Exception as e:
            print(f"❌ Failed to add DLL directory: {e}")

    # Method 4: LoadLibrary (Windows API)
    print("\n" + "=" * 70)
    print("METHOD 4: Windows LoadLibrary")
    print("=" * 70)

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        for dll in dlls:
            dll_path = os.path.join(openfhe_path, dll)
            if os.path.exists(dll_path):
                print(f"\n📦 Testing: {dll}")

                handle = kernel32.LoadLibraryW(dll_path)
                if handle:
                    print(f"   ✅ SUCCESS! Handle: {handle}")
                    kernel32.FreeLibrary(handle)
                else:
                    error = ctypes.get_last_error()
                    print(f"   ❌ FAILED: Error code {error}")

                    # Get error message
                    FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
                    buffer = ctypes.create_unicode_buffer(256)
                    kernel32.FormatMessageW(
                        FORMAT_MESSAGE_FROM_SYSTEM,
                        None,
                        error,
                        0,
                        buffer,
                        256,
                        None
                    )
                    print(f"   Error: {buffer.value}")
    except Exception as e:
        print(f"❌ LoadLibrary test failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)
    print()

    print("If ALL methods failed, the issue is:")
    print("  ❌ Missing system dependencies (Visual C++ Runtime)")
    print()
    print("SOLUTION:")
    print("  1. Download and install:")
    print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("  2. Restart your computer")
    print()
    print("  3. Run this test again")
    print()

    print("If at least ONE method worked:")
    print("  ✅ DLLs are loadable!")
    print("  → Update openfhe_wrapper.py to use the working method")
    print()

    print("For detailed DLL dependencies, use:")
    print("  • Dependencies.exe: https://github.com/lucasg/Dependencies/releases")
    print("  • dumpbin /dependents libOPENFHEcore.dll")
    print()

    print("=" * 70)
    print("ALTERNATIVE: Use Simulation Mode")
    print("=" * 70)
    print()
    print("The application already runs in simulation mode if OpenFHE fails.")
    print("This is PERFECTLY FINE for:")
    print("  ✅ Development")
    print("  ✅ Testing")
    print("  ✅ Demonstrations")
    print("  ✅ Learning FHE concepts")
    print()
    print("You only need real OpenFHE for:")
    print("  • Production cryptographic operations")
    print("  • Real homomorphic encryption")
    print()


if __name__ == "__main__":
    test_dll_loading()
    input("\nPress Enter to exit...")