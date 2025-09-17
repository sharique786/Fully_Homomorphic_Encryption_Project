# build_openfhe_windows.ps1
Write-Host "This script will run MSYS2 commands to build OpenFHE and the Python wrapper."
Write-Host "Make sure MSYS2 (MINGW64) is installed. Open MSYS2 MinGW 64-bit shell and run this script from that shell or copy commands manually."


# Recommended MSYS2 packages (run inside MINGW64 shell):
# pacman -Syu
# pacman -S --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-python mingw-w64-x86_64-python-pip mingw-w64-x86_64-openmp


# Example commands you can run inside MINGW64 shell:
# 1. Update
# pacman -Syu
# (close and reopen shell if updated MSYS2 core packages)


# 2. Install build deps
# pacman -S --noconfirm base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-python mingw-w64-x86_64-python-pip mingw-w64-x86_64-openmp git


# 3. Clone OpenFHE
# git clone https://github.com/openfheorg/openfhe-development.git
# mkdir -p openfhe-development/build && cd openfhe-development/build
# cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
# make -j$(nproc)
# make install


# 4. Clone python wrapper and build
# git clone https://github.com/openfheorg/openfhe-python.git
# cd openfhe-python
# pip install wheel
# pip wheel . -w ../wheels
# pip install ../wheels/*.whl


Write-Host "If any step fails, inspect the error and ensure MSYS2 packages (toolchain, cmake, python) are installed.";