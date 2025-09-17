@echo off
REM Windows build helper for OpenFHE using MSYS2

where pacman >nul 2>nul
if %errorlevel% neq 0 (
    echo "MSYS2 not found. Please install from https://www.msys2.org/"
    exit /b 1
)

echo "Launching MSYS2 build environment..."
C:\msys64\usr\bin\bash -lc "./build_openfhe.sh"
