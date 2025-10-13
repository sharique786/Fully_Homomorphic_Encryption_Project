@echo off
echo ========================================
echo OpenFHE DLL Dependency Checker
echo ========================================
echo.

set OPENFHE_PATH=C:\Program Files (x86)\OpenFHE\lib

echo Checking DLLs in: %OPENFHE_PATH%
echo.

cd "%OPENFHE_PATH%"

echo Listing all DLL files:
dir *.dll /b
echo.

echo ========================================
echo Checking dependencies with dumpbin...
echo ========================================
echo.

REM Check if dumpbin is available (part of Visual Studio)
where dumpbin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] dumpbin not found. Install Visual Studio Build Tools.
    echo.
    echo Alternative: Use Dependencies.exe
    echo Download from: https://github.com/lucasg/Dependencies/releases
    echo.
    goto :manual_check
)

echo Checking libOPENFHEcore.dll dependencies:
dumpbin /dependents libOPENFHEcore.dll
echo.

echo Checking libOPENFHEpke.dll dependencies:
dumpbin /dependents libOPENFHEpke.dll
echo.

echo Checking libOPENFHEbinfhe.dll dependencies:
dumpbin /dependents libOPENFHEbinfhe.dll
echo.

:manual_check
echo ========================================
echo Manual Checks
echo ========================================
echo.

echo Checking for Visual C++ Redistributables:
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Visual C++ 2015-2022 Redistributable found
) else (
    echo [MISSING] Visual C++ 2015-2022 Redistributable
    echo Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
)
echo.

echo ========================================
echo Quick Fixes
echo ========================================
echo.
echo 1. Install Visual C++ Redistributable:
echo    https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo 2. Add OpenFHE to PATH:
echo    setx PATH "%%PATH%%;%OPENFHE_PATH%"
echo.
echo 3. Use Dependencies.exe to see missing DLLs:
echo    https://github.com/lucasg/Dependencies/releases
echo.
echo 4. Run application - it will use simulation mode if needed
echo.

pause