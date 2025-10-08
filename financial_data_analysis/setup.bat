@echo off
REM FHE Financial Processor - Windows Installation Script

echo ==================================
echo FHE Financial Processor Setup
echo ==================================
echo.

REM Check Python installation
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

python --version
echo [OK] Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists
    set /p recreate="Do you want to recreate it? (y/n): "
    if /i "%recreate%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo [OK] Virtual environment created
    )
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

REM Install requirements
echo Installing dependencies...
echo This may take a few minutes...
echo.

REM Install basic requirements
pip install streamlit pandas numpy plotly python-dateutil

REM Try to install TenSEAL
echo.
echo Installing TenSEAL...
pip install tenseal
if errorlevel 1 (
    echo [WARNING] TenSEAL installation failed
    echo You can try manual installation later with:
    echo   pip install tenseal --no-binary tenseal
    echo   or use conda: conda install -c conda-forge tenseal
) else (
    echo [OK] TenSEAL installed successfully
)
echo.

REM Create directory structure
echo Creating directory structure...
if not exist utils mkdir utils
if not exist fhe mkdir fhe
if not exist ui mkdir ui
type nul > utils\__init__.py
type nul > fhe\__init__.py
type nul > ui\__init__.py
echo [OK] Directory structure created
echo.

REM Check for OpenFHE
echo Checking for OpenFHE...
set OPENFHE_PATH=C:\Program Files (x86)\OpenFHE
if exist "%OPENFHE_PATH%" (
    echo [OK] OpenFHE installation directory found
) else (
    echo [WARNING] OpenFHE not found
    echo The application will run in simulation mode.
    echo To install OpenFHE, visit:
    echo   https://openfhe-development.readthedocs.io/
)
echo.

REM Final checks
echo Running final checks...
python -c "import streamlit, pandas, numpy, plotly" 2>nul
if errorlevel 1 (
    echo [ERROR] Some core packages missing
    pause
    exit /b 1
) else (
    echo [OK] All core packages installed
)
echo.

REM Create run script
echo Creating run script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo streamlit run main.py
echo pause
) > run.bat
echo [OK] Run script created
echo.

REM Summary
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To start the application:
echo   1. Run: run.bat
echo   Or manually:
echo      - Activate: venv\Scripts\activate.bat
echo      - Run: streamlit run main.py
echo.
echo For detailed instructions, see:
echo   - README.md
echo   - QUICKSTART.md
echo.
echo Happy encrypting! [lock emoji]
echo.
pause