#!/bin/bash

# FHE Financial Processor - Installation Script
# For Linux/Mac systems

echo "=================================="
echo "FHE Financial Processor Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓ Python $python_version detected${NC}"
else
    echo -e "${RED}✗ Python 3.8 or higher required${NC}"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}! Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

# Install basic requirements first
pip install streamlit pandas numpy plotly python-dateutil

# Try to install TenSEAL
echo ""
echo "Installing TenSEAL..."
if pip install tenseal; then
    echo -e "${GREEN}✓ TenSEAL installed successfully${NC}"
else
    echo -e "${YELLOW}! TenSEAL installation failed${NC}"
    echo "You can try manual installation later with:"
    echo "  pip install tenseal --no-binary tenseal"
    echo "  or: conda install -c conda-forge tenseal"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p utils fhe ui
touch utils/__init__.py fhe/__init__.py ui/__init__.py
echo -e "${GREEN}✓ Directory structure created${NC}"

# Check for OpenFHE
echo ""
echo "Checking for OpenFHE..."
openfhe_path="/usr/local/lib/libOPENFHEcore.so"
if [ -f "$openfhe_path" ]; then
    echo -e "${GREEN}✓ OpenFHE library found${NC}"
else
    echo -e "${YELLOW}! OpenFHE library not found${NC}"
    echo "The application will run in simulation mode."
    echo "To install OpenFHE, visit:"
    echo "  https://openfhe-development.readthedocs.io/"
fi

# Final checks
echo ""
echo "Running final checks..."

# Check if all required packages are installed
python3 << EOF
import sys
try:
    import streamlit
    import pandas
    import numpy
    import plotly
    print("${GREEN}✓ All core packages installed${NC}")
    sys.exit(0)
except ImportError as e:
    print("${RED}✗ Missing package:", str(e), "${NC}")
    sys.exit(1)
EOF

# Create run script
echo ""
echo "Creating run script..."
cat > run.sh << 'RUNEOF'
#!/bin/bash
source venv/bin/activate
streamlit run main.py
RUNEOF
chmod +x run.sh
echo -e "${GREEN}✓ Run script created${NC}"

# Summary
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run main.py"
echo "  Or simply: ./run.sh"
echo ""
echo "For detailed instructions, see:"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo ""
echo -e "${GREEN}Happy encrypting! 🔐${NC}"