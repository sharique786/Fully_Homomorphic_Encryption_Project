#!/bin/bash
set -e

# Install dependencies
sudo yum groupinstall "Development Tools" -y || sudo apt-get update && sudo apt-get install -y build-essential cmake g++

# Clone OpenFHE if not exists
if [ ! -d "openfhe-development" ]; then
  git clone https://github.com/openfheorg/openfhe-development.git
fi

cd openfhe-development
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Build Python wrapper
cd ../..
if [ ! -d "openfhe-python" ]; then
  git clone https://github.com/openfheorg/openfhe-python.git
fi
cd openfhe-python
pip install .
