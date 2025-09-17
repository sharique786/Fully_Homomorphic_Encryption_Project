FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl \
    libgmp-dev libmpfr-dev libssl-dev \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build OpenFHE from source
RUN git clone https://github.com/openfheorg/openfhe-development.git /tmp/openfhe \
    && cd /tmp/openfhe \
    && mkdir build && cd build \
    && cmake .. -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd / && rm -rf /tmp/openfhe

# Ensure OpenFHE shared libraries are discoverable
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py ./

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]