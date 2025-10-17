# Multi-stage Dockerfile for FHE Financial Data Processor
# Base: RHEL/Rocky Linux 9

# ============================================================
# Stage 1: Build OpenFHE
# ============================================================
FROM rockylinux:9 AS openfhe-builder

# Install build dependencies
RUN dnf install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    git \
    autoconf \
    automake \
    libtool \
    python3 \
    python3-devel \
    && dnf clean all

# Set working directory
WORKDIR /opt

# Clone and build OpenFHE
RUN git clone https://github.com/openfheorg/openfhe-development.git && \
    cd openfhe-development && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local/openfhe && \
    make -j$(nproc) && \
    make install

# ============================================================
# Stage 2: Application Runtime
# ============================================================
FROM rockylinux:9

LABEL maintainer="FHE Financial Processor"
LABEL description="Fully Homomorphic Encryption Application with OpenFHE"
LABEL version="1.0"

# Install runtime dependencies
RUN dnf install -y \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    gcc-c++ \
    make \
    && dnf clean all

# Copy OpenFHE from builder stage
COPY --from=openfhe-builder /usr/local/openfhe /usr/local/openfhe
COPY --from=openfhe-builder /opt/openfhe-development/src /usr/local/openfhe/src

# Set environment variables for OpenFHE (fix the undefined variable warnings)
ENV OPENFHE_ROOT=/usr/local/openfhe \
    LD_LIBRARY_PATH=/usr/local/openfhe/lib \
    PATH=/usr/local/openfhe/bin:${PATH} \
    PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3.11 install --no-cache-dir --upgrade pip && \
    pip3.11 install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY config.py .
COPY utils/ ./utils/
COPY fhe/ ./fhe/
COPY ui/ ./ui/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/temp && \
    chmod -R 755 /app

# Set user (non-root for security) - do this before changing ownership
RUN useradd -m -u 1000 fheuser && \
    chown -R fheuser:fheuser /app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Switch to non-root user
USER fheuser

# Run the application
CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.serverAddress=0.0.0.0", \
     "--browser.gatherUsageStats=false"]