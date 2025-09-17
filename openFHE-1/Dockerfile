# Use Ubuntu LTS as base
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive


# Install prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential cmake git wget ca-certificates python3-dev python3-pip python3-venv \
libgomp1 libomp-dev libssl-dev libbz2-dev liblzma-dev pkg-config curl unzip \
&& rm -rf /var/lib/apt/lists/*


# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python || true


# Create app dir
WORKDIR /workspace


# Install pip packages needed for building and running examples
RUN pip install --upgrade pip setuptools wheel
RUN pip install streamlit pandas numpy plotly scikit-learn


# Build OpenFHE from source
ARG OPENFHE_TAG=main
RUN git clone --depth 1 https://github.com/openfheorg/openfhe-development.git /workspace/openfhe-development && \
mkdir -p /workspace/openfhe-development/build && cd /workspace/openfhe-development/build && \
cmake -DCMAKE_BUILD_TYPE=Release .. && \
make -j"$(nproc)" && make install


# Build Python wrapper (openfhe-python)
RUN git clone --depth 1 https://github.com/openfheorg/openfhe-python.git /workspace/openfhe-python && \
cd /workspace/openfhe-python && \
pip wheel . --no-deps -w /workspace/wheels && \
pip install /workspace/wheels/*.whl


# Copy app files
COPY ./app_extended.py /workspace/app_extended.py
COPY ./requirements.txt /workspace/requirements.txt


# Install runtime requirements (if any extras)
RUN pip install -r /workspace/requirements.txt || true


# Expose Streamlit default port
EXPOSE 8501


CMD ["streamlit", "run", "app_extended.py", "--server.port=8501", "--server.address=0.0.0.0"]