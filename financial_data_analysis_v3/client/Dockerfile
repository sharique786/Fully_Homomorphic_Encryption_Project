# Dockerfile for FHE Client Application (Streamlit)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY client.py .

# Create .streamlit directory and config
RUN mkdir -p ~/.streamlit && \
    echo '[server]' > ~/.streamlit/config.toml && \
    echo 'headless = true' >> ~/.streamlit/config.toml && \
    echo 'port = 8501' >> ~/.streamlit/config.toml && \
    echo 'address = "0.0.0.0"' >> ~/.streamlit/config.toml && \
    echo 'enableCORS = false' >> ~/.streamlit/config.toml && \
    echo 'enableXsrfProtection = false' >> ~/.streamlit/config.toml && \
    echo '' >> ~/.streamlit/config.toml && \
    echo '[browser]' >> ~/.streamlit/config.toml && \
    echo 'gatherUsageStats = false' >> ~/.streamlit/config.toml && \
    echo '' >> ~/.streamlit/config.toml && \
    echo '[theme]' >> ~/.streamlit/config.toml && \
    echo 'primaryColor = "#6366f1"' >> ~/.streamlit/config.toml && \
    echo 'backgroundColor = "#ffffff"' >> ~/.streamlit/config.toml && \
    echo 'secondaryBackgroundColor = "#f0f2f6"' >> ~/.streamlit/config.toml && \
    echo 'textColor = "#262730"' >> ~/.streamlit/config.toml && \
    echo 'font = "sans serif"' >> ~/.streamlit/config.toml

# Create cache directory
RUN mkdir -p /app/.streamlit/cache && \
    chown -R streamlit:streamlit /app ~/.streamlit

# Switch to non-root user
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "client.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]