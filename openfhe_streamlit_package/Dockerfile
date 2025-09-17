FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y build-essential cmake g++ git wget curl && rm -rf /var/lib/apt/lists/*

# Install OpenFHE Python wrapper
RUN pip install --no-cache-dir openfhe streamlit pandas matplotlib seaborn

# Copy app
WORKDIR /app
COPY app_extended.py /app/app_extended.py
COPY requirements.txt /app/requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app_extended.py", "--server.port=8501", "--server.headless=true"]
