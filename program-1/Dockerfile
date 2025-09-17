# Dockerfile (starter)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential cmake git wget libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Try pip install — may succeed if openfhe wheel is available for your platform
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# If pip install openfhe fails, you must install OpenFHE C++ (see openfhe docs) and then build openfhe-python
COPY . /app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
