#!/bin/bash

echo "🚀 Starting FHE Financial Data Processor locally..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build if image doesn't exist
if ! docker images | grep -q "fhe-processor"; then
    echo "📦 Building Docker image..."
    ./build.sh
fi

# Create necessary directories
mkdir -p data uploads exports logs

# Run the container
docker run -d \
    --name fhe-processor \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/exports:/app/exports \
    -v $(pwd)/logs:/app/logs \
    fhe-processor:latest

echo "✅ FHE Financial Data Processor is running!"
echo "🌐 Access the application at: http://localhost:8501"
echo ""
echo "📋 Useful commands:"
echo "  docker logs fhe-processor        - View logs"
echo "  docker stop fhe-processor        - Stop the application"
echo "  docker rm fhe-processor          - Remove the container"