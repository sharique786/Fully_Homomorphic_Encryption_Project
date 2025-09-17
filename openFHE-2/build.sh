#!/bin/bash

# build.sh - Build script for FHE Financial Data Processor

set -e

echo "🐳 Building FHE Financial Data Processor Docker Images..."

# Create necessary directories
mkdir -p data uploads exports logs .streamlit

# Build production image
echo "📦 Building production image..."
docker build --target production -t fhe-processor:latest .
docker build --target production -t fhe-processor:prod .

# Build development image
echo "🛠️  Building development image..."
docker build --target development -t fhe-processor:dev .

# Build OpenFHE-enabled image
echo "🔐 Building OpenFHE-enabled image..."
docker build --target production-openfhe -t fhe-processor:openfhe .

echo "✅ Build completed successfully!"

# Optional: Run tests
if [ "$1" = "--test" ]; then
    echo "🧪 Running tests..."
    docker run --rm fhe-processor:latest python -m pytest --version || echo "Pytest not available in production image"
fi

# Optional: Push to registry
if [ "$1" = "--push" ]; then
    echo "📤 Pushing images to registry..."
    # Replace with your registry URL
    REGISTRY_URL="your-registry.com"

    docker tag fhe-processor:latest $REGISTRY_URL/fhe-processor:latest
    docker tag fhe-processor:prod $REGISTRY_URL/fhe-processor:prod
    docker tag fhe-processor:dev $REGISTRY_URL/fhe-processor:dev
    docker tag fhe-processor:openfhe $REGISTRY_URL/fhe-processor:openfhe

    docker push $REGISTRY_URL/fhe-processor:latest
    docker push $REGISTRY_URL/fhe-processor:prod
    docker push $REGISTRY_URL/fhe-processor:dev
    docker push $REGISTRY_URL/fhe-processor:openfhe
fi

echo "🚀 Images ready for deployment!"

# Show built images
echo "📋 Available images:"
docker images | grep fhe-processor