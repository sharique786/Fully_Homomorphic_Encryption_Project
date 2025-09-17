set -e

echo "🚀 Deploying FHE Financial Data Processor..."

# Function to deploy to different environments
deploy_local() {
    echo "🏠 Deploying locally with Docker Compose..."
    docker-compose up -d fhe-processor
    echo "✅ Local deployment complete!"
    echo "🌐 Access the application at: http://localhost:8501"
}

deploy_dev() {
    echo "🛠️  Deploying development environment..."
    docker-compose up -d fhe-processor-dev
    echo "✅ Development deployment complete!"
    echo "🌐 Access the application at: http://localhost:8502"
    echo "📓 Jupyter available at: http://localhost:8888"
}

deploy_openfhe() {
    echo "🔐 Deploying with OpenFHE support..."
    docker-compose up -d fhe-processor-openfhe
    echo "✅ OpenFHE deployment complete!"
    echo "🌐 Access the application at: http://localhost:8503"
}

deploy_production() {
    echo "🏭 Deploying production environment with Nginx..."
    docker-compose up -d
    echo "✅ Production deployment complete!"
    echo "🌐 Access the application at: http://localhost"
}

deploy_gcp() {
    echo "☁️  Deploying to Google Cloud Platform..."

    # Set your GCP project ID
    PROJECT_ID=${GCP_PROJECT_ID:-"your-gcp-project"}

    # Build and push to Google Container Registry
    docker tag fhe-processor:latest gcr.io/$PROJECT_ID/fhe-processor:latest
    docker push gcr.io/$PROJECT_ID/fhe-processor:latest

    # Deploy to Cloud Run
    gcloud run deploy fhe-processor \
        --image gcr.io/$PROJECT_ID/fhe-processor:latest \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --port 8501 \
        --memory 2Gi \
        --cpu 1

    echo "✅ GCP deployment complete!"
}

deploy_aws() {
    echo "☁️  Deploying to AWS ECS..."

    # This is a simplified example - you'll need to configure ECS properly
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

    docker tag fhe-processor:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fhe-processor:latest
    docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fhe-processor:latest

    echo "✅ AWS deployment initiated!"
}

# Parse command line arguments
case "$1" in
    "local"|"")
        deploy_local
        ;;
    "dev"|"development")
        deploy_dev
        ;;
    "openfhe")
        deploy_openfhe
        ;;
    "prod"|"production")
        deploy_production
        ;;
    "gcp")
        deploy_gcp
        ;;
    "aws")
        deploy_aws
        ;;
    "stop")
        echo "🛑 Stopping all services..."
        docker-compose down
        echo "✅ All services stopped!"
        ;;
    "logs")
        echo "📋 Showing logs..."
        docker-compose logs -f
        ;;
    "status")
        echo "📊 Service status:"
        docker-compose ps
        ;;
    *)
        echo "Usage: $0 {local|dev|openfhe|prod|gcp|aws|stop|logs|status}"
        echo ""
        echo "Commands:"
        echo "  local     - Deploy locally (default)"
        echo "  dev       - Deploy development environment"
        echo "  openfhe   - Deploy with OpenFHE support"
        echo "  prod      - Deploy production with Nginx"
        echo "  gcp       - Deploy to Google Cloud Platform"
        echo "  aws       - Deploy to AWS ECS"
        echo "  stop      - Stop all services"
        echo "  logs      - Show service logs"
        echo "  status    - Show service status"
        exit 1
        ;;
esac