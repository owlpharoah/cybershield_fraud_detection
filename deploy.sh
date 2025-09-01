#!/bin/bash

# Fraud Detection System Deployment Script
# Usage: ./deploy.sh [option]

echo "🚀 Fraud Detection System Deployment"
echo "====================================="

case $1 in
    "local")
        echo "📍 Starting local development server..."
        streamlit run app.py
        ;;
    
    "docker")
        echo "🐳 Deploying with Docker..."
        docker build -t fraud-detection-app .
        docker run -p 8501:8501 fraud-detection-app
        ;;
    
    "docker-compose")
        echo "🐳 Deploying with Docker Compose..."
        docker-compose up --build
        ;;
    
    "heroku")
        echo "☁️ Deploying to Heroku..."
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI not found. Please install it first."
            exit 1
        fi
        heroku create fraud-detection-$(date +%s)
        git add .
        git commit -m "Deploy fraud detection system"
        git push heroku main
        heroku open
        ;;
    
    "test")
        echo "🧪 Testing the model..."
        python test_model.py
        ;;
    
    "retrain")
        echo "🔄 Retraining the model..."
        python retrain_model.py
        ;;
    
    "setup")
        echo "⚙️ Setting up the environment..."
        pip install -r requirements.txt
        python retrain_model.py
        echo "✅ Setup complete!"
        ;;
    
    *)
        echo "Usage: ./deploy.sh [option]"
        echo ""
        echo "Options:"
        echo "  local          - Run locally with Streamlit"
        echo "  docker         - Deploy with Docker"
        echo "  docker-compose - Deploy with Docker Compose"
        echo "  heroku         - Deploy to Heroku"
        echo "  test           - Test the model"
        echo "  retrain        - Retrain the model"
        echo "  setup          - Initial setup"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh local"
        echo "  ./deploy.sh docker"
        echo "  ./deploy.sh heroku"
        ;;
esac
