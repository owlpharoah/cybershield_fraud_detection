@echo off
REM Fraud Detection System Deployment Script for Windows
REM Usage: deploy.bat [option]

echo 🚀 Fraud Detection System Deployment
echo =====================================

if "%1"=="local" (
    echo 📍 Starting local development server...
    streamlit run app.py
    goto :eof
)

if "%1"=="docker" (
    echo 🐳 Deploying with Docker...
    docker build -t fraud-detection-app .
    docker run -p 8501:8501 fraud-detection-app
    goto :eof
)

if "%1"=="docker-compose" (
    echo 🐳 Deploying with Docker Compose...
    docker-compose up --build
    goto :eof
)

if "%1"=="test" (
    echo 🧪 Testing the model...
    python test_model.py
    goto :eof
)

if "%1"=="retrain" (
    echo 🔄 Retraining the model...
    python retrain_model.py
    goto :eof
)

if "%1"=="setup" (
    echo ⚙️ Setting up the environment...
    pip install -r requirements.txt
    python retrain_model.py
    echo ✅ Setup complete!
    goto :eof
)

echo Usage: deploy.bat [option]
echo.
echo Options:
echo   local          - Run locally with Streamlit
echo   docker         - Deploy with Docker
echo   docker-compose - Deploy with Docker Compose
echo   test           - Test the model
echo   retrain        - Retrain the model
echo   setup          - Initial setup
echo.
echo Examples:
echo   deploy.bat local
echo   deploy.bat docker
echo   deploy.bat test
