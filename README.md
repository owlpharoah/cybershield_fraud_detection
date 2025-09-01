# 🔍 Financial Fraud Detection System

A machine learning-powered web application for detecting fraudulent financial transactions in real-time.

## 🎯 Features

- **Real-time Fraud Detection**: Analyze transactions instantly
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Advanced Feature Engineering**: 10+ engineered features for better detection
- **Risk Assessment**: Probability-based fraud scoring
- **Transaction Analysis**: Detailed insights and risk factors

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fraud-detection-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if needed)
   ```bash
   python retrain_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Local: http://localhost:8501
   - Network: http://your-ip:8501

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Deploy!

**Advantages:**
- ✅ Free hosting
- ✅ Automatic deployments
- ✅ No server management
- ✅ Built-in CI/CD

### Option 2: Heroku

**Steps:**
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker Deployment

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Deploy:**
```bash
docker build -t fraud-detection-app .
docker run -p 8501:8501 fraud-detection-app
```

### Option 4: AWS/GCP/Azure

**For production environments:**
- Use EC2 (AWS) or Compute Engine (GCP)
- Set up load balancers
- Configure auto-scaling
- Use managed databases for model storage

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 30+ engineered features
- **Performance**: 88.76% accuracy
- **Training Data**: 50,000 synthetic transactions
- **Fraud Detection Rate**: Optimized for low false positives

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set for production
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Retraining
```bash
# Retrain with current scikit-learn version
python retrain_model.py

# Test the model
python test_model.py
```

## 📁 Project Structure

```
fraud-detection-system/
├── app.py                      # Main Streamlit application
├── retrain_model.py            # Model training script
├── test_model.py               # Model testing script
├── requirements.txt            # Python dependencies
├── fraud_detection_model_final.pkl  # Trained model
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── DEPLOYMENT_GUIDE.md        # Detailed deployment guide
└── README.md                  # This file
```

## 🛠️ Development

### Adding New Features
1. Modify `create_features()` in `app.py`
2. Update `retrain_model.py` with same features
3. Retrain the model
4. Test with `test_model.py`

### Customizing the UI
- Edit CSS in `app.py` (lines 18-45)
- Modify sidebar layout
- Add new input fields
- Customize visualizations

## 🔒 Security Considerations

### Production Checklist
- [ ] Use HTTPS in production
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Secure model file access
- [ ] Monitor application logs
- [ ] Set up error tracking

### Model Security
- [ ] Version control your models
- [ ] Implement model drift detection
- [ ] Regular model retraining
- [ ] A/B testing for new models

## 📈 Monitoring & Maintenance

### Health Checks
```bash
# Test model loading
python -c "import joblib; model = joblib.load('fraud_detection_model_final.pkl'); print('✅ Model OK')"

# Test application
curl http://localhost:8501/_stcore/health
```

### Performance Monitoring
- Monitor response times
- Track prediction accuracy
- Watch for model drift
- Monitor resource usage

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error:**
```bash
# Solution: Retrain model
python retrain_model.py
```

**Port Already in Use:**
```bash
# Solution: Use different port
streamlit run app.py --server.port=8502
```

**Dependencies Missing:**
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the deployment guide
3. Test with the provided scripts
4. Open an issue on GitHub

---

**Built with ❤️ using Streamlit and scikit-learn**
