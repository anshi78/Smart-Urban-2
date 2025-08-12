# Deployment Guide

## Overview

This guide covers local development setup, production deployment options, and configuration for the Smart Urban Settlement Management System. The system is designed for easy deployment across different environments.

## Prerequisites

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Storage**: 2GB free space for models and data
- **Network**: Internet connection for package installation

### Software Dependencies
- Python 3.7+
- pip (Python package manager)
- Git (for version control)
- Virtual environment tool (venv/virtualenv)

---

## Local Development Setup

### 1. Environment Setup

#### Clone Repository
```bash
# Clone the repository
git clone <repository-url>
cd smart-urban-settlement

# Alternative: Download and extract ZIP
# wget <repository-zip-url>
# unzip smart-urban-settlement.zip
# cd smart-urban-settlement
```

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv myvenv

# Activate virtual environment
# Windows
myvenv\Scripts\activate

# macOS/Linux
source myvenv/bin/activate

# Verify activation (should show virtual environment path)
which python
```

#### Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### 2. Data Setup

#### Download/Prepare Datasets
```bash
# Create data directory if not exists
mkdir -p data

# Ensure dataset files are in place:
# data/air_aquality.csv
# data/traffic_congestion.csv  
# data/house_prediction.csv
# data/new_housinf_data.csv

# Check data files
ls -la data/
```

#### Update Data Paths (if needed)
```bash
# Edit source files to update hardcoded paths
# src/air_quality.py - line 12
# src/traffic_congestion.py - line 10
# src/house_price.py - line 11

# Use find/replace or sed command
sed -i 's|C:/Users/91860/OneDrive/Desktop/Smart Urban 2/||g' src/*.py
```

### 3. Model Training

#### Train All Models
```bash
# Train air quality model
python src/air_quality.py

# Train traffic congestion model  
python src/traffic_congestion.py

# Train housing price model
python src/house_price.py

# Verify model files created
ls -la models/
```

#### Model Files Generated
```
models/
├── air_quality_model.pkl          (~99MB)
├── label_encoder_country.pkl      (~2.7KB)
├── label_encoder_status.pkl       (~635B)
├── traffic_congestion_model.pkl   (~2.3MB)
├── housing_models.pkl             (~921B)
└── kmeans_model.pkl               (varies)
```

### 4. Static Files Setup

#### Create Required Directories
```bash
# Create static directories
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images

# Move any existing CSS files
# mv style.css static/css/ (if exists)
```

#### Verify Template Structure
```bash
# Ensure templates directory exists
ls -la templates/
# Should contain: dashboard.html, index.html
```

### 5. Run Application

#### Start Development Server
```bash
# Run Flask application
python app.py

# Alternative with environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
flask run

# Application should start on http://localhost:5000
```

#### Verify Application
```bash
# Test dashboard access
curl http://localhost:5000

# Test API endpoints
curl -X POST http://localhost:5000/predict/air_quality \
  -d "Country=India&Status=Good&year=2024&month=7&day=15"

curl -X POST http://localhost:5000/predict/traffic \
  -d "hour=17&day_of_week=4"

curl -X POST http://localhost:5000/predict/housing \
  -d "population_density=25000&average_income=75000"
```

---

## Production Deployment

### 1. Docker Deployment

#### Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/css static/js static/images

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

#### Create docker-compose.yml
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./static/images:/app/static/images
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

#### Build and Run
```bash
# Build Docker image
docker build -t smart-urban-settlement .

# Run with docker-compose
docker-compose up -d

# Check running containers
docker-compose ps

# View logs
docker-compose logs -f web
```

### 2. Cloud Deployment (Heroku)

#### Prepare for Heroku
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create Heroku app
heroku create smart-urban-settlement-app
```

#### Create Heroku Configuration Files

**Procfile**:
```
web: python app.py
```

**runtime.txt**:
```
python-3.9.20
```

**requirements.txt** (update if needed):
```
Flask==2.1.0
pandas==1.3.3
scikit-learn==0.24.2
joblib==1.1.0
matplotlib==3.4.3
gunicorn==20.1.0
```

#### Deploy to Heroku
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit"

# Add Heroku remote
heroku git:remote -a smart-urban-settlement-app

# Deploy
git push heroku main

# Open application
heroku open

# View logs
heroku logs --tail
```

### 3. AWS EC2 Deployment

#### Launch EC2 Instance
```bash
# Launch Ubuntu 20.04 LTS instance
# Security Group: Allow HTTP (80), HTTPS (443), SSH (22)
# Instance Type: t3.medium (recommended)
```

#### Server Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv nginx -y

# Clone repository
git clone <repository-url>
cd smart-urban-settlement

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Configure Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Create Gunicorn service file
sudo nano /etc/systemd/system/smart-urban.service
```

**Service Configuration**:
```ini
[Unit]
Description=Smart Urban Settlement Management
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/smart-urban-settlement
Environment="PATH=/home/ubuntu/smart-urban-settlement/venv/bin"
ExecStart=/home/ubuntu/smart-urban-settlement/venv/bin/gunicorn --workers 3 --bind unix:smart-urban.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
```

#### Configure Nginx
```nginx
# /etc/nginx/sites-available/smart-urban
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/smart-urban-settlement/smart-urban.sock;
    }

    location /static {
        alias /home/ubuntu/smart-urban-settlement/static;
    }
}
```

#### Start Services
```bash
# Enable and start services
sudo systemctl daemon-reload
sudo systemctl start smart-urban
sudo systemctl enable smart-urban

# Configure Nginx
sudo ln -s /etc/nginx/sites-available/smart-urban /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx

# Check status
sudo systemctl status smart-urban
sudo systemctl status nginx
```

### 4. Google Cloud Platform (GCP)

#### App Engine Deployment

**app.yaml**:
```yaml
runtime: python39

env_variables:
  FLASK_ENV: production

handlers:
- url: /static
  static_dir: static

- url: /.*
  script: auto

automatic_scaling:
  min_instances: 1
  max_instances: 10
```

#### Deploy to GCP
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Initialize gcloud
gcloud init

# Create project (if needed)
gcloud projects create smart-urban-settlement

# Set project
gcloud config set project smart-urban-settlement

# Enable App Engine
gcloud app create --region=us-central

# Deploy application
gcloud app deploy

# Open application
gcloud app browse
```

---

## Environment Configuration

### 1. Environment Variables

#### Development (.env file)
```bash
# .env
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key
DATA_PATH=./data
MODEL_PATH=./models
STATIC_PATH=./static
```

#### Production
```bash
# Production environment variables
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
export DATA_PATH=/app/data
export MODEL_PATH=/app/models
```

### 2. Application Configuration

#### Update app.py for Production
```python
import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.environ.get('FLASK_ENV') == 'development'

# Model paths
MODEL_PATH = os.environ.get('MODEL_PATH', 'models')
DATA_PATH = os.environ.get('DATA_PATH', 'data')

# Load models with error handling
try:
    air_quality_model = joblib.load(f'{MODEL_PATH}/air_quality_model.pkl')
    traffic_model = joblib.load(f'{MODEL_PATH}/traffic_congestion_model.pkl')
    housing_model = joblib.load(f'{MODEL_PATH}/housing_model.pkl')
    country_encoder = joblib.load(f'{MODEL_PATH}/label_encoder_country.pkl')
    status_encoder = joblib.load(f'{MODEL_PATH}/label_encoder_status.pkl')
except FileNotFoundError as e:
    print(f"Model loading error: {e}")
    print("Please ensure all model files are available.")

# Rest of your application code...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
```

---

## Security Configuration

### 1. Production Security

#### Flask Security Headers
```python
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)

# Security headers for production
if os.environ.get('FLASK_ENV') == 'production':
    Talisman(app, force_https=True)
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
```

#### Environment Variables Security
```bash
# Use environment variables for sensitive data
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(16))')
export DATABASE_URL=your-database-url
export API_KEYS=your-api-keys
```

### 2. Input Validation

#### Enhanced Form Validation
```python
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, validators

class AirQualityForm(FlaskForm):
    country = StringField('Country', [
        validators.DataRequired(),
        validators.Length(min=2, max=50)
    ])
    status = StringField('Status', [
        validators.DataRequired(),
        validators.AnyOf(['Good', 'Moderate', 'Poor'])
    ])
    year = IntegerField('Year', [
        validators.DataRequired(),
        validators.NumberRange(min=2020, max=2025)
    ])
    month = IntegerField('Month', [
        validators.DataRequired(),
        validators.NumberRange(min=1, max=12)
    ])
    day = IntegerField('Day', [
        validators.DataRequired(),
        validators.NumberRange(min=1, max=31)
    ])
```

---

## Performance Optimization

### 1. Caching Configuration

#### Redis Caching
```python
import redis
from flask_caching import Cache

# Configure Redis cache
cache_config = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}

app.config.from_mapping(cache_config)
cache = Cache(app)

# Cache predictions
@cache.memoize(timeout=300)  # 5 minutes
def cached_air_quality_prediction(country, status, year, month, day):
    # Your prediction logic here
    pass
```

#### Memory Caching
```python
from flask_caching import Cache

# Simple memory cache for development
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def get_prediction(model_type, **kwargs):
    # Cached prediction logic
    pass
```

### 2. Database Optimization

#### Model Persistence
```python
import sqlite3
import pickle

# Store predictions in database
def store_prediction(prediction_type, inputs, output):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (type, inputs, output, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    ''', (prediction_type, pickle.dumps(inputs), output))
    
    conn.commit()
    conn.close()
```

### 3. Model Loading Optimization

#### Lazy Loading
```python
class ModelManager:
    def __init__(self):
        self._models = {}
    
    def get_model(self, model_name):
        if model_name not in self._models:
            self._models[model_name] = joblib.load(f'models/{model_name}.pkl')
        return self._models[model_name]

model_manager = ModelManager()
```

---

## Monitoring and Logging

### 1. Application Logging

#### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/smart-urban.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Smart Urban Settlement startup')
```

### 2. Health Checks

#### Health Check Endpoint
```python
@app.route('/health')
def health_check():
    try:
        # Test model loading
        air_quality_model.predict([[1, 1, 2024, 7, 15]])
        return {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 500
```

### 3. Monitoring Integration

#### Prometheus Metrics
```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')
```

---

## Backup and Recovery

### 1. Data Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/smart-urban-$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup models
cp -r models/ $BACKUP_DIR/

# Backup data
cp -r data/ $BACKUP_DIR/

# Backup static files
cp -r static/ $BACKUP_DIR/

# Create tarball
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### 2. Model Versioning

#### Model Version Management
```python
import shutil
from datetime import datetime

def backup_model(model_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'models/backups/{model_name}_{timestamp}.pkl'
    
    # Create backup directory
    os.makedirs('models/backups', exist_ok=True)
    
    # Copy current model
    shutil.copy(f'models/{model_name}.pkl', backup_path)
    
    return backup_path
```

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Issue: FileNotFoundError for model files
# Solution: Train models first
python src/air_quality.py
python src/traffic_congestion.py  
python src/house_price.py
```

#### 2. Memory Issues
```bash
# Issue: Out of memory during model training
# Solution: Use smaller datasets or increase memory
# For large air quality model, ensure 8GB+ RAM
```

#### 3. Path Issues
```bash
# Issue: Hardcoded paths in source files
# Solution: Update paths in src/ files
sed -i 's|C:/Users/91860/OneDrive/Desktop/Smart Urban 2/||g' src/*.py
```

#### 4. Port Conflicts
```bash
# Issue: Port 5000 already in use
# Solution: Use different port
python app.py --port 8000

# Or set environment variable
export PORT=8000
python app.py
```

### Log Analysis

#### Check Application Logs
```bash
# View Flask logs
tail -f logs/smart-urban.log

# Check system logs (Linux)
sudo journalctl -u smart-urban -f

# Docker logs
docker-compose logs -f web
```

### Performance Monitoring

#### Resource Usage
```bash
# Monitor CPU/Memory usage
htop
free -h
df -h

# Docker resource usage
docker stats
```

---

## Maintenance

### 1. Regular Updates

#### Dependency Updates
```bash
# Update packages
pip list --outdated
pip install --upgrade package-name

# Update requirements.txt
pip freeze > requirements.txt
```

#### Security Updates
```bash
# Check for security vulnerabilities
pip-audit

# Update security-related packages
pip install --upgrade flask werkzeug
```

### 2. Model Retraining

#### Scheduled Retraining
```bash
#!/bin/bash
# retrain_models.sh

echo "Starting model retraining..."

# Backup current models
python -c "
from datetime import datetime
import shutil
import os

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(f'models/backups/{timestamp}', exist_ok=True)
shutil.copytree('models/', f'models/backups/{timestamp}/', ignore=shutil.ignore_patterns('backups'))
"

# Retrain models
python src/air_quality.py
python src/traffic_congestion.py
python src/house_price.py

echo "Model retraining completed"
```

### 3. Database Maintenance

#### Clean Old Predictions
```sql
-- Delete predictions older than 30 days
DELETE FROM predictions 
WHERE timestamp < datetime('now', '-30 days');

-- Vacuum database
VACUUM;
```

---

This deployment guide provides comprehensive instructions for setting up and maintaining the Smart Urban Settlement Management System across different environments. Follow the appropriate section based on your deployment target and requirements.