# Development Guide - Smart Urban Settlement Management System

## 🚀 Getting Started

### Prerequisites

- **Python 3.7+** - Latest stable version recommended
- **Git** - For version control
- **pip** - Python package manager
- **Virtual Environment** - For dependency isolation

### Development Environment Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd smart-urban-settlement
   ```

2. **Create Virtual Environment**
   ```bash
   # On Linux/Mac
   python3 -m venv myvenv
   source myvenv/bin/activate
   
   # On Windows
   python -m venv myvenv
   myvenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import flask, joblib, numpy; print('All dependencies installed successfully!')"
   ```

## 📁 Project Structure Deep Dive

```
smart-urban-settlement/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── API_DOCUMENTATION.md           # API reference
├── DEVELOPMENT_GUIDE.md           # This file
├── TESTING_GUIDE.md               # Testing documentation
├── models/                        # Trained ML models
│   ├── air_quality_model.pkl      # Air quality prediction model
│   ├── traffic_congestion_model.pkl # Traffic prediction model
│   ├── housing_model.pkl          # Housing prediction model
│   ├── label_encoder_country.pkl  # Country label encoder
│   └── label_encoder_status.pkl   # Status label encoder
├── templates/                     # HTML templates
│   ├── dashboard.html             # Main dashboard interface
│   └── index.html                 # Landing page
├── static/                        # Static assets
│   ├── css/                       # Stylesheets
│   │   └── style.css              # Main stylesheet
│   └── images/                    # Images and plots
│       ├── actual_vs_predicted.png
│       ├── actual_vs_predicted_congestion.png
│       └── housing_demand_plot.png
├── data/                          # Dataset files
│   ├── air_quality_data.csv       # Air quality dataset
│   ├── traffic_data.csv           # Traffic dataset
│   └── housing_data.csv           # Housing dataset
├── src/                          # Source code modules
│   ├── models/                    # Model training scripts
│   ├── utils/                     # Utility functions
│   └── preprocessing/             # Data preprocessing scripts
├── tests/                        # Test files
│   ├── test_api.py               # API tests
│   ├── test_models.py            # Model tests
│   └── test_utils.py             # Utility tests
├── docs/                         # Documentation
│   ├── architecture.md           # System architecture
│   └── deployment.md             # Deployment guide
└── scripts/                      # Utility scripts
    ├── train_models.py           # Model training script
    ├── data_preprocessing.py     # Data preprocessing script
    └── deployment.py             # Deployment script
```

## 🔧 Development Workflow

### 1. Code Organization

#### Backend Structure
```python
# app.py - Main application file
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load models at startup
def load_models():
    """Load all trained models and encoders"""
    try:
        global air_quality_model, traffic_model, housing_model
        global country_encoder, status_encoder
        
        air_quality_model = joblib.load('models/air_quality_model.pkl')
        traffic_model = joblib.load('models/traffic_congestion_model.pkl')
        housing_model = joblib.load('models/housing_model.pkl')
        country_encoder = joblib.load('models/label_encoder_country.pkl')
        status_encoder = joblib.load('models/label_encoder_status.pkl')
        
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# Initialize models
if not load_models():
    raise RuntimeError("Failed to load required models")

# Routes
@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('dashboard.html')

@app.route('/predict/air_quality', methods=['POST'])
def predict_air_quality():
    """Air quality prediction endpoint"""
    logger.info("Air quality prediction request received")
    
    try:
        # Extract and validate input
        data = extract_air_quality_input(request.form)
        if not data['valid']:
            return render_template('dashboard.html', 
                                 air_quality_error=data['errors'])
        
        # Make prediction
        prediction = make_air_quality_prediction(data['features'])
        
        # Format response
        response = format_air_quality_response(prediction)
        
        logger.info(f"Air quality prediction completed: {prediction}")
        return render_template('dashboard.html', 
                             air_quality_prediction=response)
    
    except Exception as e:
        logger.error(f"Air quality prediction error: {e}")
        return render_template('dashboard.html', 
                             air_quality_error="Prediction failed")

# Similar structure for other endpoints...
```

#### Frontend Structure
```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Urban Settlement Management</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <!-- Header Section -->
    <header class="main-header">
        <h1>Smart Urban Settlement Management</h1>
        <nav class="main-nav">
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="#air-quality">Air Quality</a></li>
                <li><a href="#traffic">Traffic</a></li>
                <li><a href="#housing">Housing</a></li>
            </ul>
        </nav>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <section id="air-quality" class="prediction-section">
            <h2>Air Quality Prediction</h2>
            <form class="prediction-form" action="/predict/air_quality" method="POST">
                <!-- Form fields -->
            </form>
            {% if air_quality_prediction %}
            <div class="prediction-result">
                {{ air_quality_prediction }}
            </div>
            {% endif %}
        </section>

        <!-- Similar sections for traffic and housing -->
    </main>

    <!-- Footer -->
    <footer class="main-footer">
        <p>&copy; 2025 Smart Urban Management</p>
    </footer>

    <!-- JavaScript -->
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

### 2. Coding Standards

#### Python Style Guide (PEP 8)

```python
# Good example
def predict_air_quality(country: str, status: str, year: float, 
                       month: float, day: float) -> dict:
    """
    Predict air quality for given parameters.
    
    Args:
        country: Country name
        status: Development status
        year: Year (YYYY)
        month: Month (1-12)
        day: Day (1-31)
    
    Returns:
        Dictionary containing prediction results
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate inputs
    if not all([country, status, year, month, day]):
        raise ValueError("All parameters are required")
    
    # Process inputs
    features = preprocess_air_quality_inputs(country, status, year, month, day)
    
    # Make prediction
    prediction = air_quality_model.predict(features)[0]
    
    # Format response
    return {
        'prediction': prediction,
        'category': categorize_aqi(prediction),
        'confidence': calculate_confidence(prediction)
    }

# Bad example
def predictAQI(c, s, y, m, d):
    """Predict AQI"""
    if not c or not s or not y or not m or not d:
        return None
    f = [c, s, y, m, d]
    p = air_quality_model.predict(f)[0]
    return p
```

#### HTML/CSS Standards

```html
<!-- Good HTML structure -->
<section class="prediction-module" id="air-quality-prediction">
    <header class="module-header">
        <h2 class="module-title">Air Quality Prediction</h2>
        <p class="module-description">
            Predict air quality index based on geographical and temporal factors.
        </p>
    </header>
    
    <form class="prediction-form" action="/predict/air_quality" method="POST">
        <div class="form-group">
            <label for="country" class="form-label">Country</label>
            <input type="text" 
                   id="country" 
                   name="Country" 
                   class="form-input"
                   placeholder="Enter country name"
                   required>
        </div>
        
        <button type="submit" class="btn btn-primary">
            Get Prediction
        </button>
    </form>
</section>
```

```css
/* Good CSS structure */
.prediction-module {
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 2rem;
    margin-bottom: 2rem;
}

.module-header {
    margin-bottom: 1.5rem;
}

.module-title {
    color: #8b0000;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}

.module-description {
    color: #666;
    font-size: 0.9rem;
    margin: 0;
}

.form-group {
    margin-bottom: 1rem;
}

.form-label {
    display: block;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #333;
}

.form-input {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.form-input:focus {
    outline: none;
    border-color: #8b0000;
    box-shadow: 0 0 0 2px rgba(139, 0, 0, 0.1);
}

.btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.btn-primary {
    background-color: #246d00;
    color: white;
}

.btn-primary:hover {
    background-color: #aa1e4d;
}
```

### 3. Error Handling

#### Comprehensive Error Handling

```python
import logging
from functools import wraps
from flask import jsonify

logger = logging.getLogger(__name__)

def handle_errors(f):
    """Decorator for comprehensive error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({
                'error': 'Invalid input parameters',
                'details': str(e),
                'status': 'error'
            }), 400
        except KeyError as e:
            logger.warning(f"Missing parameter in {f.__name__}: {e}")
            return jsonify({
                'error': 'Missing required parameter',
                'details': str(e),
                'status': 'error'
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({
                'error': 'Internal server error',
                'status': 'error'
            }), 500
    return decorated_function

@app.route('/predict/air_quality', methods=['POST'])
@handle_errors
def predict_air_quality():
    """Air quality prediction with error handling"""
    # Extract inputs
    data = request.form
    
    # Validate inputs
    validate_air_quality_inputs(data)
    
    # Process inputs
    features = preprocess_air_quality_inputs(data)
    
    # Make prediction
    prediction = air_quality_model.predict(features)[0]
    
    # Return response
    return jsonify({
        'prediction': prediction,
        'category': categorize_aqi(prediction),
        'status': 'success'
    })

def validate_air_quality_inputs(data):
    """Validate air quality prediction inputs"""
    required_fields = ['Country', 'Status', 'year', 'month', 'day']
    
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Missing required field: {field}")
        
        if not data[field]:
            raise ValueError(f"Field {field} cannot be empty")
    
    # Validate numerical ranges
    try:
        year = float(data['year'])
        month = float(data['month'])
        day = float(data['day'])
        
        if not (1900 <= year <= 2100):
            raise ValueError("Year must be between 1900 and 2100")
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
            
    except ValueError as e:
        raise ValueError(f"Invalid numerical input: {e}")
```

### 4. Testing Strategy

#### Unit Testing

```python
# tests/test_api.py
import unittest
import json
from app import app

class TestAirQualityAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_valid_air_quality_prediction(self):
        """Test air quality prediction with valid inputs"""
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        response = self.app.post('/predict/air_quality', data=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted AQI', response.get_data(as_text=True))
    
    def test_invalid_air_quality_prediction(self):
        """Test air quality prediction with invalid inputs"""
        data = {
            'Country': 'InvalidCountry',
            'Status': 'InvalidStatus',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        response = self.app.post('/predict/air_quality', data=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Invalid category input', response.get_data(as_text=True))
    
    def test_missing_parameters(self):
        """Test air quality prediction with missing parameters"""
        data = {
            'Country': 'United States',
            'Status': 'Developed'
            # Missing year, month, day
        }
        
        response = self.app.post('/predict/air_quality', data=data)
        
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
```

#### Integration Testing

```python
# tests/test_integration.py
import unittest
import requests
import time

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.base_url = "http://localhost:5000"
        self.session = requests.Session()
    
    def test_full_workflow(self):
        """Test complete workflow from dashboard to prediction"""
        # 1. Access dashboard
        response = self.session.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('Smart Urban Settlement Management', response.text)
        
        # 2. Submit air quality prediction
        air_quality_data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/air_quality", 
            data=air_quality_data
        )
        self.assertEqual(response.status_code, 200)
        
        # 3. Submit traffic prediction
        traffic_data = {
            'hour': '14',
            'day_of_week': '2'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/traffic", 
            data=traffic_data
        )
        self.assertEqual(response.status_code, 200)
        
        # 4. Submit housing prediction
        housing_data = {
            'population_density': '5000',
            'average_income': '75000'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/housing", 
            data=housing_data
        )
        self.assertEqual(response.status_code, 200)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        
        def make_request():
            data = {
                'Country': 'United States',
                'Status': 'Developed',
                'year': '2024',
                'month': '6',
                'day': '15'
            }
            response = self.session.post(
                f"{self.base_url}/predict/air_quality", 
                data=data
            )
            return response.status_code
        
        # Create multiple threads
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(make_request()))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests were successful
        for status_code in results:
            self.assertEqual(status_code, 200)

if __name__ == '__main__':
    unittest.main()
```

### 5. Performance Optimization

#### Caching Implementation

```python
from functools import lru_cache
import hashlib
import json
import redis

# Redis cache configuration
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_prediction(ttl=3600):
    """Decorator for caching prediction results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Make prediction
            result = func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

def generate_cache_key(func_name, args, kwargs):
    """Generate unique cache key for function call"""
    key_data = {
        'function': func_name,
        'args': args,
        'kwargs': kwargs
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

@cache_prediction(ttl=1800)  # Cache for 30 minutes
def predict_air_quality_cached(country, status, year, month, day):
    """Cached version of air quality prediction"""
    features = preprocess_air_quality_inputs(country, status, year, month, day)
    prediction = air_quality_model.predict(features)[0]
    return {
        'prediction': prediction,
        'category': categorize_aqi(prediction)
    }
```

#### Database Integration

```python
import sqlite3
from datetime import datetime

class PredictionDatabase:
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_type TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction_result TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_prediction(self, prediction_type, input_data, result):
        """Save prediction to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (prediction_type, input_data, prediction_result)
                VALUES (?, ?, ?)
            ''', (prediction_type, json.dumps(input_data), json.dumps(result)))
            
            conn.commit()
    
    def get_prediction_history(self, prediction_type=None, limit=100):
        """Get prediction history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if prediction_type:
                cursor.execute('''
                    SELECT * FROM predictions 
                    WHERE prediction_type = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (prediction_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            return cursor.fetchall()

# Initialize database
db = PredictionDatabase()

# Use in prediction endpoints
@app.route('/predict/air_quality', methods=['POST'])
def predict_air_quality():
    # ... existing code ...
    
    # Save prediction to database
    db.save_prediction('air_quality', data, result)
    
    return render_template('dashboard.html', 
                         air_quality_prediction=result)
```

### 6. Security Best Practices

#### Input Sanitization

```python
import re
from html import escape

def sanitize_input(input_string):
    """Sanitize user input to prevent injection attacks"""
    if not input_string:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', str(input_string))
    
    # HTML escape
    sanitized = escape(sanitized)
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized.strip()

def validate_country_name(country):
    """Validate country name input"""
    # Only allow letters, spaces, and common punctuation
    if not re.match(r'^[A-Za-z\s\-\.]+$', country):
        raise ValueError("Invalid country name format")
    
    return sanitize_input(country)

def validate_numeric_input(value, min_val, max_val, field_name):
    """Validate numeric input within range"""
    try:
        num_value = float(value)
        if not (min_val <= num_value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return num_value
    except ValueError as e:
        raise ValueError(f"Invalid {field_name}: {e}")
```

#### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict/air_quality', methods=['POST'])
@limiter.limit("10 per minute")
def predict_air_quality():
    # ... existing code ...
```

### 7. Monitoring and Logging

#### Advanced Logging

```python
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            
            # File handler for all logs
            logging.handlers.RotatingFileHandler(
                'logs/app.log',
                maxBytes=10240000,  # 10MB
                backupCount=5
            ),
            
            # File handler for errors only
            logging.handlers.RotatingFileHandler(
                'logs/errors.log',
                maxBytes=10240000,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set error handler to only log errors
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/errors.log',
        maxBytes=10240000,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.addHandler(error_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# Usage in endpoints
@app.route('/predict/air_quality', methods=['POST'])
def predict_air_quality():
    logger.info("Air quality prediction request received")
    
    try:
        # ... prediction logic ...
        logger.info(f"Air quality prediction successful: {prediction}")
        return result
    except Exception as e:
        logger.error(f"Air quality prediction failed: {e}", exc_info=True)
        return error_response
```

#### Health Monitoring

```python
import psutil
import os

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'checks': {}
    }
    
    # Check models
    try:
        if all([air_quality_model, traffic_model, housing_model]):
            health_status['checks']['models'] = 'healthy'
        else:
            health_status['checks']['models'] = 'unhealthy'
            health_status['status'] = 'unhealthy'
    except Exception as e:
        health_status['checks']['models'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status['checks']['system'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        }
        
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            health_status['status'] = 'warning'
            
    except Exception as e:
        health_status['checks']['system'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # Check database
    try:
        db.get_prediction_history(limit=1)
        health_status['checks']['database'] = 'healthy'
    except Exception as e:
        health_status['checks']['database'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return jsonify(health_status), status_code
```

## 🚀 Deployment Preparation

### Production Configuration

```python
# config.py
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Production settings
    PREFERRED_URL_SCHEME = 'https'
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
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

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  redis_data:
```

## 📊 Performance Monitoring

### Application Metrics

```python
import time
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['type'])

def track_metrics(f):
    """Decorator to track request metrics"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            REQUEST_COUNT.labels(method='POST', endpoint=f.__name__).inc()
            return result
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
    
    return decorated_function

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.route('/predict/air_quality', methods=['POST'])
@track_metrics
def predict_air_quality():
    # ... existing code ...
    PREDICTION_COUNT.labels(type='air_quality').inc()
    return result
```

This comprehensive development guide provides everything needed to understand, develop, and maintain the Smart Urban Settlement Management System for your hackathon project. The guide covers coding standards, testing strategies, performance optimization, security best practices, and deployment preparation.