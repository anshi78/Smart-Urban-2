# API Documentation - Smart Urban Settlement Management System

## Overview

This document provides comprehensive API documentation for the Smart Urban Settlement Management System. The system exposes three main prediction endpoints for air quality, traffic congestion, and housing allocation.

## Base Configuration

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Response Format**: HTML (rendered templates) or JSON (for API calls)

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API key authentication
- JWT tokens
- OAuth 2.0

## Rate Limiting

No rate limiting is currently implemented. For production, consider:
- Request per minute limits
- IP-based throttling
- User-based quotas

## Error Handling

### Common Error Responses

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| 400 | Bad Request | Invalid input parameters |
| 404 | Not Found | Endpoint not found |
| 500 | Internal Server Error | Model prediction failed |

### Error Response Format

```json
{
  "error": "Invalid category input!",
  "status": "error",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Endpoints

### 1. Air Quality Prediction

#### POST /predict/air_quality

Predicts Air Quality Index (AQI) based on geographical and temporal factors.

**Request Parameters:**

| Parameter | Type | Required | Description | Validation Rules |
|-----------|------|----------|-------------|-----------------|
| Country | string | Yes | Country name | Must be in trained dataset |
| Status | string | Yes | Development status | Must be in trained dataset |
| year | float | Yes | Year (YYYY) | 1900-2100 |
| month | float | Yes | Month (1-12) | 1-12 |
| day | float | Yes | Day (1-31) | 1-31 |

**Example Request:**

```bash
curl -X POST http://localhost:5000/predict/air_quality \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Country=United States&Status=Developed&year=2024&month=6&day=15"
```

**Python Example:**

```python
import requests

def predict_air_quality(country, status, year, month, day):
    """
    Predict air quality for given parameters.
    
    Args:
        country (str): Country name
        status (str): Development status
        year (float): Year
        month (float): Month (1-12)
        day (float): Day (1-31)
    
    Returns:
        dict: Prediction results
    """
    url = "http://localhost:5000/predict/air_quality"
    data = {
        'Country': country,
        'Status': status,
        'year': year,
        'month': month,
        'day': day
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Usage
result = predict_air_quality("United States", "Developed", 2024, 6, 15)
print(result)
```

**JavaScript Example:**

```javascript
async function predictAirQuality(country, status, year, month, day) {
    const url = 'http://localhost:5000/predict/air_quality';
    const formData = new FormData();
    
    formData.append('Country', country);
    formData.append('Status', status);
    formData.append('year', year);
    formData.append('month', month);
    formData.append('day', day);
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.text();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return { error: error.message };
    }
}

// Usage
predictAirQuality('United States', 'Developed', 2024, 6, 15)
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

**Response Format:**

```html
<!-- HTML Response (rendered template) -->
<p><strong>Prediction:</strong> Predicted AQI: 45.23 (Good)</p>
```

**AQI Categories:**

| AQI Range | Category | Description |
|-----------|----------|-------------|
| 0-50 | Good | Air quality is satisfactory |
| 51-100 | Moderate | Air quality is acceptable |
| 101+ | Bad | Air quality is unhealthy |

### 2. Traffic Congestion Prediction

#### POST /predict/traffic

Predicts traffic congestion levels based on time and day patterns.

**Request Parameters:**

| Parameter | Type | Required | Description | Validation Rules |
|-----------|------|----------|-------------|-----------------|
| hour | int | Yes | Hour of day | 0-23 |
| day_of_week | int | Yes | Day of week | 0-6 (0=Monday, 6=Sunday) |

**Example Request:**

```bash
curl -X POST http://localhost:5000/predict/traffic \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "hour=14&day_of_week=2"
```

**Python Example:**

```python
import requests

def predict_traffic_congestion(hour, day_of_week):
    """
    Predict traffic congestion for given time parameters.
    
    Args:
        hour (int): Hour of day (0-23)
        day_of_week (int): Day of week (0-6)
    
    Returns:
        dict: Prediction results
    """
    url = "http://localhost:5000/predict/traffic"
    data = {
        'hour': hour,
        'day_of_week': day_of_week
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Usage
result = predict_traffic_congestion(14, 2)  # 2 PM on Wednesday
print(result)
```

**JavaScript Example:**

```javascript
async function predictTrafficCongestion(hour, dayOfWeek) {
    const url = 'http://localhost:5000/predict/traffic';
    const formData = new FormData();
    
    formData.append('hour', hour);
    formData.append('day_of_week', dayOfWeek);
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.text();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return { error: error.message };
    }
}

// Usage
predictTrafficCongestion(14, 2)
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

**Response Format:**

```html
<!-- HTML Response (rendered template) -->
<p><strong>Prediction:</strong> Traffic Congestion Level: 65.45 (High)</p>
```

**Congestion Levels:**

| Congestion Level | Severity | Description |
|------------------|----------|-------------|
| < 50 | Low | Normal traffic conditions |
| ≥ 50 | High | Heavy traffic congestion |

### 3. Housing Allocation Prediction

#### POST /predict/housing

Predicts housing prices and affordability based on demographic factors.

**Request Parameters:**

| Parameter | Type | Required | Description | Validation Rules |
|-----------|------|----------|-------------|-----------------|
| population_density | int | Yes | Population density per square km | Positive integer |
| average_income | int | Yes | Average income in currency units | Positive integer |

**Example Request:**

```bash
curl -X POST http://localhost:5000/predict/housing \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "population_density=5000&average_income=75000"
```

**Python Example:**

```python
import requests

def predict_housing_price(population_density, average_income):
    """
    Predict housing price for given demographic factors.
    
    Args:
        population_density (int): Population density per square km
        average_income (int): Average income in currency units
    
    Returns:
        dict: Prediction results
    """
    url = "http://localhost:5000/predict/housing"
    data = {
        'population_density': population_density,
        'average_income': average_income
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Usage
result = predict_housing_price(5000, 75000)
print(result)
```

**JavaScript Example:**

```javascript
async function predictHousingPrice(populationDensity, averageIncome) {
    const url = 'http://localhost:5000/predict/housing';
    const formData = new FormData();
    
    formData.append('population_density', populationDensity);
    formData.append('average_income', averageIncome);
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.text();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return { error: error.message };
    }
}

// Usage
predictHousingPrice(5000, 75000)
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

**Response Format:**

```html
<!-- HTML Response (rendered template) -->
<p><strong>Prediction:</strong> Predicted Housing Price: 45000.00 (Affordable)</p>
```

**Affordability Thresholds:**

| Price Range | Affordability | Description |
|-------------|---------------|-------------|
| ≤ 50000 | Affordable | Within reach for average income |
| > 50000 | Not Affordable | Above average affordability threshold |

## Batch Processing

For multiple predictions, you can make sequential API calls:

```python
import requests
import time

def batch_predictions(predictions_data):
    """
    Process multiple predictions in batch.
    
    Args:
        predictions_data (list): List of prediction parameters
    
    Returns:
        list: Results for all predictions
    """
    results = []
    
    for data in predictions_data:
        try:
            if data['type'] == 'air_quality':
                response = requests.post('http://localhost:5000/predict/air_quality', 
                                       data=data['params'])
            elif data['type'] == 'traffic':
                response = requests.post('http://localhost:5000/predict/traffic', 
                                       data=data['params'])
            elif data['type'] == 'housing':
                response = requests.post('http://localhost:5000/predict/housing', 
                                       data=data['params'])
            
            results.append({
                'type': data['type'],
                'params': data['params'],
                'result': response.text,
                'status': 'success'
            })
            
            # Add delay to prevent overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            results.append({
                'type': data['type'],
                'params': data['params'],
                'error': str(e),
                'status': 'error'
            })
    
    return results

# Usage example
batch_data = [
    {
        'type': 'air_quality',
        'params': {'Country': 'United States', 'Status': 'Developed', 'year': 2024, 'month': 6, 'day': 15}
    },
    {
        'type': 'traffic',
        'params': {'hour': 14, 'day_of_week': 2}
    },
    {
        'type': 'housing',
        'params': {'population_density': 5000, 'average_income': 75000}
    }
]

results = batch_predictions(batch_data)
for result in results:
    print(f"{result['type']}: {result['status']}")
```

## Testing

### Unit Tests

```python
import unittest
import requests

class TestUrbanSettlementAPI(unittest.TestCase):
    
    def setUp(self):
        self.base_url = "http://localhost:5000"
    
    def test_air_quality_prediction(self):
        """Test air quality prediction endpoint"""
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': 2024,
            'month': 6,
            'day': 15
        }
        
        response = requests.post(f"{self.base_url}/predict/air_quality", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted AQI", response.text)
    
    def test_traffic_prediction(self):
        """Test traffic prediction endpoint"""
        data = {
            'hour': 14,
            'day_of_week': 2
        }
        
        response = requests.post(f"{self.base_url}/predict/traffic", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Traffic Congestion Level", response.text)
    
    def test_housing_prediction(self):
        """Test housing prediction endpoint"""
        data = {
            'population_density': 5000,
            'average_income': 75000
        }
        
        response = requests.post(f"{self.base_url}/predict/housing", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted Housing Price", response.text)
    
    def test_invalid_input(self):
        """Test error handling for invalid inputs"""
        data = {
            'Country': 'InvalidCountry',
            'Status': 'InvalidStatus',
            'year': 2024,
            'month': 6,
            'day': 15
        }
        
        response = requests.post(f"{self.base_url}/predict/air_quality", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Invalid category input", response.text)

if __name__ == '__main__':
    unittest.main()
```

### Load Testing

```python
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

def load_test(endpoint, data, num_requests=100):
    """Perform load testing on an endpoint"""
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    
    def make_request():
        nonlocal successful_requests, failed_requests
        try:
            response = requests.post(f"http://localhost:5000{endpoint}", data=data)
            if response.status_code == 200:
                successful_requests += 1
            else:
                failed_requests += 1
        except:
            failed_requests += 1
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in futures:
            future.result()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Load Test Results for {endpoint}:")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Requests per second: {num_requests/total_time:.2f}")

# Run load tests
air_quality_data = {
    'Country': 'United States',
    'Status': 'Developed',
    'year': 2024,
    'month': 6,
    'day': 15
}

load_test('/predict/air_quality', air_quality_data, 50)
```

## Monitoring and Logging

### Health Check Endpoint

```python
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if models are loaded
        if (air_quality_model is not None and 
            traffic_model is not None and 
            housing_model is not None):
            return jsonify({
                'status': 'healthy',
                'models_loaded': True,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'models_loaded': False,
                'timestamp': datetime.now().isoformat()
            }), 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
```

### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('app.log', maxBytes=10240000, backupCount=5),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Usage in endpoints
@app.route('/predict/air_quality', methods=['POST'])
def predict_air_quality():
    logger.info("Air quality prediction request received")
    # ... existing code ...
    logger.info(f"Air quality prediction completed: {prediction}")
```

## Best Practices

### 1. Input Validation

```python
def validate_air_quality_input(data):
    """Validate air quality prediction inputs"""
    errors = []
    
    # Check required fields
    required_fields = ['Country', 'Status', 'year', 'month', 'day']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate year
    if 'year' in data:
        try:
            year = float(data['year'])
            if not (1900 <= year <= 2100):
                errors.append("Year must be between 1900 and 2100")
        except ValueError:
            errors.append("Year must be a valid number")
    
    # Validate month
    if 'month' in data:
        try:
            month = float(data['month'])
            if not (1 <= month <= 12):
                errors.append("Month must be between 1 and 12")
        except ValueError:
            errors.append("Month must be a valid number")
    
    # Validate day
    if 'day' in data:
        try:
            day = float(data['day'])
            if not (1 <= day <= 31):
                errors.append("Day must be between 1 and 31")
        except ValueError:
            errors.append("Day must be a valid number")
    
    return errors
```

### 2. Response Caching

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def cached_prediction(prediction_type, input_hash):
    """Cache prediction results for repeated inputs"""
    # Implementation for caching predictions
    pass

def get_input_hash(data):
    """Generate hash for input data for caching"""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
```

### 3. Rate Limiting

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

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all model files are present in the `models/` directory
   - Check file permissions
   - Verify model file integrity

2. **Input Validation Errors**
   - Check parameter names and types
   - Ensure categorical values are in the training dataset
   - Validate numerical ranges

3. **Performance Issues**
   - Monitor server resources
   - Implement caching for repeated requests
   - Consider model optimization

### Debug Mode

Enable debug mode for detailed error information:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Log Analysis

```python
def analyze_logs(log_file='app.log'):
    """Analyze application logs for patterns"""
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    # Count requests by endpoint
    endpoint_counts = {}
    error_counts = {}
    
    for log in logs:
        if 'POST /predict/' in log:
            endpoint = log.split('POST ')[1].split(' ')[0]
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        
        if 'ERROR' in log:
            error_counts['total_errors'] = error_counts.get('total_errors', 0) + 1
    
    print("Endpoint Usage:")
    for endpoint, count in endpoint_counts.items():
        print(f"  {endpoint}: {count} requests")
    
    print(f"\nTotal Errors: {error_counts.get('total_errors', 0)}")
```