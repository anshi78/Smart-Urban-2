# Smart Urban Settlement Management System

A comprehensive web-based system for real-time predictions and management of urban settlement aspects including air quality, traffic congestion, and housing allocation.

## 🏗️ Project Overview

The Smart Urban Settlement Management System is a Flask-based web application that provides real-time predictions for three critical urban management aspects:

1. **Air Quality Prediction** - Predicts Air Quality Index (AQI) based on geographical and temporal factors
2. **Traffic Congestion Prediction** - Forecasts traffic congestion levels based on time and day patterns
3. **Housing Allocation Prediction** - Predicts housing prices and affordability based on demographic factors

## 📁 Project Structure

```
smart-urban-settlement/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── models/                        # Trained ML models
│   ├── air_quality_model.pkl
│   ├── traffic_congestion_model.pkl
│   ├── housing_model.pkl
│   ├── label_encoder_country.pkl
│   └── label_encoder_status.pkl
├── templates/                     # HTML templates
│   ├── dashboard.html
│   └── index.html
├── static/                        # Static assets
│   ├── css/
│   └── images/
├── data/                          # Dataset files
└── src/                          # Source code modules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-urban-settlement
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### 1. Air Quality Prediction API

**Endpoint:** `POST /predict/air_quality`

**Description:** Predicts Air Quality Index (AQI) based on geographical and temporal parameters.

**Request Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| Country | string | Yes | Country name | "United States" |
| Status | string | Yes | Status/condition | "Developed" |
| year | float | Yes | Year (YYYY) | 2024.0 |
| month | float | Yes | Month (1-12) | 6.0 |
| day | float | Yes | Day (1-31) | 15.0 |

**Response:**
```json
{
  "prediction": "Predicted AQI: 45.23 (Good)",
  "aqi_value": 45.23,
  "category": "Good"
}
```

**Example Usage:**
```bash
curl -X POST http://localhost:5000/predict/air_quality \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Country=United States&Status=Developed&year=2024&month=6&day=15"
```

**Python Example:**
```python
import requests

data = {
    'Country': 'United States',
    'Status': 'Developed',
    'year': 2024,
    'month': 6,
    'day': 15
}

response = requests.post('http://localhost:5000/predict/air_quality', data=data)
print(response.text)
```

### 2. Traffic Congestion Prediction API

**Endpoint:** `POST /predict/traffic`

**Description:** Predicts traffic congestion levels based on time and day patterns.

**Request Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| hour | int | Yes | Hour of day (0-23) | 14 |
| day_of_week | int | Yes | Day of week (0=Monday, 6=Sunday) | 2 |

**Response:**
```json
{
  "prediction": "Traffic Congestion Level: 65.45 (High)",
  "congestion_level": 65.45,
  "severity": "High"
}
```

**Example Usage:**
```bash
curl -X POST http://localhost:5000/predict/traffic \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "hour=14&day_of_week=2"
```

**Python Example:**
```python
import requests

data = {
    'hour': 14,
    'day_of_week': 2
}

response = requests.post('http://localhost:5000/predict/traffic', data=data)
print(response.text)
```

### 3. Housing Allocation Prediction API

**Endpoint:** `POST /predict/housing`

**Description:** Predicts housing prices and affordability based on demographic factors.

**Request Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| population_density | int | Yes | Population density per square km | 5000 |
| average_income | int | Yes | Average income in currency units | 75000 |

**Response:**
```json
{
  "prediction": "Predicted Housing Price: 45000.00 (Affordable)",
  "price": 45000.00,
  "affordability": "Affordable"
}
```

**Example Usage:**
```bash
curl -X POST http://localhost:5000/predict/housing \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "population_density=5000&average_income=75000"
```

**Python Example:**
```python
import requests

data = {
    'population_density': 5000,
    'average_income': 75000
}

response = requests.post('http://localhost:5000/predict/housing', data=data)
print(response.text)
```

## 🎯 Frontend Components

### Dashboard Interface

The main dashboard (`/`) provides a unified interface for all three prediction modules:

#### 1. Air Quality Prediction Form
- **Country Input**: Text field for country name
- **Status Input**: Text field for development status
- **Date Inputs**: Year, month, and day number inputs
- **Prediction Display**: Shows AQI value and category (Good/Moderate/Bad)

#### 2. Traffic Congestion Prediction Form
- **Hour Input**: Number field (0-23) for time of day
- **Day of Week Input**: Number field (0-6) for day selection
- **Prediction Display**: Shows congestion level and severity (High/Low)

#### 3. Housing Allocation Prediction Form
- **Population Density Input**: Number field for density per square km
- **Average Income Input**: Number field for income in currency units
- **Prediction Display**: Shows predicted price and affordability status

## 🔧 Technical Architecture

### Backend Components

#### Flask Application (`app.py`)
- **Main Application**: Flask web server with three prediction endpoints
- **Model Loading**: Pre-trained machine learning models loaded at startup
- **Data Processing**: Input validation and feature encoding
- **Response Generation**: Formatted predictions with categorization

#### Machine Learning Models
- **Air Quality Model**: Predicts AQI based on geographical and temporal features
- **Traffic Model**: Predicts congestion levels using time-based patterns
- **Housing Model**: Predicts housing prices using demographic features
- **Label Encoders**: Convert categorical variables to numerical format

### Frontend Components

#### HTML Templates
- **Dashboard Template**: Main interface with three prediction forms
- **Responsive Design**: Mobile-friendly layout with CSS styling
- **Form Validation**: Client-side input validation
- **Dynamic Content**: Real-time prediction display

#### Styling
- **Color Scheme**: Dark red/pink theme with green accents
- **Layout**: Card-based design with clean typography
- **Interactive Elements**: Hover effects and transitions

## 📊 Data Processing

### Input Validation
- **Type Checking**: Ensures correct data types for all inputs
- **Range Validation**: Validates numerical ranges (hours, days, etc.)
- **Categorical Validation**: Checks against encoded categories

### Feature Engineering
- **Label Encoding**: Converts categorical variables to numerical
- **Feature Scaling**: Normalizes numerical inputs
- **Array Formatting**: Prepares features for model prediction

### Output Processing
- **Threshold Classification**: Categorizes predictions into meaningful groups
- **Formatting**: Formats numerical outputs with appropriate precision
- **Error Handling**: Graceful handling of invalid inputs

## 🛠️ Development Guidelines

### Adding New Features

1. **Model Integration**
   ```python
   # Load new model
   new_model = joblib.load('models/new_model.pkl')
   
   # Add prediction endpoint
   @app.route('/predict/new_feature', methods=['POST'])
   def predict_new_feature():
       # Process inputs
       # Make prediction
       # Return formatted response
   ```

2. **Frontend Integration**
   ```html
   <!-- Add new form section -->
   <div>
       <h3>New Feature Prediction</h3>
       <form action="/predict/new_feature" method="POST">
           <!-- Add form fields -->
       </form>
   </div>
   ```

### Error Handling

```python
try:
    # Model prediction
    prediction = model.predict(features)
except ValueError as e:
    return render_template('dashboard.html', 
                         error="Invalid input parameters")
except Exception as e:
    return render_template('dashboard.html', 
                         error="Prediction failed")
```

### Testing

```python
# Unit test example
def test_air_quality_prediction():
    with app.test_client() as client:
        response = client.post('/predict/air_quality', data={
            'Country': 'United States',
            'Status': 'Developed',
            'year': 2024,
            'month': 6,
            'day': 15
        })
        assert response.status_code == 200
```

## 🔒 Security Considerations

### Input Validation
- **Sanitization**: Clean user inputs to prevent injection attacks
- **Type Checking**: Validate data types before processing
- **Range Validation**: Ensure inputs are within acceptable ranges

### Error Handling
- **Graceful Degradation**: Handle errors without exposing system details
- **Logging**: Log errors for debugging without exposing to users
- **User Feedback**: Provide meaningful error messages

## 📈 Performance Optimization

### Model Loading
- **Lazy Loading**: Load models only when needed
- **Caching**: Cache model predictions for repeated inputs
- **Compression**: Use compressed model formats

### Response Optimization
- **Async Processing**: Handle multiple requests concurrently
- **Response Caching**: Cache common prediction results
- **Compression**: Compress responses for faster transmission

## 🚀 Deployment

### Production Setup

1. **Environment Configuration**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

2. **WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Reverse Proxy (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

## 🔮 Future Enhancements

### Planned Features
- **Real-time Data Integration**: Connect to live data sources
- **Advanced Analytics**: Add data visualization and trends
- **Mobile Application**: Develop native mobile apps
- **API Rate Limiting**: Implement request throttling
- **User Authentication**: Add user management system
- **Historical Data**: Store and analyze prediction history
- **Multi-language Support**: Internationalization
- **Advanced ML Models**: Implement deep learning models

### Technical Improvements
- **Microservices Architecture**: Split into separate services
- **Database Integration**: Add persistent data storage
- **Message Queues**: Implement async processing
- **Monitoring**: Add application performance monitoring
- **CI/CD Pipeline**: Automated testing and deployment