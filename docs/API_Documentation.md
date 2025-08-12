# API Documentation

## Overview

The Smart Urban Settlement Management System provides RESTful APIs for three main prediction services: Air Quality, Traffic Congestion, and Housing Price prediction.

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. Dashboard - GET /

**Description**: Main dashboard interface for the Smart Urban Settlement Management System.

**Method**: `GET`

**URL**: `/`

**Response**: HTML dashboard page with prediction forms

**Example Request**:
```bash
curl -X GET http://localhost:5000/
```

**Response**: Returns the main dashboard HTML page with all prediction forms.

---

### 2. Air Quality Prediction - POST /predict/air_quality

**Description**: Predicts Air Quality Index (AQI) based on country, status, and date parameters.

**Method**: `POST`

**URL**: `/predict/air_quality`

**Content-Type**: `application/x-www-form-urlencoded`

#### Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| Country | string | Yes | Country name | "India" |
| Status | string | Yes | Environmental status | "Good" |
| year | float | Yes | Year (2020-2025) | 2024 |
| month | float | Yes | Month (1-12) | 7 |
| day | float | Yes | Day (1-31) | 15 |

#### Response

**Success Response** (200 OK):
Returns HTML page with prediction result.

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict/air_quality \
  -d "Country=India&Status=Good&year=2024&month=7&day=15" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

**Example Form Data**:
```
Country=India
Status=Good
year=2024
month=7
day=15
```

**Response Format**:
The response includes the prediction in the format:
```
"Predicted AQI: {value} ({category})"
```

Where:
- `value`: Predicted AQI numerical value
- `category`: 
  - "Good" (≤ 50)
  - "Moderate" (51-100)
  - "Bad" (> 100)

**Example Response Text**:
```
"Predicted AQI: 45.32 (Good)"
```

#### Error Responses

**400 Bad Request**: Invalid category input
```
"Invalid category input!"
```

---

### 3. Traffic Congestion Prediction - POST /predict/traffic

**Description**: Predicts traffic congestion level based on hour and day of the week.

**Method**: `POST`

**URL**: `/predict/traffic`

**Content-Type**: `application/x-www-form-urlencoded`

#### Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| hour | integer | Yes | Hour of day (0-23) | 17 |
| day_of_week | integer | Yes | Day of week (0=Monday, 6=Sunday) | 4 |

#### Response

**Success Response** (200 OK):
Returns HTML page with traffic prediction result.

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict/traffic \
  -d "hour=17&day_of_week=4" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

**Example Form Data**:
```
hour=17
day_of_week=4
```

**Response Format**:
```
"Traffic Congestion Level: {value} ({level})"
```

Where:
- `value`: Predicted congestion numerical value
- `level`: 
  - "High" (≥ 50)
  - "Low" (< 50)

**Example Response Text**:
```
"Traffic Congestion Level: 67.45 (High)"
```

---

### 4. Housing Price Prediction - POST /predict/housing

**Description**: Predicts housing prices and affordability based on population density and average income.

**Method**: `POST`

**URL**: `/predict/housing`

**Content-Type**: `application/x-www-form-urlencoded`

#### Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| population_density | integer | Yes | Population density per area | 25000 |
| average_income | integer | Yes | Average income in area | 75000 |

#### Response

**Success Response** (200 OK):
Returns HTML page with housing prediction result.

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict/housing \
  -d "population_density=25000&average_income=75000" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

**Example Form Data**:
```
population_density=25000
average_income=75000
```

**Response Format**:
```
"Predicted Housing Price: {value} ({affordability})"
```

Where:
- `value`: Predicted housing price
- `affordability`: 
  - "Affordable" (≤ 50,000)
  - "Not Affordable" (> 50,000)

**Example Response Text**:
```
"Predicted Housing Price: 45000.00 (Affordable)"
```

---

## Error Handling

### Common Error Responses

**405 Method Not Allowed**: When using wrong HTTP method
```json
{
  "error": "Method not allowed"
}
```

**400 Bad Request**: When required parameters are missing
```json
{
  "error": "Missing required parameters"
}
```

**500 Internal Server Error**: When model prediction fails
```json
{
  "error": "Prediction failed"
}
```

---

## Data Models

### Air Quality Input Model
```json
{
  "Country": "string",
  "Status": "string", 
  "year": "float",
  "month": "float",
  "day": "float"
}
```

### Traffic Input Model
```json
{
  "hour": "integer (0-23)",
  "day_of_week": "integer (0-6)"
}
```

### Housing Input Model
```json
{
  "population_density": "integer",
  "average_income": "integer"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

---

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

---

## Examples

### Complete Air Quality Prediction Example

```python
import requests

url = "http://localhost:5000/predict/air_quality"
data = {
    "Country": "India",
    "Status": "Good", 
    "year": 2024,
    "month": 7,
    "day": 15
}

response = requests.post(url, data=data)
print(response.text)  # Returns HTML with prediction
```

### Complete Traffic Prediction Example

```python
import requests

url = "http://localhost:5000/predict/traffic"
data = {
    "hour": 17,
    "day_of_week": 4
}

response = requests.post(url, data=data)
print(response.text)  # Returns HTML with prediction
```

### Complete Housing Prediction Example

```python
import requests

url = "http://localhost:5000/predict/housing"
data = {
    "population_density": 25000,
    "average_income": 75000
}

response = requests.post(url, data=data)
print(response.text)  # Returns HTML with prediction
```

---

## Testing

### Using curl

Test each endpoint with curl commands:

```bash
# Test Air Quality
curl -X POST http://localhost:5000/predict/air_quality \
  -d "Country=India&Status=Good&year=2024&month=7&day=15"

# Test Traffic
curl -X POST http://localhost:5000/predict/traffic \
  -d "hour=17&day_of_week=4"

# Test Housing
curl -X POST http://localhost:5000/predict/housing \
  -d "population_density=25000&average_income=75000"
```

### Using Python requests

```python
import requests

base_url = "http://localhost:5000"

# Test all endpoints
endpoints = [
    ("/predict/air_quality", {"Country": "India", "Status": "Good", "year": 2024, "month": 7, "day": 15}),
    ("/predict/traffic", {"hour": 17, "day_of_week": 4}),
    ("/predict/housing", {"population_density": 25000, "average_income": 75000})
]

for endpoint, data in endpoints:
    response = requests.post(f"{base_url}{endpoint}", data=data)
    print(f"Status: {response.status_code}")
    print(f"Endpoint: {endpoint}")
    print("---")
```