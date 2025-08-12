# Machine Learning Documentation

## Overview

The Smart Urban Settlement Management System employs three distinct machine learning modules for different urban planning aspects. Each module uses optimized algorithms for their specific prediction tasks.

## Architecture

```
ML Pipeline
├── Air Quality Module (Random Forest)
├── Traffic Congestion Module (Random Forest)  
└── Housing Price Module (Linear Regression + K-Means)
```

---

## 1. Air Quality Module (`src/air_quality.py`)

### Purpose
Predicts Air Quality Index (AQI) values based on geographical and temporal features.

### Algorithm
**Random Forest Regressor** with 100 estimators

### Features
- `Country` (encoded): Geographical location
- `Status` (encoded): Current environmental status  
- `year`: Year component of date
- `month`: Month component of date
- `day`: Day component of date

### Functions

#### `preprocess_and_train()`
Preprocesses air quality data and trains the prediction model.

**Process**:
1. Loads CSV data from `data/air_aquality.csv`
2. Converts date strings to datetime objects
3. Extracts temporal features (year, month, day)
4. Encodes categorical variables using LabelEncoder
5. Splits data (80% train, 20% test)
6. Trains Random Forest model
7. Saves model and encoders to `models/` directory

**Returns**: 
- `RandomForestRegressor`: Trained model object

**Files Generated**:
- `models/air_quality_model.pkl`
- `models/label_encoder_country.pkl`
- `models/label_encoder_status.pkl`

**Example Usage**:
```python
from src.air_quality import preprocess_and_train

# Train the model
model = preprocess_and_train()
print("Air quality model trained successfully!")
```

#### `predict_aqi(country, status, year, month, day)`
Predicts AQI for given parameters.

**Parameters**:
- `country` (str): Country name (e.g., "India", "Albania")
- `status` (str): Environmental status (e.g., "Good", "Moderate") 
- `year` (int): Year (2020-2025)
- `month` (int): Month (1-12)
- `day` (int): Day (1-31)

**Returns**:
- `prediction` (float): Predicted AQI value
- `category` (str): AQI category ("Good", "Moderate", "Bad")

**AQI Categories**:
- **Good**: ≤ 50
- **Moderate**: 51-100  
- **Bad**: > 100

**Example Usage**:
```python
from src.air_quality import predict_aqi

# Predict AQI for Albania
aqi_value, aqi_category = predict_aqi('Albania', 'Good', 2024, 7, 21)
print(f"Predicted AQI: {aqi_value:.2f} ({aqi_category})")
```

### Model Performance
- **Metric**: Mean Squared Error (MSE)
- **Evaluation**: Train/test split validation
- **Output**: MSE value printed during training

---

## 2. Traffic Congestion Module (`src/traffic_congestion.py`)

### Purpose
Predicts traffic congestion levels based on temporal patterns.

### Algorithm
**Random Forest Regressor** with 100 estimators

### Features
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)

### Functions

#### `preprocess_traffic_data()`
Loads and preprocesses traffic congestion data.

**Process**:
1. Loads CSV without headers from `data/traffic_congestion.csv`
2. Assigns column names: timestamp, location, congestion_level, id
3. Converts timestamps to datetime
4. Extracts hour and day_of_week features
5. Removes unnecessary columns

**Returns**:
- `DataFrame`: Preprocessed traffic data

**Example Usage**:
```python
from src.traffic_congestion import preprocess_traffic_data

# Load and preprocess data
data = preprocess_traffic_data()
print(f"Data shape: {data.shape}")
print(data.head())
```

#### `train_traffic_model()`
Trains the traffic congestion prediction model.

**Process**:
1. Preprocesses traffic data
2. Splits features (hour, day_of_week) and target (congestion_level)
3. Trains Random Forest model
4. Evaluates using Mean Absolute Error (MAE)
5. Generates prediction visualization
6. Saves model to `models/traffic_congestion_model.pkl`

**Files Generated**:
- `models/traffic_congestion_model.pkl`
- `static/images/actual_vs_predicted_congestion.png`

**Example Usage**:
```python
from src.traffic_congestion import train_traffic_model

# Train the model
train_traffic_model()
print("Traffic model training completed!")
```

#### `predict_traffic_congestion(hour, day_of_week)`
Predicts traffic congestion level for given time parameters.

**Parameters**:
- `hour` (int): Hour of day (0-23)
- `day_of_week` (int): Day of week (0-6, Monday=0)

**Returns**:
- `float`: Predicted congestion level

**Congestion Categories**:
- **High**: ≥ 50
- **Low**: < 50

**Example Usage**:
```python
from src.traffic_congestion import predict_traffic_congestion

# Predict for Friday 5 PM
congestion = predict_traffic_congestion(17, 4)
category = "High" if congestion >= 50 else "Low"
print(f"Congestion Level: {congestion:.2f} ({category})")
```

#### `plot_predictions(y_test, y_pred)`
Generates visualization comparing actual vs predicted values.

**Parameters**:
- `y_test` (array): Actual congestion values
- `y_pred` (array): Predicted congestion values

**Output**: Saves scatter plot to `static/images/actual_vs_predicted_congestion.png`

### Model Performance
- **Metric**: Mean Absolute Error (MAE)
- **Visualization**: Actual vs Predicted scatter plot
- **Evaluation**: Train/test split validation

---

## 3. Housing Price Module (`src/house_price.py`)

### Purpose
Predicts housing prices and affordability using clustering and regression.

### Algorithms
1. **K-Means Clustering** (3 clusters) for area segmentation
2. **Linear Regression** for price prediction per cluster

### Features
- `population_density`: Calculated as area_population / avg_area_house_age
- `avg_area_income`: Average income in the area

### Functions

#### `preprocess_housing_data()`
Loads and preprocesses housing dataset.

**Process**:
1. Loads CSV from `data/house_prediction.csv`
2. Assigns proper column names
3. Removes address column
4. Converts numeric columns and handles missing values
5. Calculates population_density feature

**Returns**:
- `DataFrame`: Preprocessed housing data

**Example Usage**:
```python
from src.house_price import preprocess_housing_data

# Load and preprocess data
data = preprocess_housing_data()
print(f"Data shape: {data.shape}")
print(data.columns.tolist())
```

#### `apply_kmeans_clustering(data, n_clusters=3)`
Applies K-Means clustering to segment areas.

**Parameters**:
- `data` (DataFrame): Preprocessed housing data
- `n_clusters` (int): Number of clusters (default: 3)

**Returns**:
- `data` (DataFrame): Data with cluster assignments
- `kmeans` (KMeans): Fitted clustering model

**Clustering Features**:
- Population density
- Average area income

**Example Usage**:
```python
from src.house_price import preprocess_housing_data, apply_kmeans_clustering

data = preprocess_housing_data()
clustered_data, kmeans_model = apply_kmeans_clustering(data)
print(f"Cluster distribution:\n{clustered_data['cluster'].value_counts()}")
```

#### `train_housing_model()`
Trains housing price prediction models for each cluster.

**Process**:
1. Preprocesses housing data
2. Applies K-Means clustering
3. Trains separate Linear Regression model for each cluster
4. Saves all models to pickle files

**Files Generated**:
- `models/kmeans_model.pkl`
- `models/housing_models.pkl`

**Returns**:
- `models` (dict): Dictionary of trained models per cluster
- `kmeans` (KMeans): Fitted clustering model

**Example Usage**:
```python
from src.house_price import train_housing_model

# Train all housing models
models, kmeans = train_housing_model()
print(f"Trained models for {len(models)} clusters")
```

#### `predict_housing_demand(new_data, affordability_threshold=50000)`
Predicts housing prices and affordability for new data.

**Parameters**:
- `new_data` (DataFrame): Input data with population_density and avg_area_income
- `affordability_threshold` (int): Price threshold for affordability (default: 50,000)

**Returns**:
- `predictions` (array): Predicted housing prices
- `affordability` (list): Affordability status for each prediction

**Affordability Categories**:
- **Affordable**: ≤ 50,000
- **Not Affordable**: > 50,000

**Example Usage**:
```python
import pandas as pd
from src.house_price import predict_housing_demand

# Create sample data
new_data = pd.DataFrame({
    'population_density': [24212, 53504],
    'avg_area_income': [74213, 69423]
})

# Predict prices and affordability
prices, affordability = predict_housing_demand(new_data)
for i, (price, afford) in enumerate(zip(prices, affordability)):
    print(f"Area {i+1}: ${price:,.2f} ({afford})")
```

#### `plot_predictions(actual, predicted)`
Generates visualization comparing actual vs predicted housing prices.

**Parameters**:
- `actual` (list): Actual housing prices
- `predicted` (array): Predicted housing prices

**Output**: Saves line plot to `static/images/housing_demand_plot.png`

### Model Performance
- **Metric**: Mean Absolute Error (MAE)
- **Visualization**: Actual vs Predicted line plot
- **Approach**: Cluster-specific model evaluation

---

## Model Files

### Generated Model Files
All trained models are saved in the `models/` directory:

| File | Purpose | Size |
|------|---------|------|
| `air_quality_model.pkl` | Air quality Random Forest model | ~99MB |
| `label_encoder_country.pkl` | Country label encoder | ~2.7KB |
| `label_encoder_status.pkl` | Status label encoder | ~635B |
| `traffic_congestion_model.pkl` | Traffic Random Forest model | ~2.3MB |
| `housing_model.pkl` | Housing Linear Regression models | ~921B |
| `kmeans_model.pkl` | K-Means clustering model | N/A |

### Loading Pre-trained Models

```python
import joblib

# Load air quality models
air_model = joblib.load('models/air_quality_model.pkl')
country_encoder = joblib.load('models/label_encoder_country.pkl')
status_encoder = joblib.load('models/label_encoder_status.pkl')

# Load traffic model
traffic_model = joblib.load('models/traffic_congestion_model.pkl')

# Load housing models
housing_models = joblib.load('models/housing_models.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')
```

---

## Data Requirements

### Input Data Formats

#### Air Quality Data (`data/air_aquality.csv`)
```csv
Country,Status,Date,AQI Value
India,Good,2024-07-15,45.2
Albania,Moderate,2024-07-16,67.8
```

#### Traffic Data (`data/traffic_congestion.csv`)
```csv
# No headers - timestamp,location,congestion_level,id
2024-07-15 09:00:00,Location1,65.5,1
2024-07-15 17:30:00,Location2,89.2,2
```

#### Housing Data (`data/house_prediction.csv`)
```csv
avg_area_income,avg_area_house_age,avg_area_rooms,avg_area_bedrooms,area_population,price,address
74213.0,5.682,7.009,1.04,24212.0,1231412.0,"Address1"
69423.0,6.498,6.652,1.06,53504.0,1245817.0,"Address2"
```

---

## Performance Optimization

### Model Selection Rationale

1. **Random Forest for Air Quality & Traffic**:
   - Handles non-linear relationships
   - Robust to outliers
   - Good performance with temporal features

2. **Linear Regression + K-Means for Housing**:
   - Clustering captures area-specific patterns
   - Linear regression suitable for price prediction
   - Interpretable results for stakeholders

### Memory Considerations
- Air quality model is largest (~99MB) due to high feature dimensionality
- Consider model compression for production deployment
- Lazy loading of models to reduce startup time

### Training Time
- Air quality: ~2-3 minutes (large dataset)
- Traffic: ~30 seconds (medium dataset)  
- Housing: ~10 seconds (clustering + regression)

---

## Future Enhancements

### Model Improvements
1. **Ensemble Methods**: Combine multiple algorithms
2. **Deep Learning**: Neural networks for complex patterns
3. **Time Series**: LSTM/ARIMA for temporal predictions
4. **Feature Engineering**: Additional derived features

### Real-time Capabilities
1. **Online Learning**: Incremental model updates
2. **Streaming Data**: Real-time prediction pipeline
3. **Model Versioning**: A/B testing for model updates

### Evaluation Metrics
1. **Cross-validation**: More robust evaluation
2. **Multiple Metrics**: R², RMSE, MAPE for regression
3. **Business Metrics**: Cost-based evaluation functions