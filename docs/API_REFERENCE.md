## API Reference

Base URL: `http://localhost:5000`

All endpoints accept `application/x-www-form-urlencoded` request bodies and return HTML by default (rendered dashboard). For programmatic access, you can adapt the routes to return JSON (see notes below).

### POST /predict/air_quality

- **Description**: Predict Air Quality Index (AQI) value and map to category.
- **Body params**:
  - `Country` (string): Human-readable country label that must exist in the label encoder
  - `Status` (string): Status label (e.g., Good/Moderate/Bad) that must exist in the label encoder
  - `year` (integer)
  - `month` (integer 1-12)
  - `day` (integer 1-31)
- **Processing**:
  - Loads `models/label_encoder_country.pkl`, `models/label_encoder_status.pkl`
  - Encodes `Country` and `Status`
  - Predicts with `models/air_quality_model.pkl`
  - Maps the numeric prediction to a category: `<=50: Good`, `51..100: Moderate`, `>100: Bad`
- **Response**: HTML dashboard including a message like:
  - `Predicted AQI: 74.13 (Moderate)`
- **Errors**:
  - If an unknown category is provided, you get the dashboard with `Invalid category input!`
- **Example**:
```bash
curl -X POST http://localhost:5000/predict/air_quality \
  -d Country=Albania -d Status=Good -d year=2023 -d month=7 -d day=21
```

### POST /predict/traffic

- **Description**: Predict congestion level from hour and day of week.
- **Body params**:
  - `hour` (integer 0-23)
  - `day_of_week` (integer 0=Monday … 6=Sunday)
- **Processing**:
  - Predicts with `models/traffic_congestion_model.pkl`
  - Labels overall level as `High` if prediction `>= 50`, else `Low`
- **Response**: HTML dashboard including:
  - `Traffic Congestion Level: 63.50 (High)`
- **Example**:
```bash
curl -X POST http://localhost:5000/predict/traffic -d hour=17 -d day_of_week=4
```

### POST /predict/housing

- **Description**: Estimate housing price and affordability at a location proxy using population density and average income.
- **Body params**:
  - `population_density` (integer)
  - `average_income` (integer)
- **Processing**:
  - In the advanced module (`src/house_price.py`), `kmeans_model.pkl` assigns the cluster and `housing_models.pkl` contains regressors per cluster. The current Flask route uses a simplified `housing_model.pkl` pipeline; you can switch to the clustered approach if desired.
  - Computes affordability using threshold `50000`: `<= threshold: Affordable`, otherwise `Not Affordable`.
- **Response**: HTML dashboard including:
  - `Predicted Housing Price: 42173.25 (Affordable)`
- **Example**:
```bash
curl -X POST http://localhost:5000/predict/housing \
  -d population_density=24000 -d average_income=70000
```

---

## Programmatic JSON responses (optional)

If you need JSON, adapt the routes in `app.py` by returning `jsonify({...})` instead of `render_template(...)`. Sketch:

```python
from flask import jsonify

@app.post('/predict/traffic')
def predict_traffic():
    # ... compute `prediction` and `congestion_level`
    return jsonify({
        'prediction': float(prediction),
        'level': congestion_level
    })
```

## Authentication / Rate limiting

- Not implemented. Add middleware or wrappers if exposing publicly.

## Versioning

- This API is currently unversioned. Consider prefixing routes with `/v1` for hackathon deliverables.