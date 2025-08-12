## Smart Urban Settlement Management System

A Flask-based application that brings together three urban analytics capabilities:

- Air quality prediction (AQI category)
- Traffic congestion prediction (congestion level)
- Affordable housing estimation (price and affordability)

This repository includes a simple HTML frontend, Python model training scripts, and server endpoints to make real-time predictions via web forms.

### Table of contents
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [APIs overview](#apis-overview)
- [Python module usage](#python-module-usage)
- [Datasets](#datasets)
- [Training models](#training-models)
- [Examples](#examples)
- [Docs](#docs)

## Quick start

```bash
# 1) Create a virtual environment (optional but recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Ensure trained models exist in `models/` (see Training models below)
# 4) Run the web app
python app.py

# The app will run on http://localhost:5000
```

## Project structure

```
/workspace/
├── app.py                        # Flask server with prediction endpoints
├── requirements.txt
├── src/
│   ├── air_quality.py            # AQI preprocessing, training, and prediction helpers
│   ├── traffic_congestion.py     # Traffic preprocessing, training, and prediction helpers
│   └── house_price.py            # Housing preprocessing, clustering + regression, prediction helpers
├── templates/
│   ├── dashboard.html            # Single-page dashboard with three forms
│   └── index.html                # Basic landing page (not routed by default)
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/                   # Plots saved by training scripts
├── models/                       # Saved .pkl models and encoders (created by training scripts)
└── data/                         # Place your CSV datasets here (update paths in src/*.py)
```

## Architecture

- **Frontend**: HTML templates rendered by Flask (`templates/dashboard.html`) with simple forms posting to server endpoints.
- **Backend**: Flask app (`app.py`) exposes three POST routes that load pre-trained models from `models/` and return rendered results.
- **Models**: `src/` contains training utilities using scikit-learn. Trained artifacts are serialized with `joblib` into the `models/` directory.

## APIs overview

The app primarily serves HTML. All endpoints expect `application/x-www-form-urlencoded` form fields and respond with an HTML page embedding the prediction.

- POST `/predict/air_quality`
  - Fields: `Country` (str), `Status` (str), `year` (int), `month` (int), `day` (int)
  - Loads encoders `label_encoder_country.pkl`, `label_encoder_status.pkl` and `air_quality_model.pkl`
  - Response: Dashboard HTML with "Predicted AQI: <value> (<Good|Moderate|Bad>)"

- POST `/predict/traffic`
  - Fields: `hour` (int, 0-23), `day_of_week` (int, 0=Monday … 6=Sunday)
  - Loads `traffic_congestion_model.pkl`
  - Response: Dashboard HTML with "Traffic Congestion Level: <value> (<Low|High>)"

- POST `/predict/housing`
  - Fields: `population_density` (int), `average_income` (int)
  - Loads `kmeans_model.pkl` and `housing_models.pkl` (clustered regressors); the Flask app currently loads `housing_model.pkl` for a simplified demo
  - Response: Dashboard HTML with "Predicted Housing Price: <value> (<Affordable|Not Affordable>)"

For complete parameter details and cURL examples, see `docs/API_REFERENCE.md`.

## Python module usage

You can also use the training and prediction helpers directly in Python. See `docs/PYTHON_MODULES.md` for full signatures and examples of:

- `src/air_quality.py` → `preprocess_and_train()`, `predict_aqi(...)`
- `src/traffic_congestion.py` → `train_traffic_model()`, `predict_traffic_congestion(...)`
- `src/house_price.py` → `train_housing_model()`, `predict_housing_demand(...)`, `plot_predictions(...)`

## Datasets

The training scripts expect CSV files for each domain. Update the dataset paths in `src/*.py` to point to your local `data/` directory. See `docs/DATA.md` for schema details and column expectations.

## Training models

Run any of the following from the repository root to generate model artifacts in `models/`:

```bash
# Air quality (saves air_quality_model.pkl and encoders)
python src/air_quality.py

# Traffic (saves traffic_congestion_model.pkl and a plot in static/images)
python src/traffic_congestion.py

# Housing (saves kmeans_model.pkl and housing_models.pkl, plus plot)
python src/house_price.py
```

Important: The scripts currently reference absolute Windows-style file paths. Edit them to point to your local CSVs, for example:

```python
# Example: change in src/air_quality.py
pd.read_csv('data/air_aquality.csv')
```

## Examples

### Call endpoints via cURL

```bash
# Air quality
curl -X POST http://localhost:5000/predict/air_quality \
  -d Country=Albania -d Status=Good -d year=2022 -d month=7 -d day=21

# Traffic
curl -X POST http://localhost:5000/predict/traffic \
  -d hour=17 -d day_of_week=4

# Housing
curl -X POST http://localhost:5000/predict/housing \
  -d population_density=24000 -d average_income=70000
```

### Use Python helpers

```python
from src.air_quality import preprocess_and_train, predict_aqi
model = preprocess_and_train()
pred, category = predict_aqi('Albania', 'Good', 2022, 7, 21)
print(pred, category)
```

## Docs

- `docs/API_REFERENCE.md` — endpoints, parameters, responses, and examples
- `docs/PYTHON_MODULES.md` — public functions and usage examples
- `docs/FRONTEND.md` — templates, forms, and how the UI ties to endpoints
- `docs/DATA.md` — dataset schemas and preprocessing expectations
- `docs/DEPLOYMENT.md` — running locally and deployment notes