## Python Modules Reference

This document describes the public functions intended for reuse from the `src/` package. All functions assume you have installed dependencies from `requirements.txt`.

---

### Module: `src/air_quality.py`

#### preprocess_and_train() -> RandomForestRegressor
- **Description**: Loads the air quality dataset, applies date parsing and label encoding for `Country` and `Status`, splits data, trains a `RandomForestRegressor`, saves the model and encoders into `models/`.
- **Saves**:
  - `models/air_quality_model.pkl`
  - `models/label_encoder_country.pkl`
  - `models/label_encoder_status.pkl`
- **Returns**: The trained `RandomForestRegressor` instance.
- **Usage**:
```python
from src.air_quality import preprocess_and_train
model = preprocess_and_train()
```

#### predict_aqi(country: str, status: str, year: int, month: int, day: int) -> tuple[float, str]
- **Description**: Loads the saved model and encoders, transforms inputs, predicts AQI, and maps it to category.
- **Returns**: `(prediction_value, category)` where category is one of `Good | Moderate | Bad`.
- **Usage**:
```python
from src.air_quality import predict_aqi
pred, category = predict_aqi('Albania', 'Good', 2023, 7, 21)
```

---

### Module: `src/traffic_congestion.py`

#### preprocess_traffic_data() -> pandas.DataFrame
- **Description**: Loads and cleans the raw traffic CSV, converting `timestamp` to `hour` and `day_of_week`, and dropping unused columns.
- **Usage**:
```python
from src.traffic_congestion import preprocess_traffic_data
df = preprocess_traffic_data()
```

#### train_traffic_model() -> None
- **Description**: Trains a `RandomForestRegressor` on `hour` and `day_of_week`, evaluates with MAE, saves the model, and creates a diagnostic scatter plot at `static/images/actual_vs_predicted_congestion.png`.
- **Saves**:
  - `models/traffic_congestion_model.pkl`
  - `static/images/actual_vs_predicted_congestion.png`
- **Usage**:
```python
from src.traffic_congestion import train_traffic_model
train_traffic_model()
```

#### predict_traffic_congestion(hour: int, day_of_week: int) -> float | None
- **Description**: Loads the serialized model and predicts congestion level for the provided time.
- **Returns**: The numeric congestion level or `None` if the model file is missing.
- **Usage**:
```python
from src.traffic_congestion import predict_traffic_congestion
pred = predict_traffic_congestion(17, 4)
```

#### plot_predictions(y_test, y_pred) -> None
- **Description**: Utility to visualize predicted vs actual values and save to `static/images/actual_vs_predicted_congestion.png`.

---

### Module: `src/house_price.py`

#### preprocess_housing_data() -> pandas.DataFrame
- **Description**: Loads the housing dataset, cleans columns, converts numeric fields, and constructs `population_density`.

#### apply_kmeans_clustering(data: pandas.DataFrame, n_clusters: int = 3) -> tuple[pandas.DataFrame, KMeans]
- **Description**: Assigns a `cluster` to each row using KMeans over `population_density` and `avg_area_income`.

#### train_housing_model() -> tuple[dict[int, LinearRegression], KMeans]
- **Description**: Performs clustering, trains a `LinearRegression` per cluster, and persists artifacts.
- **Saves**:
  - `models/kmeans_model.pkl`
  - `models/housing_models.pkl` (dict from cluster id to model)
- **Usage**:
```python
from src.house_price import train_housing_model
models, kmeans = train_housing_model()
```

#### predict_housing_demand(new_data: pandas.DataFrame, affordability_threshold: int = 50000) -> tuple[numpy.ndarray | None, list[str] | None]
- **Description**: Loads artifacts, predicts prices using the correct cluster-specific model, and computes affordability labels.
- **Parameters**:
  - `new_data`: DataFrame with columns `population_density`, `avg_area_income`.
  - `affordability_threshold`: Price threshold for affordability classification.
- **Returns**: `(predictions_array, affordability_labels)` or `(None, None)` if artifacts are missing.
- **Usage**:
```python
import pandas as pd
from src.house_price import predict_housing_demand

X = pd.DataFrame({
    'population_density': [24000, 53000],
    'avg_area_income': [70000, 69000]
})
prices, labels = predict_housing_demand(X)
```

#### plot_predictions(actual: list | numpy.ndarray, predicted: list | numpy.ndarray) -> None
- **Description**: Saves a line plot of actual vs predicted prices to `static/images/housing_demand_plot.png`.

---

## Notes and caveats

- File paths to CSVs in the modules are currently absolute and Windows-specific. Replace with relative paths into the `data/` directory before training.
- Models are persisted with `joblib` into the `models/` folder expected by `app.py`.
- Matplotlib requires a display backend; when running on a server, ensure a non-interactive backend or rely on `plt.savefig` only (already used here).