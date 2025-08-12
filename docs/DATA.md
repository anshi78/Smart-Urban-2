## Data Documentation

Place your datasets in the `data/` directory and update file paths inside `src/*.py` to point to these relative paths. The training scripts currently reference absolute Windows paths and should be changed.

### Air Quality dataset (`air_aquality.csv`)
- Required columns:
  - `Date` (parseable date string)
  - `Country` (category)
  - `Status` (category)
  - `AQI Value` (numeric target)
- Preprocessing performed in `src/air_quality.py`:
  - `Date` → extract `year`, `month`, `day`
  - Label encode `Country` and `Status`
  - Split into train/test

### Traffic dataset (`traffic_congestion.csv`)
- Suggested columns (as used):
  - `timestamp` (datetime string)
  - `location` (string) — dropped in preprocessing
  - `congestion_level` (numeric target)
  - `id` (any) — dropped
- Preprocessing in `src/traffic_congestion.py`:
  - Convert `timestamp` to `hour` and `day_of_week`
  - Drop `timestamp`, `id`, and `location`

### Housing dataset (`house_prediction.csv`)
- Expected columns (renamed by script):
  - `avg_area_income`, `avg_area_house_age`, `avg_area_rooms`, `avg_area_bedrooms`, `area_population`, `price`, `address`
- Preprocessing in `src/house_price.py`:
  - Drop `address`
  - Cast numeric columns; drop rows with NaNs
  - Create `population_density = area_population / avg_area_house_age`
  - Optional: KMeans clustering by `population_density`, `avg_area_income`

### File paths

Edit the CSV paths in the modules to be relative, e.g.:
```python
# src/air_quality.py
pd.read_csv('data/air_aquality.csv')

# src/traffic_congestion.py
pd.read_csv('data/traffic_congestion.csv', header=None, names=[...])

# src/house_price.py
pd.read_csv('data/house_prediction.csv')
```

### Reproducibility tips
- Fix random seeds: `random_state=42` is already used where applicable.
- Keep a copy of preprocessed datasets if training is expensive.
- Track metrics (MSE/MAE) across runs and datasets for the hackathon report.