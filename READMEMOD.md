# Model Artifacts Documentation

## preprocessing_artifacts.pkl

### Contents
- `scaler`: MinMaxScaler instance fitted on training data
- `numeric_cols`: List of numerical column names that were scaled
- `expected_columns`: Full list of feature names after one-hot encoding

### Usage
```python
import joblib

# Load artifacts
artifacts = joblib.load('preprocessing_artifacts.pkl')

# Preprocess new data
new_data[numeric_cols] = artifacts['scaler'].transform(new_data[numeric_cols])