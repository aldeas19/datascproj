import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import yaml
from pathlib import Path

def load_config():
    """Load configuration file."""
    with open('config.yml') as f:
        return yaml.safe_load(f)

def load_data(filepath):
    """Load dataset with error handling."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def save_artifacts(scaler, numeric_cols, columns, path, label_encoder=None):
    """Saves preprocessing artifacts to a single file, optionally including label encoder."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    artifact_path = path / "preprocessing_artifacts.joblib"
    artifacts = {
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'expected_columns': columns
    }
    if label_encoder is not None:
        artifacts['label_encoder'] = label_encoder

    joblib.dump(artifacts, artifact_path)

def preprocess_new_data(df, artifacts_path):
    """Preprocess new data using saved artifacts."""
    artifacts = joblib.load(artifacts_path)
    df = df[artifacts['numeric_cols']]
    return artifacts['scaler'].transform(df)

def generate_report(output_path, **stats):
    """Generate preprocessing report."""
    with open(output_path, 'w') as f:
        f.write("=== PREPROCESSING REPORT ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")