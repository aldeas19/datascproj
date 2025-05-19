import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import joblib
from pathlib import Path
from datetime import datetime
from .utils import load_config, load_data, save_artifacts, generate_report

def clean_raw_data(df, config):
    """Apply basic cleaning: clipping and rare category replacement."""
    # Clip absences
    absences_clip = config['preprocessing']['clip']['absences_clip']
    df['absences'] = df['absences'].clip(upper=absences_clip)

    # Replace rare categories
    for col in ['Mjob', 'Fjob']:
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < config['preprocessing']['encoding']['rare_category_threshold']].index
        df[col] = df[col].replace(rare, config['preprocessing']['encoding']['rare_category_name'])

    return df

def validate_preprocessing(df, config):
    """Validate that preprocessing addressed EDA findings."""
    absences_clip = config['preprocessing']['clip']['absences_clip']
    assert df['absences'].max() <= absences_clip, "Absences not clipped properly"

    for col in ['Mjob', 'Fjob']:
        counts = df[col].value_counts(normalize=True)
        assert all(counts >= config['preprocessing']['encoding']['rare_category_threshold']), f"Rare categories not handled in {col}"

    # Ensure both classes are represented in the target variable
    assert df['passed'].nunique() == 2, "Target variable does not have both classes represented"

def preprocess_data(df, config):
    """Main preprocessing pipeline."""
    # Separate features and target BEFORE any preprocessing
    X = df.drop('passed', axis=1)
    y = df['passed']

    # Encode target labels ('yes'/'no' -> 1/0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config['preprocessing']['sampling']['test_size'],
        random_state=config['preprocessing']['sampling']['random_state'],
        stratify=y_encoded
    )

    # Encode categorical features with one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Ensure test set has same columns as train set
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Balance training set classes
    ros = RandomOverSampler(random_state=config['preprocessing']['sampling']['random_state'])
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # Scale numeric columns only
    numeric_cols = config['eda']['numeric_cols']
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler, numeric_cols, le

def main():
    config = load_config()
    df = load_data(config['paths']['raw_data'])

    # Clean and validate raw data
    df = clean_raw_data(df, config)
    validate_preprocessing(df, config)

    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, numeric_cols, label_encoder = preprocess_data(df, config)

    # Combine features and encoded target for saving
    train_processed = pd.concat([X_train, pd.Series(y_train, name='passed')], axis=1)
    test_processed = pd.concat([X_test, pd.Series(y_test, name='passed')], axis=1)

    # Create output directories if needed
    output_dir = Path(config['paths']['processed']['train']).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_processed.to_csv(config['paths']['processed']['train'], index=False)
    test_processed.to_csv(config['paths']['processed']['test'], index=False)

    # Save preprocessing artifacts (scaler, numeric cols, columns order, label encoder)
    save_artifacts(
        scaler=scaler,
        numeric_cols=numeric_cols,
        columns=X_train.columns.tolist(),
        label_encoder=label_encoder,
        path=config['paths']['artifacts']
    )

    # Create reports directory if missing
    reports_dir = Path(config['paths']['reports'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    generate_report(
        report_path,
        Original_samples=len(df),
        Train_samples=len(X_train),
        Test_samples=len(X_test),
        Numerical_features=numeric_cols,
        Absences_clipping_threshold=config['preprocessing']['clip']['absences_clip']
    )

if __name__ == "__main__":
    main()
