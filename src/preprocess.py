import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
from pathlib import Path
from .utils import load_config, load_data, save_artifacts, generate_report

def preprocess_data(df, config):
    """Main preprocessing pipeline."""
    # 1. Clean numerical features - usa valor fixo para clipping
    absences_clip = config['preprocessing']['clip']['absences_clip']  # valor fixo, ex: 10
    df['absences'] = df['absences'].clip(upper=absences_clip)
    
    # 2. Handle categorical features
    for col in ['Mjob', 'Fjob']:
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < config['preprocessing']['encoding']['rare_category_threshold']].index
        df[col] = df[col].replace(rare, config['preprocessing']['encoding']['rare_category_name'])
    
    # 3. Split data
    X = df.drop('passed', axis=1)
    y = df['passed']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['preprocessing']['sampling']['test_size'],
        random_state=config['preprocessing']['sampling']['random_state'],
        stratify=y
    )
    
    # 4. Encode categoricals
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    
    # Ensure consistent columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # 5. Handle class imbalance
    ros = RandomOverSampler(random_state=config['preprocessing']['sampling']['random_state'])
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    # 6. Scale numerical features
    numeric_cols = config['eda']['numeric_cols']
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler, numeric_cols, absences_clip 

def validate_preprocessing(df, config):
    """Validate that preprocessing addressed EDA findings"""
    # Check absences clipping - compara com valor fixo do config
    absences_clip = config['preprocessing']['clip']['absences_clip']
    assert df['absences'].max() <= absences_clip, "Absences not clipped properly"
    
    # Check rare categories
    for col in ['Mjob', 'Fjob']:
        counts = df[col].value_counts(normalize=True)
        assert all(counts >= config['preprocessing']['encoding']['rare_category_threshold']), f"Rare categories not handled in {col}"
    
    print("All preprocessing validations passed!")

def main():
    config = load_config()
    df = load_data(config['paths']['raw_data'])
    
    # Process data primeiro para aplicar clipping e encoding
    X_train, X_test, y_train, y_test, scaler, numeric_cols, absences_clip = preprocess_data(df, config)
    
    # Crie um df pós-processado só com treino para validação
    df_processed = X_train.copy()
    df_processed['passed'] = y_train
    
    # Agora valide usando o df já com clip aplicado
    validate_preprocessing(df_processed, config)
    
    # Prepare outputs
    train_processed = pd.concat([X_train, y_train], axis=1)
    test_processed = pd.concat([X_test, y_test], axis=1)
    
    # Salva arquivos
    Path(config['paths']['processed']['train']).parent.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(config['paths']['processed']['train'], index=False)
    test_processed.to_csv(config['paths']['processed']['test'], index=False)
    
    save_artifacts(
        scaler=scaler,
        numeric_cols=numeric_cols,
        columns=X_train.columns.tolist(),
        path=config['paths']['artifacts']
    )
    
    # Gera relatório
    report_stats = {
        "Original samples": len(df),
        "Train samples": len(X_train),
        "Test samples": len(X_test),
        "Numerical features": numeric_cols,
        "Absences clipping threshold": absences_clip
    }
    generate_report(config['paths']['reports'], **report_stats)
