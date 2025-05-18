import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import ttest_ind
from pathlib import Path
from .utils import load_config

def main():
    config = load_config()
    Path(config['paths']['plots']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(config['paths']['raw_data'])
    
    # 1. Basic Inspection
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    
    # 2. Target Analysis
    plt.figure(figsize=config['eda']['figure_size']['small'])
    ax = sns.countplot(x='passed', data=df, order=['no', 'yes'])
    plt.title('Class Distribution')
    ax.bar_label(ax.containers[0])
    plt.savefig(f"{config['paths']['plots']}target_dist.png", 
                bbox_inches='tight', 
                dpi=config['eda']['dpi'])
    plt.close()
    
    # 3. Numerical Features Analysis
    numeric_cols = config['eda']['numeric_cols']
    
    # Distribution plots
    for col in numeric_cols:
        plt.figure(figsize=config['eda']['figure_size']['medium'])
        sns.histplot(data=df, x=col, hue='passed', kde=True, element='step')
        plt.title(f'Distribution of {col} by Pass/Fail')
        plt.savefig(f"{config['paths']['plots']}{col}_dist.png", 
                    bbox_inches='tight',
                    dpi=config['eda']['dpi'])
        plt.close()
    
    # Correlation
    plt.figure(figsize=config['eda']['figure_size']['large'])
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Numerical Features Correlation Matrix')
    plt.savefig(f"{config['paths']['plots']}correlation_matrix.png", 
                bbox_inches='tight',
                dpi=config['eda']['dpi'])
    plt.close()
    
    # 4. Categorical Features Analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('passed')
    
    for col in categorical_cols:
        plt.figure(figsize=config['eda']['figure_size']['medium'])
        order = df[col].value_counts().index
        ax = sns.countplot(x=col, data=df, order=order)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        ax.bar_label(ax.containers[0])
        plt.savefig(f"{config['paths']['plots']}{col}_countplot.png", 
                    bbox_inches='tight',
                    dpi=config['eda']['dpi'])
        plt.close()
    
    # 5. Advanced Analysis
    # Hypothesis testing
    results = []
    for col in numeric_cols:
        group1 = df[df['passed'] == 'yes'][col]
        group2 = df[df['passed'] == 'no'][col]
        t, p = ttest_ind(group1, group2)
        results.append({
            'feature': col,
            't-statistic': t,
            'p-value': p,
            'significant': p < 0.05
        })
    
    pd.DataFrame(results).to_csv(f"{config['paths']['plots']}hypothesis_tests.csv", index=False)
    
    # Missing values visualization
    msno.matrix(df)
    plt.savefig(f"{config['paths']['plots']}missing_values.png",
                bbox_inches='tight',
                dpi=config['eda']['dpi'])
    plt.close()

if __name__ == "__main__":
    main()