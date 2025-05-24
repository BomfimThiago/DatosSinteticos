import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compare_distributions(original_df, synthetic_df, column):
    """Compare distributions of a column between original and synthetic data"""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    sns.histplot(data=original_df[column], label='Original', alpha=0.5, stat='density')
    sns.histplot(data=synthetic_df[column], label='Synthetic', alpha=0.5, stat='density')
    
    plt.title(f'Distribution Comparison - {column}')
    plt.legend()
    plt.show()
    
    # Calculate statistics
    stats_original = original_df[column].describe()
    stats_synthetic = synthetic_df[column].describe()
    
    # Calculate KS test
    ks_stat, p_value = stats.ks_2samp(original_df[column], synthetic_df[column])
    
    print(f"\nStatistical Comparison for {column}:")
    print("\nOriginal Data Statistics:")
    print(stats_original)
    print("\nSynthetic Data Statistics:")
    print(stats_synthetic)
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")


def compare_datasets(original_df, synthetic_df):
    """Compare overall dataset statistics"""
    print("=== Overall Dataset Comparison ===")
    print("\nOriginal Dataset Shape:", original_df.shape)
    print("Synthetic Dataset Shape:", synthetic_df.shape)
    
    # Compare basic statistics for all numerical columns
    print("\n=== Numerical Columns Statistics ===")
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        print(f"\n{col}:")
        print("Original - Mean:", original_df[col].mean(), "Std:", original_df[col].std())
        print("Synthetic - Mean:", synthetic_df[col].mean(), "Std:", synthetic_df[col].std())
        
        # KS test for distribution similarity
        ks_stat, p_value = stats.ks_2samp(original_df[col], synthetic_df[col])
        print(f"KS Test p-value: {p_value:.4f}")
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=original_df[col], label='Original', alpha=0.5)
        sns.kdeplot(data=synthetic_df[col], label='Synthetic', alpha=0.5)
        plt.title(f'Distribution of {col}')
        plt.legend()
        plt.show()

    # Compare categorical columns
    print("\n=== Categorical Columns Statistics ===")
    categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        print(f"\n{col}:")
        # Calculate proportions
        orig_prop = original_df[col].value_counts(normalize=True)
        synth_prop = synthetic_df[col].value_counts(normalize=True)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Original': orig_prop,
            'Synthetic': synth_prop
        }).fillna(0)
        
        print(comparison)
        
        # Plot categorical distributions
        plt.figure(figsize=(10, 6))
        comparison.plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Compare correlations
    print("\n=== Correlation Comparison ===")
    # Calculate correlations
    corr_original = original_df[numerical_cols].corr()
    corr_synthetic = synthetic_df[numerical_cols].corr()
    
    # Plot correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_title('Original Data Correlations')
    sns.heatmap(corr_synthetic, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Synthetic Data Correlations')
    plt.tight_layout()
    plt.show()
    
    # Show correlation differences
    corr_diff = corr_original - corr_synthetic
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_diff, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Differences (Original - Synthetic)')
    plt.tight_layout()
    plt.show()
