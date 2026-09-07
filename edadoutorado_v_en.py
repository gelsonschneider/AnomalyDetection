
###################################################
# Gelson André Schneider
# Title : A Taxonomic Framework Based on Machine Learning for Unsupervised Anomaly Detection in Healthcare Data
# Journal: Healthcare
# Section: Artificial Intelligence in Healthcare
# Special Issue: AI-Driven Healthcare Insights
# Script for Descriptive Statistics of the data
###################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera, shapiro
import warnings
warnings.filterwarnings('ignore')

# Settings for views
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("="*80)
print("EXPLORATORY DATA ANALYSIS - INPATIENT CLAIMS SAMPLE 1")
print("="*80)

# Load processed data
df = pd.read_csv('inpatient_claims_sample1_model_features_V_EN.csv')
print(
    f"\nDataset loaded: {df.shape[0]} registros e {df.shape[1]} attributes")

# =============================================================================
# 2. GENERAL DESCRIPTIVE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("2. GENERAL DESCRIPTIVE ANALYSIS")
print("="*80)

print("\nData Types by Column:")
print(df.dtypes.value_counts())

print("\nGeneral Information About the Dataset:")
print(df.info())

print("\nStatistical summary of numerical variables:")
display(df.describe())

# =============================================================================
# 3. ANALYSIS OF MISSING VALUES
# =============================================================================

print("\n" + "="*80)
print("3. ANALYSIS OF MISSING VALUES")
print("="*80)

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    print(
        f"\nTotal number of columns with missing values: {len(missing_data)}")
    print("\nMissing values by column:")
    print(missing_data)

    # Preview
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_data.plot(kind='bar', ax=ax)
    ax.set_title('Missing Values by Column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Number of Missing Values')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo missing values found!")

# =============================================================================
# 4. DISTRIBUTION CHARTS - NUMERICAL ATTRIBUTES
# =============================================================================

print("\n" + "="*80)
print("4. DISTRIBUTION CHARTS")
print("="*80)

# Function to create a grid of charts


def plot_distribution_grid(df, columns, n_cols=3, figsize=(18, 12)):
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            # Histograma
            axes[i].hist(df[col].dropna(), bins=50, alpha=0.7,
                         edgecolor='black', linewidth=0.5)
            axes[i].axvline(df[col].mean(), color='red',
                            linestyle='--', label=f'Average: {df[col].mean():.2f}')
            axes[i].axvline(df[col].median(), color='green',
                            linestyle='--', label=f'Median: {df[col].median():.2f}')
            axes[i].set_title(f'Distribution: {col}')
            axes[i].set_xlabel('Amount')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Plot the distributions of the main variables
print("\nDistribution of the main numerical variables:")
plot_distribution_grid(
    df, main_numeric_existentes[:6], n_cols=3, figsize=(18, 12))

# =============================================================================
# 5. ANALYSIS OF CATEGORICAL VARIABLES
# =============================================================================

print("\n" + "="*80)
print("5. ANALYSIS OF CATEGORICAL VARIABLES")
print("="*80)

# Identificar variáveis categóricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariáveis categóricas: {len(categorical_cols)} colunas")
print(categorical_cols)

# Análise de DRG
if 'CLM_DRG_CD_freq' in df.columns:
    print("\n" + "="*50)
    print("DRG Analysis (Diagnosis Related Group)")
    print("="*50)

    drg_stats = df['CLM_DRG_CD_freq'].value_counts()

    # Gráfico dos top DRGs
    fig, ax = plt.subplots(figsize=(12, 8))
    drg_stats.head(15).plot(kind='bar', ax=ax)
    ax.set_title('Most Common Diagnosis-Related Groups (DRG)')
    ax.set_xlabel('DRG Code')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("\n No categorical variables were found!")

# Diagnostic Evaluation upon Admission
if 'ADMTNG_CATEGORY_freq' in df.columns:
    print("\n" + "="*50)
    print("Analysis of the Admission Diagnosis Category")
    print("="*50)

    diag_cat_stats = df['ADMTNG_CATEGORY_freq'].value_counts()

    # Chart PLOT
    fig, ax = plt.subplots(figsize=(12, 8))
    diag_cat_stats.head(15).plot(kind='bar', ax=ax)
    ax.set_title('Admission Diagnosis Categories')
    ax.set_xlabel('Admission Diagnosis')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. ANALYSIS OF BIVARIATE RELATIONSHIPS
# =============================================================================

print("\n" + "="*80)
print("6. ANALYSIS OF BIVARIATE RELATIONSHIPS")
print("="*80)


# Analysis of Correlations Between Categorical and Continuous Variables
if 'CLM_DRG_CD_freq' in df.columns and 'TOTALCOST' in df.columns:
    print("\nAverage Cost per DRG (Top 10):")
    custo_por_drg = df.groupby('CLM_DRG_CD_freq')['TOTALCOST'].agg(
        ['mean', 'median', 'count']).sort_values('mean', ascending=False)
    print(custo_por_drg.head(10))

    # Chart Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    custo_por_drg.head(10)['mean'].plot(kind='bar', ax=ax)
    ax.set_title('DRGs with the Highest Average Cost')
    ax.set_xlabel('DRG Code')
    ax.set_ylabel('Average Cost')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
