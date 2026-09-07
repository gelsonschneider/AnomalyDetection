###################################################
# Gelson André Schneider
# Title : A Taxonomic Framework Based on Machine Learning for Unsupervised Anomaly Detection in Healthcare Data
# Journal: Healthcare
# Section: Artificial Intelligence in Healthcare
# Special Issue: AI-Driven Healthcare Insights
# Script for Annomaly Detection
###################################################
from matplotlib_venn import venn3
from matplotlib_venn import venn2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')
!pip install matplotlib-venn  # Install the missing library

# Settings for views
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("ANOMALY DETECTION - (16 FEATURES)")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Load processed data
df = pd.read_csv('inpatient_claims_sample1_model_features_V_EN.csv')
print(
    f"\nDataset carregado: {df.shape[0]} registros e {df.shape[1]} variáveis")

# DataSet Information
print("\nGeneral Information About the Dataset:")
print(df.info())
print("\nData Types by Column:")
print(df.dtypes.value_counts())

# =============================================================================
# 2.  2. DATA PREPARATION (ALL 16 FEATURES)
# =============================================================================

print("\n2. Preparing data with all 16 features...")

# Numerical features (13)
features_numericas = [
    'LOS', 'CLM_PMT_AMT', 'TOTALCOST', 'COST_PER_DAY',
    'NUM_DIAGNOSTICS', 'NUM_PROCEDURES', 'NUM_HCPCS',
    'PROPORTION_MEDICAR', 'EFFICIENCY', 'COMPLEXITY',
    'FLAG_PAYMENT_ZERO', 'FLAG_HIGH_COST', 'FLAG_LONG_STAY']

# Categorical features (3)
features_categoricas = ['CLM_DRG_CD_freq', 'MDC_freq', 'ADMTNG_CATEGORY_freq']

# Filter existing features
print(f"\nNumerical features: {len(features_numericas)}")
features_numericas = [col for col in features_numericas if col in df.columns]
features_categoricas = [
    col for col in features_categoricas if col in df.columns]

print(f"\nNumerical features {len(features_numericas)}")
print(f"Categorical features: {len(features_categoricas)}")
print(f"Total: {len(features_numericas) + len(features_categoricas)} features")

# Create a dataset
X = df[features_numericas].copy()

# Handling Missing Values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())  # We use the median

# Create a Complete Dataset
X = df[features_numericas + features_categoricas].copy()

# Coding Categorical Variables
print("\nEncoding Categorical Variables (Label Encoding)")
for col in features_categoricas:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('missing').astype(str)
        X[col] = le.fit_transform(X[col])

print(f"Dataset final: {X.shape[0]} records e {X.shape[1]} attributes")

# Standardization
print("\nNormalizing Data (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
print("\nApplying PCA for visualization")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(
    f"Variância explicada: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# =============================================================================
# 3. K-MEANS WITH OPTIMIZATION
# =============================================================================

print("\n" + "="*80)
print("3. K-MEANS COM OTIMIZAÇÃO")
print("="*80)

# Optimizing the Number of Clusters
print("\nOptimizing the Number of Clusters.")
n_clusters_range = range(2, 11)
silhouette_scores = []
inertias = []

for k in n_clusters_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)

    if len(set(labels_temp)) > 1:
        score = silhouette_score(X_scaled, labels_temp)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(-1)

    inertias.append(kmeans_temp.inertia_)

# View Optimization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(n_clusters_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(n_clusters_range[np.argmax(silhouette_scores)], color='red',
                linestyle='--', label=f'Melhor k={n_clusters_range[np.argmax(silhouette_scores)]}')
axes[0].set_title('The Elbow Method')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Silhouette Score
axes[1].plot(n_clusters_range, silhouette_scores,
             'ro-', linewidth=2, markersize=8)
axes[1].axvline(n_clusters_range[np.argmax(silhouette_scores)], color='red',
                linestyle='--', label=f'Best k={n_clusters_range[np.argmax(silhouette_scores)]}')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Choose the best k
best_k = n_clusters_range[np.argmax(silhouette_scores)]
print(
    f"\nOptimal number of clusters: {best_k} (Silhouette Score: {max(silhouette_scores):.3f})")

# Apply K-Means with the optimal k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Calculate the distance to the center of the cluster
kmeans_distances = np.zeros(len(X_scaled))
for i in range(best_k):
    cluster_mask = (kmeans_labels == i)
    if np.sum(cluster_mask) > 0:
        kmeans_distances[cluster_mask] = np.linalg.norm(
            X_scaled[cluster_mask] - kmeans.cluster_centers_[i], axis=1
        )

# Define anomalies (top 10%)
percentile_kmeans = 90
threshold_kmeans = np.percentile(kmeans_distances, percentile_kmeans)
kmeans_anomalies = kmeans_distances > threshold_kmeans

print(f"\nK-Means Results (k={best_k}):")
print(
    f"  - Detected anomalies: {np.sum(kmeans_anomalies)} ({np.mean(kmeans_anomalies)*100:.1f}%)")
print(f"  - Average distance: {kmeans_distances.mean():.3f}")
print(f"  - Maximum distance: {kmeans_distances.max():.3f}")
print(f"  - Threshold (p{percentile_kmeans}): {threshold_kmeans:.3f}")

# Cluster-Based Silhouette Analysis
silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
sample_silhouette_values = silhouette_samples(X_scaled, kmeans_labels)

print(f"\nCluster-based Silhouette Analysis:")
for i in range(best_k):
    cluster_silhouette = sample_silhouette_values[kmeans_labels == i].mean()
    print(f"  - Cluster {i}: {cluster_silhouette:.3f}")

# =============================================================================
# 4. ISOLATION FOREST WITH OPTIMIZATION
# =============================================================================

print("\n" + "="*80)
print("4. ISOLATION FOREST WITH OPTIMIZATION")
print("="*80)

# Grid de parâmetros para otimização
param_grid_iforest = {
    'contamination': [0.05, 0.10, 0.15],
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.5, 0.8]
}

print("\nOptimizing Isolation Forest parameters.")
iforest_results = {}
best_iforest_score = np.inf
best_iforest_params = None

for params in ParameterGrid(param_grid_iforest):
    try:
        iforest = IsolationForest(
            contamination=params['contamination'],
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            random_state=42
        )
        iforest_labels = iforest.fit_predict(X_scaled)
        iforest_scores = iforest.decision_function(X_scaled)

        # Metric: average of the scores (the lower the score, the more anomalies are detected)
        mean_score = iforest_scores.mean()

        if mean_score < best_iforest_score:
            best_iforest_score = mean_score
            best_iforest_params = params

        iforest_results[str(params)] = {
            'labels': iforest_labels,
            'scores': iforest_scores,
            'anomalies': iforest_labels == -1,
            'mean_score': mean_score
        }
    except:
        continue

print(f"\nBest parameters: {best_iforest_params}")
print(f"Highest average score: {best_iforest_score:.4f}")

# Use better parameters
best_params = best_iforest_params
iforest = IsolationForest(
    contamination=best_params['contamination'],
    n_estimators=best_params['n_estimators'],
    max_samples=best_params['max_samples'],
    random_state=42
)
iforest_labels = iforest.fit_predict(X_scaled)
iforest_scores = iforest.decision_function(X_scaled)
iforest_anomalies = iforest_labels == -1

print(f"\nIsolation Forest Results (optimized):")
print(
    f"  - Anomalies detected: {np.sum(iforest_anomalies)} ({np.mean(iforest_anomalies)*100:.1f}%)")
print(f"  - Average score: {iforest_scores.mean():.4f}")
print(f"  - Minimum score: {iforest_scores.min():.4f}")
print(f" - Maximum score: {iforest_scores.max():.4f}")

# =============================================================================
# 5.  LOCAL OUTLIER FACTOR (LOF) WITH OPTIMIZATION
# =============================================================================

print("\n" + "="*80)
print("5. LOCAL OUTLIER FACTOR (LOF) WITH OPTIMIZATION")
print("="*80)

# Parameter grid for optimization
param_grid_lof = {
    'contamination': [0.05, 0.10, 0.15],
    'n_neighbors': [10, 20, 30, 50],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

print("\nOptimizing LOF Parameters.")
lof_results = {}
best_lof_score = np.inf
best_lof_params = None

for params in ParameterGrid(param_grid_lof):
    try:
        n_neighbors = min(params['n_neighbors'], len(X_scaled) - 1)
        if n_neighbors < 2:
            continue

        lof = LocalOutlierFactor(
            contamination=params['contamination'],
            n_neighbors=n_neighbors,
            metric=params['metric']
        )
        lof_labels = lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_

        # Metric: average of the scores (the higher the score, the more anomalies are detected)
        mean_score = lof_scores.mean()

        if mean_score < best_lof_score:
            best_lof_score = mean_score
            best_lof_params = params

        lof_results[str(params)] = {
            'labels': lof_labels,
            'scores': lof_scores,
            'anomalies': lof_labels == -1,
            'mean_score': mean_score
        }
    except:
        continue

print(f"\nBest parameters: {best_lof_params}")
print(f"Highest average score: {best_lof_score:.4f}")

# Use better parameters
best_params_lof = best_lof_params
n_neighbors_opt = min(best_params_lof['n_neighbors'], len(X_scaled) - 1)
lof = LocalOutlierFactor(
    contamination=best_params_lof['contamination'],
    n_neighbors=n_neighbors_opt,
    metric=best_params_lof['metric']
)
lof_labels = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_
lof_anomalies = lof_labels == -1

print(f"\nLOF Results (optimized):")
print(
    f"  - Detected anomalies: {np.sum(lof_anomalies)} ({np.mean(lof_anomalies)*100:.1f}%)")
print(f"  - Average score: {lof_scores.mean():.4f}")
print(f"  - Minimum score: {lof_scores.min():.4f}")
print(f"  - Maximum score: {lof_scores.max():.4f}")

# =============================================================================
# 6. ADVANCED COMPARISON OF ALGORITHMS
# =============================================================================

print("\n" + "="*80)
print("6. ADVANCED COMPARISON OF ALGORITHMS")
print("="*80)

# Create a dataframe with the results
results_df = pd.DataFrame({
    'Index': np.arange(len(X_scaled)),
    'KMeans_Anomaly': kmeans_anomalies,
    'IForest_Anomaly': iforest_anomalies,
    'LOF_Anomaly': lof_anomalies,
    'KMeans_Distance': kmeans_distances,
    'IForest_Score': iforest_scores,
    'LOF_Score': lof_scores,
    'KMeans_Cluster': kmeans_labels
})

# Comparative statistics
print("\nComparison of Anomaly Counts:")
print(
    f"  - K-Means: {results_df['KMeans_Anomaly'].sum()} ({results_df['KMeans_Anomaly'].mean()*100:.1f}%)")
print(
    f"  - Isolation Forest: {results_df['IForest_Anomaly'].sum()} ({results_df['IForest_Anomaly'].mean()*100:.1f}%)")
print(
    f"  - LOF: {results_df['LOF_Anomaly'].sum()} ({results_df['LOF_Anomaly'].mean()*100:.1f}%)")

# Intersections
all_anomalies = results_df['KMeans_Anomaly'] & results_df['IForest_Anomaly'] & results_df['LOF_Anomaly']
any_anomalies = results_df['KMeans_Anomaly'] | results_df['IForest_Anomaly'] | results_df['LOF_Anomaly']
kmeans_iforest = results_df['KMeans_Anomaly'] & results_df['IForest_Anomaly']
kmeans_lof = results_df['KMeans_Anomaly'] & results_df['LOF_Anomaly']
iforest_lof = results_df['IForest_Anomaly'] & results_df['LOF_Anomaly']

print(f"\nIntersection of anomalies:")
print(
    f"  - Detected by all 3 algorithms: {all_anomalies.sum()} ({all_anomalies.mean()*100:.1f}%)")
print(
    f"  - Detected by at least one algorithm: {any_anomalies.sum()} ({any_anomalies.mean()*100:.1f}%)")
print(
    f"  - K-Means ∩ Isolation Forest: {kmeans_iforest.sum()} ({kmeans_iforest.mean()*100:.1f}%)")
print(f"  - K-Means ∩ LOF: {kmeans_lof.sum()} ({kmeans_lof.mean()*100:.1f}%)")
print(
    f"  - Isolation Forest ∩ LOF: {iforest_lof.sum()} ({iforest_lof.mean()*100:.1f}%)")

# Detailed Overlay Analysis
print(f"\nDetailed Overlay Analysis:")
categories = [
    ('Just K-Means', kmeans_anomalies & ~iforest_anomalies & ~lof_anomalies),
    ('Just IForest', ~kmeans_anomalies & iforest_anomalies & ~lof_anomalies),
    ('Just LOF', ~kmeans_anomalies & ~iforest_anomalies & lof_anomalies),
    ('K-Means + IForest', kmeans_anomalies & iforest_anomalies & ~lof_anomalies),
    ('K-Means + LOF', kmeans_anomalies & ~iforest_anomalies & lof_anomalies),
    ('IForest + LOF', ~kmeans_anomalies & iforest_anomalies & lof_anomalies),
    ('All', all_anomalies)
]

for name, mask in categories:
    print(f"  - {name}: {mask.sum()} ({mask.mean()*100:.1f}%)")

# =============================================================================
# 7. ADVANCED VIEWS
# =============================================================================

print("\n" + "="*80)
print("7. GENERATING ADVANCED VISUALIZATIONS.")
print("="*80)

# Plot 1: PCA projection with anomalies
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# K-Means
scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=kmeans_anomalies, cmap='RdYlGn_r', alpha=0.6, s=8)
axes[0, 0].set_title(f'K-Means - Anomalies (k={best_k})')
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Anomalies (1=Yes)')

# Isolation Forest
scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=iforest_anomalies, cmap='RdYlGn_r', alpha=0.6, s=8)
axes[0, 1].set_title(f'Isolation Forest - Anomalies (optimized)')
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Anomalies (1=Yes)')

# LOF
scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=lof_anomalies, cmap='RdYlGn_r', alpha=0.6, s=8)
axes[1, 0].set_title(f'LOF - Anomalies (optimized)')
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 0], label='Anomalies (1=Yes)')

# Consenso
scatter4 = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=all_anomalies, cmap='RdYlGn_r', alpha=0.6, s=8)
axes[1, 1].set_title(f'Consensus - Anomalies Detected by Everyone')
axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter4, ax=axes[1, 1], label='Anomalies (1=Yes)')

plt.tight_layout()
plt.show()

# Plot : Venn Diagram (Intersection)

fig, ax = plt.subplots(figsize=(10, 8))

# Create sets
set_kmeans = set(results_df[results_df['KMeans_Anomaly']].index)
set_iforest = set(results_df[results_df['IForest_Anomaly']].index)
set_lof = set(results_df[results_df['LOF_Anomaly']].index)

venn = venn3([set_kmeans, set_iforest, set_lof],
             ('K-Means', 'Isolation Forest', 'LOF'),
             ax=ax)
ax.set_title('Venn Diagram - Intersection of Anomalies')

# Add counts
for subset in ('100', '010', '001', '110', '101', '011', '111'):
    if subset in venn.subset_labels:
        label = venn.subset_labels[subset]
        if label is not None:
            label.set_fontsize(12)

plt.tight_layout()
plt.show()

# =============================================================================
# 8. ANALYSIS OF THE DETECTED ANOMALIES
# =============================================================================

print("\n" + "="*80)
print("8. ANALYSIS OF THE DETECTED ANOMALIES")
print("="*80)

# Add flags to the original dataset
df_anomalies = df.copy()
df_anomalies['ANOMALIA_KMEANS'] = kmeans_anomalies
df_anomalies['ANOMALIA_IFOREST'] = iforest_anomalies
df_anomalies['ANOMALIA_LOF'] = lof_anomalies
df_anomalies['ANOMALIA_CONSENSO'] = all_anomalies
df_anomalies['ANOMALIA_ANY'] = any_anomalies
df_anomalies['CLUSTER_KMEANS'] = kmeans_labels
df_anomalies['SCORE_KMEANS'] = kmeans_distances
df_anomalies['SCORE_IFOREST'] = iforest_scores
df_anomalies['SCORE_LOF'] = lof_scores

# Comparison of Means (Normal vs. Anomaly - Consensus)
print("\nComparison of Means (Normal vs. Anomaly - Consensus):")
features_analise = ['LOS', 'CLM_PMT_AMT', 'TOTALCOST', 'NUM_DIAGNOSTICS',
                    'NUM\_PROCEDURES', 'NUM_HCPCS', 'COMPLEXITY']
features_analise = [
    col for col in features_analise if col in df_anomalies.columns]

comparison_data = []
for feature in features_analise:
    normal_mean = df_anomalies[~df_anomalies['ANOMALIA_CONSENSO']][feature].mean(
    )
    anomaly_mean = df_anomalies[df_anomalies['ANOMALIA_CONSENSO']][feature].mean(
    )
    diff_pct = ((anomaly_mean - normal_mean) /
                normal_mean * 100) if normal_mean != 0 else 0

    comparison_data.append({
        'Feature': feature,
        'Normal': normal_mean,
        'Anomaly': anomaly_mean,
        'Difference %': diff_pct
    })

    print(f"\n{feature}:")
    print(f"  - Normal: {normal_mean:.2f}")
    print(f"  - Anomaly (consensus): {anomaly_mean:.2f}")
    print(f"  - Difference: {diff_pct:.1f}%")

# Comparative DataFrame
comparison_df = pd.DataFrame(comparison_data)
print("\nComparative Summary:")
print(comparison_df.to_string(index=False))

# DRG Analysis
if 'CLM_DRG_CD_freq' in df_anomalies.columns:
    print("\nTop 10 DRGs with the Most Anomalies (consensus):")
    drg_anomaly = df_anomalies[df_anomalies['ANOMALIA_CONSENSO']
                               ]['CLM_DRG_CD_freq'].value_counts()
    drg_total = df_anomalies['CLM_DRG_CD_freq'].value_counts()

    drg_analysis = pd.DataFrame({
        'Total': drg_total,
        'Anomalies': drg_anomaly,
        'Anomaly_Rate': (drg_anomaly / drg_total * 100).fillna(0)
    }).sort_values('AnomalAnomaliesias', ascending=False)

    print(drg_analysis.head(10))

    # VisualizaçãoPreview
    fig, ax = plt.subplots(figsize=(12, 6))
    drg_analysis.head(10)['Taxa_Anomalia'].plot(
        kind='bar', ax=ax, color='coral')
    ax.set_title('Anomaly Rate by DRG (Consensus)')
    ax.set_xlabel('DRG Code')
    ax.set_ylabel('Anomaly Rate (%)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================

print("\n" + "="*80)
print("9. SAVING RESULTS")
print("="*80)

# Save dataset with flags
output_file = 'inpatient_anomalies_improved_16features.csv'
df_anomalies.to_csv(output_file, index=False)
print(f"\nDataset with flags saved as: {output_file}")
print(f"  - Shape: {df_anomalies.shape}")
print(f"  - Columns: {df_anomalies.columns.tolist()}")

# Save metrics
metrics_data = {
    'Algorithm': ['K-Means', 'Isolation Forest', 'LOF'],
    'N_Anomalias': [
        np.sum(kmeans_anomalies),
        np.sum(iforest_anomalies),
        np.sum(lof_anomalies)
    ],
    'Percentual': [
        np.mean(kmeans_anomalies) * 100,
        np.mean(iforest_anomalies) * 100,
        np.mean(lof_anomalies) * 100
    ],
    'Score_Medio': [
        kmeans_distances.mean(),
        iforest_scores.mean(),
        lof_scores.mean()
    ],
    'Parametros': [
        f'k={best_k} (Silhouette={max(silhouette_scores):.3f})',
        f'cont={best_params["contamination"]:.0%}, n_estim={best_params["n_estimators"]}',
        f'n_neighbors={best_params_lof["n_neighbors"]}, metric={best_params_lof["metric"]}'
    ]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_file = 'anomaly_detection_improved_16features_metrics.csv'
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics saved as: {metrics_file}")

# Save feature comparison
comparison_file = 'anomaly_features_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"Feature comparison saved as: {comparison_file}")

# Save DRG Analysis
if 'CLM_DRG_CD' in df_anomalies.columns:
    drg_file = 'anomaly_drg_analysis.csv'
    drg_analysis.to_csv(drg_file)
    print(f" DRG Analysis saved as: {drg_file}")

# Save Detailed Intersection
intersection_data = []
for name, mask in categories:
    intersection_data.append({
        'Category': name,
        'Quantity': mask.sum(),
        'Percentage': mask.mean() * 100
    })
intersection_df = pd.DataFrame(intersection_data)
intersection_file = 'anomaly_intersection_analysis.csv'
intersection_df.to_csv(intersection_file, index=False)
print(f"Intersection analysis saved as: {intersection_file}")
