# AnomalyDetection

**A Taxonomic Framework Based on Machine Learning for Unsupervised Anomaly Detection in Healthcare Data**

---

## Overview

This repository contains the data and scripts developed for the research article:

> **"A Taxonomic Framework Based on Machine Learning for Unsupervised Anomaly Detection in Healthcare Data"**  
> *Submitted to the Journal Healthcare*

The project implements and evaluates an unsupervised machine learning framework for detecting anomalies in healthcare datasets, with a focus on inpatient claims data.

---

## Repository Structure

### Scripts Description

| File | Description |
|------|-------------|
| `annomaly_detection_v_en.py` | Core implementation of the taxonomic framework for unsupervised anomaly detection. Contains machine learning models and evaluation metrics. |
| `edadoutorado_v_en.py` | Comprehensive exploratory data analysis script, including data preprocessing, visualization, and statistical summaries. |
| `inpatient_claims_sample1_model_features_V_EN.csv` | Sample dataset containing anonymized inpatient claims with engineered features for model training and evaluation. |

---

## Methodology

The proposed taxonomic framework organizes unsupervised anomaly detection methods into a structured hierarchy tailored for healthcare data. The approach:

1. **Preprocesses** healthcare claims data with feature engineering
2. **Applies** multiple unsupervised learning algorithms (isolation forest, autoencoders, clustering-based methods, etc.)
3. **Evaluates** detection performance using healthcare-specific metrics
4. **Provides** a taxonomic classification of anomalies found in the data

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see script imports):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib/seaborn (for visualization)
  - Additional ML libraries as specified in the scripts

### Installation

```bash
# Clone the repository
git clone https://github.com/gelsonschneider/AnomalyDetection.git

# Navigate to the directory
cd AnomalyDetection

# Install dependencies (recommended to use a virtual environment)
pip install -r requirements.txt  # If available
