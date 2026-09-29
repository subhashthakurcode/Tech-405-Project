# European Weather Prediction: EDA, Feature Engineering & Neural Network Implementation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.7053722-blue)](https://doi.org/10.5281/zenodo.7053722)

An end-to-end pipeline covering Exploratory Data Analysis (EDA), stationarity analysis, sequence window optimisation, feature engineering, and a fully evaluated **MLP Neural Network Classifier** on the **European Weather Prediction Dataset** (ECA&D / Zenodo). Implements binary classification for Basel picnic weather suitability with hyperparameter tuning and full evaluation (Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC).

---

## 📁 Repository Structure

```
Weather Prediction/
├── .gitignore                                 # Git ignore patterns
├── README.md                                  # Project overview and GitHub push instructions
├── requirements.txt                           # Frozen Python environment dependencies
├── run_pipeline.py                            # End-to-end automated pipeline executor
├── build_notebook.py                          # Programmatic notebook generator & runner
├── data/                                      # European Weather Prediction Dataset files
│   ├── weather_prediction_dataset.csv         # Main dataset (165 features, 3,654 daily timesteps)
│   ├── weather_prediction_dataset_light.csv   # Light version (91 features)
│   ├── weather_prediction_picnic_labels.csv   # Picnic weather classification binary labels
│   ├── weather_prediction_dataset_map.jpg     # Map of 18 European weather stations
│   ├── metadata.txt                           # Variable descriptions and physical units
│   └── readme.md                              # Dataset documentation
├── src/                                       # Modular Python source package
│   ├── __init__.py
│   ├── data_loader.py                         # Data ingestion, schema & integrity validation
│   ├── eda_analysis.py                        # Statistical metrics, ADF test & ACF/PACF windowing
│   ├── feature_engineering.py                 # Cyclical, lag, rolling & spatial differential transforms
│   └── visualization.py                       # High-resolution figure generator (300 DPI)
├── figures/                                   # 10 Publication-Grade Analytical Snapshots
│   ├── 01_temperature_distributions.png       # Multi-station temperature KDE & Histograms
│   ├── 02_full_series_and_outliers.png        # 10-year continuity and outlier boxplots
│   ├── 03_seasonal_decomposition.png          # Additive decomposition (Trend, Season, Residual)
│   ├── 04_autocorrelation_acf_pacf.png        # ACF / PACF and LSTM lookback window decision (W=14)
│   ├── 05_stationarity_and_transformation.png # Raw vs Differenced series ADF tests
│   ├── 06_cross_station_correlation_heatmap.png # Spatial cross-station temperature matrix
│   ├── 07_inter_variable_relationships.png    # Radiation vs Sunshine vs Cloud Cover correlations
│   ├── 08_cyclical_and_lag_features.png       # Continuous sin/cos polar encodings and lag scatter
│   ├── 09_target_distributions_and_imbalance.png # Continuous temperature and binary picnic targets
│   └── 10_chronological_splits_and_nn_scaling.png# Time-series train/val/test splits & scaling
├── notebooks/
│   ├── weather_prediction_eda_feature_engineering.ipynb   # Week 2: EDA & Feature Engineering
│   └── weather_prediction_neural_network.ipynb            # Week 3: MLP Neural Network (NEW)
└── reports/
    └── WEATHER_PREDICTION_EDA_REPORT.md       # Full Academic Report
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd "Weather Prediction"

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Full Pipeline
```bash
python run_pipeline.py
```
This runs the full data validation, statistical tests, feature engineering transformations, PyTorch tensor builds, and saves all 10 high-resolution figures to `figures/`.

### 3. Run or View the Jupyter Notebook
```bash
# Re-generate and execute notebook from scratch
python build_notebook.py

# Or launch JupyterLab
jupyter lab notebooks/weather_prediction_eda_feature_engineering.ipynb
```

---

## 📊 Summary of Key Findings

### Week 2 — EDA & Feature Engineering
1. **Dataset Integrity (Zero Gaps):** 3,654 consecutive daily steps with **0 missing dates, 0 duplicates, 0 corrupt codes**.
2. **Stationarity (ADF Test):** Raw temperature ADF = −4.70 (p = 1.04×10⁻⁴); first-differenced ADF = −17.35 (p = 5.28×10⁻³⁰).
3. **LSTM Sequence Window W = 14 Days:** Derived from PACF sharp drop after Lag 3–5.
4. **Spatial Front Tracking:** West-to-East cross-correlation r > 0.85 (UK/France → Central Europe).
5. **Class Imbalance:** Picnic target ~25% positive / 75% negative → Weighted BCE applied in Week 3.

### Week 3 — Neural Network Implementation
6. **MLP Classifier:** 3-layer WeatherMLP (256→128→64, GELU, BatchNorm, Dropout=0.3) trained with `BCEWithLogitsLoss(pos_weight≈2.9)`.
7. **Hyperparameter Tuning:** 8 configurations compared; `Medium-LR1e3` with AdamW + CosineAnnealingLR achieved best F1.
8. **Evaluation Metrics:** See `notebooks/weather_prediction_neural_network.ipynb` for full Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC results.
9. **Threshold Optimisation:** Moving from threshold=0.5 to the optimal F1 threshold improved F1 by several percentage points.
10. **Cyclical Embeddings:** sin/cos of day-of-year and month as input features improved val F1 by ~0.03 over raw meteorological features.

---

## 📤 Publishing to Your GitHub Account

To submit this assignment on GitHub, execute the following commands in your terminal:

```bash
# 1. Initialize git
git init

# 2. Stage all files
git add .

# 3. Commit
git commit -m "feat: complete European Weather Prediction EDA, feature engineering, and report"

# 4. Create repository on GitHub (via GitHub CLI)
gh repo create weather-prediction-neural-network --public --source=. --remote=origin --push

# OR set remote manually and push:
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

---

## 📑 Full Academic Report
For the exhaustive analysis with mathematical derivations, code snapshots, and neural network readiness architecture specifications, please see:
👉 **[WEATHER_PREDICTION_EDA_REPORT.md](reports/WEATHER_PREDICTION_EDA_REPORT.md)**
# Tech-405-Project
