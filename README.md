# European Weather Prediction: Exploratory Data Analysis & Feature Engineering

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.7053722-blue)](https://doi.org/10.5281/zenodo.7053722)

An end-to-end Exploratory Data Analysis (EDA), stationarity analysis, sequence window optimization, and feature engineering pipeline on the **European Weather Prediction Dataset** (ECA&D / Zenodo), prepared for deep learning and neural network implementations (MLP, 1D-CNN, LSTM/GRU, Spatial-Temporal Transformers).

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
│   └── weather_prediction_eda_feature_engineering.ipynb  # Executed, self-contained Jupyter Notebook
└── reports/
    └── WEATHER_PREDICTION_EDA_REPORT.md       # Full Academic Submission Report with Snapshots
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

1. **Dataset Integrity (Zero Gaps):** The 10-year dataset (2000-01-01 to 2010-01-01) contains exactly 3,654 consecutive daily steps with **0 missing dates, 0 duplicates, and 0 corrupt codes**.
2. **Stationarity Diagnostics (ADF Test):** Raw temperature yields $\text{ADF} = -4.6952$ ($p = 1.04 \times 10^{-4}$); first-differenced $\Delta T$ yields $\text{ADF} = -17.3493$ ($p = 5.28 \times 10^{-30}$).
3. **LSTM Sequence Window ($W = 14$ Days):** Derived from PACF sharp drop after Lag 3–5 and the 3–7 day European mid-latitude cyclonic storm lifespan.
4. **Spatial Front Tracking:** Strong West-to-East cross-correlation ($r > 0.85$ between UK/France and Central Europe) proves that upstream stations provide 24–48 hour predictive lead time.
5. **Class Imbalance & Loss Strategy:** The picnic target is imbalanced (~25% positive / 75% negative). In the next assignment, we will use **Weighted BCE ($w_{\text{pos}}=3.0$) or Focal Loss** and evaluate on **PR-AUC and F1-score**.

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
