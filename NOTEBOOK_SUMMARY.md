# 📓 Notebook Summary: European Weather Prediction — EDA, Feature Engineering & Neural Network

**Files:**  
- `notebooks/weather_prediction_eda_feature_engineering.ipynb` — Week 2: EDA & Feature Engineering  
- `notebooks/weather_prediction_neural_network.ipynb` — **Week 3: MLP Neural Network** *(NEW)*  

**Course:** Neural Networks & Deep Learning  
**Dataset:** European Climate Assessment & Dataset (ECA&D) / Zenodo  
**GitHub:** [github.com/subhashthakurcode/Tech-405-Project](https://github.com/subhashthakurcode/Tech-405-Project)  
**Date:** September 2026  

---

## 🗂️ Dataset Card

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | European Weather Prediction Dataset |
| **Source** | ECA&D — Klein Tank et al. (2002) / Zenodo DOI: `10.5281/zenodo.7053722` |
| **License** | Open Access / CC-BY 4.0 |
| **Temporal Coverage** | 2000-01-01 → 2010-01-01 (3,654 daily steps) |
| **Spatial Coverage** | 18 stations across 9 European countries |
| **Features** | 165 meteorological variables + 18 binary picnic weather labels |
| **Disk Size** | Full CSV: ~2.77 MB · Light CSV: ~1.54 MB · Picnic Labels: ~394 KB |
| **Sampling** | Regular daily observations (24-hour aggregates) |
| **Known Limitations** | Daily aggregation obscures diurnal swings; Roma station lacks precipitation; mean imputation may attenuate extreme peaks |

---

## 📋 Notebook Sections

### 1. Dataset Card & Source Attribution
- Full metadata table: source, licence, temporal/spatial coverage, disk size, collection method, and known limitations.

### 2. File Structure, Data Types & Five Random Samples
- Column dtype distribution across all 183+ columns.
- 5 random samples from key European stations (Basel, Heathrow, Roma, Sonnblick).

### 3. Data Integrity Verification
- Date continuity check: **0 gaps** in 3,654 consecutive daily records.
- Duplicate timestamp scan: **0 duplicates**.
- Null/NaN cell count: **0 missing values**.
- Corrupt sentinel scan (`-9999`): **0 corrupt entries**.
- ✅ **Integrity Status: PASSED**

### 4. Distribution Analysis & Outlier Profiling
- Descriptive statistics (mean, std, min, max, skewness, kurtosis) for Basel station.
- IQR-based outlier detection for mean temperature — only **~5%** flagged, consistent with extreme seasonal swings.
- **Figure 1:** Multi-station temperature probability density distributions (6 climate zones).

### 5. Time-Series Dynamics: 10-Year Continuity & Seasonal Decomposition
- **Figure 2:** Full 10-year temperature series with 30-day rolling average; boxplots of temperature and precipitation across stations.
- **Figure 3:** Additive decomposition (period=365 days) into Observed, Trend, Seasonal, and Residual components.

### 6. Autocorrelation (ACF), PACF & LSTM Window Selection
- **Figure 4:** ACF (lags 0–400) confirms strong annual harmonic at lag 365 (r ≈ 0.74).
- PACF (lags 0–40) shows direct AR dependency truncates sharply at lag 3–5.
- **LSTM Lookback Window Decision: W = 14 days** — covers two full synoptic cyclone cycles.

### 7. Stationarity Assessment (ADF Test) & Transformation Strategy
- **Raw series ADF:** stat = −4.70, p = 1.04×10⁻⁴ → stationary in mean, but seasonally non-stationary.
- **First-differenced series ADF:** stat = −17.35, p = 5.28×10⁻³⁰ → strictly stationary.
- **Figure 5:** Raw vs. differenced series and their distribution comparisons.
- Transformation plan: cyclical time embeddings + StandardScaler on train partition only.

### 8. Spatial Correlation & Multivariate Atmospheric Relationships
- **Figure 6:** 18×18 cross-station mean temperature correlation heatmap — Western European cluster (r > 0.95) vs. Alpine/Nordic outliers.
- **Figure 7:** Basel intra-station heatmap + Solar Radiation vs. Sunshine vs. Cloud Cover scatter.

### 9. Feature Engineering Pipeline
All logic is self-contained in the notebook. Engineered feature categories:

| Category | Features Created |
| :--- | :--- |
| **Cyclical Time Embeddings** | `sin/cos` of day-of-year, month, day-of-week (6 features) |
| **Autoregressive Lags** | T−1, T−2, T−3, T−7 for target + 3 upstream stations (40 features) |
| **Multi-Scale Rolling Stats** | 3d, 7d, 14d, 30d mean/std/min/max for temp, pressure, humidity (48 features) |
| **Spatial Pressure Differentials** | ΔP and ΔT vs. Heathrow, Tours, De Bilt, Montélimar (8 features) |
| **Prediction Targets** | Next-day temp (T+1), next-day picnic label (Y+1), rain event (R+1) |
| **Total** | **~266 engineered features** |

- **Figure 8:** Cyclical polar space visualization + autoregressive lag-1 scatter (r = 0.902).

### 10. Target Formulations & Class Imbalance Assessment
- **Regression Target:** Next-day mean temperature — near-Gaussian, μ ≈ 10.4°C, σ ≈ 7.8°C.
- **Classification Target:** Next-day picnic suitability — **~75% Negative / ~25% Positive** (imbalanced).
- Remedy: Weighted Binary Cross-Entropy (w_pos ≈ 3.0), evaluate with PR-AUC and F1-Score.
- **Figure 9:** Continuous target histogram + binary class imbalance bar chart.

### 11. Chronological Splitting & Leakage-Free Normalization

| Partition | Years | Samples | Share |
| :--- | :--- | :--- | :--- |
| **Train** | 2000–2006 | ~2,527 | ~69.7% |
| **Validation** | 2007–2008 | ~731 | ~20.2% |
| **Test** | 2009–2010 | ~365 | ~10.1% |

- `StandardScaler` fitted **strictly on Train** → applied to Val and Test (zero leakage).
- **Figure 10:** Chronological partition plot + standardized feature density curves.

### 12. PyTorch 3D Sequence Tensor Construction
- Sliding-window conversion: tabular → `(N, W, F)` tensors where W=14, F≈266.
- Shapes confirmed: `X_train (2513, 14, 266)`, `X_val (717, 14, 266)`, `X_test (351, 14, 266)`.
- Ready for direct `torch.utils.data.DataLoader` instantiation.

### 13. Three Critical Findings for Model Design

| # | Finding | Architectural Impact |
| :--- | :--- | :--- |
| **1** | Temperature has a 365-day harmonic AND 3–7 day synoptic shocks (lag-1 r=0.902) | Add **cyclical positional embeddings** fused with LSTM/Dilated CNN (W=14) |
| **2** | Western European stations lead Basel by 24–48 hrs (Atlantic jet stream, r>0.85) | Add **Cross-Station Spatial Attention** layers using upstream station features |
| **3** | Precipitation is zero-inflated (>60% zero days, skew>3.5); picnic target 75/25 imbalanced | Use **Log1p + Huber loss** for rain; **Focal Loss / Weighted BCE** for classification |

### 14. Conclusion & Next Steps
- Complete baseline established: 3,654 clean daily records, 266 features, PyTorch tensors ready.
- **Upcoming model roadmap:**
  1. MLP with Residual connections and Dropout
  2. 1D Dilated Temporal CNN
  3. Bidirectional LSTM / GRU with Temporal Attention
  4. Spatial-Temporal Transformer with Cross-Station Multi-Head Self-Attention

---

## 📊 Figures Produced (Inline in Notebook)

| Figure | Description |
| :--- | :--- |
| Fig 1 | Multi-station temperature distributions (6 European climate zones) |
| Fig 2 | Full 10-year time series + outlier boxplots |
| Fig 3 | Additive seasonal decomposition (T=365 days) |
| Fig 4 | ACF (lags 0–400) & PACF (lags 0–40) — LSTM window justification |
| Fig 5 | Stationarity check: raw vs. first-differenced distributions |
| Fig 6 | 18-station spatial cross-correlation heatmap |
| Fig 7 | Basel inter-variable relationships + solar radiation scatter |
| Fig 8 | Cyclical encodings (polar space) + autoregressive lag-1 |
| Fig 9 | Continuous target distribution + binary class imbalance |
| Fig 10 | Chronological splits + StandardScaler feature density |

---

## 🛠️ Dependencies

```
numpy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, torch, jupyter
```

Install all via:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/subhashthakurcode/Tech-405-Project.git
cd Tech-405-Project

# Install dependencies
pip install -r requirements.txt

# Open the EDA notebook (Week 2)
jupyter lab notebooks/weather_prediction_eda_feature_engineering.ipynb

# Open the Neural Network notebook (Week 3)
jupyter lab notebooks/weather_prediction_neural_network.ipynb
```

> **Note:** Both notebooks are fully executable. Run cells top-to-bottom; all outputs, figures, and metrics will be generated automatically.

---

## 🧠 Week 3 — Neural Network Implementation

### Notebook: `weather_prediction_neural_network.ipynb`

**Task:** Binary Classification — Basel Picnic Weather Suitability  
**Model:** Multi-Layer Perceptron (MLP) in PyTorch  

### Architecture: `WeatherMLP`

```
Input (169 features) 
  → Linear(169→256) → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(256→128) → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(128→64)  → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(64→1)    [logit output — BCEWithLogitsLoss]
```

- **Loss:** `BCEWithLogitsLoss` with `pos_weight ≈ 2.9` (handles 75/25 class imbalance)
- **Optimizer:** AdamW with CosineAnnealingLR scheduler
- **Gradient clipping:** max_norm = 1.0

### Hyperparameter Configurations Evaluated

| Config | Hidden Dims | Dropout | LR | Weight Decay |
| :--- | :--- | :--- | :--- | :--- |
| Small-LR1e3 | (128, 64) | 0.3 | 1e-3 | 1e-4 |
| **Medium-LR1e3 ★** | **(256, 128, 64)** | **0.3** | **1e-3** | **1e-4** |
| Large-LR5e4 | (512, 256, 128) | 0.3 | 5e-4 | 1e-4 |
| Medium-HighDropout | (256, 128, 64) | 0.5 | 1e-3 | 1e-4 |
| Medium-LowDropout | (256, 128, 64) | 0.2 | 1e-3 | 1e-4 |
| Medium-LR1e4 | (256, 128, 64) | 0.3 | 1e-4 | 1e-4 |
| Medium-HighWD | (256, 128, 64) | 0.3 | 1e-3 | 1e-3 |
| Large-LR1e3-LowDrop | (512, 256, 128) | 0.2 | 1e-3 | 1e-4 |

### Figures Produced (Fig 11–17)

| Figure | Description |
| :--- | :--- |
| Fig 11 | Hyperparameter comparison — F1, Precision, Recall, ROC-AUC bar charts |
| Fig 12 | Best model training curves — Loss & Accuracy (Train vs Val) |
| Fig 13 | Confusion matrix — raw counts + row-normalised percentages |
| Fig 14 | ROC Curve (AUC) + Precision-Recall Curve |
| Fig 15 | Threshold optimisation — F1/P/R vs decision threshold |
| Fig 16 | All 8 HP configs — val loss & val accuracy training curves |
| Fig 17 | Per-class Precision, Recall, F1 bar chart |

### Key Discoveries

1. **Class Imbalance:** Weighted BCE (`pos_weight≈2.9`) is essential — without it recall collapses below 0.30.
2. **Depth Sweet Spot:** Medium 3-layer (256→128→64) outperforms shallower and deeper architectures for this tabular dataset.
3. **Dropout:** 0.3 generalises best; 0.5 is over-regularised; 0.2 leads to mild overfitting.
4. **LR Schedule:** AdamW LR=1e-3 + CosineAnnealingLR converges fastest and most stably.
5. **Threshold Tuning:** Optimising the decision threshold yields more F1 gain than architecture changes.
6. **Cyclical Embeddings:** sin/cos of day-of-year + month improved val F1 by ~0.03 over raw features.
