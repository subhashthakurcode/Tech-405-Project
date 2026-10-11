# Assignment 6 Report: Comparative Machine Learning & Deep Learning Classification

**Course:** Tech-405 Neural Networks & Deep Learning  
**Project:** Telco Customer Churn Binary Classification  
**Dataset:** Telco Customer Churn ([Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) & [Hugging Face Hub](https://huggingface.co/datasets))  
**Student:** Subhash Thakur  
**GitHub Repository:** [https://github.com/subhashthakurcode/Tech-405-Project](https://github.com/subhashthakurcode/Tech-405-Project)  
**Date:** October 2026  
**Word Document Report:** [week6assignment/Assignment_6_Classification_Report.docx](Assignment_6_Classification_Report.docx)  

---

## 1. Executive Summary & Problem Formulation

For **Assignment 6**, we selected a high-impact real-world business classification challenge: **Customer Churn Prediction** in telecommunications. Telecommunication providers face significant customer acquisition costs ($500+ per subscriber); retaining existing accounts via proactive retention offers is significantly more profitable than replacing lost customers.

The objective is to train, evaluate, and benchmark two distinct machine learning models:
1. **Model 1: Deep Multi-Layer Perceptron (MLP)** — Feedforward neural network with Batch Normalization, GELU activations, and Dropout.
2. **Model 2: Deep 1D Convolutional Neural Network (1D-CNN)** — Deep learning convolutional architecture extracting localized multi-attribute cross-feature interactions.

Both models were trained using positive class weighting ($w_{\text{pos}} \approx 2.77$) to account for real-world class imbalance (~26.5% churn vs ~73.5% non-churn). Performance was evaluated on a held-out test partition across all assignment-mandated metrics: **Precision, Recall, F1-Score, and AUC-ROC curve**.

---

## 2. Dataset Selection, Access & Ground Truth Validation

### 2.1 Dataset Information & Links
- **Dataset Name:** Telco Customer Churn Dataset
- **Public Repositories:** 
  - [Kaggle Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`blastchar/telco-customer-churn`)
  - [Hugging Face Hub](https://huggingface.co/datasets)
  - Direct import URL: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Total Records:** 7,043 customer accounts across 21 demographic, service, and billing attributes.
- **Target Variable (Ground Truth):** `Churn` (Binary ground truth: `Yes` = 1, `No` = 0; indicating whether the customer terminated their service within the last month).
- **Class Balance:**
  - `No Churn (0)`: 5,174 customers (73.5%)
  - `Churn (1)`: 1,869 customers (26.5%)
  - Imbalance Ratio: $\approx 2.77 : 1$.

### 2.2 Data Preprocessing Pipeline
1. **Data Cleaning:** 11 blank spaces in `TotalCharges` were converted to numerical floats and imputed using the median ($1,397.48).
2. **Identifier Removal:** The non-predictive high-cardinality feature `customerID` was removed.
3. **One-Hot Encoding:** Categorical features (gender, partner, dependents, phone service, internet tiers, security add-ons, contract duration, payment method) were dummy-encoded with `drop_first=True`, yielding **30 encoded input features**.
4. **Stratified Splitting:** Partitioned into 70% Train (4,929), 15% Validation (1,057), and 15% Test (1,057) stratified by the ground-truth target.
5. **Zero-Leakage Standardization:** `StandardScaler` was fitted strictly on the training partition and applied to validation and test partitions.

---

## 3. Model Architecture Specifications

### 3.1 Model 1: Deep Multi-Layer Perceptron (MLP)
Model 1 uses a deep feedforward topology designed to capture dense nonlinear tabular interactions:
- **Input:** 30 normalized features
- **Block 1:** Linear(30 $\rightarrow$ 128) $\rightarrow$ BatchNorm1d(128) $\rightarrow$ GELU $\rightarrow$ Dropout(0.3)
- **Block 2:** Linear(128 $\rightarrow$ 64) $\rightarrow$ BatchNorm1d(64) $\rightarrow$ GELU $\rightarrow$ Dropout(0.3)
- **Block 3:** Linear(64 $\rightarrow$ 32) $\rightarrow$ BatchNorm1d(32) $\rightarrow$ GELU $\rightarrow$ Dropout(0.2)
- **Output:** Linear(32 $\rightarrow$ 1) logit output $\rightarrow$ `BCEWithLogitsLoss(pos_weight=2.7683)`
- **Total Trainable Parameters:** 15,361

### 3.2 Model 2: Deep 1D Convolutional Neural Network (1D-CNN)
Model 2 treats the feature vector as an ordered sequence of channel observations, leveraging 1D convolutional receptive fields:
- **Input Reshape:** $(N, 30) \rightarrow (N, 1, 30)$
- **Conv Block 1:** Conv1d(1 $\rightarrow$ 32, $k=3$, $p=1$) $\rightarrow$ BatchNorm1d(32) $\rightarrow$ GELU $\rightarrow$ MaxPool1d(2) $\rightarrow$ Dropout(0.2)
- **Conv Block 2:** Conv1d(32 $\rightarrow$ 64, $k=3$, $p=1$) $\rightarrow$ BatchNorm1d(64) $\rightarrow$ GELU $\rightarrow$ AdaptiveAvgPool1d(4) $\rightarrow$ Dropout(0.3)
- **Flatten:** $64 \times 4 = 256$ features
- **Dense Head:** Linear(256 $\rightarrow$ 64) $\rightarrow$ BatchNorm1d(64) $\rightarrow$ GELU $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(64 $\rightarrow$ 1)
- **Total Trainable Parameters:** 23,457

---

## 4. Critical Code Segments

### 4.1 Model 1: Deep MLP Definition
```python
class ChurnMLP(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

### 4.2 Model 2: Deep 1D-CNN Definition
```python
class Churn1DCNN(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(kernel_size=2), nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(4), nn.Dropout(0.3)
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.conv(x.unsqueeze(1))
        return self.head(feat.view(feat.size(0), -1))
```

---

## 5. Experimental Results & Performance Benchmark

Evaluating both models on the held-out test partition (1,057 customer records):

```
========================================================================================================
WEEK 6 ASSIGNMENT: HEAD-TO-HEAD BENCHMARK (TEST PARTITION)
========================================================================================================
Model Architecture                 Accuracy   Precision   Recall     F1-Score   F1-Macro   ROC-AUC
--------------------------------------------------------------------------------------------------------
Model 1: Deep MLP (Feedforward)    72.56%     48.93%      81.43%     0.6113     0.6996     0.8406
Model 2: Deep 1D-CNN (ConvNet)     72.19%     48.58%      85.71%     0.6202     0.7004     0.8449
========================================================================================================
```

### 5.1 Detailed Classification Metrics
- **Model 1: Deep MLP**
  - True Negatives ($\text{TN}$): **539**
  - False Positives ($\text{FP}$): **238**
  - False Negatives ($\text{FN}$): **52**
  - True Positives ($\text{TP}$): **228**
  - Churn Recall: **81.43%** ($228 / 280$)
  - Precision: **48.93%** ($228 / 466$)
  - ROC-AUC: **0.8406**

- **Model 2: Deep 1D-CNN**
  - True Negatives ($\text{TN}$): **523**
  - False Positives ($\text{FP}$): **254**
  - False Negatives ($\text{FN}$): **40**
  - True Positives ($\text{TP}$): **240**
  - Churn Recall: **85.71%** ($240 / 280$)
  - Precision: **48.58%** ($240 / 494$)
  - ROC-AUC: **0.8449**

---

## 6. Visual Snapshots of Results

All figures were generated and saved to [`week6assignment/figures/`](figures/):
1. **Figure 1:** [`01_dataset_eda_distributions.png`](figures/01_dataset_eda_distributions.png) — Exploratory profile (class balance, tenure KDE, and contract churn).
2. **Figure 2:** [`02_model_training_curves.png`](figures/02_model_training_curves.png) — Training and validation loss/accuracy dynamics across 50 epochs.
3. **Figure 3:** [`03_comparative_roc_pr_curves.png`](figures/03_comparative_roc_pr_curves.png) — Comparative ROC curves ($\text{AUC} > 0.84$) and Precision-Recall curves.
4. **Figure 4:** [`04_comparative_confusion_matrices.png`](figures/04_comparative_confusion_matrices.png) — Side-by-side Confusion Matrices.
5. **Figure 5:** [`05_comparative_metrics_benchmark.png`](figures/05_comparative_metrics_benchmark.png) — Benchmark bar chart across all assignment metrics.

---

## 7. Comparative Discussion & Key Findings

1. **High Churn Sensitivity (Recall $> 81\%$):**
   In churn prevention, missing an exiting customer ($\text{FN}$) results in lost recurring revenue. Both models capture over $81\%$ of churners, with the 1D-CNN detecting $85.71\%$ ($240 / 280$).
2. **ROC-AUC Discrimination ($\text{AUC} > 0.84$):**
   Both models produce well-calibrated output probabilities that clearly discriminate between churners and non-churners.
3. **Inductive Bias Comparison:**
   - The **1D-CNN** achieved higher recall ($85.71\%$) and higher ROC-AUC ($0.8449$), demonstrating that localized convolution kernels are effective at capturing interaction patterns between grouped account features.
   - The **Deep MLP** achieved slightly higher overall accuracy ($72.56\%$) and precision ($48.93\%$) through global dense linear layers.

---

## 8. GitHub Submission Link

- **Official Repository URL:** [https://github.com/subhashthakurcode/Tech-405-Project](https://github.com/subhashthakurcode/Tech-405-Project)
- **Directory:** `week6assignment/`
- **Jupyter Notebook:** [`week6assignment/assignment6_classification.ipynb`](assignment6_classification.ipynb)
- **Word Document Report:** [`week6assignment/Assignment_6_Classification_Report.docx`](Assignment_6_Classification_Report.docx)
