# Bankruptcy Prediction on Imbalanced Financial Dataset

This project addresses the challenge of predicting bankruptcies using a highly imbalanced financial dataset, where only **3% of the companies are bankrupt** and the remaining **97% are not**. The model aims to **minimize false negatives**, prioritizing **early identification of at-risk firms** to support financial risk mitigation.

---

## 📊 Dataset

- **Type**: Binary classification
- **Imbalance Ratio**: ~3% bankrupt vs 97% non-bankrupt
- **Features**: Financial ratios and firm metrics (anonymized)
- **Target**: Bankruptcy (1) or Not (0)

---

## 🧠 Methodology

### 🔁 Handling Imbalanced Data
- **Oversampling**:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Random Oversampling
- **Undersampling**:
  - Cluster Centroid Sampling
  - Random Undersampling
- **Hybrid Sampling**:
  - Combined techniques to maintain balance and distribution

### 🎯 Threshold Calibration
- Manually tuned classification threshold for:
  - Minimizing **false negatives**
  - Maximizing **recall** and **G-Mean**

---

## 🧪 Models Implemented

- **Decision Tree**
- **Random Forest**

> All models were implemented from scratch using **Python** and **NumPy**, without using high-level libraries like scikit-learn.

---

## 📈 Results

| Metric       | Score   |
|--------------|---------|
| Recall       | 69.7%   |
| AUC (ROC)    | 0.94    |
| G-Mean       | 0.82    |

- Performance was evaluated using **stratified cross-validation**.
- The focus was on **maximizing recall** to avoid missing bankrupt firms.

---

## ⚙️ Tech Stack

- Python (native)
- NumPy
- Matplotlib (optional, for visualizations)
- No use of external ML frameworks (e.g., scikit-learn, TensorFlow)

---

## 🧾 Key Learnings

- Importance of sampling strategy in imbalanced datasets
- Impact of threshold tuning on recall and precision tradeoff
- Implementing interpretable models in a low-level environment (NumPy only)

---
