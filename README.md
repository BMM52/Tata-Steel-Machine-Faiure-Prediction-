# 🏭 Tata Steel – Machine Failure Prediction using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Predictive%20Maintenance-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange"/>
  <img src="https://img.shields.io/badge/XGBoost-Boosting-red"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-purple"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-yellow"/>
</p>

---

## 📌 Project Overview

Unexpected machine failures in large-scale steel manufacturing can cause:

- ⏱️ Unplanned production downtime  
- 💰 High repair and maintenance costs  
- ⚠️ Safety risks for operators  

This project builds a **machine learning–based predictive maintenance system** for **Tata Steel**, capable of predicting machine failures **before they occur**, enabling **proactive maintenance decisions**.

The dataset is synthetically generated but closely reflects **real-world industrial machine behavior**.

---

## 🎯 Business Objective

- Predict **machine failure (binary classification)**
- Minimize **false negatives (missed failures)**
- Enable **early intervention**
- Improve **Overall Equipment Effectiveness (OEE)**
- Reduce **operational cost and safety risk**

---

## 📂 Dataset Description

### 📊 Data Files
- **Training Dataset (`train.csv`)**
  - 136,429 records
  - Includes target variable
- **Test Dataset (`test.csv`)**
  - 90,954 records
  - Used for final predictions

### 🧾 Key Features
- 🌡️ Air temperature  
- 🔥 Process temperature  
- ⚙️ Rotational speed  
- 🧲 Torque  
- 🛠️ Tool wear  
- 🏷️ Machine type (categorical)  
- 🚨 Failure indicators (TWF, HDF, PWF, OSF, RNF)

### 🎯 Target Variable
- **Machine failure**
  - `1` → Failure occurred  
  - `0` → No failure  

> ⚠️ **Highly Imbalanced Dataset**  
> Only ~**1.57%** of observations correspond to failures.

---

## 🔍 Exploratory Data Analysis (EDA)

Key EDA steps performed:

- ✔ Missing value and duplicate checks  
- 📈 Feature distribution analysis  
- 🔁 Failure class imbalance visualization  
- 🔥 Correlation heatmaps  
- ⚠️ Failure-type relationship analysis  
- 📉 Outlier detection using **Z-score** and **IQR methods**

---

## 🛠️ Feature Engineering

Domain-driven features were engineered to better capture machine stress patterns:

| Feature | Description |
|-------|------------|
| `temp_diff` | Process temperature − Air temperature |
| `torque_per_rpm` | Torque / Rotational speed |
| `temp_ratio` | Process temperature / Air temperature |
| `temp_interaction` | Process × Air temperature |
| `high_wear_flag` | 1 if Tool wear > 150 |

These features significantly improved predictive performance.

---

## ⚙️ Preprocessing Pipeline

- 📏 **StandardScaler** for numerical features  
- 🏷️ **OneHotEncoder** for machine type  
- 🔀 **Train–Validation Split**
  - 80% Train / 20% Validation  
  - Stratified to preserve class imbalance  

---

## 🤖 Model Building & Comparison

### 🔎 Models Evaluated
- Logistic Regression  
- Random Forest  
- XGBoost (baseline)

### 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- **ROC-AUC (primary metric)**

| Model | ROC-AUC | Recall | Precision |
|------|--------|--------|-----------|
| Logistic Regression | 0.87 | High | Very Low |
| Random Forest | 0.91 | Medium | Medium |
| ⭐ **XGBoost (Baseline)** | **0.915** | **High** | **Best Balance** |

✅ **XGBoost was selected for advanced tuning**.

---

## 🚀 Hyperparameter Tuning

- Performed using **RandomizedSearchCV**
- Tuned parameters include:
  - `max_depth`
  - `learning_rate`
  - `n_estimators`
  - `subsample`
  - `colsample_bytree`
  - `scale_pos_weight`

🎯 **Best ROC-AUC achieved:** **~0.9366**

---

## 🎚️ Threshold Optimization

The default probability threshold (0.5) is unsuitable for highly imbalanced data.

A threshold sweep from **0.10 → 0.90** was conducted.

### ✅ Final Threshold: **0.40**

- **Recall:** ~81%
- **Precision:** ~12%
- **ROC-AUC:** ~0.92

> In industrial environments, **missing a failure is far more costly** than investigating a false alarm — hence recall is prioritized.

---

## 🧠 Model Explainability (SHAP)

SHAP was used to interpret model predictions.

### 🔑 Most Influential Features:
- Torque per RPM  
- Temperature Difference  
- Tool Wear  
- Process Temperature  
- Machine Type  

This ensures **model transparency and trust**, critical for industrial deployment.

---

## 📈 Final Model & Predictions

- Final XGBoost model trained on **100% of training data**
- Predictions generated for unseen test data
- Threshold = 0.40 applied
- Final output contains:
  - `id`
  - `Machine failure`

---

## 💼 Business Impact

Implementing this predictive maintenance system enables:

- 🔻 Reduced unplanned downtime  
- 💰 Lower maintenance and repair costs  
- ⚙️ Optimized maintenance scheduling  
- 📈 Improved Overall Equipment Effectiveness (OEE)  
- 🦺 Enhanced operator safety  

💡 Preventing even **one major failure per month** can save **lakhs of rupees** in operational losses.

---

## 🔮 Future Enhancements

- Real-time sensor data streaming  
- Vibration and acoustic data integration  
- REST API deployment (FastAPI / Flask)  
- Live machine health monitoring dashboard  
- Continuous model retraining

---

## 🧑‍💻 Tech Stack

- 🐍 Python  
- 📊 Pandas, NumPy  
- 🤖 Scikit-Learn  
- 🚀 XGBoost  
- 🧠 SHAP  
- 📈 Matplotlib, Seaborn  
- ☁️ Google Colab  

---

## 👤 Author

**Burhanuddin Motiwala**  
📊 Aspiring Data Scientist | Machine Learning Enthusiast  

🔗 **GitHub:** https://github.com/burhanuddinmo  
🔗 **LinkedIn:** https://www.linkedin.com/in/burhanuddinmotiwala  



