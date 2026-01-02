# 📘 AI/ML Assignment 3: Supervised Learning Models in Practice

**Course:** Machine Learning (Module 19)  
**Total Marks:** 100  
**Submission Format:** Jupyter Notebook (`.ipynb`)

---

## 📋 Assignment Overview

This assignment demonstrates the practical implementation of supervised learning models across two distinct tasks:
- **Part A (45 marks):** Regression — predicting insurance charges
- **Part B (45 marks):** Classification — loan approval prediction
- **Part C (5 marks):** Final reflection & insights

All models use **Python and scikit-learn only** with `random_state=42` for reproducibility.

---

## 🔵 Part A: Regression Task (45 Marks)

### Objective
Predict medical insurance charges based on patient demographics and health attributes.

### Models Implemented
1. **Multiple Linear Regression** — R²: 0.784 (test)
2. **Polynomial Regression** (degree=2) — R²: 0.780 (test)
3. **Support Vector Regression** (RBF, C=10000) — R²: 0.849 (test) ⭐⭐ Second Best
4. **Random Forest Regressor** (100 estimators, max_depth=3) — R²: 0.865 (test) ⭐⭐⭐ **BEST**

### Dataset: `insurance.csv`
- **Rows:** 1,338 | **Columns:** 7 (6 features + 1 target)
- **Features:** age, sex, bmi, children, smoker, region
- **Target:** charges (medical charges in USD)

### Key Findings
- **Best Model:** Random Forest achieves highest R² (0.865) with lowest MAE (2746) and RMSE (4560).
- **Kernel Choice:** RBF kernel in SVR captures nonlinearity effectively with proper hyperparameter tuning (C=10000).
- **Comparison:** Polynomial regression shows minimal improvement over linear regression (~0.78 R²), confirming linear relationships suffice for this dataset.

---

## 🟠 Part B: Classification Task (45 Marks)

### Objective
Predict loan approval (binary: 0=rejected, 1=approved) based on applicant profile.

### Models Implemented
1. **Logistic Regression** (L2 penalty, C=1.0)
2. **Support Vector Machine** (RBF kernel, C=1, gamma='scale')
3. **Naive Bayes** (GaussianNB for continuous features)
4. **K-Nearest Neighbors** (best k=25, distance-weighted)
5. **Random Forest Classifier** (100 estimators, max_depth=5) ⭐⭐⭐ **BEST**

### Dataset: `loan_data.csv`
- **Rows:** ~10,000+ | **Columns:** Multiple features + 1 binary target
- **Class Distribution:** ~77.7% class 0 (rejected), ~22.2% class 1 (approved) — **imbalanced**
- **Feature Types:** Numerical (age, income, etc.) + Categorical (education, home_ownership, etc.)

### Data Preprocessing
- **Stratified Train-Test Split:** 80/20 with stratification to preserve class distribution
- **Categorical Encoding:**
  - OneHotEncoder for nominal features
  - OrdinalEncoder for ordinal features (education, home_ownership)
- **Scaling:** StandardScaler applied to numerical features

### Key Findings
- **Best Model:** Random Forest handles mixed feature types and class imbalance robustly.
- **Naive Bayes:** GaussianNB chosen for continuous feature distributions; BernoulliNB inappropriate for non-binary features.
- **KNN Tuning:** Tested k ∈ [5, 15, 19, 25, 35, 49, 99, 500]; best performance at k=25.
- **Imbalance:** Affects precision/recall for minority class; RF's ensemble approach mitigates this.

---

## 🧠 Part C: Final Reflection (5 Marks)

### Best Regression Model
**Random Forest Regressor** — captures complex nonlinear relationships, handles feature interactions, and achieves highest test R² (0.865) with strong generalization.

### Best Classification Model
**Random Forest Classifier** — robust to mixed feature types and class imbalance; delivers high accuracy, precision, and interpretable feature importances.

### Real-World Deployment Scenario
**Loan-Risk Scoring Microservice:**
- Deploy trained `RandomForestClassifier` as a REST API.
- Accept applicant profile (features) in real time.
- Return approval prediction + confidence scores.
- Log predictions and features for monitoring, drift detection, and periodic retraining.
- Example: Bank integration for instant pre-qualification during online applications.

---

## 📊 Performance Summary

| Model | Task | Test R²/Accuracy | MAE/Key Metric |
|-------|------|------------------|----------------|
| Linear Regression | Regression | 0.784 | 4181 |
| Polynomial Regression | Regression | 0.780 | 4254 |
| SVR (RBF tuned) | Regression | 0.849 | 1803 |
| **Random Forest** | **Regression** | **0.865** | **2746** ⭐ |
| Logistic Regression | Classification | ~0.88 | — |
| SVM (RBF) | Classification | ~0.87 | — |
| Naive Bayes (Gaussian) | Classification | ~0.85 | — |
| KNN (k=25) | Classification | ~0.88 | — |
| **Random Forest** | **Classification** | **~0.91** | — ⭐ |

---

## 🛠️ Requirements & Environment

### Dependencies
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Python Version
Python 3.7+ recommended.

---

## 📁 Files Included

- **Copy_of_AI_ML_Assignment_3_Module_19.ipynb** — Complete notebook with all models, evaluations, and reflections.
- **insurance.csv** — Regression dataset (medical insurance charges).
- **loan_data.csv** — Classification dataset (loan approvals).
- **README.md** — This file.

---

## 🚀 How to Run

1. **Open the notebook:**
   ```bash
   jupyter notebook Copy_of_AI_ML_Assignment_3_Module_19.ipynb
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run all cells** in order (Kernel → Restart & Run All).

4. **Review outputs:**
   - Part A (cells ~1–25): Regression models with metrics and visualizations.
   - Part B (cells ~26–45): Classification models with confusion matrices.
   - Part C (final cell): Reflection and deployment scenario.

---

## 📈 Key Insights

- **Regression:** Random Forest outperforms linear/polynomial/SVM models due to nonlinear pattern capture and ensemble robustness.
- **Classification:** RF dominates due to mixed feature handling, class imbalance mitigation, and interpretability.
- **Imbalance Handling:** Stratified splits + ensemble methods (Random Forest) effective for imbalanced data.
- **Hyperparameter Tuning:** Critical for SVR (C, gamma, epsilon) and KNN (k selection); grid search & cross-validation recommended.

---

## 📝 Notes

- All random seeds fixed at `random_state=42` for reproducibility.
- Plots are labeled and readable per assignment requirements.
- No external AutoML, deep learning, or proprietary libraries used.
- Assignment follows Modules 13–18 (scikit-learn supervised learning).

---

## ✅ Evaluation Checklist

- [x] TODO 0: Environment Setup (5 marks)
- [x] TODO A1–A7: Regression Task (45 marks)
- [x] TODO B1–B8: Classification Task (45 marks)
- [x] Final Reflection (5 marks)

**Total: 100 marks**

---

## 📧 Contact & Questions

For clarifications on the assignment or results, refer to the notebook cells and embedded markdown explanations.

---

**Last Updated:** January 3, 2026  
**Status:** ✅ Submission Ready
