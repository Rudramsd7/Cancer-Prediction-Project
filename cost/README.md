# 🎗️ Cancer Patient ML Prediction

> Predicting **Treatment Cost**, **Severity Score**, and **Survival Years** for cancer patients using classical ML regression models.

**Author:** Rudra Dahikar  
**Kaggle:** https://www.kaggle.com/rudradahikar  

---

## 📊 Dataset

**Global Cancer Patients Dataset** — `cancer.csv`

| Column | Type | Description |
|---|---|---|
| Patient_ID | str | Unique identifier (dropped) |
| Age | int | Patient age |
| Gender | str | Male / Female |
| Country_Region | str | Country of patient |
| Year | int | Year of diagnosis |
| Genetic_Risk | float | Genetic risk score |
| Air_Pollution | float | Air pollution exposure |
| Alcohol_Use | float | Alcohol use score |
| Smoking | float | Smoking score |
| Obesity_Level | float | Obesity level |
| Cancer_Type | str | Type of cancer |
| Cancer_Stage | str | Stage (0–IV) |
| Treatment_Cost_USD | float | ✅ **Target A** |
| Survival_Years | float | ✅ **Target C** |
| Target_Severity_Score | float | ✅ **Target B** |

---

## 🧠 Project Structure

```
cancer_ml_prediction/
├── cancer.csv                          ← Raw dataset
├── cancer_ml_prediction_master.ipynb   ← ⭐ Master Kaggle notebook (all 3 targets)
│
├── cost/                               ← Treatment Cost prediction
│   ├── preprocessed.py
│   ├── cancer_cost_prediction.ipynb
│   ├── DecisionTree/   train.py  test.py
│   ├── RandomForest/   train.py  test.py
│   ├── GradientBoosting/ train.py test.py
│   ├── knn/            train.py  test.py
│   ├── linear/         train.py  test.py
│   └── svr/            train.py  test.py
│
├── severity/                           ← Severity Score prediction
│   ├── preprocessed.py
│   ├── *_encoder.enc
│   ├── linear/   train.py  test.py  prediction.py
│   ├── knn/      train.py  test.py
│   └── SVR/      train.py  test.py
│
├── years/                              ← Survival Years prediction
│   ├── preprocessed.py
│   ├── decisiontree/   train.py  test.py
│   ├── gradientboosting/ train.py test.py
│   ├── knn/            train.py  test.py
│   ├── linear/         train.py  test.py
│   ├── randomforest/   train.py  test.py
│   └── svr/            train.py  test.py
│
└── test/                               ← Encoding utility scripts
    ├── program1.py … program9.py
```

---

## 🤖 Models Used

| Algorithm | Cost | Severity | Survival Years |
|---|:---:|:---:|:---:|
| Linear Regression | ✅ | ✅ | ✅ |
| Decision Tree | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ✅ |
| K-Nearest Neighbors | ✅ | ✅ | ✅ |
| Support Vector Regressor | ✅ | ✅ | ✅ |
| Gradient Boosting | ✅ | ✅ | ✅ |

---

## 📈 Metrics

Every model is evaluated with:
- **R² Score** (higher = better)
- **MSE** — Mean Squared Error (lower = better)
- **MAE** — Mean Absolute Error (lower = better)

---

## 🚀 Run on Kaggle

1. Upload `cancer.csv` as a Kaggle Dataset → Title: **cancer-prediction-dataset**
2. Open `cancer_ml_prediction_master.ipynb` as a new Kaggle Notebook
3. Click **Add Data** → attach `cancer-prediction-dataset`
4. Click **Run All**

---

## 🖥️ Run Locally

```bash
pip install pandas scikit-learn joblib matplotlib

# Step 1 — preprocess
python cost/preprocessed.py

# Step 2 — train (example: Random Forest for cost)
python cost/RandomForest/train.py

# Step 3 — evaluate
python cost/RandomForest/test.py

# Step 4 — live prediction (severity pipeline)
python severity/linear/prediction.py
```

---

## 🔗 Links

- Kaggle: https://www.kaggle.com/rudradahikar
