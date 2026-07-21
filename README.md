# 🎗️ Cancer ML Prediction — Multi-Target Clinical Prognosis & Cost Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Rudra%20Dahikar-orange)](https://www.kaggle.com/rudradahikar)

An end-to-end Machine Learning project designed to analyze clinical, environmental, genetic, and demographic factors to predict three critical healthcare outcomes for cancer patients:
1. 💵 **Treatment Cost (USD)** — Financial estimation and healthcare cost forecasting.
2. 🏥 **Target Severity Score** — Clinical risk and patient severity indexing.
3. ⏳ **Survival Years** — Patient prognosis and life expectancy modeling.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Machine Learning Models](#-machine-learning-models)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Workflow & How to Run](#-workflow--how-to-run)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Metrics](#3-evaluation--metrics)
  - [4. Interactive Live Inference](#4-interactive-live-inference)
  - [5. Kaggle / Jupyter Notebooks](#5-kaggle--jupyter-notebooks)
- [Evaluation Metrics](#-evaluation-metrics)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Project Overview

Cancer prognosis and healthcare financial planning depend on an intricate interplay of individual genetics, behavioral risk factors, ambient environment, and tumor staging. This repository implements multiple classical and ensemble regression architectures across three dedicated prediction tasks:

* **Target A: Treatment Cost (`Treatment_Cost_USD`)**: Predicts total financial burden based on cancer type, stage, demographics, and clinical parameters.
* **Target B: Patient Severity Score (`Target_Severity_Score`)**: Predicts clinical severity index to help triage high-risk patients.
* **Target C: Survival Years (`Survival_Years`)**: Models estimated post-diagnosis survival duration.

---

## 📊 Dataset Architecture

The project operates on the **Global Cancer Patients Dataset** (`cancer.csv`), containing **50,000 patient records** across 15 attributes:

| Column | Type | Role | Description |
|:---|:---:|:---:|:---|
| `Patient_ID` | String | Identifier | Unique patient hash (dropped during preprocessing) |
| `Age` | Integer | Feature | Patient age at diagnosis |
| `Gender` | String | Feature | Biological sex (`Male` / `Female`) |
| `Country_Region` | String | Feature | Geographic origin (USA, UK, Canada, Germany, etc.) |
| `Year` | Integer | Feature | Diagnosis timestamp year |
| `Genetic_Risk` | Float | Feature | Genetic predisposition score (0.0 – 10.0) |
| `Air_Pollution` | Float | Feature | Environmental pollution exposure index (0.0 – 10.0) |
| `Alcohol_Use` | Float | Feature | Alcohol consumption score (0.0 – 10.0) |
| `Smoking` | Float | Feature | Tobacco use index (0.0 – 10.0) |
| `Obesity_Level` | Float | Feature | Obesity score index (0.0 – 10.0) |
| `Cancer_Type` | String | Feature | Primary tumor classification (Lung, Breast, Colon, Prostate, etc.) |
| `Cancer_Stage` | String | Feature | Tumor stage (`Stage 0`, `Stage I`, `Stage II`, `Stage III`, `Stage IV`) |
| `Treatment_Cost_USD` | Float | 🎯 **Target A** | Total medical and therapeutic expenditure ($) |
| `Target_Severity_Score` | Float | 🎯 **Target B** | Composite clinical severity score |
| `Survival_Years` | Float | 🎯 **Target C** | Patient survival duration in years |

---

## 🤖 Machine Learning Models

Each prediction target is evaluated against multiple regression algorithms to discover optimal predictive power:

| Algorithm | Category | Cost Prediction | Severity Prediction | Survival Years |
|:---|:---:|:---:|:---:|:---:|
| **Linear Regression** | Parametric | ✅ | ✅ | ✅ |
| **Decision Tree Regressor** | Tree-based | ✅ | — | ✅ |
| **Random Forest Regressor** | Ensemble Bagging | ✅ | — | ✅ |
| **Gradient Boosting Regressor** | Ensemble Boosting | ✅ | — | ✅ |
| **K-Nearest Neighbors (KNN)** | Instance-based | ✅ | ✅ | ✅ |
| **Support Vector Regressor (SVR)** | Kernel-based | ✅ | ✅ | ✅ |

---

## 📁 Directory Structure

```text
cancer_ml_prediction/
├── cancer.csv                              # Primary dataset (50,000 records)
├── requirements.txt                        # Environment dependencies
├── README.md                               # Project documentation
│
├── cost/                                   # 💰 Treatment Cost Prediction Pipeline
│   ├── preprocessed.py                     # Encoding, cleaning & 80/20 train/test split
│   ├── preprocessed.csv                    # Full processed dataset
│   ├── train.csv / test.csv                # Split datasets
│   ├── cancer_cost_prediction.ipynb        # Single-target Jupyter exploration
│   ├── cancer_ml_prediction_master.ipynb   # Master notebook across all targets
│   ├── DecisionTree/                       # Decision Tree train & test scripts + model
│   ├── RandomForest/                       # Random Forest train & test scripts + model
│   ├── GradientBoosting/                   # Gradient Boosting train & test scripts + model
│   ├── knn/                                # KNN Regressor train & test scripts + model
│   ├── linear/                             # Linear Regression train & test scripts + model
│   └── svr/                                # SVR train & test scripts + model
│
├── severity/                               # 🏥 Target Severity Score Pipeline
│   ├── preprocessed.py                     # Feature encoding & dataset preparation
│   ├── Gender_encoded.enc                  # Serialized LabelEncoder for Gender
│   ├── Country_Region_encoded.enc          # Serialized LabelEncoder for Country
│   ├── Cancer_Type_encoded.enc             # Serialized LabelEncoder for Cancer Type
│   ├── Cancer_Stage_encoded.enc            # Serialized LabelEncoder for Cancer Stage
│   ├── linear/                             # Linear model + interactive live prediction tool
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── train_model.dat
│   │   └── prediction.py                   # ⚡ Real-time terminal prediction CLI
│   ├── knn/                                # KNN train & test scripts + model
│   └── SVR/                                # SVR train & test scripts + model
│
├── years/                                  # ⏳ Survival Years Prediction Pipeline
│   ├── preprocessed.py                     # Preprocessing & train/test generation
│   ├── decisiontree/                       # Decision Tree train & test scripts + model
│   ├── randomforest/                       # Random Forest train & test scripts + model
│   ├── gradientboosting/                   # Gradient Boosting train & test scripts + model
│   ├── knn/                                # KNN train & test scripts + model
│   ├── linear/                             # Linear Regression train & test scripts + model
│   └── svr/                                # SVR train & test scripts + model
│
└── test/                                   # 🧪 Modular Encoder & Serializer Verification
    ├── city_encoded.enc
    └── program1.py ... program9.py         # Incremental verification and encoding tests
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Clone the Repository
```bash
git clone git@github.com:Rudramsd7/Cancer-Preditction-Project.git
cd Cancer-Preditction-Project
```

### Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Workflow & How to Run

### 1. Data Preprocessing
Each target pipeline contains its own dedicated preprocessing logic to encode categorical fields (`Gender`, `Country_Region`, `Cancer_Type`, `Cancer_Stage`), remove identifiers (`Patient_ID`), handle missing values, and produce stratified `train.csv` (80%) and `test.csv` (20%) splits:

```bash
# Preprocess for Treatment Cost
python cost/preprocessed.py

# Preprocess for Severity Score
python severity/preprocessed.py

# Preprocess for Survival Years
python years/preprocessed.py
```

### 2. Model Training
Train any of the regression algorithms. Trained weights and serialized artifacts are automatically exported to `train_model.dat`:

```bash
# Train Random Forest on Treatment Cost
python cost/RandomForest/train.py

# Train Linear Regression on Patient Severity
python severity/linear/train.py

# Train Gradient Boosting on Survival Years
python years/gradientboosting/train.py
```

### 3. Evaluation & Metrics
Run the test scripts to evaluate against the unseen 20% holdout test partition:

```bash
# Evaluate Random Forest on Cost
python cost/RandomForest/test.py

# Evaluate Linear Regression on Severity
python severity/linear/test.py

# Evaluate Decision Tree on Survival Years
python years/decisiontree/test.py
```

### 4. Interactive Live Inference
Test live clinical inputs with the interactive predictor in `severity/linear/prediction.py`:

```bash
python severity/linear/prediction.py
```
This utility loads the saved label encoders (`*.enc`) and trained model checkpoint to provide instantaneous severity scoring from user prompts.

### 5. Kaggle / Jupyter Notebooks
For exploratory data analysis, visual correlation heatmaps, and side-by-side benchmark plots:
- `cost/cancer_ml_prediction_master.ipynb`: Complete unified notebook covering all three targets.
- `cost/cancer_cost_prediction.ipynb`: Deep dive into treatment cost distributions and model diagnostics.

---

## 📈 Evaluation Metrics

All models are benchmarked using industry standard regression metrics:
- **$R^2$ Score (Coefficient of Determination)**: Evaluates the proportion of variance explained by the model ($-\infty$ to $1.0$).
- **Mean Squared Error (MSE)**: Penalizes larger prediction errors.
- **Mean Absolute Error (MAE)**: Measures average magnitude of residual errors.

---

## 👨‍💻 Author & Acknowledgments

- **Author**: Rudra Atul Dahikar
- **GitHub**: [@Rudramsd7](https://github.com/Rudramsd7)
- **Kaggle**: [rudradahikar](https://www.kaggle.com/rudradahikar)
- **Repository**: [Cancer-Preditction-Project](https://github.com/Rudramsd7/Cancer-Preditction-Project)
