# 📘 NYC Taxi Trip Duration & Ride Acceptance Prediction

### *Machine Learning Regression + Classification Project*

This repository contains two complete machine learning pipelines built on the **NYC Taxi Trip Duration** dataset:

1. **Regression Task:** Predicting taxi trip duration  
2. **Classification Task:** Predicting ride acceptance using rule-based labels  

Both implementations are written in **Python 3.10** and provided as Jupyter notebooks.

---

## 🚀 Project Structure

```
.
├── regression.ipynb       # Linear, Polynomial, Ridge, Lasso (from scratch) + RFR, SVR, XGBoost
├── classification.ipynb   # Logistic Regression (from scratch) + Trees, RF, GB, SVM
├── train.csv              # User must manually add this file (not included due to large size)
├── test.csv               # User must manually add this file
└── README.md
```

> **Note:**  
> The dataset files **train.csv** and **test.csv** must be manually placed in the root directory of the project because of their large size.

---

## 🐍 Python Version

This project requires **Python 3.10**.

---

## 📦 Required Libraries

Below is the complete list of libraries used in both notebooks:

### Core Libraries
- numpy  
- pandas  
- matplotlib  
- seaborn  

### Machine Learning (Regression)
- scikit-learn  
  - RandomForestRegressor  
  - StandardScaler  
  - train_test_split  
  - GridSearchCV  
  - RandomizedSearchCV  
- xgboost  
  - XGBRegressor  

### Machine Learning (Classification)
- scikit-learn  
  - DecisionTreeClassifier  
  - RandomForestClassifier  
  - GradientBoostingClassifier  
  - LinearSVC  
  - accuracy_score  
  - precision_score  
  - recall_score  
  - f1_score  

---

## 📥 Installation

Install all required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Add Dataset Files

Place the following files in the root folder of the project:
- `train.csv`
- `test.csv`

These are not included in the repository due to size limits.

### 3. Launch Jupyter Notebook

If Jupyter is not installed:

```bash
pip install notebook
```

Start Jupyter:

```bash
jupyter notebook
```

### 4. Run the Notebooks

#### Regression Task

Open:
```
regression.ipynb
```

This notebook implements:
- Linear Regression (closed-form, from scratch)
- Polynomial Regression (degree as hyperparameter)
- Ridge & Lasso Regression (from scratch)
- Grid Search
- Random Forest Regressor (RFR)
- XGBoostRegressor

#### Classification Task

Open:
```
classification.ipynb
```

This notebook implements:
- Logistic Regression (from scratch)
- Ridge and Lasso regularization
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

Both notebooks can be run cell by cell in order without any additional configuration.

---

## 📊 Dataset Source

The dataset is obtained from the Kaggle competition:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

---

## 🙌 Contributions

Feel free to open issues or create pull requests for improvements or bug fixes!