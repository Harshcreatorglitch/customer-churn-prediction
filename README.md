# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, service usage, contract details, billing information, and payment method.

The project covers the complete data science workflow: **data cleaning → exploratory data analysis → preprocessing → model training → hyperparameter tuning → evaluation → prediction on new customer data**.

---

## Project Overview

Customer churn is a major business problem for subscription-based companies. Identifying customers who are likely to leave allows businesses to take proactive retention measures.

In this project, machine learning classification models are used to predict customer churn and identify customers at higher risk of leaving.

### Objective

Build a machine learning model that can:

* Predict whether a customer is likely to churn.
* Estimate the probability of churn.
* Identify patterns associated with customer churn.
* Compare multiple classification algorithms.
* Prioritize detection of potential churners using class balancing and model tuning.

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset**, containing information about telecom customers and whether they discontinued their service.

### Dataset dimensions

* **Rows:** 7,043
* **Features:** 19 after removing `customerID`
* **Target:** `Churn`
* **Churned customers:** 1,869
* **Non-churned customers:** 5,174
* **Overall churn rate:** approximately 26.5%

### Important features

| Feature           | Description                              |
| ----------------- | ---------------------------------------- |
| `gender`          | Customer gender                          |
| `SeniorCitizen`   | Whether the customer is a senior citizen |
| `Partner`         | Whether the customer has a partner       |
| `Dependents`      | Whether the customer has dependents      |
| `tenure`          | Number of months with the company        |
| `PhoneService`    | Whether the customer has phone service   |
| `InternetService` | Type of internet service                 |
| `OnlineSecurity`  | Online security subscription             |
| `TechSupport`     | Technical support subscription           |
| `Contract`        | Contract type                            |
| `PaymentMethod`   | Customer payment method                  |
| `MonthlyCharges`  | Monthly customer charges                 |
| `TotalCharges`    | Total amount charged                     |
| `Churn`           | Target variable                          |

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   └── predict.py
│
├── models/
│   └── customer_churn_model.joblib
│
├── visualizations/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Exploratory Data Analysis

Several important patterns were identified during exploratory analysis.

## 1. Churn Distribution

The dataset is imbalanced, with significantly more customers who stayed than customers who churned.

* Non-churn: **5,174**
* Churn: **1,869**
* Churn rate: **~26.5%**

This class imbalance was considered during model development.

---

## 2. Contract Type

Contract type showed one of the strongest relationships with churn.

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month | **42.71%** |
| One year       | **11.27%** |
| Two year       |  **2.83%** |

Customers with month-to-month contracts had substantially higher churn rates than customers with longer contracts.

---

## 3. Payment Method

Payment method also showed noticeable differences in churn rates.

| Payment Method            | Churn Rate |
| ------------------------- | ---------: |
| Bank transfer (automatic) | **16.71%** |
| Credit card (automatic)   | **15.24%** |
| Electronic check          | **45.29%** |
| Mailed check              | **19.11%** |

Customers using electronic checks had the highest observed churn rate in the dataset.

---

## 4. Senior Citizen Status

Churn rates differed between senior and non-senior customers.

| Customer Group | Churn Rate |
| -------------- | ---------: |
| Non-senior     | **23.61%** |
| Senior citizen | **41.68%** |

---

## 5. Tenure

Customers with shorter tenure showed higher churn levels, while customers who had stayed with the company for longer periods generally showed lower churn.

This suggests that the early customer lifecycle may be an important period for retention efforts.

---

## 6. Monthly Charges

Customers who churned generally showed relatively higher monthly charges than customers who remained with the company.

Monthly charges were therefore included as an important numerical feature in the machine learning models.

---

## 7. Internet Service

Fiber optic customers represented a large share of churn observations.

However, observed relationships in exploratory analysis should not automatically be interpreted as causal relationships.

---

# Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing values

`TotalCharges` was originally stored as a text field.

It was converted to numeric values:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

The resulting missing values were filled with `0`.

### 2. Duplicate records

No duplicate records were identified.

### 3. Customer ID

`customerID` was removed because it is an identifier rather than a meaningful predictive feature.

### 4. Target encoding

The target variable was converted:

```text
No  → 0
Yes → 1
```

### 5. Categorical features

Categorical variables were transformed using **One-Hot Encoding**.

### 6. Numerical features

Numerical features were standardized using `StandardScaler`.

### 7. Train/Test Split

The dataset was divided using an 80/20 split with stratification:

* Training set: **5,634 records**
* Test set: **1,409 records**

Stratification was used to preserve the churn/non-churn class distribution.

---

# Machine Learning Models

Multiple classification approaches were evaluated.

## Models tested

1. Logistic Regression
2. Logistic Regression with a 0.40 classification threshold
3. Random Forest
4. Balanced Logistic Regression
5. Tuned Balanced Logistic Regression

The balanced models used:

```python
class_weight="balanced"
```

This gives additional importance to the minority churn class.

---

# Model Comparison

Performance on the held-out test set:

| Model                                  |   Accuracy |  Precision |     Recall |         F1 |   ROC-AUC |
| -------------------------------------- | ---------: | ---------: | ---------: | ---------: | --------: |
| Logistic Regression                    | **80.55%** |     65.72% |     55.88% |     60.40% | **0.842** |
| Logistic Regression (0.40)             |     77.71% |     56.82% |     66.84% |     61.43% | **0.842** |
| Random Forest                          |     77.50% |     59.60% |     47.33% |     52.76% |     0.819 |
| Balanced Logistic Regression           |     73.81% |     50.43% |     78.34% |     61.36% |     0.842 |
| **Tuned Balanced Logistic Regression** | **74.17%** | **50.87%** | **78.61%** | **61.76%** | **0.841** |

---

# Final Model

The final candidate model is a **Tuned Balanced Logistic Regression**.

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation.

### Best parameter

```text
C = 0.1
```

### Cross-validation F1

```text
0.633
```

### Test-set performance

```text
Accuracy:   74.17%
Precision:  50.87%
Recall:     78.61%
F1-score:   61.76%
ROC-AUC:    0.841
```

The model was selected because the project's primary objective is to identify customers who are at risk of churning. Therefore, **recall for the churn class** is particularly important.

---

# Confusion Matrix

The final model produced the following confusion matrix on the test set:

```text
                 Predicted
                 No     Yes

Actual No       751    284
Actual Yes       80    294
```

### Interpretation

* **True Negatives:** 751
* **False Positives:** 284
* **False Negatives:** 80
* **True Positives:** 294

The model correctly identified **294 of 374 actual churners**, resulting in approximately **78.6% churn recall**.

---

# Example Prediction

The trained model can be used to predict churn for a new customer.

Example output:

```text
Prediction: Likely to Churn
Churn Probability: 86.16%
```

This demonstrates that the trained pipeline can process a new customer's information and return both a classification and estimated churn probability.

---

# How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

Move into the project:

```bash
cd customer-churn-prediction
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_data_exploration.ipynb
```

The notebook contains the data exploration, preprocessing, model training, tuning, and evaluation workflow.

---

## 5. Run a prediction

The trained model is stored in:

```text
models/customer_churn_model.joblib
```

Run:

```bash
python src/predict.py
```

Example:

```text
Prediction: Likely to Churn
Churn Probability: 86.16%
```

---

# Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* GridSearchCV
* One-Hot Encoding
* StandardScaler

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# Key Business Insights

The analysis suggests several customer groups that may deserve additional retention attention:

* Customers on **month-to-month contracts** showed substantially higher churn.
* Customers with **shorter tenure** showed higher churn levels.
* Customers using **electronic check** had a substantially higher observed churn rate.
* **Senior citizens** showed a higher observed churn rate than non-senior customers.
* Customers with relatively higher **monthly charges** showed greater churn levels.

These findings represent patterns observed in the dataset and should be validated with additional business and customer research before being treated as causal relationships.

---

# Future Improvements

Potential future improvements include:

* Hyperparameter optimization of additional algorithms.
* Gradient boosting and XGBoost experimentation.
* Precision-Recall curve analysis.
* Feature importance and model interpretability using SHAP.
* Cross-validation comparison across multiple models.
* Interactive Streamlit prediction dashboard.
* Customer risk segmentation.
* Automated retention recommendations.
* Deployment using a cloud platform or API.

---

# Author

**Harsh Kumar**

B.Tech Computer Science & Engineering Student

Interested in **Data Science, Machine Learning, Python, and AI**.

---

## Project Goal

This project was developed as a hands-on Data Science portfolio project to demonstrate the complete machine learning workflow, from raw data exploration to model deployment-ready prediction.