# Loan Approval & Financial Risk Prediction Using Machine Learning

This is a Machine Learning project focused on helping financial institutions make safer and more informed lending decisions.

The project combines both classification and regression techniques to:

* Predict whether a loan application should be approved or rejected
* Predict the maximum loan amount that can be safely offered to a client

The system was developed using Python and Scikit-learn, with a strong focus on data preprocessing, financial risk analysis, model evaluation, and explainable machine learning.

---

## Project Overview

Financial institutions process large numbers of loan applications every day. Incorrect lending decisions can lead to:

* Increased loan defaults
* Financial losses
* Higher operational risk
* Poor customer trust

This project demonstrates how machine learning can be used to automate and improve loan-related decision making.

The project is divided into two major machine learning tasks:

## 1. Loan Approval Prediction (Classification)

The model predicts whether a client should be:

* Approved
* Rejected

based on:

* Income
* Employment history
* Credit history
* Interest rate
* Loan intent
* Home ownership
* Debt-related features

---

## 2. Maximum Loan Amount Prediction (Regression)

The regression model predicts the maximum loan amount a client may safely receive.

This helps simulate real-world financial decision making while minimizing lending risk.

---

# Technologies & Libraries Used

The project was developed using the following tools and libraries:

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Core programming language             |
| Pandas           | Data loading, cleaning, preprocessing |
| NumPy            | Numerical operations                  |
| Matplotlib       | Data visualization                    |
| Seaborn          | Statistical visualization             |
| Scikit-learn     | Machine learning models & evaluation  |
| Jupyter Notebook | Development environment               |

---

# Machine Learning Models Used

## Classification Models

### Logistic Regression

Used as the primary classification model.

Why it was used:

* Simple and interpretable
* Efficient on structured/tabular data
* Strong performance on binary classification problems
* Produces probability outputs

---

### Naïve Bayes

A probabilistic classifier based on Bayes Theorem.

Why it was used:

* Fast training
* Performs well on large datasets
* Useful baseline model

---

### K-Nearest Neighbour (KNN)

A non-parametric distance-based classification algorithm.

Why it was used:

* Simple implementation
* No strong assumptions about data distribution
* Useful for comparison with parametric models

---

## Regression Models

### Decision Tree Regressor (DT-1)

A fully grown regression tree.

Why it was used:

* Handles non-linear relationships
* Easy to interpret
* Strong predictive performance

---

### Pruned Decision Tree Regressor (DT-2)

A pre-pruned decision tree using restricted depth.

Why it was used:

* Reduce overfitting
* Improve generalization
* Compare performance against the fully grown tree

---

## Ensemble Learning

### Soft Voting Classifier

An ensemble combining:

* Logistic Regression
* Naïve Bayes

Why it was used:

* Combine strengths of multiple models
* Improve classification robustness
* Compare ensemble performance against individual models

---

# Dataset Preprocessing

A major part of the project focused on cleaning and preparing the dataset before model training.

Real-world financial datasets often contain:

* Missing values
* Invalid entries
* Outliers
* Inconsistent data

Improper preprocessing can significantly reduce machine learning performance.

---

## Preprocessing Steps Performed

### Missing Value Handling

Different techniques were used depending on the data type:

| Data Type   | Method Used              |
| ----------- | ------------------------ |
| Numerical   | Mean / Median Imputation |
| Categorical | Mode Imputation          |

Why?

* Median is more robust to outliers
* Mode is appropriate for categorical features

---

### Invalid Data Removal

The following issues were corrected:

* Unrealistic age values
* Negative employment lengths
* Negative interest rates
* Invalid loan amounts

Why?

* Invalid data negatively affects model learning and accuracy

---

### Feature Selection

Unnecessary columns were removed:

| Removed Feature                              | Reason                 |
| -------------------------------------------- | ---------------------- |
| ID                                           | No predictive value    |
| Maximum Loan Amount (classification dataset) | Prevent target leakage |

---

### Encoding Categorical Variables

Categorical values were converted into numerical representations for machine learning compatibility.

Example:

* Yes → 1
* No → 0

---

### Feature Scaling

Feature scaling was applied where necessary.

Why?

* Distance-based models like KNN are sensitive to feature scale
* Scaling ensures fair contribution from all features

---

# Model Evaluation

The models were evaluated using multiple performance metrics.

---

# Classification Metrics

| Metric    | Purpose                                      |
| --------- | -------------------------------------------- |
| Precision | Measures correctness of positive predictions |
| Recall    | Measures ability to identify risky clients   |
| F1-score  | Balance between precision and recall         |
| AUC-ROC   | Measures class separation ability            |

---

## Why Recall Was Important

Recall was treated as the most important classification metric.

In financial systems:

* False negatives are dangerous
* Approving risky clients may lead to loan defaults

Therefore, the model needed to correctly identify high-risk applicants.

---

# Regression Metrics

| Metric                   | Purpose                             |
| ------------------------ | ----------------------------------- |
| Mean Squared Error (MSE) | Measures prediction error magnitude |

---

## Why MSE Was Used

MSE heavily penalizes large prediction errors.

This is important in finance because:

* large loan prediction mistakes can cause major financial losses

---

# Hyperparameter Tuning

GridSearchCV was used for hyperparameter optimization.

Parameters tuned included:

* Logistic Regression `C`
* KNN `n_neighbors`
* Decision Tree `max_depth`

Why tuning was important:

* Improve model performance
* Reduce overfitting
* Improve generalization ability
* Find optimal model configuration

---

# Final Results

# Best Classification Model

## Logistic Regression

### Performance Highlights

* Recall: 0.98
* Strong Precision
* Strong F1-score
* Better AUC performance than KNN

### Why It Was Selected

Logistic Regression best satisfied the business objective of detecting risky clients while minimizing financial risk.

It produced highly reliable classification performance while remaining interpretable and efficient.

---

# Best Regression Model

## Fully Grown Decision Tree (DT-1)

### Why It Performed Best

* Lowest Mean Squared Error
* Better handling of complex relationships
* Fewer large prediction mistakes

This made it more suitable for financial loan amount estimation.

---

# Example Prediction

The trained regression model was used to estimate the maximum loan amount for a sample client based on:

* Financial information
* Credit history
* Employment details
* Interest rate
* Loan intent

This demonstrates how machine learning models can support real-world lending decisions.

---

# Project Structure

```bash
LoanPrediction_ML/
│
├── data/
│   ├── cleaned_classification_data.csv
│   ├── cleaned_regression_data.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── classification_models.ipynb
│   ├── regression_models.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── decision_tree_DT-1.png
│   ├── decision_tree_DT-2.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Key Learning Outcomes

This project demonstrates:

* Data preprocessing techniques
* Classification modelling
* Regression modelling
* Ensemble learning
* Hyperparameter tuning
* Financial risk analysis
* Model evaluation and comparison
* Practical machine learning workflow development

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/RiskLens.git
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# Future Improvements

Possible future enhancements:

* Deploy as a web application
* Add explainable AI techniques
* Use advanced ensemble methods
* Perform cross-validation
* Integrate real-time prediction APIs
* Add feature importance analysis

---

# Author

Inuka Wijerathna
