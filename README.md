# Predicting FMCG Stockout Risk Using Machine Learning

## Run the app and live analytics

From the repository folder:

```sh
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/analytics.

- **Inventory dataset:** recalculates summaries and charts from the CSV on every request. Data is read from the first available path: `data/inventory_data.csv`, `data/processed/processed_inventory_data.csv`, or the checked-in `processed_inventory_data.csv`. Replace that file and click **Refresh analytics** to see updated results.
- **Saved predictions:** reads `outputs/predictions.csv`. Submit predictions on the Predict page, then refresh this view. Predicted risk bands and probabilities are estimates; they are not counted as actual stockout labels.
- **Analyse CSV:** upload a UTF-8 CSV up to 16 MB. The upload is used only for that response and does not overwrite data or retrain the model. It requires `current_stock`; other processed dataset columns are optional. Raw Kaggle headers must be converted to the processed schema first.

The dashboard uses calculated HTML bar charts with exact values, not the repository's saved PNGs. Missing, empty, or invalid data produces a visible error instead of old charts. Recognized numeric columns must have finite, non-negative values; stockout/promotion labels must be 0 or 1, and probabilities must be between 0 and 1. Missing stockout labels are treated as unavailable, not zero.

The app supports both this repository's flat files and conventional `templates/`, `static/`, `models/`, and `outputs/results/` locations. If a separate `templates/analytics.html` exists in your local copy, update it with this repository's `analytics.html` too.

Run the regression checks with:

```sh
python -m unittest test_analytics -v
```

## Introduction

In retail and FMCG businesses, maintaining the right inventory level is an important challenge. If inventory becomes too low, customers may find products unavailable. If inventory is too high, businesses may face unnecessary holding costs and excess stock.

This project explores how **Machine Learning can be used to estimate stockout risk within the next seven days** using historical inventory, sales, demand, pricing, promotion, and store/product information.

The goal is not to automatically place purchase orders, but to provide an **early warning signal that can support inventory-management decisions**.

---

## The Business Problem

A simple question motivated this project:

> **Can historical inventory and sales information help identify whether a product is at risk of a stockout within the next seven days?**

For an inventory manager, an early risk signal could help them investigate:

* Current stock levels
* Recent sales velocity
* Demand variability
* Promotions
* Pricing
* Product category
* Region
* Seasonal patterns

This transforms inventory management from simply looking at current stock into a more data-driven risk assessment process.

---

## Project Objective

The objective of this project is to develop a **binary classification model** that estimates whether an SKU-store-day observation has a derived stockout risk within seven days.

The target is represented as:

* **0 — No derived stockout risk**
* **1 — Derived stockout risk**

The application then converts the model's predicted probability into three project-defined risk bands:

* **Low Risk:** below 35%
* **Medium Risk:** 35%–64.9%
* **High Risk:** 65% or above

These thresholds are defined for this project and should not be interpreted as universal industry standards.

---

## Dataset

The project uses the **Retail Store Inventory Forecasting Dataset** from Kaggle.

The dataset contains approximately **73,100 records and 15 columns** covering inventory and retail-related information.

Important variables include:

* Date
* Store ID
* Product ID
* Category
* Region
* Inventory Level
* Units Sold
* Units Ordered
* Demand Forecast
* Price
* Discount
* Weather Condition
* Holiday/Promotion
* Competitor Pricing
* Seasonality

One important point is that the dataset is described as **synthetic but realistic**. Therefore, the project is presented as a machine-learning prototype rather than a system trained on confidential real-world company transactions.

---

## Creating the Target Variable

The original dataset does not contain the exact stockout label required for this project.

Therefore, a project-specific target called:

`stockout_within_7_days`

was created.

For each Store ID and Product ID, the following seven future days were examined.

A positive target is assigned when:

```text
Inventory Level <= Units Sold
```

occurs during the future seven-day period.

This means the target represents a **derived future inventory-shortfall proxy**, rather than a directly observed stockout event.

This distinction is important when interpreting the model results.

---

## Feature Engineering

Raw data is not always directly suitable for machine learning. Therefore, several useful features were created.

### Current Stock

Represents the available inventory level.

### Sales Velocity

Calculated from historical sales to represent how quickly the product is moving.

### Demand Variability

Measures variation in recent sales and helps represent demand uncertainty.

### Lead Time

The original dataset does not contain supplier lead time, so the project uses a disclosed three-day planning assumption.

### Promotion Status

Indicates whether a promotion or holiday-related event is present.

### Price and Discount

These variables can influence purchasing behavior and demand.

### Calendar Features

Additional time-related information includes:

* Day of week
* Month
* Seasonality

---

## Preventing Data Leakage

One of the important machine-learning concepts in this project is **data leakage**.

The model should not use information from the future to make a prediction about the present.

For example, when calculating sales velocity and demand variability, the project uses historical observations through shifted rolling features.

The future information is used for creating the target, but not as predictor information.

This helps maintain a more realistic prediction setup.

---

## Machine Learning Models

Instead of selecting one algorithm immediately, six classification models were compared:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Linear Support Vector Machine
5. Decision Tree
6. Random Forest

The purpose of comparing multiple models is to determine how different algorithms behave on the same prediction problem.

---

## Model Evaluation

The models were evaluated using a chronological holdout test set.

The following metrics were considered:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

For a stockout-risk problem, accuracy alone is not enough.

For example, if stockout-risk cases are relatively uncommon, a model could obtain high accuracy simply by predicting the majority class most of the time.

That is why **Recall, Precision, F1-score, and ROC-AUC** are also important.

---

## Model Comparison Results

The actual project comparison produced the following results:

| Model                | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| -------------------- | -------: | --------: | -----: | ----: | ------: |
| Logistic Regression  |   49.59% |     3.73% | 51.41% | 6.96% |   0.511 |
| KNN                  |     ~96% |        0% |     0% |    0% |   ~0.50 |
| Gaussian Naive Bayes |     ~96% |        0% |     0% |    0% |   ~0.50 |
| Linear SVM           |     ~96% |        0% |     0% |    0% |   ~0.50 |
| Decision Tree        |     ~96% |        0% |     0% |    0% |   ~0.50 |
| Random Forest        |     ~96% |        0% |     0% |    0% |   ~0.50 |

The exact values are stored in the project's:

```text
outputs/results/model_comparison.csv
```

### What Do These Results Tell Us?

An important observation is that several models achieve approximately 96% accuracy while predicting essentially no positive stockout-risk cases.

Therefore, their high accuracy should **not** be interpreted as strong stockout prediction performance.

Logistic Regression identifies substantially more positive cases, producing:

* **51.41% Recall**
* **6.96% F1-score**
* **0.511 ROC-AUC**

However, its low precision shows that many of its positive predictions are false positives.

This result demonstrates an important machine-learning lesson:

> **A high accuracy score does not automatically mean that a classification model is useful.**

---

## Why Logistic Regression Was Selected

The project's model-selection process prioritizes:

1. F1-score
2. Recall
3. ROC-AUC

Using this selection rule, **Logistic Regression** is currently selected.

The model is saved as:

```text
models/best_model.joblib
```

The selection is based on the project's evaluation results and does not mean that Logistic Regression is universally the best algorithm for stockout prediction.

---

## Flask Web Application

After training, the selected model is integrated into a **Flask web application**.

The application provides a user-friendly interface where users can enter inventory-related information and receive a stockout-risk prediction.

The workflow is:

```text
User Input
     ↓
Feature Preparation
     ↓
Saved ML Pipeline
     ↓
Prediction Probability
     ↓
Risk Classification
     ↓
Business Recommendation
```

The application displays the prediction as a decision-support signal rather than automatically making inventory purchases.

---

## Example Business Interpretation

Suppose the application estimates a high probability of stockout risk.

An inventory manager could then investigate:

* Is current stock too low?
* Has sales velocity increased?
* Is demand becoming more variable?
* Is a promotion increasing demand?
* Is replenishment required?
* Is the product affected by seasonal demand?

The model therefore acts as an **early-warning tool**.

---

## Project Architecture

The project follows a structured machine-learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Target Construction
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model
   ↓
Flask Application
   ↓
Stockout Risk Prediction
```

The project separates data processing, model training, saved models, outputs, templates, and application logic.

---

## Technology Stack

The project uses:

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Flask**
* **HTML**
* **CSS**
* **Matplotlib**

These technologies cover the complete workflow from data processing and machine learning to web deployment.

---

## Project Limitations

Like any machine-learning prototype, this project has limitations.

### 1. Synthetic Dataset

The source dataset is synthetic rather than actual company inventory data.

### 2. Derived Target

The stockout target is constructed from inventory and sales information rather than coming from a directly recorded stockout field.

### 3. Assumed Lead Time

A three-day planning assumption is used because the original dataset does not provide supplier lead-time information.

### 4. Class Imbalance

The evaluation results show that the positive class is difficult for several models to identify.

### 5. Not a Production System

The application should be considered a **coursework/prototype decision-support system**, not a production inventory-management solution.

---

## Future Improvements

A stronger real-world version could use:

* Actual company stockout records
* Real supplier lead times
* Supplier reliability
* Safety-stock information
* Reorder points
* Historical purchase orders
* Cost-sensitive learning
* Class-imbalance techniques
* Probability calibration
* Threshold optimization
* Walk-forward validation
* Model monitoring
* Data-drift detection
* Automated retraining

With real operational data, these improvements could make the system more suitable for practical inventory decision support.

---

## What I Learned

This project helped demonstrate that machine learning is not simply about obtaining the highest accuracy score.

The important lessons include:

* Defining the business problem clearly
* Creating an appropriate target
* Preventing data leakage
* Engineering meaningful features
* Comparing multiple algorithms
* Understanding class imbalance
* Evaluating models using multiple metrics
* Saving and deploying a trained model
* Connecting machine learning with a real business decision

Most importantly, the project showed that **model evaluation must be connected to the actual business problem**.

---

## Conclusion

The FMCG Stockout Risk Prediction project demonstrates how machine learning can be applied to inventory risk analysis.

The system takes historical retail information, creates a seven-day stockout-risk target, engineers useful inventory and demand features, compares six classification algorithms, selects a model using multiple evaluation metrics, and deploys the trained pipeline through a Flask web application.

Although the current results show important limitations, especially class imbalance and the use of a derived target, the project provides a complete and reproducible foundation for understanding **machine-learning-based inventory risk prediction**.

The next step toward a production-quality solution would be to train and validate the system using **real observed stockout events and operational supply-chain data**.

---

### Final Project Statement

> **“This project uses machine learning to estimate seven-day stockout risk from historical inventory, sales, demand variability, pricing, promotion, and store/product information. Multiple classification models are compared using accuracy, precision, recall, F1-score, and ROC-AUC, and the selected model is deployed through a Flask web application to provide inventory decision support.”**
