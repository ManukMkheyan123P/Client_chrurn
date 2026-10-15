# Customer Churn Prediction — Flask & Machine Learning

A Flask web application that estimates the probability of customer churn using a trained machine learning model. Enter customer details, view a churn prediction, and explore a simple analytics dashboard and a session-level customer list.

## Features

- **Churn prediction:** Predict whether a customer is likely to leave (`1`) or stay (`0`).
- **Risk probability:** Display the model's estimated probability of churn as a percentage.
- **Customer history:** View predictions submitted since the Flask server started at `/customers`.
- **Analytics dashboard:** View model performance metrics at `/analytics`.
- **Categorical preprocessing:** Encode contract and internet types using a saved OneHotEncoder.

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn-compatible model and encoder (loaded with joblib)
- HTML templates (Jinja2)

## Input Features

| Feature | Description |
| --- | --- |
| `age` | Customer age |
| `monthly_spend` | Monthly spending |
| `months_active` | Months as a customer |
| `support_calls` | Number of support calls |
| `late_payments` | Number of late payments |
| `satisfaction` | Customer satisfaction score |
| `contract_type` | Contract category |
| `internet_type` | Internet service category |
| `discount` | Discount amount |

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── clientMLprediction.pkl       # Trained model (required; provide separately)
├── churn_ohe.pkl                # Fitted encoder (required; provide separately)
├── templates/
│   ├── index.html               # Prediction form and result
│   ├── analytics.html           # Model metrics
│   └── customers.html           # Submitted customer predictions
└── README.md
```

> **Note:** The `.pkl` files and HTML templates are required to run this app. They are referenced in `app.py` but are not included with this README. Make sure they are present in your repository or local project.

## Installation & Usage

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install flask pandas scikit-learn joblib
   ```

   If the saved model uses XGBoost, also install `xgboost` and use package versions compatible with the versions used during training.

4. Place the trained model, encoder, and HTML templates in the locations shown above.

5. Run the app locally:

   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000/` in your browser. Visit `/analytics` and `/customers` for the additional pages.

## Model Metrics

The analytics page currently displays these **hard-coded values** from `app.py` (they are not recalculated when the app runs):

| Metric | Displayed value |
| --- | ---: |
| Accuracy | 73% |
| ROC-AUC | 0.779 |
| Precision | 68% |
| Recall | 53% |
| F1 score | 60% |

These values should be verified against the saved model's evaluation results before presenting them as independently reproduced benchmarks.

## How Prediction Works

1. The user submits customer data through the homepage form.
2. The app builds a one-row pandas DataFrame.
3. `contract_type` and `internet_type` are transformed with the saved encoder.
4. The encoded columns are joined to the numeric input features.
5. The saved model returns a predicted class and churn probability via `predict()` and `predict_proba()`.
6. The result is rendered on the homepage and appended to an in-memory customer list.

## Limitations & Future Improvements

- Customer records are stored **in memory only** and disappear when the server restarts; use a database for persistence.
- The analytics metrics are static; calculate them from a held-out test set for reproducibility.
- Input validation, exception handling, and automated tests should be added before production use.
- The app currently uses `debug=True`; disable debug mode in production.
- `app.py` defines the `/analytics` route twice. Remove the duplicate route definition before publishing or deploying.
- Predictions are estimates, not guarantees of future customer behavior.

## Purpose

This project demonstrates an end-to-end machine learning inference workflow: preprocessing user input, loading trained artifacts, serving predictions through Flask, and displaying results in a web interface.
