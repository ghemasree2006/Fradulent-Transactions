# FraudDetection_101
This repository encomposes of fraud detection modelling scripts

A core ML project build to detect the fraudulent transactions using various classical classification algorithms in scale.

## Overview

This project implements classical machine learning models to detect potentially fraudulent transactions in the stream data. The systen is being trained on multiple models to classify transactions as fraudulent or legitimate class.

## Classical Model - View

Three different models have been trained and saved:
- *Logistic Regression* (lr_model.pkl): A baseline model providing good interpretability with the cost of f1-score
- *Random Forest* (rf_model.pkl): An ensemble method with high accuracy and realted metrics
- *XGBoost* (xgb_model.pkl): A gradient boosting implementation offering state-of-the-art performance (SOTA)

## Analysis

The notebook fraudulent_transactions_analysis.ipynb contains exploratory data analysis of the transaction dataset, including:
- Feature distributions
- Correlation analysis
- Anomaly detection
- Visualization of fraudulent patterns

## Model Training and Evaluation

The notebook modelling.ipynb documents:
- Data preprocessing steps
- Feature engineering
- Model training
- Performance evaluation metrics
- Model comparison

## Deployment

The project includes an app.py file to serve the trained models (ensemble combination) through an API endpoint.

## Installation

To set up the environment for this project:

1. Clone this repository
2. Install the required packages:

pip install -r models/requirements.txt


## Usage

### Running the Application

python
python models/app.py


### Using the Models

python
import pickle

# Load the model
with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Prepare your data (X)
# Make predictions
predictions = model.predict(X)

## Project Structure

├── .ipynb_checkpoints
├── data
│   └── Fraud.csv
├── notebooks
│   ├── _pycache_
│   ├── .ipynb_checkpoints
│   ├── fraudulent_transactions_analysis.ipynb
│   └── modelling.ipynb
├── scripts
└── models
    ├── lr_model.pkl
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── app.py
    └── requirements.txt


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
