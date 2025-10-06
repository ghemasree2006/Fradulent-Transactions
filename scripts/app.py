# app.py
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API",
              description="API for detecting fraudulent transactions using ensemble ML models",
              version="1.0.0")

# Load pickled models
try:
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open('models/lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # You might also need to load any preprocessing pipelines or scalers
    # with open('models/scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    # We'll handle this gracefully when the app runs

# Define request body model based on your features
class TransactionData(BaseModel):
    # Core transaction features
    step: int
    amount: float
    type_CASH_OUT: int = 0
    type_DEBIT: int = 0
    type_PAYMENT: int = 0
    type_TRANSFER: int = 0
    
    # Engineered features
    sender_balance_pct_change: float
    receiver_balance_pct_change: float
    log_amount: float
    orig_zero_balance: int
    dest_zero_balance: int
    errorBalanceOrig: int
    errorBalanceDest: int
    is_merchant_dest: int

    class Config:
        schema_extra = {
            "example": {
                "step": 1,
                "amount": 9839.64,
                "type_CASH_OUT": 1,
                "type_PAYMENT": 0,
                "type_TRANSFER": 0,
                "type_DEBIT": 0,
                "sender_balance_pct_change": -100.0,
                "receiver_balance_pct_change": 0.0,
                "log_amount": 9.19,
                "orig_zero_balance": 0,
                "dest_zero_balance": 1,
                "errorBalanceOrig": 0,
                "errorBalanceDest": 0,
                "is_merchant_dest": 0
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    model_agreement: dict
    explanation: str

@app.get('/')
def home():
    return {"message": "Fraud Detection API is up and running!",
            "docs": "Visit /docs for the API documentation"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: TransactionData):
    try:
        # Convert input data to DataFrame with correct feature order
        input_data = pd.DataFrame([data.dict()])
        
        # Make predictions with all models
        rf_prob = rf_model.predict_proba(input_data)[0][1]
        lr_prob = lr_model.predict_proba(input_data)[0][1]
        xgb_prob = xgb_model.predict_proba(input_data)[0][1]
        
        # Get the maximum probability
        max_prob = max(rf_prob, lr_prob, xgb_prob)
        
        # Define a threshold (you might want to tune this based on your needs)
        threshold = 0.5
        prediction = 1 if max_prob > threshold else 0
        
        # Create model agreement dictionary
        model_agreement = {
            "random_forest": {"fraud_probability": float(rf_prob), "prediction": 1 if rf_prob > threshold else 0},
            "logistic_regression": {"fraud_probability": float(lr_prob), "prediction": 1 if lr_prob > threshold else 0},
            "xgboost": {"fraud_probability": float(xgb_prob), "prediction": 1 if xgb_prob > threshold else 0}
        }
        
        # Generate simple explanation based on the prediction
        if prediction == 1:
            explanation = "This transaction appears to be fraudulent. High risk indicators include: "
            # Add explanations based on feature values
            if data.errorBalanceOrig == 1:
                explanation += "sender balance error, "
            if data.errorBalanceDest == 1:
                explanation += "receiver balance error, "
            if data.orig_zero_balance == 1:
                explanation += "sender started with zero balance, "
            if data.sender_balance_pct_change < -99:
                explanation += "sender's balance was depleted, "
            explanation = explanation.rstrip(", ") + "."
        else:
            explanation = "This transaction appears to be legitimate based on our models' analysis."
        
        return PredictionResponse(
            prediction=prediction,
            fraud_probability=float(max_prob),
            model_agreement=model_agreement,
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Helper endpoint to get feature information
@app.get('/features')
def features():
    return {
        "features": [
            "step", "amount", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", 
            "type_TRANSFER", "sender_balance_pct_change", "receiver_balance_pct_change", 
            "log_amount", "orig_zero_balance", "dest_zero_balance", "errorBalanceOrig", 
            "errorBalanceDest", "is_merchant_dest"
        ],
        "description": "These are the required features for making predictions. The 'type_*' features are one-hot encoded transaction types."
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)