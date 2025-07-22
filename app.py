from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Any

# Load model and columns
model = joblib.load('catboost_salary_model.joblib')
model_columns = joblib.load('model_columns.joblib')

app = FastAPI(title="Employee Salary Prediction API")

# Dynamically create Pydantic model for input validation
class InputData(BaseModel):
    # Define all columns as optional Any, will validate in code
    __annotations__ = {col: Any for col in model_columns}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    input_dict = data.dict()
    # Check for missing columns
    missing_cols = [col for col in model_columns if col not in input_dict]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    # Create DataFrame in correct order
    X = pd.DataFrame([[input_dict[col] for col in model_columns]], columns=model_columns)
    # Predict
    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X)[0][1])
    return {"prediction": int(pred), "probability": proba} 