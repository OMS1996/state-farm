from fastapi import APIRouter, HTTPException
#from .models import PredictionInput, PredictionOutput
from typing import List, Dict, Union
from ml_model.model import load_model, predict
from ml_model.preprocessing import run_preprocess, create_preprocessors
from .dependencies import MODEL_PATH
import numpy as np
import pandas as pd

# Create a router object so we can define multiple API endpoints
router = APIRouter()

# Load your model
model = load_model(MODEL_PATH)

# Load imputers and scaler
# Assuming you have a mechanism to load these, similar to your model
numeric_imputer, std_scaler = create_preprocessors()

# Health check endpoint
@router.get("/")
def root():
    return {"message": "Welcome to the API! By Omar M. Hussein"}

@router.post("/predict")
async def get_prediction(input_data: Union[Dict[str, float], List[Dict[str, float]]]):
    try:
        # Convert the input list of dictionaries to a DataFrame
        df = pd.DataFrame(input_data)

        # Preprocess the data
        processed_data = run_preprocess(df, numeric_imputer, std_scaler)

        # Check for NaN or infinity values in processed_data
        if processed_data.isnull().values.any() or np.isinf(processed_data).any():
            raise ValueError("Invalid input data after preprocessing.")

        # Make predictions for the entire batch
        predictions = predict(model, processed_data)

        # Format and return the predictions
        results = [{"probability": pred[0],
                    "predicted_class": "customer_purchased" if pred[0] > 0.5 else "customer_did_not_purchase"}
                   for pred in predictions]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))