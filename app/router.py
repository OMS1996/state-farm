from fastapi import APIRouter, HTTPException
from .models import InputData, PredictionOutput
from typing import List, Dict
from ml_model.model import load_model, predict
from ml_model.preprocessing import preprocess_data, create_preprocessors
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

# Here is an example curl call to your API:

# curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"x0": "-1.018506", 
# "x1": "-4.180869", "x2": "5.70305872366547", 
# "x3": "-0.522021597308617", ...,"x99": "2.55535888"}'

# or a batch curl call:

# curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"},{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}]'

@router.post("/predict")
async def get_prediction(input_data: InputData):
    try:
        # Convert the input list of dictionaries to a DataFrame
        df = pd.DataFrame(input_data)

        # Preprocess the data
        processed_data = preprocess_data(df, numeric_imputer, std_scaler)

        print("DATA HAS BEEN PROCESSED")

        # Check for NaN or infinity values in processed_data
        if processed_data.isnull().values.any() or np.isinf(processed_data).any():
            print("Invalid input data after preprocessing.")
            raise ValueError("Invalid input data after preprocessing.")

        # Make predictions for the entire batch
        predictions = predict(model, processed_data)

        #
        print("PREDICTIONS HAVE BEEN MADE")

        # Format and return the predictions
        results = [{"probability": pred[0],
                    "predicted_class": "customer_purchased" if pred[0] > 0.5 else "customer_did_not_purchase"}
                   for pred in predictions]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 
