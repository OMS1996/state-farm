# Router for the API endpoints
from fastapi import APIRouter, HTTPException
from .models import PredictionInput, PredictionOutput
from ml_model.model import load_model, predict
from ml_model.preprocessing import preprocess_data
from .dependencies import MODEL_PATH
from typing import List, Dict
import numpy as np

# Create a router object so we can define multiple API endpoints
router = APIRouter()

# Load your model
model = load_model(MODEL_PATH)

# Default endpoint
@router.get("/")
def root():
    return {"message": "Welcome to the Machine Learning API! By Omar M. Hussein  Please go to /docs for the API documentation."}

@router.post("/predict", response_model=List[PredictionOutput])
async def get_prediction(input_data: PredictionInput):
    try:
        results = []
        for single_input in input_data.data:
            # Preprocess data
            processed_data = preprocess_data(single_input.features)
            
            # Check for NaN or infinity values in processed_data
            if processed_data.isnull().values.any() or np.isinf(processed_data).any():
                raise ValueError("Invalid input data after preprocessing.")
            
            print("Processed data:  *()*!!!!!!")
            print(processed_data)
            # Make prediction
            prediction = predict(model, processed_data)
            
            # Convert prediction to your desired format
            probability = prediction[0]
            predicted_class = "customer_purchased" if probability > 0.5 else "customer_did_not_purchase"
            results.append(PredictionOutput(probability=probability, predicted_class=predicted_class))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

