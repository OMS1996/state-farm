from fastapi import APIRouter, HTTPException
from .models import PredictionInput, PredictionOutput
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

@router.post("/predict")
def get_prediction(input_data: PredictionInput):
    print(input_data)
    try:
        # Convert the input data to a DataFrame
        df = pd.DataFrame([item.features for item in input_data.data])

        # Preprocess the entire batch of data
        processed_data = preprocess_data(df, numeric_imputer, std_scaler)

        # Check for NaN or infinity values in processed_data
        if processed_data.isnull().values.any() or np.isinf(processed_data).any():
            raise ValueError("Invalid input data after preprocessing.")
        
        print("Processed data:\n", processed_data)

        # Make predictions for the entire batch
        predictions = predict(model, processed_data)

        # Create the response
        results = [PredictionOutput(probability=pred[0], 
                                    predicted_class="customer_purchased" if pred[0] > 0.5 else "customer_did_not_purchase")
                   for pred in predictions]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
