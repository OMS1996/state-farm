from fastapi import APIRouter, HTTPException
import traceback

# from .models import PredictionInput, PredictionOutput
from typing import Any, Dict, List, Union
from ml_model.model import load_model, predict
from ml_model.preprocessing import run_preprocess, create_preprocessors
from .dependencies import MODEL_PATH, VARIABLES
import numpy as np
import pandas as pd

# Create a router object so we can define multiple API endpoints
router = APIRouter()

# Load your model
model = load_model(MODEL_PATH)

# Load imputers and scaler
# Assuming you have a mechanism to load these, similar to your model
numeric_imputer, std_scaler = create_preprocessors()


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Health check endpoint
@router.get("/")
def root():
    return {"message": "Welcome to the API! By Omar M. Hussein"}


@router.post("/predict")
def get_prediction(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    selected_variables: List[str] = VARIABLES,
):
    try:
        # Convert the input data to a DataFrame
        df = pd.DataFrame(input_data)

        # Preprocess the data
        processed_data = run_preprocess(df)

        # Select only the columns you trained on
        processed_data = processed_data[selected_variables]

        # Make predictions row by row and return as a list of results.
        for col in processed_data.columns:
            processed_data[col] = processed_data[col].astype(float)
        predictions = predict(model, processed_data)

        # Format the predictions into a response
        results = [
            {
                "probability": pred,
                "predicted_class": "customer_purchased"
                if pred > 0.5
                else "customer_did_not_purchase",
            }
            for pred in predictions
        ]

        return results
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
