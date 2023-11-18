# Router for the API endpoints
from fastapi import APIRouter, HTTPException
from .models import PredictionInput, PredictionOutput
from ml_model.model import load_model, predict
from ml_model.preprocessing import preprocess_data
from dependencies import MODEL_PATH

router = APIRouter()

# Load your model
model = load_model(MODEL_PATH)

@router.post("/predict", response_model=PredictionOutput)
async def get_prediction(input_data: PredictionInput):
    try:
        # Preprocess data
        processed_data = preprocess_data(input_data.features)  # Assuming input_data.features is a dictionary
        # Make prediction
        prediction = predict(model, processed_data)
        # Convert prediction to your desired format
        # Example: Extracting probability and determining class
        # Adjust this based on your model's output and class determination logic
        probability = prediction[0]
        predicted_class = "customer_purchased" if probability > 0.5 else "customer_did_not_purchase"  # Example threshold
        return PredictionOutput(probability=probability, predicted_class=predicted_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
