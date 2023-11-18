# Router for the API endpoints
from fastapi import APIRouter, HTTPException
from .models import PredictionInput, PredictionOutput
from ml_model.model import load_model, predict
from ml_model.preprocessing import preprocess_data
from dependencies import MODEL_PATH

router = APIRouter()

# Load your model (consider loading it at startup rather than on each call)
# model path and model.pkl
model = load_model(MODEL_PATH + "/model.pkl")

@router.post("/predict", response_model=PredictionOutput)
async def get_prediction(input_data: PredictionInput):
    try:
        # Preprocess data
        processed_data = preprocess_data(input_data.dict())
        # Make prediction
        prediction = predict(model, processed_data)
        # Convert prediction to your desired format
        return PredictionOutput(probability=prediction, predicted_class="your_class")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
