# Pydantic models for request and response data
from typing import Dict
from pydantic import BaseModel

class PredictionInput(BaseModel):
    features: Dict[str, float]  # Accepting a dictionary of features

class PredictionOutput(BaseModel):
    probability: float
    predicted_class: str  # Adjust based on your classification labels
