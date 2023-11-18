# Pydantic models for request and response data
from typing import List, Dict
from pydantic import BaseModel

class SinglePredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionInput(BaseModel):
    data: List[SinglePredictionInput]

class PredictionOutput(BaseModel):
    probability: float
    predicted_class: str

