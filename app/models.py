# Pydantic models for request and response data
from typing import List, Dict, Union
from pydantic import BaseModel

class SinglePredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    probability: float
    predicted_class: str

class InputData(BaseModel):
    input_data: List[Dict[str, any]]