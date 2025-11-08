from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    text: str
    model_name: str
    parameters: Optional[dict] = None

class PredictionResponse(BaseModel):
    result: str
    confidence: float
    model_name: str
