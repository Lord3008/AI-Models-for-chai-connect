from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    text: str
    model_name: str
    parameters: Optional[dict] = None

class PredictionResponse(BaseModel):
    result: str
    confidence: float
    model_name: str
    sources: Optional[List[Dict[str, Any]]] = None
