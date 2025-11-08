from fastapi import APIRouter, HTTPException
from models.schemas import PredictionRequest, PredictionResponse
from services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = await model_service.get_prediction(
            text=request.text,
            model_name=request.model_name,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    return {"models": model_service.get_available_models()}
