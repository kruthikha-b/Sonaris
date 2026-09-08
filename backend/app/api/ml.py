from fastapi import APIRouter

from backend.app.services.ml_service import predict

router = APIRouter()


@router.get("/predict")
def run_prediction():

    results = predict("sample.jpg")

    return {
        "detections": results
    }