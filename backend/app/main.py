from fastapi import FastAPI

from backend.app.api.upload import router as upload_router

from backend.app.core.database import engine
from backend.app.core.database import Base

from backend.app.models.survey import Survey
from backend.app.api.surveys import router as survey_router
from backend.app.models.detection import Detection

from backend.app.api.detections import router as detection_router

from backend.app.api.ml import router as ml_router
from backend.app.api.reports import router as report_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SONARIS API")

app.include_router(upload_router)
app.include_router(survey_router)
app.include_router(detection_router)
app.include_router(ml_router)
app.include_router(report_router)

@app.get("/")
def root():
    return {
        "message": "SONARIS backend running"
    }