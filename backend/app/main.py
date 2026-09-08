from fastapi import FastAPI

from backend.app.api.upload import router as upload_router
from backend.app.api.surveys import router as survey_router
from backend.app.api.detections import router as detection_router
from backend.app.api.ml import router as ml_router
from backend.app.api.reports import router as report_router
from backend.app.api.analytics import router as analytics_router
from backend.app.api.status import router as status_router
from backend.app.api.auth import router as auth_router

from backend.app.models.user import User

app = FastAPI(
    title="SONARIS API",
    description="Underwater Sonar Anomaly Detection System",
    version="1.0.0"
)

# Routes
app.include_router(upload_router)
app.include_router(survey_router)
app.include_router(detection_router)
app.include_router(ml_router)
app.include_router(report_router)
app.include_router(analytics_router)
app.include_router(status_router)
app.include_router(auth_router)

@app.get("/")
def root():
    return {
        "message": "SONARIS backend running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }