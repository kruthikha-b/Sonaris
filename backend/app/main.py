from fastapi import FastAPI

from backend.app.api.upload import router as upload_router

from backend.app.core.database import engine
from backend.app.core.database import Base

from backend.app.models.survey import Survey
from backend.app.api.surveys import router as survey_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SONARIS API")

app.include_router(upload_router)
app.include_router(survey_router)

@app.get("/")
def root():
    return {
        "message": "SONARIS backend running"
    }