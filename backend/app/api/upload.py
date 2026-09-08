from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import Depends

from sqlalchemy.orm import Session

import os

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.survey import Survey

router = APIRouter()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_sonar_file(
    current_user: str = Depends(get_current_user),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    survey = Survey(
        filename=file.filename,
        status="uploaded",
        progress=0
    )

    db.add(survey)
    db.commit()
    db.refresh(survey)

    return {
        "message": f"File uploaded by {current_user}",
        "survey_id": survey.id,
        "filename": survey.filename,
        "status": survey.status
    }