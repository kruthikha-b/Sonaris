from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.survey import Survey

router = APIRouter()


@router.put("/surveys/{survey_id}/status")
def update_status(
    survey_id: int,
    status: str,
    progress: int,
    db: Session = Depends(get_db)
):
    survey = db.query(Survey).filter(
        Survey.id == survey_id
    ).first()

    if not survey:
        return {"error": "Survey not found"}

    survey.status = status
    survey.progress = progress

    db.commit()
    db.refresh(survey)

    return survey