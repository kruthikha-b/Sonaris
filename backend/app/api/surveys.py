from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.survey import Survey

router = APIRouter()


@router.get("/surveys")
def get_surveys(db: Session = Depends(get_db)):
    return db.query(Survey).all()


@router.get("/surveys/{survey_id}")
def get_survey(survey_id: int, db: Session = Depends(get_db)):
    return db.query(Survey).filter(
        Survey.id == survey_id
    ).first()