from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.detection import Detection

router = APIRouter()


@router.post("/detections")
def create_detection(
    survey_id: int,
    class_name: str,
    confidence: float,
    db: Session = Depends(get_db)
):
    detection = Detection(
        survey_id=survey_id,
        class_name=class_name,
        confidence=confidence
    )

    db.add(detection)
    db.commit()
    db.refresh(detection)

    return detection


@router.get("/detections")
def get_detections(
    db: Session = Depends(get_db)
):
    return db.query(Detection).all()


@router.get("/detections/{detection_id}")
def get_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    return db.query(Detection).filter(
        Detection.id == detection_id
    ).first()