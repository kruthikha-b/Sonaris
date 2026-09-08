from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.survey import Survey
from backend.app.models.detection import Detection

router = APIRouter()


@router.get("/analytics/summary")
def analytics_summary(
    db: Session = Depends(get_db)
):
    total_surveys = db.query(Survey).count()

    total_detections = db.query(Detection).count()

    pending = db.query(Detection).filter(
        Detection.status == "Pending"
    ).count()

    confirmed = db.query(Detection).filter(
        Detection.status == "Confirmed"
    ).count()

    rejected = db.query(Detection).filter(
        Detection.status == "Rejected"
    ).count()

    high_priority = db.query(Detection).filter(
        Detection.priority == "HIGH"
    ).count()

    return {
        "total_surveys": total_surveys,
        "total_detections": total_detections,
        "pending": pending,
        "confirmed": confirmed,
        "rejected": rejected,
        "high_priority": high_priority
    }