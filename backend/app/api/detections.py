from fastapi import APIRouter, Depends
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

    # Calculate priority
    if confidence >= 0.90:
        priority = "HIGH"
    elif confidence >= 0.75:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # Calculate anomaly score
    anomaly_score = confidence * 100

    detection = Detection(
        survey_id=survey_id,
        class_name=class_name,
        confidence=confidence,
        anomaly_score=anomaly_score,
        priority=priority
    )

    db.add(detection)
    db.commit()
    db.refresh(detection)

    return detection


@router.get("/detections")
def get_detections(
    status: str = None,
    priority: str = None,
    class_name: str = None,
    db: Session = Depends(get_db)
):
    query = db.query(Detection)

    if status:
        query = query.filter(
            Detection.status == status
        )

    if priority:
        query = query.filter(
            Detection.priority == priority
        )

    if class_name:
        query = query.filter(
            Detection.class_name == class_name
        )

    return query.all()


@router.get("/detections/{detection_id}")
def get_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    return db.query(Detection).filter(
        Detection.id == detection_id
    ).first()


@router.put("/detections/{detection_id}/review")
def review_detection(
    detection_id: int,
    status: str,
    review_notes: str = "",
    db: Session = Depends(get_db)
):
    detection = db.query(Detection).filter(
        Detection.id == detection_id
    ).first()

    if not detection:
        return {"error": "Detection not found"}

    detection.status = status
    detection.review_notes = review_notes

    db.commit()
    db.refresh(detection)

    return detection


@router.get("/detections/high-priority")
def get_high_priority(
    db: Session = Depends(get_db)
):
    return db.query(Detection).filter(
        Detection.priority == "HIGH"
    ).all()