from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

import os

from backend.app.core.database import get_db
from backend.app.models.survey import Survey
from backend.app.services.ml_service import predict

from backend.app.models.detection import Detection

router = APIRouter()


@router.post("/run-ml/{survey_id}")
def run_ml(
    survey_id: int,
    db: Session = Depends(get_db)
):
    survey = db.query(Survey).filter(
        Survey.id == survey_id
    ).first()

    if not survey:
        return {
            "error": "Survey not found"
        }

    file_path = os.path.join(
        "uploads",
        survey.filename
    )

    if not os.path.exists(file_path):
        return {
            "error": "Uploaded file not found"
        }

    results = predict(file_path)

    saved=[]

    for item in results["detections"]:
        confidence = item.get("confidence")

        if confidence >= 0.90:
            priority = "HIGH"
        elif confidence >= 0.75:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        anomaly_score = confidence * 100

        detection = Detection(
            survey_id=survey_id,
            class_name=item["class"],
            confidence=confidence,
            anomaly_score=anomaly_score,
            priority=priority,
            status="PENDING"
        )

        db.add(detection)
        saved.append({
            "class_name": item["class"],
            "confidence": confidence
        })
    db.commit()
    survey.status = "completed"
    survey.progress = 100
    db.commit()
    return{
        "survey_id": survey_id,
        "filename": survey.filename,
        "detections_saved": len(saved),
        "detections": saved
    }