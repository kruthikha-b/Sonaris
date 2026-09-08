from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from reports.json_report import generate_json_report
from reports.csv_report import generate_csv_report

from fastapi.responses import FileResponse

from backend.app.services.pdf_service import generate_pdf_report
from backend.app.models.detection import Detection

router = APIRouter()


@router.get("/report/json")
def json_report():

    data = [
        {
            "survey_id": 1,
            "class_name": "pipeline",
            "confidence": 0.95
        }
    ]

    filename = generate_json_report(
        data,
        "report.json"
    )

    return {
        "report": filename
    }


@router.get("/report/csv")
def csv_report():

    data = [
        {
            "survey_id": 1,
            "class_name": "pipeline",
            "confidence": 0.95
        }
    ]

    filename = generate_csv_report(
        data,
        "report.csv"
    )

    return {
        "report": filename
    }

@router.get("/reports/pdf")
def pdf_report(
    db: Session = Depends(get_db)
):
    detections = db.query(Detection).all()

    pdf_path = generate_pdf_report(detections)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="sonaris_report.pdf"
    )