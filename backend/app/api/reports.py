from fastapi import APIRouter

from reports.json_report import generate_json_report
from reports.csv_report import generate_csv_report

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