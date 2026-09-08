from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf_report(detections):

    pdf = SimpleDocTemplate("report.pdf")

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("SONARIS Detection Report", styles["Title"])
    )

    content.append(
        Paragraph(f"Total Detections: {len(detections)}",
                  styles["Normal"])
    )

    for d in detections:

        content.append(
            Paragraph(
                f"""
                ID: {d.id}<br/>
                Class: {d.class_name}<br/>
                Confidence: {d.confidence}<br/>
                Priority: {d.priority}<br/>
                Status: {d.status}<br/>
                """,
                styles["Normal"]
            )
        )

    pdf.build(content)

    return "report.pdf"