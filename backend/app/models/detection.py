from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float

from backend.app.core.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    survey_id = Column(Integer, nullable=False)

    class_name = Column(String, nullable=False)

    anomaly_score = Column(Float, default=0)

    priority = Column(String, default="LOW")

    confidence = Column(Float, nullable=False)

    status = Column(String, default="Pending")

    review_notes = Column(String, nullable=True)

    latitude = Column(Float, nullable=True)

    longitude = Column(Float, nullable=True)

    uncertainty = Column(Float, nullable=True)