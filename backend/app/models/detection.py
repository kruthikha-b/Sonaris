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

    confidence = Column(Float, nullable=False)