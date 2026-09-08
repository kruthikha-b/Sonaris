from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

from backend.app.core.database import Base


class Survey(Base):
    __tablename__ = "surveys"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String, nullable=False)

    status = Column(String, default="uploaded")