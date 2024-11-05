from datetime import datetime

from pgvector.sqlalchemy import Vector, SPARSEVEC
from sqlalchemy import Integer, BigInteger, String, DateTime, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Publication(Base):
    __tablename__ = "publication"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    openalex_id: Mapped[int] = mapped_column(BigInteger, unique=True)  # OpenAlex ids are too large for an Integer
    title: Mapped[str] = mapped_column(String, nullable=True)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String, dimensions=1), nullable=True)
    publication_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accessed_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    abstract: Mapped[str] = mapped_column(String, nullable=True)
    bm25: Mapped[list[float]] = mapped_column(SPARSEVEC, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))
