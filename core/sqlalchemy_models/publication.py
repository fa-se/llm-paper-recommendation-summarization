from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, BigInteger, String, DateTime, ForeignKey, Float, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Publication(Base):
    __tablename__ = "publication"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    openalex_id: Mapped[int] = mapped_column(BigInteger, unique=True)  # OpenAlex ids are too large for an Integer
    title: Mapped[str] = mapped_column(String, nullable=True)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String, dimensions=1), nullable=True)
    publication_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accessed_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))


class UserConfigPublicationAssociation(Base):
    __tablename__ = "userconfig_publication_association"

    user_config_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_config.id"), primary_key=True)
    publication_id: Mapped[int] = mapped_column(Integer, ForeignKey("publication.id"), primary_key=True)
    relevance_score: Mapped[float] = mapped_column(Float)

    # Define relationships
    user_config: Mapped["UserConfig"] = relationship(back_populates="scored_publications")
    publication: Mapped["Publication"] = relationship()
