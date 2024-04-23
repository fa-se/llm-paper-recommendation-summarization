from sqlalchemy import Integer, String, ForeignKey, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import OpenAlexBase


class Topic(OpenAlexBase):
    __tablename__ = "openalex_topic"

    keywords: Mapped[[str]] = mapped_column(ARRAY(String, dimensions=1))

    subfield_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_subfield.id"))
    subfield: Mapped["Subfield"] = relationship(back_populates="topics")
