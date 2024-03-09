from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import OpenAlexBase


class Domain(OpenAlexBase):
    __tablename__ = "openalex_domain"

    wikidata: Mapped[str] = mapped_column(String)

    fields: Mapped[list["Field"]] = relationship(back_populates="domain")
