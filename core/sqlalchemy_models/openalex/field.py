from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import OpenAlexBase


class Field(OpenAlexBase):
    __tablename__ = "openalex_field"

    wikidata: Mapped[str] = mapped_column(String)

    domain_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_domain.id"))
    domain: Mapped["Domain"] = relationship(back_populates="fields")

    subfields: Mapped[list["Subfield"]] = relationship(back_populates="field")
