from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import OpenAlexBase


class Subfield(OpenAlexBase):
    __tablename__ = "openalex_subfield"

    wikidata: Mapped[str] = mapped_column(String)

    field_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_field.id"))
    field: Mapped["Field"] = relationship(back_populates="subfields")

    topics: Mapped[list["Topic"]] = relationship(back_populates="subfield")
