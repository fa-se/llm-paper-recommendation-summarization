from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base


class OpenAlexBase(Base):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)  # ids come from OpenAlex
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    wikipedia: Mapped[str] = mapped_column(String)
    updated_date: Mapped[DateTime] = mapped_column(DateTime)
    embedding: Mapped[Vector] = mapped_column(Vector(1024))

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id},name={self.name})"
