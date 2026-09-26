"""Database tables: the OpenAlex topic taxonomy (domain > field > subfield > topic), and the stored publications.

The taxonomy with its embeddings is loaded once from setup/openalex_embeddings.sql; only topics are used, for routing a
query to the parts of OpenAlex to fetch works from. The schema in setup/ddl.sql is generated from these models
(scripts/save_ddl.py).
"""

from datetime import datetime

from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import ARRAY, BigInteger, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class OpenAlexEntity(Base):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)  # ids come from OpenAlex
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    wikipedia: Mapped[str] = mapped_column(String)
    updated_date: Mapped[datetime] = mapped_column(DateTime)
    # embedding of "Topic: {name}; Description: {description}; Keywords: {keywords}" (notebooks/create_topic_embeddings)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id},name={self.name})"


class Domain(OpenAlexEntity):
    __tablename__ = "openalex_domain"

    wikidata: Mapped[str] = mapped_column(String)

    fields: Mapped[list["Field"]] = relationship(back_populates="domain")


class Field(OpenAlexEntity):
    __tablename__ = "openalex_field"

    wikidata: Mapped[str] = mapped_column(String)

    domain_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_domain.id"))
    domain: Mapped[Domain] = relationship(back_populates="fields")

    subfields: Mapped[list["Subfield"]] = relationship(back_populates="field")


class Subfield(OpenAlexEntity):
    __tablename__ = "openalex_subfield"

    wikidata: Mapped[str] = mapped_column(String)

    field_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_field.id"))
    field: Mapped[Field] = relationship(back_populates="subfields")

    topics: Mapped[list["Topic"]] = relationship(back_populates="subfield")


class Topic(OpenAlexEntity):
    __tablename__ = "openalex_topic"

    keywords: Mapped[list[str]] = mapped_column(ARRAY(String, dimensions=1))

    subfield_id: Mapped[int] = mapped_column(Integer, ForeignKey("openalex_subfield.id"))
    subfield: Mapped[Subfield] = relationship(back_populates="topics")


class Publication(Base):
    __tablename__ = "publication"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    openalex_id: Mapped[int] = mapped_column(BigInteger, unique=True)  # OpenAlex ids are too large for an Integer
    title: Mapped[str] = mapped_column(String, nullable=True)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String, dimensions=1), nullable=True)
    publication_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accessed_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    abstract: Mapped[str] = mapped_column(String, nullable=True)
    # BM25 document vector of the abstract; recomputed for all rows on every ingest (PublicationRepository.rebuild_bm25)
    bm25: Mapped[list[float]] = mapped_column(SPARSEVEC, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))
    # hashes of the normalized title and abstract, used to skip duplicates (see core/services/deduplication.py)
    title_key: Mapped[str] = mapped_column(String(40), nullable=True, index=True)
    abstract_key: Mapped[str] = mapped_column(String(40), nullable=True, index=True)
