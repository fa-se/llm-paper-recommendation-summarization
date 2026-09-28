from datetime import datetime

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session, scoped_session

from core.services.deduplication import text_key
from core.sqlalchemy_models import Publication
from core.works import Work

# OpenAlex ID -> score, best first
type Ranking = dict[int, float]

# VectorChord-bm25's index on publication.bm25, and the pg_tokenizer tokenizer of abstracts and queries (setup/ddl.sql)
BM25_INDEX = "publication_bm25"
TOKENIZER = "bert"


class PublicationRepository:
    def __init__(self, session: Session | scoped_session[Session]):
        self.session = session

    def add(self, work: Work, embedding: list[float], accessed: datetime) -> Publication:
        publication = Publication(
            openalex_id=work.id,
            title=work.title,
            authors=work.authors,
            abstract=work.abstract,
            publication_datetime_utc=work.publication_date,
            accessed_datetime_utc=accessed,
            embedding=embedding,
            title_key=text_key(work.title),
            abstract_key=text_key(work.abstract),
        )
        self.session.add(publication)
        return publication

    def commit(self):
        self.session.commit()

    def count(self) -> int:
        return self.session.scalar(select(func.count()).select_from(Publication))

    def get_by_openalex_ids(self, openalex_ids: list[int]) -> list[Publication]:
        """Returns the publications in the order of the given IDs, skipping unknown IDs."""
        publications = self.session.scalars(select(Publication).where(Publication.openalex_id.in_(openalex_ids)))
        by_id = {publication.openalex_id: publication for publication in publications}
        return [by_id[openalex_id] for openalex_id in openalex_ids if openalex_id in by_id]

    def get_titles(self, openalex_ids: list[int]) -> dict[int, str]:
        query = select(Publication.openalex_id, Publication.title).where(Publication.openalex_id.in_(openalex_ids))
        return dict(self.session.execute(query).all())

    def get_all_openalex_ids(self) -> set[int]:
        return set(self.session.scalars(select(Publication.openalex_id)))

    def get_all_dedupe_keys(self) -> tuple[list[str], list[str]]:
        """Returns the title keys and the abstract keys of all publications."""
        rows = self.session.execute(select(Publication.title_key, Publication.abstract_key)).all()
        title_keys = [title_key for title_key, _ in rows if title_key]
        abstract_keys = [abstract_key for _, abstract_key in rows if abstract_key]
        return title_keys, abstract_keys

    def search_by_embedding(self, embedding: list[float], n: int, start_date: datetime | None = None) -> Ranking:
        """The n publications with the highest cosine similarity to the embedding (exact search, no index)."""
        similarity = (1 - Publication.embedding.cosine_distance(embedding)).label("similarity")
        query = select(Publication.openalex_id, similarity)
        if start_date is not None:
            query = query.where(Publication.publication_datetime_utc >= start_date)
        query = query.order_by(desc("similarity")).limit(n)
        return dict(self.session.execute(query).all())

    def search_by_bm25(self, query: str, n: int, start_date: datetime | None = None) -> Ranking:
        """The n publications whose abstracts have the highest BM25 score for the query (VectorChord-bm25)."""
        # exact: the index scores every abstract that shares a token with the query (brute force). By default it stops
        # at its best 100 by block-max WAND, and a date filter afterwards could leave fewer than n.
        self.session.execute(text("SET LOCAL bm25_catalog.bm25_limit = -1"))
        start_date_filter = "WHERE publication_datetime_utc >= :start_date" if start_date is not None else ""
        # the CTE tokenizes the query once, not again for every row returned; <&> is the negative BM25 score
        sql = f"""
        WITH query AS MATERIALIZED (
            SELECT to_bm25query('{BM25_INDEX}', tokenize(:query, '{TOKENIZER}')) AS bm25
        )
        SELECT openalex_id, -(publication.bm25 <&> (SELECT bm25 FROM query)) AS score
        FROM publication
        {start_date_filter}
        ORDER BY publication.bm25 <&> (SELECT bm25 FROM query)
        LIMIT :n;
        """
        params = {"query": query, "n": n}
        if start_date is not None:
            params["start_date"] = start_date
        return dict(self.session.execute(text(sql), params).all())

    def index_bm25(self):
        """Tokenizes the abstracts that have no BM25 vector yet, i.e. the new publications.

        The BM25 index adds them to its corpus statistics (document count, average length, token frequencies) itself,
        with the same scores as a rebuild, so the other publications need no update.
        """
        self.session.execute(
            text(
                f"UPDATE publication SET bm25 = tokenize(abstract, '{TOKENIZER}') "
                "WHERE bm25 IS NULL AND abstract IS NOT NULL"
            )
        )
        self.commit()

    def rebuild_bm25(self):
        """Tokenizes any new abstracts and rebuilds the BM25 index. Needed after deleting publications: the index drops
        deleted rows from its statistics only when vacuumed, and a rebuild makes the scores exact at once."""
        self.index_bm25()
        self.session.execute(text(f"REINDEX INDEX {BM25_INDEX}"))
        self.commit()
