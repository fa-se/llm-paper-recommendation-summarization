from datetime import datetime

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session, scoped_session

from core.services.deduplication import text_key
from core.sqlalchemy_models import Publication
from core.works import Work

# OpenAlex ID -> score, best first
type Ranking = dict[int, float]


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
        """The n publications whose abstracts have the highest BM25 score for the query (pg_bestmatch)."""
        start_date_filter = "AND publication_datetime_utc >= :start_date" if start_date is not None else ""
        # MATERIALIZED: tokenize the query once, not once per row (for a paragraph-long query, 2 s -> 10 ms).
        # <#> is pgvector's negative inner product.
        sql = f"""
        WITH query_vector AS MATERIALIZED (
            SELECT bm25_query_to_svector('publication_abstract_bm25', :query, 'pgvector')::sparsevec AS bm25
        )
        SELECT openalex_id, score
        FROM
        (
            SELECT openalex_id, publication.publication_datetime_utc,
                    -(publication.bm25 <#> query_vector.bm25) AS score
            FROM publication, query_vector
        ) subquery
        WHERE score != double precision 'NaN'
        {start_date_filter}
        ORDER BY score DESC
        LIMIT :n;
        """
        params = {"query": query, "n": n}
        if start_date is not None:
            params["start_date"] = start_date
        return dict(self.session.execute(text(sql), params).all())

    def rebuild_bm25(self):
        """Updates the BM25 statistics (a materialized view of pg_bestmatch) and every publication's BM25 vector.

        All vectors, not only the new publications' ones: a document vector depends on the corpus statistics (average
        document length), which change with every ingest.
        """
        view_exists = self.session.scalar(
            text(
                "SELECT EXISTS (SELECT FROM pg_matviews "
                "WHERE schemaname = 'public' AND matviewname = 'publication_abstract_bm25')"
            )
        )
        if view_exists:
            self.session.execute(text("SELECT bm25_refresh('publication_abstract_bm25')"))
        else:
            self.session.execute(text("SELECT bm25_create('publication', 'abstract', 'publication_abstract_bm25')"))
        self.session.execute(
            text(
                "UPDATE publication "
                "SET bm25 = bm25_document_to_svector('publication_abstract_bm25', abstract, 'pgvector')::sparsevec"
            )
        )
        self.commit()
