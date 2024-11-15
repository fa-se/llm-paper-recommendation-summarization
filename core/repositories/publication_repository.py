from datetime import datetime

from sqlalchemy import select, desc, text, func

from core.sqlalchemy_models import Publication
from db import Session


class PublicationRepository:
    def __init__(self, session: Session):
        self.session = session

    def commit(self):
        self.session.commit()

    def create(
        self,
        openalex_id: int,
        title: str,
        authors: list[str],
        abstract: str,
        published: datetime,
        accessed: datetime,
        embedding: list[float],
    ) -> Publication:
        publication = Publication(
            openalex_id=openalex_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_datetime_utc=published,
            accessed_datetime_utc=accessed,
            embedding=embedding,
        )
        self.session.add(publication)
        return publication

    def get_by_openalex_id(self, openalex_id: int) -> Publication:
        return self.session.query(Publication).filter(Publication.openalex_id == openalex_id).one_or_none()

    def get_all_openalex_ids(self) -> list[int]:
        results = self.session.query(Publication.openalex_id).all()
        return [result[0] for result in results]

    def get_random_publications(self, n: int) -> list[Publication]:
        query = self.session.query(Publication).order_by(func.random()).limit(n)
        return query.all()

    def get_openalex_ids_by_embedding_similarity(
        self, embedding: list[float], top_n: int, start_date: datetime = None
    ) -> tuple[list[int], list[float]]:
        # Query to find the n most similar topics with similarity score (cosine similarity)
        query = select(
            Publication.openalex_id, (1 - Publication.embedding.cosine_distance(embedding)).label("similarity")
        )
        if start_date is not None:
            query = query.where(Publication.publication_datetime_utc >= start_date)
        query = query.order_by(desc("similarity")).limit(top_n)

        results = self.session.execute(query).all()

        ids = [result.openalex_id for result in results]
        similarities = [result.similarity for result in results]

        return ids, similarities

    def get_openalex_ids_by_bm25_similarity(
        self, query: str, top_n: int, start_date: datetime = None
    ) -> tuple[list[int], list[float]]:
        start_date_filter = ""
        if start_date is not None:
            start_date_filter = "AND publication_datetime_utc >= :start_date"

        query_raw = f"""
        SELECT openalex_id, score
        FROM
        (
            SELECT openalex_id, publication.publication_datetime_utc,
                    -(bm25 <#> bm25_query_to_svector('publication_abstract_bm25', :query, 'pgvector')::sparsevec) AS score
            FROM publication
        ) subquery   
        WHERE score != double precision 'NaN'
        {start_date_filter}
        ORDER BY score DESC
        LIMIT :top_n;
        """

        query_text = text(query_raw)
        params = {"query": query, "top_n": top_n}
        if start_date is not None:
            params["start_date"] = start_date

        results = self.session.execute(query_text, params).fetchall()

        ids = [result[0] for result in results]
        scores = [result[1] for result in results]

        return ids, scores

    def rebuild_bm25(self):
        # First, check if the materialized statistics view already exists
        view_exists = self.session.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM pg_matviews 
                    WHERE schemaname = 'public' AND matviewname = 'publication_abstract_bm25'
                );
                """)
        ).scalar()
        if view_exists:
            # refresh
            self.session.execute(
                text("""
                SELECT bm25_refresh('publication_abstract_bm25');
                """)
            )
        else:
            self.session.execute(
                text("""
                SELECT bm25_create('publication', 'abstract', 'publication_abstract_bm25'); 
                """)
            )

        self.session.execute(
            text("""
            UPDATE publication
            SET bm25 = bm25_document_to_svector('publication_abstract_bm25', abstract, 'pgvector')::sparsevec;
            """)
        )
        self.commit()

    def count(self) -> int:
        return self.session.query(Publication).count()

    def truncate(self):
        self.session.query(Publication).delete()
        self.commit()
