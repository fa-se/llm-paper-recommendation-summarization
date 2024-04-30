from datetime import datetime

from sqlalchemy import select, desc

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
        published: datetime,
        accessed: datetime,
        embedding: list[float],
    ) -> Publication:
        publication = Publication(
            openalex_id=openalex_id,
            title=title,
            authors=authors,
            publication_datetime_utc=published,
            accessed_datetime_utc=accessed,
            embedding=embedding,
        )
        self.session.add(publication)
        return publication

    def get_all_openalex_ids(self) -> list[int]:
        results = self.session.query(Publication.openalex_id).all()
        return [result[0] for result in results]

    def get_openalex_ids_by_embedding_similarity(
        self, embedding: list[float], top_n: int
    ) -> tuple[list[int], list[float]]:
        # Query to find the n most similar topics with similarity score (cosine similarity)
        query = (
            select(Publication.openalex_id, (1 - Publication.embedding.cosine_distance(embedding)).label("similarity"))
            .order_by(desc("similarity"))
            .limit(top_n)
        )
        results = self.session.execute(query).all()

        ids = [result.openalex_id for result in results]
        similarities = [result.similarity for result in results]

        return ids, similarities
