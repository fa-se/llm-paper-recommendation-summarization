from sqlalchemy import select, desc

from core.sqlalchemy_models.openalex.topic import Topic
from db import Session


class TopicRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_topics_by_embedding_similarity(self, embedding: list[float], top_n: int) -> tuple[list[Topic], list[float]]:
        # Query to find the n most similar topics with similarity score (cosine similarity)
        query = (
            select(Topic, (1 - Topic.embedding.cosine_distance(embedding)).label("similarity"))
            .order_by(desc("similarity"))
            .limit(top_n)
        )
        results = self.session.execute(query).all()

        topics = [result.Topic for result in results]
        similarities = [result.similarity for result in results]

        return topics, similarities
