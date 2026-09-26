from sqlalchemy import desc, select
from sqlalchemy.orm import Session, scoped_session

from core.sqlalchemy_models import Topic


class TopicRepository:
    def __init__(self, session: Session | scoped_session[Session]):
        self.session = session

    def most_similar(self, embedding: list[float], n: int) -> list[tuple[Topic, float]]:
        """The n topics with the highest cosine similarity to the embedding, most similar first."""
        similarity = (1 - Topic.embedding.cosine_distance(embedding)).label("similarity")
        query = select(Topic, similarity).order_by(desc("similarity")).limit(n)
        return [(row.Topic, row.similarity) for row in self.session.execute(query)]
