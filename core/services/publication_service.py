import datetime
import logging
from itertools import chain
from os import environ

import pyalex

from core.dataclasses.data_classes import Work, ScoredWork
from core.llm_interfaces import LLMInterface
from core.services.user_service import UserService

logger = logging.getLogger(__name__)
pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")


class PublicationService:
    def __init__(self, user_service: UserService, llm_interface: LLMInterface):
        self.user_service = user_service
        self.llm_interface = llm_interface

    def get_works_by_topics(self, topic_ids: [int], published_after: datetime.date, n_max: int = 2000) -> list[Work]:
        topic_filter_str = "|".join(f"T{str(topic_id)}" for topic_id in topic_ids)
        # ISO 8601 date format
        from_publication_date_str = published_after.strftime("%Y-%m-%d")
        to_publication_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        query = (
            pyalex.Works()
            .filter(primary_topic={"id": topic_filter_str})
            .filter(from_publication_date=from_publication_date_str)
            .filter(to_publication_date=to_publication_date_str)
        )

        logger.info(
            f"Querying OpenAlex for works with topics {topic_ids} published after {published_after} (n_max={n_max}) Query URL: {query.url}"
        )

        works = []
        for pyalex_work in chain(*query.paginate(per_page=200, n_max=n_max)):
            works.append(Work(pyalex_work))

        return works

    def compute_relevance_scores_by_topics(
        self, works: list[Work], topics: list[int], topics_user_relevances: list[float]
    ) -> list[ScoredWork]:
        scored_works: list[ScoredWork] = []

        for work in works:
            score = 0
            # TODO Maybe use set intersection?
            for topic_id, user_relevance in zip(topics, topics_user_relevances):
                if topic_id in work.topics:
                    topic_score = work.topics[topic_id]["score"]
                    # there currently is a bug in the OpenAlex API, where the score is sometimes equal to the topic id. use -inf to ignore such cases
                    if topic_score == topic_id:
                        topic_score = float("-inf")
                    # topic_score -> how well does this work match the topic | user_relevance -> how relevant is this topic for the user
                    score += topic_score * user_relevance
            # Normalize by number of maximum possible matches
            score = score / (min(len(topics), len(work.topics)))
            if score == float("-inf"):
                logger.info(f"Ignoring invalid topic score for work {work}")
            else:
                scored_works.append(ScoredWork(work, score))

        return scored_works

    def compute_relevance_score_by_embedding_similarity(
        self,
        user_name: str,
        works: list[Work],
    ) -> list[ScoredWork]:
        # get user embedding
        area_of_interest_description = self.user_service.area_of_interest_description(user_name)
        area_of_interest_embedding = self.llm_interface.create_embedding(area_of_interest_description)

        # need to extract all abstracts first for batch embedding
        abstracts = []
        for work in works:
            if work.abstract:
                abstracts.append(work.abstract)

        embeddings = self.llm_interface.create_embedding_batch(abstracts)

        scored_works: list[ScoredWork] = []
        for work, embedding in zip(works, embeddings):
            # cosine similarity
            score = sum([a * b for a, b in zip(embedding, area_of_interest_embedding)])
            scored_works.append(ScoredWork(work, score))

        return scored_works
