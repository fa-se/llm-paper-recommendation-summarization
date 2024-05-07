import datetime
import logging
from itertools import chain
from os import environ

import pyalex

from core.dataclasses.data_classes import Work, ScoredWork, SummarizedWork
from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import CustomizedSummaryTask
from core.repositories.publication_repository import PublicationRepository
from core.services.user_service import UserService
from core.sqlalchemy_models import UserConfigTopicAssociation

logger = logging.getLogger(__name__)
pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")


class PublicationService:
    def __init__(
        self, publication_repository: PublicationRepository, user_service: UserService, llm_interface: LLMInterface
    ):
        self.publication_repository = publication_repository
        self.user_service = user_service
        self.llm_interface = llm_interface

    def get_works_by_topics(
        self, topic_ids: [int], published_after: datetime.date, require_abstract=True, n_max: int = 2000
    ) -> list[Work]:
        topic_filter_str = "|".join(f"T{str(topic_id)}" for topic_id in topic_ids)
        # ISO 8601 date format
        from_publication_date_str = published_after.strftime("%Y-%m-%d")
        to_publication_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        no_limit = n_max == -1

        query = (
            pyalex.Works()
            .filter(primary_topic={"id": topic_filter_str})
            .filter(from_publication_date=from_publication_date_str)
            .filter(to_publication_date=to_publication_date_str)
        )
        if require_abstract:
            query = query.filter(has_abstract=True)

        logger.info(
            f"Querying OpenAlex for works with topics {topic_ids} published after {published_after} (n_max={n_max if not no_limit else "unlimited"}) Query URL: {query.url}"
        )

        works = []
        for pyalex_work in chain(*query.paginate(per_page=200, n_max=(n_max if not no_limit else None))):
            # don't return works without title
            works.append(Work(pyalex_work))
        logger.info(f"Found {len(works)} works.")

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
            else:
                raise ValueError(f"Work {work} has no abstract, but the abstract is required for scoring.")

        embeddings = self.llm_interface.create_embedding_batch(abstracts)

        scored_works: list[ScoredWork] = []
        for work, embedding in zip(works, embeddings):
            # cosine similarity
            score = sum([a * b for a, b in zip(embedding, area_of_interest_embedding)])
            scored_works.append(ScoredWork(work, score))

        return scored_works

    def summarize_works_for_user(self, user_name: str, works: list[Work]) -> list[SummarizedWork]:
        area_of_interest_description = self.user_service.area_of_interest_description(user_name)

        # only consider works with abstracts
        works_with_abstracts = [work for work in works if work.abstract]

        summarized_works: list[SummarizedWork] = []
        for work in works_with_abstracts:
            task = CustomizedSummaryTask(
                area_of_research=area_of_interest_description,
                abstract=work.abstract,
                prioritize_quality=True,
            )
            summary = self.llm_interface.handle_task(task)
            summarized_works.append(SummarizedWork(work, summary))

        return summarized_works

    # Fetches all potentially relevant works for a user published after a certain date, embeds the abstracts and stores them in the database
    # Does not yet score publications
    def initialize_for_user(self, user_name: str, start_date: datetime.datetime) -> list[Work]:
        topics: list[UserConfigTopicAssociation] = self.user_service.followed_topics(user_name)
        topic_ids = [topic.topic_id for topic in topics]

        works = self.get_works_by_topics(topic_ids, start_date, require_abstract=True, n_max=-1)
        access_timestamp = datetime.datetime.now()

        # skip works that have already been embedded
        known_works = self.publication_repository.get_all_openalex_ids()
        works_to_be_added = [work for work in works if work.id not in known_works]

        logger.info(
            f"Embedding {len(works_to_be_added)} works for user {user_name}. {len(works) - len(works_to_be_added)} works were already present."
        )
        # embed abstracts
        abstracts = []
        for work in works_to_be_added:
            if work.abstract:
                abstracts.append(work.abstract)
            else:
                raise ValueError(f"Work {work} has no abstract, but the abstract is required for embedding.")

        # TODO: Batch here (2000) to reduce memory consumption and to save intermediate results
        works_processed = 0
        for i in range(0, len(works_to_be_added), 2000):
            embeddings = self.llm_interface.create_embedding_batch(abstracts[i : i + 2000])
            for work, embedding in zip(works_to_be_added[i : i + 2000], embeddings):
                # TODO: Consistent naming? Work or Publication?
                self.publication_repository.create(
                    openalex_id=work.id,
                    title=work.title,
                    authors=work.authors,
                    published=work.publication_date,
                    accessed=access_timestamp,
                    embedding=embedding,
                )
            self.publication_repository.commit()
            works_processed += len(embeddings)
            logger.info(f"Progress: {works_processed} out of {len(works_to_be_added)} works embedded.")

        logger.info(f"Finished initialization for user {user_name}. Added {len(works_to_be_added)} works.")
        return works_to_be_added

    def get_most_relevant_works_by_embedding(
        self, user_name: str, n: int, start_date: datetime.datetime = None
    ) -> list[ScoredWork]:
        # get user embedding
        area_of_interest_description = self.user_service.area_of_interest_description(user_name)
        area_of_interest_embedding = self.llm_interface.create_embedding(area_of_interest_description)

        work_ids, similarities = self.publication_repository.get_openalex_ids_by_embedding_similarity(
            area_of_interest_embedding, n, start_date
        )
        works = self.get_works_by_openalex_ids(work_ids)
        scored_works = [ScoredWork(work, similarity) for work, similarity in zip(works, similarities)]

        return scored_works

    def get_work_by_openalex_id(self, openalex_id: str) -> Work:
        pyalex_work = pyalex.Works()[openalex_id]
        return Work(pyalex_work)

    def get_works_by_openalex_ids(self, openalex_ids: list[str] | list[int]) -> list[Work]:
        if not openalex_ids:
            return []
        # copy list to avoid modifying the original list
        ids = list(openalex_ids)
        # if the ids are integers, convert them to the OpenAlex format by prepending "W"
        if isinstance(ids[0], int):
            ids = [f"W{str(work_id)}" for work_id in openalex_ids]

        query = pyalex.Works().filter(openalex="|".join(ids))
        works = []
        for pyalex_work in chain(*query.paginate(per_page=200, n_max=None)):
            works.append(Work(pyalex_work))

        # because the API returns works in an arbitrary order, we need to restore the original order
        works = sorted(works, key=lambda work: openalex_ids.index(work.id))
        return works
