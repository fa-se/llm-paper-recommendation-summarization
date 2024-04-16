import logging

from data_classes import ScoredWork
from llm_interfaces import LLMInterface
from open_alex_interface import Work

logger = logging.getLogger(__name__)


def compute_relevance_scores_by_topics(
    works: list[Work], topics: list[int], topics_user_relevances: list[float]
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
    llm_interface: LLMInterface, works: list[Work], user_embedding: list[float]
) -> list[ScoredWork]:
    # need to extract all abstracts first for batch embedding
    abstracts = []
    for work in works:
        if work.abstract:
            abstracts.append(work.abstract)
        else:
            raise ValueError("Work does not have an abstract.")

    embeddings = llm_interface.create_embedding_batch(abstracts)

    scored_works: list[ScoredWork] = []
    for work, embedding in zip(works, embeddings):
        # cosine similarity
        score = sum([a * b for a, b in zip(embedding, user_embedding)])
        scored_works.append(ScoredWork(work, score))

    return scored_works
