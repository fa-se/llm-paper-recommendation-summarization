import logging

from open_alex_interface import Work

logger = logging.getLogger(__name__)


# function that takes a list of Works and a list of topics. It should check the topic tags of each work, consider the score of each topic, and return the weighted average of the scores of the topics that match the work's tags.
def compute_relevance_scores_by_topics(works: list[Work], topics: list[int]) -> list[float]:
    relevance_scores = []

    for work in works:
        score = 0
        # TODO Maybe use set intersection?
        for topic_id in topics:
            if topic_id in work.topics:
                topic_score = work.topics[topic_id]['score']
                # there currently is a bug in the OpenAlex API, where the score is sometimes equal to the topic id. use -inf to ignore such cases
                if topic_score == topic_id:
                    topic_score = float('-inf')
                score += topic_score
        relevance_scores.append(score / len(topics))
        if score == float('-inf'):
            logger.info(f"Ignoring invalid topic score for work {work}")

    return relevance_scores
