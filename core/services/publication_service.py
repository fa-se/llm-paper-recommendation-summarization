import datetime
import functools
import logging
from enum import Enum
from itertools import chain
from os import environ

import pyalex

from core.dataclasses.data_classes import Work, ScoredWork
from core.instrumentation import current_trace
from core.llm_interfaces import LLMInterface
from core.repositories.publication_repository import PublicationRepository
from core.repositories.topic_repository import TopicRepository
from core.services.deduplication import DuplicateFilter, has_usable_abstract
from core.services.setwise_reranker import SetwiseHeapsortReranker
from core.sqlalchemy_models.openalex.topic import Topic

# from core.services.user_service import UserService

logger = logging.getLogger(__name__)
pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")


def _normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    # avoid division by zero
    if min_score == max_score:
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


class SearchType(Enum):
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


def get_work_by_openalex_id(openalex_id: str) -> Work:
    pyalex_work = pyalex.Works()[openalex_id]
    return Work(pyalex_work)


def get_works_by_openalex_ids(openalex_ids: list[str] | list[int]) -> list[Work]:
    if not openalex_ids:
        return []
    # copy list to avoid modifying the original list
    ids = list(openalex_ids)
    # if the ids are integers, convert them to the OpenAlex format by prepending "W"
    if isinstance(ids[0], int):
        ids = [f"W{str(work_id)}" for work_id in openalex_ids]

    works = []
    # OpenAlex API only allows 100 ids per query
    for i in range(0, len(ids), 100):
        query = pyalex.Works().filter(openalex="|".join(ids[i : i + 100]))
        for pyalex_work in chain(*query.paginate(per_page=200, n_max=None)):
            works.append(Work(pyalex_work))

    # because the API returns works in an arbitrary order, we need to restore the original order.
    # OpenAlex may return a merged work under a different ID; those go last
    order = {openalex_id: i for i, openalex_id in enumerate(openalex_ids)}
    works = sorted(works, key=lambda work: order.get(work.id, len(order)))
    return works


def get_works_by_topics(
    topic_ids: [int], published_after: datetime.date, require_abstract=True, n_max: int = 2000, most_recent_first=True
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

    if most_recent_first:
        query = query.sort(publication_date="desc")

    logger.info(
        f"Querying OpenAlex for works with topics {topic_ids} published after {published_after} (n_max={n_max if not no_limit else "unlimited"}) Query URL: {query.url}"
    )

    works = []
    work_ids = {}
    for pyalex_work in chain(*query.paginate(per_page=200, n_max=(n_max if not no_limit else None))):
        # OpenAlex sometimes returns the same work multiple times, so we need to deduplicate
        if pyalex_work["id"] not in work_ids:
            work_ids[pyalex_work["id"]] = True
            works.append(Work(pyalex_work))

    logger.info(f"Found {len(works)} works.")

    return works


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


class PublicationService:
    def __init__(
        self,
        publication_repository: PublicationRepository,
        topic_repository: TopicRepository,
        llm_interface: LLMInterface,
    ):
        self.publication_repository = publication_repository
        self.topic_repository = topic_repository
        # self.user_service = user_service
        self.llm_interface = llm_interface

    # Fetches all potentially relevant works for a user published after a certain date, embeds the abstracts and stores them in the database
    # Does not yet score publications
    def initialize_for_query(
        self, query: str, start_date: datetime.datetime, limit: int = -1, num_topics: int = 5
    ) -> tuple[list[Topic], list[Work]]:
        trace = current_trace()
        topics = self._get_matching_topics_for_query(query, num_topics)
        topic_ids = [topic.id for topic in topics]

        with trace.stage("fetch", limit=limit):
            works = get_works_by_topics(topic_ids, start_date, require_abstract=True, n_max=limit)
            access_timestamp = datetime.datetime.now()

        with trace.stage("filter"):
            # skip works that have already been embedded, duplicates of those, and works without a real abstract
            known_works = set(self.publication_repository.get_all_openalex_ids())
            new_works = [work for work in works if work.id not in known_works]
            usable_works = [work for work in new_works if has_usable_abstract(work.abstract)]
            duplicate_filter = DuplicateFilter(*self.publication_repository.get_all_dedupe_keys())
            works_to_be_added = [work for work in usable_works if duplicate_filter.is_new(work.title, work.abstract)]
            trace.emit(
                "filtered",
                fetched=len(works),
                already_present=len(works) - len(new_works),
                unusable=len(new_works) - len(usable_works),
                duplicates=len(usable_works) - len(works_to_be_added),
                new=len(works_to_be_added),
            )

        logger.info(
            f"Embedding {len(works_to_be_added)} works. Skipped {len(works) - len(new_works)} already present, "
            f"{len(new_works) - len(usable_works)} without a usable abstract, "
            f"{len(usable_works) - len(works_to_be_added)} duplicates."
        )
        abstracts = [work.abstract for work in works_to_be_added]

        with trace.stage("embed", works=len(works_to_be_added)):
            works_processed = 0
            for i in range(0, len(works_to_be_added), 2000):
                embeddings = self.llm_interface.create_embedding_batch(abstracts[i : i + 2000])
                for work, embedding in zip(works_to_be_added[i : i + 2000], embeddings):
                    # TODO: Consistent naming? Work or Publication?
                    self.publication_repository.create(
                        openalex_id=work.id,
                        title=work.title,
                        authors=work.authors,
                        abstract=work.abstract,
                        published=work.publication_date,
                        accessed=access_timestamp,
                        embedding=embedding,
                    )
                self.publication_repository.commit()
                works_processed += len(embeddings)
                logger.info(f"Progress: {works_processed} out of {len(works_to_be_added)} works embedded.")

        if works_to_be_added:
            with trace.stage("index"):
                self.publication_repository.rebuild_bm25()
        logger.info(f"Finished initialization. Added {len(works_to_be_added)} works.")
        return topics, works_to_be_added

    def get_relevant_works_for_query(
        self,
        query: str,
        n: int,
        start_date: datetime.datetime,
        search_type: SearchType = SearchType.HYBRID,
        rerank: bool = True,
    ) -> list[Work]:
        # if reranking is enabled, fetch more candidate publications so that reranking can push up
        # publications missed by bm25/embedding retrieval
        n_initial = n * 10 if rerank else n
        trace = current_trace()

        logger.info(f"Getting top {n} publications using {search_type} search. Reranking enabled: {rerank}")
        if search_type == SearchType.SEMANTIC:
            work_ids, scores = self._semantic_search(query, n_initial, start_date, normalize=True)
        elif search_type == SearchType.BM25:
            work_ids, scores = self._bm25_search(query, n_initial, start_date, normalize=True)
        elif search_type == SearchType.HYBRID:
            work_ids, scores = self._hybrid_search(query, n_initial, start_date, normalize=True)
        else:
            raise ValueError(f"Invalid search type {search_type}")

        with trace.stage("load"):
            # title, authors and abstract are stored at ingest, so no need to fetch the works from OpenAlex again
            works = [
                Work.from_publication(publication)
                for publication in self.publication_repository.get_by_openalex_ids(work_ids)
            ]
            trace.emit("candidates", works=[work.to_dict() for work in works])

        if rerank:
            logger.info(f"Reranking to identify top {n} among {len(work_ids)} publications.")
            with trace.stage("rerank", k=n, candidates=len(works)):
                works = self._rerank(query, works, k=n)

        return works

    @functools.lru_cache(maxsize=32)
    def _embed_query(self, query: str) -> list[float]:
        # both topic matching and semantic search need it
        return self.llm_interface.create_embedding(query)

    def _get_matching_topics_for_query(self, query: str, n_topics: int) -> list[Topic]:
        trace = current_trace()
        with trace.stage("topics", n=n_topics):
            query_embedding = self._embed_query(query)
            topics, similarities = self.topic_repository.get_topics_by_embedding_similarity(
                query_embedding, top_n=n_topics
            )
            trace.emit(
                "topics",
                topics=[
                    {"id": topic.id, "name": topic.name, "similarity": round(similarity, 4)}
                    for topic, similarity in zip(topics, similarities)
                ],
            )
        return topics

    def _emit_ranking(self, method: str, work_ids: list[int], scores: list[float], **per_work: list[float]):
        """Emits a ranking with titles; per_work adds further per-work values, e.g. normalized scores."""
        trace = current_trace()
        if not trace.enabled:
            return
        titles = self.publication_repository.get_titles_by_openalex_ids(work_ids)
        results = []
        for i, (work_id, score) in enumerate(zip(work_ids, scores)):
            result = {"id": work_id, "title": titles.get(work_id), "score": round(score, 4)}
            result.update({key: round(values[i], 4) for key, values in per_work.items()})
            results.append(result)
        trace.emit("ranking", method=method, results=results)

    def _semantic_search(
        self, query: str, n: int, start_date: datetime.datetime, normalize: bool = False
    ) -> tuple[list[int], list[float]]:
        with current_trace().stage("semantic", n=n):
            query_embedding = self._embed_query(query)
            work_ids, scores = self.publication_repository.get_openalex_ids_by_embedding_similarity(
                query_embedding, n, start_date
            )
            self._emit_ranking("semantic", work_ids, scores, normalized=_normalize_scores(scores) if scores else [])
        if normalize:
            scores = _normalize_scores(scores)
        return work_ids, scores

    def _bm25_search(
        self, query: str, n: int, start_date: datetime.datetime, normalize: bool = False
    ) -> tuple[list[int], list[float]]:
        with current_trace().stage("bm25", n=n):
            work_ids, scores = self.publication_repository.get_openalex_ids_by_bm25_similarity(query, n, start_date)
            self._emit_ranking("bm25", work_ids, scores, normalized=_normalize_scores(scores) if scores else [])
        if normalize:
            scores = _normalize_scores(scores)
        return work_ids, scores

    def _hybrid_search(
        self,
        query: str,
        n: int,
        start_date: datetime.datetime,
        normalize: bool = False,
        weights: tuple[float, float] = (0.8, 0.2),
    ) -> tuple[list[int], list[float]]:
        with current_trace().stage("hybrid", n=n, weights=list(weights)):
            work_ids_semantic, scores_semantic = self._semantic_search(query, n * 2, start_date, normalize)
            work_ids_bm25, scores_bm25 = self._bm25_search(query, n * 2, start_date, normalize)

            # scale by weight of retrieval method
            scores_semantic = [weights[0] * score for score in scores_semantic]
            scores_bm25 = [weights[1] * score for score in scores_bm25]
            # combine scores
            merged_works = {}  # deduplicate
            for work, score in zip(chain(work_ids_semantic, work_ids_bm25), chain(scores_semantic, scores_bm25)):
                if work not in merged_works:
                    merged_works[work] = score
                else:
                    merged_works[work] += score
            # get n highest scored works with their scores
            merged_works = sorted(merged_works.items(), key=lambda x: x[1], reverse=True)[:n]

            work_ids, scores = [work_id for work_id, _ in merged_works], [score for _, score in merged_works]
            # each method's (weighted) contribution to the hybrid score
            semantic, bm25 = dict(zip(work_ids_semantic, scores_semantic)), dict(zip(work_ids_bm25, scores_bm25))
            self._emit_ranking(
                "hybrid",
                work_ids,
                scores,
                semantic=[semantic.get(work_id, 0.0) for work_id in work_ids],
                bm25=[bm25.get(work_id, 0.0) for work_id in work_ids],
            )
        return work_ids, scores

    def _rerank(self, query: str, works: list[Work | ScoredWork], k: int = 10) -> list[Work]:
        if isinstance(works[0], ScoredWork):
            works = [scored_work.work for scored_work in works]

        # check if the works have abstracts
        if not all(work.abstract for work in works):
            raise ValueError("All works must have abstracts for reranking.")

        return SetwiseHeapsortReranker(self.llm_interface).rerank(query, works, k=k)
