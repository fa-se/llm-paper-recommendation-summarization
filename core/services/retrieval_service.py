"""Retrieval: build a corpus for a query (ingest), then find the most relevant works in it (search).

ingest: query --embedding--> OpenAlex topics --> newest works on those topics --> skip known works, duplicates and
        junk --> embed the abstracts, store them, update the BM25 index
search: query --> embedding similarity + BM25 over the whole corpus, blended (hybrid) --> top n * 10 candidates
        --> setwise LLM reranking picks the top n

The same free-text query is used at every stage, at increasing fidelity: topics (coarse), embeddings and BM25 (medium),
the LLM reading query and abstracts (high). Each stage is cheap enough for the number of works it sees.
"""

import functools
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum

from core.instrumentation import current_trace
from core.llm_interfaces import LLMInterface
from core.openalex import fetch_works_by_topics
from core.repositories.publication_repository import PublicationRepository, Ranking
from core.repositories.topic_repository import TopicRepository
from core.services.deduplication import DuplicateFilter, has_usable_abstract
from core.services.setwise_reranker import SetwiseHeapsortReranker
from core.sqlalchemy_models import Topic
from core.works import Work

logger = logging.getLogger(__name__)

# reranking can promote works that the first-stage search ranks lower, so it gets this many candidates per result
CANDIDATES_PER_RESULT = 10
# weights of embedding similarity and BM25 in hybrid search (0.8/0.2 from Mandikal & Mooney, not tuned)
HYBRID_WEIGHTS = (0.8, 0.2)
# works embedded and committed at a time during ingest, so that an interrupted ingest keeps what it has paid for
EMBEDDING_CHUNK_SIZE = 2000


class SearchType(Enum):
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


def min_max_normalize(ranking: Ranking) -> Ranking:
    """Scales the scores to [0, 1]; all 0 if they're all equal."""
    if not ranking:
        return {}
    low, high = min(ranking.values()), max(ranking.values())
    if low == high:
        return dict.fromkeys(ranking, 0.0)
    return {work_id: (score - low) / (high - low) for work_id, score in ranking.items()}


def weighted_sum(rankings: Sequence[Ranking], weights: Sequence[float]) -> Ranking:
    """Each work's weighted scores summed over the rankings, best first; a work missing from a ranking gets 0 there."""
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for work_id, score in ranking.items():
            scores[work_id] = scores.get(work_id, 0.0) + weight * score
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class RetrievalService:
    def __init__(
        self, publication_repository: PublicationRepository, topic_repository: TopicRepository, llm: LLMInterface
    ):
        self.publication_repository = publication_repository
        self.topic_repository = topic_repository
        self.llm = llm
        # topic matching and semantic search both need the query's embedding; embed it only once
        self._embed_query = functools.lru_cache(maxsize=32)(llm.create_embedding)

    def ingest(
        self, query: str, start_date: datetime, limit: int | None = None, num_topics: int = 5
    ) -> tuple[list[Topic], list[Work]]:
        """Adds the newest works (up to limit) on the query's num_topics topics, published since start_date.

        Returns the matched topics and the added works. search() ranks the whole corpus, not only the works fetched
        for this query, so works ingested for earlier queries can show up as well.
        """
        trace = current_trace()
        topics = self.match_topics(query, num_topics)

        with trace.stage("fetch", limit=limit):
            works = fetch_works_by_topics([topic.id for topic in topics], start_date, limit)
            accessed = datetime.now(UTC)

        with trace.stage("filter"):
            new_works = self._filter_new_works(works)

        with trace.stage("embed", works=len(new_works)):
            for start in range(0, len(new_works), EMBEDDING_CHUNK_SIZE):
                chunk = new_works[start : start + EMBEDDING_CHUNK_SIZE]
                embeddings = self.llm.create_embedding_batch([work.abstract for work in chunk])
                for work, embedding in zip(chunk, embeddings, strict=True):
                    self.publication_repository.add(work, embedding, accessed)
                self.publication_repository.commit()
                logger.info(f"Progress: {start + len(chunk)} out of {len(new_works)} works embedded.")

        if new_works:
            with trace.stage("index"):
                self.publication_repository.rebuild_bm25()
        logger.info(f"Finished ingest. Added {len(new_works)} works.")
        return topics, new_works

    def match_topics(self, query: str, n: int) -> list[Topic]:
        """The n OpenAlex topics whose embeddings are most similar to the query's."""
        trace = current_trace()
        with trace.stage("topics", n=n):
            matches = self.topic_repository.most_similar(self._embed_query(query), n)
            trace.emit(
                "topics",
                topics=[
                    {"id": topic.id, "name": topic.name, "similarity": round(similarity, 4)}
                    for topic, similarity in matches
                ],
            )
        return [topic for topic, _ in matches]

    def _filter_new_works(self, works: list[Work]) -> list[Work]:
        """Skips works that are already stored, have no usable abstract, or duplicate a stored or an earlier work."""
        known_ids = self.publication_repository.get_all_openalex_ids()
        new = [work for work in works if work.id not in known_ids]
        usable = [work for work in new if has_usable_abstract(work.abstract)]
        duplicate_filter = DuplicateFilter(*self.publication_repository.get_all_dedupe_keys())
        unique = [work for work in usable if duplicate_filter.is_new(work.title, work.abstract)]
        current_trace().emit(
            "filtered",
            fetched=len(works),
            already_present=len(works) - len(new),
            unusable=len(new) - len(usable),
            duplicates=len(usable) - len(unique),
            new=len(unique),
        )
        logger.info(
            f"Embedding {len(unique)} works. Skipped {len(works) - len(new)} already present, "
            f"{len(new) - len(usable)} without a usable abstract, {len(usable) - len(unique)} duplicates."
        )
        return unique

    def search(
        self,
        query: str,
        n: int,
        start_date: datetime,
        search_type: SearchType = SearchType.HYBRID,
        rerank: bool = True,
    ) -> list[Work]:
        """The n works of the corpus most relevant to the query, published since start_date.

        With rerank, the LLM picks the top n out of n * CANDIDATES_PER_RESULT candidates, which follow in their
        original order.
        """
        n_candidates = n * CANDIDATES_PER_RESULT if rerank else n
        logger.info(f"Getting top {n} publications using {search_type} search. Reranking enabled: {rerank}")
        match search_type:
            case SearchType.SEMANTIC:
                ranking = self._semantic_search(query, n_candidates, start_date)
            case SearchType.BM25:
                ranking = self._bm25_search(query, n_candidates, start_date)
            case SearchType.HYBRID:
                ranking = self._hybrid_search(query, n_candidates, start_date)
            case _:
                raise ValueError(f"Invalid search type {search_type}")

        with current_trace().stage("load"):
            # title, authors and abstract are stored at ingest, so no need to fetch the works from OpenAlex again
            publications = self.publication_repository.get_by_openalex_ids(list(ranking))
            works = [Work.from_publication(publication) for publication in publications]
            current_trace().emit("candidates", works=[work.to_dict() for work in works])

        if rerank:
            works = self.rerank(query, works, k=n)
        return works

    def rerank(self, query: str, works: list[Work], k: int) -> list[Work]:
        """The k works most relevant to the query according to the LLM, in order, followed by the others."""
        if not all(work.abstract for work in works):
            raise ValueError("All works must have abstracts for reranking.")
        logger.info(f"Reranking to identify top {k} among {len(works)} publications.")
        with current_trace().stage("rerank", k=k, candidates=len(works)):
            return SetwiseHeapsortReranker(self.llm).rerank(query, works, k=k)

    def _semantic_search(self, query: str, n: int, start_date: datetime) -> Ranking:
        with current_trace().stage("semantic", n=n):
            ranking = self.publication_repository.search_by_embedding(self._embed_query(query), n, start_date)
            self._emit_ranking("semantic", ranking, normalized=min_max_normalize(ranking))
        return ranking

    def _bm25_search(self, query: str, n: int, start_date: datetime) -> Ranking:
        with current_trace().stage("bm25", n=n):
            ranking = self.publication_repository.search_by_bm25(query, n, start_date)
            self._emit_ranking("bm25", ranking, normalized=min_max_normalize(ranking))
        return ranking

    def _hybrid_search(
        self, query: str, n: int, start_date: datetime, weights: tuple[float, float] = HYBRID_WEIGHTS
    ) -> Ranking:
        """Blends the min-max normalized scores of embedding and BM25 search, each over its own top 2n."""
        with current_trace().stage("hybrid", n=n, weights=list(weights)):
            semantic = min_max_normalize(self._semantic_search(query, n * 2, start_date))
            bm25 = min_max_normalize(self._bm25_search(query, n * 2, start_date))
            ranking = dict(list(weighted_sum([semantic, bm25], weights).items())[:n])
            # each method's weighted contribution to the hybrid score
            self._emit_ranking(
                "hybrid",
                ranking,
                semantic={work_id: weights[0] * score for work_id, score in semantic.items()},
                bm25={work_id: weights[1] * score for work_id, score in bm25.items()},
            )
        return ranking

    def _emit_ranking(self, method: str, ranking: Ranking, **per_work: Mapping[int, float]):
        """Emits a ranking with titles; per_work adds further values per work (0 if missing), e.g. normalized scores."""
        trace = current_trace()
        if not trace.enabled:
            return
        titles = self.publication_repository.get_titles(list(ranking))
        results = [
            {
                "id": work_id,
                "title": titles.get(work_id),
                "score": round(score, 4),
                **{key: round(values.get(work_id, 0.0), 4) for key, values in per_work.items()},
            }
            for work_id, score in ranking.items()
        ]
        trace.emit("ranking", method=method, results=results)
