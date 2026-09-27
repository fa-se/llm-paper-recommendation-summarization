"""OpenAlex API access via pyalex (https://docs.openalex.org).

Needs no API key; create_services sets a contact email for faster responses (the "polite pool").
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pyalex

from core.works import Work

logger = logging.getLogger(__name__)


def fetch_works_by_topics(topic_ids: list[int], start_date: datetime, limit: int | None = None) -> list[Work]:
    """The newest works (up to limit; None: all) with an abstract whose primary topic is one of topic_ids, published
    between start_date and today."""
    query = (
        pyalex.Works()
        .filter(primary_topic={"id": "|".join(f"T{topic_id}" for topic_id in topic_ids)})
        .filter(from_publication_date=start_date.strftime("%Y-%m-%d"))
        # OpenAlex also has works with a publication date in the future
        .filter(to_publication_date=datetime.now().strftime("%Y-%m-%d"))
        .filter(has_abstract=True)
        .sort(publication_date="desc")
    )
    logger.info(
        f"Querying OpenAlex for works with topics {topic_ids} published after {start_date} "
        f"(limit={limit if limit is not None else 'unlimited'}) Query URL: {query.url}"
    )

    works: list[Work] = []
    seen_ids: set[str] = set()
    # pyalex yields whole pages until it has at least n_max works, so the list is cut to the limit below
    for page in query.paginate(per_page=min(200, limit or 200), n_max=limit):
        for work in page:
            # OpenAlex sometimes returns the same work more than once
            if work["id"] not in seen_ids:
                seen_ids.add(work["id"])
                works.append(Work.from_openalex(work))

    works = works[:limit] if limit is not None else works
    logger.info(f"Found {len(works)} works.")
    return works


def fetch_works_per_topic(
    topic_ids: list[int], start_date: datetime, per_topic: int, max_workers: int = 5
) -> dict[int, list[Work]]:
    """Like fetch_works_by_topics, but the newest per_topic works of each topic, so that big topics don't crowd out
    small ones. The topics are fetched concurrently; the result maps each topic id to its works."""
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(topic_ids)))) as pool:
        results = pool.map(lambda topic_id: fetch_works_by_topics([topic_id], start_date, per_topic), topic_ids)
        return dict(zip(topic_ids, results, strict=True))
