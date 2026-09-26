"""OpenAlex API access via pyalex (https://docs.openalex.org).

Needs no API key; create_services sets a contact email for faster responses (the "polite pool").
"""

import logging
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
    for page in query.paginate(per_page=200, n_max=limit):
        for work in page:
            # OpenAlex sometimes returns the same work more than once
            if work["id"] not in seen_ids:
                seen_ids.add(work["id"])
                works.append(Work.from_openalex(work))

    logger.info(f"Found {len(works)} works.")
    return works
