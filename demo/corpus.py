"""The demo corpus: a baseline (the pre-ingested papers of the demo queries), and a reset to it.

A live run with "fetch new papers first" adds papers to the corpus, and so changes the BM25 index's corpus statistics
(document count, average length, token frequencies), and with them the scores of all papers. Ingest only adds rows, so
deleting the papers added after the baseline and rebuilding the index restores the corpus exactly; a checksum over all
papers' BM25 token vectors confirms the rows.

    uv run --env-file .env python -m demo.corpus [status | reset | snapshot]

snapshot makes the current corpus the baseline (demo/corpus_baseline.json).
"""

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

import core

BASELINE = Path(__file__).parent / "corpus_baseline.json"


@contextmanager
def _session() -> Iterator:
    # the repositories share a thread-local session; close this thread's one afterwards, so that no transaction
    # stays open in a web server thread
    session = core.retrieval.publication_repository.session
    try:
        yield session
    finally:
        session.remove()


def _checksum(session) -> str | None:
    return session.execute(text("SELECT md5(string_agg(bm25::text, chr(10) ORDER BY id)) FROM publication")).scalar()


def load_baseline() -> dict | None:
    return json.loads(BASELINE.read_text()) if BASELINE.exists() else None


def status() -> dict:
    baseline = load_baseline()
    with _session() as session:
        rows = session.execute(text("SELECT count(*) FROM publication")).scalar()
        # how far back the pool reaches: the demo queries fetched the newest papers, so this is a matter of weeks
        oldest, newest = session.execute(
            text("SELECT min(publication_datetime_utc)::date, max(publication_datetime_utc)::date FROM publication")
        ).one()
        added = None
        if baseline:
            added = session.execute(
                text("SELECT count(*) FROM publication WHERE id > :max_id"), {"max_id": baseline["max_id"]}
            ).scalar()
    return {
        "rows": rows,
        "baseline_rows": baseline and baseline["rows"],
        "added": added,
        "oldest": oldest and oldest.isoformat(),
        "newest": newest and newest.isoformat(),
    }


def reset() -> dict:
    """Deletes the papers added after the baseline and rebuilds the BM25 index."""
    baseline = load_baseline()
    if baseline is None:
        raise RuntimeError(f"No baseline: create one with `python -m demo.corpus snapshot` ({BASELINE})")
    with _session() as session:
        deleted = session.execute(
            text("DELETE FROM publication WHERE id > :max_id"), {"max_id": baseline["max_id"]}
        ).rowcount
        session.commit()
        if deleted:
            core.retrieval.publication_repository.rebuild_bm25()
        rows = session.execute(text("SELECT count(*) FROM publication")).scalar()
        exact = rows == baseline["rows"] and _checksum(session) == baseline["bm25_md5"]
    return {"deleted": deleted, "rows": rows, "exact": exact}


def snapshot() -> dict:
    with _session() as session:
        rows, max_id = session.execute(text("SELECT count(*), coalesce(max(id), 0) FROM publication")).one()
        baseline = {
            "max_id": max_id,
            "rows": rows,
            "bm25_md5": _checksum(session),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "reset", "snapshot"], nargs="?", default="status")
    command = parser.parse_args().command
    print({"status": status, "reset": reset, "snapshot": snapshot}[command]())
