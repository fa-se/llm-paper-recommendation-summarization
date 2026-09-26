"""One-off migration: adds the dedupe key columns and removes duplicates and junk abstracts from an existing DB.

New ingests skip such works themselves (see core/services/deduplication.py); this cleans up rows ingested before.
Of each group of duplicates, the row ingested first is kept. Safe to run repeatedly.
Usage (with a running db): uv run --env-file .env setup/dedupe_publications.py [--dry-run]
"""

import argparse

from sqlalchemy import text

from core.repositories.publication_repository import PublicationRepository
from core.services.deduplication import MIN_ABSTRACT_WORDS, DuplicateFilter, has_usable_abstract, text_key
from db import Session

MIGRATION = [
    "ALTER TABLE publication ADD COLUMN IF NOT EXISTS title_key VARCHAR(40)",
    "ALTER TABLE publication ADD COLUMN IF NOT EXISTS abstract_key VARCHAR(40)",
    "CREATE INDEX IF NOT EXISTS ix_publication_title_key ON publication (title_key)",
    "CREATE INDEX IF NOT EXISTS ix_publication_abstract_key ON publication (abstract_key)",
]


def main(dry_run: bool):
    session = Session()
    for statement in MIGRATION:
        session.execute(text(statement))

    rows = session.execute(
        text("SELECT id, title, abstract, title_key, abstract_key FROM publication ORDER BY id")
    ).all()
    missing_keys = [row for row in rows if row.title_key is None and row.abstract_key is None]
    for row in missing_keys:
        session.execute(
            text("UPDATE publication SET title_key = :title_key, abstract_key = :abstract_key WHERE id = :id"),
            {"id": row.id, "title_key": text_key(row.title), "abstract_key": text_key(row.abstract)},
        )

    junk = [row for row in rows if not has_usable_abstract(row.abstract)]
    duplicate_filter = DuplicateFilter()
    duplicates = [
        row
        for row in rows
        if has_usable_abstract(row.abstract) and not duplicate_filter.is_new(row.title, row.abstract)
    ]
    print(
        f"{len(rows)} publications, {len(missing_keys)} without keys. To delete: {len(junk)} with an abstract under "
        f"{MIN_ABSTRACT_WORDS} words, {len(duplicates)} duplicates. {len(rows) - len(junk) - len(duplicates)} remain."
    )
    for row in duplicates[:10]:
        print(f"  duplicate: {row.title[:100]}")

    if dry_run:
        session.rollback()
        print("Dry run, nothing changed.")
        return

    delete_ids = [row.id for row in junk + duplicates]
    session.execute(text("DELETE FROM publication WHERE id = ANY(:ids)"), {"ids": delete_ids})
    session.commit()
    # the BM25 statistics (document count, term frequencies) must reflect the deleted rows
    PublicationRepository(session).rebuild_bm25()
    print("Done; BM25 statistics rebuilt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
