"""Writes the schema of the ORM models (core/sqlalchemy_models.py) to setup/ddl.sql. Needs no database.

Usage: uv run scripts/save_ddl.py
"""

from pathlib import Path

from sqlalchemy import create_mock_engine

from core.sqlalchemy_models import Base

DDL_PATH = Path(__file__).parents[1] / "setup" / "ddl.sql"

HEADER = """CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_bestmatch;
SET search_path TO public, bm_catalog;
"""


def main():
    statements: list[str] = []

    def record(sql, *multiparams, **params):
        statements.append(str(sql.compile(dialect=engine.dialect)).strip() + ";")

    engine = create_mock_engine("postgresql+psycopg://", record)
    blocks = []
    for table in Base.metadata.sorted_tables:
        statements.clear()
        table.create(engine, checkfirst=False)
        create_table, *create_indexes = statements
        # a table's indexes come in arbitrary order; sorted, the file only changes when the schema does
        blocks.append("\n".join([create_table, *sorted(create_indexes)]))
    DDL_PATH.write_text(HEADER + "\n" + "\n\n".join(blocks) + "\n")
    print(f"Wrote the schema of {len(blocks)} tables to {DDL_PATH}")


if __name__ == "__main__":
    main()
