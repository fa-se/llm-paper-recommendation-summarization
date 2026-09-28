"""Writes the schema of the ORM models (core/sqlalchemy_models.py) to setup/ddl.sql. Needs no database.

Usage: uv run scripts/save_ddl.py
"""

from pathlib import Path

from sqlalchemy import create_mock_engine

from core.sqlalchemy_models import Base

DDL_PATH = Path(__file__).parents[1] / "setup" / "ddl.sql"

HEADER = """CREATE EXTENSION IF NOT EXISTS vector;
-- BM25: VectorChord-bm25 (index, scoring) and pg_tokenizer; both must be in shared_preload_libraries (setup/Dockerfile)
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
SET search_path TO public, bm25_catalog, tokenizer_catalog;
-- the tokenizer of abstracts and queries: BERT's lowercased WordPiece vocabulary, as with pg_bestmatch.rs before.
-- Tokenizers live in extension tables that pg_dump leaves out: create it again in a restored database.
SELECT create_tokenizer('bert', $$model = "bert_base_uncased"$$);
-- load it at server start, instead of the default llmlingua2 (200 MB, unused)
SELECT add_preload_model('bert_base_uncased');
SELECT remove_preload_model('llmlingua2');
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
