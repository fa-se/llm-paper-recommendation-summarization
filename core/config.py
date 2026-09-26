"""Configuration from environment variables (see .env.example)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from sqlalchemy import URL


@dataclass(frozen=True)
class ModelConfig:
    name: str
    # sent only to reasoning models (see core/llm_interfaces/openai.py); None: the model's default
    reasoning_effort: str | None


@dataclass(frozen=True)
class Settings:
    database_url: URL
    openai_api_key: str | None
    # OpenAlex answers faster for requests with a contact email ("polite pool"); it needs no API key
    openalex_contact_email: str | None
    # which model handles which kind of task (Task.model_tier): tailored summaries, reranking comparisons
    models: Mapping[str, ModelConfig]
    # keep: the topic embeddings in setup/openalex_embeddings.sql were created with this model and dimension
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    # logs all SQL statements
    debug: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Self:
        """Models can be overridden, e.g. to run the thesis' original 2024 models:
        OPENAI_QUALITY_MODEL=gpt-4o-2024-05-13 OPENAI_RERANK_MODEL=gpt-4o-mini-2024-07-18.
        """

        def model(tier: str, default_name: str, default_effort: str) -> ModelConfig:
            prefix = f"OPENAI_{tier.upper()}"
            effort = env.get(f"{prefix}_REASONING_EFFORT", default_effort)
            return ModelConfig(env.get(f"{prefix}_MODEL", default_name), effort or None)

        return cls(
            database_url=URL.create(
                "postgresql+psycopg",
                username=env.get("DB_USER"),
                password=env.get("DB_PASSWORD"),
                host=env.get("DB_HOST"),
                database=env.get("DB_NAME"),
                # pg_bestmatch keeps its functions and BM25 statistics in the bm_catalog schema
                query={"options": "-csearch_path=public,bm_catalog"},
            ),
            openai_api_key=env.get("OPENAI_API_KEY"),
            openalex_contact_email=env.get("OPENALEX_CONTACT_EMAIL"),
            # https://developers.openai.com/api/docs/models
            models={
                "quality": model("quality", "gpt-6-sol", "low"),
                # GPT-6 accepts temperature 0 only with reasoning effort "none"
                "rerank": model("rerank", "gpt-6-luna", "none"),
            },
            debug=env.get("DEBUG") == "1",
        )
