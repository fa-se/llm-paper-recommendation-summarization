FROM postgres:18.6-trixie

# pgvector (dense search) from the PGDG apt repo that the postgres image already has; VectorChord-bm25 (BM25 index) and
# pg_tokenizer (the tokenizer it is used with) from their release packages, for amd64 and arm64
ARG PGVECTOR=0.8.6
ARG VCHORD_BM25=0.3.0
ARG PG_TOKENIZER=0.1.1

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fsSL -o /tmp/vchord-bm25.deb \
        "https://github.com/supervc-stack/VectorChord-bm25/releases/download/${VCHORD_BM25}/postgresql-${PG_MAJOR}-vchord-bm25_${VCHORD_BM25}-1_${arch}.deb"; \
    curl -fsSL -o /tmp/pg-tokenizer.deb \
        "https://github.com/supervc-stack/pg_tokenizer.rs/releases/download/${PG_TOKENIZER}/postgresql-${PG_MAJOR}-pg-tokenizer_${PG_TOKENIZER}-1_${arch}.deb"; \
    apt-get install -y "postgresql-${PG_MAJOR}-pgvector=${PGVECTOR}-*" /tmp/vchord-bm25.deb /tmp/pg-tokenizer.deb; \
    apt-get purge -y --auto-remove curl ca-certificates; \
    rm -rf /tmp/*.deb /var/lib/apt/lists/*

# both BM25 extensions must be preloaded; pg_tokenizer refuses to work otherwise. The default max_wal_size (1 GB) is
# more than this database's size; Postgres keeps that much WAL around after a bulk write (e.g. an ingest)
CMD ["postgres", "-c", "shared_preload_libraries=vchord_bm25,pg_tokenizer", "-c", "max_wal_size=256MB"]
