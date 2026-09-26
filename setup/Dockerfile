FROM pgvector/pgvector:0.7.2-pg16

RUN apt-get update && apt-get install -y curl git build-essential clang postgresql-server-dev-16

# Rust toolchain from mid-2024, matching cargo-pgrx 0.12.0-alpha.1 (newer rustc/deps break the old pgrx build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.79.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Pinned to upstream 3126173 (merge of PR #15, which fixes bm25_refresh). Later commits need pgrx 0.12.7
# and change the tokenizer API.
ARG PG_BESTMATCH_COMMIT=312617392c8a32121907496f05c23fce1e3d056c
RUN git clone https://github.com/tensorchord/pg_bestmatch.rs.git \
    && git -C pg_bestmatch.rs checkout ${PG_BESTMATCH_COMMIT}
WORKDIR pg_bestmatch.rs

RUN cargo install --locked cargo-pgrx@=0.12.0-alpha.1 \
    && cargo pgrx init --pg16=$(which pg_config) \
    && cargo pgrx install --release

WORKDIR ..
RUN rm -rf pg_bestmatch.rs
