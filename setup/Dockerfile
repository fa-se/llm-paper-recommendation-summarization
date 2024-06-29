FROM pgvector/pgvector:0.7.2-pg16

RUN apt-get update && apt-get install -y curl git build-essential clang postgresql-server-dev-16

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/tensorchord/pg_bestmatch.rs.git
WORKDIR pg_bestmatch.rs


RUN cargo install cargo-pgrx@=0.12.0-alpha.1  \
    && cargo pgrx init --pg16=$(which pg_config) \
    && cargo pgrx install --release

WORKDIR ..
RUN rm -rf pg_bestmatch.rs