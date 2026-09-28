CREATE EXTENSION IF NOT EXISTS vector;
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

CREATE TABLE openalex_domain (
	wikidata VARCHAR NOT NULL, 
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR NOT NULL, 
	wikipedia VARCHAR NOT NULL, 
	updated_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	embedding VECTOR(1024) NOT NULL, 
	PRIMARY KEY (id)
);

CREATE TABLE publication (
	id SERIAL NOT NULL, 
	openalex_id BIGINT NOT NULL, 
	title VARCHAR, 
	authors VARCHAR[], 
	publication_datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL, 
	accessed_datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL, 
	abstract VARCHAR, 
	bm25 bm25vector, 
	embedding VECTOR(1024) NOT NULL, 
	title_key VARCHAR(40), 
	abstract_key VARCHAR(40), 
	PRIMARY KEY (id), 
	UNIQUE (openalex_id)
);
CREATE INDEX ix_publication_abstract_key ON publication (abstract_key);
CREATE INDEX ix_publication_title_key ON publication (title_key);
CREATE INDEX publication_bm25 ON publication USING bm25 (bm25 bm25_ops);

CREATE TABLE openalex_field (
	wikidata VARCHAR NOT NULL, 
	domain_id INTEGER NOT NULL, 
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR NOT NULL, 
	wikipedia VARCHAR NOT NULL, 
	updated_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	embedding VECTOR(1024) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(domain_id) REFERENCES openalex_domain (id)
);

CREATE TABLE openalex_subfield (
	wikidata VARCHAR NOT NULL, 
	field_id INTEGER NOT NULL, 
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR NOT NULL, 
	wikipedia VARCHAR NOT NULL, 
	updated_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	embedding VECTOR(1024) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(field_id) REFERENCES openalex_field (id)
);

CREATE TABLE openalex_topic (
	keywords VARCHAR[] NOT NULL, 
	subfield_id INTEGER NOT NULL, 
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR NOT NULL, 
	wikipedia VARCHAR NOT NULL, 
	updated_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	embedding VECTOR(1024) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(subfield_id) REFERENCES openalex_subfield (id)
);
