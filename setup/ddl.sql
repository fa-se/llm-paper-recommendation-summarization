CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_bestmatch;
SET search_path TO public, bm_catalog;

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
	bm25 SPARSEVEC, 
	embedding VECTOR(1024) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (openalex_id)
);


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

