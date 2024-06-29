import logging
import os
from os import environ

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg://{environ.get('DB_USER')}:{environ.get('DB_PASSWORD')}@{environ.get('DB_HOST')}/{environ.get('DB_NAME')}"

engine = create_engine(f"{SQLALCHEMY_DATABASE_URL}?options=-csearch_path=public,bm_catalog")
if os.getenv("DEBUG") == "1":
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
Session = sessionmaker(autocommit=False, autoflush=True, bind=engine)
