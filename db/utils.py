from sqlalchemy import MetaData

import core.sqlalchemy_models as models
from db.database import Session


def recreate_database():
    with Session() as session:
        models.Base.metadata.drop_all(session.get_bind())
        models.Base.metadata.create_all(session.get_bind())


def create_database():
    with Session() as session:
        models.Base.metadata.create_all(session.get_bind())


def save_ddl():
    import os
    from sqlalchemy import create_mock_engine

    if os.path.exists("ddl.sql"):
        os.remove("ddl.sql")

    def dump(sql, *multiparams, **params):
        ddl = sql.compile(dialect=engine.dialect)
        # write to file
        with open("ddl.sql", "a") as f:
            f.write(str(ddl) + ";")

    engine = create_mock_engine("postgresql+psycopg://", dump)
    models.Base.metadata.create_all(engine, checkfirst=False)


def recreate_tables(tables_or_models):
    with Session() as session:
        # Extract Table objects from models, if necessary
        tables = [item.__table__ if hasattr(item, "__table__") else item for item in tables_or_models]
        print(f"Recreating tables: {tables}")
        # Reflect the current database schema into a new MetaData object
        metadata = MetaData()
        metadata.reflect(bind=session.get_bind())
        print(f"Dropping tables: {tables}")
        # Drop only the specified tables, respecting dependencies
        metadata.drop_all(bind=session.get_bind(), tables=tables)
        print(f"Creating tables: {tables}")
        # Recreate the tables using the Base metadata
        models.Base.metadata.create_all(bind=session.get_bind(), tables=tables)
        print(f"Tables created: {tables}")
        # Commit the transaction to ensure changes are applied
        session.commit()


def add_test_user():
    from sqlalchemy import func

    # create dummy user with associated config and topics
    with Session() as session:
        models.Base.metadata.create_all(session.get_bind())
        user = models.User(name="test", display_name="Test User", email="test@test.dev")
        user.config = models.UserConfig()
        # get random topics

        user.config.followed_topics = session.query(models.Topic).order_by(func.random()).limit(5).all()
        session.add(user)
        session.commit()
        print("User added")


if __name__ == "__main__":
    save_ddl()
