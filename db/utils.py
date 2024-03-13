from sqlalchemy import MetaData

from db.database import Session
from db.models import *


def recreate_database():
    with Session() as session:
        Base.metadata.drop_all(session.get_bind())
        Base.metadata.create_all(session.get_bind())


def create_database():
    with Session() as session:
        Base.metadata.create_all(session.get_bind())


def recreate_tables(tables_or_models):
    with Session() as session:
        # Extract Table objects from models, if necessary
        tables = [item.__table__ if hasattr(item, '__table__') else item for item in tables_or_models]

        # Reflect the current database schema into a new MetaData object
        metadata = MetaData()
        metadata.reflect(bind=session.get_bind())

        # Drop only the specified tables, respecting dependencies
        metadata.drop_all(bind=session.get_bind(), tables=tables)

        # Recreate the tables using the Base metadata
        Base.metadata.create_all(bind=session.get_bind(), tables=tables)

        # Commit the transaction to ensure changes are applied
        session.commit()


def add_test_user():
    from db.models import User, UserConfig, Topic
    from sqlalchemy import func
    # create dummy user with associated config and topics
    with Session() as session:
        Base.metadata.create_all(session.get_bind())
        user = User(name="test", display_name="Test User", email="test@test.dev")
        user.config = UserConfig()
        # get random topics

        user.config.topics_of_interest = session.query(Topic).order_by(func.random()).limit(5).all()
        session.add(user)
        session.commit()
        print("User added")


if __name__ == "__main__":
    from db.models.user import userconfig_topic_of_interest_association

    recreate_tables([User, UserConfig, userconfig_topic_of_interest_association])
    add_test_user()

    # select user with name "test
    with Session() as session:
        user = session.query(User).filter_by(name="test").first()
        print(user)
        print(user.config.topics_of_interest)
