from database import Session

from models.base import Base

def recreate_database():
    with Session() as session:
        Base.metadata.drop_all(session.get_bind())
        Base.metadata.create_all(session.get_bind())
