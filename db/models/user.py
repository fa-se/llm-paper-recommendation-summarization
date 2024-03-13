from sqlalchemy import Integer, String, ForeignKey, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# Association table for Config and Topic
userconfig_topic_of_interest_association = Table(
    'userconfig_topic_of_interest_association', Base.metadata,
    Column('user_config_id', Integer, ForeignKey('user_config.id'), primary_key=True),
    Column('topic_id', Integer, ForeignKey('openalex_topic.id'), primary_key=True)
)


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    display_name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    # one-to-one relationship with UserConfig
    config: Mapped["UserConfig"] = relationship(back_populates="user")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id},name={self.name})"


class UserConfig(Base):
    __tablename__ = "user_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.id"))

    # each config belongs to a user
    user: Mapped["User"] = relationship(back_populates="config")
    # Many-to-Many relationship with Topic through association table
    topics_of_interest: Mapped[list["Topic"]] = relationship(secondary=userconfig_topic_of_interest_association)
