from typing import Optional

from sqlalchemy import Integer, String, Float, ForeignKey, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class UserTopicAssociation(Base):
    __tablename__ = 'userconfig_followed_topic_association'
    user_config_id = Column(Integer, ForeignKey('user_config.id'), primary_key=True)
    topic_id = Column(Integer, ForeignKey('openalex_topic.id'), primary_key=True)
    relevance_score = Column(Float, nullable=False, default=0.0)

    # Define relationships
    user_config = relationship("UserConfig", back_populates="followed_topics")
    topic = relationship("Topic")


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    display_name: Mapped[str] = mapped_column(String)
    email: Mapped[Optional[str]] = mapped_column(String)
    # one-to-one relationship with UserConfig
    config: Mapped["UserConfig"] = relationship(back_populates="user")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id},name={self.name})"


class UserConfig(Base):
    __tablename__ = "user_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.id"))
    area_of_interest_description: Mapped[Optional[str]] = mapped_column(String)
    # each config belongs to a user
    user: Mapped["User"] = relationship(back_populates="config")
    # Many-to-Many relationship with Topic through association class
    followed_topics: Mapped[list["UserTopicAssociation"]] = relationship(
        "UserTopicAssociation", back_populates="user_config"
    )
