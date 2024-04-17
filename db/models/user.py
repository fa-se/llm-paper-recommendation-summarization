from typing import Optional

from sqlalchemy import Integer, String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class UserTopicAssociation(Base):
    __tablename__ = "userconfig_followed_topic_association"
    user_config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_config.id"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("openalex_topic.id"), primary_key=True
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Define relationships
    user_config: Mapped["UserConfig"] = relationship(back_populates="followed_topics")
    topic: Mapped["Topic"] = relationship()


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
        back_populates="user_config"
    )
    # Relationship to track publications
    scored_publications: Mapped[list["UserConfigPublicationAssociation"]] = (
        relationship(back_populates="user_config")
    )
