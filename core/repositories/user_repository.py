from sqlalchemy import func

from core.sqlalchemy_models import User, UserConfig, UserConfigTopicAssociation
from db import Session


# noinspection PyMethodMayBeStatic
class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def commit(self):
        self.session.commit()

    def create(self, name: str, display_name: str, email: str) -> User:
        user = User(name=name, display_name=display_name, email=email)
        user.configs.append(UserConfig(is_active=True))
        self.session.add(user)
        return user

    def delete(self, user: User) -> None:
        # delete user, associated configs, followed topics, and scored publications
        self.session.delete(user)

    def get_by_name(self, name: str) -> User | None:
        return self.session.query(User).filter(User.name == name).one_or_none()

    def get_by_id(self, user_id: int) -> User | None:
        return self.session.query(User).filter(User.id == user_id).one_or_none()

    def create_new_config_version(self, user: User) -> UserConfig:
        max_version = self.session.query(func.max(UserConfig.version)).filter(UserConfig.user_id == user.id).scalar()

        if max_version is not None:  # at least one config already exists
            # copy active config as a basis
            active_config = self.get_active_config(user)
            new_version = max_version + 1
            new_config = UserConfig(
                user=user,
                version=new_version,
                area_of_interest_description=active_config.area_of_interest_description,
                is_active=False,  # configs should be explicitly activated
            )

            self.session.add(new_config)
            self.session.flush()  # Flush here to ensure new_config.id is available
            associations: list[UserConfigTopicAssociation] = []
            for association in active_config.followed_topics:
                associations.append(
                    UserConfigTopicAssociation(
                        user_config_id=new_config.id,
                        topic_id=association.topic_id,
                        relevance_score=association.relevance_score,
                    )
                )
            new_config.followed_topics = associations
        else:  # create initial config
            new_config = UserConfig(user=user, version=1, is_active=False)
            self.session.add(new_config)

        return new_config

    def get_active_config(self, user: User) -> UserConfig:
        active_config = (
            self.session.query(UserConfig).filter(UserConfig.user_id == user.id, UserConfig.is_active).one_or_none()
        )
        return active_config

    def set_active_config(self, user_config: UserConfig) -> None:
        active_config = self.get_active_config(user_config.user)
        if active_config:
            active_config.is_active = False
        user_config.is_active = True

    def add_followed_topic_to_config(
        self, user_config: UserConfig, topic_id: int, relevance_score: float
    ) -> UserConfigTopicAssociation:
        new_association = UserConfigTopicAssociation(
            user_config_id=user_config.id,
            topic_id=topic_id,
            relevance_score=relevance_score,
        )
        user_config.followed_topics.append(new_association)
        return new_association

    def set_followed_topics_for_config(
        self, user_config: UserConfig, topic_ids: list[int], relevance_scores: list[float]
    ) -> None:
        # first, delete all existing associations
        user_config.followed_topics.clear()

        new_associations: list[UserConfigTopicAssociation] = []

        for topic_id, relevance_score in zip(topic_ids, relevance_scores):
            new_associations.append(
                UserConfigTopicAssociation(
                    user_config_id=user_config.id,
                    topic_id=topic_id,
                    relevance_score=relevance_score,
                )
            )
        user_config.followed_topics.extend(new_associations)

    def remove_followed_topic_from_config(self, user_config: UserConfig, topic_id: int) -> None:
        association = (
            self.session.query(UserConfigTopicAssociation)
            .filter_by(user_config_id=user_config.id, topic_id=topic_id)
            .one_or_none()
        )
        self.session.delete(association)
