from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import AlignToExamplesTask
from core.repositories import UserRepository, TopicRepository
from core.sqlalchemy_models import Topic, UserConfigTopicAssociation


class UserService:
    def __init__(self, user_repository: UserRepository, topic_repository: TopicRepository, llm_interface: LLMInterface):
        self.user_repository = user_repository
        self.topic_repository = topic_repository
        self.llm_interface = llm_interface

    def create_user(self, user_name: str, display_name: str, email: str):
        user_exists = self.user_repository.get_by_name(user_name)
        if user_exists:
            raise ValueError(f"User {user_name} already exists.")

        user = self.user_repository.create(user_name, display_name, email)

        self.user_repository.commit()
        return user

    def delete_user(self, user_name: str) -> None:
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")

        self.user_repository.delete(user)

        self.user_repository.commit()

    def set_followed_topics(self, user_name: str, topic_ids: list[int], relevance_scores: list[float]) -> None:
        # create a new version of the user's config
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")

        new_config = self.user_repository.create_new_config_version(user)
        self.user_repository.set_followed_topics_for_config(new_config, topic_ids, relevance_scores)
        self.user_repository.set_active_config(new_config)

        self.user_repository.commit()

    def remove_followed_topic(self, user_name: str, topic_id: int) -> None:
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")
        user_config = self.user_repository.create_new_config_version(user)
        self.user_repository.remove_followed_topic_from_config(user_config, topic_id)
        self.user_repository.set_active_config(user_config)

        self.user_repository.commit()

    def followed_topics(self, user_name: str) -> list[UserConfigTopicAssociation]:
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")
        user_config = self.user_repository.get_active_config(user)
        return user_config.followed_topics

    def set_area_of_interest(
        self, user_name: str, area_of_interest_description: str, align_description: bool = True, n_topics: int = 5
    ) -> tuple[list[UserConfigTopicAssociation], str]:
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")

        topics, similarities, area_of_interest_description = self._get_matching_topics_for_area_of_interest(
            area_of_interest_description, align_description, n_topics
        )

        user_config = self.user_repository.create_new_config_version(user)
        user_config.area_of_interest_description = area_of_interest_description
        self.user_repository.set_followed_topics_for_config(user_config, [topic.id for topic in topics], similarities)
        # commit new config before setting it as active
        self.user_repository.commit()
        self.user_repository.set_active_config(user_config)
        self.user_repository.commit()

        return user_config.followed_topics, area_of_interest_description

    def _get_matching_topics_for_area_of_interest(
        self, area_of_interest_description: str, align_description: bool, n_topics: int
    ) -> tuple[list[Topic], list[float], str]:
        if align_description:
            task = AlignToExamplesTask(
                area_of_interest_description,
                [
                    "This cluster of papers focuses on research related to ad hoc wireless networks, including topics such as routing protocols, mobile ad hoc networks, security, multi-hop wireless routing, and mobility models. It covers various aspects of network capacity, performance analysis, and optimization techniques for ad hoc wireless communication. Additionally, it explores challenges and solutions in areas like interference management, topology control, and channel assignment in wireless mesh networks.",
                    "This cluster of papers focuses on the resilience of coral reef ecosystems to the impacts of climate change, including ocean acidification, bleaching, and disease. It explores the role of marine reserves, symbiotic dinoflagellates, and population connectivity in maintaining the health and biodiversity of coral reefs. The cluster also addresses the importance of the coral microbiome and the potential effects of nutrient pollution on coral reef ecosystems.",
                    "This cluster of papers explores the impact of social media, particularly Facebook and online communication, on well-being, addictive behavior, and psychological effects, especially among adolescents. It delves into the concept of digital natives, examines the addictive nature of social media use, and investigates the relationship between social media use and various psychological outcomes.",
                ],
            )
            area_of_interest_description = self.llm_interface.handle_task(task)

        query_embedding = self.llm_interface.create_embedding(area_of_interest_description)
        topics, similarities = self.topic_repository.get_topics_by_embedding_similarity(query_embedding, top_n=n_topics)
        return topics, similarities, area_of_interest_description

    def area_of_interest_description(self, user_name: str) -> str:
        user = self.user_repository.get_by_name(user_name)
        if not user:
            raise ValueError(f"User {user_name} does not exist.")
        user_config = self.user_repository.get_active_config(user)
        return user_config.area_of_interest_description
