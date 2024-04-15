from typing import Annotated, Optional

import typer
from sqlalchemy import select, desc
from sqlalchemy.exc import NoResultFound

from db import Session
from db.models import Topic
from db.models.user import User, UserConfig, UserTopicAssociation
from llm_interfaces import LLMInterface
from llm_interfaces.tasks import AlignToExamplesTask
from utils.decorator import handle_db_exceptions

# Define subcommands for the user command to manage users
user_commands = typer.Typer()


@user_commands.command()
@handle_db_exceptions
def create(user_name: str, display_name: str, email: Annotated[Optional[str], typer.Argument()] = None) -> int:
    with Session() as session:
        user = User(name=user_name, display_name=display_name, email=email)
        user.config = UserConfig()
        session.add(user)
        session.commit()
        typer.echo(f"Successfully created user {user_name} with id {user.id}")
    return user.id


@user_commands.command()
@handle_db_exceptions
def add_topic(user_name: str, topic_id: int, relevance_score: float) -> None:
    if not 0 <= relevance_score <= 1:
        typer.echo("Relevance score must be between 0 and 1.")
        raise typer.Exit(code=-1)

    with Session() as session:
        try:
            # Retrieve the user's config
            user_config: UserConfig = session.query(UserConfig).join(User).filter(User.name == user_name).one()
        except NoResultFound:
            typer.echo(f"User {user_name} does not exist.")
            raise typer.Exit(code=-1)

        # Check if the topic exists
        topic: Topic = session.get(Topic, topic_id)
        if topic is None:
            typer.echo(f"Topic with id {topic_id} does not exist.")
            raise typer.Exit(code=-1)

        # Check if the topic is already followed by the user
        association_check = session.query(UserTopicAssociation).filter_by(
            user_config_id=user_config.id,
            topic_id=topic_id
        ).one_or_none()

        if association_check is not None:
            typer.echo(f"User {user_name} already follows topic \"{topic.name}\" [{topic_id}].")
            raise typer.Exit(code=-1)

        # Create a new association
        new_association = UserTopicAssociation(
            user_config_id=user_config.id,
            topic_id=topic_id,
            relevance_score=relevance_score
        )
        session.add(new_association)
        session.commit()
        typer.echo(
            f"Successfully added topic \"{topic.name}\" [{topic_id}] with relevance score {relevance_score:.2f} to {user_name}'s followed topics.")


@user_commands.command()
@handle_db_exceptions
def remove_topic(user_name: str, topic_id: int) -> None:
    with Session() as session:
        try:
            # Retrieve the user and their config
            user_config: UserConfig = session.query(UserConfig).join(User).filter(User.name == user_name).one()
        except NoResultFound:
            typer.echo(f"User {user_name} does not exist.")
            raise typer.Exit(code=-1)

        # Retrieve the association instance between user_config and the topic
        association = session.query(UserTopicAssociation).filter_by(
            user_config_id=user_config.id,
            topic_id=topic_id
        ).one_or_none()

        if association is None:
            typer.echo(f"Topic with id {topic_id} is not followed by {user_name}.")
            raise typer.Exit(code=-1)

        # Remove the association
        session.delete(association)
        session.commit()
        typer.echo(f"Successfully removed topic {topic_id} from {user_name}'s followed topics.")


@user_commands.command()
@handle_db_exceptions
def followed_topics(user_name: str) -> None:
    with Session() as session:
        try:
            user_config: UserConfig = session.query(UserConfig).join(User).filter(User.name == user_name).one()
        except NoResultFound:
            typer.echo(f"User {user_name} does not exist.")
            raise typer.Exit(code=-1)

        followed_topics_associations = user_config.followed_topics
        if len(followed_topics_associations) == 0:
            typer.echo(f"User {user_name} does not follow any topics.")
        else:
            typer.echo(f"User {user_name} follows these topics:")
            for association in followed_topics_associations:
                # Display both topic details and relevance score
                typer.echo(
                    f"Id: {association.topic.id}, Name: {association.topic.name}, Relevance Score: {association.relevance_score:.2f}")


@user_commands.command()
@handle_db_exceptions
def set_area_of_interest(
        context: typer.Context, user_name: str,
        area_of_interest_description: Annotated[str, typer.Argument
            (..., help="A natural language description of the user's area of interest.")],
        dont_align: bool = typer.Option(False, "--no-align",
                                        help="Don't rewrite the description to improve topic matching.")) -> None:
    align = not dont_align  # Flip the value, because the feature should be enabled by default
    llm_interface = context.obj
    with Session() as session:
        try:
            user: User = session.query(User).filter(User.name == user_name).one()
        except NoResultFound:
            typer.echo(f"User {user_name} does not exist.")
            raise typer.Exit(code=-1)

        if align: typer.echo(f"Rewriting the area of interest description to improve similarity search...")
        topics, scores, aligned_description = get_topics_from_description(session, llm_interface,
                                                                          area_of_interest_description, align=align)
        if align: typer.echo(f"Aligned description:\n{aligned_description}")

        typer.echo(f"\nTop 5 matching topics:")
        for topic, score in zip(topics, scores):
            typer.echo(f"Id: {topic.id} | Name: {topic.name} | Score: {score:.2f}")
        typer.echo()
        user.config.area_of_interest_description = area_of_interest_description
        user.config.followed_topics = [
            UserTopicAssociation(topic=topic, relevance_score=score) for topic, score in zip(topics, scores)
        ]

        session.commit()
        typer.echo(f"Successfully added these topics to the list of topics followed by user {user_name}.")
        typer.echo(f"\nIf you want to add or remove topics, use the 'user add-topic' and 'user remove-topic' commands.")


def get_topics_from_description(session: Session, llm_interface: LLMInterface, description: str, align: bool = True) \
        -> tuple[list[Topic], list[float], str]:
    task = AlignToExamplesTask(description, [
        "This cluster of papers focuses on research related to ad hoc wireless networks, including topics such as routing protocols, mobile ad hoc networks, security, multi-hop wireless routing, and mobility models. It covers various aspects of network capacity, performance analysis, and optimization techniques for ad hoc wireless communication. Additionally, it explores challenges and solutions in areas like interference management, topology control, and channel assignment in wireless mesh networks.",
        "This cluster of papers focuses on the resilience of coral reef ecosystems to the impacts of climate change, including ocean acidification, bleaching, and disease. It explores the role of marine reserves, symbiotic dinoflagellates, and population connectivity in maintaining the health and biodiversity of coral reefs. The cluster also addresses the importance of the coral microbiome and the potential effects of nutrient pollution on coral reef ecosystems.",
        "This cluster of papers explores the impact of social media, particularly Facebook and online communication, on well-being, addictive behavior, and psychological effects, especially among adolescents. It delves into the concept of digital natives, examines the addictive nature of social media use, and investigates the relationship between social media use and various psychological outcomes."
    ])
    if align:
        description = llm_interface.handle_task(task)

    query_embedding = llm_interface.create_embedding(description)
    statement = select(Topic, (1 - Topic.embedding.cosine_distance(query_embedding)).label(
        'cosine_similarity')).order_by(desc('cosine_similarity')).limit(5)
    # get Topics and cosine similarities lists
    results = session.execute(statement).fetchall()
    topics = [result.Topic for result in results]
    cosine_similarities = [result.cosine_similarity for result in results]
    return topics, cosine_similarities, description
