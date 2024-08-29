from core.llm_interfaces import LLMInterface, OpenAIInterface
from core.repositories import TopicRepository
from core.repositories.publication_repository import PublicationRepository
from core.services.publication_service import PublicationService
from core.services.summarization_service import SummarizationService
from db import Session

session = Session()
llm_interface: LLMInterface = OpenAIInterface()
publication_repository = PublicationRepository(session)
topic_repository = TopicRepository(session)
retrieval = PublicationService(publication_repository, topic_repository, llm_interface)
summarization = SummarizationService(llm_interface)

__all__ = ["retrieval", "summarization"]
