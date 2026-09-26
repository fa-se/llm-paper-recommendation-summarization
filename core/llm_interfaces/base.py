from enum import Enum, auto
from typing import Self


class LLMType(Enum):
    GPT = auto()


class Message:
    roles = ["system", "user", "assistant"]

    def __init__(self, role: str, content: str):
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}")
        self.role = role
        self.content = content

    def format(self, **kwargs) -> Self:
        """
        This function returns a formatted copy of the Message instance.

        Parameters:
            **kwargs: The keyword arguments to pass to str.format.

        Returns:
            Message
        """
        return Message(self.role, self.content.format(**kwargs))


# TODO: maybe use Abstract Base Classes (ABC)
class Task:
    """
    Base class that encapsulates a task to be handled by an LLM interface.
    Inheriting classes can define custom logic to generate the corresponding prompt, e.g. to adapt the prompt to different LLMs.
    """

    # sampling temperature (None: the model's default) and request timeout in seconds (None: the client's default)
    temperature: float | None = None
    timeout: float | None = None

    def __init__(self, prioritize_quality: bool = False):
        """
        Parameters:
            prioritize_quality (bool): Indicates whether to prioritize quality over cost for this task, e.g. by using larger models.
        """
        self.prioritize_quality = prioritize_quality

    @property
    def model_tier(self) -> str:
        """Which of the LLM interface's configured models handles this task: "quality", "budget" or "rerank"."""
        return "quality" if self.prioritize_quality else "budget"

    def get_response_format(self, llm_type: LLMType) -> dict | None:
        """Structured output format for the response, or None for free text."""
        return None

    def get_prompt(self, llm_type: LLMType) -> [Message]:
        """
        Generate the prompt for the specified LLM type.

        Parameters:
            llm_type: The LLM type for which to get the prompt.

        Returns:
            [Message]: The messages representing the prompt for the specified LLM type.
        """
        raise NotImplementedError


class LLMInterface:
    def create_embedding(self, text: str, config: dict = None) -> list[float]:
        raise NotImplementedError

    def create_embedding_batch(
        self, texts: list[str], config: dict = None
    ) -> list[list[float]]:
        raise NotImplementedError

    def handle_task(self, task: Task) -> str:
        raise NotImplementedError
