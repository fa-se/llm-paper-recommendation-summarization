from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class Task(ABC):
    """A prompt plus what the LLM interface needs to send it: which model, sampling settings, structured output."""

    # which of the configured models handles this task (Settings.models): "quality" or "rerank"
    model_tier: ClassVar[str]
    # sampling temperature (None: the model's default) and request timeout in seconds (None: the client's default)
    temperature: ClassVar[float | None] = None
    timeout: ClassVar[float | None] = None

    @abstractmethod
    def messages(self) -> list[Message]: ...

    def response_format(self) -> dict | None:
        """Structured output format for the response, or None for free text."""
        return None


class LLMInterface(Protocol):
    def create_embedding(self, text: str) -> list[float]: ...

    def create_embedding_batch(self, texts: list[str]) -> list[list[float]]: ...

    def handle_task(self, task: Task) -> str:
        """Sends the task's prompt; returns the response text."""
        ...
