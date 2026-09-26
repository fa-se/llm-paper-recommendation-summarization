import logging
import time

import tiktoken
from openai import OpenAI

from core.config import Settings
from core.instrumentation import current_trace

from .base import LLMInterface, Message, Task

logger = logging.getLogger(__name__)

REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# $ per token; https://developers.openai.com/api/docs/pricing (standard tier, checked 2026-09-26)
PRICES = {
    "gpt-6-astra": {"input": 10.00 / 1e6, "output": 50.00 / 1e6},
    "gpt-6-sol": {"input": 2.00 / 1e6, "output": 10.00 / 1e6},
    "gpt-6-luna": {"input": 0.10 / 1e6, "output": 0.50 / 1e6},
    "gpt-4o-2024-05-13": {"input": 5.00 / 1e6, "output": 15.00 / 1e6},
    "gpt-4o-2024-08-06": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4o-mini-2024-07-18": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "gpt-3.5-turbo-0125": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "text-embedding-3-large": {"input": 0.13 / 1e6, "output": 0.0},
}


def completion_params(model: str, reasoning_effort: str | None, temperature: float | None = None) -> dict:
    """Returns the optional chat completion parameters that the given model accepts."""
    params = {}
    is_reasoning_model = model.startswith(REASONING_MODEL_PREFIXES)
    # older models (gpt-4o, gpt-4.1) reject reasoning_effort
    if is_reasoning_model and reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    # reasoning models only accept a custom temperature with reasoning_effort "none" (checked for gpt-6, 2026-09-26)
    if temperature is not None and (not is_reasoning_model or reasoning_effort == "none"):
        params["temperature"] = temperature
    return params


class OpenAIInterface(LLMInterface):
    # limits of the embeddings endpoint (https://platform.openai.com/docs/api-reference/embeddings/create)
    embedding_max_tokens_per_input = 8191
    embedding_max_inputs_per_request = 2048
    # A request may have 300,000 "tokens" over all inputs, but the endpoint estimates them as UTF-8 bytes / 4, not with
    # the tokenizer: for the demo abstracts that's 23% more than cl100k_base counts (checked 2026-09-26).
    embedding_max_request_size = 290_000

    def __init__(self, settings: Settings, client: OpenAI | None = None):
        # more retries than the SDK's default 2: concurrent reranking can briefly exceed the tokens-per-minute limit
        self.client = client or OpenAI(api_key=settings.openai_api_key, max_retries=6)
        self.models = settings.models
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions

    def handle_task(self, task: Task) -> str:
        model = self.models[task.model_tier]
        return self.create_completion(
            task.messages(),
            model=model.name,
            reasoning_effort=model.reasoning_effort,
            temperature=task.temperature,
            timeout=task.timeout,
            response_format=task.response_format(),
        )

    def create_completion(
        self,
        messages: list[Message],
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        params = completion_params(model, reasoning_effort, temperature)
        if timeout is not None:
            params["timeout"] = timeout
        if response_format is not None:
            params["response_format"] = response_format
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            messages=[{"role": message.role, "content": message.content} for message in messages],
            model=model,
            **params,
        )
        # completion_tokens includes reasoning tokens, which are billed as output
        self._track_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, started)

        message = response.choices[0].message
        if message.content is None:
            raise ValueError(f"{model} returned no content (refusal: {message.refusal!r})")
        return message.content.strip()

    def create_embedding(self, text: str) -> list[float]:
        return self.create_embedding_batch([text])[0]

    def create_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._embedding_batches(texts):
            started = time.perf_counter()
            response = self.client.embeddings.create(
                input=batch, model=self.embedding_model, dimensions=self.embedding_dimensions
            )
            embeddings.extend(embedding.embedding for embedding in response.data)
            self._track_usage(self.embedding_model, response.usage.total_tokens, 0, started)
        return embeddings

    def _embedding_batches(self, texts: list[str]) -> list[list[str]]:
        """Splits texts into as few requests as the limits allow; truncates texts that exceed the per-input limit."""
        encoding = tiktoken.get_encoding("cl100k_base")  # the tokenizer of the text-embedding-3 models
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_batch_size = 0
        for text in texts:
            tokens = encoding.encode(text)
            if len(tokens) > self.embedding_max_tokens_per_input:
                logger.warning(f"Truncating a text of {len(tokens)} tokens for embedding")
                text = encoding.decode(tokens[: self.embedding_max_tokens_per_input])
            size = -(-len(text.encode()) // 4)  # the endpoint's estimate, rounded up
            if current_batch and (
                current_batch_size + size > self.embedding_max_request_size
                or len(current_batch) == self.embedding_max_inputs_per_request
            ):
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            current_batch.append(text)
            current_batch_size += size
        if current_batch:
            batches.append(current_batch)
        return batches

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int, started: float):
        """Reports the tokens and cost of one API request to the current trace."""
        prices = PRICES.get(model)
        if prices is None:
            logger.warning(f"No price known for model {model}; its cost is not tracked")
            prices = {"input": 0.0, "output": 0.0}
        cost = input_tokens * prices["input"] + output_tokens * prices["output"]
        current_trace().record_llm_call(model, input_tokens, output_tokens, cost, time.perf_counter() - started)
