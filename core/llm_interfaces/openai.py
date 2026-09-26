import logging
import threading
import time
from os import environ

import tiktoken
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from core.instrumentation import current_trace

from .base import LLMInterface, LLMType, Message, Task

logger = logging.getLogger(__name__)

REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


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
    # Models can be overridden via env vars, e.g. to run the thesis' original 2024 models
    # (OPENAI_QUALITY_MODEL=gpt-4o-2024-05-13, OPENAI_RERANK_MODEL=gpt-4o-mini-2024-07-18).
    # https://developers.openai.com/api/docs/models
    defaults = {
        "quality_model": environ.get("OPENAI_QUALITY_MODEL", "gpt-6-sol"),
        "quality_reasoning_effort": environ.get("OPENAI_QUALITY_REASONING_EFFORT", "low"),
        "budget_model": environ.get("OPENAI_BUDGET_MODEL", "gpt-6-luna"),
        "budget_reasoning_effort": environ.get("OPENAI_BUDGET_REASONING_EFFORT", "none"),
        # used for the comparisons of setwise reranking (core/services/setwise_reranker.py)
        "rerank_model": environ.get("OPENAI_RERANK_MODEL", "gpt-6-luna"),
        "rerank_reasoning_effort": environ.get("OPENAI_RERANK_REASONING_EFFORT", "none"),
        # keep: the topic embeddings in setup/openalex_embeddings.sql were created with this model and dimension
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024,
    }

    model_to_cost_per_token = {
        # https://developers.openai.com/api/docs/pricing (standard tier, checked 2026-09-26)
        "gpt-6-astra": {"input": 10.00 / 1e6, "output": 50.00 / 1e6},
        "gpt-6-sol": {"input": 2.00 / 1e6, "output": 10.00 / 1e6},
        "gpt-6-luna": {"input": 0.10 / 1e6, "output": 0.50 / 1e6},
        "gpt-4o-2024-05-13": {"input": 5.00 / 1e6, "output": 15.00 / 1e6},
        "gpt-4o-2024-08-06": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
        "gpt-4o-mini-2024-07-18": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
        "gpt-3.5-turbo-0125": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
        "text-embedding-3-large": 0.13 / 1e6,
    }

    def __init__(self, print_usage_info: bool = False):
        # more retries than the SDK's default 2: concurrent reranking can briefly exceed the tokens-per-minute limit
        self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"), max_retries=6)
        # self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"), base_url="http://host.docker.internal:10080/v1")
        self.accumulated_costs = 0.0
        self._costs_lock = threading.Lock()  # reranking and summarization call the API from several threads
        self.print_usage_info = print_usage_info

    def handle_task(self, task: Task) -> str:
        messages = task.get_prompt(LLMType.GPT)
        tier = task.model_tier
        completion = self.create_completion(
            messages=messages,
            model=self.defaults[f"{tier}_model"],
            reasoning_effort=self.defaults[f"{tier}_reasoning_effort"],
            temperature=task.temperature,
            timeout=task.timeout,
            response_format=task.get_response_format(LLMType.GPT),
        )

        return completion

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int, started: float) -> float:
        """Adds the cost of one API request to the accumulated costs and the current trace; returns the cost."""
        prices = self.model_to_cost_per_token.get(model)
        if prices is None:
            logger.warning(f"No price known for model {model}; its cost is not tracked")
            prices = {"input": 0.0, "output": 0.0}
        elif not isinstance(prices, dict):  # embedding models have a single price
            prices = {"input": prices, "output": 0.0}
        cost = (input_tokens * prices["input"]) + (output_tokens * prices["output"])
        with self._costs_lock:
            self.accumulated_costs += cost
        current_trace().record_llm_call(model, input_tokens, output_tokens, cost, time.perf_counter() - started)
        return cost

    def create_embedding(self, text: str, config: dict = None) -> list[float]:
        if config is None:
            config = {}
        # merge provided config with defaults
        config = {**self.defaults, **config}

        started = time.perf_counter()
        response = self.client.embeddings.create(
            input=[text],
            model=config["embedding_model"],
            dimensions=config["embedding_dimensions"],
        )

        used_tokens = response.usage.total_tokens
        cost = self._track_usage(config["embedding_model"], used_tokens, 0, started)
        if self.print_usage_info:
            print(
                f"Model: {config['embedding_model']}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        return response.data[0].embedding

    def create_embedding_batch(self, texts: list[str], config: dict = None) -> list[list[float]]:
        if config is None:
            config = {}
        # merge provided config with defaults
        config = {**self.defaults, **config}

        batches = self._embedding_batches(texts)

        embeddings: list[list[float]] = []
        used_tokens = 0
        cost = 0.0
        for batch in batches:
            started = time.perf_counter()
            response = self.client.embeddings.create(
                input=batch,
                model=config["embedding_model"],
                dimensions=config["embedding_dimensions"],
            )
            embeddings.extend([embedding.embedding for embedding in response.data])
            used_tokens += response.usage.total_tokens
            cost += self._track_usage(config["embedding_model"], response.usage.total_tokens, 0, started)
        if self.print_usage_info:
            print(
                f"Model: {config['embedding_model']}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        return embeddings

    # limits of the embeddings endpoint (https://platform.openai.com/docs/api-reference/embeddings/create)
    embedding_max_tokens_per_input = 8191
    embedding_max_inputs_per_request = 2048
    # A request may have 300,000 "tokens" over all inputs, but the endpoint estimates them as UTF-8 bytes / 4, not with
    # the tokenizer: for the demo abstracts that's 23% more than cl100k_base counts (checked 2026-09-26).
    embedding_max_request_size = 290_000

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

    def create_completion(
        self,
        messages: list[Message],
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        completion_messages: [ChatCompletionMessageParam] = []
        for message in messages:
            # Due to ChatCompletionMessageParam being a union, we need to check the role and instantiate the correct type
            if message.role == "system":
                message_param = ChatCompletionSystemMessageParam(role=message.role, content=message.content)
            elif message.role == "user":
                message_param = ChatCompletionUserMessageParam(role=message.role, content=message.content)
            else:
                raise ValueError(f"Unsupported message role: {message.role}")
            completion_messages.append(message_param)

        optional_params = completion_params(model, reasoning_effort, temperature)
        if timeout is not None:
            optional_params["timeout"] = timeout
        if response_format is not None:
            optional_params["response_format"] = response_format
        started = time.perf_counter()
        response = self.client.chat.completions.create(messages=completion_messages, model=model, **optional_params)

        input_tokens = response.usage.prompt_tokens
        # includes reasoning tokens, which are billed as output
        output_tokens = response.usage.completion_tokens
        cost = self._track_usage(model, input_tokens, output_tokens, started)
        if self.print_usage_info:
            print(
                f"Model: {model}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens},\
                Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        message = response.choices[0].message
        if message.content is None:
            raise ValueError(f"{model} returned no content (refusal: {message.refusal!r})")
        return message.content.strip()
