import logging
from os import environ

import tiktoken
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

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
        # used by PublicationService._rerank (setwise reranking via llm-rankers)
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
        self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))
        # self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"), base_url="http://host.docker.internal:10080/v1")
        self.accumulated_costs = 0.0
        self.print_usage_info = print_usage_info

    def handle_task(self, task: Task) -> str:
        messages = task.get_prompt(LLMType.GPT)
        tier = "quality" if task.prioritize_quality else "budget"
        completion = self.create_completion(
            messages=messages,
            model=self.defaults[f"{tier}_model"],
            reasoning_effort=self.defaults[f"{tier}_reasoning_effort"],
        )

        return completion

    def create_embedding(self, text: str, config: dict = None) -> list[float]:
        if config is None:
            config = {}
        # merge provided config with defaults
        config = {**self.defaults, **config}

        response = self.client.embeddings.create(
            input=[text],
            model=config["embedding_model"],
            dimensions=config["embedding_dimensions"],
        )

        used_tokens = response.usage.total_tokens
        cost = used_tokens * self.model_to_cost_per_token[config["embedding_model"]]
        self.accumulated_costs += cost
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

        # the embedding api currently accepts a maximum of 8191 tokens per call, so we need to batch the input
        max_tokens_per_batch = 8191
        batches: list[list[str]] = []
        current_batch: list[str] = []
        total_tokens = 0
        current_batch_tokens = 0
        tiktoken_encoding = (
            "cl100k_base"
            if config["embedding_model"] in ["text-embedding-3-large", "text-embedding-3-small"]
            else "cl100k_base"
        )
        for text in texts:
            num_tokens = num_tokens_from_string(text, tiktoken_encoding)
            if num_tokens > max_tokens_per_batch:
                # TODO: How to handle this case? For now, add ' ' to batch, '' fails
                current_batch.append(" ")
                continue

            if current_batch_tokens + num_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
            current_batch.append(text)
            current_batch_tokens += num_tokens
            total_tokens += num_tokens

        if current_batch:
            batches.append(current_batch)

        embeddings: list[list[float]] = []
        used_tokens = 0
        for batch in batches:
            response = self.client.embeddings.create(
                input=batch,
                model=config["embedding_model"],
                dimensions=config["embedding_dimensions"],
            )
            embeddings.extend([embedding.embedding for embedding in response.data])
            used_tokens += response.usage.total_tokens

        cost = used_tokens * self.model_to_cost_per_token[config["embedding_model"]]
        self.accumulated_costs += cost
        if self.print_usage_info:
            print(
                f"Model: {config['embedding_model']}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        return embeddings

    def create_completion(self, messages: list[Message], model: str, reasoning_effort: str | None = None) -> str:
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

        response = self.client.chat.completions.create(
            messages=completion_messages, model=model, **completion_params(model, reasoning_effort)
        )

        input_tokens = response.usage.prompt_tokens
        # includes reasoning tokens, which are billed as output
        output_tokens = response.usage.completion_tokens
        prices = self.model_to_cost_per_token.get(model)
        if prices is None:
            logger.warning(f"No price known for model {model}; its cost is not tracked")
            prices = {"input": 0.0, "output": 0.0}
        cost = (input_tokens * prices["input"]) + (output_tokens * prices["output"])
        self.accumulated_costs += cost
        if self.print_usage_info:
            print(
                f"Model: {model}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens},\
                Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        return response.choices[0].message.content.strip()


# function to calculate the number of tokens in a string
# taken from https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
