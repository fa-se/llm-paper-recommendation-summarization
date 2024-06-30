from os import environ

import tiktoken
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .base import LLMInterface, LLMType, Message, Task


class OpenAIInterface(LLMInterface):
    defaults = {
        # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        # "quality_model": "gpt-4-0125-preview",
        "quality_model": "gpt-4o-2024-05-13",
        "budget_model": "gpt-3.5-turbo-0125",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024,
    }

    model_to_cost_per_token = {
        # https://openai.com/pricing
        # "gpt-4-0125-preview": 10.00 / 1e6,
        "gpt-4o-2024-05-13": 5.00 / 1e6,
        "gpt-3.5-turbo-0125": 0.50 / 1e6,
        "text-embedding-3-large": 0.13 / 1e6,
    }

    def __init__(self, print_usage_info: bool = True):
        self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))
        self.accumulated_costs = 0.0
        self.print_usage_info = print_usage_info

    def handle_task(self, task: Task) -> str:
        messages = task.get_prompt(LLMType.GPT)
        model = self.defaults["quality_model"] if task.prioritize_quality else self.defaults["budget_model"]
        completion = self.create_completion(messages=messages, model=model)

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

    def create_completion(self, messages: list[Message], model: str) -> str:
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

        response = self.client.chat.completions.create(messages=completion_messages, model=model)

        used_tokens = response.usage.total_tokens
        cost = used_tokens * self.model_to_cost_per_token[model]
        self.accumulated_costs += cost
        if self.print_usage_info:
            print(
                f"Model: {model}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}"
            )

        return response.choices[0].message.content.strip()


# function to calculate the number of tokens in a string
# taken from https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
