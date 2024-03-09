from os import environ

from openai import OpenAI

from .base import LLMInterface


class OpenAIInterface(LLMInterface):
    defaults = {
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024
    }

    model_to_cost_per_token = {
        "text-embedding-3-large": 0.13 / 1e6
        # $0.13 per 1,000,000 tokens, as of March 2024 (https://openai.com/pricing)
    }

    def __init__(self, print_usage_info: bool = True):
        self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))
        self.accumulated_costs = 0.0
        self.print_usage_info = print_usage_info

    def create_embedding(self, text: str, config: dict = None) -> list[float]:
        if config is None:
            config = {}
        # merge provided config with defaults
        config = {**self.defaults, **config}

        response = self.client.embeddings.create(input=[text], model=config["embedding_model"],
                                                 dimensions=config["embedding_dimensions"])

        used_tokens = response.usage.total_tokens
        cost = used_tokens * self.model_to_cost_per_token[config['embedding_model']]
        self.accumulated_costs += cost
        if self.print_usage_info:
            print(
                f"Model: {config['embedding_model']}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}")

        return response.data[0].embedding
