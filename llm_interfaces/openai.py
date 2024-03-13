from os import environ

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam

from .base import LLMInterface, LLMType, Message, Task


class OpenAIInterface(LLMInterface):
    defaults = {
        # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        "quality_model": "gpt-4-0125-preview",
        "budget_model": "gpt-3.5-turbo-0125",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024
    }

    model_to_cost_per_token = {
        # https://openai.com/pricing
        "gpt-4-0125-preview": 10.00 / 1e6,
        "gpt-3.5-turbo-0125": 0.50 / 1e6,
        "text-embedding-3-large": 0.13 / 1e6
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

        response = self.client.embeddings.create(input=[text], model=config["embedding_model"],
                                                 dimensions=config["embedding_dimensions"])

        used_tokens = response.usage.total_tokens
        cost = used_tokens * self.model_to_cost_per_token[config['embedding_model']]
        self.accumulated_costs += cost
        if self.print_usage_info:
            print(
                f"Model: {config['embedding_model']}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}")

        return response.data[0].embedding

    def create_completion(self, messages: [Message], model: str) -> str:
        completion_messages: [ChatCompletionMessageParam] = []
        for message in messages:
            # Due to ChatCompletionMessageParam being a union, we need to check the role and instantiate the correct type
            if message.role == 'system':
                message_param = ChatCompletionSystemMessageParam(role=message.role, content=message.content)
            elif message.role == 'user':
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
                f"Model: {model}, Tokens: {used_tokens}, Cost: ${cost:.2f}, Accumulated cost: ${self.accumulated_costs:.2f}")

        return response.choices[0].message.content.strip()
