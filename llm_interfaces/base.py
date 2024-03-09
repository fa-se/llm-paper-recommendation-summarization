class LLMInterface:
    def create_embedding(self, text: str, config: dict = None) -> list[float]:
        raise NotImplementedError
