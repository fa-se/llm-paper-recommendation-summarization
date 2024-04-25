from .base import LLMType, Message, Task


class AlignToExamplesTask(Task):
    """
    Task to align the input to match the structure of the provided example(s).
    Example use case: Rewrite user provided text to improve embedding similarity search by matching the structure used to generate the embeddings.
    """

    prompt_templates = {
        LLMType.GPT: {
            "system": Message(
                "system",
                "You will be provided with a free form text for which embeddings will be created at a later stage. "
                "In order to improve embedding similarity search, the input needs to be aligned with the input used "
                "to generate the embeddings. Your task is to align the input to the structure of the following examples, "
                "while preserving its meaning. In your answers, respond only with the resulting aligned text.\n\n"
                "{examples}",
            ),
            "user": Message("user", "{input_text}"),
        }
    }

    def __init__(self, input_text: str, examples: list[str], prioritize_quality: bool = True):
        """
        Parameters:
            input_text: Input text to be aligned.
            examples: List of example texts to align the input to.
            prioritize_quality: Indicates whether to prioritize quality over cost for this task, e.g. by using larger models.
        """
        super().__init__(prioritize_quality=prioritize_quality)

        self.input = input_text
        self.examples = examples

    def get_prompt(self, llm_type: LLMType) -> [Message]:
        """
        Generate the prompt for the specified LLMType.

        Parameters:
            llm_type (LLMType): The LLM type to specify which template to use.

        Returns:
            list[Message]: The messages representing the prompt for the specified LLM type.
        """
        template = self.prompt_templates[llm_type]
        system_message = template["system"].format(examples="\n".join(self.examples))
        user_message = template["user"].format(input_text=self.input)
        return [system_message, user_message]


class CustomizedSummaryTask(Task):
    """
    Task to generate a summary of the input, customized to the user's area of interest.
    """

    prompt_templates = {
        LLMType.GPT: {
            "system": Message(
                "system",
                "You will be provided with the abstract of a scientific publication, as well as a description of an area of research. "
                "Your task is to generate a summary of the abstract, explaining the topic of the publication and highlighting how it might be relevant to the area of research. "
                "Additionally, you are to provide a short assessment of how well the publication matches the area of research, which is following below."
                "Formulate your answer so that it can be presented alongside the publication metadata to assist the user in deciding whether they should read the publication or not. "
                "Do not use markdown or any special formatting.\n\n"
                "Description of the area of research: {area_of_interest_description}",
            ),
            "user": Message("user", "{abstract}"),
        }
    }

    def __init__(self, area_of_research: str, abstract: str, prioritize_quality: bool = True):
        """
        Parameters:
            area_of_research: Description of the area of research to which the summary should be customized.
            abstract: Abstract of the publication to summarize.
        """
        super().__init__(prioritize_quality=prioritize_quality)

        self.area_of_research = area_of_research
        self.abstract = abstract

    def get_prompt(self, llm_type: LLMType) -> [Message]:
        """
        Generate the prompt for the specified LLMType.

        Parameters:
            llm_type (LLMType): The LLM type to specify which template to use.

        Returns:
            list[Message]: The messages representing the prompt for the specified LLM type.
        """
        template = self.prompt_templates[llm_type]
        system_message = template["system"].format(area_of_interest_description=self.area_of_research)
        user_message = template["user"].format(abstract=self.abstract)
        return [system_message, user_message]
