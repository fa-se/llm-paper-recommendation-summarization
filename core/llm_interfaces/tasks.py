import string
import textwrap

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

    self_discover_prompt_templates = string.Template(
        textwrap.dedent("""# Given Reasoning Structure

            ```json
            $reasoning_structure
            ```
            
            # Given Task
            
            $task
            
            ## Example Output 
            
            ```json
            {{
              ...
            }}
            ```
            
            # Detailed Instructions 
            
            You must use the given REASONING STRUCTURE to solve the GIVEN TASK, both are provided above.
            The REASONING STRUCTURE will guide your answer for the GIVEN TASK. 
            You must fill out ALL of the empty strings on the value side of the key-value pairs in the JSON structure of the REASONING STRUCTURE.
            Your output will consist of one codeblock. The codeblock will be a json codeblock enclosed by triple back-ticks with the json language specifier as shown in the "Example Output".
            The json codeblock will contain the completely filled out reasoning structure.
        """)
    )

    task_template = string.Template(
        """Below, you will be provided with the abstract of a research publication and the description of a user's research interests. Your task is to write a summary of the publication that focuses on aspects related to these research interests.
        The target of the summary is the user who provided the description of their research interests, and thus the summary should be tailored to their interests and adopt their terminology. It should not include any verbatim text from the abstract or the user-provided description. The summary should be as brief as possible while still being informative, helping the user to quickly grasp the essential points. It should not directly address the user.
    
        ## Research Interest Description
        $research_interest_description
    
        ## Abstract
        $abstract
        """
    )

    reasoning_structure = """{
        "Reasoning Structure": {
            "Step 1: Identify Key Findings": {
                "Action": "Extract key findings from the abstract.",
                "Key Findings": ""
            },
            "Step 2: Describe Methodologies": {
                "Action": "Outline the methodologies used in the publication.",
                "Methodologies": ""
            },
            "Step 3: Summarize Conclusions": {
                "Action": "Summarize the conclusions drawn from the research.",
                "Conclusions": ""
            },
            "Step 5: Align with Research Interests": {
                "Action": "Highlight aspects related to the user's research interests.",
                "Related Aspects": ""
            },
            "Step 6: Assess Significance": {
                "Action": "Assess the significance of the publication for the user's research areas.",
                "Significance": ""
            },
            "FINAL_ANSWER": ""
        }
    }
    """

    prompt_templates = {
        LLMType.GPT: {
            "system": Message(
                "system",
                "You are a helpful AI chatbot who pays very close attention to instructions"
                "from the user - especially any instructions on how to format your response.",
            )
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
        task = self.task_template.substitute(
            research_interest_description=self.area_of_research, abstract=self.abstract
        )
        prompt = self.self_discover_prompt_templates.substitute(reasoning_structure=self.reasoning_structure, task=task)
        user_message = Message("user", prompt)
        return [system_message, user_message]
