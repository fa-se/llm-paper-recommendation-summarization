"""The prompts of the pipeline: setwise reranking comparisons and tailored summaries."""

import json
import re
import string
import textwrap

from .base import Message, Task


class TailoredSummaryTask(Task):
    """A summary of a publication tailored to the user's research interests: their focus, their terminology.

    The prompt follows Self-Discover (Zhou et al. 2024): the model fills out a reasoning structure step by step, ending
    with the summary as FINAL_ANSWER. The structure was generated offline with Self-Discover on example pairs, then
    merged by hand and frozen.

    The prompt text is exactly the one evaluated in the thesis, including its quirks (odd indentation, no step 4, a
    missing space in the system prompt, the code block instruction that structured outputs now make moot).
    """

    model_tier = "quality"

    self_discover_prompt_template = string.Template(
        textwrap.dedent("""# Given Reasoning Structure

            ```json
            $reasoning_structure
            ```
            
            # Given Task
            
            $task
            
            ## Example Output 
            
            ```json
            {
              ...
            }
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

    # (sic) no space between "instructions" and "from"
    system_prompt = (
        "You are a helpful AI chatbot who pays very close attention to instructions"
        "from the user - especially any instructions on how to format your response."
    )

    def __init__(self, research_interests: str, abstract: str):
        """
        Parameters:
            research_interests: the user's free-form description of their research interests (the query).
            abstract: abstract of the publication to summarize.
        """
        self.research_interests = research_interests
        self.abstract = abstract

    def messages(self) -> list[Message]:
        task = self.task_template.substitute(
            research_interest_description=self.research_interests, abstract=self.abstract
        )
        prompt = self.self_discover_prompt_template.substitute(reasoning_structure=self.reasoning_structure, task=task)
        return [Message("system", self.system_prompt), Message("user", prompt)]

    def response_format(self) -> dict:
        # Structured outputs guarantee a complete reasoning structure (they replace the code block the prompt asks for).
        # The schema keeps the template's key order, so the model still works through the steps before FINAL_ANSWER.
        schema = _json_schema_for_template(json.loads(self.reasoning_structure))
        return {"type": "json_schema", "json_schema": {"name": "reasoning_structure", "strict": True, "schema": schema}}

    def parse_response(self, response: str) -> dict:
        """The filled out reasoning structure; its FINAL_ANSWER is the summary."""
        return json.loads(response)["Reasoning Structure"]


def _json_schema_for_template(template: dict | str) -> dict:
    """Strict JSON schema for a JSON template whose leaves are strings to fill out."""
    if isinstance(template, dict):
        return {
            "type": "object",
            "properties": {key: _json_schema_for_template(value) for key, value in template.items()},
            "required": list(template),
            "additionalProperties": False,
        }
    return {"type": "string"}


class SetwiseComparisonTask(Task):
    """One comparison of setwise LLM reranking: which of the passages is the most relevant to the query?

    Prompt of llm-rankers' OpenAiSetwiseLlmRanker (https://github.com/ielab/llm-rankers), which the thesis used.
    """

    model_tier = "rerank"
    temperature = 0.0
    # a comparison normally takes ~1s; retry a stuck request early instead of stalling the whole ranking
    timeout = 15.0

    labels = "ABCDEFGHIJKLMNOPQRSTUVW"
    system_prompt = (
        "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of "
        "passages based on their relevance to the query."
    )

    def __init__(self, query: str, passages: list[str]):
        self.query = query
        self.passages = passages

    def messages(self) -> list[Message]:
        passages = "\n\n".join(f'Passage {self.labels[i]}: "{passage}"' for i, passage in enumerate(self.passages))
        prompt = (
            f'Given a query "{self.query}", which of the following passages is the most relevant one to the query?\n\n'
            f"{passages}\n\nOutput only the passage label of the most relevant passage."
        )
        return [Message("system", self.system_prompt), Message("user", prompt)]

    def parse_response(self, response: str) -> int | None:
        """Index of the chosen passage, or None if the response names none of them."""
        match = re.search(r"Passage ([A-Z])", response)
        label = match.group(1) if match else response.strip()
        if len(label) == 1 and label in self.labels[: len(self.passages)]:
            return self.labels.index(label)
        return None
