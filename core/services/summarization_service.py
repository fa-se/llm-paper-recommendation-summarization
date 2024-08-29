import json

from core.dataclasses.data_classes import Work, SummarizedWork
from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import CustomizedSummaryTask


class SummarizationService:
    def __init__(
        self,
        llm_interface: LLMInterface,
    ):
        self.llm_interface = llm_interface

    def summarize_works_for_query(self, query: str, works: list[Work]) -> list[SummarizedWork]:
        # only consider works with abstracts
        works_with_abstracts = [work for work in works if work.abstract]

        summarized_works: list[SummarizedWork] = []
        for work in works_with_abstracts:
            task = CustomizedSummaryTask(
                area_of_research=query,
                abstract=work.abstract,
                prioritize_quality=True,
            )
            response = self.llm_interface.handle_task(task)
            reasoning_structure_json = response.strip("```json").strip("```")
            reasoning_structure = json.loads(reasoning_structure_json)["Reasoning Structure"]
            summary = reasoning_structure["FINAL_ANSWER"]
            summarized_works.append(SummarizedWork(work, summary))

        return summarized_works
