from concurrent.futures import ThreadPoolExecutor

from core.dataclasses.data_classes import Work, SummarizedWork
from core.instrumentation import current_trace, submit_in_context
from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import CustomizedSummaryTask


class SummarizationService:
    def __init__(
        self,
        llm_interface: LLMInterface,
    ):
        self.llm_interface = llm_interface

    def summarize_works_for_query(self, query: str, works: list[Work], max_workers: int = 5) -> list[SummarizedWork]:
        # only consider works with abstracts
        works_with_abstracts = [work for work in works if work.abstract]

        # the summaries are independent, so they are generated concurrently (one takes several seconds)
        with current_trace().stage("summarize", works=len(works_with_abstracts)):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [submit_in_context(executor, self._summarize, query, work) for work in works_with_abstracts]
                return [future.result() for future in futures]

    def _summarize(self, query: str, work: Work) -> SummarizedWork:
        task = CustomizedSummaryTask(
            area_of_research=query,
            abstract=work.abstract,
            prioritize_quality=True,
        )
        reasoning_structure = task.parse_response(self.llm_interface.handle_task(task))
        summary = reasoning_structure["FINAL_ANSWER"]
        current_trace().emit("summary", docid=work.id, summary=summary, reasoning=reasoning_structure)
        return SummarizedWork(work, summary, reasoning_structure)
