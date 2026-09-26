from concurrent.futures import ThreadPoolExecutor

from core.instrumentation import current_trace, submit_in_context
from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import TailoredSummaryTask
from core.works import SummarizedWork, Work


class SummarizationService:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def summarize(self, query: str, works: list[Work], max_workers: int = 5) -> list[SummarizedWork]:
        """Summaries of the works tailored to the query, in the order of the works; works without an abstract are
        skipped."""
        works_with_abstracts = [work for work in works if work.abstract]
        # the summaries are independent, so they are generated concurrently (one takes several seconds)
        with (
            current_trace().stage("summarize", works=len(works_with_abstracts)),
            ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            futures = [submit_in_context(executor, self._summarize, query, work) for work in works_with_abstracts]
            return [future.result() for future in futures]

    def _summarize(self, query: str, work: Work) -> SummarizedWork:
        task = TailoredSummaryTask(query, work.abstract)
        reasoning = task.parse_response(self.llm.handle_task(task))
        summary = reasoning["FINAL_ANSWER"]
        current_trace().emit("summary", docid=work.id, summary=summary, reasoning=reasoning)
        return SummarizedWork(work, summary, reasoning)
