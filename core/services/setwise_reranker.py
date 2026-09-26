"""Setwise LLM reranking with heapsort (Zhuang et al., SIGIR 2024, https://arxiv.org/abs/2310.09497).

Each LLM call sees a heap node and its children (3 abstracts with num_child=2) and names the most relevant one; heapsort
then yields the top k. This reimplements the heapsort of llm-rankers (https://github.com/ielab/llm-rankers), which the
thesis used: same prompt, same comparisons, same result. What's new is concurrency, because ~55 sequential calls took
~45 s:
- Parallel heap construction: a node is heapified as soon as its children's subtrees are heaps. Different nodes'
  subtrees are disjoint, so their comparisons run concurrently.
- Speculative lookahead: an element sinking down the heap is compared with the children of each node on its path. The
  comparison at a child node involves the same sinking element and that child's (still unchanged) children, so it can
  be sent before the comparison at the parent returns. Speculative calls off the path taken are wasted.
Every comparison and swap is emitted as a trace event (core/instrumentation.py), e.g. for a live heap animation.
"""

import itertools
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from core.dataclasses.data_classes import Work
from core.instrumentation import current_trace, submit_in_context
from core.llm_interfaces import LLMInterface
from core.llm_interfaces.tasks import SetwiseComparisonTask

logger = logging.getLogger(__name__)


class SetwiseHeapsortReranker:
    def __init__(self, llm_interface: LLMInterface, num_child: int = 2, lookahead: int = 1, max_workers: int = 32):
        """
        Parameters:
            num_child: children per heap node; each comparison shows num_child + 1 passages.
            lookahead: how many levels below the current node comparisons are sent speculatively (0: none).
            max_workers: maximum number of concurrent LLM requests.
        """
        self.llm_interface = llm_interface
        self.num_child = num_child
        self.lookahead = lookahead
        self.max_workers = max_workers

    def rerank(self, query: str, works: list[Work], k: int) -> list[Work]:
        """Returns the k most relevant works in order, followed by the remaining works in their original order."""
        return _HeapsortRun(self, query, works).run(k)


class _HeapsortRun:
    """State of one reranking. Heap positions follow llm-rankers: the children of node i are num_child*i+1, ..."""

    def __init__(self, reranker: SetwiseHeapsortReranker, query: str, works: list[Work]):
        self.reranker = reranker
        self.num_child = reranker.num_child
        self.query = query
        self.works = works
        self.heap = list(works)
        self.trace = current_trace()
        self.llm_pool: ThreadPoolExecutor | None = None
        self.call_ids = itertools.count()
        self.used_calls = 0
        self.lock = threading.Lock()

    def children(self, node: int, size: int) -> list[int]:
        first = self.num_child * node + 1
        return list(range(first, min(first + self.num_child, size)))

    def is_in_subtree(self, node: int, root: int) -> bool:
        while node > root:
            node = (node - 1) // self.num_child
        return node == root

    def run(self, k: int) -> list[Work]:
        size = len(self.heap)
        self.trace.emit("heap_init", docids=[work.id for work in self.heap], num_child=self.num_child, k=k)
        ranked: list[Work] = []
        with ThreadPoolExecutor(max_workers=self.reranker.max_workers) as self.llm_pool:
            self.build_heap(size)
            # the root is the most relevant work left; move it behind the heap, then restore the heap on the rest
            for last in range(size - 1, -1, -1):
                if last > 0:
                    self.swap(0, last)
                ranked.append(self.heap[last])
                self.trace.emit("ranked", rank=len(ranked), position=last, docid=self.heap[last].id)
                if len(ranked) == k or last == 0:
                    break
                self.sift_down(0, last)

        top_ids = {work.id for work in ranked}
        calls = next(self.call_ids)
        self.trace.emit("reranked", docids=[work.id for work in ranked], calls=calls, used_calls=self.used_calls)
        logger.info(
            f"Reranked top {len(ranked)} of {size} with {calls} LLM calls ({self.used_calls} on the path taken)"
        )
        return ranked + [work for work in self.works if work.id not in top_ids]

    def build_heap(self, size: int):
        internal_nodes = [node for node in range(size) if self.children(node, size)]
        done: dict[int, Future] = {}
        # one thread per node, so that nodes waiting for their children never starve the ones they wait for
        with ThreadPoolExecutor(max_workers=max(len(internal_nodes), 1)) as drivers:
            for node in reversed(internal_nodes):
                subtrees = [done[child] for child in self.children(node, size) if child in done]
                done[node] = submit_in_context(drivers, self.sift_down_after, node, size, subtrees)
            for future in done.values():
                future.result()

    def sift_down_after(self, node: int, size: int, subtrees: list[Future]):
        for subtree in subtrees:
            subtree.result()
        self.sift_down(node, size)

    def sift_down(self, node: int, size: int):
        """Moves heap[node] down until the LLM prefers it over its children ("heapify" in llm-rankers)."""
        sinking = self.heap[node]
        pending: dict[int, Future] = {}
        while self.children(node, size):
            self.speculate(pending, sinking, node, size)
            winner = pending.pop(node).result()
            with self.lock:
                self.used_calls += 1
            # comparisons outside the winner's subtree are off the path now
            for other in [other for other in pending if not self.is_in_subtree(other, winner)]:
                pending.pop(other).cancel()
            if winner == node:
                break
            self.swap(node, winner)
            node = winner
        for future in pending.values():
            future.cancel()

    def speculate(self, pending: dict[int, Future], sinking: Work, node: int, size: int):
        """Sends the comparisons of the sinking work at node and at its descendants up to `lookahead` levels below."""
        level = [node]
        for _ in range(self.reranker.lookahead + 1):
            for candidate in level:
                children = self.children(candidate, size)
                if children and candidate not in pending:
                    positions = [candidate] + children
                    docs = [sinking] + [self.heap[child] for child in children]
                    pending[candidate] = submit_in_context(
                        self.llm_pool, self.compare, positions, docs, speculative=candidate != node
                    )
            level = [child for candidate in level for child in self.children(candidate, size)]

    def compare(self, positions: list[int], docs: list[Work], speculative: bool) -> int:
        """Asks the LLM which of docs is the most relevant; returns its heap position."""
        call = next(self.call_ids)
        self.trace.emit(
            "compare_start", call=call, positions=positions, docids=[doc.id for doc in docs], speculative=speculative
        )
        task = SetwiseComparisonTask(self.query, [doc.abstract for doc in docs])
        response = self.reranker.llm_interface.handle_task(task)
        best = task.parse_response(response)
        if best is None:
            # like llm-rankers: keep the parent
            logger.warning(f"Unexpected reranker output: {response!r}")
            best = 0
        self.trace.emit("compare_end", call=call, winner=positions[best])
        return positions[best]

    def swap(self, a: int, b: int):
        self.heap[a], self.heap[b] = self.heap[b], self.heap[a]
        self.trace.emit("swap", positions=[a, b])
