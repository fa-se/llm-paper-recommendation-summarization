"""Pipeline instrumentation: timings, token counts, cost and domain events per pipeline stage.

A Trace records events as JSON-serializable dicts. It's the backbone of the demo's live pipeline view (each event is
passed to a listener as it happens), its cost counter, and replay mode (save a run, replay it with its original timing):

    with Trace(on_event=print) as trace:
        works = retrieval.get_relevant_works_for_query(query, n=5, start_date=start_date)
    trace.save("run.json")
    for event in replay(load_events("run.json")):
        ...

Pipeline code gets the active trace from current_trace(); outside of a Trace, that's a no-op trace.
Every event has the keys seq, t (seconds since the trace started), stage (innermost open stage or None) and type.
"""

import contextvars
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("current_trace", default=None)

USAGE_KEYS = ("llm_calls", "input_tokens", "output_tokens", "cost_usd")


def _empty_usage() -> dict:
    return {key: 0 for key in USAGE_KEYS}


class Trace:
    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self.events: list[dict] = []
        self.on_event = on_event
        self.usage = _empty_usage()  # totals over all stages
        self._stages: list[dict] = []  # open stages, innermost last; each accumulates its own usage
        self._lock = threading.RLock()
        self._start = time.perf_counter()
        self._token = None

    def __enter__(self) -> "Trace":
        self._token = _current_trace.set(self)
        return self

    def __exit__(self, *exc_info):
        _current_trace.reset(self._token)

    def emit(self, type: str, **data):
        with self._lock:
            event = {
                "seq": len(self.events),
                "t": round(time.perf_counter() - self._start, 3),
                "stage": self._stages[-1]["name"] if self._stages else None,
                "type": type,
                **data,
            }
            self.events.append(event)
            # called under the lock, so that the listener sees events in order even when they come from worker threads
            if self.on_event:
                self.on_event(event)

    @contextmanager
    def stage(self, name: str, **data):
        """Brackets a pipeline stage with stage_start/stage_end events; stage_end carries its duration and LLM usage."""
        stage = {"name": name, "start": time.perf_counter(), **_empty_usage()}
        with self._lock:
            self._stages.append(stage)
            self.emit("stage_start", **data)
        try:
            yield
        finally:
            with self._lock:
                usage = {key: stage[key] for key in USAGE_KEYS}
                usage["cost_usd"] = round(usage["cost_usd"], 6)
                self.emit("stage_end", duration_s=round(time.perf_counter() - stage["start"], 3), **usage)
                self._stages.remove(stage)

    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int, cost: float, latency_s: float):
        """Called by the LLM interface for every API request; attributes the usage to all open stages."""
        with self._lock:
            for usage in [self.usage, *self._stages]:
                usage["llm_calls"] += 1
                usage["input_tokens"] += input_tokens
                usage["output_tokens"] += output_tokens
                usage["cost_usd"] += cost
            self.emit(
                "llm_call",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost, 6),
                latency_s=round(latency_s, 3),
                total_cost_usd=round(self.usage["cost_usd"], 6),
            )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"events": self.events}, f, ensure_ascii=False, indent=1)


class _NullTrace(Trace):
    def emit(self, type: str, **data):
        pass

    @contextmanager
    def stage(self, name: str, **data):
        yield

    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int, cost: float, latency_s: float):
        pass


_NULL_TRACE = _NullTrace()


def current_trace() -> Trace:
    return _current_trace.get() or _NULL_TRACE


def submit_in_context(executor, fn, *args, **kwargs):
    """executor.submit, but fn runs with the caller's context variables (and so with the caller's trace)."""
    return executor.submit(contextvars.copy_context().run, fn, *args, **kwargs)


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["events"]


def replay(events: list[dict], speed: float = 1.0) -> Iterator[dict]:
    """Yields recorded events with their original timing (divided by speed)."""
    start = time.perf_counter()
    for event in events:
        delay = event["t"] / speed - (time.perf_counter() - start)
        if delay > 0:
            time.sleep(delay)
        yield event
