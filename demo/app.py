"""Demo web app: runs the pipeline live and streams its trace events to the browser, or replays a recorded run.

Start it from the repo root (with a running db; replays work without db and API key):
    uv run --env-file .env python -m demo [--port 8000]

The page (static/) renders everything from the event stream of core/instrumentation.py: stage timings and cost, the
topic map, the retrieval rankings, the setwise heap and the summaries. A live run streams its events over server-sent
events (SSE) while it runs; every successful live run is saved to recordings/, and a replay loads such a file and plays
it in the browser with its original timing. So a replay looks exactly like the live run it was recorded from.
"""

import argparse
import asyncio
import contextlib
import itertools
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

import core
from core.instrumentation import Trace
from demo import corpus
from scripts.demo_queries import DEMO_QUERIES, START_DATE

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
RECORDINGS = Path(__file__).parent / "recordings"
# topics a query is routed to (as in the pre-ingest of the demo queries, scripts/demo_queries.py)
NUM_TOPICS = 10
# newest works fetched per live ingest (the old "fetch new papers first", over all topics together)
INGEST_LIMIT = 500
# the most works a live run may fetch per topic: 1,000 per topic for 10 topics take ~16 s to fetch and ~$0.45 to embed
MAX_PER_TOPIC = 1000


class RunRequest(BaseModel):
    query: str = Field(min_length=20, max_length=3000)
    query_name: str | None = None
    n: int = Field(default=5, ge=1, le=10)
    summaries: int = Field(default=3, ge=0, le=5)
    # fetch and embed the newest works on the query's topics first: needed for a query outside the demo corpus.
    # fetch_per_topic: the newest N works of each topic; ingest (older clients): INGEST_LIMIT over all topics
    fetch_per_topic: int = Field(default=0, ge=0, le=MAX_PER_TOPIC)
    ingest: bool = False


@dataclass
class Run:
    """A live run: its events so far, which any number of SSE streams follow."""

    id: int
    request: RunRequest
    loop: asyncio.AbstractEventLoop
    events: list[dict] = field(default_factory=list)
    done: bool = False
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)

    def push(self, event: dict):
        # the Trace calls this from worker threads while it holds its lock, so it must not block
        self.events.append(event)
        self._notify()

    def finish(self):
        self.done = True
        self._notify()

    def _notify(self):
        with contextlib.suppress(RuntimeError):  # the server is shutting down
            self.loop.call_soon_threadsafe(self.wakeup.set)


app = FastAPI(title="Paper recommendation demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# one pipeline at a time: two concurrent reranks would exceed the reranker model's tokens-per-minute limit. The single
# worker thread also keeps one DB session (sessions are thread-local) for all runs.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
_runs: dict[int, Run] = {}
_run_ids = itertools.count(1)
_busy = threading.Lock()


def _execute(run: Run):
    request = run.request
    with Trace(on_event=run.push) as trace:
        trace.emit(
            "run_start",
            query_name=request.query_name,
            query=request.query,
            start_date=START_DATE.date().isoformat(),
            n=request.n,
            summaries=request.summaries,
            ingest=bool(request.ingest or request.fetch_per_topic),
            fetch_per_topic=request.fetch_per_topic,
        )
        try:
            if request.fetch_per_topic:
                core.retrieval.ingest(
                    request.query, START_DATE, num_topics=NUM_TOPICS, per_topic=request.fetch_per_topic
                )
            elif request.ingest:
                core.retrieval.ingest(request.query, START_DATE, limit=INGEST_LIMIT, num_topics=NUM_TOPICS)
            else:
                core.retrieval.match_topics(request.query, NUM_TOPICS)
            # search() ranks the whole corpus, not only the works fetched for this query; oldest/newest: how far back
            # the searched papers reach
            trace.emit("corpus", size=core.retrieval.publication_repository.count(), **_date_range())
            works = core.retrieval.search(request.query, n=request.n, start_date=START_DATE)
            if request.summaries:
                core.summarization.summarize(request.query, works[: request.summaries])
            trace.emit("run_end", top=[work.id for work in works[: request.n]], usage=trace.usage)
        except Exception as error:
            logger.exception("Run failed")
            with contextlib.suppress(Exception):
                core.retrieval.publication_repository.session.rollback()
            trace.emit("run_error", message=f"{type(error).__name__}: {error}")
            return
    RECORDINGS.mkdir(exist_ok=True)
    trace.save(RECORDINGS / f"{datetime.now():%Y-%m-%d_%H%M%S}_{request.query_name or 'custom'}.json")


def _date_range() -> dict:
    """The publication dates of the papers search() ranks (the corpus since START_DATE)."""
    session = core.retrieval.publication_repository.session
    oldest, newest = session.execute(
        text(
            "SELECT min(publication_datetime_utc)::date, max(publication_datetime_utc)::date FROM publication "
            "WHERE publication_datetime_utc >= :start"
        ),
        {"start": START_DATE},
    ).one()
    return {"oldest": oldest and oldest.isoformat(), "newest": newest and newest.isoformat()}


def _run_in_worker(run: Run):
    try:
        _execute(run)
    finally:
        run.finish()
        _busy.release()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {
        "queries": DEMO_QUERIES,
        "start_date": START_DATE.date().isoformat(),
        "num_topics": NUM_TOPICS,
        "ingest_limit": INGEST_LIMIT,
        "max_per_topic": MAX_PER_TOPIC,
        "busy": _busy.locked(),
    }


@app.get("/api/corpus")
def corpus_status():
    """The corpus size, and how many papers live runs have added since the baseline (demo/corpus.py)."""
    try:
        return corpus.status()
    except Exception as error:  # e.g. no database: replays still work
        return {"error": f"{type(error).__name__}: {error}"}


@app.post("/api/corpus/reset")
async def corpus_reset():
    """Deletes the papers that live runs added since the baseline, and rebuilds the BM25 index."""
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress")
    try:
        return await asyncio.wrap_future(_worker.submit(corpus.reset))
    except RuntimeError as error:
        raise HTTPException(400, str(error)) from error
    finally:
        _busy.release()


@app.get("/api/recordings")
def recordings():
    """Recorded runs, newest first, with what the replay menu shows about them."""
    result = []
    for path in sorted(RECORDINGS.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            events = json.loads(path.read_text())["events"]
        except (OSError, ValueError, KeyError):
            continue
        start = next((event for event in events if event["type"] == "run_start"), {})
        end = next((event for event in events if event["type"] == "run_end"), None)
        result.append(
            {
                "name": path.stem,
                "query_name": start.get("query_name"),
                "query": start.get("query"),
                "duration_s": events[-1]["t"] if events else None,
                "cost_usd": end["usage"]["cost_usd"] if end and "usage" in end else None,
            }
        )
    return result


@app.get("/api/recordings/{name}")
def recording(name: str):
    path = (RECORDINGS / f"{name}.json").resolve()
    if path.parent != RECORDINGS.resolve() or not path.is_file():
        raise HTTPException(404, "No such recording")
    return FileResponse(path, media_type="application/json")


@app.post("/api/runs")
async def start_run(request: RunRequest):
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress")
    run = Run(next(_run_ids), request, asyncio.get_running_loop())
    _runs[run.id] = run
    _worker.submit(_run_in_worker, run)
    return {"id": run.id}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: int, request: Request):
    """The run's events as server-sent events: those so far, then each as it happens, then an "end" event.

    Events carry their seq as SSE id, so a reconnecting EventSource resumes where it left off (Last-Event-ID)."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run")
    last_id = request.headers.get("last-event-id")
    start = int(last_id) + 1 if last_id and last_id.isdigit() else 0

    async def stream():
        i = start
        while True:
            while i < len(run.events):
                yield f"id: {i}\ndata: {json.dumps(run.events[i], ensure_ascii=False)}\n\n"
                i += 1
            if run.done:
                yield "event: end\ndata: {}\n\n"
                return
            run.wakeup.clear()
            if i < len(run.events) or run.done:
                continue
            try:
                await asyncio.wait_for(run.wakeup.wait(), timeout=10)
            except TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Paper recommendation demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
