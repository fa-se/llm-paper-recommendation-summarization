"""Would rewriting the query help topic routing? A label-free check with the thesis' citation proxy.

For a sample of recent OpenAlex papers, a paper's references stand in for what its author wants to find, and their
primary topics for the topics that routing should reach. Each paper gives these queries:
- abstract: the paper's abstract (the query kind of the thesis' reranking eval)
- fftd: a free-form description of the author's research interests, written by an LLM from the abstract; the kind of
  input the pipeline gets, and the baseline
- template: the fftd rewritten into the format the topic embeddings were made from (HyDE-style: bring the query into
  "topic space"): "Topic: ...; Description: ...; Keywords: ..."
- facets_max / facets_rrf: the fftd split into 2-5 facets, each routed on its own; a topic scores its best similarity
  to any facet (max), or the sum of 1 / (60 + rank) over the facets (reciprocal rank fusion)
Routing recall@k: the share of the paper's references whose primary topic is among the k topics most similar to the
query (the pipeline fetches works by primary topic, with k = 10).

An experiment, not part of the pipeline: the library and the demo route with the raw description, as in the thesis.
    uv run --env-file .env scripts/eval_topic_routing.py [--n 150] [--seed 42] [--out topic_routing_eval.json]
"""

import argparse
import json
import logging
import random
import statistics
from concurrent.futures import ThreadPoolExecutor

import pyalex

import core
from core.instrumentation import Trace, submit_in_context
from core.llm_interfaces.base import Message

logger = logging.getLogger(__name__)

KS = (5, 10, 20)
# ranks kept per query; enough for fusing facets exactly up to rank 20 (a topic below a facet's rank 50 has 50 topics
# above it in the fused ranking, too)
DEPTH = 50
VARIANTS = ("abstract", "fftd", "template", "facets_max", "facets_rrf")
MIN_REFERENCES = 10

FFTD_PROMPT = """Below is the abstract of a paper. Its first author signs up for a service that recommends new papers \
and describes their research interests in a text box. Write that description: first person, 3 to 5 sentences, as if \
telling a colleague what they work on, how, and what kind of new work they want to follow. Describe the interests \
behind the paper, not the paper or its results, and don't copy phrases from the abstract.

Abstract:
{abstract}"""

TEMPLATE_PROMPT = """Below is a researcher's description of their research interests. Rewrite it as an entry of the \
OpenAlex topic taxonomy, on one line, in exactly this format:
Topic: <topic name, 3 to 8 words>; Description: <one or two sentences in the third person about the research area>; \
Keywords: <5 to 10 comma-separated keywords>

Description of interests:
{fftd}"""

FACETS_PROMPT = """Below is a researcher's description of their research interests. Split it into its distinct facets \
(2 to 5), e.g. the subject studied, the methods, and the applications or questions they care about. Write each facet \
as a short phrase or sentence that makes sense on its own.

Description of interests:
{fftd}"""

FACETS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "facets",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"facets": {"type": "array", "items": {"type": "string"}}},
            "required": ["facets"],
            "additionalProperties": False,
        },
    },
}


def short_id(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def sample_papers(n: int, seed: int) -> list[dict]:
    query = (
        pyalex.Works()
        .filter(has_abstract=True, type="article", language="en", from_publication_date="2024-01-01")
        .filter(referenced_works_count=f">{2 * MIN_REFERENCES - 1}")
        .sample(n, seed=seed)
        .select(["id", "title", "abstract_inverted_index", "referenced_works", "primary_topic"])
    )
    papers = []
    for work in query.get(per_page=min(n, 200)):
        abstract = work["abstract"]
        if not abstract or len(abstract.split()) < 50 or not work.get("primary_topic"):
            continue
        papers.append(
            {
                "id": short_id(work["id"]),
                "title": work["title"],
                "abstract": abstract,
                "topic": int(short_id(work["primary_topic"]["id"]).lstrip("T")),
                "references": [short_id(url) for url in work["referenced_works"]],
            }
        )
    return papers


def primary_topics(work_ids: list[str]) -> dict[str, int]:
    topics = {}
    for start in range(0, len(work_ids), 100):
        batch = work_ids[start : start + 100]
        query = pyalex.Works().filter(openalex_id="|".join(batch)).select(["id", "primary_topic"])
        for work in query.get(per_page=100):
            if work.get("primary_topic"):
                topics[short_id(work["id"])] = int(short_id(work["primary_topic"]["id"]).lstrip("T"))
    return topics


def complete(prompt: str, response_format: dict | None = None) -> str:
    llm = core.retrieval.llm
    model = llm.models["rerank"]
    return llm.create_completion(
        [Message("user", prompt)],
        model=model.name,
        reasoning_effort=model.reasoning_effort,
        temperature=0.0,
        response_format=response_format,
    )


def in_parallel(fn, items: list, workers: int = 8) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [submit_in_context(pool, fn, item) for item in items]
        results = []
        for item, future in zip(items, futures, strict=True):
            try:
                results.append(future.result())
            except Exception as error:
                logger.warning(f"{item.get('id', '?')}: {error}")
                results.append(None)
        return results


def ranking(embedding: list[float]) -> list[tuple[int, float]]:
    matches = core.retrieval.topic_repository.most_similar(embedding, DEPTH)
    return [(topic.id, similarity) for topic, similarity in matches]


def fuse(rankings: list[list[tuple[int, float]]], method: str) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, (topic_id, similarity) in enumerate(ranked, start=1):
            if method == "max":
                scores[topic_id] = max(scores.get(topic_id, -1.0), similarity)
            else:
                scores[topic_id] = scores.get(topic_id, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda topic_id: scores[topic_id], reverse=True)


def bootstrap_ci(values: list[float], rng: random.Random, rounds: int = 5000) -> tuple[float, float]:
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main(n: int, seed: int, out: str):
    with Trace() as trace:
        papers = sample_papers(n, seed)
        logger.info(f"Sampled {len(papers)} papers; fetching the primary topics of their references")
        topics = primary_topics(sorted({ref for paper in papers for ref in paper["references"]}))
        for paper in papers:
            paper["reference_topics"] = [topics[ref] for ref in paper["references"] if ref in topics]
        papers = [paper for paper in papers if len(paper["reference_topics"]) >= MIN_REFERENCES]
        logger.info(f"{len(papers)} papers with at least {MIN_REFERENCES} references with a primary topic")

        fftds = in_parallel(lambda p: complete(FFTD_PROMPT.format(abstract=p["abstract"])), papers)
        for paper, fftd in zip(papers, fftds, strict=True):
            paper["fftd"] = fftd
        papers = [paper for paper in papers if paper["fftd"]]
        templates = in_parallel(lambda p: complete(TEMPLATE_PROMPT.format(fftd=p["fftd"])), papers)
        facets = in_parallel(
            lambda p: json.loads(complete(FACETS_PROMPT.format(fftd=p["fftd"]), FACETS_FORMAT))["facets"], papers
        )
        for paper, template, paper_facets in zip(papers, templates, facets, strict=True):
            paper["template"], paper["facets"] = template, paper_facets
        papers = [paper for paper in papers if paper["template"] and paper["facets"]]

        texts = [
            text for paper in papers for text in (paper["abstract"], paper["fftd"], paper["template"], *paper["facets"])
        ]
        embeddings = iter(core.retrieval.llm.create_embedding_batch(texts))
        for paper in papers:
            abstract, fftd, template = (ranking(next(embeddings)) for _ in range(3))
            facet_rankings = [ranking(next(embeddings)) for _ in paper["facets"]]
            routed = {
                "abstract": [topic_id for topic_id, _ in abstract],
                "fftd": [topic_id for topic_id, _ in fftd],
                "template": [topic_id for topic_id, _ in template],
                "facets_max": fuse(facet_rankings, "max"),
                "facets_rrf": fuse(facet_rankings, "rrf"),
            }
            refs = paper["reference_topics"]
            paper["recall"] = {
                variant: {k: sum(topic in set(routed[variant][:k]) for topic in refs) / len(refs) for k in KS}
                for variant in VARIANTS
            }
            paper["own_topic_rank"] = {
                variant: routed[variant].index(paper["topic"]) + 1 if paper["topic"] in routed[variant] else None
                for variant in VARIANTS
            }
            # how flat the similarity is around the cut at rank 10
            paper["fftd_gap_10_11"] = fftd[9][1] - fftd[10][1]
            paper["fftd_spread_1_10"] = fftd[0][1] - fftd[9][1]

    report(papers, trace.usage["cost_usd"], seed)
    with open(out, "w") as f:
        json.dump({"seed": seed, "papers": papers}, f, ensure_ascii=False, indent=1)
    print(f"Saved per-paper results to {out}")


def report(papers: list[dict], cost: float, seed: int):
    rng = random.Random(seed)
    print(
        f"\n{len(papers)} papers, median {statistics.median(len(p['reference_topics']) for p in papers):.0f} "
        f"references with a primary topic; API cost ${cost:.3f}\n"
    )
    print(f"{'routing recall':<14}" + "".join(f"{f'@{k} (95% CI)':>22}" for k in KS) + f"{'own topic in top 10':>22}")
    for variant in VARIANTS:
        cells = []
        for k in KS:
            values = [p["recall"][variant][k] for p in papers]
            low, high = bootstrap_ci(values, rng)
            cells.append(f"{statistics.fmean(values):.3f} ({low:.3f}-{high:.3f})")
        own = statistics.fmean((p["own_topic_rank"][variant] or DEPTH + 1) <= 10 for p in papers)
        print(f"{variant:<14}" + "".join(f"{cell:>22}" for cell in cells) + f"{own:>22.2f}")
    print("\npaired difference to fftd at k = 10: mean (95% CI), papers better / equal / worse")
    for variant in VARIANTS:
        if variant == "fftd":
            continue
        diffs = [p["recall"][variant][10] - p["recall"]["fftd"][10] for p in papers]
        low, high = bootstrap_ci(diffs, rng)
        better, worse = sum(d > 1e-9 for d in diffs), sum(d < -1e-9 for d in diffs)
        print(
            f"  {variant:<12} {statistics.fmean(diffs):+.3f} ({low:+.3f} to {high:+.3f})   "
            f"{better} / {len(diffs) - better - worse} / {worse}"
        )
    gains = [p["recall"]["fftd"][20] - p["recall"]["fftd"][10] for p in papers]
    gap = statistics.median(p["fftd_gap_10_11"] for p in papers)
    spread = statistics.median(p["fftd_spread_1_10"] for p in papers)
    print(
        f"\nfftd: similarity gap between rank 10 and 11: median {gap:.4f}; rank 1 to 10 spread: median {spread:.3f}; "
        f"recall gained by 20 instead of 10 topics: {statistics.fmean(gains):+.3f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=150, help="papers to sample (at most 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="topic_routing_eval.json")
    args = parser.parse_args()
    main(args.n, args.seed, args.out)
