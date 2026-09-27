// The details drawer: per step, the technical and procedural details behind the plain-language view, numbers from the
// run on screen, and answers to likely questions. It's the presenter's reference as much as the audience's.
// Sources: the thesis and its defense slides, and the measurements in the demo prep notes (docs/ in the prep folder).

import { STEP_LABELS, fmt, h } from "./util.js";

const section = (title, ...children) => h("section", { class: "d-section" }, h("h3", {}, title), ...children);
const p = (...children) => h("p", {}, ...children);
const ul = (...items) => h("ul", {}, ...items.filter(Boolean).map((item) => h("li", {}, item)));
const qa = (...pairs) => h("dl", { class: "qa" }, ...pairs.flatMap(([q, a]) => [h("dt", {}, q), h("dd", {}, a)]));
const facts = (...rows) => h("table", { class: "facts" }, h("tbody", {}, ...rows.filter(Boolean).map(([k, v]) => h("tr", {}, h("th", {}, k), h("td", {}, v)))));
const b = (text) => h("strong", {}, text);
const noRun = () => p(h("em", {}, "Start a run or a replay to see its numbers here."));

// the reranking eval with the current pipeline (scripts/eval_reranking.py, 30 queries; prep notes: "the reranking eval")
const EVAL_ANSWER =
  "Measured with citations as the right answers: a paper's abstract is the query, and the papers it cites should come back. 30 papers, with the current pipeline: out of ~1,800 candidates, hybrid search keeps 100, and its own top 10 already holds 6.7 cited papers, 3× a random 10 of those 100 (2.2). The LLM's top 10 holds 7.4: about +10\u00a0%, better in 16 of 30 queries and worse in 6 (p = 0.015). So the cheap searches do most of the ranking, and the LLM adds the last step, for about a cent per query. Citations undersell it a bit: the LLM can rank a relevant paper the author simply didn't cite. In the user study, researchers rated relevance 2.89 of 5 (n = 19).";

const CONTENT = {
  intro: (run) => [
    section(
      "The thesis",
      p("“Leveraging LLMs for Personalized Suggestion and Summarization of Scientific Publications”, Master's thesis in computer science at TU Berlin (Chair of Open Distributed Systems). Topic chosen in late 2023, a year after ChatGPT's release; submitted August 2024, defended April 2025."),
      p("Two research questions: how to combine LLMs with classic retrieval to judge a paper's relevance to one person's interests, and how to use LLMs for summaries that bring out that relevance."),
    ),
    section(
      "Requirements it was designed for",
      ul(
        [b("Easy input: "), "a free-text description as the query, no syntax."],
        [b("Discovery: "), "no citation counts or click history, because brand-new papers have neither."],
        [b("Efficiency: "), "the volume of papers and the price of LLM calls make cost the central constraint."],
      ),
    ),
    section(
      "The funnel in numbers",
      facts(
        ["OpenAlex", "327 M works; 14.7 M published in 2025 (7.0 M of them journal articles). Queried 27 Sep 2026."],
        ["Pool", "the newest papers of the matched topics, embedded once at ingest: about $0.00005 per paper."],
        ["Search", "in the database: milliseconds, no API cost."],
        ["LLM ranking", "about $0.0002 per candidate (50 candidates: ~$0.01)."],
        ["Summary", "about $0.007 each."],
      ),
    ),
    section(
      "If asked",
      qa(
        [
          "Why not just ask ChatGPT, or an AI research assistant?",
          "The thesis didn't benchmark against them. Its goals: the newest papers (days old, no citations yet), a paragraph as the query, a known cost per query, and a ranking you can inspect step by step. A general assistant with web search guarantees none of these. Related research tools the thesis cites: LitLLM, ChatCite.",
        ],
        ["Is it used anywhere?", "No. It's a thesis prototype, a Python library without a UI; this page was built for today."],
        [
          "Why OpenAlex?",
          "Nonprofit, free, an open API with generous rate limits, no key. The lesson from the thesis: other academic databases had tight rate limits, needed commercial access, or had usage policies against LLM processing.",
        ],
      ),
    ),
  ],

  system: () => [
    section(
      "Stack",
      facts(
        ["Pipeline", "Python 3.14 library (uv): SQLAlchemy 2 + psycopg 3, pyalex, openai"],
        ["Database", "PostgreSQL 16 in Docker, with pgvector 0.7 and pg_bestmatch.rs (BM25)"],
        ["This page", "FastAPI, server-sent events, plain JavaScript with SVG and canvas; no framework and no CDN, so replays work offline"],
        ["Models", "text-embedding-3-large (1,024 dimensions), gpt-6-luna for comparisons, gpt-6-sol for summaries. In 2024: gpt-4o-mini and gpt-4o."],
      ),
    ),
    section(
      "Data model",
      ul(
        [b("publication: "), "OpenAlex id, title, first 3 authors, date, abstract, embedding vector(1024), BM25 sparse vector, and keys for duplicate detection."],
        [b("openalex_topic: "), "4,516 rows with name, description, keywords and embedding (plus the domain, field and subfield levels above them). Embedded once from “Topic: …; Description: …; Keywords: …”, for $0.07."],
        [b("Vector search "), "is an exact scan without an index: milliseconds at this size (the thesis corpus had ~50k papers)."],
        [b("BM25: "), "pg_bestmatch.rs keeps corpus statistics and stores each abstract as a sparse vector; the query becomes one too, and the score is an inner product. Rebuilt after each ingest."],
        [b("Duplicates: "), "OpenAlex has duplicate records (about 12 % of the papers fetched here): the same title or abstract under different ids. Dropped at ingest, as are “abstracts” under 20 words."],
      ),
    ),
    section(
      "What changed for this demo (2026)",
      ul(
        "Runs again on current Python and models (the thesis's summary model, gpt-4o-2024-05-13, retires in October 2026).",
        "Own parallel implementation of the LLM ranking: a query takes 23 s instead of 70 s.",
        "Duplicates filtered; strict JSON output for the summaries.",
        "Every step is traced (time, tokens, $) as events; they drive this page, and each run is recorded for replay.",
        "Refactored, with a test harness showing identical prompts, events and database rows before and after.",
      ),
    ),
    section(
      "Two fixes merged upstream during the thesis",
      ul(
        [b("pyalex PR #35 "), "(March 2024, release 0.14): support for Topics, Domains, Fields and Subfields. OpenAlex had replaced its Concepts taxonomy with Topics mid-thesis, and the Python client didn't know them yet; the topic routing is built on this."],
        [b("pg_bestmatch.rs PR #15 "), "(July 2024): a one-line fix for a wrong table name that made the BM25 refresh fail; the pipeline runs that refresh after every ingest."],
      ),
    ),
    section(
      "If asked",
      qa(
        ["Why Postgres, not a vector database?", "Vectors, keyword index and metadata live in one database with one query language. At this size an exact vector scan takes milliseconds; a dedicated vector store would matter at millions of vectors."],
        ["What does it cost?", "Per query about $0.03 and 25 s. Ingest about $0.09 per 2,000 new papers (the embeddings). The topic embeddings once: $0.07."],
        ["What's next for the stack?", "After today: Postgres 18, current pgvector, and VectorChord-bm25, because pg_bestmatch.rs is no longer maintained."],
      ),
    ),
  ],

  topics: (run) => {
    const sims = run.topics.map((t) => t.similarity);
    const notes = {
      coral_reefs:
        "Strong, tight matches (reef resilience, ocean acidification, marine ecology). #2, “Species Distribution Modeling and Climate Change Impacts”, sits apart on the map: it matches another part of the description, modelling future changes, not the reefs.",
      rag_hallucinations:
        "Weak matches, scattered across computer science plus one psychology topic: the 2024 taxonomy has no topic for LLMs or RAG. The later steps still find relevant papers, because they search the papers' own abstracts.",
    };
    return [
      section(
        "How it works",
        p("The description is embedded (text-embedding-3-large, 1,024 dimensions) and compared by cosine similarity with the embeddings of all 4,516 OpenAlex topics, an exact search in Postgres. The 10 most similar topics are kept."),
        p("At ingest, OpenAlex returns the newest papers (since 1 Jan 2025, with an abstract) whose primary topic is one of the 10. The next steps search every paper fetched so far, not only this query's."),
      ),
      section(
        "How far back the pool reaches",
        p("The fetch takes the newest papers up to a count, not a date. So the time window depends on how busy a topic is."),
        ul(
          [b("The demo pool: "), "fetched beforehand, the newest 2,000 papers over the 10 topics of each demo query together: published 30 Aug – 26 Sep 2026, about four weeks."],
          [b("A live fetch "), "(the menu next to “Run live”): the newest 100–1,000 papers of each topic, so small topics aren't crowded out by big ones. Measured on the RAG query's topics: 100 per topic reach back 2 days to 7 weeks; 500 per topic 11 days to 6½ months; 1,000 per topic 3 weeks (NLP) to 15 months (a small topic)."],
          [b("Why it matters: "), "a very specific description can find nothing relevant simply because the window holds no such paper. The search isn't failing then; the haystack has no needle. Fetch more per topic for such a description."],
          [b("Cost: "), "the OpenAlex fetch is free (1,000 per topic: ~16 s); embedding the new abstracts costs about $0.045 per 1,000 papers, and the keyword index is rebuilt afterwards (seconds, growing with the pool)."],
        ),
        run.fetched
          ? facts(
              ["This run fetched", `${fmt.int(run.fetched.count)} papers${run.fetched.per_topic ? ` (${fmt.int(run.fetched.per_topic)} per topic)` : ""}, back to ${run.fetched.oldest ?? "–"}`],
              ...(run.fetched.topics ?? []).map((topic) => [run.topics.find((t) => t.id === topic.id)?.name ?? `T${topic.id}`, `${fmt.int(topic.count)} papers, back to ${topic.oldest ?? "–"}`]),
            )
          : null,
        run.pool ? p(`Searched in this run: ${fmt.int(run.corpus)} papers published ${run.pool.oldest} to ${run.pool.newest}.`) : null,
      ),
      section(
        "In this run",
        sims.length
          ? ul(`Similarities ${Math.min(...sims).toFixed(2)}–${Math.max(...sims).toFixed(2)}; best match: ${run.topics[0].name}.`, notes[run.meta.query_name])
          : noRun(),
      ),
      section(
        "Reading the map",
        p("A 2D projection (UMAP) of the 4,516 topic embeddings: nearby dots are similar topics. Distances between far-apart clusters mean little."),
        p("The description itself isn't on the map: it's about equally similar to all 10 matches, and less similar to them than they are to each other (coral: 0.56–0.64 to the description, 0.63–0.86 among the matches). No point on the map would place it honestly."),
      ),
      section(
        "Measured later (2026): what improves the routing?",
        p("Setup: ~195 random recent papers, twice; the target is the topics of each paper's references (where its author wants to look)."),
        ul(
          "Rewriting the description into the topics' format (HyDE-style): no gain (±0).",
          "Splitting it into facets: no gain; fusing the facets with RRF: 7–10 points worse.",
          "Richer content helps: the paper's abstract instead of a description: +4 points.",
          "The biggest lever is the cut: 20 topics instead of 10 cover 7–9 points more. Ranks 10 and 11 differ by a median 0.002 in similarity, so the cut at 10 is arbitrary.",
        ),
      ),
      section(
        "If asked",
        qa(
          ["Why topics at all?", "Embedding all 327 M abstracts in OpenAlex is out of reach for a thesis. Topics narrow the fetch to a few thousand recent papers that cost cents to embed. The price: this coarse step can miss papers, and it's blind to fields newer than the taxonomy."],
          ["Why are the similarities so low (0.4–0.6)?", "Cosine similarities between texts of different kinds (a personal paragraph vs a topic's short description) stay far from 1. Only the order matters here."],
          ["Is the taxonomy up to date?", "OpenAlex introduced Topics in early 2024, replacing Concepts. As of September 2026 it's still the same 4,516 topics, with none for LLMs or RAG."],
        ),
      ),
    ];
  },

  retrieval: (run) => {
    const { bm25, semantic, hybrid } = run.rankings;
    let stats = null;
    if (bm25 && semantic && hybrid) {
      const inBm25 = new Set(bm25.map((r) => r.id));
      const inDense = new Set(semantic.map((r) => r.id));
      const both = hybrid.filter((r) => inBm25.has(r.id) && inDense.has(r.id)).length;
      const denseOnly = hybrid.filter((r) => inDense.has(r.id) && !inBm25.has(r.id)).length;
      const bm25Only = hybrid.filter((r) => inBm25.has(r.id) && !inDense.has(r.id)).length;
      const top10 = bm25.slice(0, 10).filter((r) => semantic.slice(0, 10).some((s) => s.id === r.id)).length;
      stats = ul(
        `Both searches over ${fmt.int(run.corpus ?? 0)} papers: ${fmt.seconds(run.groups.retrieval.duration)}, no API call.`,
        `The two top-10 lists share ${top10} paper${top10 === 1 ? "" : "s"}.`,
        `Of the ${hybrid.length} candidates, ${both} were in both top-100 lists, ${denseOnly} only in the meaning search, ${bm25Only} only in the keyword search.`,
      );
    }
    return [
      section(
        "How it works",
        ul(
          [b("Keyword (BM25): "), "the classic search-engine score. A word counts more when it's rare in the corpus and frequent in the paper, with diminishing returns. Computed in Postgres by pg_bestmatch.rs."],
          [b("Meaning (dense): "), "cosine similarity between the description's embedding and each abstract's embedding (pgvector)."],
          [b("Blend: "), "each search returns its top 100; scores are min-max normalized (its #1 = 1, its #100 = 0) and added with weights 0.8 meaning + 0.2 keyword. The top 50 go to the LLM: 10 × the 5 results wanted."],
        ),
      ),
      section("In this run", stats ?? noRun()),
      section(
        "Why both",
        p("Keywords catch exact terms: names, acronyms, jargon. Meaning catches paraphrases. Each alone fails in its own way: in the thesis's coral example, keyword search alone brought up land-based false positives (arthropods, leafhoppers); the blend promoted a paper that was #10 by keyword and #6 by meaning."),
      ),
      section(
        "The weights, and “What if?”",
        p("0.8 / 0.2 comes from Mandikal & Mooney (“Sparse Meets Dense”), not tuned here. Min-max has a flaw: each list's #1 gets 1, however weak it is. Reciprocal rank fusion (RRF) uses ranks only."),
        p("“What if?” re-blends the recorded scores in the browser and says how many of the 50 candidates the LLM actually saw would stay. For coral, meaning weight 0.3 keeps only 23 of 50; RRF keeps 30. The LLM's result exists only for the recorded pool."),
      ),
      section(
        "If asked",
        qa(
          ["Why not give the LLM every paper?", `Reading every abstract once would be about 1.1 M tokens per query (${fmt.int(run.corpus ?? 3509)} abstracts × ~320 tokens), and thousands of comparisons. The search narrows it to 50 in milliseconds.`],
          ["No vector index?", "An exact scan over a few thousand vectors takes milliseconds. An HNSW index would matter at millions."],
          ["The bars under the combined column?", "The combined score, split into its meaning part (dark) and keyword part (light). Full length would be #1 in both searches. Hover a row for the arithmetic."],
        ),
      ),
    ];
  },

  rerank: (run) => {
    const g = run.groups.rerank;
    return [
      section(
        "How it works",
        p("Setwise prompting (Zhuang et al. 2023): the prompt holds the description and three abstracts, labelled A, B and C, and asks for the label of the most relevant one. The answer is a single token."),
        p("Heapsort organizes these comparisons. Build a heap, a tree in which each paper beats its two children; then take the top off five times, repairing the tree after each. For the top 5 of 50 that's about 55 comparisons. Ranks 6–50 are never computed, which is what makes top-k cheap."),
        p("Model: gpt-6-luna with reasoning off and temperature 0, at $0.10 / $0.50 per million tokens in / out. The thesis used gpt-4o-mini."),
      ),
      section(
        "In this run",
        run.rerank
          ? facts(
              ["LLM calls", `${run.rerank.calls}, of which ${run.rerank.used} were on the path the sort took`],
              ["Time", fmt.seconds(g.duration)],
              ["Cost", fmt.usd(g.cost)],
              ["Tokens", `${fmt.int(g.input)} in, ${fmt.int(g.output)} out`],
            )
          : noRun(),
      ),
      section(
        "Why it's fast (added in 2026)",
        p("The original library ran the comparisons one after another: 45 s. Now independent branches of the tree run in parallel (25 s), and comparisons are asked ahead (dashed lines): 12–15 s, with about 90 calls instead of 55."),
        p("Asking ahead: while a paper waits for its comparison with its two children, it can only stay, or move down into one of them. So the comparisons it would face one level down are sent right away. At most one of them is needed, often neither; unneeded calls still queued are cancelled, sent ones are paid. The extra calls cost about $0.004 per query."),
        p("Unit tests check that the decisions are identical to the original library's, for many tree sizes and settings."),
      ),
      section(
        "If asked",
        qa(
          ["Why not have the LLM score each paper 1–10?", "Scores from separate calls aren't calibrated against each other; comparing papers side by side is more reliable. The setwise paper reports a better cost/quality trade-off than pointwise, pairwise and listwise prompting."],
          ["Why not all 50 in one prompt?", "About 16k tokens of abstracts in one prompt, and an answer of 50 labels that must be parsed and can be inconsistent. Three at a time keeps each prompt small and each answer one token."],
          ["Is it deterministic?", "No: even at temperature 0 the model isn't fully deterministic, and the order of the top 5 varies slightly between runs. That's one reason the demo can replay a recorded run."],
          ["A cross-encoder or a rerank API instead?", "Cheaper and faster; not compared in the thesis. On the citation measure, it would only need to beat the LLM's +10\u00a0% over hybrid search: a good baseline to add."],
          ["Does it work?", EVAL_ANSWER],
        ),
      ),
    ];
  },

  summaries: (run) => {
    const done = [...run.summaries.values()];
    return [
      section(
        "How it works",
        p("One call per paper, with the description and the abstract, all in parallel. The prompt has three parts:"),
        ul(
          [b("Task: "), "tailor the summary to the reader's interests, use their terminology, don't copy sentences, be brief, don't address the reader."],
          [b("Worksheet: "), "a JSON structure to fill in: key findings, methods, conclusions, related aspects, significance, then the final answer."],
          [b("Instruction "), "to fill in the worksheet step by step."],
        ),
        p("Strict JSON-schema output keeps the key order, so the model writes the worksheet before the final answer. Only the final answer is meant for the reader. (2026; the thesis parsed JSON out of free text.)"),
        p("Model: gpt-6-sol with low reasoning effort, about $0.007 per summary. The thesis used gpt-4o."),
      ),
      section(
        "In this run",
        done.length ? p(`${done.length} summaries in ${fmt.seconds(run.groups.summaries.duration)}, ${fmt.usd(run.groups.summaries.cost)}.`) : noRun(),
      ),
      section(
        "Where the worksheet comes from",
        p("Self-Discover (Zhou et al. 2024): the model picks general reasoning modules (“break it into steps”, “critical thinking”, …), adapts them to the task, and turns them into a step-by-step plan in JSON. I ran it offline on example pairs, merged the generated plans by hand (dropping steps that leaked example content or weren't actionable, like “be brief”), and froze the result. So Self-Discover was a prompt-engineering tool, not part of the runtime."),
        p("The step numbers skip 4: the prompt is kept word for word as evaluated in the thesis."),
      ),
      section(
        "How good are they? (thesis user study, 1–5)",
        facts(["Fluency", "4.25"], ["Coherence", "≈ 4.06"], ["Informativeness", "≈ 3.19"], ["Faithfulness to the abstract", "3.00 (my own rating: 3.93)"], ["Focus on the reader's interests", "≈ 2.89"]),
        p("n = 16 ratings from the participating researchers (Fraunhofer FOKUS / DCAITI)."),
      ),
      section(
        "The main failure: tailoring vs. faithfulness",
        p("4 of 30 summaries overstated how relevant the paper was to the reader: an obstacle-detection paper “aligns closely with HD-map creation”. Another recast a V2X forensics study as “improving localization”. The instruction to relate the paper to the reader invites this, much like sycophancy."),
        p("The 2026 model hedges more on its own, e.g. “does not measure whether responses are grounded in retrieved records”."),
        p("What I'd add now: an LLM-as-judge check for faithfulness, regenerating summaries that fail it; and the full text instead of only the abstract."),
      ),
    ];
  },

  results: (run) => [
    section(
      "Reading the chart",
      ul(
        "Each line is one of the final top 5, colored by its final rank, as in the other views.",
        "The y-axis is its rank in each search, on a log scale; “none” means it wasn't in that search's top 100.",
        "Every final paper has a combined rank of at most 50: only those 50 reach the LLM.",
        "The LLM only ranks its top 5; heapsort never computes ranks 6–50.",
      ),
    ),
    section("Better, or just different?", p(EVAL_ANSWER)),
    section(
      "If asked",
      qa(
        ["How recent are these?", "Weeks old: the demo pool holds papers published 30 Aug – 26 Sep 2026, fetched newest first."],
        ["Can users give feedback?", "Not yet. The thesis proposed ratings and “not relevant because …” feedback as future work."],
        ["Why only 3 summaries?", "A setting of the demo, to keep a run short and cheap. Each summary costs about $0.007."],
      ),
    ),
    run.rerank ? null : section("In this run", noRun()),
  ],

  takeaways: () => [
    section(
      "What I'd do differently now",
      ul(
        [b("Negation: "), "a description saying “not UAVs” still got a UAV paper at #3. Embeddings and BM25 can't express NOT; an LLM could extract exclusions and filter on them."],
        [b("Faithfulness: "), "an LLM-as-judge check on each summary, regenerating those that fail."],
        [b("Blending: "), "rank fusion (RRF) or tuned weights instead of min-max with untuned 0.8 / 0.2."],
        [b("The topic cut: "), "20 topics instead of 10 (measured: +7 to +9 points), or a threshold instead of a fixed cut."],
        [b("Feedback: "), "move the query toward liked papers (Rocchio), and learn from “not relevant because …”."],
        [b("Evaluation: "), "a proper test set of paragraph-long descriptions with relevance labels."],
      ),
    ),
    section(
      "For RAG in general (hypotheses from this project, not measured on our data)",
      ul(
        [b("Rewrite for missing content, not style: "), "making the description look like a topic didn't help; more content did."],
        [b("Multi-query + RRF "), "is a popular default, but here fusing sub-queries cost 7–10 points. Measure before relying on it."],
        [b("The top-k cut sits on a plateau: "), "similarities around the cut differ by ~0.002. Retrieve wide and cheap, then rerank the few."],
        [b("A weak best match "), "is a (weak) signal that the answer isn't in the knowledge base: RAG's best topic scored 0.45, coral's 0.64. Thresholds need calibrating per embedding model."],
        [b("Label-free evaluation: "), "wherever a “right source” exists (say, the article a human agent used to resolve a ticket), a retrieval change can be scored offline before it ships."],
      ),
    ),
    section(
      "The 2026 revival, in steps",
      ul(
        "Got it running again: a broken virtualenv, a database image that no longer built, models about to retire.",
        "Replaced the ranking library with an own parallel version, verified to make identical decisions.",
        "Filtered OpenAlex duplicates (~12 % of the papers), and traced every step.",
        "Moved to uv and Python 3.14; refactored with a harness that compared prompts, events and database rows before and after.",
        "Built this page on the trace events. All of it pair-programmed with Claude Code.",
      ),
    ),
  ],
};

export function details(step, run) {
  const build = CONTENT[step];
  return {
    title: `${STEP_LABELS[step]}: details`,
    body: h("div", { class: "details" }, ...(build ? build(run) : []).filter(Boolean)),
  };
}
