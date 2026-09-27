// Step "Takeaways": what I learned, as a few cards. Numbers come from the run on screen where it has them; the thesis
// numbers are from its evaluation (see the details drawer, and docs/thesis_project.md §4 in the prep notes).

import { fmt, h, viewHead } from "./util.js";

export class TakeawaysView {
  constructor(root) {
    this.grid = h("div", { class: "takeaways" });
    root.append(viewHead("takeaways", "What I took away", null), this.grid);
    this.render();
  }

  reset(run) {
    this.run = run;
    this.render();
  }

  show(run) {
    this.run = run;
    this.render();
  }

  finish(run) {
    this.run = run;
    this.render();
  }

  render() {
    const run = this.run?.finished && !this.run.error ? this.run : null;
    const g = run?.groups;
    const searchTime = g ? fmt.seconds(g.retrieval.duration) : "0.04 s";
    const pool = fmt.int(run?.corpus ?? 3509);
    const total = run ? `${fmt.seconds(run.t)} and ${fmt.usd(run.cost)}` : "about 24 s and $0.03";
    const rerankCalls = g ? g.rerank.calls : 92;
    const tokensIn = g ? g.rerank.input : 102766;
    const tokensOut = g ? g.rerank.output : 552;
    const cards = [
      {
        title: "Spend the LLM where it counts",
        body: `Searching all ${pool} papers takes ${searchTime} in the database and costs nothing. The LLM only reads the 50 candidates and writes 3 summaries: ${total} per query.`,
        tag: "architecture",
      },
      {
        title: "Ask the model small questions",
        body: `“Which of these three fits best?” with a one-letter answer: ${rerankCalls} calls, ${fmt.int(tokensIn)} tokens in, ${fmt.int(tokensOut)} out. Input is cheap; the short answer keeps each call at about a second, and many run in parallel.`,
        tag: "LLM as judge",
      },
      {
        title: "Tailoring pushes toward flattery",
        body: "Asked to relate a paper to the reader, the model sometimes invented relevance: an obstacle-detection paper “aligns closely with HD-map creation”. Participants rated faithfulness 3.0 of 5 (I had rated it 3.9). The same risk applies to any assistant that personalizes its answers.",
        tag: "user study, n = 16",
      },
      {
        title: "The builder is a lenient judge",
        body: "I judged every top-5 result as relevant: each touched a topic of the description. The researchers themselves rated relevance 2.89 of 5: roughly “some relevance, but its focus is not central to your research”.",
        tag: "user study, n = 19",
      },
      {
        title: "No labels? Borrow them",
        body: "No dataset has paragraph-long interest descriptions with relevance labels. So a paper's abstract is the query, and the papers it cites are the right answers. Hybrid search's top 10 held 6.7 cited papers, the LLM's 7.4. My thesis said +60\u00a0% for the LLM: an eval bug had lost the hybrid order. It's about +10\u00a0%.",
        tag: "evaluation, 30 queries, re-run 2026",
      },
      {
        title: "Two years later, with an AI agent",
        body: "The code had been dormant since 2024: old dependencies, retired models. Over a weekend with Claude Code it runs again on current models, 3× faster (70 s → 23 s), refactored with tests that show the behaviour unchanged, and it got this demo page.",
        tag: "2026",
      },
    ];
    this.grid.replaceChildren(
      ...cards.map((card, i) =>
        h("article", { class: "takeaway" }, h("div", { class: "takeaway-num" }, i + 1), h("h3", {}, card.title), h("p", {}, card.body), h("div", { class: "takeaway-tag" }, card.tag)),
      ),
    );
  }
}
