// Step "The idea": the problem, the idea, and the pipeline as a funnel. The funnel's bars are the number of papers left
// after each stage (log scale); between them, each stage with the question it answers, how, and its time and cost (from
// the run on screen, or typical values before a run). A click on a stage opens its view.

import { GROUPS, STEP_LABELS, dayRange, fmt, h, s, viewHead } from "./util.js";

// OpenAlex, queried 2026-09-27: all works, and works published in 2025
const OPENALEX_WORKS = 327_331_874;
const WORKS_2025 = 14_654_407;
const DEMO_POOL = 3509;

const STAGES = {
  topics: { question: "Which research areas?", how: "Match the description to 10 of 4,516 OpenAlex topics, and fetch their newest papers (done beforehand for the demo)." },
  retrieval: { question: "Which papers could fit?", how: "Keyword and meaning search over every paper in the pool, in the database." },
  rerank: { question: "Which fit best?", how: "The LLM compares three abstracts at a time, and picks the best one." },
  summaries: { question: "Why should I read it?", how: "The LLM writes a short summary for this reader." },
};

// a typical run (the recorded RAG query), shown until a run is on screen
const TYPICAL = {
  topics: { duration: 0.93, cost: 0.000014, calls: 1 },
  retrieval: { duration: 0.04, cost: 0, calls: 0 },
  rerank: { duration: 15.3, cost: 0.0106, calls: 92 },
  summaries: { duration: 7.8, cost: 0.0206, calls: 3 },
};

const BAND = 140; // px: height of the funnel's bar band
const barHeight = (count) => Math.max(8, (Math.log10(Math.max(count, 1)) / Math.log10(OPENALEX_WORKS)) * BAND);

export class IntroView {
  constructor(root, onSelect) {
    this.root = root;
    this.onSelect = onSelect;
    this.pool = DEMO_POOL;
    this.funnel = h("div", { class: "funnel" });
    this.funnelNote = h("p", { class: "funnel-note" });
    root.append(
      viewHead(
        "intro",
        "Papers worth reading, from a description of your research",
        "Late 2023, a year after ChatGPT came out, my Master's thesis asked how LLMs could help researchers in their daily work. One chore every researcher knows: keeping up with what's new in their own field.",
      ),
      h(
        "div",
        { class: "intro-grid" },
        h(
          "section",
          { class: "intro-card" },
          h("h3", { class: "kicker" }, "The problem"),
          h(
            "div",
            { class: "big-stat" },
            h("span", { class: "big-number" }, `${(WORKS_2025 / 1e6).toFixed(1)} million`),
            h("span", { class: "big-label" }, `works published in 2025, ~${fmt.int(Math.round(WORKS_2025 / 365 / 1000) * 1000)} a day`, h("span", { class: "source" }, " (OpenAlex)")),
          ),
          h("p", {}, "Search engines for research want keywords and Boolean logic, like this Scopus query:"),
          h("pre", { class: "boolean" }, '( SUBJAREA ( comp ) OR SUBJAREA ( engi ) )\nAND ( KEY ( "vehicle to vehicle communications" )\n      OR TITLE-ABS-KEY ( v2x ) )\nAND PUBYEAR > 2023'),
          h("p", {}, "And the newest papers have no citations yet, so “most cited” can't find them."),
        ),
        h(
          "section",
          { class: "intro-card idea" },
          h("h3", { class: "kicker" }, "The idea"),
          h("p", { class: "idea-line" }, h("span", { class: "idea-tag" }, "In"), h("span", {}, "a paragraph about your research, the way you'd describe it to a colleague.")),
          h("p", { class: "idea-line" }, h("span", { class: "idea-tag" }, "Out"), h("span", {}, "the newest papers that match it, ranked, each with a short summary of why it matters to you.")),
          h("p", { class: "idea-note" }, "No keywords, no citation counts, no click history."),
          h("p", { class: "idea-note" }, "Master's thesis at TU Berlin: topic chosen late 2023, submitted August 2024, brought back to life for this demo."),
        ),
        // where the papers come from, and the topic taxonomy that the first stage routes into (facts: help.openalex.org)
        h(
          "section",
          { class: "intro-card openalex" },
          h("h3", { class: "kicker" }, "Where the papers come from"),
          h(
            "p",
            {},
            h("strong", {}, "OpenAlex"),
            `: an open index of the world's research, successor to Microsoft Academic Graph. ${Math.round(OPENALEX_WORKS / 1e6)} M works with authors, venues and citations; free data (CC0), free API.`,
          ),
          h("p", {}, "It files every paper under research topics, in four levels:"),
          h(
            "div",
            { class: "taxonomy", role: "img", "aria-label": "4 domains, 26 fields, 252 subfields, 4,516 topics" },
            ...[
              ["4", "domains"],
              ["26", "fields"],
              ["252", "subfields"],
              ["4,516", "topics"],
            ].flatMap(([count, level], i) => [i ? h("span", { class: "tax-arrow", "aria-hidden": "true" }, "→") : null, h("span", { class: "tax-level" }, h("strong", {}, count), ` ${level}`)]),
          ),
          h("p", { class: "taxonomy-example" }, "e.g. Physical Sciences → Computer Science → AI → Natural Language Processing"),
          h(
            "p",
            { class: "idea-note" },
            "Topics are clusters of the citation network, named by an LLM; a classifier files each paper under them. ",
            h("a", { class: "ext-link", href: "https://help.openalex.org/data/topics/", target: "_blank", rel: "noopener" }, "How OpenAlex topics work ↗"),
          ),
        ),
      ),
      h("section", { class: "funnel-section" }, h("h3", { class: "kicker" }, "How: a funnel, from cheap to expensive"), this.funnel, this.funnelNote),
    );
    this.render();
  }

  reset(run) {
    this.run = run;
    this.render();
  }

  // the pool before a run: the corpus in the database, and its publication dates
  setCorpus(rows, range) {
    this.pool = rows;
    this.poolRange = range;
    if (!this.run?.corpus) this.render();
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
    const run = this.run;
    const live = run?.finished && !run.error;
    const n = run?.meta?.n ?? 5;
    const range = run?.pool ?? (run?.corpus ? null : this.poolRange);
    const levels = [
      { count: OPENALEX_WORKS, text: `${Math.round(OPENALEX_WORKS / 1e6)} M`, label: "works in OpenAlex" },
      {
        count: run?.corpus ?? this.pool,
        text: fmt.int(run?.corpus ?? this.pool),
        label: "recent papers in the pool",
        sub: range ? `published ${dayRange(range.oldest, range.newest)}` : null,
      },
      { count: n * 10, text: `${n * 10}`, label: "candidates" },
      { count: n, text: `${n}`, label: "best matches" },
      { count: run?.meta?.summaries ?? 3, text: `${run?.meta?.summaries ?? 3}`, label: "summaries" },
    ];
    const heights = levels.map((level) => barHeight(level.count));
    const cells = [];
    levels.forEach((level, i) => {
      cells.push(
        h(
          "div",
          { class: "f-level" },
          h("div", { class: "f-count" }, level.text),
          h("div", { class: "f-band" }, h("div", { class: "f-bar", style: { height: `${heights[i]}px` } })),
          h("div", { class: "f-label" }, level.label, level.sub ? h("div", { class: "f-sub" }, level.sub) : null),
        ),
      );
      const step = GROUPS[i];
      if (!step) return;
      const metrics = live ? run.groups[step] : TYPICAL[step];
      const [a, b] = [heights[i], heights[i + 1]];
      // the connector from this level's bar to the next one's, both centered on the band
      const shape = s(
        "svg",
        { class: "f-shape", viewBox: `0 0 100 ${BAND}`, preserveAspectRatio: "none", "aria-hidden": "true" },
        s("polygon", { points: `0,${(BAND - a) / 2} 100,${(BAND - b) / 2} 100,${(BAND + b) / 2} 0,${(BAND + a) / 2}` }),
      );
      cells.push(
        h(
          "button",
          { class: "f-stage", type: "button", onclick: () => this.onSelect(step), title: `Show ${STEP_LABELS[step]}` },
          h("div", { class: "f-stage-name" }, h("span", { class: "view-num" }, i + 1), STEP_LABELS[step]),
          h("div", { class: "f-band" }, shape),
          h("div", { class: "f-question" }, STAGES[step].question),
          h("div", { class: "f-how" }, STAGES[step].how),
          h("div", { class: "f-metrics" }, `${fmt.seconds(metrics.duration)} · ${metrics.calls ? `${fmt.usd(metrics.cost)}, ${metrics.calls} LLM call${metrics.calls === 1 ? "" : "s"}` : "no LLM"}`),
        ),
      );
    });
    this.funnel.replaceChildren(...cells);
    this.funnelNote.textContent = `Bars: papers left after each stage (log scale). Time and cost: ${live ? "the run on screen" : "a typical run"}. The LLM reads 50 abstracts per query, not ${fmt.int(levels[1].count)}.`;
  }
}
