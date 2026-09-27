// Step "Results": what the reader gets, the final top k with their summaries, and where each one came from: its rank
// in the keyword search, the meaning search, the combined search and the LLM's ranking, as a slope chart on a log scale.
// Colors are the final ranks, the same as in every other view.

import { attachTip, h, openOnClick, paperLink, rankColor, s, viewHead } from "./util.js";

const AXES = [
  { key: "bm25", label: "Keyword" },
  { key: "semantic", label: "Meaning" },
  { key: "hybrid", label: "Combined" },
  { key: "llm", label: "LLM" },
];
const TICKS = [1, 2, 5, 10, 20, 50, 100];
const MAX_RANK = 100;

export class ResultsView {
  constructor(root) {
    this.list = h("ol", { class: "result-list" });
    this.chart = s("svg", { class: "journey", role: "img", "aria-label": "Rank of each of the final top 5 in each search" });
    this.callout = h("div", { class: "journey-callout" });
    root.append(
      viewHead(
        "results",
        "What the reader gets",
        "The best matches among the newest papers, each with a summary in the reader's terms (the top 3, in this demo). And where they came from: the LLM often picks papers that the cheap searches ranked far down.",
      ),
      h(
        "div",
        { class: "results-grid" },
        h("section", { class: "results-list-panel" }, this.list),
        h(
          "section",
          { class: "journey-panel" },
          h("h3", {}, "Where the top 5 came from"),
          h("p", { class: "note" }, "Rank of each paper in each search; higher is better (log scale)."),
          this.chart,
          this.callout,
        ),
      ),
    );
    new ResizeObserver(() => this.draw()).observe(this.chart);
    this.highlighted = null;
  }

  reset(run) {
    this.run = run;
    this.render();
  }

  onEvent(event, run) {
    this.run = run;
    if (["reranked", "summary"].includes(event.type)) this.render();
  }

  show(run) {
    this.run = run;
    this.render();
  }

  top() {
    return this.run?.rerank ? this.run.top : [];
  }

  ranksOf(id) {
    const position = (list) => {
      const i = list?.findIndex((result) => result.id === id) ?? -1;
      return i >= 0 ? i + 1 : null;
    };
    const { rankings } = this.run;
    return { bm25: position(rankings.bm25), semantic: position(rankings.semantic), hybrid: position(rankings.hybrid), llm: this.run.finalRank(id) };
  }

  render() {
    const top = this.top();
    if (!top.length) {
      this.list.replaceChildren(h("li", { class: "empty" }, "The LLM's top 5 appear here once the ranking is done."));
      this.callout.replaceChildren();
      this.draw();
      return;
    }
    const run = this.run;
    const show = (rank) => (rank ? `#${rank}` : "–");
    this.list.replaceChildren(
      ...top.map((id, i) => {
        const work = run.work(id);
        const ranks = this.ranksOf(id);
        const summary = run.summaries.get(id)?.summary;
        const meta = [work.authors?.join(", "), work.publication_date].filter(Boolean).join(" · ");
        const item = h(
          "li",
          { class: "result", "data-id": id },
          h("span", { class: "result-rank" }, h("span", { class: "marker", style: { background: rankColor(i + 1) } }), `#${i + 1}`),
          h(
            "div",
            { class: "result-main" },
            h("div", { class: "result-title" }, paperLink(id, work.title ?? `W${id}`)),
            meta ? h("div", { class: "result-meta" }, meta) : null,
            summary
              ? h("p", { class: "result-summary" }, summary)
              : h("p", { class: "result-summary muted" }, i < (run.meta.summaries ?? 3) ? "Summary on its way…" : "No summary: the demo summarizes the top 3."),
            h("div", { class: "result-journey" }, `keyword ${show(ranks.bm25)} · meaning ${show(ranks.semantic)} · combined ${show(ranks.hybrid)} → LLM #${i + 1}`),
          ),
        );
        item.addEventListener("mouseenter", () => this.highlight(id));
        item.addEventListener("mouseleave", () => this.highlight(null));
        return item;
      }),
    );
    this.renderCallout(top);
    this.draw();
  }

  renderCallout(top) {
    const ranks = top.map((id) => this.ranksOf(id));
    const deepest = ranks.reduce((best, r) => (r.hybrid > (best?.hybrid ?? 0) ? r : best), null);
    const inTop10 = ranks.filter((r) => r.hybrid && r.hybrid <= 10).length;
    const noKeyword = ranks.filter((r) => !r.bm25).length;
    const facts = [];
    if (deepest && deepest.hybrid > 10) facts.push(`The LLM's #${deepest.llm} was only #${deepest.hybrid} of 50 in the combined search.`);
    facts.push(`${inTop10} of the LLM's top ${top.length} were in the combined top 10.`);
    if (noKeyword) facts.push(`${noKeyword} ${noKeyword === 1 ? "wasn't" : "weren't"} in the keyword search's top 100 at all.`);
    this.callout.replaceChildren(...facts.map((fact) => h("p", {}, fact)));
  }

  highlight(id) {
    this.highlighted = id;
    for (const path of this.chart.querySelectorAll(".journey-line")) path.classList.toggle("dim", id !== null && path.dataset.id !== String(id));
    for (const item of this.list.querySelectorAll(".result")) item.classList.toggle("hl", id !== null && item.dataset.id === String(id));
  }

  draw() {
    const svg = this.chart;
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    svg.replaceChildren();
    if (!width || !height) return;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const top = this.top();
    const margin = { left: 44, right: 150, top: 30, bottom: 12 };
    const missingGap = 34; // the "not in its top 100" row, below the scale
    const plotHeight = height - margin.top - margin.bottom - missingGap;
    const plotWidth = width - margin.left - margin.right;
    const x = (i) => margin.left + (i / (AXES.length - 1)) * plotWidth;
    const y = (rank) => (rank ? margin.top + (Math.log(rank) / Math.log(MAX_RANK)) * plotHeight : margin.top + plotHeight + missingGap);

    const grid = s("g", { class: "journey-grid" });
    for (const tick of TICKS) {
      grid.append(
        s("line", { x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick) }),
        s("text", { class: "tick", x: margin.left - 8, y: y(tick) + 4, "text-anchor": "end" }, `#${tick}`),
      );
    }
    grid.append(s("text", { class: "tick", x: margin.left - 8, y: y(null) + 4, "text-anchor": "end" }, "none"));
    // between two axes, clear of the lines' end dots
    grid.append(s("text", { class: "tick missing", x: (x(1) + x(2)) / 2, y: y(null) - 10, "text-anchor": "middle" }, "none: not in that search's top 100"));
    AXES.forEach((axis, i) => {
      grid.append(
        s("line", { class: "axis-line", x1: x(i), x2: x(i), y1: margin.top - 6, y2: y(null) + 6 }),
        s("text", { class: "axis-label", x: x(i), y: margin.top - 14, "text-anchor": "middle" }, axis.label),
      );
    });
    svg.append(grid);
    if (!top.length) return;

    // one group per paper (line, dots, label), so that hovering one dims the others as a whole
    const lines = s("g");
    top.forEach((id, i) => {
      const ranks = this.ranksOf(id);
      const points = AXES.map((axis, a) => [x(a), y(ranks[axis.key])]);
      const color = rankColor(i + 1);
      const work = this.run.work(id);
      const tip = () =>
        h(
          "div",
          {},
          h("div", { class: "tip-title" }, work.title),
          h("div", { class: "tip-meta" }, AXES.map((axis) => `${axis.label} ${ranks[axis.key] ? `#${ranks[axis.key]}` : "not in its top 100"}`).join(" · ")),
          h("div", { class: "tip-hint" }, `Click to open W${id} in OpenAlex ↗`),
        );
      const d = points.map(([px, py], p) => `${p ? "L" : "M"}${px},${py}`).join(" ");
      const line = s("g", { class: "journey-line", "data-id": id }, s("path", { class: "hit", d }), s("path", { class: "stroke", d, style: `stroke: ${color}` }));
      attachTip(line, tip);
      openOnClick(line, id);
      line.addEventListener("mouseenter", () => this.highlight(id));
      line.addEventListener("mouseleave", () => this.highlight(null));
      for (const [px, py] of points) line.append(s("circle", { class: "journey-dot", cx: px, cy: py, r: 4.5, style: `fill: ${color}` }));
      const [ex, ey] = points.at(-1);
      const title = work.title ?? "";
      const maxChars = Math.floor((margin.right - 14) / 6.6);
      line.append(s("text", { class: "journey-label", x: ex + 12, y: ey + 4 }, `#${i + 1} ${title.length > maxChars ? `${title.slice(0, maxChars - 1)}…` : title}`));
      lines.append(line);
    });
    svg.append(lines);
    if (this.highlighted !== null) this.highlight(this.highlighted);
  }
}
