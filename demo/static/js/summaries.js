// Stage 4, tailored summaries: for each of the top papers, the abstract, the reasoning structure the model filled out
// (normally discarded), and its FINAL_ANSWER, the summary tailored to the query. The steps are revealed in order.

import { h, paperLink, rankColor, viewHead, words } from "./util.js";

export class SummariesView {
  constructor(root) {
    this.root = root;
    this.tabs = h("div", { class: "summary-tabs", role: "tablist" });
    this.body = h("div", { class: "summary-body" });
    root.append(
      viewHead(
        "summaries",
        "Why should I read it?",
        "For each of the top 3, the LLM reads the description and the abstract, first fills in a fixed worksheet (findings, methods, conclusions, significance for this reader), and only then writes a short summary in the reader's terms.",
      ),
      h("div", { class: "summary-head" }, this.tabs),
      this.body,
    );
    this.reset();
  }

  reset(run) {
    this.run = run;
    this.items = []; // {docid, summary?, reasoning?}
    this.selected = null;
    this.render();
  }

  onEvent(event, run) {
    this.run = run;
    if (event.type === "stage_start" && event.stage === "summarize") {
      this.items = run.top.slice(0, event.works).map((docid) => ({ docid }));
      this.selected = this.items[0]?.docid ?? null;
      this.render();
    } else if (event.type === "summary") {
      let item = this.items.find((i) => i.docid === event.docid);
      if (!item) this.items.push((item = { docid: event.docid }));
      Object.assign(item, { summary: event.summary, reasoning: event.reasoning, fresh: true });
      // follow the first summary that arrives, unless one is already on screen
      const current = this.items.find((i) => i.docid === this.selected);
      if (!current?.summary) this.selected = event.docid;
      this.render();
    }
  }

  select(docid) {
    this.selected = docid;
    const item = this.items.find((i) => i.docid === docid);
    if (item) item.fresh = true; // reveal again
    this.render();
  }

  render() {
    this.tabs.replaceChildren(
      ...this.items.map((item) => {
        const rank = this.run.finalRank(item.docid);
        return h(
          "button",
          { class: `summary-tab${item.docid === this.selected ? " active" : ""}${item.summary ? "" : " pending"}`, role: "tab", onclick: () => this.select(item.docid) },
          h("span", { class: "marker", style: { background: rankColor(rank) } }),
          `#${rank} `,
          truncate(this.run.work(item.docid).title ?? "", 60),
        );
      }),
    );
    const item = this.items.find((i) => i.docid === this.selected);
    if (!item) {
      this.body.replaceChildren(h("p", { class: "empty" }, this.run?.meta?.summaries === 0 ? "This run has no summaries." : "Summaries of the top papers appear here after reranking."));
      return;
    }
    const work = this.run.work(item.docid);
    const meta = [work.authors?.join(", "), work.publication_date].filter(Boolean).join(" · ");
    const steps = item.reasoning
      ? Object.entries(item.reasoning)
          .filter(([key]) => key !== "FINAL_ANSWER")
          .map(([key, value]) => {
            const [title, action] = [key.replace(/^Step (\d+): /, "$1 · "), value?.Action];
            const content = Object.entries(value ?? {})
              .filter(([k]) => k !== "Action")
              .map(([, v]) => v)
              .join("\n");
            return h("li", { class: "step" }, h("div", { class: "step-title" }, title), action ? h("div", { class: "step-action" }, action) : null, h("div", { class: "step-content" }, content));
          })
      : [h("li", { class: "empty" }, "The model is filling in the worksheet…")];
    const summary = item.reasoning?.FINAL_ANSWER ?? item.summary;
    const finalCard = h(
      "div",
      { class: "final" },
      h("h3", {}, "Summary for this reader"),
      summary ? h("p", { class: "final-text" }, summary) : h("p", { class: "empty" }, "…"),
      summary ? h("div", { class: "final-meta" }, `${words(work.abstract)} words of abstract → ${words(summary)} words of summary`) : null,
    );
    this.body.replaceChildren(
      h(
        "div",
        { class: "summary-grid" },
        h(
          "section",
          { class: "abstract" },
          h("h3", {}, "The paper's abstract"),
          h("div", { class: "abstract-title" }, paperLink(item.docid, work.title ?? `W${item.docid}`)),
          h("div", { class: "abstract-meta" }, meta, meta ? " · " : null, paperLink(item.docid, `OpenAlex W${item.docid} ↗`)),
          h("p", { class: "abstract-text" }, work.abstract ?? ""),
        ),
        h("section", { class: "reasoning" }, h("h3", {}, "The model's worksheet, filled in first"), h("ol", { class: "steps" }, ...steps)),
        finalCard,
      ),
    );
    if (item.fresh && item.reasoning) {
      item.fresh = false;
      // reveal the steps one after the other, then the summary
      const parts = [...this.body.querySelectorAll(".step"), finalCard];
      parts.forEach((part, i) => {
        part.classList.add("reveal");
        part.style.animationDelay = `${i * 0.45}s`;
      });
    }
  }
}

function truncate(text, n) {
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}
