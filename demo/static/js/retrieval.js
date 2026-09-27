// Stage 2, hybrid retrieval: BM25, dense and hybrid rankings side by side. Lines follow the same paper across the
// columns. The LLM's result isn't shown here (the Results step shows where the final top 5 came from), but once it's
// known, the final top-k papers keep their color, and the columns list where they rank further down. The blend can be
// changed client-side from the recorded per-method scores ("What if?"): another weight, or reciprocal rank fusion.

import { attachTip, h, openOnClick, paperLink, paperTip, rankColor, s, viewHead } from "./util.js";

const SHOWN = 10;
const RRF_K = 60;
const RECORDED_WEIGHT = 0.8; // HYBRID_WEIGHTS in core/services/retrieval_service.py

const COLUMNS = [
  { key: "bm25", title: "Keyword match", sub: "BM25: shared words, weighted by rarity · top 10 of 100", score: (r) => r.score.toFixed(2) },
  { key: "semantic", title: "Meaning match", sub: "embeddings: similar meaning, other words · top 10 of 100", score: (r) => r.score.toFixed(2) },
  { key: "hybrid", title: "Combined: the candidates", sub: "", score: (r) => r.score.toFixed(2) },
];

// Blends the two rankings like RetrievalService._hybrid_search (weighted sum of min-max normalized scores; a work
// missing from one ranking gets 0 there), or by reciprocal rank fusion. Insertion order breaks ties, as in Python.
export function blend(semantic, bm25, mode, weight) {
  const scores = new Map();
  // each part keeps its input (the normalized score, or the rank for RRF), so that tooltips can show the arithmetic
  const add = (id, part, value, input) => {
    const entry = scores.get(id) ?? { id, dense: 0, bm25: 0, denseInput: null, bm25Input: null };
    entry[part] += value;
    entry[`${part}Input`] = input;
    scores.set(id, entry);
  };
  if (mode === "rrf") {
    semantic.forEach((result, i) => add(result.id, "dense", 1 / (RRF_K + i + 1), i + 1));
    bm25.forEach((result, i) => add(result.id, "bm25", 1 / (RRF_K + i + 1), i + 1));
  } else {
    semantic.forEach((result) => add(result.id, "dense", weight * result.normalized, result.normalized));
    bm25.forEach((result) => add(result.id, "bm25", (1 - weight) * result.normalized, result.normalized));
  }
  return [...scores.values()].map((entry) => ({ ...entry, score: entry.dense + entry.bm25 })).sort((a, b) => b.score - a.score);
}

export class RetrievalView {
  constructor(root) {
    this.root = root;
    this.mode = "weighted";
    this.weight = RECORDED_WEIGHT;
    this.slider = h("input", { type: "range", min: 0, max: 1, step: 0.05, value: RECORDED_WEIGHT, "aria-label": "Weight of dense retrieval" });
    this.slider.addEventListener("input", () => {
      this.weight = Number(this.slider.value);
      this.update();
    });
    this.modeSelect = h(
      "select",
      { "aria-label": "Blending method" },
      h("option", { value: "weighted" }, "Weighted sum of min-max scores"),
      h("option", { value: "rrf" }, `Reciprocal rank fusion (k = ${RRF_K})`),
    );
    this.modeSelect.addEventListener("change", () => {
      this.mode = this.modeSelect.value;
      this.update();
    });
    this.weightLabel = h("span", { class: "weight-label" });
    this.poolStat = h("span", { class: "pool-stat" });
    this.resetButton = h("button", { class: "small", onclick: () => this.resetBlend() }, "As recorded");
    this.grid = h("div", { class: "rank-grid" });
    this.lines = s("svg", { class: "rank-lines", "aria-hidden": "true" });
    this.blendBar = h(
      "div",
      { class: "blend-bar", hidden: true },
      h("span", { class: "blend-title" }, "Blend"),
      this.modeSelect,
      h("span", { class: "slider-end" }, "keyword"),
      this.slider,
      h("span", { class: "slider-end" }, "meaning"),
      this.weightLabel,
      this.resetButton,
      this.poolStat,
    );
    this.whatIf = h("button", { class: "small what-if", type: "button", onclick: () => this.toggleBlend() }, "What if? Change the blend");
    this.legend = h("p", { class: "rank-legend" });
    const head = viewHead(
      "retrieval",
      "Which papers could fit?",
      "Two cheap searches score every paper in the pool: by shared keywords, and by meaning. A blend of both picks 50 candidates; only these go to the LLM.",
    );
    head.insertBefore(this.whatIf, head.lastChild);
    root.append(head, this.blendBar, h("div", { class: "rank-wrap" }, this.grid, this.lines), this.legend);
    new ResizeObserver(() => this.drawLines()).observe(this.grid);
    this.reset();
  }

  reset(run) {
    this.run = run;
    this.rankings = {};
    this.render();
  }

  toggleBlend(open = this.blendBar.hidden) {
    this.blendBar.hidden = !open;
    this.whatIf.classList.toggle("active", open);
    if (!open) this.resetBlend();
    requestAnimationFrame(() => this.drawLines());
  }

  resetBlend() {
    this.mode = "weighted";
    this.modeSelect.value = "weighted";
    this.weight = RECORDED_WEIGHT;
    this.slider.value = RECORDED_WEIGHT;
    this.update();
  }

  onEvent(event, run) {
    this.run = run;
    if (event.type === "ranking") {
      this.rankings[event.method] = event.results;
      if (event.method === "hybrid") this.checkBlend();
      this.render();
    } else if (["reranked", "candidates"].includes(event.type)) this.render();
  }

  show() {
    requestAnimationFrame(() => this.drawLines());
  }

  update() {
    this.render();
  }

  // the client-side blend must reproduce the recorded hybrid ranking at the recorded weight
  checkBlend() {
    const { semantic, bm25, hybrid } = this.rankings;
    if (!semantic || !bm25 || !hybrid) return;
    const ids = blend(semantic, bm25, "weighted", RECORDED_WEIGHT).slice(0, hybrid.length).map((r) => r.id);
    const mismatches = hybrid.filter((result, i) => result.id !== ids[i]).length;
    if (mismatches) console.warn(`Client-side blend differs from the recorded hybrid ranking at ${mismatches} of ${hybrid.length} ranks`);
  }

  hybridRanking() {
    const { semantic, bm25, hybrid } = this.rankings;
    if (!semantic || !bm25) return hybrid ?? null;
    const poolSize = hybrid?.length ?? 50;
    return blend(semantic, bm25, this.mode, this.weight).slice(0, poolSize);
  }

  render() {
    const run = this.run;
    const recorded = this.mode === "weighted" && Math.abs(this.weight - RECORDED_WEIGHT) < 1e-9;
    this.weightLabel.textContent = this.mode === "rrf" ? "weights unused" : `meaning ${this.weight.toFixed(2)} · keyword ${(1 - this.weight).toFixed(2)}`;
    this.slider.disabled = this.mode === "rrf";
    this.resetButton.disabled = recorded;
    this.root.classList.toggle("modified", !recorded);

    const hybrid = this.hybridRanking();
    const lists = {
      bm25: this.rankings.bm25 ?? null,
      semantic: this.rankings.semantic ?? null,
      hybrid,
    };
    const pool = new Set(run?.candidates ?? []);
    // the final top k, once the LLM is done: this view then marks where they rank in each column
    const top = run?.rerank ? run.top : [];
    this.top = top;
    this.legend.textContent = top.length
      ? `Lines connect the same paper across the columns. Colored: the LLM's final top ${top.length}, which the Results step follows.`
      : "Lines connect the same paper across the columns.";
    this.ranks = Object.fromEntries(Object.entries(lists).map(([key, list]) => [key, new Map((list ?? []).map((r, i) => [r.id, i + 1]))]));

    // the pool statistic: how much of the candidate pool the LLM actually reranked this blend would keep
    if (hybrid && pool.size && !recorded) {
      const blended = new Set(hybrid.map((r) => r.id));
      const kept = [...pool].filter((id) => blended.has(id)).length;
      const topKept = top.filter((id) => blended.has(id)).length;
      this.poolStat.textContent = `Top 50 of this blend: ${kept} of the 50 reranked candidates${top.length ? `, ${topKept} of the final top ${top.length}` : ""}`;
    } else this.poolStat.textContent = hybrid ? "Blend as in the recorded run" : "";

    // the bar's full length is the highest possible score: first in both lists (weighted: 1, RRF: 2 / (k + 1))
    const maxHybrid = this.mode === "rrf" ? 2 / (RRF_K + 1) : 1;
    this.grid.replaceChildren(
      ...COLUMNS.map((column) => {
        const list = lists[column.key];
        const sub =
          column.key === "hybrid"
            ? this.mode === "rrf"
              ? `Σ 1 / (${RRF_K} + rank) · top 10 of 50`
              : `${Math.round(this.weight * 100)} % meaning + ${Math.round((1 - this.weight) * 100)} % keyword · top 10 of 50`
            : column.sub;
        const body = h("ol", { class: "rank-list" });
        if (!list) body.append(h("li", { class: "empty" }, "–"));
        else {
          const shown = list.slice(0, SHOWN);
          shown.forEach((result, i) => body.append(this.row(column, result, i + 1, column.key === "hybrid" ? maxHybrid : null)));
          if (column.key === "hybrid" && list.length > SHOWN) body.append(h("li", { class: "more" }, `+ ${list.length - SHOWN} more candidates → the LLM`));
          // the final top k papers that this column ranks below its shown rows
          const below = top.filter((id) => !shown.some((r) => r.id === id));
          if (below.length) {
            body.append(h("li", { class: "fold" }, `the LLM's top ${top.length}, further down`));
            for (const id of below) body.append(this.foldRow(column, id, pool));
          }
        }
        const legend =
          column.key === "hybrid"
            ? h(
                "span",
                { class: "contrib-legend", title: "Bar: the combined score, split into the meaning part and the keyword part. Full length: #1 in both searches" },
                h("i", { class: "seg dense" }),
                "meaning",
                h("i", { class: "seg keyword" }),
                "keyword",
              )
            : null;
        return h("section", { class: `rank-col col-${column.key}` }, h("h3", {}, column.title, legend), h("div", { class: "col-sub" }, sub), body);
      }),
    );
    requestAnimationFrame(() => this.drawLines());
  }

  row(column, result, rank, maxHybrid) {
    const run = this.run;
    const finalRank = this.top.includes(result.id) ? run.finalRank(result.id) : null;
    const work = run?.work(result.id) ?? { id: result.id, title: result.title };
    const color = rankColor(finalRank);
    const element = h(
      "li",
      { class: `rank-row${color ? " tracked" : ""}`, "data-id": result.id },
      h("span", { class: "rank" }, rank),
      h("span", { class: "marker", style: color ? { background: color } : {} }),
      h(
        "div",
        { class: "row-main" },
        h("div", { class: "row-title" }, paperLink(result.id, work.title ?? result.title ?? `W${result.id}`)),
        column.key === "hybrid" && maxHybrid
          ? h(
              "div",
              { class: "contrib" },
              h("span", { class: "seg dense", style: { width: `${(result.dense / maxHybrid) * 100}%` } }),
              h("span", { class: "seg keyword", style: { width: `${(result.bm25 / maxHybrid) * 100}%` } }),
            )
          : null,
      ),
      h("span", { class: "score" }, column.score(result)),
    );
    attachTip(element, () => {
      const tip = paperTip(work, this.rankSummary(result.id));
      if (column.key === "hybrid") tip.insertBefore(h("div", { class: "tip-meta" }, this.scoreSummary(result)), tip.querySelector(".tip-body"));
      return tip;
    });
    openOnClick(element, result.id);
    this.hoverable(element, result.id);
    return element;
  }

  // the arithmetic behind a hybrid score and its bar
  scoreSummary(result) {
    const part = (name, value, input, weight) => {
      if (input === null) return `${name} 0 (not in its top 100)`;
      return this.mode === "rrf" ? `${name} 1 / (${RRF_K} + ${input}) = ${value.toFixed(4)}` : `${name} ${weight.toFixed(2)} × ${input.toFixed(3)} = ${value.toFixed(3)}`;
    };
    const max = this.mode === "rrf" ? (2 / (RRF_K + 1)).toFixed(4) : "1";
    return `${part("meaning", result.dense, result.denseInput, this.weight)} + ${part("keyword", result.bm25, result.bm25Input, 1 - this.weight)} = ${result.score.toFixed(this.mode === "rrf" ? 4 : 3)} (max ${max})`;
  }

  foldRow(column, id, pool) {
    const rank = this.ranks[column.key].get(id);
    const color = rankColor(this.run.finalRank(id));
    const text = rank ? `#${rank}` : "not in its top 100";
    const element = h(
      "li",
      { class: `rank-row fold-row tracked${rank ? "" : " missing"}`, "data-id": id },
      h("span", { class: "rank" }, ""),
      h("span", { class: "marker", style: { background: color } }),
      h("div", { class: "row-main" }, h("div", { class: "row-title" }, text)),
    );
    attachTip(element, () => paperTip(this.run.work(id), this.rankSummary(id)));
    openOnClick(element, id);
    this.hoverable(element, id);
    return element;
  }

  // where a paper ranks in each column; the LLM only orders its top k, the other candidates stay unranked
  rankSummary(id) {
    const ranks = this.ranks;
    const k = this.run?.meta?.n ?? 5;
    const final = this.top.indexOf(id) + 1;
    const llm = !this.top.length
      ? null
      : final
        ? `LLM #${final}`
        : this.run?.candidates?.includes(id)
          ? `LLM: not in its top ${k} (ranks below ${k} aren't computed)`
          : "LLM: not a candidate";
    return [
      `keyword ${ranks.bm25.has(id) ? `#${ranks.bm25.get(id)}` : "not in top 100"}`,
      `meaning ${ranks.semantic.has(id) ? `#${ranks.semantic.get(id)}` : "not in top 100"}`,
      `combined ${ranks.hybrid.has(id) ? `#${ranks.hybrid.get(id)}` : "not in top 50"}`,
      llm,
    ]
      .filter(Boolean)
      .join(" · ");
  }

  hoverable(element, id) {
    element.addEventListener("mouseenter", () => this.highlight(id));
    element.addEventListener("mouseleave", () => this.highlight(null));
  }

  highlight(id) {
    for (const row of this.grid.querySelectorAll(".rank-row")) row.classList.toggle("hl", id !== null && row.dataset.id === String(id));
    for (const line of this.lines.querySelectorAll("path")) line.classList.toggle("hl", id !== null && line.dataset.id === String(id));
    this.grid.classList.toggle("hovering", id !== null);
    this.lines.classList.toggle("hovering", id !== null);
  }

  // connectors between the same paper in adjacent columns
  drawLines() {
    const wrap = this.lines.parentElement;
    const box = wrap.getBoundingClientRect();
    this.lines.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    this.lines.setAttribute("width", box.width);
    this.lines.setAttribute("height", box.height);
    this.lines.replaceChildren();
    const columns = [...this.grid.querySelectorAll(".rank-col")];
    const rowsOf = (column) => new Map([...column.querySelectorAll(".rank-row")].map((row) => [row.dataset.id, row.getBoundingClientRect()]));
    for (let i = 0; i + 1 < columns.length; i++) {
      const left = rowsOf(columns[i]);
      const right = rowsOf(columns[i + 1]);
      for (const [id, a] of left) {
        const b = right.get(id);
        if (!b) continue;
        const finalRank = this.top.includes(Number(id)) ? this.run.finalRank(Number(id)) : null;
        const x1 = a.right - box.left + 2;
        const y1 = a.top + a.height / 2 - box.top;
        const x2 = b.left - box.left - 2;
        const y2 = b.top + b.height / 2 - box.top;
        const mid = (x1 + x2) / 2;
        const path = s("path", { d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`, class: finalRank && finalRank <= 5 ? "tracked" : "", "data-id": id });
        if (finalRank && finalRank <= 5) path.style.stroke = rankColor(finalRank);
        this.lines.append(path);
      }
    }
  }
}

