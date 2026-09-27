// Stage 3, setwise LLM reranking: the heap of the 50 candidates as the reranker builds and drains it. Each node is a
// paper, labelled with its hybrid rank; comparisons in flight are outlined (dashed: sent speculatively), the winner
// flashes, swaps move the nodes, and each extracted paper moves to the podium on the right. Nodes and podium entries
// link to the papers' OpenAlex pages.
// The heap is replayed from the trace events like tests/test_setwise_reranker.py: test_events_reconstruct_the_heap.

import { attachTip, fmt, h, openalexUrl, openOnClick, paperLink, paperTip, rankColor, s, viewHead } from "./util.js";

const SYSTEM_PROMPT =
  "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query.";

export class HeapView {
  constructor(root) {
    this.root = root;
    this.svg = s("svg", { class: "heap-svg", role: "img", "aria-label": "Heap of the reranking candidates" });
    this.decision = h("div", { class: "decision" });
    this.counters = h("dl", { class: "counters" });
    this.prompt = h("pre", { class: "prompt" });
    // what the sort is doing right now, in plain words
    this.phase = h("div", { class: "heap-phase" });
    root.append(
      viewHead(
        "rerank",
        "Which fit best? The LLM as the judge",
        "The LLM gets the description and three abstracts, and answers with one letter: the best fit. A tournament tree (a heap) turns these small decisions into the top 5 of the 50 candidates, with about 55 of them.",
      ),
      h(
        "div",
        { class: "heap-layout" },
        h("div", { class: "heap-panel" }, this.phase, this.svg),
        h(
          "div",
          { class: "side" },
          h("h3", { class: "legend-title" }, "How to read the tree"),
          h(
            "ul",
            { class: "heap-legend" },
            h("li", {}, h("i", { class: "lg-node" }, "12"), "a candidate; the number is its rank in the search"),
            h("li", {}, h("i", { class: "lg-cmp" }), "three being compared right now: a parent and its two children"),
            h("li", {}, h("i", { class: "lg-spec" }), "a comparison asked ahead, in case it's needed"),
            h("li", {}, h("i", { class: "lg-flash" }), "flash: the LLM's pick among the three"),
            h("li", {}, h("i", { class: "lg-swap" }, "↕"), "two circles trade places: the pick moves up, the parent down"),
            h("li", {}, h("i", { class: "lg-final" }), "a circle flies right: the top of the tree is the next best paper"),
          ),
          this.counters,
          h("h3", {}, "Latest decision"),
          this.decision,
          h("details", { class: "prompt-details" }, h("summary", {}, "The prompt of this comparison"), this.prompt),
        ),
      ),
    );
    new ResizeObserver(() => this.layout()).observe(this.svg);
    this.reset();
  }

  reset(run) {
    this.run = run;
    this.heap = null;
    this.nodes = new Map(); // docid -> {g, circle}
    this.inflight = new Map(); // call -> {positions, docids, speculative, lines}
    this.comparing = new Map(); // docid -> number of comparisons in flight
    this.stats = { sent: 0, speculative: 0, answered: 0, used: null };
    this.size = 0;
    this.ranked = [];
    this.svg.replaceChildren();
    this.edgeLayer = s("g", { class: "edges" });
    this.overlay = s("g", { class: "cmp-layer" });
    this.nodeLayer = s("g");
    this.podium = s("g", { class: "podium" });
    this.svg.append(this.edgeLayer, this.overlay, this.podium, this.nodeLayer);
    this.decision.replaceChildren(h("p", { class: "empty" }, "No comparison yet."));
    this.setPhase(null);
    this.prompt.textContent = "";
    this.renderCounters();
    this.layout();
  }

  onEvent(event, run) {
    this.run = run;
    switch (event.type) {
      case "heap_init":
        this.init(event.docids, event.num_child, event.k);
        this.setPhase("build");
        break;
      case "compare_start":
        this.compareStart(event);
        break;
      case "compare_end":
        this.compareEnd(event);
        break;
      case "swap": {
        const [a, b] = event.positions;
        [this.heap[a], this.heap[b]] = [this.heap[b], this.heap[a]];
        this.placeNode(this.heap[a], a);
        this.placeNode(this.heap[b], b);
        break;
      }
      case "ranked":
        this.rank(event);
        this.setPhase(event.rank >= (this.k ?? 5) ? "done" : "extract", event.rank);
        break;
      case "reranked":
        this.stats.used = event.used_calls;
        this.stats.sent = event.calls;
        this.renderCounters();
        break;
    }
  }

  init(docids, numChild, k) {
    this.heap = [...docids];
    this.numChild = numChild;
    this.k = k;
    this.size = docids.length;
    this.levels = Math.floor(Math.log(this.size * (numChild - 1) + 1) / Math.log(numChild)) + 1;
    for (const docid of docids) {
      const work = this.run.work(docid);
      const circle = s("circle", { r: 12 });
      const g = s("a", { class: "node", href: openalexUrl(docid), target: "_blank", rel: "noopener" }, circle, s("text", { "text-anchor": "middle", dy: "0.35em" }, work.hybridRank ?? ""));
      attachTip(g, () => paperTip(work, `search #${work.hybridRank}${this.ranked.includes(docid) ? ` · LLM #${this.ranked.indexOf(docid) + 1}` : ""}`));
      this.nodeLayer.append(g);
      this.nodes.set(docid, { g, circle });
    }
    this.layout();
    this.renderCounters();
  }

  // geometry: the tree on the left, the podium for the top k on the right
  layout() {
    const width = this.svg.clientWidth || 1000;
    const height = this.svg.clientHeight || 400;
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.width = width;
    this.podiumX = width - Math.min(360, width * 0.33);
    if (!this.heap) return;
    const treeWidth = this.podiumX - 30;
    const slots = this.numChild ** (this.levels - 1);
    this.radius = Math.max(7, Math.min(15, treeWidth / slots / 2 - 2));
    const levelGap = (height - 2 * 26) / Math.max(1, this.levels - 1);
    this.positions = this.heap.map((_, position) => {
      let level = 0;
      let first = 0;
      while (position >= first + this.numChild ** level) {
        first += this.numChild ** level;
        level++;
      }
      const index = position - first;
      return { x: 10 + ((index + 0.5) / this.numChild ** level) * treeWidth, y: 26 + level * levelGap };
    });
    this.edgeLayer.replaceChildren(
      ...this.heap.slice(1).map((_, i) => {
        const child = i + 1;
        const parent = Math.floor((child - 1) / this.numChild);
        const a = this.positions[parent];
        const b = this.positions[child];
        return s("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, "data-child": child, class: child >= this.size ? "gone" : "" });
      }),
    );
    for (const { circle } of this.nodes.values()) circle.setAttribute("r", this.radius);
    this.heap.forEach((docid, position) => this.placeNode(docid, position));
    this.ranked.forEach((docid, i) => this.placeNode(docid, null, i + 1));
    this.drawPodium();
    for (const comparison of this.inflight.values()) this.drawComparison(comparison);
  }

  placeNode(docid, position, podiumRank) {
    const node = this.nodes.get(docid);
    if (!node || !this.positions) return;
    const rank = podiumRank ?? (this.ranked.includes(docid) ? this.ranked.indexOf(docid) + 1 : null);
    const point = rank ? this.podiumSlot(rank) : this.positions[position];
    node.g.style.transform = `translate(${point.x}px, ${point.y}px)`;
  }

  podiumSlot(rank) {
    return { x: this.podiumX + 22, y: 40 + (rank - 1) * 62 };
  }

  drawPodium() {
    this.podium.replaceChildren(s("text", { class: "podium-title", x: this.podiumX + 6, y: 12 }, `LLM's top ${this.k ?? 5}`));
    for (let rank = 1; rank <= (this.k ?? 5); rank++) {
      const slot = this.podiumSlot(rank);
      this.podium.append(s("circle", { class: "slot", cx: slot.x, cy: slot.y, r: this.radius + 3 }));
      const docid = this.ranked[rank - 1];
      if (docid === undefined) continue;
      const work = this.run.work(docid);
      const maxChars = Math.floor((this.width - slot.x - 40) / 7);
      const title = work.title ?? "";
      this.podium.append(
        s(
          "a",
          { href: openalexUrl(docid), target: "_blank", rel: "noopener" },
          s("text", { class: "podium-rank", x: slot.x + this.radius + 12, y: slot.y - 5 }, `#${rank} · was #${work.hybridRank} in the search`),
          s("text", { class: "podium-name", x: slot.x + this.radius + 12, y: slot.y + 12 }, title.length > maxChars ? `${title.slice(0, maxChars - 1)}…` : title),
        ),
      );
    }
  }

  compareStart(event) {
    this.stats.sent++;
    if (event.speculative) this.stats.speculative++;
    const comparison = { ...event, lines: [] };
    this.inflight.set(event.call, comparison);
    for (const docid of event.docids) this.comparing.set(docid, (this.comparing.get(docid) ?? 0) + 1);
    this.drawComparison(comparison);
    this.updateComparing(event.docids);
    this.renderCounters();
  }

  drawComparison(comparison) {
    for (const line of comparison.lines) line.remove();
    if (!this.positions) return;
    const [parent, ...children] = comparison.positions;
    comparison.lines = children.map((child) => {
      const a = this.positions[parent];
      const b = this.positions[child];
      const line = s("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: `cmp${comparison.speculative ? " spec" : ""}` });
      this.overlay.append(line);
      return line;
    });
  }

  compareEnd(event) {
    const comparison = this.inflight.get(event.call);
    if (!comparison) return;
    this.inflight.delete(event.call);
    this.stats.answered++;
    for (const line of comparison.lines) line.remove();
    for (const docid of comparison.docids) this.comparing.set(docid, this.comparing.get(docid) - 1);
    this.updateComparing(comparison.docids);
    const winnerIndex = comparison.positions.indexOf(event.winner);
    const winner = comparison.docids[winnerIndex];
    const node = this.nodes.get(winner);
    if (node) {
      node.g.classList.remove("won");
      void node.g.getBBox(); // restart the animation
      node.g.classList.add("won");
    }
    this.showDecision(comparison, winnerIndex);
    this.renderCounters();
  }

  updateComparing(docids) {
    for (const docid of docids) {
      const node = this.nodes.get(docid);
      if (node) node.g.classList.toggle("comparing", (this.comparing.get(docid) ?? 0) > 0);
    }
  }

  rank(event) {
    this.ranked.push(event.docid);
    this.size = event.position;
    for (const line of this.edgeLayer.querySelectorAll("line")) line.classList.toggle("gone", Number(line.dataset.child) >= this.size);
    const node = this.nodes.get(event.docid);
    if (node) {
      node.g.classList.add("ranked");
      node.circle.style.fill = rankColor(event.rank);
      this.nodeLayer.append(node.g); // on top while it moves
    }
    this.placeNode(event.docid, null, event.rank);
    this.drawPodium();
  }

  showDecision(comparison, winnerIndex) {
    const labels = "ABCDEFG";
    this.decision.replaceChildren(
      h("div", { class: "decision-meta" }, `call ${comparison.call}${comparison.speculative ? " · asked ahead" : ""} · the description, plus:`),
      ...comparison.docids.map((docid, i) => {
        const work = this.run.work(docid);
        const row = h(
          "div",
          { class: `passage${i === winnerIndex ? " winner" : ""}` },
          h("span", { class: "label" }, labels[i]),
          h("span", { class: "passage-title" }, paperLink(docid, work.title ?? `W${docid}`)),
          h("span", { class: "passage-rank" }, `search #${work.hybridRank}`),
        );
        attachTip(row, () => paperTip(work));
        openOnClick(row, docid);
        return row;
      }),
      h("div", { class: "answer" }, "The model's whole answer: ", h("code", {}, labels[winnerIndex] ?? "?")),
    );
    // the prompt as SetwiseComparisonTask builds it (core/llm_interfaces/tasks.py)
    const passages = comparison.docids.map((docid, i) => `Passage ${labels[i]}: "${this.run.work(docid).abstract ?? "…"}"`).join("\n\n");
    this.prompt.textContent = `[system] ${SYSTEM_PROMPT}\n\n[user] Given a query "${this.run.meta.query}", which of the following passages is the most relevant one to the query?\n\n${passages}\n\nOutput only the passage label of the most relevant passage.`;
  }

  setPhase(phase, rank) {
    const k = this.k ?? 5;
    const texts = {
      build: [
        "1 · Build the tree.",
        ` The LLM looks at each family of three (a parent and its two children) and picks the best; the pick moves up. Once every parent beats its children, the top of the tree is the best of all ${this.size || 50}.`,
      ],
      extract: [
        `2 · Take the best off the top: #${rank} found.`,
        " It trades places with the tree's last paper and flies to the podium. The paper now on top sinks down, family by family, until it beats both its children; then the top is the best of the rest.",
      ],
      done: [
        `Done: the top ${k}.`,
        ` The sort stops here. The paper now at the top was only moved there, never compared, and the other ${Math.max(0, (this.heap?.length ?? 50) - k)} stay only partly sorted. That's what makes a top ${k} cheap.`,
      ],
    };
    this.phase.hidden = !phase;
    if (phase) this.phase.replaceChildren(h("strong", {}, texts[phase][0]), texts[phase][1]);
  }

  renderCounters() {
    const { sent, speculative } = this.stats;
    const items = [
      ["LLM calls", fmt.int(sent)],
      ["running in parallel now", fmt.int(this.inflight.size)],
      ["of them asked ahead", fmt.int(speculative)],
    ];
    this.counters.replaceChildren(...items.flatMap(([label, value]) => [h("dt", {}, label), h("dd", {}, value)]));
  }
}
