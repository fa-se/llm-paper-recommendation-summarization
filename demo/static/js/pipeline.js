// The pipeline strip (one card per view: what goes in and out, time, cost), the run totals, and the call timeline.

import { GROUPS, fmt, groupOf, h, s, attachTip } from "./util.js";

const CARDS = {
  topics: { name: "Topic routing", key: "1" },
  retrieval: { name: "Hybrid retrieval", key: "2" },
  rerank: { name: "LLM reranking", key: "3" },
  summaries: { name: "Tailored summaries", key: "4" },
};

export class Pipeline {
  constructor(strip, timeline, stats, onSelect) {
    this.strip = strip;
    this.timelineRoot = timeline;
    this.statsRoot = stats;
    this.cards = {};
    for (const group of GROUPS) {
      const card = h(
        "button",
        { class: "stage", "data-group": group, onclick: () => onSelect(group), title: `Show this stage (key ${CARDS[group].key})` },
        h("div", { class: "stage-head" }, h("span", { class: "stage-key" }, CARDS[group].key), h("span", { class: "stage-name" }, CARDS[group].name)),
        h("div", { class: "stage-funnel" }, h("span", { class: "stage-out" }, "–"), h("span", { class: "stage-in" }, "")),
        h("div", { class: "stage-detail" }, ""),
        h("div", { class: "stage-metrics" }, h("span", { class: "stage-time" }, ""), h("span", { class: "stage-cost" }, "")),
      );
      this.cards[group] = card;
      this.strip.append(card);
      if (group !== "summaries") this.strip.append(h("div", { class: "stage-arrow", "aria-hidden": "true" }, "→"));
    }
    this.strip.append(
      h(
        "label",
        { class: "follow", title: "Switch to each stage's view as it starts" },
        h("input", { type: "checkbox", id: "follow", checked: true }),
        " follow the run",
      ),
    );
    this.stat = {};
    this.statsRoot.replaceChildren(
      ...[
        ["cost", "API cost", "hero"],
        ["time", "Elapsed"],
        ["calls", "LLM calls"],
        ["tokens", "Tokens in / out"],
      ].map(([key, label, cls]) => {
        this.stat[key] = h("span", { class: "stat-value" }, "–");
        return h("div", { class: `stat ${cls ?? ""}` }, this.stat[key], h("span", { class: "stat-label" }, label));
      }),
    );
    this.timeline = new Timeline(timeline);
    window.addEventListener("resize", () => this.timeline.layout());
  }

  reset(run) {
    this.metrics = Object.fromEntries(GROUPS.map((group) => [group, { duration: 0, cost: 0, calls: 0, open: 0, started: false, done: false }]));
    this.totals = { calls: 0, input: 0, output: 0, cost: 0 };
    for (const group of GROUPS) {
      const card = this.cards[group];
      card.classList.remove("running", "done", "failed");
      card.querySelector(".stage-out").textContent = "–";
      card.querySelector(".stage-in").textContent = "";
      card.querySelector(".stage-detail").textContent = "";
      card.querySelector(".stage-time").textContent = "";
      card.querySelector(".stage-cost").textContent = "";
    }
    this.setFunnel("topics", "10", "of 4,516 OpenAlex topics");
    this.setFunnel("retrieval", "50", "of the corpus");
    this.setFunnel("rerank", "5", "of 50 candidates");
    this.setFunnel("summaries", "3", "top papers");
    for (const card of Object.values(this.cards)) card.classList.add("idle");
    this.stat.cost.textContent = "$0.0000";
    this.stat.time.textContent = "0.0 s";
    this.stat.calls.textContent = "0";
    this.stat.tokens.textContent = "0 / 0";
    this.timeline.reset();
    this.shownTime = 0;
    this.summaryCount = 0;
  }

  setFunnel(group, out, input) {
    const card = this.cards[group];
    card.querySelector(".stage-out").textContent = out;
    card.querySelector(".stage-in").textContent = input;
  }

  detail(group, text) {
    this.cards[group].querySelector(".stage-detail").textContent = text;
  }

  onEvent(event, run) {
    const group = groupOf(event.stage);
    const metrics = group ? this.metrics[group] : null;
    switch (event.type) {
      case "run_start":
        this.setFunnel("rerank", `${event.n}`, `of ${event.n * 10} candidates`);
        this.setFunnel("summaries", `${event.summaries}`, "top papers");
        this.setFunnel("retrieval", `${event.n * 10}`, "of the corpus");
        for (const card of Object.values(this.cards)) card.classList.remove("idle");
        if (!event.summaries) this.cards.summaries.classList.add("idle");
        break;
      case "stage_start":
        if (!metrics) break;
        metrics.open++;
        metrics.started = true;
        this.cards[group].classList.add("running");
        if (event.stage === "fetch") this.detail("topics", `fetching up to ${event.limit ?? "all"} newest papers…`);
        if (event.stage === "embed") this.detail("topics", `embedding ${fmt.int(event.works)} new abstracts…`);
        if (event.stage === "index") this.detail("topics", "rebuilding the BM25 index…");
        if (event.stage === "rerank") this.detail("rerank", "3 abstracts per call, 1-token answer");
        if (event.stage === "summarize") this.detail("summaries", "abstract + interest → reasoning JSON");
        break;
      case "stage_end":
        if (!metrics) break;
        metrics.open--;
        // nested stages (semantic and bm25 inside hybrid) are already in their parent's duration
        if (!["semantic", "bm25"].includes(event.stage)) metrics.duration += event.duration_s;
        if (metrics.open === 0) {
          metrics.done = true;
          this.cards[group].classList.remove("running");
          this.cards[group].classList.add("done");
        }
        this.cards[group].querySelector(".stage-time").textContent = fmt.seconds(metrics.duration);
        break;
      case "llm_call":
        this.totals.calls++;
        this.totals.input += event.input_tokens;
        this.totals.output += event.output_tokens;
        this.totals.cost = event.total_cost_usd;
        if (metrics) {
          metrics.cost += event.cost_usd;
          metrics.calls++;
          this.cards[group].querySelector(".stage-cost").textContent = `${fmt.usd(metrics.cost)} · ${metrics.calls} call${metrics.calls === 1 ? "" : "s"}`;
        }
        this.stat.cost.textContent = fmt.usd(this.totals.cost);
        this.stat.calls.textContent = fmt.int(this.totals.calls);
        this.stat.tokens.textContent = `${fmt.compact(this.totals.input)} / ${fmt.compact(this.totals.output)}`;
        break;
      case "filtered":
        this.detail("topics", `fetched ${fmt.int(event.fetched)}: ${fmt.int(event.new)} new, ${fmt.int(event.already_present)} known, ${fmt.int(event.duplicates + event.unusable)} dupes/junk`);
        break;
      case "corpus":
        this.setFunnel("retrieval", `${run.meta.n * 10}`, `of ${fmt.int(event.size)} papers`);
        break;
      case "ranking":
        if (event.method === "hybrid") this.detail("retrieval", "BM25 + dense, min-max blended 0.2 / 0.8");
        break;
      case "reranked":
        this.detail("rerank", `${event.calls} calls sent, ${event.used_calls} on the path`);
        break;
      case "summary":
        this.summaryCount++;
        this.detail("summaries", `${this.summaryCount} written`);
        break;
      case "run_error":
        for (const g of GROUPS) if (this.metrics[g].open) this.cards[g].classList.add("failed");
        break;
    }
    this.timeline.onEvent(event, run);
    this.setClock(event.t, run);
  }

  setClock(t, run) {
    if (t < this.shownTime) return;
    this.shownTime = t;
    this.stat.time.textContent = `${t.toFixed(1)} s`;
    this.timeline.setClock(t, run);
  }

  finish(run) {
    for (const card of Object.values(this.cards)) card.classList.remove("running");
    this.setClock(run.t, run);
    this.timeline.setClock(run.t, run, true);
  }

  select(group) {
    for (const [name, card] of Object.entries(this.cards)) card.classList.toggle("selected", name === group);
  }
}

// Every API call of the run as a bar over time; concurrent calls stack in lanes.
class Timeline {
  constructor(root) {
    this.root = root;
    this.svg = s("svg", { class: "timeline-svg", role: "img", "aria-label": "Timeline of all API calls" });
    this.legend = h(
      "div",
      { class: "timeline-legend" },
      h("span", { class: "timeline-title" }, "API calls over time"),
      h("span", { class: "key" }, h("i", { class: "swatch call" }), "needed"),
      h("span", { class: "key" }, h("i", { class: "swatch speculative" }), "rerank call sent speculatively"),
      h("span", { class: "key" }, h("i", { class: "swatch inflight" }), "in flight"),
    );
    this.root.append(this.legend, this.svg);
  }

  reset() {
    this.bars = new Map(); // key -> {start, end|null, lane, kind, element}
    this.lanes = []; // end time of each lane's last bar (Infinity while in flight)
    this.stages = []; // {group, start, end}
    this.clock = 0;
    this.domain = 20;
    this.svg.replaceChildren();
    this.bandLayer = s("g");
    this.barLayer = s("g");
    this.axisLayer = s("g", { class: "axis" });
    this.svg.append(this.bandLayer, this.barLayer, this.axisLayer);
    this.layout();
  }

  lane(start) {
    let lane = this.lanes.findIndex((end) => end <= start + 1e-3);
    if (lane === -1) {
      lane = this.lanes.length;
      this.lanes.push(0);
    }
    return lane;
  }

  addBar(key, start, end, kind, tip) {
    const lane = this.lane(start);
    this.lanes[lane] = end ?? Infinity;
    const element = s("rect", { class: `bar ${kind}${end === null ? " inflight" : ""}`, rx: 1 });
    attachTip(element, tip);
    this.barLayer.append(element);
    const bar = { start, end, lane, kind, element };
    this.bars.set(key, bar);
    if (this.lanes.length > this.shownLanes) this.layout();
    else this.place(bar);
  }

  onEvent(event, run) {
    const group = groupOf(event.stage);
    switch (event.type) {
      case "stage_start": {
        if (!group) break;
        const last = this.stages.at(-1);
        if (last?.group === group) last.end = null;
        else this.stages.push({ group, start: event.t, end: null });
        break;
      }
      case "stage_end": {
        const last = this.stages.at(-1);
        if (last && last.group === group) last.end = event.t;
        this.drawBands();
        break;
      }
      case "compare_start": {
        const titles = event.docids.map((id, i) => `${"ABC"[i]}: ${run.work(id).title}`);
        this.addBar(`c${event.call}`, event.t, null, event.speculative ? "speculative" : "call", () =>
          h("div", {}, h("div", { class: "tip-title" }, `Rerank call ${event.call}${event.speculative ? " (speculative)" : ""}`), ...titles.map((t) => h("div", { class: "tip-meta" }, t))),
        );
        break;
      }
      case "compare_end": {
        const bar = this.bars.get(`c${event.call}`);
        if (!bar) break;
        bar.end = event.t;
        this.lanes[bar.lane] = event.t;
        bar.element.classList.remove("inflight");
        this.place(bar);
        break;
      }
      case "llm_call": {
        // rerank calls are drawn from compare_start/compare_end, which say which comparison they were
        if (group === "rerank") break;
        const start = Math.max(0, event.t - event.latency_s);
        this.addBar(`l${event.seq}`, start, event.t, "call", `${event.model}: ${fmt.int(event.input_tokens)} tokens in, ${fmt.int(event.output_tokens)} out, ${fmt.seconds(event.latency_s)}, ${fmt.usd(event.cost_usd)}`);
        break;
      }
    }
  }

  setClock(t, run, final = false) {
    this.clock = t;
    const domain = final ? Math.max(t * 1.02, 1) : Math.max(20, Math.ceil((t + 2) / 10) * 10);
    if (domain !== this.domain) {
      this.domain = domain;
      this.layout();
      return;
    }
    for (const bar of this.bars.values()) if (bar.end === null) this.place(bar);
    this.drawBands();
  }

  x(t) {
    return this.padLeft + (t / this.domain) * (this.width - this.padLeft - 12);
  }

  place(bar) {
    const end = bar.end ?? this.clock;
    const x = this.x(bar.start);
    bar.element.setAttribute("x", x);
    bar.element.setAttribute("width", Math.max(1.5, this.x(end) - x));
    bar.element.setAttribute("y", 4 + bar.lane * this.laneHeight);
    bar.element.setAttribute("height", this.laneHeight - 1.5);
  }

  layout() {
    this.width = this.root.clientWidth || 1200;
    this.padLeft = 8;
    this.shownLanes = Math.max(this.lanes?.length ?? 0, 6);
    this.laneHeight = this.shownLanes > 14 ? 5 : 7;
    this.plotHeight = 4 + this.shownLanes * this.laneHeight;
    const height = this.plotHeight + 20;
    this.svg.setAttribute("viewBox", `0 0 ${this.width} ${height}`);
    this.svg.setAttribute("height", height);
    for (const bar of this.bars?.values() ?? []) this.place(bar);
    this.drawAxis();
    this.drawBands();
  }

  drawAxis() {
    this.axisLayer.replaceChildren();
    const step = this.domain > 60 ? 10 : this.domain > 25 ? 5 : 2;
    for (let t = 0; t <= this.domain + 1e-6; t += step) {
      const x = this.x(t);
      this.axisLayer.append(
        s("line", { x1: x, x2: x, y1: this.plotHeight + 2, y2: this.plotHeight + 6 }),
        s("text", { x, y: this.plotHeight + 17, "text-anchor": t === 0 ? "start" : "middle" }, `${t} s`),
      );
    }
    this.axisLayer.append(s("line", { x1: this.padLeft, x2: this.width - 12, y1: this.plotHeight + 2, y2: this.plotHeight + 2 }));
  }

  drawBands() {
    if (!this.bandLayer) return;
    this.bandLayer.replaceChildren();
    for (const stage of this.stages) {
      const x1 = this.x(stage.start);
      const x2 = this.x(stage.end ?? this.clock);
      this.bandLayer.append(s("rect", { class: `band band-${stage.group}`, x: x1, width: Math.max(1, x2 - x1), y: 0, height: this.plotHeight + 2 }));
      if (x2 - x1 > 70) this.bandLayer.append(s("text", { class: "band-label", x: x1 + 4, y: this.plotHeight - 3 }, CARDS[stage.group].name));
    }
  }
}
