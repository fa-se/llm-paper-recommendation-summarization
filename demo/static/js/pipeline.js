// The step bar (the intro steps, one card per pipeline stage with what goes in and out, time and cost, and the closing
// steps), the run totals in the header, and the timeline of API calls.

import { GROUPS, STEPS, STEP_LABELS, attachTip, fmt, groupOf, h, s } from "./util.js";

const PILL_TITLES = {
  intro: "The problem, and the idea in one picture",
  system: "The parts: OpenAlex, Postgres, the OpenAI models",
  results: "The final top 5, and where they came from",
  takeaways: "What I learned",
};

export class Pipeline {
  constructor(stepper, timeline, stats, onSelect) {
    this.items = {};
    const stages = h("div", { class: "stage-group" });
    for (const step of STEPS) {
      const number = GROUPS.indexOf(step) + 1;
      if (!number) {
        const pill = h("button", { class: "step-pill", type: "button", "data-step": step, onclick: () => onSelect(step), title: PILL_TITLES[step] }, STEP_LABELS[step]);
        this.items[step] = pill;
        // the stage cards sit between the intro and the closing pills
        if (step === "results") stepper.append(stages);
        stepper.append(pill);
        continue;
      }
      const card = h(
        "button",
        { class: "stage idle", type: "button", "data-step": step, onclick: () => onSelect(step), title: `Show this stage (key ${number})` },
        h("div", { class: "stage-head" }, h("span", { class: "stage-key" }, number), h("span", { class: "stage-name" }, STEP_LABELS[step])),
        h("div", { class: "stage-funnel" }, h("span", { class: "stage-out" }, ""), h("span", { class: "stage-in" }, "")),
        h("div", { class: "stage-metrics" }, h("span", { class: "stage-time" }, ""), h("span", { class: "stage-cost" }, "")),
      );
      this.items[step] = card;
      if (number > 1) stages.append(h("div", { class: "stage-arrow", "aria-hidden": "true" }, "→"));
      stages.append(card);
    }
    this.stat = {};
    stats.replaceChildren(
      ...[
        ["cost", "API cost", "hero"],
        ["time", "elapsed"],
        ["calls", "LLM calls"],
      ].map(([key, label, cls]) => {
        this.stat[key] = h("span", { class: "stat-value" }, "–");
        return h("div", { class: `stat ${cls ?? ""}` }, this.stat[key], h("span", { class: "stat-label" }, label));
      }),
    );
    this.timeline = new Timeline(timeline);
    window.addEventListener("resize", () => this.timeline.layout());
  }

  reset(run) {
    for (const group of GROUPS) {
      const card = this.items[group];
      card.classList.remove("running", "done", "failed", "pending");
      card.classList.add("idle");
      card.querySelector(".stage-time").textContent = "";
      card.querySelector(".stage-cost").textContent = "";
    }
    this.setFunnel("topics", "10", "of 4,516 topics");
    this.setFunnel("retrieval", "50", "of the papers");
    this.setFunnel("rerank", "5", "of 50 candidates");
    this.setFunnel("summaries", "3", "summaries");
    this.stat.cost.textContent = "$0.00";
    this.stat.time.textContent = "0.0 s";
    this.stat.calls.textContent = "0";
    this.timeline.reset();
    this.shownTime = 0;
    this.run = run;
  }

  setFunnel(group, out, input) {
    const card = this.items[group];
    card.querySelector(".stage-out").textContent = out;
    card.querySelector(".stage-in").textContent = input;
  }

  // the stage cards read their numbers from the run state, which has already applied the event
  onEvent(event, run) {
    const group = groupOf(event.stage);
    switch (event.type) {
      case "run_start":
        this.setFunnel("retrieval", `${event.n * 10}`, "of the papers");
        this.setFunnel("rerank", `${event.n}`, `of ${event.n * 10} candidates`);
        this.setFunnel("summaries", `${event.summaries}`, "summaries");
        for (const g of GROUPS) this.items[g].classList.remove("idle");
        if (!event.summaries) this.items.summaries.classList.add("idle");
        break;
      case "stage_start":
        if (group) this.items[group].classList.add("running");
        if (event.stage === "fetch") this.setFunnel("topics", "10", "topics: fetching their newest papers…");
        if (event.stage === "embed") this.setFunnel("topics", "10", `topics: embedding ${fmt.int(event.works)} new papers…`);
        if (event.stage === "index") this.setFunnel("topics", "10", "topics: updating the keyword index…");
        break;
      case "stage_end":
        if (!group) break;
        if (run.groups[group].done) {
          this.items[group].classList.remove("running");
          this.items[group].classList.add("done");
        }
        this.items[group].querySelector(".stage-time").textContent = fmt.seconds(run.groups[group].duration);
        if (!run.groups[group].calls) this.items[group].querySelector(".stage-cost").textContent = "no LLM";
        if (event.stage === "index" || (event.stage === "topics" && !run.meta.ingest)) this.setFunnel("topics", "10", "of 4,516 topics");
        break;
      case "llm_call": {
        this.stat.cost.textContent = fmt.usd(run.cost);
        this.stat.calls.textContent = fmt.int(run.calls);
        if (!group) break;
        const { cost, calls } = run.groups[group];
        this.items[group].querySelector(".stage-cost").textContent = `${fmt.usd(cost)} · ${calls} call${calls === 1 ? "" : "s"}`;
        break;
      }
      case "corpus":
        this.setFunnel("retrieval", `${run.meta.n * 10}`, `of ${fmt.int(event.size)} papers`);
        break;
      case "run_error":
        for (const g of GROUPS) if (run.groups[g].open) this.items[g].classList.add("failed");
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
    for (const g of GROUPS) this.items[g].classList.remove("running", "pending");
    this.setClock(run.t, run);
    this.timeline.setClock(run.t, run, true);
  }

  select(step) {
    for (const [name, item] of Object.entries(this.items)) item.classList.toggle("selected", name === step);
  }

  // the stage a paused run continues with
  pending(group) {
    for (const g of GROUPS) this.items[g].classList.toggle("pending", g === group);
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
      h("span", { class: "key" }, h("i", { class: "swatch call" }), "call"),
      h("span", { class: "key" }, h("i", { class: "swatch speculative" }), "ranking call sent ahead (speculative)"),
      h("span", { class: "key" }, h("i", { class: "swatch inflight" }), "waiting for the answer"),
      h("span", { class: "timeline-hint" }, "stacked bars run in parallel"),
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
          h("div", {}, h("div", { class: "tip-title" }, `Ranking call ${event.call}${event.speculative ? " (sent ahead)" : ""}`), ...titles.map((t) => h("div", { class: "tip-meta" }, t))),
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
        // ranking calls are drawn from compare_start/compare_end, which say which comparison they were
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
      if (x2 - x1 > 70) this.bandLayer.append(s("text", { class: "band-label", x: x1 + 4, y: this.plotHeight - 3 }, STEP_LABELS[stage.group]));
    }
  }
}
