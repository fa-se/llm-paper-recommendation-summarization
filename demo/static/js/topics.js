// Stage 1, topic routing: all 4,516 OpenAlex topics on a 2D map (UMAP of their embeddings, scripts/make_topic_map.py),
// the query's 10 most similar topics highlighted, and the list of those topics with their cosine similarity.
// The query itself isn't drawn: a paragraph-long description is about equally similar to all of its matches, and less
// similar to them than they are to each other (coral: 0.56-0.64 vs 0.63-0.86), so no point on the map represents it.
// The map shows where the matches sit in the taxonomy; the list shows how similar they are. The details drawer
// (details.js) explains this for the audience.

import { cssVar, fmt, h, openalexUrl, tooltip, viewHead } from "./util.js";

export class TopicsView {
  constructor(root) {
    this.root = root;
    this.canvas = h("canvas", { class: "map-canvas", "aria-label": "Map of all OpenAlex topics" });
    this.zoomButton = h("button", { class: "small", onclick: () => this.zoom(!this.zoomed) }, "Whole map");
    this.list = h("ol", { class: "topic-list" });
    this.note = h("p", { class: "note" });
    root.append(
      viewHead(
        "topics",
        "Which research areas?",
        "OpenAlex files every paper under one of 4,516 research topics. The description is compared by meaning with all of them; the newest papers of the 10 closest topics form the pool that the next steps search.",
      ),
      h(
        "div",
        { class: "map-layout" },
        h(
          "div",
          { class: "map-panel" },
          this.canvas,
          h(
            "div",
            { class: "map-legend" },
            h("span", { class: "key" }, h("i", { class: "dot all" }), "one of 4,516 topics; similar topics sit close together"),
            h("span", { class: "key" }, h("i", { class: "dot match" }), "the 10 closest to the description"),
            this.zoomButton,
          ),
        ),
        h("div", { class: "side" }, h("h3", {}, "The 10 closest topics"), h("div", { class: "list-head" }, h("span"), h("span", {}, "topic"), h("span", {}, "similarity to the description")), this.list, this.note),
      ),
    );
    this.view = { x: 0, y: 0, k: 1 }; // the visible part of the unit square: origin and zoom
    this.matches = [];
    this.data = fetch("/static/topic_map.json")
      .then((response) => response.json())
      .then((data) => this.prepare(data))
      .catch(() => (this.note.textContent = "Topic map not available (demo/static/topic_map.json)."));
    this.canvas.addEventListener("mousemove", (event) => this.hover(event));
    this.canvas.addEventListener("mouseleave", () => {
      this.hovered = null;
      tooltip.hide();
      this.draw();
    });
    new ResizeObserver(() => this.resize()).observe(this.canvas);
    window.addEventListener("themechange", () => this.draw());
  }

  prepare(data) {
    const [id, x, y, name, subfield, field] = [0, 1, 2, 3, 4, 5];
    this.topics = data.topics.map((row) => ({ id: row[id], x: row[x], y: 1 - row[y], name: row[name], subfield: data.subfields[row[subfield]], field: data.fields[row[field]] }));
    this.byId = new Map(this.topics.map((topic) => [topic.id, topic]));
    // a label per field, at the median position of its topics
    const byField = new Map();
    for (const topic of this.topics) (byField.get(topic.field) ?? byField.set(topic.field, []).get(topic.field)).push(topic);
    const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
    this.fieldLabels = [...byField].map(([name, topics]) => ({ name, count: topics.length, x: median(topics.map((t) => t.x)), y: median(topics.map((t) => t.y)) }));
    this.draw();
  }

  reset() {
    this.matches = [];
    this.list.replaceChildren(h("li", { class: "empty" }, "Run a query or replay a recording to see where it lands."));
    this.note.textContent = "";
    this.zoom(false, false);
  }

  onEvent(event, run) {
    if (event.type === "topics") this.showMatches(event.topics);
    if (event.type === "corpus") this.corpus = event.size;
    if (event.type === "filtered") this.note.textContent = `Fetched ${fmt.int(event.fetched)} papers for these topics: ${fmt.int(event.new)} new, ${fmt.int(event.already_present)} already in the corpus, ${fmt.int(event.duplicates)} duplicates, ${fmt.int(event.unusable)} without a usable abstract.`;
  }

  async showMatches(topics) {
    await this.data;
    if (!this.topics) return;
    this.matches = topics.map((match, i) => ({ ...match, rank: i + 1, topic: this.byId.get(match.id) })).filter((match) => match.topic);
    this.list.replaceChildren(
      ...this.matches.map((match) =>
        h(
          "li",
          {
            onmouseenter: () => {
              this.hovered = match.topic;
              this.draw();
            },
            onmouseleave: () => {
              this.hovered = null;
              this.draw();
            },
          },
          h("span", { class: "rank" }, match.rank),
          h(
            "div",
            { class: "topic-text" },
            h("div", { class: "topic-name" }, h("a", { class: "openalex-link", href: openalexUrl(match.id, "T"), target: "_blank", rel: "noopener" }, match.name)),
            h("div", { class: "topic-meta" }, `${match.topic.subfield} · ${match.topic.field}`),
          ),
          h(
            "div",
            { class: "sim" },
            // on an absolute scale (0 to 1), so a weak match looks weak: coral's best is 0.64, RAG's 0.45
            h("div", { class: "sim-track", title: "cosine similarity, 0 to 1" }, h("div", { class: "sim-bar", style: { width: `${match.similarity * 100}%` } })),
            h("span", { class: "sim-value" }, match.similarity.toFixed(2)),
          ),
        ),
      ),
    );
    this.draw();
    // show the whole map first, then zoom in on the matches
    clearTimeout(this.zoomTimer);
    this.zoomTimer = setTimeout(() => this.zoom(true), 900);
  }

  zoom(inward, animate = true) {
    this.zoomed = inward && this.matches.length > 0;
    this.zoomButton.textContent = this.zoomed ? "Whole map" : "Zoom to matches";
    this.zoomButton.disabled = !this.matches.length;
    let target = { x: 0, y: 0, k: 1 };
    if (this.zoomed) {
      const xs = this.matches.map((m) => m.topic.x);
      const ys = this.matches.map((m) => m.topic.y);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const size = Math.min(1, Math.max(x1 - x0, y1 - y0, 0.18) * 1.5);
      target = { x: (x0 + x1) / 2 - size / 2, y: (y0 + y1) / 2 - size / 2, k: 1 / size };
    }
    cancelAnimationFrame(this.animation);
    if (!animate) {
      this.view = target;
      this.draw();
      return;
    }
    const from = { ...this.view };
    const started = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - started) / 700);
      const e = p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
      this.view = { x: from.x + (target.x - from.x) * e, y: from.y + (target.y - from.y) * e, k: from.k + (target.k - from.k) * e };
      this.draw();
      if (p < 1) this.animation = requestAnimationFrame(step);
    };
    this.animation = requestAnimationFrame(step);
  }

  show() {
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width) return;
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.size = { width: rect.width, height: rect.height, ratio };
    this.draw();
  }

  // unit-square coordinates -> canvas pixels
  px(point) {
    const pad = 18;
    const { width, height } = this.size;
    return [pad + (point.x - this.view.x) * this.view.k * (width - 2 * pad), pad + (point.y - this.view.y) * this.view.k * (height - 2 * pad)];
  }

  draw() {
    if (!this.topics || !this.size) return;
    const ctx = this.canvas.getContext("2d");
    const { width, height, ratio } = this.size;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const color = {
      point: cssVar("--map-point"),
      accent: cssVar("--accent"),
      ink: cssVar("--text-primary"),
      label: cssVar("--text-secondary"),
      surface: cssVar("--surface-1"),
    };
    const radius = Math.min(3, 1.4 * Math.sqrt(this.view.k));
    ctx.fillStyle = color.point;
    for (const topic of this.topics) {
      const [x, y] = this.px(topic);
      if (x < -5 || y < -5 || x > width + 5 || y > height + 5) continue;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
    // field labels, with a halo in the surface color
    ctx.font = `${this.view.k > 1.5 ? 12 : 11}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.lineJoin = "round";
    for (const label of this.fieldLabels) {
      if (label.count < 40 && this.view.k < 1.5) continue;
      const [x, y] = this.px(label);
      ctx.lineWidth = 3;
      ctx.strokeStyle = color.surface;
      ctx.strokeText(label.name, x, y);
      ctx.fillStyle = color.label;
      ctx.fillText(label.name, x, y);
    }
    if (this.matches.length) {
      ctx.font = "600 12px system-ui, sans-serif";
      for (const match of this.matches) {
        const [x, y] = this.px(match.topic);
        const r = match.topic === this.hovered ? 8 : 6;
        ctx.beginPath();
        ctx.arc(x, y, r + 2, 0, 2 * Math.PI);
        ctx.fillStyle = color.surface;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color.accent;
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = color.surface;
        ctx.strokeText(match.rank, x + r + 7, y + 4);
        ctx.fillStyle = color.ink;
        ctx.fillText(match.rank, x + r + 7, y + 4);
      }
    }
    if (this.hovered && !this.matches.some((match) => match.topic === this.hovered)) {
      const [x, y] = this.px(this.hovered);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.strokeStyle = color.ink;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  hover(event) {
    if (!this.topics) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    let best = null;
    let bestDistance = 10 ** 2;
    for (const topic of this.topics) {
      const [x, y] = this.px(topic);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestDistance) {
        best = topic;
        bestDistance = d;
      }
    }
    if (best !== this.hovered) {
      this.hovered = best;
      this.draw();
    }
    if (best) {
      const match = this.matches.find((m) => m.topic === best);
      tooltip.show(
        h(
          "div",
          {},
          h("div", { class: "tip-title" }, best.name),
          h("div", { class: "tip-meta" }, `${best.subfield} · ${best.field}`),
          match ? h("div", { class: "tip-meta" }, `match #${match.rank}, cosine similarity ${match.similarity.toFixed(3)}`) : null,
        ),
        event.clientX,
        event.clientY,
      );
    } else tooltip.hide();
  }
}
