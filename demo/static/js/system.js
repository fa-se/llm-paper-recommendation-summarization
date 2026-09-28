// Step "The system": the parts and how they talk (a hand-laid SVG diagram), what the database stores, and the two phases
// of the pipeline (ingest once per interest area, then search per query).

import { h, s, viewHead } from "./util.js";

// a box with a bold title and lines of text; a cylinder for the database
function box({ x, y, w, height, title, lines = [], kind = "" }) {
  const g = s("g", { class: `arch-box ${kind}` });
  if (kind === "db") {
    const ry = 12;
    g.append(
      s("path", { class: "shape", d: `M${x},${y + ry} v${height - 2 * ry} a${w / 2},${ry} 0 0 0 ${w},0 v${-(height - 2 * ry)} a${w / 2},${ry} 0 0 0 ${-w},0 z` }),
      s("path", { class: "rim", d: `M${x},${y + ry} a${w / 2},${ry} 0 0 0 ${w},0` }),
    );
  } else g.append(s("rect", { class: "shape", x, y, width: w, height, rx: 10 }));
  const top = kind === "db" ? y + 2 * 12 + 14 : y + 26;
  g.append(s("text", { class: "arch-title", x: x + 16, y: top }, title));
  lines.forEach((line, i) => g.append(s("text", { class: "arch-line", x: x + 16, y: top + 22 + i * 19 }, line)));
  return g;
}

// an arrow from (x1, y1) to (x2, y2), with a label next to its middle
function arrow(x1, y1, x2, y2, label, { both = false, dx = 8, anchor = "start" } = {}) {
  const g = s("g", { class: "arch-arrow" }, s("line", { x1, y1, x2, y2, "marker-end": "url(#arch-head)", "marker-start": both ? "url(#arch-head-start)" : null }));
  if (label) g.append(s("text", { class: "arch-label", x: (x1 + x2) / 2 + dx, y: (y1 + y2) / 2 + 4, "text-anchor": anchor }, label));
  return g;
}

function chip(x, y, number, title, sub) {
  return s(
    "g",
    { class: "arch-chip" },
    s("rect", { x, y, width: 196, height: 64, rx: 8 }),
    s("circle", { class: "num", cx: x + 20, cy: y + 22, r: 10 }),
    s("text", { class: "num-text", x: x + 20, y: y + 26, "text-anchor": "middle" }, number),
    s("text", { class: "arch-title", x: x + 38, y: y + 27 }, title),
    s("text", { class: "arch-line small", x: x + 14, y: y + 50 }, sub),
  );
}

function diagram() {
  const defs = s(
    "defs",
    {},
    s("marker", { id: "arch-head", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z" })),
    s("marker", { id: "arch-head-start", viewBox: "0 0 10 10", refX: 1, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M10,0 L0,5 L10,10 z" })),
  );
  const chips = [
    ["1", "Topic routing", "description → 10 topics"],
    ["2", "Search", "keyword + meaning"],
    ["3", "LLM ranking", "best of 3, in parallel"],
    ["4", "Summaries", "worksheet, then summary"],
  ];
  const chipX = (i) => 62 + i * 225;
  return s(
    "svg",
    { class: "arch", viewBox: "0 0 1000 520", role: "img", "aria-label": "System architecture" },
    defs,
    box({ x: 40, y: 10, w: 270, height: 82, title: "This page (browser)", lines: ["shows a run's events as they come"] }),
    box({ x: 365, y: 10, w: 270, height: 82, title: "Demo server (FastAPI)", lines: ["runs the pipeline, streams events"] }),
    box({ x: 690, y: 10, w: 270, height: 82, title: "Recorded runs (JSON)", lines: ["replays need no network or API"] }),
    arrow(310, 51, 365, 51, null, { both: true }),
    arrow(635, 51, 690, 51, null),
    arrow(500, 92, 500, 138, "starts a run, gets its events", { dx: 10 }),
    s("rect", { class: "arch-core", x: 40, y: 140, width: 920, height: 176, rx: 12 }),
    s("text", { class: "arch-title", x: 58, y: 166 }, "Python library: the pipeline"),
    ...chips.map(([number, title, sub], i) => chip(chipX(i), 184, number, title, sub)),
    ...[0, 1, 2].map((i) => arrow(chipX(i) + 196, 216, chipX(i + 1), 216, null)),
    s("text", { class: "arch-line small", x: 58, y: 280 }, "ingest (once per interest area): fetch newest papers → drop duplicates → embed abstracts → update keyword index"),
    s("text", { class: "arch-line small", x: 58, y: 301 }, "trace: time, tokens and $ of every step, as events (they drive this page)"),
    arrow(175, 316, 175, 378, "fetch newest papers", { dx: 10 }),
    arrow(500, 316, 500, 378, "store · search", { dx: 10 }),
    arrow(825, 316, 825, 378, "embed · compare · summarize", { dx: -10, anchor: "end" }),
    box({ x: 40, y: 380, w: 270, height: 130, title: "OpenAlex API", lines: ["open catalogue of research", "327 M works, 4,516 topics", "free data (CC0), free API", "Python client: pyalex"] }),
    box({ x: 365, y: 380, w: 270, height: 136, kind: "db", title: "PostgreSQL 18 (Docker)", lines: ["pgvector: meaning search", "VectorChord-bm25: BM25 search", "4,516 topics, 3,509 papers", "no separate vector database"] }),
    box({ x: 690, y: 380, w: 270, height: 130, title: "OpenAI API", lines: ["text-embedding-3-large: vectors", "gpt-6-luna: comparisons", "gpt-6-sol: summaries", "≈ $0.03 per query"] }),
  );
}

export class SystemView {
  constructor(root) {
    root.append(
      viewHead("system", "Under the hood", null),
      h(
        "div",
        { class: "system-grid" },
        h("div", { class: "arch-panel" }, diagram()),
        h(
          "div",
          { class: "system-side" },
          h("h3", { class: "kicker" }, "What the database stores"),
          h(
            "dl",
            { class: "stored" },
            h("dt", {}, "4,516 topics"),
            h("dd", {}, "OpenAlex's research topics: name, description, keywords, and an embedding (1,024 numbers for its meaning). Embedded once, for $0.07."),
            h("dt", {}, "3,509 papers"),
            h("dd", {}, "Title, authors, date, abstract, plus the abstract's embedding and its keyword (BM25) vector. The demo pool: papers published 30 Aug – 26 Sep 2026."),
          ),
          h("h3", { class: "kicker" }, "Two phases"),
          h(
            "dl",
            { class: "stored" },
            h("dt", {}, "Ingest, once per interest area (< 1 min)"),
            h("dd", {}, "Match topics, fetch their newest papers from OpenAlex, drop duplicates, embed the abstracts, update the keyword index. Done beforehand for the demo."),
            h("dt", {}, "Search, every query (~25 s, ~$0.03)"),
            h("dd", {}, "Match topics, search the pool, LLM ranking, summaries: the four steps that follow."),
          ),
          h("h3", { class: "kicker" }, "Along the way"),
          h("p", { class: "side-text" }, "Two fixes merged upstream: pyalex (support for OpenAlex's new Topics, PR #35) and pg_bestmatch.rs (a broken index refresh, PR #15)."),
        ),
      ),
    );
  }
}
