// Small DOM helpers, formatting, the shared tooltip, and the mapping of pipeline stages to the page's views.

export const $ = (selector, root = document) => root.querySelector(selector);

// h("div", {class: "x", onclick: fn}, "text", child) -> element
export function h(tag, attrs = {}, ...children) {
  return build(document.createElement(tag), attrs, children);
}

export function s(tag, attrs = {}, ...children) {
  return build(document.createElementNS("http://www.w3.org/2000/svg", tag), attrs, children);
}

function build(element, attrs, children) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else if (key === "style" && typeof value === "object") Object.assign(element.style, value);
    else element.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

export const fmt = {
  usd: (value) => (value < 0.001 ? `$${value.toFixed(5)}` : value < 0.1 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`),
  seconds: (value) => (value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(1)} s`),
  int: (value) => Math.round(value).toLocaleString("en-US"),
  compact: (value) => (value >= 10000 ? `${(value / 1000).toFixed(0)}k` : value >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${value}`),
};

export const words = (text) => (text ? text.trim().split(/\s+/).length : 0);

// the steps of the page in presenting order: two intro steps, the four pipeline stages, and two closing steps
export const STEPS = ["intro", "system", "topics", "retrieval", "rerank", "summaries", "results", "takeaways"];
export const STEP_LABELS = {
  intro: "The idea",
  system: "The system",
  topics: "Topic routing",
  retrieval: "Search",
  rerank: "LLM ranking",
  summaries: "Summaries",
  results: "Results",
  takeaways: "Takeaways",
};

// the pipeline stages among the steps; each groups the trace stages it shows
export const GROUPS = ["topics", "retrieval", "rerank", "summaries"];
const STAGE_GROUP = {
  topics: "topics",
  fetch: "topics",
  filter: "topics",
  embed: "topics",
  index: "topics",
  hybrid: "retrieval",
  semantic: "retrieval",
  bm25: "retrieval",
  load: "retrieval",
  rerank: "rerank",
  summarize: "summaries",
};
export const groupOf = (stage) => STAGE_GROUP[stage] ?? null;

// colors of the final top-k papers, by final rank; the same paper keeps its color in every view
export const rankColor = (rank) => (rank >= 1 && rank <= 5 ? `var(--series-${rank})` : null);

export const tooltip = {
  element: null,
  show(content, x, y) {
    this.element ??= $("#tooltip");
    this.element.replaceChildren(content instanceof Node ? content : document.createTextNode(content));
    this.element.hidden = false;
    const { innerWidth, innerHeight } = window;
    const box = this.element.getBoundingClientRect();
    const left = x + 16 + box.width > innerWidth ? x - 16 - box.width : x + 16;
    const top = y + 16 + box.height > innerHeight ? Math.max(8, y - 16 - box.height) : y + 16;
    this.element.style.left = `${Math.max(8, left)}px`;
    this.element.style.top = `${top}px`;
  },
  hide() {
    if (this.element) this.element.hidden = true;
  },
};

// OpenAlex pages of works ("W") and topics ("T"); the trace carries ids without the prefix
export const openalexUrl = (id, prefix = "W") => `https://openalex.org/${prefix}${id}`;

// a paper's title (or other content) as a link to its OpenAlex page, opened in a new tab
export function paperLink(id, ...children) {
  return h("a", { class: "openalex-link", href: openalexUrl(id), target: "_blank", rel: "noopener" }, ...children);
}

// a click anywhere on the element opens the paper too, except on a link inside it or when text was selected
export function openOnClick(element, id) {
  element.classList.add("clickable");
  element.addEventListener("click", (event) => {
    if (event.target.closest("a") || getSelection()?.toString()) return;
    window.open(openalexUrl(id), "_blank", "noopener");
  });
}

// tooltip content for a paper; every element that shows it opens the paper on click
export function paperTip(work, extra) {
  const meta = [work.authors?.length ? work.authors.join(", ") : null, work.publication_date].filter(Boolean).join(" · ");
  const abstract = work.abstract ? (work.abstract.length > 420 ? `${work.abstract.slice(0, 420)}…` : work.abstract) : null;
  return h(
    "div",
    {},
    h("div", { class: "tip-title" }, work.title ?? `W${work.id}`),
    meta ? h("div", { class: "tip-meta" }, meta) : null,
    extra ? h("div", { class: "tip-meta" }, extra) : null,
    abstract ? h("div", { class: "tip-body" }, abstract) : null,
    h("div", { class: "tip-hint" }, `Click to open W${work.id} in OpenAlex ↗`),
  );
}

export function attachTip(element, content) {
  element.addEventListener("mousemove", (event) => tooltip.show(typeof content === "function" ? content() : content, event.clientX, event.clientY));
  element.addEventListener("mouseleave", () => tooltip.hide());
}

// the head of a step's view: plain-language title and lede, and the button that opens the step's technical details
export function viewHead(step, title, lede) {
  const number = GROUPS.indexOf(step) + 1;
  return h(
    "header",
    { class: "view-head" },
    h("div", { class: "view-titles" }, h("h2", {}, number ? h("span", { class: "view-num" }, number) : null, title), lede ? h("p", { class: "lede" }, lede) : null),
    h("button", { class: "info-button", type: "button", "data-details": step, title: "Technical details and likely questions (key i)" }, h("span", { class: "info-icon" }, "i"), "Details"),
  );
}

export function cssVar(name, element = document.body) {
  return getComputedStyle(element).getPropertyValue(name).trim();
}
