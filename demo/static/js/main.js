// Wires the page together: the query controls, the event sources (a live run over SSE, or a replay of a recording),
// the shared run state, and the views that render the events.

import { $, GROUPS, groupOf, h } from "./util.js";
import { Pipeline } from "./pipeline.js";
import { TopicsView } from "./topics.js";
import { RetrievalView } from "./retrieval.js";
import { HeapView } from "./heap.js";
import { SummariesView } from "./summaries.js";

// What several views need to know about the run so far.
class RunState {
  constructor() {
    this.meta = {}; // the run_start event
    this.works = new Map(); // id -> {id, title, authors?, abstract?, publication_date?}
    this.candidates = []; // ids in hybrid order, i.e. the reranker's input
    this.top = []; // ids in final order, growing while the reranker extracts them
    this.t = 0;
    this.cost = 0;
    this.finished = false;
    this.error = null;
  }

  apply(event) {
    this.t = Math.max(this.t, event.t ?? 0);
    switch (event.type) {
      case "run_start":
        this.meta = event;
        break;
      case "ranking":
        for (const result of event.results) this.works.set(result.id, { id: result.id, title: result.title, ...this.works.get(result.id) });
        break;
      case "candidates":
        this.candidates = event.works.map((work) => work.id);
        event.works.forEach((work, i) => this.works.set(work.id, { ...this.works.get(work.id), ...work, hybridRank: i + 1 }));
        break;
      case "ranked":
        if (!this.top.includes(event.docid)) this.top.push(event.docid);
        break;
      case "reranked":
        this.top = event.docids;
        break;
      case "llm_call":
        this.cost = event.total_cost_usd;
        break;
      case "run_end":
        this.finished = true;
        break;
      case "run_error":
        this.finished = true;
        this.error = event.message;
        break;
    }
  }

  finalRank(id) {
    const i = this.top.indexOf(id);
    return i >= 0 ? i + 1 : null;
  }

  work(id) {
    return this.works.get(id) ?? { id, title: `W${id}` };
  }
}

// Plays events into the views, from a live run or a recording.
class Player {
  constructor(app) {
    this.app = app;
    this.source = null; // EventSource of a live run
    this.replay = null; // state of a replay
    this.clockStart = null;
    this.speed = 1;
  }

  get playing() {
    return Boolean(this.source || this.replay);
  }

  async live(request) {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `HTTP ${response.status}`);
    }
    const { id } = await response.json();
    this.app.reset({ mode: "live" });
    this.speed = 1;
    this.clockStart = performance.now();
    const seen = new Set();
    this.source = new EventSource(`/api/runs/${id}/events`);
    this.source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (seen.has(event.seq)) return;
      seen.add(event.seq);
      this.app.dispatch(event);
    };
    this.source.addEventListener("end", () => this.stop());
    this.source.onerror = () => {
      // EventSource reconnects by itself (and resumes via Last-Event-ID); give up only if the server is gone for good
      if (this.source?.readyState === EventSource.CLOSED) {
        this.app.toast("Lost the connection to the server");
        this.stop();
      }
    };
  }

  async play(name, { speed = 1, stepwise = false } = {}) {
    const response = await fetch(`/api/recordings/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`Recording ${name}: HTTP ${response.status}`);
    const { events } = await response.json();
    this.app.reset({ mode: "replay", name });
    this.speed = speed;
    // the virtual clock: recorded seconds = offset + (now - wallStart) * speed
    this.replay = { events, next: 0, offset: 0, wallStart: performance.now(), stepwise, paused: false, group: null };
    this.clockStart = performance.now();
    this.tick();
  }

  virtualTime() {
    const replay = this.replay;
    if (!replay) return null;
    if (replay.paused) return replay.offset;
    return replay.offset + ((performance.now() - replay.wallStart) / 1000) * this.speed;
  }

  tick() {
    const replay = this.replay;
    if (!replay || replay.paused) return;
    const now = this.virtualTime();
    while (replay.next < replay.events.length && replay.events[replay.next].t <= now) {
      const event = replay.events[replay.next];
      const group = event.type === "stage_start" ? groupOf(event.stage) : null;
      if (replay.stepwise && group && replay.group && group !== replay.group) {
        // stage by stage: wait for "Continue" before the next view's stage starts
        replay.group = group;
        replay.paused = true;
        replay.offset = event.t;
        this.app.setPaused(true, group);
        return;
      }
      if (group) replay.group = group;
      this.app.dispatch(event);
      replay.next++;
    }
    if (replay.next >= replay.events.length) {
      this.stop();
      return;
    }
    this.timer = setTimeout(() => this.tick(), 16);
  }

  resume() {
    const replay = this.replay;
    if (!replay?.paused) return;
    replay.paused = false;
    replay.wallStart = performance.now();
    this.app.setPaused(false);
    // the event that caused the pause is due now
    const event = replay.events[replay.next];
    this.app.dispatch(event);
    replay.next++;
    this.tick();
  }

  setSpeed(speed) {
    if (this.replay && !this.replay.paused) {
      this.replay.offset = this.virtualTime();
      this.replay.wallStart = performance.now();
    }
    this.speed = speed;
  }

  skipToEnd() {
    const replay = this.replay;
    if (!replay) return;
    clearTimeout(this.timer);
    while (replay.next < replay.events.length) this.app.dispatch(replay.events[replay.next++]);
    this.stop();
  }

  stop() {
    clearTimeout(this.timer);
    this.source?.close();
    this.source = null;
    this.replay = null;
    this.app.stopped();
  }
}

class App {
  constructor() {
    this.pipeline = new Pipeline($("#pipeline"), $("#timeline"), $("#stats"), (group) => this.select(group, true));
    this.views = {
      topics: new TopicsView($("#view-topics")),
      retrieval: new RetrievalView($("#view-retrieval")),
      rerank: new HeapView($("#view-rerank")),
      summaries: new SummariesView($("#view-summaries")),
    };
    this.player = new Player(this);
    this.run = new RunState();
    this.follow = true;
    this.current = null;
    this.select("topics");
    for (const view of Object.values(this.views)) view.reset(this.run);
    this.pipeline.reset(this.run);
    this.clock();
  }

  reset({ mode, name }) {
    this.run = new RunState();
    this.run.mode = mode;
    this.run.recording = name;
    this.follow = true;
    this.pipeline.reset(this.run);
    for (const view of Object.values(this.views)) view.reset(this.run);
    this.select("topics");
    this.updateControls();
  }

  dispatch(event) {
    this.run.apply(event);
    this.pipeline.onEvent(event, this.run);
    for (const view of Object.values(this.views)) view.onEvent(event, this.run);
    if (event.type === "stage_start" && this.follow) {
      const group = groupOf(event.stage);
      if (group && group !== this.current) this.select(group);
    }
    if (event.type === "run_error") this.toast(`Run failed: ${event.message}`);
    // a replay shows the query it was recorded with
    if (event.type === "run_start" && this.run.mode === "replay") setQuery(event.query_name, event.query, false);
  }

  select(group, byUser = false) {
    if (byUser && this.player.playing) this.follow = false;
    this.current = group;
    for (const name of GROUPS) {
      const active = name === group;
      $(`#view-${name}`).hidden = !active;
      if (active) this.views[name].show?.(this.run);
    }
    this.pipeline.select(group);
  }

  stopped() {
    this.pipeline.finish(this.run);
    for (const view of Object.values(this.views)) view.finish?.(this.run);
    this.setPaused(false);
    this.updateControls();
    loadRecordings();
  }

  setPaused(paused, group) {
    $("#continue").hidden = !paused;
    $("#status").textContent = paused ? `Paused before “${$(`[data-group="${group}"] .stage-name`)?.textContent ?? group}”` : "";
  }

  updateControls() {
    const playing = this.player.playing;
    $("#run").disabled = playing;
    $("#replay").disabled = playing;
    $("#skip").hidden = !this.player.replay;
    $("#stop").hidden = !this.player.replay;
    $("#follow").checked = this.follow;
  }

  // the elapsed-time display runs on its own clock between events
  clock() {
    const update = () => {
      if (this.player.playing && !this.run.finished) {
        const t = this.player.replay ? this.player.virtualTime() : (performance.now() - this.player.clockStart) / 1000;
        this.pipeline.setClock(Math.max(t ?? 0, this.run.t), this.run);
      }
      requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }

  toast(message) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (toast.hidden = true), 8000);
  }
}

// --- controls ---

const app = new App();
let config = null;

async function loadConfig() {
  try {
    config = await (await fetch("/api/config")).json();
  } catch {
    app.toast("Server not reachable");
    return;
  }
  const chips = $("#query-chips");
  chips.replaceChildren(
    ...Object.entries(config.queries).map(([name, text]) =>
      h("button", { class: "chip", "data-name": name, onclick: () => setQuery(name, text) }, LABELS[name] ?? name),
    ),
    h("button", { class: "chip", "data-name": "", onclick: () => setQuery(null, "") }, "Your own"),
  );
  const [name, text] = Object.entries(config.queries).find(([key]) => key === "rag_hallucinations") ?? Object.entries(config.queries)[0];
  setQuery(name, text);
}

const LABELS = { coral_reefs: "Coral reefs & climate", rag_hallucinations: "RAG hallucinations" };
let queryName = null;

function setQuery(name, text, focus = true) {
  queryName = name;
  const textarea = $("#query");
  textarea.value = text;
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("active", chip.dataset.name === (name ?? ""));
  if (!name && focus) textarea.focus();
  // a custom query is probably outside the pre-ingested corpus
  $("#ingest").checked = !name;
}

async function loadRecordings() {
  let list = [];
  try {
    list = await (await fetch("/api/recordings")).json();
  } catch {
    return;
  }
  const select = $("#recordings");
  const previous = select.value;
  // curated recordings (named after their query) first, then the auto-saved ones, newest first
  list.sort((a, b) => Number(/^\d/.test(a.name)) - Number(/^\d/.test(b.name)));
  select.replaceChildren(
    ...list.map((recording) => {
      const label = [
        LABELS[recording.query_name] ?? recording.query_name ?? "custom query",
        /^\d/.test(recording.name) ? recording.name.replace(/^\d{4}-(\d\d)-(\d\d)_(\d\d)(\d\d).*/, "$2.$1. $3:$4") : "recorded",
        recording.duration_s ? `${recording.duration_s.toFixed(0)} s` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      return h("option", { value: recording.name }, label);
    }),
  );
  if (list.some((recording) => recording.name === previous)) select.value = previous;
  $("#replay").disabled = !list.length || app.player.playing;
}

$("#query").addEventListener("input", () => {
  // edited text is a custom query, unless it's still a demo query verbatim
  const text = $("#query").value;
  const match = Object.entries(config?.queries ?? {}).find(([, query]) => query === text);
  queryName = match ? match[0] : null;
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("active", chip.dataset.name === (queryName ?? ""));
});

$("#run").addEventListener("click", async () => {
  const query = $("#query").value.trim();
  if (query.length < 20) {
    app.toast("Describe the research interest in a sentence or two");
    return;
  }
  try {
    await app.player.live({ query, query_name: queryName, ingest: $("#ingest").checked, n: 5, summaries: 3 });
  } catch (error) {
    app.toast(`Could not start the run: ${error.message}`);
  }
  app.updateControls();
});

$("#replay").addEventListener("click", async () => {
  const name = $("#recordings").value;
  if (!name) return;
  try {
    await app.player.play(name, { speed: Number($("#speed").value), stepwise: $("#stepwise").checked });
  } catch (error) {
    app.toast(error.message);
  }
  app.updateControls();
});

$("#speed").addEventListener("change", () => app.player.setSpeed(Number($("#speed").value)));
$("#continue").addEventListener("click", () => app.player.resume());
$("#skip").addEventListener("click", () => app.player.skipToEnd());
$("#stop").addEventListener("click", () => app.player.stop());
$("#follow").addEventListener("change", (event) => (app.follow = event.target.checked));

// keyboard: space continues a paused replay, 1-4 select the views
document.addEventListener("keydown", (event) => {
  if (event.target.matches("textarea, input, select")) return;
  if (event.key === " " && app.player.replay?.paused) {
    event.preventDefault();
    app.player.resume();
  } else if (/^[1-4]$/.test(event.key)) {
    app.select(GROUPS[Number(event.key) - 1], true);
    $("#follow").checked = app.follow;
  }
});

// theme: follows the OS unless toggled; the choice is remembered
const THEMES = ["auto", "light", "dark"];
function applyTheme(theme) {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $("#theme").textContent = { auto: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[theme];
  window.dispatchEvent(new Event("themechange"));
}
let theme = "auto";
try {
  theme = localStorage.getItem("theme") ?? "auto";
} catch {}
applyTheme(theme);
$("#theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try {
    localStorage.setItem("theme", theme);
  } catch {}
  applyTheme(theme);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => window.dispatchEvent(new Event("themechange")));

loadConfig();
loadRecordings();
