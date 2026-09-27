// Wires the page together: the steps and their navigation, the query controls, the player (a live run over SSE or a
// replay of a recording, both paced stage by stage), the shared run state, the views, and the details drawer.

import { $, GROUPS, STEPS, STEP_LABELS, groupOf, h } from "./util.js";
import { Pipeline } from "./pipeline.js";
import { IntroView } from "./intro.js";
import { SystemView } from "./system.js";
import { TopicsView } from "./topics.js";
import { RetrievalView } from "./retrieval.js";
import { HeapView } from "./heap.js";
import { SummariesView } from "./summaries.js";
import { ResultsView } from "./results.js";
import { TakeawaysView } from "./takeaways.js";
import { details } from "./details.js";

// nested trace stages whose time is already in their parent's (semantic and bm25 run inside hybrid)
const NESTED = new Set(["semantic", "bm25"]);

// What several views need to know about the run so far.
class RunState {
  constructor() {
    this.meta = {}; // the run_start event
    this.works = new Map(); // id -> {id, title, authors?, abstract?, publication_date?, hybridRank?}
    this.topics = []; // the matched topics
    this.corpus = null; // number of papers searched
    this.pool = null; // {oldest, newest}: publication dates of the searched papers (recordings since 2026-09-27)
    this.fetched = null; // the fetched event of a live fetch: how many papers, how far back, per topic
    this.rankings = {}; // method -> results (semantic and bm25: top 100, hybrid: the candidates)
    this.candidates = []; // ids in hybrid order, i.e. the reranker's input
    this.top = []; // ids in final order, growing while the reranker extracts them
    this.rerank = null; // {calls, used}
    this.summaries = new Map(); // id -> {summary, reasoning}
    this.groups = Object.fromEntries(GROUPS.map((g) => [g, { duration: 0, cost: 0, calls: 0, input: 0, output: 0, open: 0, started: false, done: false }]));
    this.t = 0;
    this.cost = 0;
    this.calls = 0;
    this.input = 0;
    this.output = 0;
    this.finished = false;
    this.error = null;
  }

  get started() {
    return Boolean(this.meta.query);
  }

  apply(event) {
    this.t = Math.max(this.t, event.t ?? 0);
    const group = this.groups[groupOf(event.stage)];
    switch (event.type) {
      case "run_start":
        this.meta = event;
        break;
      case "topics":
        this.topics = event.topics;
        break;
      case "corpus":
        this.corpus = event.size;
        if (event.oldest) this.pool = { oldest: event.oldest, newest: event.newest };
        break;
      case "fetched":
        this.fetched = event;
        break;
      case "ranking":
        this.rankings[event.method] = event.results;
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
        this.rerank = { calls: event.calls, used: event.used_calls };
        break;
      case "summary":
        this.summaries.set(event.docid, { summary: event.summary, reasoning: event.reasoning });
        break;
      case "stage_start":
        if (!group) break;
        group.open++;
        group.started = true;
        break;
      case "stage_end":
        if (!group) break;
        group.open--;
        if (!NESTED.has(event.stage)) group.duration += event.duration_s;
        if (group.open === 0) group.done = true;
        break;
      case "llm_call":
        this.cost = event.total_cost_usd;
        this.calls++;
        this.input += event.input_tokens;
        this.output += event.output_tokens;
        if (group) {
          group.cost += event.cost_usd;
          group.calls++;
          group.input += event.input_tokens;
          group.output += event.output_tokens;
        }
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

// Plays a run's events into the page. A live run and a replay work the same way: events go into a buffer (from the SSE
// stream, or all at once from a recording), and a virtual clock plays them with their recorded timing. With "pause
// between stages", the clock stops before each stage, and Next continues; a live run keeps running in the background
// meanwhile, so the page then shows it with a delay.
class Player {
  constructor(app) {
    this.app = app;
    this.session = null;
    this.timer = null;
  }

  get playing() {
    return Boolean(this.session);
  }

  async live(request, { stepwise }) {
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
    const session = this.begin({ mode: "live", events: [], ended: false, speed: 1, stepwise });
    const seen = new Set();
    session.source = new EventSource(`/api/runs/${id}/events`);
    session.source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (seen.has(event.seq)) return;
      seen.add(event.seq);
      session.events.push(event);
      if (this.session === session) this.tick();
    };
    session.source.addEventListener("end", () => {
      session.source.close();
      session.ended = true;
      if (this.session === session) this.tick();
    });
    session.source.onerror = () => {
      // EventSource reconnects by itself (and resumes via Last-Event-ID); give up only if the server is gone for good
      if (session.source.readyState === EventSource.CLOSED && !session.ended) {
        session.ended = true;
        this.app.toast("Lost the connection to the server");
        if (this.session === session) this.tick();
      }
    };
  }

  async play(name, { speed = 1, stepwise = false } = {}) {
    const response = await fetch(`/api/recordings/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`Recording ${name}: HTTP ${response.status}`);
    const { events } = await response.json();
    this.begin({ mode: "replay", name, events, ended: true, speed, stepwise });
  }

  begin(options) {
    this.halt();
    const session = { next: 0, offset: 0, wallStart: performance.now(), paused: false, group: null, pending: null, ...options };
    this.session = session;
    this.app.reset({ mode: session.mode, name: session.name });
    this.tick();
    return session;
  }

  // recorded seconds = offset + (now - wallStart) * speed
  virtualTime() {
    const session = this.session;
    if (!session) return null;
    if (session.paused) return session.offset;
    return session.offset + ((performance.now() - session.wallStart) / 1000) * session.speed;
  }

  // the pipeline stage the event opens, if it opens another one than the current
  nextGroup(event) {
    const group = event.type === "stage_start" ? groupOf(event.stage) : null;
    return group && group !== this.session.group ? group : null;
  }

  tick() {
    clearTimeout(this.timer);
    const session = this.session;
    if (!session || session.paused) return;
    const now = this.virtualTime();
    while (session.next < session.events.length && session.events[session.next].t <= now) {
      const event = session.events[session.next];
      const group = this.nextGroup(event);
      if (group && session.stepwise && session.group) {
        // wait for Next before the next stage starts
        session.paused = true;
        session.pending = group;
        session.offset = event.t;
        this.app.setPaused(group);
        return;
      }
      this.dispatch(event);
    }
    if (session.ended && session.next >= session.events.length) {
      this.finish();
      return;
    }
    this.timer = setTimeout(() => this.tick(), 16);
  }

  dispatch(event) {
    const session = this.session;
    const group = this.nextGroup(event);
    if (group) session.group = group;
    session.next++;
    this.app.dispatch(event);
  }

  resume() {
    const session = this.session;
    if (!session?.paused) return;
    session.paused = false;
    session.pending = null;
    session.wallStart = performance.now();
    this.app.setPaused(null);
    // the event that caused the pause is due now
    this.dispatch(session.events[session.next]);
    this.tick();
  }

  // what Next does to the run: "continue" into the next stage (paused before it, or the current one is complete and the
  // next one is due soon), "finish" the current stage at once (as far as a live run has got), or nothing
  upcoming() {
    const session = this.session;
    if (!session) return null;
    if (session.paused) return { action: "continue", group: session.pending };
    const event = session.events[session.next];
    if (!event) return null;
    const group = this.nextGroup(event);
    return group ? { action: "continue", group } : { action: "finish" };
  }

  advance() {
    const session = this.session;
    const upcoming = this.upcoming();
    if (!upcoming) return;
    if (session.paused) {
      this.resume();
      return;
    }
    if (upcoming.action === "continue") {
      // start the next stage now: move the clock to its start
      session.offset = session.events[session.next].t;
      session.wallStart = performance.now();
      this.dispatch(session.events[session.next]);
      this.tick();
      return;
    }
    let last = null;
    while (session.next < session.events.length && !this.nextGroup(session.events[session.next])) {
      last = session.events[session.next];
      this.dispatch(last);
    }
    session.offset = Math.max(this.virtualTime(), last?.t ?? 0);
    session.wallStart = performance.now();
    this.tick();
  }

  setSpeed(speed) {
    const session = this.session;
    if (!session || session.mode === "live") return;
    if (!session.paused) {
      session.offset = this.virtualTime();
      session.wallStart = performance.now();
    }
    session.speed = speed;
  }

  setStepwise(stepwise) {
    if (this.session) this.session.stepwise = stepwise;
  }

  // everything received so far, at once; a live run then plays on without pauses
  skipToEnd() {
    const session = this.session;
    if (!session) return;
    clearTimeout(this.timer);
    session.stepwise = false;
    if (session.paused) {
      session.paused = false;
      this.app.setPaused(null);
    }
    while (session.next < session.events.length) this.dispatch(session.events[session.next]);
    session.offset = Math.max(session.offset, session.events.at(-1)?.t ?? 0);
    session.wallStart = performance.now();
    this.tick();
  }

  halt() {
    clearTimeout(this.timer);
    this.session?.source?.close();
    this.session = null;
  }

  finish() {
    this.halt();
    this.app.stopped();
  }

  stop() {
    if (!this.session) return;
    this.finish();
  }
}

class App {
  constructor() {
    this.pipeline = new Pipeline($("#stepper"), $("#timeline"), $("#stats"), (step) => this.select(step, true));
    this.views = {
      intro: new IntroView($("#view-intro"), (step) => this.select(step, true)),
      system: new SystemView($("#view-system")),
      topics: new TopicsView($("#view-topics")),
      retrieval: new RetrievalView($("#view-retrieval")),
      rerank: new HeapView($("#view-rerank")),
      summaries: new SummariesView($("#view-summaries")),
      results: new ResultsView($("#view-results")),
      takeaways: new TakeawaysView($("#view-takeaways")),
    };
    this.player = new Player(this);
    this.run = new RunState();
    this.follow = true;
    this.current = null;
    this.drawerOpen = false;
    for (const view of Object.values(this.views)) view.reset?.(this.run);
    this.pipeline.reset(this.run);
    this.select("intro");
    this.clock();
  }

  reset({ mode, name }) {
    this.run = new RunState();
    this.run.mode = mode;
    this.run.recording = name;
    this.follow = $("#follow").checked;
    this.pipeline.reset(this.run);
    for (const view of Object.values(this.views)) view.reset?.(this.run);
    this.setPaused(null);
    this.updateControls();
    collapseQuery(true);
  }

  dispatch(event) {
    this.run.apply(event);
    this.pipeline.onEvent(event, this.run);
    for (const view of Object.values(this.views)) view.onEvent?.(event, this.run);
    if (event.type === "stage_start" && this.follow) {
      const group = groupOf(event.stage);
      if (group && group !== this.current) this.select(group);
    }
    if (event.type === "run_error") this.runFailed(event.message);
    // a replay shows the query it was recorded with
    if (event.type === "run_start" && this.run.mode === "replay") setQuery(event.query_name, event.query, false);
    if (["stage_start", "stage_end", "run_start"].includes(event.type)) this.updateControls();
  }

  select(step, byUser = false) {
    if (!STEPS.includes(step)) return;
    // choosing the stage a paused run waits for continues the run
    const session = this.player.session;
    if (byUser && session?.paused && step === session.pending) {
      this.follow = $("#follow").checked;
      this.player.resume();
      if (this.current !== step) this.select(step);
      return;
    }
    if (byUser && this.player.playing && GROUPS.includes(step)) this.follow = false;
    this.current = step;
    document.body.dataset.step = step;
    for (const name of STEPS) {
      const active = name === step;
      $(`#view-${name}`).hidden = !active;
      if (active) this.views[name].show?.(this.run);
    }
    this.pipeline.select(step);
    // the timeline is hidden on the non-stage steps, and needs its width again when it reappears
    if (GROUPS.includes(step)) requestAnimationFrame(() => this.pipeline.timeline.layout());
    if (this.drawerOpen) this.renderDetails();
    this.updateControls();
  }

  // Next: on the stage the run is at, continue into the next stage or finish the current one's animation; anywhere
  // else (behind the run, or no run), go to the next step
  next() {
    const session = this.player.session;
    const i = STEPS.indexOf(this.current);
    if (session && session.group === this.current) {
      const upcoming = this.player.upcoming();
      if (upcoming) {
        if (upcoming.action === "continue") this.follow = $("#follow").checked;
        this.player.advance();
        this.updateControls();
        return;
      }
      if (!this.run.groups[this.current]?.done) return; // a live stage still running: nothing to skip
    }
    if (i < STEPS.length - 1) this.select(STEPS[i + 1], true);
  }

  back() {
    const i = STEPS.indexOf(this.current);
    if (i > 0) this.select(STEPS[i - 1], true);
  }

  nextLabel() {
    const session = this.player.session;
    const i = STEPS.indexOf(this.current);
    if (session && session.group === this.current) {
      const upcoming = this.player.upcoming();
      if (upcoming?.action === "continue") return { text: `Continue: ${STEP_LABELS[upcoming.group]} ▸`, cue: session.paused };
      if (upcoming?.action === "finish") return { text: "Finish this stage ▸▸" };
      if (!this.run.groups[this.current]?.done) return { text: "Running…", disabled: true };
    }
    if (i < STEPS.length - 1) return { text: `${STEP_LABELS[STEPS[i + 1]]} ▸` };
    return { text: "End", disabled: true };
  }

  stopped() {
    this.pipeline.finish(this.run);
    for (const view of Object.values(this.views)) view.finish?.(this.run);
    this.setPaused(null);
    this.updateControls();
    if (this.drawerOpen) this.renderDetails();
    if (this.run.mode === "live" && !this.run.error) lastLiveRunEnd = Date.now();
    loadRecordings();
    loadCorpus();
  }

  setPaused(group) {
    this.pipeline.pending(group);
    $("#status").textContent = group ? (this.run.mode === "live" ? "Paused · the live run goes on in the background" : `Paused before ${STEP_LABELS[group]}`) : "";
    this.updateControls();
  }

  runFailed(message) {
    const recording = recordingFor(queryName);
    this.toast(`Run failed: ${message}`, recording ? { label: "Play the recording instead", onclick: () => startReplay(recording) } : null);
  }

  updateControls() {
    const session = this.player.session;
    $("#run").disabled = Boolean(session);
    $("#fetch").disabled = Boolean(session);
    $("#replay").disabled = Boolean(session) || !recordingList.length;
    $("#skip").hidden = !session;
    $("#stop").hidden = !session;
    $("#follow").checked = this.follow;
    const next = this.nextLabel();
    const button = $("#next");
    button.textContent = next.text;
    button.disabled = Boolean(next.disabled);
    button.classList.toggle("cue", Boolean(next.cue));
    $("#back").disabled = this.current === STEPS[0];
    // stage and result views stay empty until a run has started
    const needsRun = !["intro", "system", "takeaways"].includes(this.current);
    $("#no-run").hidden = !needsRun || this.run.started || Boolean(session);
    $("#no-run-replay").disabled = !recordingList.length;
  }

  // the elapsed-time display runs on the player's clock between events
  clock() {
    const update = () => {
      if (this.player.playing && !this.run.finished) {
        this.pipeline.setClock(Math.max(this.player.virtualTime() ?? 0, this.run.t), this.run);
        // the Next button's label depends on what the buffer holds
        const label = this.nextLabel().text;
        if (label !== $("#next").textContent) this.updateControls();
      }
      requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }

  toggleDetails(open = !this.drawerOpen) {
    this.drawerOpen = open;
    $("#drawer").hidden = !open;
    document.body.classList.toggle("drawer-open", open);
    if (open) this.renderDetails();
  }

  renderDetails() {
    const { title, body } = details(this.current, this.run);
    $("#drawer-title").textContent = title;
    $("#drawer-body").replaceChildren(body);
    $("#drawer-body").scrollTop = 0;
  }

  toast(message, action) {
    const toast = $("#toast");
    toast.replaceChildren(
      h("span", {}, message),
      action
        ? h(
            "button",
            {
              class: "small",
              onclick: () => {
                toast.hidden = true;
                action.onclick();
              },
            },
            action.label,
          )
        : null,
    );
    toast.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (toast.hidden = true), action ? 15000 : 8000);
  }
}

// --- controls ---

// declared before the App, whose constructor already reads them
let config = null;
let recordingList = [];
let lastLiveRunEnd = 0;
const LABELS = { coral_reefs: "Coral reefs & climate", rag_hallucinations: "RAG hallucinations" };
let queryName = null;

const app = new App();

async function loadConfig() {
  try {
    config = await (await fetch("/api/config")).json();
  } catch {
    app.toast("Server not reachable");
    return;
  }
  $("#query-chips").replaceChildren(
    ...Object.entries(config.queries).map(([name, text]) =>
      h("button", { class: "chip", type: "button", "data-name": name, onclick: () => setQuery(name, text) }, LABELS[name] ?? name),
    ),
    h(
      "button",
      {
        class: "chip",
        type: "button",
        "data-name": "",
        onclick: () => {
          setQuery(null, "");
          collapseQuery(false);
        },
      },
      "Your own",
    ),
  );
  const [name, text] = Object.entries(config.queries).find(([key]) => key === "rag_hallucinations") ?? Object.entries(config.queries)[0];
  setQuery(name, text);
}

function setQuery(name, text, focus = true) {
  queryName = name;
  const textarea = $("#query");
  textarea.value = text;
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("active", chip.dataset.name === (name ?? ""));
  if (!name && focus) textarea.focus();
  // a custom query is probably outside the pre-fetched demo pool: fetch the newest papers of its topics first
  $("#fetch").value = name ? "0" : "500";
  selectRecording();
}

// the query bar shows one line of the description while a run is on screen; click it to see all of it
function collapseQuery(collapsed) {
  $("#query-bar").classList.toggle("collapsed", collapsed);
  $("#query").rows = collapsed ? 1 : 4;
  $("#expand-query").textContent = collapsed ? "▾" : "▴";
  $("#expand-query").title = collapsed ? "Show the whole description" : "Show one line";
}

$("#expand-query").addEventListener("click", () => collapseQuery(!$("#query-bar").classList.contains("collapsed")));
$("#query").addEventListener("focus", () => collapseQuery(false));

$("#query").addEventListener("input", () => {
  // edited text is a custom query, unless it's still a demo query verbatim
  const text = $("#query").value;
  const match = Object.entries(config?.queries ?? {}).find(([, query]) => query === text);
  queryName = match ? match[0] : null;
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("active", chip.dataset.name === (queryName ?? ""));
  selectRecording();
});

// curated recordings (named after their query) first, then the auto-saved ones, newest first
async function loadRecordings() {
  try {
    recordingList = await (await fetch("/api/recordings")).json();
  } catch {
    return;
  }
  const select = $("#recordings");
  const previous = select.value;
  recordingList.sort((a, b) => Number(/^\d/.test(a.name)) - Number(/^\d/.test(b.name)));
  select.replaceChildren(
    ...recordingList.map((recording) => {
      const label = [
        LABELS[recording.query_name] ?? recording.query_name ?? "custom query",
        /^\d/.test(recording.name) ? recording.name.replace(/^\d{4}-(\d\d)-(\d\d)_(\d\d)(\d\d).*/, "$2.$1. $3:$4") : "curated",
        recording.duration_s ? `${recording.duration_s.toFixed(0)} s` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      return h("option", { value: recording.name }, label);
    }),
  );
  if (recordingList.some((recording) => recording.name === previous)) select.value = previous;
  else selectRecording();
  app.updateControls();
}

// the recording to replay for a query: its curated one, else its newest
function recordingFor(name) {
  if (!name) return null;
  return recordingList.find((r) => r.name === name)?.name ?? recordingList.find((r) => r.query_name === name)?.name ?? null;
}

function selectRecording() {
  const name = recordingFor(queryName);
  if (name) $("#recordings").value = name;
}

const stepwise = () => $("#stepwise").checked;

async function startLive() {
  const query = $("#query").value.trim();
  if (query.length < 20) {
    app.toast("Describe the research interest in a sentence or two");
    collapseQuery(false);
    return;
  }
  const since = (Date.now() - lastLiveRunEnd) / 1000;
  try {
    await app.player.live({ query, query_name: queryName, fetch_per_topic: Number($("#fetch").value), n: 5, summaries: 3 }, { stepwise: stepwise() });
    if (since < 60) app.toast(`The last live run ended ${Math.round(since)} s ago: this one may hit the API's rate limit, and the ranking then takes longer (it retries).`);
  } catch (error) {
    const recording = recordingFor(queryName);
    app.toast(`Could not start the run: ${error.message}`, recording ? { label: "Play the recording instead", onclick: () => startReplay(recording) } : null);
  }
  app.updateControls();
}

async function startReplay(name = $("#recordings").value) {
  if (!name) return;
  try {
    await app.player.play(name, { speed: Number($("#speed").value), stepwise: stepwise() });
  } catch (error) {
    app.toast(error.message);
  }
  app.updateControls();
}

$("#run").addEventListener("click", startLive);
$("#replay").addEventListener("click", () => startReplay());
$("#no-run-live").addEventListener("click", startLive);
$("#no-run-replay").addEventListener("click", () => startReplay());
$("#next").addEventListener("click", () => app.next());
$("#back").addEventListener("click", () => app.back());
$("#skip").addEventListener("click", () => app.player.skipToEnd());
$("#stop").addEventListener("click", () => app.player.stop());

// details drawer: the Details button of each view, or the i key
document.addEventListener("click", (event) => {
  if (event.target.closest("[data-details]")) app.toggleDetails();
});
$("#drawer-close").addEventListener("click", () => app.toggleDetails(false));

// settings popover
function toggleSettings(open = $("#settings").hidden) {
  $("#settings").hidden = !open;
  $("#settings-button").setAttribute("aria-expanded", String(open));
}
$("#settings-button").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleSettings();
});
document.addEventListener("click", (event) => {
  if (!$("#settings").hidden && !event.target.closest("#settings")) toggleSettings(false);
});

// settings that persist across reloads
const PREFS = { stepwise: "demo.stepwise", "show-timeline": "demo.timeline", speed: "demo.speed" };
function loadPrefs() {
  try {
    for (const [id, key] of Object.entries(PREFS)) {
      const value = localStorage.getItem(key);
      if (value === null) continue;
      const input = $(`#${id}`);
      if (input.type === "checkbox") input.checked = value === "1";
      else input.value = value;
    }
  } catch {}
  document.body.classList.toggle("no-timeline", !$("#show-timeline").checked);
}
function savePref(id) {
  const input = $(`#${id}`);
  try {
    localStorage.setItem(PREFS[id], input.type === "checkbox" ? (input.checked ? "1" : "0") : input.value);
  } catch {}
}
loadPrefs();

$("#speed").addEventListener("change", () => {
  app.player.setSpeed(Number($("#speed").value));
  savePref("speed");
});
$("#stepwise").addEventListener("change", () => {
  app.player.setStepwise(stepwise());
  if (!stepwise() && app.player.session?.paused) app.player.resume();
  savePref("stepwise");
});
$("#follow").addEventListener("change", (event) => (app.follow = event.target.checked));
$("#show-timeline").addEventListener("change", () => {
  document.body.classList.toggle("no-timeline", !$("#show-timeline").checked);
  app.pipeline.timeline.layout();
  savePref("show-timeline");
});

// the corpus: its size, and a reset that removes what live runs with "fetch new papers first" added
async function loadCorpus() {
  let status;
  try {
    status = await (await fetch("/api/corpus")).json();
  } catch {
    return;
  }
  const button = $("#reset-corpus");
  if (status.error) {
    $("#corpus").textContent = "database not reachable: replays only";
    button.hidden = true;
    return;
  }
  app.views.intro.setCorpus?.(status.rows, status.oldest && { oldest: status.oldest, newest: status.newest });
  $("#corpus").textContent = `corpus: ${status.rows.toLocaleString("en-US")} papers${status.added ? `, ${status.added.toLocaleString("en-US")} of them from live fetches` : ""}`;
  button.hidden = !status.added;
  button.dataset.added = status.added ?? 0;
  resetArmed(false);
}

const papers = (n) => `${n.toLocaleString("en-US")} paper${n === 1 ? "" : "s"}`;

// a reset deletes papers, so it takes a second click
function resetArmed(armed) {
  const button = $("#reset-corpus");
  button.dataset.armed = armed ? "1" : "";
  const n = Number(button.dataset.added);
  button.textContent = armed ? `Click again: delete ${papers(n)}` : "Reset corpus";
  button.classList.toggle("danger", armed);
  clearTimeout(resetArmed.timer);
  if (armed) resetArmed.timer = setTimeout(() => resetArmed(false), 4000);
}

$("#reset-corpus").addEventListener("click", async () => {
  const button = $("#reset-corpus");
  if (!button.dataset.armed) {
    resetArmed(true);
    return;
  }
  resetArmed(false);
  button.disabled = true;
  button.textContent = "Resetting…";
  try {
    const response = await fetch("/api/corpus/reset", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`);
    app.toast(
      result.exact
        ? `Removed ${papers(result.deleted)} and rebuilt the BM25 index: the corpus is the demo baseline again.`
        : `Removed ${papers(result.deleted)}, but the corpus differs from the baseline (${papers(result.rows)}).`,
    );
  } catch (error) {
    app.toast(`Reset failed: ${error.message}`);
  }
  button.disabled = false;
  loadCorpus();
});

// keyboard: → / space / PgDn next (a presentation clicker sends PgDn/PgUp), ← / PgUp back, 1-4 stages, i details
document.addEventListener("keydown", (event) => {
  if (event.target.matches("textarea, input, select") || event.altKey || event.ctrlKey || event.metaKey) return;
  if (["ArrowRight", " ", "PageDown"].includes(event.key)) {
    event.preventDefault();
    // a focused button would also take the space as a click
    if (event.target.closest("button")) event.target.blur();
    app.next();
  } else if (["ArrowLeft", "PageUp"].includes(event.key)) {
    event.preventDefault();
    app.back();
  } else if (/^[1-4]$/.test(event.key)) {
    app.select(GROUPS[Number(event.key) - 1], true);
  } else if (event.key === "i") {
    app.toggleDetails();
  } else if (event.key === "Escape") {
    if (!$("#settings").hidden) toggleSettings(false);
    else app.toggleDetails(false);
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
loadCorpus();
