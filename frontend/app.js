/* ===========================================================================
   RAG Lab front-end logic.
   Calls the backend one stage at a time, animates the pipeline, and renders
   each stage's real output in the inspector so the concepts are visible.
   =========================================================================== */

const $ = (sel) => document.querySelector(sel);
const stageEls = {};
document.querySelectorAll(".stage").forEach((el) => (stageEls[el.dataset.stage] = el));

// Cache the latest output of each stage so clicking a node re-shows it.
const results = {};

/* ---- small helpers ------------------------------------------------------ */
async function api(path, body, isForm = false) {
  const opts = { method: "POST" };
  if (isForm) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || "Request failed");
  }
  return res.json();
}

function setStage(stage, statusClass) {
  const el = stageEls[stage];
  if (!el) return;
  el.classList.remove("active");
  if (statusClass === "active") el.classList.add("active");
  if (statusClass === "done") el.classList.add("done");
}

function selectStage(stage) {
  Object.values(stageEls).forEach((el) => el.classList.remove("selected"));
  if (stageEls[stage]) stageEls[stage].classList.add("selected");
}

function toast(msg, isErr = false) {
  let t = $(".toast");
  if (!t) {
    t = document.createElement("div");
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/* ---- inspector rendering, one renderer per stage ------------------------ */
function showInspector(stage, title, tag, html) {
  $("#inspectorTitle").textContent = title;
  $("#inspectorTag").textContent = tag || "";
  $("#inspectorBody").innerHTML = html;
  selectStage(stage);
}

function metric(val, lab) {
  return `<div class="metric"><div class="m-val">${esc(val)}</div><div class="m-lab">${esc(lab)}</div></div>`;
}

function renderLoad(d) {
  const html = `
    <div class="metrics">
      ${metric(d.characters, "characters")}
      ${metric(d.words, "words")}
      ${metric(d.source, "source")}
    </div>
    <div class="section-label">Preview of raw text</div>
    <div class="chunk">${esc(d.preview)}…</div>`;
  showInspector("load", "Loaded text", "step 01", html);
}

function renderChunk(d) {
  const chunks = d.chunks
    .map(
      (c) => `<div class="chunk"><div class="c-head">chunk ${c.id} · ${c.length} chars</div>${esc(c.text)}</div>`
    )
    .join("");
  const html = `
    <div class="metrics">
      ${metric(d.chunk_count, "chunks")}
      ${metric(d.chunk_size, "size")}
      ${metric(d.chunk_overlap, "overlap")}
    </div>
    <p class="hint">Each piece becomes one searchable unit. Notice how overlap repeats text across boundaries.</p>
    ${chunks}`;
  showInspector("chunk", "Chunks", "step 02", html);
}

function renderEmbed(d) {
  const html = `
    <div class="metrics">
      ${metric(d.chunks_embedded, "vectors")}
      ${metric(d.vector_dimensions, "dimensions")}
      ${metric(d.seconds + "s", "time")}
    </div>
    <p class="hint">Every chunk is now a point in ${d.vector_dimensions}-dimensional space, stored in Chroma. Model: <code>${esc(d.model)}</code></p>
    <div class="section-label">First 12 numbers of chunk ${d.sample_chunk_id}'s vector</div>
    <div class="vector">[ ${d.sample_vector_preview.join(", ")} … ]</div>`;
  showInspector("embed", "Embeddings stored", "steps 03–04", html);
  // Step 04 (store) shares this result — mark both done.
  setStage("store", "done");
}

function renderRetrieve(d) {
  const items = d.retrieved
    .map((r) => {
      const pct = Math.max(0, Math.min(100, Math.round(r.similarity * 100)));
      return `<div class="chunk matched">
        <div class="c-head">chunk ${r.chunk_id}<span class="score">sim ${r.similarity}</span></div>
        ${esc(r.text)}
        <div class="bar"><i style="width:${pct}%"></i></div>
      </div>`;
    })
    .join("");
  const html = `
    <div class="section-label">Question</div>
    <div class="chunk">${esc(d.question)}</div>
    <p class="hint">The question was embedded and matched against the stored vectors. These ${d.top_k} chunks are closest in meaning — no LLM involved yet.</p>
    ${items}`;
  showInspector("retrieve", "Retrieved context", "step 05", html);
}

function renderGenerate(d) {
  const matched = d.retrieved
    .map((r) => `<div class="chunk matched"><div class="c-head">chunk ${r.chunk_id}<span class="score">sim ${r.similarity}</span></div>${esc(r.text)}</div>`)
    .join("");
  const html = `
    <div class="section-label">Answer ${d.dry_run ? "(dry run)" : ""} · ${esc(d.model)}</div>
    <div class="answer">${esc(d.answer)}</div>
    <div class="section-label">Exact prompt sent to the model</div>
    <pre class="code">${esc(d.prompt_preview)}</pre>
    <div class="section-label">Context chunks used</div>
    ${matched}`;
  showInspector("generate", "Generated answer", "step 06", html);
}

/* ---- pipeline actions --------------------------------------------------- */
function knobs() {
  return {
    chunk_size: +$("#chunkSize").value,
    chunk_overlap: +$("#chunkOverlap").value,
    top_k: +$("#topK").value,
  };
}

async function doChunk() {
  const k = knobs();
  setStage("chunk", "active");
  const d = await api("/api/chunk", { chunk_size: k.chunk_size, chunk_overlap: k.chunk_overlap });
  results.chunk = d;
  setStage("chunk", "done");
  renderChunk(d);
  // chunks changed -> downstream stages reset
  stageEls.embed.classList.remove("done");
  stageEls.store.classList.remove("done");
  return d;
}

async function doEmbed() {
  setStage("embed", "active");
  setStage("store", "active");
  const d = await api("/api/embed");
  results.embed = d;
  setStage("embed", "done");
  setStage("store", "done");
  renderEmbed(d);
  return d;
}

async function doRetrieve() {
  const q = $("#question").value.trim();
  if (!q) { toast("Type a question first.", true); return; }
  setStage("retrieve", "active");
  const d = await api("/api/retrieve", { question: q, top_k: +$("#topK").value });
  results.retrieve = d;
  setStage("retrieve", "done");
  renderRetrieve(d);
  return d;
}

async function doGenerate() {
  const q = $("#question").value.trim();
  if (!q) { toast("Type a question first.", true); return; }
  setStage("generate", "active");
  const d = await api("/api/generate", { question: q, top_k: +$("#topK").value });
  results.generate = d;
  results.retrieve = { question: q, top_k: +$("#topK").value, retrieved: d.retrieved };
  setStage("retrieve", "done");
  setStage("generate", "done");
  renderGenerate(d);
  return d;
}

async function runAll() {
  const q = $("#question").value.trim();
  if (!results.load && !$("#sourceText").value.trim()) {
    toast("Load a source first.", true); return;
  }
  if (!q) { toast("Type a question first.", true); return; }
  try {
    await doChunk();
    await doEmbed();
    await doRetrieve();
    await doGenerate();
    toast("Pipeline complete — click any stage to inspect it.");
  } catch (e) {
    toast(e.message, true);
  }
}

/* ---- source loading ----------------------------------------------------- */
async function loadPasted() {
  const text = $("#sourceText").value.trim();
  if (!text) { toast("Paste some text first.", true); return; }
  try {
    const d = await api("/api/load-text", { text, source_name: "pasted-text" });
    results.load = d;
    setStage("load", "done");
    renderLoad(d);
    toast("Text loaded.");
  } catch (e) { toast(e.message, true); }
}

async function loadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  try {
    const d = await api("/api/load-file", fd, true);
    results.load = d;
    setStage("load", "done");
    renderLoad(d);
    toast(`Loaded ${file.name}.`);
  } catch (e) { toast(e.message, true); }
}

const SAMPLE = `Retrieval-Augmented Generation (RAG) is a technique for making language models answer from a specific body of knowledge instead of only their training data.

A plain language model answers from memory. It can be confidently wrong, and it cannot know about your private documents or anything newer than its training cutoff. RAG fixes this by retrieving relevant text at question time and placing it in the model's context.

The pipeline has six stages. First, loading turns a source such as a PDF or web page into plain text. Second, chunking splits that text into small overlapping pieces, because models have a limited context window and because search is sharper on focused passages. Third, embedding converts each chunk into a vector, a list of numbers that captures meaning, so that similar passages sit close together. Fourth, the vectors are stored in a vector database built for fast similarity search.

When a user asks a question, the question itself is embedded with the same model. The vector store returns the chunks whose vectors are nearest to the question's vector. This retrieval step uses no language model at all; it is pure similarity search over your own data.

Finally, generation hands the retrieved chunks and the question to a language model with an instruction to answer only from the supplied context. Because the answer is grounded in retrieved passages, it is more accurate, can cite its sources, and can cover private or recent information the model never saw during training.

Two knobs matter most for quality. Chunk size controls how much text sits in each piece: too large and retrieval is vague, too small and meaning is fragmented. The number of chunks retrieved, called top-k, controls how much context the model receives: too few and the answer may miss facts, too many and the prompt fills with noise.`;

function loadSample() {
  $("#sourceText").value = SAMPLE;
  loadPasted();
}

/* ---- provider status ---------------------------------------------------- */
async function refreshProvider() {
  // A tiny generate dry-run-free probe: ask status, then infer from a no-op.
  try {
    const s = await api("/api/generate", { question: "ping", top_k: 1 }).catch(() => null);
    // The status endpoint is cleaner; fall back to it.
  } catch (_) {}
  try {
    const st = await fetch("/api/status").then((r) => r.json());
    $("#provider").textContent = st.embedded ? "vectors ready" : "model: see .env";
  } catch (_) {
    $("#provider").textContent = "backend offline";
  }
}

/* ---- wire up the UI ----------------------------------------------------- */
function bind() {
  $("#btnLoadText").onclick = loadPasted;
  $("#btnSample").onclick = loadSample;
  $("#btnRunAll").onclick = runAll;
  $("#fileInput").onchange = (e) => e.target.files[0] && loadFile(e.target.files[0]);

  // live knob labels
  $("#chunkSize").oninput = (e) => ($("#csVal").textContent = e.target.value);
  $("#chunkOverlap").oninput = (e) => ($("#coVal").textContent = e.target.value);
  $("#topK").oninput = (e) => ($("#kVal").textContent = e.target.value);

  // manual step buttons
  document.querySelectorAll("[data-step]").forEach((b) => {
    b.onclick = async () => {
      try {
        if (b.dataset.step === "chunk") await doChunk();
        if (b.dataset.step === "embed") await doEmbed();
        if (b.dataset.step === "retrieve") await doRetrieve();
        if (b.dataset.step === "generate") await doGenerate();
      } catch (e) { toast(e.message, true); }
    };
  });

  // clicking a pipeline node re-shows its last output
  const renderers = {
    load: () => results.load && renderLoad(results.load),
    chunk: () => results.chunk && renderChunk(results.chunk),
    embed: () => results.embed && renderEmbed(results.embed),
    store: () => results.embed && renderEmbed(results.embed),
    retrieve: () => results.retrieve && renderRetrieve(results.retrieve),
    generate: () => results.generate && renderGenerate(results.generate),
  };
  Object.entries(stageEls).forEach(([name, el]) => {
    el.onclick = () => {
      if (renderers[name] && results[name === "store" ? "embed" : name]) renderers[name]();
      else { selectStage(name); toast("That stage hasn't run yet."); }
    };
  });
}

bind();
fetch("/api/status").then((r) => r.json()).then((st) => {
  $("#provider").textContent = "ready · add a source to begin";
}).catch(() => ($("#provider").textContent = "backend offline"));
