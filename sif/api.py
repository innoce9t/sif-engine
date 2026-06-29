"""
SIF Engine web API + demo UI (FastAPI).

A minimal frontend for inspecting the engine: upload an image and see the full
SIF JSON it produces (objects, caption, OCR, embeddings, backends used).
Optionally save it to the index and run semantic search.

Run:
    python -m sif.cli serve            # then open http://127.0.0.1:8000
    # or: uvicorn sif.api:app --reload

It uses whatever extractors are available in the running environment: real
models if installed (run from the Stage-1 venv with Ollama up), otherwise the
deterministic stubs. The first real-model request is slow (model load); after
that it's fast.
"""
from __future__ import annotations

import os
import threading

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .pipeline import process
from .query import search
from .store import Store
from . import dedup

DATA_ROOT = os.environ.get("SIF_DATA", "./sif_data")
UPLOAD_DIR = os.path.join(DATA_ROOT, "uploads")

app = FastAPI(title="SIF Engine", version="3.0")

# One shared store, guarded by a lock. SQLite isn't safe across threads and
# FastAPI runs sync endpoints in a threadpool, so serialize store access. This
# is a single-user demo server; Stage 4's dedicated writer is the real answer.
_store: Store | None = None
_lock = threading.Lock()


def _get_store() -> Store:
    global _store
    if _store is None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        _store = Store(DATA_ROOT)
    return _store


def _save_upload(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = os.path.basename(file.filename or "upload.bin")
    dest = os.path.join(UPLOAD_DIR, name)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return dest


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


@app.post("/api/process")
def api_process(file: UploadFile = File(...), save: str = Form("false")):
    """Run the pipeline on an uploaded image; optionally add it to the index.

    Unlike the CLI's dedup-skip, the API always processes so it can show the
    JSON — even for a duplicate — and only the *storage* decision varies.
    """
    path = _save_upload(file)
    sif = process(path)  # one inference, outside the lock
    status = "not-saved"
    if save.lower() in ("1", "true", "on", "yes"):
        with _lock:
            st = _get_store()
            meta = st.get_meta(path)
            if meta is not None and meta["sha256"] == sif.file.sha256:
                status = "unchanged"
            elif meta is not None:
                st.update(sif); status = "updated"
            elif st.find_duplicate(dedup.Hashes(
                    sif.file.sha256, sif.file.pixel_hash, sif.file.phash)) is not None:
                status = "duplicate"
            else:
                st.insert(sif); status = "indexed"
    return JSONResponse({"status": status, "sif": sif.to_dict()})


@app.get("/api/search")
def api_search(q: str = Query(...), limit: int = 10):
    with _lock:
        results = search(_get_store(), q, limit=limit)
    return JSONResponse({"query": q, "count": len(results), "results": results})


@app.get("/api/stats")
def api_stats():
    with _lock:
        s = _get_store()
        return JSONResponse({
            "indexed": s.count(),
            "visual_vectors": s.visual.count(),
            "text_vectors": s.text.count(),
        })


# --------------------------------------------------------------------------
# Single-page UI (inlined so the server is one self-contained module).
# --------------------------------------------------------------------------
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SIF Engine</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
          --muted:#8b949e; --accent:#2f81f7; --good:#3fb950; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--border); }
  header h1 { margin:0; font-size:18px; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .wrap { display:grid; grid-template-columns:380px 1fr; gap:20px; padding:24px;
          align-items:start; }
  @media (max-width:880px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--border);
          border-radius:10px; padding:18px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
       color:var(--muted); margin:0 0 12px; }
  #drop { border:2px dashed var(--border); border-radius:10px; padding:28px;
          text-align:center; color:var(--muted); cursor:pointer; transition:.15s; }
  #drop.hover { border-color:var(--accent); color:var(--fg); }
  #preview { max-width:100%; border-radius:8px; margin-top:14px; display:none; }
  .row { display:flex; align-items:center; gap:8px; margin-top:14px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
           padding:9px 16px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  input[type=text]{ flex:1; background:var(--bg); border:1px solid var(--border);
           color:var(--fg); border-radius:8px; padding:9px 12px; }
  label.chk { color:var(--muted); display:flex; align-items:center; gap:6px; }
  .summary div { margin:6px 0; }
  .k { color:var(--muted); }
  .pill { display:inline-block; background:var(--bg); border:1px solid var(--border);
          border-radius:999px; padding:2px 9px; margin:2px 4px 2px 0; font-size:12px; }
  .bar { height:6px; background:var(--bg); border-radius:3px; overflow:hidden;
         display:inline-block; width:90px; vertical-align:middle; margin-left:6px; }
  .bar > i { display:block; height:100%; background:var(--good); }
  pre { background:var(--bg); border:1px solid var(--border); border-radius:8px;
        padding:14px; overflow:auto; max-height:60vh; font-size:12.5px;
        white-space:pre-wrap; word-break:break-word; }
  .muted { color:var(--muted); }
  .results div.hit { padding:8px 0; border-bottom:1px solid var(--border); }
  .err { color:#f85149; }
</style>
</head>
<body>
<header>
  <h1>SIF Engine — semantic image inspector</h1>
  <p>Upload an image to extract its Semantic Index File (objects · caption · OCR · embeddings).</p>
</header>
<div class="wrap">
  <div class="card">
    <h2>Upload</h2>
    <div id="drop">Drop an image here, or click to choose</div>
    <input id="file" type="file" accept="image/*" hidden/>
    <img id="preview"/>
    <div class="row">
      <label class="chk"><input id="save" type="checkbox" checked/> add to index</label>
      <button id="go" disabled>Extract SIF</button>
    </div>
    <div id="status" class="muted" style="margin-top:10px"></div>

    <h2 style="margin-top:24px">Search the index</h2>
    <div class="row">
      <input id="q" type="text" placeholder="e.g. public transport in the city"/>
      <button id="searchBtn">Search</button>
    </div>
    <div id="results" class="results"></div>
  </div>

  <div class="card">
    <h2>Result</h2>
    <div id="summary" class="summary muted">No image processed yet.</div>
    <div class="row">
      <strong style="font-size:13px">SIF JSON</strong>
      <label class="chk" style="margin-left:auto"><input id="trim" type="checkbox" checked/> trim embedding vectors</label>
    </div>
    <pre id="json" class="muted">—</pre>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
let lastSif = null;

const drop = $('#drop'), fileInput = $('#file'), go = $('#go'), preview = $('#preview');
let chosen = null;

drop.onclick = () => fileInput.click();
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => { if (ev.dataTransfer.files[0]) setFile(ev.dataTransfer.files[0]); });
fileInput.onchange = () => { if (fileInput.files[0]) setFile(fileInput.files[0]); };

function setFile(f){
  chosen = f;
  go.disabled = false;
  drop.textContent = f.name;
  const url = URL.createObjectURL(f);
  preview.src = url; preview.style.display = 'block';
}

go.onclick = async () => {
  if (!chosen) return;
  go.disabled = true; $('#status').textContent = 'Processing… (first real-model run can take a while)';
  $('#status').className = 'muted';
  const fd = new FormData();
  fd.append('file', chosen);
  fd.append('save', $('#save').checked ? 'true' : 'false');
  try {
    const r = await fetch('/api/process', { method:'POST', body:fd });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    lastSif = data.sif;
    renderSummary(data.sif, data.status);
    renderJson();
    $('#status').textContent = 'Done.' + (data.status !== 'not-saved' ? ' index: ' + data.status : '');
  } catch (e) {
    $('#status').textContent = 'Error: ' + e.message; $('#status').className = 'err';
  } finally { go.disabled = false; }
};

function renderSummary(sif, status){
  const objs = (sif.objects||[]).map(o =>
    `<span class="pill">${o.label} <span class="bar"><i style="width:${Math.round(o.confidence*100)}%"></i></span></span>`).join('') || '<span class="muted">none</span>';
  const tags = (sif.scene.tags||[]).map(t => `<span class="pill">${t}</span>`).join('') || '';
  const mu = sif.meta.models_used || {};
  $('#summary').className = 'summary';
  $('#summary').innerHTML = `
    <div><span class="k">caption:</span> ${sif.scene.caption || '<span class="muted">—</span>'}</div>
    <div><span class="k">tags:</span> ${tags}</div>
    <div><span class="k">objects:</span><br>${objs}</div>
    <div><span class="k">ocr:</span> ${sif.ocr.has_text ? esc(sif.ocr.full_text) : '<span class="muted">no text</span>'}</div>
    <div><span class="k">resolution:</span> ${sif.file.resolution.join('×')} ${sif.file.format}</div>
    <div><span class="k">embedding:</span> ${sif.embeddings.model} · visual ${sif.embeddings.visual.length}d · text ${sif.embeddings.text.length}d</div>
    <div><span class="k">backends:</span> ${Object.entries(mu).map(([k,v])=>`${k}=${v}`).join(', ')}</div>`;
}

function renderJson(){
  if (!lastSif) return;
  let s = JSON.parse(JSON.stringify(lastSif));
  if ($('#trim').checked && s.embeddings){
    for (const k of ['visual','text']){
      const v = s.embeddings[k];
      if (Array.isArray(v) && v.length > 6)
        s.embeddings[k] = v.slice(0,3).concat([`… (${v.length} floats)`]);
    }
  }
  $('#json').className = '';
  $('#json').textContent = JSON.stringify(s, null, 2);
}
$('#trim').onchange = renderJson;

$('#searchBtn').onclick = async () => {
  const q = $('#q').value.trim(); if (!q) return;
  const box = $('#results'); box.innerHTML = '<span class="muted">searching…</span>';
  try {
    const r = await fetch('/api/search?q=' + encodeURIComponent(q));
    const d = await r.json();
    box.innerHTML = d.results.length ? d.results.map(h =>
      `<div class="hit"><strong>${h.distance.toFixed(3)}</strong> ${esc(h.path.split(/[\\/]/).pop())}<br>
       <span class="muted">${esc(h.caption||'')}</span></div>`).join('')
      : '<span class="muted">no results</span>';
  } catch(e){ box.innerHTML = '<span class="err">'+e.message+'</span>'; }
};

function esc(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
</script>
</body>
</html>"""
