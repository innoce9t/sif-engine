"""
SIF Engine web app (FastAPI).

A real UI over the engine, not just an inspector:
  * Library   — browse everything already indexed (thumbnails + captions)
  * Add       — analyze a single upload, or bulk-index a whole folder (async)
  * Search    — semantic search with thumbnails
  * Settings  — model status, data dir, and the faces on/off toggle

Uses whatever models are installed (real when available, stubs otherwise).
Run:  python -m sif.cli serve   (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import importlib.util
import io
import os
import threading
import uuid

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .pipeline import process
from .query import search, search_by_image, search_similar
from .store import Store
from . import dedup, clip_embed

DATA_ROOT = os.environ.get("SIF_DATA", "./sif_data")
UPLOAD_DIR = os.path.join(DATA_ROOT, "uploads")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DOC_EXTS = {".pdf"}

app = FastAPI(title="SIF Engine", version="1.2")

# One shared store for reads/single writes, guarded by a lock (SQLite isn't
# thread-safe and FastAPI runs sync endpoints in a threadpool). Bulk indexing
# jobs use their own Store instance (separate SQLite connection; WAL makes that
# safe) so a long index doesn't hold this lock.
_store: Store | None = None
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _get_store() -> Store:
    global _store
    if _store is None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        _store = Store(DATA_ROOT)
    return _store


def _save_upload(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, os.path.basename(file.filename or "upload.bin"))
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return dest


def _collect(folder: str) -> list[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS | DOC_EXTS:
                out.append(os.path.join(root, f))
    return out


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


# --------------------------------------------------------------------------
# Single upload (analyze; optionally add to index)
# --------------------------------------------------------------------------
@app.post("/api/process")
def api_process(file: UploadFile = File(...), save: str = Form("false")):
    path = _save_upload(file)
    from . import pdf
    sif = pdf.process_pdf(path) if (pdf.is_pdf(path) and pdf.deps_available()) else process(path)
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


# --------------------------------------------------------------------------
# Bulk folder indexing (async job)
# --------------------------------------------------------------------------
def _run_index_job(job_id: str, paths: list[str], force: bool = False):
    job = _jobs[job_id]
    try:
        from . import runner
        from .ingest import ingest
        st = Store(DATA_ROOT)                      # own connection for the job
        images = [p for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
        pdfs = [p for p in paths if os.path.splitext(p)[1].lower() in DOC_EXTS]
        job["total"] = len(paths)

        def prog(_label, path):
            job["done"] += 1
            job["last"] = os.path.basename(path)

        if images:
            rep = runner.index_paths(st, images, progress=prog, force=force)
            for k, v in rep["stats"].items():
                job["tally"][k] = job["tally"].get(k, 0) + v
        for p in pdfs:
            r = ingest(st, p, force=force)
            job["done"] += 1
            job["last"] = os.path.basename(p)
            job["tally"][r.status] = job["tally"].get(r.status, 0) + 1
        job["indexed_total"] = st.count()
        st.close()
        job["status"] = "done"
    except Exception as e:  # surface it to the UI rather than dying silently
        job["status"] = "error"
        job["error"] = str(e)


def _start_job(paths: list[str], force: bool = False) -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "total": len(paths), "done": 0, "last": "", "tally": {}}
    threading.Thread(target=_run_index_job, args=(job_id, paths, force), daemon=True).start()
    return job_id


@app.post("/api/index-folder")
def api_index_folder(path: str = Form(...), force: str = Form("false")):
    if not os.path.isdir(path):
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=400)
    force_b = force.lower() in ("1", "true", "on", "yes")
    return JSONResponse({"job": _start_job(_collect(path), force=force_b)})


@app.post("/api/reindex-all")
def api_reindex_all():
    """Force re-process every already-indexed asset (to add CLIP/faces to an
    index built before those were enabled). Skips files no longer on disk."""
    with _lock:
        paths = [a["path"] for a in _get_store().list_assets(limit=100000)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return JSONResponse({"error": "nothing to re-index (no existing files found)"}, status_code=400)
    return JSONResponse({"job": _start_job(paths, force=True), "count": len(paths)})


@app.get("/api/index-status")
def api_index_status(job: str = Query(...)):
    j = _jobs.get(job)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(j)


# --------------------------------------------------------------------------
# Library / assets / search / thumbnails
# --------------------------------------------------------------------------
@app.get("/api/library")
def api_library(limit: int = 200, offset: int = 0):
    with _lock:
        assets = _get_store().list_assets(limit=limit, offset=offset)
    return JSONResponse({"count": len(assets), "assets": assets})


@app.get("/api/asset")
def api_asset(id: str = Query(...)):
    with _lock:
        sif = _get_store().get(id)
    if sif is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(sif)


@app.get("/api/search")
def api_search(q: str = Query(...), limit: int = 20, kind: str = "",
               has_text: str = "", object: str = ""):
    filters = {"kind": kind, "has_text": has_text.lower() in ("1", "true", "on"),
               "object": object}
    with _lock:
        results = search(_get_store(), q, limit=limit, filters=filters)
    return JSONResponse({"query": q, "count": len(results), "results": results})


@app.post("/api/search-image")
def api_search_image(file: UploadFile = File(...)):
    """Reverse image search: find indexed images that look like the upload."""
    path = _save_upload(file)
    vec = clip_embed.embed_image(path)
    if not vec:
        return JSONResponse({"error": "CLIP unavailable — install sentence-transformers "
                             "and re-index to enable image search.", "results": []},
                            status_code=200)
    with _lock:
        results = search_by_image(_get_store(), vec, limit=24)
    return JSONResponse({"count": len(results), "results": results})


@app.get("/api/similar")
def api_similar(id: str = Query(...)):
    with _lock:
        results = search_similar(_get_store(), id, limit=12)
    return JSONResponse({"count": len(results), "results": results})


@app.get("/api/duplicates")
def api_duplicates(path: str = Query(...)):
    """Find duplicate/near-duplicate FILES in a folder (the index itself dedups,
    so this scans disk — handy for cleaning a folder before indexing)."""
    if not os.path.isdir(path):
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=400)
    groups = dedup.duplicate_groups(_collect(path))
    return JSONResponse({"count": len(groups), "groups": groups})


@app.post("/api/delete")
def api_delete(id: str = Form(...)):
    with _lock:
        _get_store().delete(id.split("#")[0])
    return JSONResponse({"deleted": id})


@app.get("/api/reveal")
def api_reveal(id: str = Query(...)):
    """Open the asset's containing folder on the (local) server machine."""
    path = id.split("#")[0]
    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        os.startfile(os.path.dirname(path))   # Windows; local-only server
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


@app.get("/api/thumbnail")
def api_thumbnail(id: str = Query(...)):
    from PIL import Image
    path = id.split("#")[0]
    if not os.path.exists(path):
        return Response(status_code=404)
    try:
        if path.lower().endswith(".pdf"):
            import pypdfium2 as pdfium       # render page 1 as the thumbnail
            pdf = pdfium.PdfDocument(path)
            try:
                im = pdf[0].render(scale=0.5).to_pil().convert("RGB")
            finally:
                pdf.close()
        else:
            im = Image.open(path)
        im.thumbnail((260, 260))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=80)
        return Response(buf.getvalue(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)


@app.get("/api/stats")
def api_stats():
    with _lock:
        s = _get_store()
        return JSONResponse({
            "indexed": s.count(),
            "visual_vectors": s.visual.count(),
            "text_vectors": s.text.count(),
            "clip_vectors": s.clip.count(),
        })


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def _settings() -> dict:
    return {
        "data_root": os.path.abspath(DATA_ROOT),
        "enable_faces": os.environ.get("SIF_ENABLE_FACES") == "1",
        "faces_available": importlib.util.find_spec("insightface") is not None,
        "vlm_model": os.environ.get("SIF_VLM_MODEL", "moondream"),
        "clip_available": importlib.util.find_spec("sentence_transformers") is not None,
        "using_stubs": os.environ.get("SIF_USE_STUBS") == "1",
    }


@app.get("/api/settings")
def api_get_settings():
    return JSONResponse(_settings())


@app.post("/api/settings")
def api_set_settings(enable_faces: str = Form(None), vlm_model: str = Form(None)):
    if enable_faces is not None:
        if enable_faces.lower() in ("1", "true", "on", "yes"):
            os.environ["SIF_ENABLE_FACES"] = "1"
        else:
            os.environ.pop("SIF_ENABLE_FACES", None)
    if vlm_model:
        os.environ["SIF_VLM_MODEL"] = vlm_model
    return JSONResponse(_settings())


@app.get("/api/health")
def api_health():
    models, ollama_up = [], False
    try:
        import ollama
        for m in ollama.list().get("models", []):
            models.append(m.get("model") or m.get("name"))
        ollama_up = True
    except Exception:
        pass
    vlm = os.environ.get("SIF_VLM_MODEL", "moondream")
    return JSONResponse({
        "ollama_up": ollama_up,
        "ollama_models": models,
        "vlm_ready": ollama_up and any(vlm in (m or "") for m in models),
        "vlm_model": vlm,
        "using_stubs": os.environ.get("SIF_USE_STUBS") == "1",
    })


@app.post("/api/ask")
def api_ask(q: str = Form(...)):
    """RAG: retrieve top matches, then have a local LLM answer from them."""
    with _lock:
        results = search(_get_store(), q, limit=5)
        st = _get_store()
        ctx_lines = []
        for i, r in enumerate(results, 1):
            sif = st.get(r["path"]) or {}
            extra = sif.get("ocr", {}).get("full_text", "")[:300]
            loc = os.path.basename(r["path"]) + (f" p{r['page'] + 1}" if r.get("page") is not None else "")
            ctx_lines.append(f"[{i}] {loc}: {r['caption']} {' '.join(r['objects'])} {extra}".strip())
    context = "\n".join(ctx_lines) or "(no matching items)"
    try:
        import ollama
        model = os.environ.get("SIF_RAG_MODEL", os.environ.get("SIF_VLM_MODEL", "gemma4"))
        prompt = ("Answer the question using ONLY the indexed items below. Cite item "
                  "numbers like [1]. If they don't contain the answer, say so briefly.\n\n"
                  f"Items:\n{context}\n\nQuestion: {q}")
        answer = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"]
    except Exception as e:
        answer = f"(LLM unavailable: {e}) — showing the top matches as sources instead."
    return JSONResponse({"answer": answer, "sources": results})


# -- watch a folder (incremental auto-index) ------------------------------
_watch = {"thread": None, "stop": False, "folder": None, "active": False}


def _watch_loop(folder: str, interval: float):
    import time
    from .ingest import ingest
    st = Store(DATA_ROOT)
    try:
        while not _watch["stop"]:
            for p in _collect(folder):
                ingest(st, p)          # dedup makes re-scans cheap/idempotent
            time.sleep(interval)
    finally:
        st.close()


@app.post("/api/watch/start")
def api_watch_start(path: str = Form(...), interval: float = Form(10.0)):
    if not os.path.isdir(path):
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=400)
    if _watch["active"]:
        return JSONResponse({"error": "already watching " + str(_watch["folder"])}, status_code=409)
    _watch.update(stop=False, folder=path, active=True)
    _watch["thread"] = threading.Thread(target=_watch_loop, args=(path, interval), daemon=True)
    _watch["thread"].start()
    return JSONResponse({"active": True, "folder": path})


@app.post("/api/watch/stop")
def api_watch_stop():
    _watch["stop"] = True
    _watch["active"] = False
    return JSONResponse({"active": False})


@app.get("/api/watch/status")
def api_watch_status():
    return JSONResponse({"active": _watch["active"], "folder": _watch["folder"]})


# --------------------------------------------------------------------------
# Single-page tabbed UI
# --------------------------------------------------------------------------
_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SIF Engine</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;
        --accent:#2f81f7;--good:#3fb950;--warn:#d29922}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
  header{padding:14px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px}
  header h1{margin:0;font-size:17px}
  nav{display:flex;gap:4px;margin-left:auto;flex-wrap:wrap}
  nav button{background:none;border:1px solid transparent;color:var(--muted);
             padding:7px 13px;border-radius:8px;cursor:pointer;font-size:14px}
  nav button.active{background:var(--panel);border-color:var(--border);color:var(--fg)}
  #health{padding:6px 24px;font-size:12.5px;border-bottom:1px solid var(--border);background:#11161d}
  main{padding:22px;max-width:1120px;margin:0 auto}
  .tab{display:none}.tab.active{display:block}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 12px}
  button.btn{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;
             font-weight:600;cursor:pointer}
  button.btn:disabled{opacity:.5;cursor:default}
  input[type=text],select{background:var(--bg);border:1px solid var(--border);color:var(--fg);
       border-radius:8px;padding:9px 12px}
  input[type=text]{flex:1;min-width:120px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-top:16px}
  .thumb{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;cursor:pointer;transition:.12s}
  .thumb:hover{border-color:var(--accent)}
  .imgwrap{position:relative;height:130px;background:#0b0e13}
  .imgwrap .ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:30px;color:var(--muted)}
  .imgwrap img{position:relative;width:100%;height:100%;object-fit:cover;opacity:0;transition:.2s;display:block}
  .acts{position:absolute;top:6px;right:6px;display:none;gap:4px}
  .thumb:hover .acts{display:flex}
  .acts button{background:rgba(13,17,23,.85);border:1px solid var(--border);color:var(--fg);
               border-radius:6px;width:26px;height:26px;cursor:pointer;font-size:13px;line-height:1}
  .meta{padding:8px 10px}
  .name{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .cap{font-size:12.5px;margin-top:2px;max-height:34px;overflow:hidden}
  .pill{display:inline-block;background:var(--bg);border:1px solid var(--border);border-radius:999px;
        padding:1px 7px;margin:3px 3px 0 0;font-size:11px}
  .pill.visual{border-color:#2f81f7}.pill.text{border-color:#3fb950}.pill.clip{border-color:#a371f7}
  #drop,.drop{border:2px dashed var(--border);border-radius:10px;padding:20px;text-align:center;color:var(--muted);cursor:pointer}
  #drop.hover,.drop.hover{border-color:var(--accent);color:var(--fg)}
  pre{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;
      max-height:50vh;font-size:12px;white-space:pre-wrap;word-break:break-word}
  .muted{color:var(--muted)}.good{color:var(--good)}.warn{color:var(--warn)}.err{color:#f85149}
  .bar{height:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:8px}
  .bar>i{display:block;height:100%;background:var(--accent);width:0}
  .switch{display:flex;align-items:center;gap:10px;margin:10px 0}
  .modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;padding:24px;z-index:5}
  .modal.open{display:flex}
  .modal .box{background:var(--panel);border:1px solid var(--border);border-radius:12px;max-width:820px;width:100%;max-height:86vh;overflow:auto;padding:20px}
  label{font-size:13px}.dim{color:var(--muted);font-size:12.5px}
  #answer{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;margin-top:12px;white-space:pre-wrap}
</style></head><body>
<header>
  <h1>🔍 SIF Engine</h1>
  <nav>
    <button data-tab="library" class="active">Library</button>
    <button data-tab="search">Search</button>
    <button data-tab="ask">Ask</button>
    <button data-tab="add">Add</button>
    <button data-tab="duplicates">Duplicates</button>
    <button data-tab="settings">Settings</button>
  </nav>
</header>
<div id="health" class="muted">checking models…</div>
<main>
  <section id="library" class="tab active">
    <div class="row"><h2 style="margin:0">Indexed assets</h2>
      <button class="btn" style="margin-left:auto;padding:6px 12px" onclick="loadLibrary()">Refresh</button></div>
    <div id="libStats" class="dim" style="margin-top:6px"></div>
    <div id="libGrid" class="grid"></div>
  </section>

  <section id="search" class="tab">
    <div class="row"><input id="q" type="text" placeholder="e.g. a code editor, an error dialog, invoice from March"/>
      <button id="searchBtn" class="btn">Search</button></div>
    <div class="row" style="margin-top:8px">
      <select id="fKind"><option value="">any type</option><option value="image">images</option><option value="pdf">PDFs</option></select>
      <label class="switch"><input type="checkbox" id="fText"/> has text</label>
      <input id="fObj" type="text" style="max-width:180px;flex:none" placeholder="object (e.g. person)"/>
      <span class="dim">or</span>
      <label class="btn" style="padding:8px 12px;cursor:pointer">Search by image<input id="imgSearch" type="file" accept="image/*" hidden/></label>
    </div>
    <div id="searchInfo" class="dim" style="margin-top:10px"></div>
    <div id="searchGrid" class="grid"></div>
  </section>

  <section id="ask" class="tab">
    <div class="card">
      <h2>Ask your archive</h2>
      <div class="row"><input id="askQ" type="text" placeholder="What does my archive say about…?"/>
        <button id="askBtn" class="btn">Ask</button></div>
      <div class="dim" style="margin-top:6px">Retrieves the top matches and has a local LLM answer from them (cited). Needs Ollama + a text model.</div>
      <div id="answer" style="display:none"></div>
    </div>
    <div id="askSources" class="grid"></div>
  </section>

  <section id="add" class="tab">
    <div class="card">
      <h2>Analyze one image</h2>
      <div id="drop">Drop an image here, or click to choose</div>
      <input id="file" type="file" accept="image/*" hidden/>
      <div class="row" style="margin-top:12px">
        <label class="switch"><input id="save" type="checkbox" checked/> add to index</label>
        <button id="go" class="btn" style="margin-left:auto" disabled>Extract SIF</button></div>
      <div id="upStatus" class="dim" style="margin-top:8px"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Bulk-index a folder</h2>
      <div class="row"><input id="folder" type="text" placeholder="C:\Users\you\Pictures\Screenshots"/>
        <button id="idxBtn" class="btn">Index folder</button>
        <button id="watchBtn" class="btn" style="background:#347d39">Watch</button></div>
      <label class="switch"><input id="forceIdx" type="checkbox"/> re-process files already indexed (adds CLIP / faces)</label>
      <div class="bar" id="idxBarWrap" style="display:none"><i id="idxBar"></i></div>
      <div id="idxStatus" class="dim" style="margin-top:8px"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Re-index everything</h2>
      <div class="row"><button id="reidxBtn" class="btn">Re-index all indexed assets</button></div>
      <div class="dim" style="margin-top:6px">Force-reprocess every asset already in the index — use this to add CLIP vectors (or faces) to an index built before they were enabled.</div>
    </div>
  </section>

  <section id="duplicates" class="tab">
    <div class="card">
      <h2>Find duplicate files in a folder</h2>
      <div class="row"><input id="dupPath" type="text" placeholder="folder to scan for duplicates"/>
        <button id="dupBtn" class="btn">Scan</button></div>
      <div class="dim" style="margin-top:6px">Exact, pixel-identical, and near-duplicate (perceptual) groups. The index itself dedups; this cleans a folder.</div>
      <div id="dupResults" style="margin-top:12px"></div>
    </div>
  </section>

  <section id="settings" class="tab">
    <div class="card"><h2>Settings</h2><div id="setBody" class="dim">loading…</div></div>
  </section>
</main>

<div class="modal" id="modal"><div class="box">
  <div class="row"><strong id="mTitle"></strong>
    <button class="btn" style="margin-left:auto;padding:4px 10px" onclick="closeModal()">Close</button></div>
  <div id="mSummary" style="margin:12px 0"></div><pre id="mJson"></pre>
</div></div>

<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const enc=encodeURIComponent, esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fname=p=>(p||'').split(/[\\/]/).pop();

$$('nav button').forEach(b=>b.onclick=()=>{
  $$('nav button').forEach(x=>x.classList.remove('active'));$$('.tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');$('#'+b.dataset.tab).classList.add('active');
  if(b.dataset.tab==='library')loadLibrary();
  if(b.dataset.tab==='settings')loadSettings();
});

async function loadHealth(){
  try{const h=await (await fetch('/api/health')).json();
    const el=$('#health');
    if(h.using_stubs){el.className='warn';el.textContent='⚠ Running in STUB mode — no real models (search quality is placeholder).';return;}
    if(!h.ollama_up){el.className='warn';el.textContent='⚠ Ollama is not running — scene captions fall back to stubs. Start Ollama for real captions.';return;}
    if(!h.vlm_ready){el.className='warn';el.textContent=`⚠ VLM "${h.vlm_model}" not found in Ollama (${h.ollama_models.length} models available). Pull it or pick another in Settings.`;return;}
    el.className='good';el.textContent=`✓ Real models · Ollama up · VLM: ${h.vlm_model}`;
  }catch(e){$('#health').textContent=''}
}

function card(a,actions){
  const src='/api/thumbnail?id='+enc(a.id);
  const badges=(a.matched||[]).map(m=>`<span class="pill ${m}">${m}</span>`).join('');
  const pg=(a.page!=null)?`<span class="pill">p${a.page+1}</span>`:'';
  const act=actions?`<div class="acts">
    <button title="More like this" onclick="event.stopPropagation();moreLike('${enc(a.id)}')">≈</button>
    <button title="Reveal in folder" onclick="event.stopPropagation();reveal('${enc(a.id)}')">📂</button>
    <button title="Delete" onclick="event.stopPropagation();del('${enc(a.id)}',this)">🗑</button></div>`:'';
  return `<div class="thumb" onclick="openAsset('${enc(a.id)}')">
    <div class="imgwrap"><span class="ph">${a.kind==='pdf'?'📄':'🖼️'}</span>
      <img loading="lazy" src="${src}" onload="this.style.opacity=1" onerror="this.remove()"/>${act}</div>
    <div class="meta"><div class="name">${esc(fname(a.path))}${pg}</div>
    <div class="cap">${esc(a.caption||'')}</div>${badges}</div></div>`;
}

async function loadLibrary(){
  const g=$('#libGrid');g.innerHTML='<span class="muted">loading…</span>';
  const d=await (await fetch('/api/library?limit=400')).json();
  const st=await (await fetch('/api/stats')).json();
  $('#libStats').textContent=`${st.indexed} assets · ${st.visual_vectors} visual · ${st.text_vectors} text · ${st.clip_vectors} CLIP vectors`;
  g.innerHTML=d.assets.length?d.assets.map(a=>card(a,true)).join(''):'<span class="muted">Nothing indexed yet — use the Add tab.</span>';
}

async function openAsset(id){
  const sif=await (await fetch('/api/asset?id='+id)).json();
  $('#mTitle').textContent=fname(sif.file&&sif.file.path||'');
  const mu=(sif.meta&&sif.meta.models_used)||{};
  $('#mSummary').innerHTML=
   `<div><span class="muted">caption:</span> ${esc(sif.scene?sif.scene.caption:'')||'<span class=muted>—</span>'}</div>
    <div><span class="muted">objects:</span> ${(sif.objects||[]).map(o=>`<span class=pill>${esc(o.label)}</span>`).join('')||'<span class=muted>none</span>'}</div>
    <div><span class="muted">ocr:</span> ${sif.ocr&&sif.ocr.has_text?esc(sif.ocr.full_text.slice(0,300)):'<span class=muted>no text</span>'}</div>
    <div><span class="muted">backends:</span> ${Object.entries(mu).map(([k,v])=>k+'='+v).join(', ')}</div>`;
  const s=JSON.parse(JSON.stringify(sif));
  if(s.embeddings)for(const k of ['visual','text','clip']){const v=s.embeddings[k];if(Array.isArray(v)&&v.length>6)s.embeddings[k]=v.slice(0,3).concat([`… (${v.length} floats)`]);}
  if(Array.isArray(s.pages))s.pages.forEach(p=>{if(Array.isArray(p.text_vector)&&p.text_vector.length>6)p.text_vector=['…('+p.text_vector.length+')'];(p.regions||[]).forEach(r=>['visual','clip'].forEach(k=>{if(Array.isArray(r[k])&&r[k].length>6)r[k]=['…('+r[k].length+')']}))});
  $('#mJson').textContent=JSON.stringify(s,null,2);$('#modal').classList.add('open');
}
function closeModal(){$('#modal').classList.remove('open')}
$('#modal').onclick=e=>{if(e.target.id==='modal')closeModal()};

async function del(id,btn){
  if(!confirm('Remove this asset from the index? (the file on disk is not deleted)'))return;
  const fd=new FormData();fd.append('id',decodeURIComponent(id));
  await fetch('/api/delete',{method:'POST',body:fd});
  const t=btn.closest('.thumb');if(t)t.remove();
}
async function reveal(id){await fetch('/api/reveal?id='+id)}
async function moreLike(id){
  $$('nav button').forEach(x=>x.classList.remove('active'));$$('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelector('nav button[data-tab=search]').classList.add('active');$('#search').classList.add('active');
  $('#searchInfo').textContent='Similar to '+fname(decodeURIComponent(id));$('#searchGrid').innerHTML='<span class="muted">…</span>';
  const d=await (await fetch('/api/similar?id='+id)).json();
  $('#searchGrid').innerHTML=d.results.length?d.results.map(a=>card(a,false)).join(''):'<span class="muted">no similar items (needs CLIP vectors — re-index with CLIP)</span>';
}

$('#searchBtn').onclick=doSearch;$('#q').addEventListener('keydown',e=>{if(e.key==='Enter')doSearch()});
async function doSearch(){
  const q=$('#q').value.trim();if(!q)return;
  const p=new URLSearchParams({q,kind:$('#fKind').value,has_text:$('#fText').checked?'1':'',object:$('#fObj').value.trim()});
  $('#searchInfo').textContent='searching…';$('#searchGrid').innerHTML='';
  try{const d=await (await fetch('/api/search?'+p)).json();
    $('#searchInfo').innerHTML=`${d.count} result(s)`+(d.results[0]&&d.results[0].reranked?' · re-ranked':'');
    $('#searchGrid').innerHTML=d.results.map(a=>card(a,false)).join('')||'<span class="muted">no results</span>';
  }catch(e){$('#searchInfo').innerHTML='<span class="err">'+e.message+'</span>'}
}
$('#imgSearch').onchange=async()=>{
  const f=$('#imgSearch').files[0];if(!f)return;
  $('#searchInfo').textContent='searching by image…';$('#searchGrid').innerHTML='';
  const fd=new FormData();fd.append('file',f);
  const d=await (await fetch('/api/search-image',{method:'POST',body:fd})).json();
  if(d.error){$('#searchInfo').innerHTML='<span class="warn">'+esc(d.error)+'</span>';return;}
  $('#searchInfo').textContent=`${d.count} visually similar`;
  $('#searchGrid').innerHTML=d.results.map(a=>card(a,false)).join('')||'<span class="muted">no results</span>';
};

$('#askBtn').onclick=doAsk;$('#askQ').addEventListener('keydown',e=>{if(e.key==='Enter')doAsk()});
async function doAsk(){
  const q=$('#askQ').value.trim();if(!q)return;
  $('#answer').style.display='block';$('#answer').textContent='thinking…';$('#askSources').innerHTML='';
  const fd=new FormData();fd.append('q',q);
  try{const d=await (await fetch('/api/ask',{method:'POST',body:fd})).json();
    $('#answer').textContent=d.answer;
    $('#askSources').innerHTML='<div class="dim" style="grid-column:1/-1">Sources:</div>'+d.sources.map(a=>card(a,false)).join('');
  }catch(e){$('#answer').innerHTML='<span class="err">'+e.message+'</span>'}
}

// single upload
const drop=$('#drop'),fileInput=$('#file'),go=$('#go');let chosen=null;
drop.onclick=()=>fileInput.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])pick(ev.dataTransfer.files[0])});
fileInput.onchange=()=>{if(fileInput.files[0])pick(fileInput.files[0])};
function pick(f){chosen=f;go.disabled=false;drop.textContent=f.name}
go.onclick=async()=>{
  if(!chosen)return;go.disabled=true;$('#upStatus').textContent='Processing… (first run loads models)';
  const fd=new FormData();fd.append('file',chosen);fd.append('save',$('#save').checked?'true':'false');
  try{const d=await (await fetch('/api/process',{method:'POST',body:fd})).json();
    $('#upStatus').innerHTML=`<span class="good">Done — ${d.status}.</span> Open the Library to view it.`;
  }catch(e){$('#upStatus').innerHTML='<span class="err">'+e.message+'</span>'}finally{go.disabled=false}
};

// bulk index
$('#idxBtn').onclick=async()=>{
  const path=$('#folder').value.trim();if(!path)return;
  const fd=new FormData();fd.append('path',path);fd.append('force',$('#forceIdx').checked?'true':'false');
  const r=await fetch('/api/index-folder',{method:'POST',body:fd});
  if(!r.ok){$('#idxStatus').innerHTML='<span class="err">'+(await r.json()).error+'</span>';return}
  $('#idxBarWrap').style.display='block';poll((await r.json()).job);
};
$('#reidxBtn').onclick=async()=>{
  if(!confirm('Re-process every indexed asset? This re-runs the models on all of them.'))return;
  const r=await fetch('/api/reindex-all',{method:'POST'});
  const j=await r.json();
  if(!r.ok){$('#idxStatus').innerHTML='<span class="err">'+j.error+'</span>';return}
  $('#idxBarWrap').style.display='block';$('#idxStatus').textContent='Re-indexing '+j.count+' assets…';poll(j.job);
};
async function poll(job){
  const j=await (await fetch('/api/index-status?job='+job)).json();
  const pct=j.total?Math.round(100*j.done/j.total):0;$('#idxBar').style.width=pct+'%';
  const tally=Object.entries(j.tally||{}).map(([k,v])=>k+'='+v).join(', ');
  $('#idxStatus').innerHTML=`${j.done}/${j.total} — ${esc(j.last||'')} <span class="muted">[${tally}]</span>`;
  if(j.status==='running')setTimeout(()=>poll(job),700);
  else if(j.status==='error')$('#idxStatus').innerHTML='<span class="err">'+esc(j.error)+'</span>';
  else $('#idxStatus').innerHTML=`<span class="good">Done.</span> ${j.indexed_total} total indexed. [${tally}]`;
}
let watching=false;
$('#watchBtn').onclick=async()=>{
  if(!watching){const path=$('#folder').value.trim();if(!path)return;
    const fd=new FormData();fd.append('path',path);
    const r=await fetch('/api/watch/start',{method:'POST',body:fd});
    if(!r.ok){$('#idxStatus').innerHTML='<span class="err">'+(await r.json()).error+'</span>';return}
    watching=true;$('#watchBtn').textContent='Stop watching';$('#watchBtn').style.background='#a1382f';
    $('#idxStatus').innerHTML='<span class="good">Watching '+esc(path)+'</span> — new/changed files auto-index.';
  }else{await fetch('/api/watch/stop',{method:'POST'});watching=false;
    $('#watchBtn').textContent='Watch';$('#watchBtn').style.background='#347d39';$('#idxStatus').textContent='Stopped watching.';}
};

// duplicates
$('#dupBtn').onclick=async()=>{
  const path=$('#dupPath').value.trim();if(!path)return;
  $('#dupResults').innerHTML='<span class="muted">scanning…</span>';
  const r=await fetch('/api/duplicates?path='+enc(path));
  if(!r.ok){$('#dupResults').innerHTML='<span class="err">'+(await r.json()).error+'</span>';return}
  const d=await r.json();
  if(!d.count){$('#dupResults').innerHTML='<span class="good">No duplicates found.</span>';return}
  $('#dupResults').innerHTML=`<div class="dim">${d.count} duplicate group(s):</div>`+d.groups.map((g,i)=>
    `<div style="margin-top:10px"><b>Group ${i+1}</b> (${g.length})<div class="dim">${g.map(esc).join('<br>')}</div></div>`).join('');
};

// settings
async function loadSettings(){
  const s=await (await fetch('/api/settings')).json();
  const h=await (await fetch('/api/health')).json();
  const opts=(h.ollama_models||[]).map(m=>`<option ${m===s.vlm_model?'selected':''}>${esc(m)}</option>`).join('');
  $('#setBody').innerHTML=`
    <div class="switch"><label><input type="checkbox" id="facesTog" ${s.enable_faces?'checked':''} ${s.faces_available?'':'disabled'}/> Enable facial recognition</label>
      ${s.faces_available?'':'<span class="warn">(install insightface to use)</span>'}</div>
    <div class="dim">Biometric-privacy surface, off by default. Affects <b>future</b> indexing — re-index to add/remove faces.</div>
    <div class="switch" style="margin-top:14px"><label>Scene VLM:</label>
      <select id="vlmSel">${opts||`<option>${esc(s.vlm_model)}</option>`}</select></div>
    <hr style="border-color:var(--border);margin:14px 0">
    <div><span class="muted">Data dir:</span> ${esc(s.data_root)}</div>
    <div><span class="muted">CLIP available:</span> ${s.clip_available?'<span class=good>yes</span>':'<span class=warn>no</span>'}</div>
    <div><span class="muted">Ollama:</span> ${h.ollama_up?'<span class=good>up ('+h.ollama_models.length+' models)</span>':'<span class=warn>down</span>'}</div>
    <div><span class="muted">Mode:</span> ${s.using_stubs?'<span class=warn>stubs</span>':'<span class=good>real models</span>'}</div>`;
  const tog=$('#facesTog');if(tog)tog.onchange=async()=>{const fd=new FormData();fd.append('enable_faces',tog.checked?'true':'false');await fetch('/api/settings',{method:'POST',body:fd})};
  const vlm=$('#vlmSel');if(vlm)vlm.onchange=async()=>{const fd=new FormData();fd.append('vlm_model',vlm.value);await fetch('/api/settings',{method:'POST',body:fd});loadHealth()};
}

loadHealth();loadLibrary();
</script></body></html>"""
