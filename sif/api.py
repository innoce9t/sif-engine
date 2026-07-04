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
from .query import search
from .store import Store
from . import dedup

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
def _run_index_job(job_id: str, folder: str):
    job = _jobs[job_id]
    try:
        from . import runner, pdf
        from .ingest import ingest
        st = Store(DATA_ROOT)                      # own connection for the job
        paths = _collect(folder)
        images = [p for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
        pdfs = [p for p in paths if os.path.splitext(p)[1].lower() in DOC_EXTS]
        job["total"] = len(paths)

        def prog(_label, path):
            job["done"] += 1
            job["last"] = os.path.basename(path)

        if images:
            rep = runner.index_paths(st, images, progress=prog)
            for k, v in rep["stats"].items():
                job["tally"][k] = job["tally"].get(k, 0) + v
        for p in pdfs:
            r = ingest(st, p)
            job["done"] += 1
            job["last"] = os.path.basename(p)
            job["tally"][r.status] = job["tally"].get(r.status, 0) + 1
        job["indexed_total"] = st.count()
        st.close()
        job["status"] = "done"
    except Exception as e:  # surface it to the UI rather than dying silently
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/index-folder")
def api_index_folder(path: str = Form(...)):
    if not os.path.isdir(path):
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=400)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "total": 0, "done": 0, "last": "", "tally": {}}
    threading.Thread(target=_run_index_job, args=(job_id, path), daemon=True).start()
    return JSONResponse({"job": job_id})


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
def api_search(q: str = Query(...), limit: int = 20):
    with _lock:
        results = search(_get_store(), q, limit=limit)
    return JSONResponse({"query": q, "count": len(results), "results": results})


@app.get("/api/thumbnail")
def api_thumbnail(id: str = Query(...)):
    from PIL import Image
    if not os.path.exists(id):
        return Response(status_code=404)
    try:
        with Image.open(id) as im:
            im.thumbnail((260, 260))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=80)
        return Response(buf.getvalue(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)   # e.g. PDFs — UI shows a placeholder


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
def api_set_settings(enable_faces: str = Form(...)):
    if enable_faces.lower() in ("1", "true", "on", "yes"):
        os.environ["SIF_ENABLE_FACES"] = "1"
    else:
        os.environ.pop("SIF_ENABLE_FACES", None)
    return JSONResponse(_settings())


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
  header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px}
  header h1{margin:0;font-size:17px}
  nav{display:flex;gap:4px;margin-left:auto}
  nav button{background:none;border:1px solid transparent;color:var(--muted);
             padding:7px 14px;border-radius:8px;cursor:pointer;font-size:14px}
  nav button.active{background:var(--panel);border-color:var(--border);color:var(--fg)}
  main{padding:24px;max-width:1100px;margin:0 auto}
  .tab{display:none}.tab.active{display:block}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 12px}
  button.btn{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;
             font-weight:600;cursor:pointer}
  button.btn:disabled{opacity:.5;cursor:default}
  input[type=text]{background:var(--bg);border:1px solid var(--border);color:var(--fg);
       border-radius:8px;padding:9px 12px;width:100%}
  .row{display:flex;gap:8px;align-items:center}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-top:16px}
  .thumb{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;
         cursor:pointer;transition:.12s}
  .thumb:hover{border-color:var(--accent)}
  .thumb .img{height:130px;background:#0b0e13 center/cover no-repeat;display:flex;
              align-items:center;justify-content:center;color:var(--muted);font-size:26px}
  .thumb .meta{padding:8px 10px}
  .thumb .name{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .thumb .cap{font-size:12.5px;margin-top:2px;max-height:34px;overflow:hidden}
  .pill{display:inline-block;background:var(--bg);border:1px solid var(--border);border-radius:999px;
        padding:1px 7px;margin:2px 3px 0 0;font-size:11px}
  #drop{border:2px dashed var(--border);border-radius:10px;padding:24px;text-align:center;
        color:var(--muted);cursor:pointer}
  #drop.hover{border-color:var(--accent);color:var(--fg)}
  pre{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;
      overflow:auto;max-height:52vh;font-size:12px;white-space:pre-wrap;word-break:break-word}
  .muted{color:var(--muted)} .good{color:var(--good)} .warn{color:var(--warn)} .err{color:#f85149}
  .bar{height:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:8px}
  .bar>i{display:block;height:100%;background:var(--accent);width:0}
  .switch{display:flex;align-items:center;gap:10px;margin:10px 0}
  .modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;padding:24px}
  .modal.open{display:flex}
  .modal .box{background:var(--panel);border:1px solid var(--border);border-radius:12px;
              max-width:820px;width:100%;max-height:86vh;overflow:auto;padding:20px}
  label{font-size:13px}
</style></head><body>
<header>
  <h1>🔍 SIF Engine</h1>
  <nav>
    <button data-tab="library" class="active">Library</button>
    <button data-tab="add">Add</button>
    <button data-tab="search">Search</button>
    <button data-tab="settings">Settings</button>
  </nav>
</header>
<main>
  <section id="library" class="tab active">
    <div class="row"><h2 style="margin:0">Indexed assets</h2>
      <button class="btn" style="margin-left:auto;padding:6px 12px" onclick="loadLibrary()">Refresh</button></div>
    <div id="libStats" class="muted" style="margin-top:6px"></div>
    <div id="libGrid" class="grid"></div>
  </section>

  <section id="add" class="tab">
    <div class="card">
      <h2>Analyze one image</h2>
      <div id="drop">Drop an image here, or click to choose</div>
      <input id="file" type="file" accept="image/*" hidden/>
      <div class="row" style="margin-top:12px">
        <label class="switch"><input id="save" type="checkbox" checked/> add to index</label>
        <button id="go" class="btn" style="margin-left:auto" disabled>Extract SIF</button>
      </div>
      <div id="upStatus" class="muted" style="margin-top:8px"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Bulk-index a folder</h2>
      <div class="row">
        <input id="folder" type="text" placeholder="C:\Users\you\Pictures\Screenshots"/>
        <button id="idxBtn" class="btn">Index folder</button>
      </div>
      <div class="bar" id="idxBarWrap" style="display:none"><i id="idxBar"></i></div>
      <div id="idxStatus" class="muted" style="margin-top:8px"></div>
    </div>
  </section>

  <section id="search" class="tab">
    <div class="row"><input id="q" type="text" placeholder="e.g. a code editor, an error dialog, invoice from March"/>
      <button id="searchBtn" class="btn">Search</button></div>
    <div id="searchInfo" class="muted" style="margin-top:8px"></div>
    <div id="searchGrid" class="grid"></div>
  </section>

  <section id="settings" class="tab">
    <div class="card">
      <h2>Settings</h2>
      <div id="setBody" class="muted">loading…</div>
    </div>
  </section>
</main>

<div class="modal" id="modal"><div class="box">
  <div class="row"><strong id="mTitle"></strong>
    <button class="btn" style="margin-left:auto;padding:4px 10px" onclick="closeModal()">Close</button></div>
  <div id="mSummary" style="margin:12px 0"></div>
  <pre id="mJson"></pre>
</div></div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

// tabs
$$('nav button').forEach(b=>b.onclick=()=>{
  $$('nav button').forEach(x=>x.classList.remove('active'));
  $$('.tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); $('#'+b.dataset.tab).classList.add('active');
  if(b.dataset.tab==='library') loadLibrary();
  if(b.dataset.tab==='settings') loadSettings();
});

function icon(kind){return kind==='pdf'?'📄':'🖼️'}
function card(a){
  const thumb = a.kind==='pdf' ? `<div class="img">📄</div>`
    : `<div class="img" style="background-image:url('/api/thumbnail?id='+encodeURIComponent(a.id)+`')"></div>`;
  const tags = (a.has_text?'<span class="pill">text</span>':'')+(a.n_faces?`<span class="pill">${a.n_faces} face</span>`:'');
  const pg = (a.page!=null)?` <span class="pill">p${a.page+1}</span>`:'';
  return `<div class="thumb" onclick="openAsset('${encodeURIComponent(a.id)}')">
    ${thumb}<div class="meta"><div class="name">${esc(a.path.split(/[\\/]/).pop())}${pg}</div>
    <div class="cap">${esc(a.caption||'')}</div>${tags}</div></div>`;
}

async function loadLibrary(){
  const g=$('#libGrid'); g.innerHTML='<span class="muted">loading…</span>';
  const d=await (await fetch('/api/library?limit=300')).json();
  const st=await (await fetch('/api/stats')).json();
  $('#libStats').textContent=`${st.indexed} assets · ${st.visual_vectors} visual · ${st.text_vectors} text · ${st.clip_vectors} CLIP vectors`;
  g.innerHTML = d.assets.length? d.assets.map(card).join('') : '<span class="muted">Nothing indexed yet. Use the Add tab.</span>';
}

async function openAsset(id){
  const sif=await (await fetch('/api/asset?id='+id)).json();
  $('#mTitle').textContent = (sif.file&&sif.file.path||'').split(/[\\/]/).pop();
  const mu=(sif.meta&&sif.meta.models_used)||{};
  $('#mSummary').innerHTML =
    `<div><span class="muted">caption:</span> ${esc(sif.scene?sif.scene.caption:'')||'<span class=muted>—</span>'}</div>
     <div><span class="muted">objects:</span> ${(sif.objects||[]).map(o=>`<span class=pill>${esc(o.label)}</span>`).join('')||'<span class=muted>none</span>'}</div>
     <div><span class="muted">ocr:</span> ${sif.ocr&&sif.ocr.has_text?esc(sif.ocr.full_text.slice(0,300)):'<span class=muted>no text</span>'}</div>
     <div><span class="muted">backends:</span> ${Object.entries(mu).map(([k,v])=>k+'='+v).join(', ')}</div>`;
  const s=JSON.parse(JSON.stringify(sif));
  if(s.embeddings){for(const k of ['visual','text','clip']){const v=s.embeddings[k];
    if(Array.isArray(v)&&v.length>6)s.embeddings[k]=v.slice(0,3).concat([`… (${v.length} floats)`]);}}
  if(Array.isArray(s.pages))s.pages.forEach(p=>{if(Array.isArray(p.text_vector)&&p.text_vector.length>6)p.text_vector=['…('+p.text_vector.length+')'];
    (p.regions||[]).forEach(r=>{['visual','clip'].forEach(k=>{if(Array.isArray(r[k])&&r[k].length>6)r[k]=['…('+r[k].length+')']})})});
  $('#mJson').textContent=JSON.stringify(s,null,2);
  $('#modal').classList.add('open');
}
function closeModal(){$('#modal').classList.remove('open')}
$('#modal').onclick=e=>{if(e.target.id==='modal')closeModal()};

// single upload
const drop=$('#drop'),fileInput=$('#file'),go=$('#go');let chosen=null;
drop.onclick=()=>fileInput.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])pick(ev.dataTransfer.files[0])});
fileInput.onchange=()=>{if(fileInput.files[0])pick(fileInput.files[0])};
function pick(f){chosen=f;go.disabled=false;drop.textContent=f.name}
go.onclick=async()=>{
  if(!chosen)return; go.disabled=true; $('#upStatus').textContent='Processing… (first run loads models)';
  const fd=new FormData();fd.append('file',chosen);fd.append('save',$('#save').checked?'true':'false');
  try{const d=await (await fetch('/api/process',{method:'POST',body:fd})).json();
    $('#upStatus').innerHTML=`<span class="good">Done — ${d.status}.</span> Open the Library to view it.`;
  }catch(e){$('#upStatus').innerHTML='<span class="err">'+e.message+'</span>'}
  finally{go.disabled=false}
};

// bulk folder
$('#idxBtn').onclick=async()=>{
  const path=$('#folder').value.trim(); if(!path)return;
  const fd=new FormData();fd.append('path',path);
  const r=await fetch('/api/index-folder',{method:'POST',body:fd});
  if(!r.ok){$('#idxStatus').innerHTML='<span class="err">'+(await r.json()).error+'</span>';return}
  const {job}=await r.json(); $('#idxBarWrap').style.display='block';
  poll(job);
};
async function poll(job){
  const j=await (await fetch('/api/index-status?job='+job)).json();
  const pct=j.total?Math.round(100*j.done/j.total):0;
  $('#idxBar').style.width=pct+'%';
  const tally=Object.entries(j.tally||{}).map(([k,v])=>k+'='+v).join(', ');
  $('#idxStatus').innerHTML=`${j.done}/${j.total} — ${esc(j.last||'')} <span class="muted">[${tally}]</span>`;
  if(j.status==='running'){setTimeout(()=>poll(job),700);}
  else if(j.status==='error'){$('#idxStatus').innerHTML='<span class="err">'+esc(j.error)+'</span>';}
  else{$('#idxStatus').innerHTML=`<span class="good">Done.</span> ${j.indexed_total} total indexed. [${tally}]`;}
}

// search
$('#searchBtn').onclick=doSearch; $('#q').addEventListener('keydown',e=>{if(e.key==='Enter')doSearch()});
async function doSearch(){
  const q=$('#q').value.trim(); if(!q)return;
  $('#searchInfo').textContent='searching…'; $('#searchGrid').innerHTML='';
  try{const d=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
    $('#searchInfo').innerHTML=`${d.count} result(s)`+(d.results[0]&&d.results[0].reranked?' · re-ranked':'');
    $('#searchGrid').innerHTML=d.results.map(r=>card({id:r.id.split('#')[0],path:r.path,kind:r.page!=null?'pdf':'image',
      caption:r.caption,page:r.page})).join('')||'<span class="muted">no results</span>';
  }catch(e){$('#searchInfo').innerHTML='<span class="err">'+e.message+'</span>'}
}

// settings
async function loadSettings(){
  const s=await (await fetch('/api/settings')).json();
  $('#setBody').innerHTML=`
    <div class="switch"><label><input type="checkbox" id="facesTog" ${s.enable_faces?'checked':''}
        ${s.faces_available?'':'disabled'}/> Enable facial recognition</label>
      ${s.faces_available?'':'<span class="warn">(install insightface to use)</span>'}</div>
    <div class="muted" style="font-size:12.5px">Faces are a biometric-privacy surface, off by default.
      Toggling affects <b>future</b> indexing only — re-index to add/remove faces.</div>
    <hr style="border-color:var(--border);margin:14px 0">
    <div><span class="muted">Data dir:</span> ${esc(s.data_root)}</div>
    <div><span class="muted">Scene VLM:</span> ${esc(s.vlm_model)}</div>
    <div><span class="muted">CLIP available:</span> ${s.clip_available?'<span class=good>yes</span>':'<span class=warn>no</span>'}</div>
    <div><span class="muted">Mode:</span> ${s.using_stubs?'<span class=warn>stubs (no real models)</span>':'<span class=good>real models</span>'}</div>`;
  const tog=$('#facesTog');
  if(tog) tog.onchange=async()=>{const fd=new FormData();fd.append('enable_faces',tog.checked?'true':'false');
    await fetch('/api/settings',{method:'POST',body:fd});};
}

loadLibrary();
</script></body></html>"""
