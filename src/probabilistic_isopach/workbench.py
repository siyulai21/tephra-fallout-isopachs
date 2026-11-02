"""Local upload workspace. Analyses run in isolated subprocesses and folders."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from urllib.parse import unquote, urlparse
import uuid
import webbrowser
import zipfile

import numpy as np
import pandas as pd

from .catalog import datasets
from .data import read_observations
from .publication import CSS
from .validation import spatial_folds


MAX_UPLOAD = 5_000_000


def prepare_upload(payload):
    """Return original bytes, analysis bytes and a safe display title."""
    if payload.get("dataset"):
        matches = [r for r in datasets() if r["slug"] == payload["dataset"]]
        if not matches:
            raise ValueError("Choose one of the listed example deposits.")
        row = matches[0]
        raw = row["path"].read_bytes()
        default_name = row["name"]
    else:
        text = payload.get("csv")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Choose a CSV file or an example deposit first.")
        raw = text.encode("utf-8")
        default_name = Path(str(payload.get("filename", "New eruption"))).stem
    if len(raw) > MAX_UPLOAD:
        raise ValueError("The CSV must be smaller than 5 MB.")
    analysis = raw
    lon, lat = payload.get("vent_lon"), payload.get("vent_lat")
    if lon is not None or lat is not None:
        if lon is None or lat is None:
            raise ValueError("Enter both vent longitude and latitude, or leave both blank.")
        lon, lat = float(lon), float(lat)
        if not np.isfinite([lon, lat]).all() or abs(lon) > 180 or abs(lat) > 90:
            raise ValueError("Vent longitude must be −180 to 180; latitude −90 to 90.")
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        if frame.empty:
            raise ValueError("The CSV contains no observations.")
        for column, value in (("Vent Lon", lon), ("Vent Lat", lat)):
            if column in frame:
                supplied = pd.to_numeric(frame[column], errors="raise").dropna()
                if len(supplied) and not np.allclose(supplied, value, atol=1e-8, rtol=0):
                    raise ValueError("Entered vent coordinates disagree with the CSV; correct one source before running.")
            frame[column] = value
        analysis = frame.to_csv(index=False).encode("utf-8")
    name = str(payload.get("name") or default_name).strip()[:100]
    if not name or any(ord(c) < 32 for c in name):
        raise ValueError("Use a nonempty eruption name without control characters.")
    return raw, analysis, name


def preview_upload(payload, workspace):
    raw, analysis, name = prepare_upload(payload)
    with tempfile.TemporaryDirectory(prefix=".preview-", dir=workspace) as temp:
        path = Path(temp)/"input.csv"
        path.write_bytes(analysis)
        data, meta = read_observations(path)
    detections = ~data.censored
    if detections.sum() < 6 or len(np.unique(data.xy[detections], axis=0)) < 4:
        raise ValueError("At least six measured positive thicknesses at four distinct locations are required.")
    def sufficient(obs):
        detected = ~obs.censored
        return detected.sum() >= 6 and len(np.unique(obs.xy[detected], axis=0)) >= 4
    folds = spatial_folds(data.xy, 4)
    for fold in range(4):
        training = data.subset(folds != fold)
        if not sufficient(training):
            raise ValueError("Some geographic training groups have fewer than six measurements at four locations. Add observations with broader coverage before analysing.")
        if payload.get("nested_validation", True):
            inner = spatial_folds(training.xy, 3)
            if any(not sufficient(training.subset(inner != k)) for k in range(3)):
                raise ValueError("This dataset is too small or clustered for independent nested validation. Add measurements or turn off the withheld-region check for an exploratory run.")
    points = [{"x": float(p[0]), "y": float(p[1]), "value_cm": float(np.exp(v)), "upper_limit": bool(c)}
              for p, v, c in zip(data.xy, data.log_value, data.censored)]
    return {"name": name, "measurements": meta["n_measurements"], "upper_limits": meta["n_upper_limits"],
            "excluded_pseudo": len(meta["ignored_pseudo_rows"]), "vent": meta["vent_lonlat"],
            "original_sha256": sha256(raw).hexdigest(), "analysis_sha256": sha256(analysis).hexdigest(),
            "ignored_columns": meta["ignored_extra_columns"], "points": points}


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.slot = threading.Semaphore(1)
        self.active = set()
        self.lock = threading.Lock()

    def list_runs(self):
        rows = []
        for p in sorted(self.root.glob("*/job.json"), reverse=True):
            try:
                row = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            with self.lock:
                active = row["id"] in self.active
            if row["status"] in {"queued", "running"} and not active:
                if (p.parent/"report/virtual_predictions.npz").is_file() and (p.parent/"report/report.json").is_file():
                    row["status"] = "complete"
                else:
                    row["status"] = "interrupted"
                    row["error"] = "The previous workspace session ended before this run completed. Start a new run to retry."
            row["report_url"] = f"/runs/{row['id']}/report/index.html" if row["status"] == "complete" else None
            rows.append(row)
        return rows

    def start(self, payload):
        preview = preview_upload(payload, self.root)
        raw, analysis, name = prepare_upload(payload)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
        folder = self.root/identifier
        folder.mkdir()
        (folder/"original.csv").write_bytes(raw)
        (folder/"input.csv").write_bytes(analysis)
        state = {"id": identifier, "name": name, "status": "queued", "preview": preview,
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "nested_validation": bool(payload.get("nested_validation", True))}
        with self.lock:
            self.active.add(identifier)
        _write_json(folder/"job.json", state)
        threading.Thread(target=self._run, args=(folder, state), daemon=True).start()
        return state

    def _run(self, folder, state):
        with self.slot:
            state["status"] = "running"
            _write_json(folder/"job.json", state)
            command = [sys.executable, "-m", "src.probabilistic_isopach", "fit", str(folder/"input.csv"),
                       "--output", str(folder/"report"), "--name", state["name"], "--draws", "1500"]
            if state["nested_validation"]:
                command.append("--nested-cv")
            env = {**os.environ, "MPLCONFIGDIR": str(self.root/".matplotlib"), "PYTHONUNBUFFERED": "1"}
            try:
                with (folder/"analysis.log").open("w") as log:
                    result = subprocess.run(command, cwd=Path(__file__).parents[2], env=env,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    lines = (folder/"analysis.log").read_text().strip().splitlines()
                    raise RuntimeError(lines[-1] if lines else "Analysis stopped before producing a report.")
                state["status"] = "complete"
                report = json.loads((folder/"report/report.json").read_text())
                state["expected_regional_bulk_km3"] = report["volume_analysis"]["regional"]["mean_km3"]
            except Exception as exc:
                state["status"] = "failed"
                state["error"] = str(exc)[:1500]
            finally:
                state["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(folder/"job.json", state)
                with self.lock:
                    self.active.discard(state["id"])


def workspace_html():
    extra = """
input,select,button{font:inherit}input[type=text],input[type=number],select{width:100%;border:1px solid #c6d7d1;padding:11px 12px;border-radius:7px;background:white;color:var(--ink)}label.field{display:block;font-size:13px;font-weight:600;margin:17px 0 7px}.drop{border:1.5px dashed #8eb8ad;border-radius:12px;background:#f1f7f2;padding:28px;text-align:center;cursor:pointer;display:block}.drop:hover,.drop.drag{background:#e5f1e9}.drop b{display:block;margin:8px 0}input[type=file]{max-width:100%;font-size:13px}.form-panel{padding:25px;background:white;border:1px solid var(--line);border-radius:12px}.button{border:0;border-radius:7px;background:var(--teal);color:white;padding:12px 22px;cursor:pointer;font-weight:600}.button.secondary{background:#e6efeb;color:var(--ink)}.button:disabled{opacity:.45;cursor:default}.actions{display:flex;gap:12px;margin-top:22px;flex-wrap:wrap}.error{color:#a23e29;background:#fff1eb;padding:14px;border-radius:7px;margin-top:16px}.success{background:#eaf4ec;padding:18px;border-radius:8px;margin-top:20px}.hidden{display:none!important}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.run{padding:18px 0;border-top:1px solid var(--line)}.run strong{display:block}.run a{display:inline-block;margin:8px 16px 0 0}.status{font-size:13px;color:var(--muted)}.spinner{display:inline-block;width:11px;height:11px;border:2px solid #adcac1;border-top-color:var(--teal);border-radius:50%;animation:spin 1s linear infinite;margin-right:7px}@keyframes spin{to{transform:rotate(360deg)}}#preview svg{width:100%;max-height:210px;background:white;border:1px solid var(--line);border-radius:7px;margin:12px 0}.step{font-size:11px;letter-spacing:.13em;color:var(--teal);font-weight:700;margin-bottom:10px}.check{display:flex;gap:9px;align-items:flex-start;font-size:14px;margin-top:18px}@media(max-width:720px){.two{grid-template-columns:1fr}}
"""
    body = r"""
<div class="eyebrow">Tephra / Local analysis workspace</div><h1>A new eruption,<br>a clear picture.</h1><p class="lead">Upload thickness measurements. Preview the data. Get closed isopachs, volume estimates, plots and a short report—all on your computer.</p><!-- atlas -->
<div class="grid" style="margin-top:34px;align-items:start"><section class="form-panel"><div class="step">01 / CHOOSE YOUR DATA</div>
<label class="drop" id="drop"><span style="font-size:28px;color:var(--teal)">↥</span><b>Drop an eruption CSV here</b><span class="muted" style="font-size:13px">or choose a file · thickness in centimetres</span><br><input id="file" type="file" accept=".csv,text/csv" aria-label="Eruption CSV" style="margin-top:16px"></label>
<label class="field" for="example">Or explore an included eruption</label><select id="example"><option value="">Choose an example…</option></select>
<label class="field" for="name">Eruption name</label><input id="name" type="text" placeholder="e.g. My eruption" maxlength="100">
<details style="margin-top:20px"><summary>Vent coordinates, if missing from the CSV</summary><p><small>Leave blank when the CSV already supplies Vent Lon and Vent Lat. Enter decimal degrees.</small></p><div class="two"><div><label class="field" for="lon">Longitude</label><input id="lon" type="number" step="any" min="-180" max="180" placeholder="−180 to 180"></div><div><label class="field" for="lat">Latitude</label><input id="lat" type="number" step="any" min="-90" max="90" placeholder="−90 to 90"></div></div></details>
<label class="check"><input id="validation" type="checkbox" checked><span>Check predictions against withheld geographic regions<br><small>Recommended; adds time to the analysis.</small></span></label>
<div class="actions"><button id="preview-button" class="button secondary">Preview measurements</button><button id="run-button" class="button" disabled>Run analysis →</button></div><div id="error" class="error hidden" role="alert"></div>
<div id="preview" class="success hidden" aria-live="polite"></div></section>
<aside><div class="step">02 / A COMPLETE REPORT</div><h2>From observations to an interpretable result.</h2><p class="muted">The same simple statistical fit is used for every eruption. No hand-placed pseudo-points or historical wind profile is required.</p><div class="panel" style="padding:24px"><h3>Maps you can share</h3><p class="muted">Closed contours, geographic context and visible extrapolation. Download PDF, SVG, PNG and GIS contour lines.</p><h3>Volumes with their scope</h3><p class="muted">Checked regional bulk volumes, numerical refinement and sensitivity to the integration region.</p><h3>A short report and the details</h3><p class="muted" style="margin-bottom:0">A concise Markdown summary, an interactive browser report, plots, CSV tables and the full reproducibility record.</p></div>
<details><summary>What should my CSV contain?</summary><p><code>Lon, Lat, Thickness</code> plus <code>Vent Lon, Vent Lat</code> in the file or entered above. Use at least six positive thickness measurements at four distinct locations.</p><p>Optional <code>Pseudo=Yes</code> rows are excluded. A non-detection needs <code>Kind=upper_limit</code> and a documented <code>Limit_cm</code>. An unsampled location is not zero thickness.</p><a href="/template.csv" download="eruption-template.csv">Download a CSV header template</a></details>
<div class="note warn">Closing a contour does not establish total eruption volume. Reports distinguish measured support, extrapolation and unresolved tails.</div></aside></div>
<div class="rule"></div><div class="step">03 / YOUR ANALYSES</div><h2>Results stay together.</h2><p class="muted">Runs are saved separately. Keep this workspace open while an analysis runs; reports can be reopened or downloaded afterwards.</p><div id="runs" aria-live="polite"><p class="muted">No analyses yet.</p></div><footer>Localhost only. No uploads to a remote service. Original and analysis CSVs are retained separately when vent coordinates are added.</footer>
<script>
const $=id=>document.getElementById(id);let chosenFile=null,approved=null,previewSerial=0;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function reset(){approved=null;previewSerial++;$('run-button').disabled=true;$('preview').classList.add('hidden');$('error').classList.add('hidden')}
function error(message){$('error').textContent=message;$('error').classList.remove('hidden')}
async function request(url,options={}){const r=await fetch(url,options);const j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');return j}
function setFile(file){const previous=$('example').selectedOptions[0]?.textContent;chosenFile=file;if(file&&(!$('name').value||$('name').value===previous))$('name').value=file.name.replace(/\.csv$/i,'');$('example').value='';reset()}
$('file').addEventListener('change',()=>setFile($('file').files[0]));
$('drop').addEventListener('dragover',e=>{e.preventDefault();$('drop').classList.add('drag')});$('drop').addEventListener('dragleave',()=>$('drop').classList.remove('drag'));$('drop').addEventListener('drop',e=>{e.preventDefault();$('drop').classList.remove('drag');if(e.dataTransfer.files.length){setFile(e.dataTransfer.files[0]);$('drop').querySelector('b').textContent=chosenFile.name}});
$('example').addEventListener('change',()=>{chosenFile=null;$('file').value='';if($('example').value)$('name').value=$('example').selectedOptions[0].textContent;reset()});
for(const id of ['name','lon','lat','validation'])$(id).addEventListener('input',reset);
async function payload(){if(chosenFile&&chosenFile.size>5000000)throw new Error('Choose a CSV smaller than 5 MB.');return {dataset:$('example').value||null,csv:chosenFile?await chosenFile.text():null,filename:chosenFile?.name,name:$('name').value,vent_lon:$('lon').value===''?null:Number($('lon').value),vent_lat:$('lat').value===''?null:Number($('lat').value),nested_validation:$('validation').checked}}
function scatter(points){let xs=[0,...points.map(p=>p.x)],ys=[0,...points.map(p=>p.y)],xmin=Math.min(...xs),ymin=Math.min(...ys),sx=Math.max(...xs)-xmin||1,sy=Math.max(...ys)-ymin||1;const scale=Math.min(360/sx,150/sy),cx=x=>30+(x-xmin)*scale,cy=y=>180-(y-ymin)*scale;return `<svg viewBox="0 0 420 210" role="img" aria-label="Projected measurement locations and vent">${points.map(p=>`<circle cx="${cx(p.x)}" cy="${cy(p.y)}" r="3" fill="${p.upper_limit?'#af6c25':'#087e8b'}" opacity=".8"/>`).join('')}<text x="${cx(0)}" y="${cy(0)+5}" text-anchor="middle" fill="#b9573c" font-size="20">★</text><text x="12" y="204" fill="#5c7075" font-size="10">Measurement locations · star = vent · projected kilometres</text></svg>`}
$('preview-button').addEventListener('click',async()=>{reset();const serial=previewSerial;$('preview-button').disabled=true;try{const p=await payload();const result=await request('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});if(serial!==previewSerial)return;approved=p;$('preview').innerHTML=`<b>${esc(result.name)} · ready to analyse</b><p style="margin-top:8px;margin-bottom:0">${result.measurements} measurements · ${result.upper_limits} upper limits · ${result.excluded_pseudo} pseudo rows excluded</p><small>Vent: ${result.vent.map(v=>v.toFixed(4)).join(', ')}</small>${scatter(result.points)}<small>Preview checks the input structure. Source provenance and preservation still require geological interpretation.</small>`;$('preview').classList.remove('hidden');$('run-button').disabled=false}catch(e){error(e.message)}finally{$('preview-button').disabled=false}});
$('run-button').addEventListener('click',async()=>{if(!approved)return;$('run-button').disabled=true;try{await request('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(approved)});approved=null;await refresh();$('runs').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){error(e.message);$('run-button').disabled=false}});
async function refresh(){try{const rows=await request('/api/runs');$('runs').innerHTML=rows.length?rows.map(r=>`<article class="run"><strong>${esc(r.name)}</strong><div class="status">${['queued','running'].includes(r.status)?'<span class="spinner"></span>':''}${esc(({queued:'Queued · another analysis is running',running:'Analysing measurements, contours and volumes…',complete:'Complete',failed:'Analysis needs attention',interrupted:'Previous run interrupted'})[r.status]||r.status)} · ${esc(new Date(r.created_utc).toLocaleString())}</div>${r.error?`<p class="error">${esc(r.error)}</p>`:''}${r.status==='complete'?`<a href="${r.report_url}" target="_blank" rel="noopener">Open report ↗</a><a href="/runs/${r.id}/report/short_report.md" download>Short report</a><a href="/api/runs/${r.id}/download">Download everything · ZIP</a>`:''}</article>`).join(''):'<p class="muted">No analyses yet. Your first report will appear here.</p>'}catch(e){$('runs').textContent='Workspace connection unavailable. Reopen the local server to view saved results.'}}
request('/api/datasets').then(rows=>{for(const row of rows){const o=document.createElement('option');o.value=row.slug;o.textContent=row.name;$('example').appendChild(o)}}).catch(e=>error(e.message));refresh();setInterval(refresh,3000);
</script>
"""
    atlas = '<div class="downloads"><a href="/atlas/index.html" target="_blank" rel="noopener">Explore the four-deposit atlas ↗</a></div>' if (Path(__file__).parents[2]/"docs/atlas/index.html").is_file() else ""
    body = body.replace("<!-- atlas -->", atlas)
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tephra · Upload workspace</title><style>'+CSS+extra+'</style></head><body><main>'+body+'</main></body></html>'


def handler_for(workspace):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type="application/json", filename=None):
            if content_type == "application/json":
                body = json.dumps(body, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            port = self.server.server_port
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (origin is None or origin in {f"http://{h}" for h in allowed})

        def do_GET(self):
            if not self.local_request():
                return self.send(403, {"error": "This workspace accepts local browser requests only."})
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self.send(200, workspace_html(), "text/html; charset=utf-8")
            if path == "/api/datasets":
                return self.send(200, [{"name": r["name"], "slug": r["slug"]} for r in datasets()])
            if path == "/api/runs":
                return self.send(200, workspace.list_runs())
            if path == "/template.csv":
                return self.send(200, "Lon,Lat,Thickness,Vent Lon,Vent Lat,Pseudo,Kind,Limit_cm,LogSD\n", "text/csv", "eruption-template.csv")
            if path.startswith("/atlas/"):
                atlas_root = (Path(__file__).parents[2]/"docs/atlas").resolve()
                file = (atlas_root/path[len("/atlas/"):]).resolve()
                if file.is_relative_to(atlas_root) and file.is_file() and file.suffix in {".html", ".json", ".geojson", ".csv", ".png", ".svg", ".pdf", ".md"}:
                    return self.send(200, file.read_bytes(), mimetypes.guess_type(str(file))[0] or "application/octet-stream")
            if path.startswith("/api/runs/") and path.endswith("/download"):
                identifier = path.split("/")[3]
                if not any(r["id"] == identifier and r["status"] == "complete" for r in workspace.list_runs()):
                    return self.send(404, {"error": "Completed report not found."})
                folder = workspace.root/identifier
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file in sorted(folder.rglob("*")):
                        if file.is_file() and file.suffix != ".log" and not file.is_symlink():
                            archive.write(file, file.relative_to(folder))
                return self.send(200, buffer.getvalue(), "application/zip", f"tephra-{identifier}.zip")
            if path.startswith("/runs/"):
                relative = Path(path[len("/runs/"):])
                file = (workspace.root/relative).resolve()
                if (file.is_relative_to(workspace.root) and len(relative.parts) >= 3
                        and relative.parts[1] == "report" and file.is_file()
                        and file.suffix in {".html", ".json", ".geojson", ".csv", ".png", ".svg", ".pdf", ".npz", ".md"}):
                    kind = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                    return self.send(200, file.read_bytes(), kind)
            self.send(404, {"error": "Not found."})

        def do_POST(self):
            if not self.local_request():
                return self.send(403, {"error": "Uploads must come from this local workspace."})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_UPLOAD+100000:
                    raise ValueError("Choose a CSV smaller than 5 MB.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Upload must be a JSON object.")
                if self.path == "/api/preview":
                    return self.send(200, preview_upload(payload, workspace.root))
                if self.path == "/api/runs":
                    return self.send(202, workspace.start(payload))
                return self.send(404, {"error": "Not found."})
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return self.send(400, {"error": str(exc)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open a local CSV-upload and analysis workspace.")
    parser.add_argument("--output", type=Path, default=Path("results/workbench"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    workspace = Workspace(args.output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(workspace))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Tephra workspace: {url}\nResults: {workspace.root}\nKeep this process open while analysing. Stop with Ctrl+C.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkspace stopped. Saved reports remain on disk.")
    finally:
        server.server_close()
