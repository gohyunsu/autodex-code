"""Local Flask UI for the live AutoDex perception checker.

The controller passed to :func:`run_dashboard` owns all camera/GPU work.  This
module only serves control/status requests and files below a completed run's
artifact directory; it never starts a second perception pipeline itself.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoDex Perception Check</title>
<style>
:root { color-scheme: dark; --bg:#111827; --panel:#1f2937; --line:#374151; --muted:#9ca3af;
        --ok:#34d399; --warn:#fbbf24; --bad:#fb7185; --accent:#60a5fa; }
* { box-sizing:border-box } body { margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:#f9fafb; }
main { max-width:1500px; margin:auto; padding:24px; } h1 { margin:0; font-size:25px; } h2 { font-size:16px; margin:0 0 12px; }
.sub { color:var(--muted); margin:6px 0 20px; font-size:13px; }.layout { display:grid; grid-template-columns:minmax(300px,.9fr) minmax(0,2.1fr); gap:16px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }.grow { flex:1 }.badge { border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; background:#334155; }
.ok { color:var(--ok) }.warn { color:var(--warn) }.bad { color:var(--bad) }.muted { color:var(--muted) }.small { font-size:12px }.mono { font-family:ui-monospace,SFMono-Regular,monospace; overflow-wrap:anywhere }
input,button { min-height:38px; border-radius:7px; border:1px solid #4b5563; padding:8px 10px; font:inherit; }
input { color:#f9fafb; background:#111827; width:100%; } button { color:white; background:#2563eb; cursor:pointer; font-weight:600; } button:disabled { opacity:.45; cursor:not-allowed; }
button.secondary { background:#374151 } .timeline { display:grid; gap:7px }.stage { display:grid; grid-template-columns:22px 125px 1fr; align-items:center; gap:8px; font-size:13px; }
.dot { width:10px; height:10px; border-radius:50%; background:#4b5563 }.dot.running { background:var(--accent); box-shadow:0 0 10px var(--accent) }.dot.done { background:var(--ok) }.dot.error { background:var(--bad) }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px }.metric { background:#111827; padding:10px; border-radius:7px }.metric label { display:block; color:var(--muted); font-size:11px; }.metric strong { font-size:18px; display:block; margin-top:3px; }
table { border-collapse:collapse; width:100%; font-size:12px } th,td { padding:7px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top } th { color:var(--muted); font-weight:600; }
.scroll { max-height:310px; overflow:auto }.tabs { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px }.tabs button { background:#374151; font-size:12px; min-height:31px }.tabs button.active { background:#2563eb }
.grid-image { width:100%; border:1px solid var(--line); background:#030712; min-height:100px; object-fit:contain; cursor:zoom-in }.notice { border-left:3px solid var(--warn); padding:8px 10px; background:#252117; font-size:13px; }
.legend { display:flex; flex-wrap:wrap; gap:8px 14px; margin:0 0 12px; font-size:12px; color:var(--muted) }.legend span { display:inline-flex; align-items:center; gap:5px }.swatch { width:12px; height:12px; border-radius:3px; display:inline-block; border:1px solid #e5e7eb55 }
.asset-map { display:flex; align-items:stretch; gap:6px; overflow:auto; padding:3px 0 10px }.asset-node { flex:1 0 105px; border:1px solid var(--line); background:#111827; border-radius:7px; padding:8px; font-size:11px }.asset-node.ok { border-color:#16885f }.asset-node.bad { border-color:#b53c55 }.asset-node .asset-title { display:block; color:#d1d5db; font-weight:700; margin-bottom:4px }.asset-arrow { align-self:center; color:var(--muted) }.object-preview { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(220px,.85fr); gap:12px; margin-top:12px }.object-preview figure { margin:0; border:1px solid var(--line); background:#030712; border-radius:7px; overflow:hidden }.object-preview figure img { display:block; width:100%; min-height:150px; object-fit:contain }.object-preview figcaption { padding:7px 9px; font-size:11px; color:var(--muted) }.profile-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px }.profile-item { background:#111827; border-radius:6px; padding:7px; min-width:0 }.profile-item label { display:block; color:var(--muted); font-size:10px }.profile-item strong { display:block; font-size:12px; margin-top:3px; overflow-wrap:anywhere }
.hidden { display:none }.resource-ok { color:var(--ok) }.resource-missing { color:var(--bad) } dialog { width:min(96vw,1500px); padding:0; background:#0b1020; border:1px solid var(--line); border-radius:10px; } dialog img { max-width:100%; display:block } dialog button { margin:10px; float:right; }
@media(max-width:900px) { main { padding:12px }.layout{grid-template-columns:1fr} }
</style>
</head>
<body><main>
  <div class="row"><div class="grow"><h1>AutoDex live perception check</h1><div class="sub">Object → SAM3 → per-view FoundPose → cross-view IoU → silhouette refinement. Pose is an output, never a manual input.</div></div><span id="session-state" class="badge">connecting</span></div>
  <div class="layout"><aside>
    <section class="card"><h2>Run perception</h2><div class="row"><div class="grow"><input id="object" list="objects" autocomplete="off" placeholder="Enter a v8 object name"><datalist id="objects"></datalist></div><button id="run">Run</button></div><p id="validation" class="small muted">Select an object to inspect its local perception resources.</p><div id="object-resources" class="scroll"></div><div id="object-visual" class="hidden"><h2 style="margin-top:14px">Selected object visual inspection</h2><div id="asset-map" class="asset-map"></div><div class="object-preview"><figure><img id="mesh-preview" alt="Perception raw mesh preview"><figcaption>Exact v8 <span class="mono">raw_mesh/&lt;object&gt;.obj</span> used for FoundPose. Three orthographic views; point colour is mesh colour when available, otherwise a stable axis cue.</figcaption></figure><div><div id="asset-profile" class="profile-grid"></div><p class="small muted">The representation is a binary FoundPose feature archive; this panel exposes its small onboarding metadata/config instead of loading the full archive into the browser.</p></div></div></div></section>
    <section class="card"><h2>Session resources</h2><div id="session-info" class="small muted">loading…</div><div id="session-resources" class="scroll"></div></section>
    <section class="card"><h2>Recent runs</h2><div id="history" class="small muted">none</div></section>
  </aside><section>
    <section class="card"><h2>Pipeline progress</h2><div id="timeline" class="timeline"></div></section>
    <section id="result-card" class="card"><h2>Result summary</h2><div id="result" class="muted">No run yet.</div></section>
    <section class="card"><h2>Stage overlays</h2><div class="legend"><span><i class="swatch" style="background:#ffbf00"></i>Amber: SAM3 segmentation mask</span><span><i class="swatch" style="background:#b450b4"></i>Purple: per-view FoundPose candidate</span><span><i class="swatch" style="background:#2896ff"></i>Blue: cross-view IoU selected pose</span><span><i class="swatch" style="background:#1ed21e"></i>Green: silhouette-refined final pose</span><span><i class="swatch" style="background:#111827"></i>Dark tile/text: missing image or payload</span></div><div id="stage-tabs" class="tabs"></div><div id="stage-empty" class="notice">After a run, select SAM3, FoundPose, IoU, or final silhouette to inspect every camera view.</div><img id="stage-image" class="grid-image hidden" alt="Perception stage grid"></section>
    <section class="card"><h2>Camera and output resources</h2><div id="run-resources" class="scroll muted">No run yet.</div><div id="cameras" class="scroll"></div></section>
  </section></div>
</main><dialog id="image-dialog"><button id="close-dialog">Close</button><img id="large-image" alt="Full stage grid"></dialog>
<script>
const $ = id => document.getElementById(id); const stageOrder = ['validate','initialize','collect','iou','silhouette','persist','complete'];
const stageNames = {validate:'Validate assets', initialize:'Initialize object', collect:'SAM3 + FoundPose collection', iou:'Cross-view IoU selection', silhouette:'Silhouette refinement', persist:'Save artifacts', complete:'Complete'};
let activeRun = null, selectedStage = null, objectsLoaded = false;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (value, digits=2) => value == null ? '—' : Number(value).toFixed(digits);
const api = async (url, options={}) => { const r = await fetch(url, options); const j = await r.json(); if (!r.ok) throw Object.assign(new Error(j.error || 'request failed'), {data:j}); return j; };
function resourceTable(items) { if (!items?.length) return '<span class="muted">none</span>'; return `<table><thead><tr><th>resource</th><th>state</th><th>source stage</th><th>saved</th><th>path</th></tr></thead><tbody>${items.map(x => { const state=x.exists === true ? 'available' : (x.exists === false ? (x.required === false ? 'optional missing' : 'missing') : 'not probed'); const cls=x.exists === true ? 'resource-ok' : (x.exists === false ? 'resource-missing' : 'muted'); const note=x.note ? `<div class="muted small">${esc(x.note)}</div>` : ''; const stage=x.source_stage ? `${esc(stageNames[x.source_stage] || x.source_stage)}${x.source_stage_completed_offset_s != null ? ` <span class="muted">@ ${fmt(x.source_stage_completed_offset_s)}s</span>` : ''}` : '—'; const saved=x.saved_offset_s != null ? `<span title="${esc(x.saved_at || '')}">+${fmt(x.saved_offset_s)}s</span>` : '—'; return `<tr><td>${esc(x.label || x.key)}${note}</td><td class="${cls}">${state}</td><td>${stage}</td><td>${saved}</td><td class="mono">${esc(x.path || '')}${x.size_bytes != null ? ` <span class="muted">(${x.size_bytes} B)</span>` : ''}</td></tr>`; }).join('')}</tbody></table>`; }
function renderObjectVisuals(data) {
  const panel = $('object-visual');
  if (!data.valid) { panel.classList.add('hidden'); return; }
  const byKey = Object.fromEntries((data.resources || []).map(x => [x.key, x]));
  const flow = [['v8_list', 'v8 allow-list'], ['object_root', 'object root'],
                ['mesh', 'perception mesh'], ['foundpose_assets', 'FoundPose assets'],
                ['representation', 'feature archive']];
  $('asset-map').innerHTML = flow.map(([key, label], index) => {
    const resource = byKey[key] || {};
    const state = resource.exists === true ? 'available' :
        (resource.exists === false ? 'missing' : 'not probed');
    const nodeClass = resource.exists === false ? 'bad' : 'ok';
    const textClass = resource.exists === true ? 'ok' :
        (resource.exists === false ? 'bad' : 'muted');
    const arrow = index ? '<span class="asset-arrow">→</span>' : '';
    return arrow + '<div class="asset-node ' + nodeClass + '">' +
        '<span class="asset-title">' + esc(label) + '</span>' +
        '<span class="' + textClass + '">' + state + '</span>' +
        '<div class="mono muted" style="margin-top:4px">' +
        esc(resource.path || '') + '</div></div>';
  }).join('');
  const profile = data.asset_profile || {};
  const geometry = profile.geometry_mm || {};
  const size = [geometry.size_x, geometry.size_y, geometry.size_z].every(x => x != null)
      ? fmt(geometry.size_x, 1) + ' × ' + fmt(geometry.size_y, 1) +
        ' × ' + fmt(geometry.size_z, 1) + ' mm'
      : '—';
  const valueOrDash = value => value == null ? '—' : value;
  const profileItems = [
    ['v8 catalogue', data.catalog_index ? '#' + data.catalog_index : '—'],
    ['templates', profile.template_count == null ? '—' : Number(profile.template_count).toLocaleString()],
    ['feature vectors', profile.feature_vector_count == null ? '—' : Number(profile.feature_vector_count).toLocaleString()],
    ['extractor', profile.extractor || '—'],
    ['geometry', size],
    ['diameter', geometry.diameter == null ? '—' : fmt(geometry.diameter, 1) + ' mm'],
    ['viewpoints / rotations', valueOrDash(profile.min_viewpoints) + ' / ' + valueOrDash(profile.inplane_rotations)],
    ['PCA / clusters', valueOrDash(profile.pca_components) + ' / ' + valueOrDash(profile.cluster_count)],
  ];
  $('asset-profile').innerHTML = profileItems.map(([label, value]) =>
      '<div class="profile-item"><label>' + esc(label) + '</label><strong>' +
      esc(value) + '</strong></div>').join('');
  const preview = $('mesh-preview');
  preview.alt = data.object + ' perception raw mesh preview';
  preview.src = '/api/objects/' + encodeURIComponent(data.object) +
      '/preview/mesh?t=' + Date.now();
  panel.classList.remove('hidden');
}
function renderTimeline(state) { const stages = state.stages || {}; $('timeline').innerHTML = stageOrder.map(key => { const x = stages[key] || {}; const status = x.state || 'waiting'; const dot = status === 'done' ? 'done' : (status === 'running' || status === 'queued' ? 'running' : (status === 'error' ? 'error' : '')); let detail = status; if (key === 'collect' && x.n_expected != null) detail += ` · masks ${x.n_masks_recv ?? 0}/${x.n_expected} · poses ${x.n_poses_recv ?? 0}/${x.n_expected} · ${fmt(x.elapsed_s)}s`; else if (x.elapsed_s != null) detail += ` · ${fmt(x.elapsed_s)}s`; else if (x.reused) detail += ' · reused object state'; return `<div class="stage"><span class="dot ${dot}"></span><span>${stageNames[key]}</span><span class="muted">${esc(detail)}</span></div>`; }).join(''); }
function poseText(pose) { if (!pose) return '—'; const xyz = [pose[0]?.[3],pose[1]?.[3],pose[2]?.[3]].map(v => fmt(v,4)).join(', '); return `[${xyz}] m`; }
function renderResult(record) { if (!record) return; const t=record.timing||{}; const status=record.status||'FAIL'; const cls=status==='PASS'?'ok':(status==='WARN'?'warn':'bad'); const persist=record.stage_timing?.persist||{}; $('result').innerHTML = `<div class="metrics"><div class="metric"><label>status</label><strong class="${cls}">${esc(status)}</strong></div><div class="metric"><label>views</label><strong>${record.n_masks_recv ?? 0}/${record.n_expected ?? 0}</strong><span class="small muted">masks / expected</span></div><div class="metric"><label>valid FoundPose</label><strong>${record.n_candidates ?? 0}</strong></div><div class="metric"><label>best IoU</label><strong>${fmt(t.best_iou,3)}</strong><span class="small muted">${esc(t.best_serial || '—')}</span></div><div class="metric"><label>silhouette loss</label><strong>${fmt(t.sil_loss,6)}</strong></div><div class="metric"><label>perception total</label><strong>${fmt(record.total_s)}s</strong><span class="small muted">artifact save excluded</span></div><div class="metric"><label>artifact save</label><strong>${fmt(persist.duration_s)}s</strong><span class="small muted">post-perception</span></div></div><p class="small ${record.reason ? 'bad' : 'muted'}">${record.reason ? `reason: ${esc(record.reason)}` : 'Pose world: ' + esc(poseText(record.pose_world))}</p><div class="mono small">${esc(record.run_dir || '')}</div>`;
  const runResources=[...(record.resources?.inputs||[]), ...(record.artifact_manifest?.length ? record.artifact_manifest : (record.resources?.artifacts||[]))]; $('run-resources').innerHTML = resourceTable(runResources);
  const cams=record.cameras||[], camRes=Object.fromEntries((record.resources?.cameras||[]).map(x=>[x.serial,x])); $('cameras').innerHTML = `<table><thead><tr><th>serial</th><th>image</th><th>mask</th><th>FoundPose</th><th>quality</th><th>inliers</th><th>SAM3 / FP</th></tr></thead><tbody>${cams.map(c=>{const r=camRes[c.serial]||{}; const found=c.foundpose_ok?'OK':(c.pose_received?'FAIL':'missing'); return `<tr><td class="mono">${esc(c.serial)}</td><td class="${r.image_saved?'ok':'bad'}">${r.image_saved?'saved':'missing'}</td><td>${c.mask_received?`${fmt(100*c.mask_ratio,1)}%`:'missing'}</td><td class="${c.foundpose_ok?'ok':'bad'}">${found}</td><td>${fmt(c.foundpose_quality,3)}</td><td>${c.foundpose_inliers ?? '—'}</td><td>${fmt(c.sam3_s)} / ${fmt(c.foundpose_s)} s</td></tr>`;}).join('')}</tbody></table>`;
  renderVisualizations(record);
}
function relativeAsset(record, value) { if (!value || !record?.run_dir) return null; const root=String(record.run_dir).replace(/\\/+$/, '') + '/'; const p=String(value); return p.startsWith(root) ? p.slice(root.length) : null; }
function renderVisualizations(record) { const views=record.visualizations||{}; const labels={ '01_sam3_foundpose':'SAM3 + FoundPose availability', '02_foundpose_candidates':'Per-view FoundPose', '03_iou_selected':'IoU selected pose', '04_silhouette_refined':'Final refined pose' }; const entries=Object.entries(views).map(([key,path])=>[key,relativeAsset(record,path)]).filter(([,rel])=>rel); const tabs=$('stage-tabs'); tabs.innerHTML=entries.map(([key])=>`<button class="${selectedStage===key?'active':''}" data-stage="${esc(key)}">${esc(labels[key]||key)}</button>`).join(''); for (const b of tabs.querySelectorAll('button')) b.onclick=()=>{selectedStage=b.dataset.stage; renderVisualizations(record);}; if (!entries.length) { $('stage-image').classList.add('hidden'); $('stage-empty').classList.remove('hidden'); return; } if (!entries.some(([key])=>key===selectedStage)) selectedStage=entries[0][0]; const rel=entries.find(([key])=>key===selectedStage)[1]; const url=`/api/runs/${encodeURIComponent(activeRun || '')}/artifact?path=${encodeURIComponent(rel)}`; $('stage-image').src=url+`&t=${Date.now()}`; $('stage-image').classList.remove('hidden'); $('stage-empty').classList.add('hidden'); }
function renderHistory(rows) { $('history').innerHTML = rows?.length ? `<table><thead><tr><th>object</th><th>status</th><th>time</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${esc(x.object)}</td><td>${esc(x.status)}</td><td>${fmt(x.total_s)}s</td></tr>`).join('')}</tbody></table>` : '<span class="muted">none</span>'; }
async function loadObjects() { if (objectsLoaded) return; const data=await api('/api/objects'); $('objects').innerHTML=data.objects.map(x=>`<option value="${esc(x)}"></option>`).join(''); objectsLoaded=true; }
async function inspect() { const name=$('object').value.trim(); if (!name) return; const data=await api('/api/objects/inspect?name='+encodeURIComponent(name)); $('validation').className='small '+(data.valid?'ok':'bad'); $('validation').textContent=data.valid?'Object and required local assets are available.':(data.problems||[]).join(' ') || 'Invalid object.'; $('object-resources').innerHTML=resourceTable(data.resources); renderObjectVisuals(data); }
async function refresh() { try { const s=await api('/api/session'); const state=s.state||{}; activeRun=state.run_id || activeRun; $('session-state').textContent=state.state || 'ready'; $('session-state').className='badge '+(state.state==='complete'?(state.result?.status==='PASS'?'ok':'warn'):(state.state==='error'?'bad':'')); $('session-info').innerHTML=`${s.active_camera_count} active cameras · calibration <span class="mono">${esc(s.calibration_dir)}</span><br>${s.camera_controller.managed ? 'camera controller managed by this session' : esc(s.camera_controller.message || 'stream state unmanaged')}`; $('session-resources').innerHTML=resourceTable(s.resources); renderTimeline(state); if (state.result) renderResult(state.result); else if (s.latest) renderResult(s.latest); $('run').disabled=state.state==='running'; const history=await api('/api/history'); renderHistory(history.runs); } catch(e) { $('session-state').textContent='connection error'; $('session-state').className='badge bad'; console.error(e); } }
$('object').addEventListener('change', inspect); $('object').addEventListener('blur', inspect); $('run').onclick=async()=>{try { const data=await api('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({object:$('object').value})}); if (!data.accepted) { $('validation').className='small bad'; $('validation').textContent=data.busy?'A perception run is already active.':(data.inspection?.problems||['Cannot start run.']).join(' '); return; } activeRun=data.run_id; selectedStage=null; await refresh(); } catch(e) { $('validation').className='small bad'; $('validation').textContent=e.message; }};
$('stage-image').onclick=()=>{ $('large-image').src=$('stage-image').src; $('image-dialog').showModal(); }; $('close-dialog').onclick=()=>$('image-dialog').close();
loadObjects().then(refresh); setInterval(refresh, 1000);
</script></body></html>"""


def create_app(controller: Any) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return _HTML

    @app.get("/api/session")
    def session():
        return jsonify(controller.session())

    @app.get("/api/objects")
    def objects():
        return jsonify({"objects": controller.supported_names})

    @app.get("/api/objects/inspect")
    def inspect_object():
        return jsonify(controller.inspect_object(request.args.get("name", "")))

    @app.get("/api/objects/<object_name>/preview/mesh")
    def mesh_preview(object_name: str):
        result = controller.mesh_preview(object_name)
        if result is None:
            return jsonify({"error": "object or required assets not found"}), 404
        if result.get("error"):
            return jsonify({"error": "could not render mesh preview",
                            "detail": result["error"]}), 500
        path = Path(result["path"]).resolve()
        preview_root = (controller.args.output_root / "_resource_previews").resolve()
        try:
            path.relative_to(preview_root)
        except ValueError:
            return jsonify({"error": "preview path outside checker output"}), 403
        return send_file(path, conditional=True, max_age=0)

    @app.post("/api/runs")
    def start_run():
        payload = request.get_json(silent=True) or {}
        result = controller.start_run(str(payload.get("object", "")))
        if result.get("accepted"):
            return jsonify(result), 202
        return jsonify(result), 409 if result.get("busy") else 422

    @app.get("/api/runs/<run_id>")
    def run(run_id: str):
        result = controller.run(run_id)
        if result is None:
            return jsonify({"error": "run not found"}), 404
        return jsonify(result)

    @app.get("/api/history")
    def history():
        return jsonify({"runs": controller.history()})

    @app.get("/api/runs/<run_id>/artifact")
    def artifact(run_id: str):
        state = controller.run(run_id)
        record = state.get("result") if isinstance(state, dict) else None
        if record is None and isinstance(state, dict) and state.get("run_dir"):
            record = state
        if not isinstance(record, dict) or not record.get("run_dir"):
            return jsonify({"error": "run artifacts are not available"}), 404
        relative = request.args.get("path", "")
        root = Path(record["run_dir"]).expanduser().resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return jsonify({"error": "artifact path is outside this run"}), 403
        if not candidate.is_file():
            return jsonify({"error": "artifact not found"}), 404
        return send_file(candidate, conditional=True, max_age=0)

    return app


def run_dashboard(controller: Any, host: str, port: int) -> None:
    """Run a single-process local dashboard without Flask's reloader."""
    app = create_app(controller)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"[web] open: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
