from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

try:
    from .raw_scoring_pipeline import OUTPUT_DIR, safe_filename, score_raw_files
except ImportError:  # Allows: python raw_prediction_dashboard.py
    from raw_scoring_pipeline import OUTPUT_DIR, safe_filename, score_raw_files


APP_DIR = OUTPUT_DIR
UPLOAD_DIR = APP_DIR / "uploads"
APP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Loan Model Raw Scoring Dashboard")


async def save_upload(upload: UploadFile | None, prefix: str) -> Path | None:
    if upload is None or not upload.filename:
        return None
    path = UPLOAD_DIR / f"{prefix}_{safe_filename(upload.filename)}"
    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return path


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/score")
async def score(
    user_file: UploadFile = File(...),
    call_file: UploadFile | None = File(None),
    label_file: UploadFile | None = File(None),
    chunksize: int = Form(500_000),
    top_n: int = Form(500),
):
    try:
        top_n = max(50, min(int(top_n), 2_000))
        chunksize = max(25_000, min(int(chunksize), 1_000_000))
        user_path = await save_upload(user_file, "user")
        call_path = await save_upload(call_file, "call")
        label_path = await save_upload(label_file, "label")
        if user_path is None:
            raise ValueError("User file is required.")
        result = score_raw_files(
            user_path=user_path,
            call_path=call_path,
            label_path=label_path,
            output_dir=APP_DIR,
            chunksize=chunksize,
            top_n=top_n,
        )
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - show dashboard-friendly error message.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/download/{job_id}/{kind}")
def download(job_id: str, kind: str):
    if not job_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid job id.")
    paths = {
        "full": APP_DIR / f"{job_id}_raw_scored_predictions.csv",
        "top_t0": APP_DIR / f"{job_id}_top_t0_call_targets.csv",
        "top_t1": APP_DIR / f"{job_id}_top_t1_conversion_predictions.csv",
        "summary": APP_DIR / f"{job_id}_summary.json",
    }
    path = paths.get(kind)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Output not found.")
    return FileResponse(path, filename=path.name)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loan Propensity Command Center</title>
  <style>
    :root {
      --ink:#101820; --muted:#60717c; --paper:#f7f2e8; --card:rgba(255,255,255,.82);
      --line:rgba(16,24,32,.13); --teal:#0f766e; --cyan:#0ea5a4; --gold:#d9911b;
      --blue:#17324c; --green:#15803d; --red:#b42318; --shadow:0 28px 80px rgba(23,50,76,.16);
    }
    * { box-sizing:border-box; }
    body {
      margin:0; color:var(--ink);
      font-family:"Bahnschrift","Candara","Trebuchet MS",sans-serif;
      background:
        radial-gradient(circle at 8% 9%, rgba(15,118,110,.24), transparent 26%),
        radial-gradient(circle at 88% 12%, rgba(217,145,27,.22), transparent 28%),
        linear-gradient(135deg,#fff9ec 0%,#eef7f3 58%,#e8edf1 100%);
      min-height:100vh;
    }
    body:before {
      content:""; position:fixed; inset:0; pointer-events:none; opacity:.22;
      background-image:linear-gradient(rgba(16,24,32,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16,24,32,.06) 1px, transparent 1px);
      background-size:36px 36px;
    }
    .wrap { position:relative; width:min(1540px, calc(100% - 34px)); margin:0 auto; padding:30px 0 60px; }
    .hero,.panel,.card,.metric {
      background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow); backdrop-filter:blur(14px);
    }
    .hero { overflow:hidden; border-radius:34px; padding:clamp(28px,5vw,58px); position:relative;
      background:linear-gradient(135deg,rgba(255,255,255,.9),rgba(232,248,242,.82)); }
    .hero:after {
      content:"T0/T1"; position:absolute; right:-10px; bottom:-46px; color:rgba(15,118,110,.08);
      font:900 clamp(92px,15vw,190px)/1 Georgia,"Palatino Linotype",serif; letter-spacing:-.09em;
    }
    h1,h2,h3 { margin:0; font-family:Georgia,"Palatino Linotype",serif; letter-spacing:-.035em; }
    h1 { font-size:clamp(42px,7vw,86px); line-height:.95; max-width:1050px; }
    h2 { font-size:clamp(26px,3vw,42px); margin-bottom:15px; }
    h3 { font-size:23px; margin-bottom:8px; }
    p { color:var(--muted); line-height:1.55; }
    .eyebrow { color:var(--teal); font-weight:900; text-transform:uppercase; letter-spacing:.16em; font-size:13px; margin-bottom:12px; }
    .grid { display:grid; gap:18px; }
    .two { grid-template-columns:1fr 1fr; }
    .three { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .four { grid-template-columns:repeat(4,minmax(0,1fr)); }
    .panel { margin-top:22px; border-radius:30px; padding:24px; }
    .card { border-radius:26px; padding:20px; }
    .metric { min-height:136px; border-radius:26px; padding:20px; }
    .metric p { margin:0; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em; font-weight:900; }
    .metric strong { display:block; margin:12px 0 8px; font-size:clamp(30px,3vw,48px); line-height:1; letter-spacing:-.045em; }
    .metric span { color:var(--muted); }
    .upload {
      border:1px dashed rgba(15,118,110,.42); border-radius:24px; padding:18px;
      background:linear-gradient(135deg,rgba(15,118,110,.07),rgba(217,145,27,.06));
    }
    label { display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.09em; font-weight:900; margin-bottom:6px; }
    input,select,button {
      min-height:44px; width:100%; border:1px solid var(--line); border-radius:15px; padding:10px 12px;
      background:rgba(255,255,255,.88); color:var(--ink); font:inherit; outline-color:var(--teal);
    }
    button { cursor:pointer; border-color:transparent; background:var(--ink); color:white; font-weight:900; }
    button.secondary { background:rgba(16,24,32,.08); color:var(--ink); border-color:var(--line); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .pill { display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:7px 12px; font-weight:900; font-size:12px;
      background:rgba(23,50,76,.08); color:var(--blue); border:1px solid rgba(23,50,76,.10); }
    .pill.good { color:#0f5132; background:#dff7e8; border-color:#b9eccd; }
    .pill.warn { color:#7a4b00; background:#fff3cd; border-color:#ffe69c; }
    .status {
      border-radius:22px; padding:18px; background:#101820; color:white; min-height:100%;
      display:flex; flex-direction:column; justify-content:space-between;
    }
    .status p { color:rgba(255,255,255,.78); }
    .progress { height:12px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,.16); }
    .progress span { display:block; height:100%; width:0%; border-radius:999px; background:linear-gradient(90deg,var(--teal),var(--gold)); transition:width .35s ease; }
    .section-head { display:flex; justify-content:space-between; gap:18px; align-items:start; margin-bottom:18px; }
    .section-head p { margin:6px 0 0; max-width:860px; }
    .workflow-card {
      border-radius:28px; padding:22px; position:relative; overflow:hidden;
      background:linear-gradient(135deg,rgba(255,255,255,.78),rgba(232,248,242,.70));
      border:1px solid rgba(16,24,32,.11);
    }
    .workflow-card strong {
      display:inline-flex; width:52px; height:52px; align-items:center; justify-content:center; border-radius:18px;
      background:#101820; color:white; font-size:19px; margin-bottom:14px;
    }
    .workflow-card.t1 strong { background:linear-gradient(135deg,var(--teal),var(--gold)); }
    .workflow-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:16px; }
    .workflow-stats div { padding:12px; border-radius:16px; background:rgba(255,255,255,.62); border:1px solid rgba(16,24,32,.08); }
    .workflow-stats b { display:block; font-size:22px; letter-spacing:-.03em; }
    .simulator {
      background:linear-gradient(135deg,rgba(16,24,32,.94),rgba(23,50,76,.88));
      color:white;
    }
    .simulator h2,.simulator h3 { color:white; }
    .simulator p,.simulator label { color:rgba(255,255,255,.76); }
    .sim-card { padding:16px; border-radius:20px; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.14); }
    .sim-card b { display:block; margin-top:7px; font-size:clamp(26px,3vw,42px); line-height:1; letter-spacing:-.04em; color:white; }
    .sim-card span { color:rgba(255,255,255,.72); }
    .simulator .graph-card { background:rgba(255,255,255,.08); border-color:rgba(255,255,255,.14); }
    .simulator canvas { height:230px; }
    .graph-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .graph-card {
      min-height:365px; border-radius:24px; padding:18px; border:1px solid rgba(16,24,32,.10);
      background:rgba(255,255,255,.62); position:relative; overflow:hidden;
    }
    .graph-card p { margin:0 0 12px; font-size:13px; }
    canvas { width:100%; height:260px; display:block; cursor:crosshair; }
    .tip {
      position:fixed; z-index:50; pointer-events:none; transform:translate(14px,-18px); opacity:0;
      padding:9px 11px; border-radius:12px; background:rgba(16,24,32,.92); color:white;
      font-size:12px; font-weight:900; box-shadow:0 18px 42px rgba(16,24,32,.25); transition:opacity .1s ease;
    }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:12px 10px; border-bottom:1px solid rgba(16,24,32,.10); text-align:left; vertical-align:middle; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    td small { display:block; color:var(--muted); font-size:11px; margin-top:3px; }
    .table-wrap { overflow-x:auto; border-radius:22px; border:1px solid rgba(16,24,32,.10); background:rgba(255,255,255,.58); }
    .tabs { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
    .tabs button { width:auto; min-height:38px; padding:8px 14px; }
    .tabs button.active { background:var(--teal); }
    .hidden { display:none !important; }
    .download-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
    .download-grid a {
      display:block; text-decoration:none; color:var(--ink); font-weight:900; padding:16px; border-radius:20px;
      background:rgba(255,255,255,.68); border:1px solid rgba(16,24,32,.11);
    }
    .source-note { font-size:12px; color:var(--muted); margin-top:10px; }
    .empty { text-align:center; padding:40px 16px; color:var(--muted); }
    @media (max-width:1100px) { .two,.three,.four,.graph-grid,.download-grid{grid-template-columns:1fr 1fr} }
    @media (max-width:720px) { .wrap{width:min(100% - 20px,1540px);padding-top:10px}.hero,.panel{border-radius:22px;padding:18px}.two,.three,.four,.graph-grid,.download-grid{grid-template-columns:1fr} }
  </style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <div class="eyebrow">Production-style raw scoring dashboard</div>
    <h1>Upload raw user and call data. Get T0 and T1 predictions.</h1>
    <p>
      T0 ranks who should be called first using user data. T1 ranks who is most likely to take the loan after call signals are available.
      Labels are optional and only used for validation, never for prediction. Charts and KPI cards are calculated on the full scored population;
      the top-user table is only a preview.
    </p>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <span class="pill good">Final mixed hybrid models</span>
      <span class="pill good">No DIY/SFDC needed for scoring</span>
      <span class="pill warn">Large call files can take a few minutes</span>
    </div>
  </section>

  <section class="panel grid two">
    <div>
      <h2>Raw Input</h2>
      <form id="scoreForm" class="upload">
        <div class="grid two">
          <div>
            <label>User data CSV/XLSX</label>
            <input id="userFile" name="user_file" type="file" accept=".csv,.xlsx,.xls" required>
          </div>
          <div>
            <label>Call data CSV/XLSX</label>
            <input id="callFile" name="call_file" type="file" accept=".csv,.xlsx,.xls">
          </div>
          <div>
            <label>Optional label CSV/XLSX</label>
            <input id="labelFile" name="label_file" type="file" accept=".csv,.xlsx,.xls">
          </div>
          <div>
            <label>Top rows returned</label>
            <select id="topN" name="top_n">
              <option>100</option><option selected>500</option><option>1000</option><option>2000</option>
            </select>
          </div>
          <div>
            <label>Call chunk size</label>
            <select id="chunksize" name="chunksize">
              <option value="100000">100,000 safer</option>
              <option value="250000">250,000</option>
              <option value="500000" selected>500,000 faster</option>
              <option value="1000000">1,000,000 fastest</option>
            </select>
          </div>
          <div style="display:flex;align-items:end">
            <button id="scoreButton" type="submit">Score Raw Dataset</button>
          </div>
        </div>
        <p>
          The app saves full ranked predictions after scoring. If labels are uploaded, it also computes AUC, precision, lift, and conversion deciles.
        </p>
      </form>
    </div>
    <div class="status">
      <div>
        <h2 style="color:white">Run Status</h2>
        <p id="statusText">Waiting for raw user and call data.</p>
      </div>
      <div>
        <div class="progress"><span id="progressBar"></span></div>
        <p id="statusSub">No run started.</p>
      </div>
    </div>
  </section>

  <section id="results" class="hidden">
    <section class="grid four" id="cards"></section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Operating Workflow</h2>
          <p>
            This mirrors the real business process: T0 builds the call queue before outreach, then T1 prioritizes post-call loan follow-up using call behavior.
          </p>
        </div>
        <span class="pill good">Full-dataset workflow</span>
      </div>
      <div class="grid two">
        <article class="workflow-card">
          <strong>T0</strong>
          <h3>Call Planning Queue</h3>
          <p>Use this before calls happen. It ranks users by who should be called first using user snapshot features only.</p>
          <div class="workflow-stats" id="t0WorkflowStats"></div>
        </article>
        <article class="workflow-card t1">
          <strong>T1</strong>
          <h3>Conversion Follow-Up</h3>
          <p>Use this after calls are available. It ranks users by loan conversion likelihood using user plus call features.</p>
          <div class="workflow-stats" id="t1WorkflowStats"></div>
        </article>
      </div>
    </section>

    <section class="panel simulator">
      <div class="section-head">
        <div>
          <h2>Decision Simulator</h2>
          <p>
            Choose a call/follow-up capacity and estimate the opportunity from the full T1-ranked population.
            If labels are uploaded, this also shows actual conversions captured at that cutoff.
          </p>
        </div>
        <span class="pill warn">Uses all scored users</span>
      </div>
      <div class="grid two">
        <div>
          <label for="capacitySelect">Daily capacity / top K users</label>
          <select id="capacitySelect"></select>
        </div>
        <div class="grid three" id="simCards"></div>
      </div>
      <div class="graph-card" style="margin-top:18px">
        <h3>Modeled Opportunity by Capacity</h3>
        <p>Columns show the sum of T1 scores inside each top-K cutoff, using the whole scored dataset.</p>
        <canvas id="decisionChart"></canvas>
      </div>
    </section>

    <section class="panel">
      <h2>Executive View: All Scored Users</h2>
      <p>
        These charts are not based on the top-user preview. They use every user scored in the uploaded dataset.
      </p>
      <div class="grid graph-grid">
        <div class="graph-card">
          <h3>3D T1 Score Deciles</h3>
          <p>Calculated across all scored users. Top decile should have the strongest predicted conversion quality.</p>
          <canvas id="decileChart"></canvas>
        </div>
        <div class="graph-card">
          <h3>3D Priority Band Split</h3>
          <p>Full-dataset split of very high, high, medium, and low T1 bands.</p>
          <canvas id="priorityPie"></canvas>
        </div>
        <div class="graph-card">
          <h3>Call Join Health</h3>
          <p>Full-dataset check of whether call rows joined correctly to user rows by user_id.</p>
          <canvas id="joinChart"></canvas>
        </div>
        <div class="graph-card">
          <h3>Month-Trend Similarity</h3>
          <p>Calculated from the full uploaded batch, showing which training months it resembles.</p>
          <canvas id="monthChart"></canvas>
        </div>
        <div class="graph-card">
          <h3>T1 Score Distribution</h3>
          <p>Full scored-population histogram. Useful for spotting drift or over-concentration.</p>
          <canvas id="scoreHistogram"></canvas>
        </div>
        <div class="graph-card">
          <h3>Top Campaigns / States</h3>
          <p>Full-dataset segments ranked by mean T1 score.</p>
          <select id="groupMode" style="margin-bottom:12px">
            <option value="campaign_id">Campaign</option>
            <option value="state">State</option>
          </select>
          <canvas id="groupChart"></canvas>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Download Outputs</h2>
      <div class="grid download-grid" id="downloads"></div>
    </section>

    <section class="panel">
      <h2>Top Users</h2>
      <p>
        This table is intentionally a ranked preview. The graphs above and KPI cards use all users, and the full ranked CSV is available in downloads.
      </p>
      <div class="tabs">
        <button id="tabT1" class="active" type="button">T1 Conversion Ranking</button>
        <button id="tabT0" type="button">T0 Call Targeting</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>User</th><th>T1 Score</th><th>T0 Score</th><th>Band</th><th>Stage</th>
              <th>Calls</th><th>Answered</th><th>Avg Duration</th><th>Campaign</th><th>State</th>
            </tr>
          </thead>
          <tbody id="userBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Model / Data Checks</h2>
      <div class="grid three" id="checks"></div>
    </section>
  </section>
</main>
<div id="tip" class="tip"></div>

<script>
let lastResult = null;
let activeTable = "t1";
const hits = {};

function fmtNum(value) { return Number(value || 0).toLocaleString("en-IN"); }
function fmtPct(value, digits=1) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "-"; }
function fmtScore(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "-"; }
function esc(value) { return String(value ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[c])); }

function setStatus(text, sub, pct) {
  document.querySelector("#statusText").textContent = text;
  document.querySelector("#statusSub").textContent = sub;
  document.querySelector("#progressBar").style.width = `${pct}%`;
}

function card(label, value, note) {
  return `<article class="metric"><p>${esc(label)}</p><strong>${esc(value)}</strong><span>${esc(note)}</span></article>`;
}

function setupCanvas(id) {
  const canvas = document.getElementById(id);
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(220, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio,0,0,ratio,0,0);
  if (!canvas.dataset.hoverReady) {
    canvas.dataset.hoverReady = "1";
    canvas.addEventListener("mousemove", event => {
      const r = canvas.getBoundingClientRect();
      const point = { x:event.clientX - r.left, y:event.clientY - r.top };
      const hit = findHit(id, point);
      const tip = document.querySelector("#tip");
      if (!hit) { tip.style.opacity = 0; return; }
      tip.innerHTML = hit.tip;
      tip.style.left = `${event.clientX}px`;
      tip.style.top = `${event.clientY}px`;
      tip.style.opacity = 1;
    });
    canvas.addEventListener("mouseleave", () => document.querySelector("#tip").style.opacity = 0);
  }
  return {canvas, ctx, width:canvas.width / ratio, height:canvas.height / ratio};
}

function clear(ctx, w, h) {
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "rgba(255,255,255,.34)";
  ctx.fillRect(0,0,w,h);
}

function shade(hex, amount) {
  const raw = String(hex || "#0f766e").replace("#","");
  const r = parseInt(raw.slice(0,2),16), g = parseInt(raw.slice(2,4),16), b = parseInt(raw.slice(4,6),16);
  const clamp = v => Math.max(0, Math.min(255, Math.round(v)));
  return `rgb(${clamp(r+amount)},${clamp(g+amount)},${clamp(b+amount)})`;
}

function findHit(id, point) {
  const list = hits[id] || [];
  for (let i=list.length-1; i>=0; i--) {
    const h = list[i];
    if (h.type === "rect" && point.x >= h.x && point.x <= h.x+h.w && point.y >= h.y && point.y <= h.y+h.h) return h;
    if (h.type === "pie") {
      const dx = point.x - h.cx, dy = (point.y - h.cy) / h.yScale;
      let angle = Math.atan2(dy, dx), start = h.start, end = h.end;
      while (end < start) end += Math.PI * 2;
      while (angle < start) angle += Math.PI * 2;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist <= h.r && angle >= start && angle <= end) return h;
    }
  }
  return null;
}

function drawNoData(id, message) {
  const {ctx,width,height} = setupCanvas(id);
  clear(ctx,width,height);
  hits[id] = [];
  ctx.fillStyle = "#60717c";
  ctx.font = "800 15px Bahnschrift, Candara, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, width/2, height/2);
}

function draw3DColumns(id, bars, options={}) {
  const {ctx,width,height} = setupCanvas(id);
  clear(ctx,width,height);
  bars = (bars || []).filter(b => Number(b.value) >= 0);
  if (!bars.length || bars.every(b => Number(b.value) === 0)) return drawNoData(id, "No chart data yet");
  const pad = {l:54,r:28,t:24,b:62};
  const x=pad.l,y=pad.t,w=width-pad.l-pad.r,h=height-pad.t-pad.b;
  const maxVal = Math.max(...bars.map(b => Number(b.value) || 0), 0.01);
  ctx.strokeStyle = "rgba(16,24,32,.16)";
  ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x,y+h); ctx.lineTo(x+w,y+h); ctx.stroke();
  ctx.strokeStyle = "rgba(16,24,32,.07)";
  for (let i=1; i<=4; i++) { const gy = y + h*i/5; ctx.beginPath(); ctx.moveTo(x,gy); ctx.lineTo(x+w,gy); ctx.stroke(); }
  const gap = Math.max(8, Math.min(14, w/(bars.length*5)));
  const slot = (w - gap*(bars.length-1))/bars.length;
  const depth = Math.min(18, Math.max(9, slot*.2));
  const bw = Math.max(8, slot-depth);
  const base = options.color || "#0f766e", end = options.color2 || "#d9911b";
  const localHits = [];
  bars.forEach((bar, idx) => {
    const val = Number(bar.value) || 0, bh = val/maxVal*h, bx = x + idx*(slot+gap), by = y+h-bh, lift=depth*.55;
    ctx.fillStyle = "rgba(16,24,32,.09)";
    ctx.beginPath(); ctx.ellipse(bx+bw/2+depth/2, y+h+7, Math.max(10,bw*.55), 6, 0, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.moveTo(bx+bw,by); ctx.lineTo(bx+bw+depth,by-lift); ctx.lineTo(bx+bw+depth,y+h-lift); ctx.lineTo(bx+bw,y+h); ctx.closePath(); ctx.fillStyle=shade(base,-48); ctx.fill();
    ctx.beginPath(); ctx.moveTo(bx,by); ctx.lineTo(bx+depth,by-lift); ctx.lineTo(bx+bw+depth,by-lift); ctx.lineTo(bx+bw,by); ctx.closePath(); ctx.fillStyle=shade(base,36); ctx.fill();
    const grad = ctx.createLinearGradient(0, by, 0, y+h); grad.addColorStop(0, base); grad.addColorStop(1, end);
    ctx.fillStyle = grad; ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = "rgba(255,255,255,.42)"; ctx.beginPath(); ctx.moveTo(bx+3, by+2); ctx.lineTo(bx+3, y+h-2); ctx.stroke();
    ctx.save(); ctx.translate(bx+bw/2, y+h+14); ctx.rotate(-Math.PI/5); ctx.fillStyle="#60717c"; ctx.font="10px Bahnschrift, Candara, sans-serif"; ctx.textAlign="center"; ctx.fillText(String(bar.label).slice(0,14), 0, 0); ctx.restore();
    localHits.push({type:"rect", x:bx, y:Math.min(by-lift, y+h-10), w:bw+depth, h:Math.max(14,bh+lift+10), tip:`<b>${esc(bar.label)}</b><br>${esc(options.valueLabel || "Value")}: ${esc(options.percent ? fmtPct(val) : fmtNum(val))}`});
  });
  hits[id] = localHits;
}

function draw3DPie(id, segments) {
  const {ctx,width,height} = setupCanvas(id);
  clear(ctx,width,height);
  segments = (segments || []).filter(s => Number(s.value) > 0);
  const total = segments.reduce((s,x)=>s+Number(x.value),0);
  if (!total) return drawNoData(id, "No split available");
  const cx = width*.36, cy = height*.42, r = Math.min(width,height)*.30, yScale=.62, depth=24;
  function slice(centerY, start, end, color) {
    ctx.save(); ctx.translate(cx, centerY); ctx.scale(1,yScale); ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0,0,r,start,end); ctx.closePath(); ctx.fillStyle=color; ctx.fill(); ctx.restore();
  }
  let start = -Math.PI/2;
  const arcs = segments.map(seg => { const end = start + Number(seg.value)/total*Math.PI*2; const out={...seg,start,end}; start=end; return out; });
  for (let layer=depth; layer>=2; layer-=2) arcs.forEach(seg => slice(cy+layer, seg.start, seg.end, shade(seg.color,-58)));
  const localHits = [];
  arcs.forEach(seg => {
    slice(cy, seg.start, seg.end, seg.color);
    ctx.save(); ctx.translate(cx,cy); ctx.scale(1,yScale); ctx.strokeStyle="rgba(255,255,255,.58)"; ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0,0,r,seg.start,seg.end); ctx.closePath(); ctx.stroke(); ctx.restore();
    localHits.push({type:"pie", cx, cy, r, yScale, start:seg.start, end:seg.end, tip:`<b>${esc(seg.label)}</b><br>${fmtNum(seg.value)} users<br>${fmtPct(seg.value/total)} of batch`});
  });
  let ly = height*.30;
  arcs.forEach(seg => {
    ctx.fillStyle = seg.color; ctx.fillRect(width*.67, ly-9, 15, 15);
    ctx.fillStyle = "#101820"; ctx.font = "800 13px Bahnschrift, Candara, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`${seg.label}: ${fmtNum(seg.value)}`, width*.67 + 24, ly+3);
    ly += 28;
  });
  hits[id] = localHits;
}

function renderCards(data) {
  const users = data.users_scored || 0;
  const join = data.call_join_info || {};
  const t1 = data.score_summary?.t1 || {};
  const labels = data.label_info || {};
  const metrics = data.metrics?.t1;
  document.querySelector("#cards").innerHTML = [
    card("Users Scored", fmtNum(users), "All cards and graphs use this full population"),
    card("Call Join Rate", fmtPct(join.join_rate || 0), `${fmtNum(join.users_with_any_joined_call)} users had calls across all users`),
    card("Very High T1", fmtNum(t1.count_ge_0_90 || 0), "Full-dataset users with T1 score >= 0.90"),
    card(labels.labels_available ? "T1 AUC" : "Validation", labels.labels_available && metrics?.auc ? Number(metrics.auc).toFixed(4) : "No labels", labels.labels_available ? "Computed from uploaded labels" : "Prediction-only mode")
  ].join("");
}

function renderDownloads(data) {
  const id = data.job_id;
  document.querySelector("#downloads").innerHTML = [
    ["Full ranked CSV", "full"],
    ["Top T0 call targets", "top_t0"],
    ["Top T1 conversions", "top_t1"],
    ["Run summary JSON", "summary"],
  ].map(([label, kind]) => `<a href="/download/${id}/${kind}">${esc(label)}<small style="display:block;color:#60717c;margin-top:4px">Download ${esc(kind)}</small></a>`).join("");
}

function statBox(label, value, note) {
  return `<div><span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(note || "")}</small></div>`;
}

function nearestCurveRow(curve, wanted) {
  if (!curve.length) return null;
  return curve.reduce((best, row) => Math.abs(row.k - wanted) < Math.abs(best.k - wanted) ? row : best, curve[0]);
}

function renderWorkflow(data) {
  const t0 = data.score_summary?.t0 || {};
  const t1 = data.score_summary?.t1 || {};
  const join = data.call_join_info || {};
  const t0Curve = data.decision_curve?.t0 || [];
  const t1Curve = data.decision_curve?.t1 || [];
  const t0Top100 = nearestCurveRow(t0Curve, 100) || {};
  const t1Top100 = nearestCurveRow(t1Curve, 100) || {};
  document.querySelector("#t0WorkflowStats").innerHTML = [
    statBox("Top 100 Avg", fmtScore(t0Top100.avg_score), "call priority"),
    statBox(">= 0.75", fmtNum(t0.count_ge_0_75 || 0), "call targets"),
    statBox("Users", fmtNum(data.users_scored || 0), "pre-call pool"),
  ].join("");
  document.querySelector("#t1WorkflowStats").innerHTML = [
    statBox("Top 100 Expected", Number(t1Top100.expected_conversions || 0).toFixed(1), "modeled conversions"),
    statBox(">= 0.90", fmtNum(t1.count_ge_0_90 || 0), "very high"),
    statBox("With Calls", fmtNum(join.users_with_any_joined_call || 0), "post-call pool"),
  ].join("");
}

function renderSimulator(data) {
  const curve = data.decision_curve?.t1 || [];
  const select = document.querySelector("#capacitySelect");
  if (!curve.length) {
    select.innerHTML = "";
    document.querySelector("#simCards").innerHTML = "";
    drawNoData("decisionChart", "No simulator data yet");
    return;
  }
  const previous = Number(select.value || 0);
  select.innerHTML = curve.map(row => `<option value="${row.k}">Top ${fmtNum(row.k)} users</option>`).join("");
  const preferred = previous || (curve.find(row => row.k >= 100)?.k || curve[0].k);
  select.value = String(nearestCurveRow(curve, preferred).k);
  updateSimulatorCards(data);
  draw3DColumns(
    "decisionChart",
    curve.map(row => ({label:`Top ${row.k}`, value:row.expected_conversions})),
    {color:"#0f766e", color2:"#d9911b", valueLabel:"Modeled conversions"}
  );
}

function updateSimulatorCards(data) {
  const curve = data.decision_curve?.t1 || [];
  const selected = Number(document.querySelector("#capacitySelect").value || 0);
  const row = curve.find(item => item.k === selected) || curve[0] || {};
  const labelInfo = data.label_info || {};
  const actual = labelInfo.labels_available && Number.isFinite(Number(row.actual_conversions))
    ? `<div class="sim-card"><span>Actual conversions</span><b>${fmtNum(row.actual_conversions)}</b><span>${fmtPct(row.actual_precision || 0)} precision</span></div>`
    : `<div class="sim-card"><span>Validation</span><b>Pending</b><span>Upload labels later</span></div>`;
  document.querySelector("#simCards").innerHTML = [
    `<div class="sim-card"><span>Selected users</span><b>${fmtNum(row.k || 0)}</b><span>${fmtPct(row.coverage || 0)} of full batch</span></div>`,
    `<div class="sim-card"><span>Modeled conversions</span><b>${Number(row.expected_conversions || 0).toFixed(1)}</b><span>sum of T1 scores</span></div>`,
    `<div class="sim-card"><span>Minimum T1 score</span><b>${fmtScore(row.min_score || 0)}</b><span>cutoff threshold</span></div>`,
    actual,
  ].join("");
}

function renderCharts(data) {
  const t1Deciles = (data.deciles?.t1 || []).map(d => ({
    label:`D${d.model_decile}`,
    value:data.label_info?.labels_available ? (d.positive_rate || 0) : (d.mean_score || 0)
  }));
  draw3DColumns("decileChart", t1Deciles, {percent:true, color:"#0f766e", color2:"#99f6e4", valueLabel:data.label_info?.labels_available ? "Conversion rate" : "Mean score"});
  const bands = data.priority_bands || {};
  draw3DPie("priorityPie", [
    {label:"Very High", value:bands["Very High"] || 0, color:"#0f766e"},
    {label:"High", value:bands["High"] || 0, color:"#d9911b"},
    {label:"Medium", value:bands["Medium"] || 0, color:"#17324c"},
    {label:"Low", value:bands["Low"] || 0, color:"#c2573a"},
  ]);
  const join = data.call_join_info || {};
  draw3DColumns("joinChart", [
    {label:"Users", value:data.users_scored || 0},
    {label:"With Calls", value:join.users_with_any_joined_call || 0},
    {label:"Answered", value:join.users_with_answered_call || 0},
    {label:"Avg >=10s", value:join.users_with_avg_duration_10s || 0},
  ], {color:"#17324c", color2:"#0f766e", valueLabel:"Users"});
  const sim = data.month_similarity?.t1 || {};
  draw3DColumns("monthChart", Object.entries(sim).map(([label,value]) => ({label, value})), {percent:true, color:"#d9911b", color2:"#0f766e", valueLabel:"Similarity weight"});
  draw3DColumns(
    "scoreHistogram",
    data.score_histogram?.t1 || [],
    {color:"#17324c", color2:"#0f766e", valueLabel:"Users"}
  );
  const mode = document.querySelector("#groupMode")?.value || "campaign_id";
  const groups = data.top_groups?.[mode] || [];
  draw3DColumns(
    "groupChart",
    groups.map(row => ({label:row.label, value:row.mean_t1_score || 0})),
    {percent:true, color:"#d9911b", color2:"#0f766e", valueLabel:"Mean T1 score"}
  );
}

function stageLabel(row) {
  return row.recommended_stage || (Number(row.total_calls) > 0 ? "T1 follow-up" : "T0 call queue");
}

function renderTable() {
  if (!lastResult) return;
  const rows = activeTable === "t1" ? lastResult.top_t1 : lastResult.top_t0;
  const rankCol = activeTable === "t1" ? "t1_conversion_rank" : "t0_call_priority_rank";
  document.querySelector("#userBody").innerHTML = (rows || []).slice(0, 100).map(row => `
    <tr>
      <td>${fmtNum(row[rankCol])}</td>
      <td><b>${esc(row.uid)}</b><small>${esc(row.user_id || "")}</small></td>
      <td>${fmtScore(row.t1_loan_conversion_score)}</td>
      <td>${fmtScore(row.t0_call_targeting_score)}</td>
      <td><span class="pill ${row.priority_band === "Very High" ? "good" : ""}">${esc(row.priority_band)}</span></td>
      <td>${esc(stageLabel(row))}</td>
      <td>${fmtNum(row.total_calls)}</td>
      <td>${fmtNum(row.answered_calls)}</td>
      <td>${Number.isFinite(Number(row.avg_call_duration)) ? Number(row.avg_call_duration).toFixed(1) + "s" : "-"}</td>
      <td>${esc(row.campaign_id || "")}</td>
      <td>${esc(row.state || "")}</td>
    </tr>
  `).join("") || `<tr><td colspan="11" class="empty">No users returned.</td></tr>`;
}

function renderChecks(data) {
  const user = data.user_info || {}, call = data.call_join_info || {}, label = data.label_info || {};
  const checks = [
    ["User cleaning", `${fmtNum(user.feature_rows_after_uid_dedupe || 0)} final users`, `${fmtNum(user.duplicate_uid_rows || 0)} duplicate uid rows removed`],
    ["Call merge", `${fmtPct(call.join_rate || 0)} joined`, `${fmtNum(call.raw_call_rows_processed || 0)} call rows processed`],
    ["Graph scope", "All scored users", "Top-user table does not control the charts"],
    ["Labels", label.labels_available ? `${fmtNum(label.matched_label_users)} matched` : "Not used", label.labels_available ? `${fmtNum(label.actual_conversions)} positives` : "Prediction mode is clean"],
  ];
  document.querySelector("#checks").innerHTML = checks.map(([a,b,c]) => `<div class="card"><h3>${esc(a)}</h3><p><b>${esc(b)}</b><br>${esc(c)}</p></div>`).join("");
}

function renderAll(data) {
  lastResult = data;
  document.querySelector("#results").classList.remove("hidden");
  renderCards(data);
  renderDownloads(data);
  renderWorkflow(data);
  renderSimulator(data);
  renderCharts(data);
  renderChecks(data);
  renderTable();
}

document.querySelector("#scoreForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  const button = document.querySelector("#scoreButton");
  button.disabled = true;
  setStatus("Scoring raw files...", "Uploading files and building model features. Keep this tab open.", 35);
  try {
    const res = await fetch("/score", {method:"POST", body:data});
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail:"Unknown scoring error"}));
      throw new Error(err.detail || "Scoring failed");
    }
    setStatus("Rendering dashboard...", "Predictions are ready. Drawing charts and tables.", 82);
    const json = await res.json();
    renderAll(json);
    setStatus("Scoring complete.", `Job ${json.job_id}: ${fmtNum(json.users_scored)} users scored.`, 100);
  } catch (err) {
    setStatus("Scoring failed.", err.message, 100);
  } finally {
    button.disabled = false;
  }
});

document.querySelector("#tabT1").addEventListener("click", () => {
  activeTable = "t1";
  document.querySelector("#tabT1").classList.add("active");
  document.querySelector("#tabT0").classList.remove("active");
  renderTable();
});
document.querySelector("#tabT0").addEventListener("click", () => {
  activeTable = "t0";
  document.querySelector("#tabT0").classList.add("active");
  document.querySelector("#tabT1").classList.remove("active");
  renderTable();
});
document.querySelector("#capacitySelect").addEventListener("change", () => { if (lastResult) updateSimulatorCards(lastResult); });
document.querySelector("#groupMode").addEventListener("change", () => { if (lastResult) renderCharts(lastResult); });
window.addEventListener("resize", () => {
  if (lastResult) {
    clearTimeout(window.__resizeTimer);
    window.__resizeTimer = setTimeout(() => {
      renderSimulator(lastResult);
      renderCharts(lastResult);
    }, 150);
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("raw_prediction_dashboard:app", host="127.0.0.1", port=8055, reload=False)
