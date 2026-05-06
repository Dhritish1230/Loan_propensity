import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "multi_month_training" / "test_outputs"
REPORT_PATH = OUT_DIR / "april_labeled_evaluation_report.json"
TOP100_PATH = OUT_DIR / "april_t1_top100_with_conversions.csv"
DASHBOARD_PATH = OUT_DIR / "april_model_validation_dashboard_live_upload.html"


def card(label: str, value: str, note: str) -> str:
    return f"""
    <article class="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{note}</span>
    </article>
    """


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    top100 = pd.read_csv(TOP100_PATH).head(100).copy()
    top100["rank"] = range(1, len(top100) + 1)
    keep_cols = [
        "rank",
        "uid",
        "user_id",
        "t1_loan_conversion_score",
        "converted_full",
        "converted_from_diy",
        "converted_from_sfdc",
        "total_calls",
        "answered_calls",
        "answered_rate",
        "avg_call_duration",
        "campaign_id",
        "state",
    ]
    top100 = top100[[col for col in keep_cols if col in top100.columns]]
    t1 = report["metrics"]["t1_vs_conversion"]
    default_payload = {
        "defaultRows": top100.to_dict(orient="records"),
        "defaultSummary": {
            "rows": t1["rows"],
            "positives": t1["positives"],
            "baseline": t1["baseline_rate"],
            "auc": t1["auc"],
            "p100": t1["precision_at_100"],
            "top100Conversions": t1["top_100_conversions"],
            "topDecileLift": t1["top_decile_lift"],
        },
    }
    data_json = json.dumps(default_payload, ensure_ascii=False)
    cards = "\n".join(
        [
            card("Default Dataset", "April", "Loaded from saved validation outputs"),
            card("April Users", f"{t1['rows']:,.0f}", "Use upload controls to replace"),
            card("T1 AUC", f"{t1['auc']:.4f}", "Recalculates when labels exist"),
            card("Precision@100", f"{t1['precision_at_100'] * 100:.1f}%", "53 / 100 converted"),
        ]
    )

    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Upload Loan Model Dashboard</title>
  <style>
    :root {
      --ink:#142027; --muted:#65737b; --line:rgba(20,32,39,.13);
      --paper:#fff8ea; --card:rgba(255,255,255,.82); --teal:#0f766e;
      --gold:#d9911b; --navy:#17324c; --green:#15803d; --red:#b42318;
    }
    * { box-sizing:border-box; }
    body {
      margin:0; color:var(--ink);
      font-family:"Bahnschrift","Candara","Trebuchet MS",sans-serif;
      background:
        radial-gradient(circle at 12% 7%, rgba(15,118,110,.24), transparent 28%),
        radial-gradient(circle at 90% 4%, rgba(217,145,27,.22), transparent 30%),
        linear-gradient(145deg,#fff8ea 0%,#f1eee5 48%,#e6f5f0 100%);
      min-height:100vh;
    }
    body:before {
      content:""; position:fixed; inset:0; pointer-events:none; opacity:.24;
      background-image:linear-gradient(rgba(20,32,39,.05) 1px,transparent 1px),
        linear-gradient(90deg,rgba(20,32,39,.05) 1px,transparent 1px);
      background-size:38px 38px;
    }
    .wrap { width:min(1480px,calc(100% - 34px)); margin:0 auto; padding:30px 0 60px; }
    .hero,.panel,.metric-card {
      border:1px solid var(--line); background:var(--card);
      box-shadow:0 24px 70px rgba(23,50,76,.13); backdrop-filter:blur(12px);
    }
    .hero { position:relative; overflow:hidden; border-radius:34px; padding:clamp(28px,5vw,58px);
      background:linear-gradient(135deg,rgba(255,255,255,.88),rgba(235,249,244,.78)); animation:rise .65s ease both; }
    .hero:after { content:"LIVE"; position:absolute; right:-18px; bottom:-48px;
      font-family:Georgia,"Palatino Linotype",serif; font-size:clamp(100px,18vw,220px);
      font-weight:900; color:rgba(15,118,110,.08); letter-spacing:-.08em; }
    h1,h2,h3 { font-family:Georgia,"Palatino Linotype",serif; letter-spacing:-.035em; }
    h1 { font-size:clamp(42px,7vw,88px); line-height:.95; margin:0 0 18px; max-width:980px; }
    h2 { font-size:clamp(26px,3vw,42px); margin:0 0 16px; }
    h3 { font-size:23px; margin:0 0 10px; }
    .eyebrow { color:var(--teal); font-weight:900; text-transform:uppercase; letter-spacing:.16em; font-size:13px; }
    .hero p,.footer { color:var(--muted); line-height:1.55; }
    .hero p { max-width:900px; font-size:19px; }
    .grid { display:grid; gap:18px; }
    .metrics { grid-template-columns:repeat(4,minmax(0,1fr)); margin:20px 0; }
    .metric-card { min-height:138px; padding:20px; border-radius:26px; animation:rise .78s ease both; }
    .metric-card p { margin:0; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.09em; font-weight:900; }
    .metric-card strong { display:block; margin:13px 0 8px; font-size:clamp(30px,3vw,46px); line-height:1; letter-spacing:-.04em; }
    .metric-card span { color:var(--muted); }
    .panel { margin-top:22px; padding:24px; border-radius:30px; animation:rise .82s ease both; }
    .two { grid-template-columns:1.1fr .9fr; }
    .three { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .upload-zone {
      border:1px dashed rgba(15,118,110,.42); border-radius:24px; padding:18px;
      background:rgba(15,118,110,.06);
    }
    .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:14px 0 20px; }
    label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:900; }
    select,input,textarea,button {
      min-height:42px; border:1px solid var(--line); border-radius:14px; padding:10px 12px;
      background:rgba(255,255,255,.88); color:var(--ink); font:inherit; outline-color:var(--teal);
    }
    input[type="file"] { max-width:360px; }
    input[type="search"] { min-width:min(420px,100%); }
    textarea { width:100%; min-height:90px; resize:vertical; }
    button { cursor:pointer; background:var(--ink); color:#fff; font-weight:900; border-color:transparent; }
    button.secondary { background:rgba(20,32,39,.08); color:var(--ink); border-color:var(--line); }
    .pill { display:inline-flex; align-items:center; border-radius:999px; padding:7px 12px;
      font-weight:900; font-size:12px; background:rgba(23,50,76,.08); color:var(--navy); border:1px solid rgba(23,50,76,.10); }
    .pill.good { color:#0f5132; background:#dff7e8; border-color:#b9eccd; }
    .pill.bad { color:var(--red); background:#fee4e2; border-color:#fecdca; }
    .pill.quiet { color:#56636b; background:#eef1f1; }
    .live-card { border-radius:28px; padding:22px; background:linear-gradient(135deg,#17324c,#0f766e); color:white; min-height:100%; }
    .live-card strong { display:block; font-size:clamp(42px,7vw,82px); line-height:.9; letter-spacing:-.06em; }
    .live-card p { color:rgba(255,255,255,.82); line-height:1.5; }
    .mini-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:18px; }
    .mini { padding:14px; border-radius:18px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.14); }
    .mini b { display:block; font-size:23px; }
    .bar-row { display:grid; grid-template-columns:92px 1fr 220px; align-items:center; gap:12px; margin:13px 0; }
    .bar-label { font-weight:900; }
    .bar-track,.inline-bar { height:15px; border-radius:999px; background:rgba(20,32,39,.08); overflow:hidden; border:1px solid rgba(20,32,39,.05); }
    .bar-track span,.inline-bar span { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--teal),var(--gold)); }
    .bar-value { font-weight:900; }
    .bar-value em { display:block; color:var(--muted); font-size:12px; font-style:normal; font-weight:700; }
    .graph-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .graph-card {
      min-height:360px; border:1px solid rgba(20,32,39,.10); border-radius:24px;
      background:rgba(255,255,255,.62); padding:18px; position:relative; overflow:hidden;
    }
    .graph-card h3 { margin-bottom:4px; }
    .graph-card p { margin:0 0 12px; color:var(--muted); font-size:13px; line-height:1.45; }
    canvas { width:100%; height:260px; display:block; cursor:crosshair; }
    .chart-tip {
      position:fixed; z-index:20; pointer-events:none; transform:translate(14px, -18px);
      padding:9px 11px; border-radius:12px; background:rgba(20,32,39,.92); color:#fff;
      font-size:12px; font-weight:800; box-shadow:0 12px 30px rgba(20,32,39,.24);
      opacity:0; transition:opacity .12s ease;
    }
    .graph-wide { grid-column:1 / -1; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:12px 10px; border-bottom:1px solid rgba(20,32,39,.10); text-align:left; vertical-align:middle; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    td small { display:block; color:var(--muted); font-size:11px; margin-top:3px; }
    .table-wrap { overflow-x:auto; border-radius:22px; border:1px solid rgba(20,32,39,.10); background:rgba(255,255,255,.58); }
    .inline-bar { width:min(170px,42vw); display:inline-block; margin-right:10px; vertical-align:middle; }
    .note { padding:18px; border-radius:22px; background:rgba(15,118,110,.08); border:1px solid rgba(15,118,110,.14); }
    .status-line { margin-top:10px; color:var(--muted); font-size:13px; }
    .hidden { display:none; }
    @media (max-width:1100px) { .metrics{grid-template-columns:repeat(2,minmax(0,1fr));}.two,.three,.graph-grid{grid-template-columns:1fr}.bar-row{grid-template-columns:80px 1fr}.bar-value{grid-column:2}.graph-wide{grid-column:auto} }
    @media (max-width:700px) { .wrap{width:min(100% - 20px,1480px);padding-top:10px}.hero,.panel{border-radius:22px;padding:18px}.metrics,.mini-grid{grid-template-columns:1fr}.controls>*{width:100%} }
    @keyframes rise { from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:translateY(0)} }
  </style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <div class="eyebrow">Live data dashboard</div>
    <h1>Upload a CSV. The dashboard recalculates.</h1>
    <p>
      Use a scored prediction CSV, optionally with labels, or upload labels separately and join by <b>uid</b>.
      This dashboard updates precision, AUC, deciles, top users, and source split directly in the browser.
    </p>
    <div class="controls">
      <span class="pill good">No server needed</span>
      <span class="pill good">Prediction + label join by UID</span>
      <span class="pill">For raw user/call scoring, use Python model pipeline first</span>
    </div>
  </section>

  <section class="grid metrics" id="cards">__CARDS__</section>

  <section class="panel grid two">
    <div>
      <h2>Data Input</h2>
      <div class="upload-zone">
        <div class="controls">
          <div>
            <label>Prediction CSV</label><br>
            <input id="predictionFile" type="file" accept=".csv,text/csv">
          </div>
          <div>
            <label>Label CSV Optional</label><br>
            <input id="labelFile" type="file" accept=".csv,text/csv">
          </div>
          <div>
            <button id="resetDefault" class="secondary">Reset April Default</button>
          </div>
        </div>
        <p class="footer">
          Prediction file needs <b>uid</b> plus a score column like <b>t1_loan_conversion_score</b>, <b>pred_prob</b>, or <b>score</b>.
          Label file needs <b>uid</b> plus <b>converted_full</b> or <b>converted</b>.
        </p>
        <details>
          <summary><b>Paste CSV instead</b></summary>
          <textarea id="csvPaste" placeholder="uid,t1_loan_conversion_score,converted_full&#10;CSD-123,0.91,1"></textarea>
          <button id="loadPasted">Load pasted CSV</button>
        </details>
        <div id="dataStatus" class="status-line">Loaded default April top-100 preview plus April summary metrics.</div>
      </div>

      <div class="controls" id="columnControls">
        <div>
          <label for="scoreCol">Score Column</label><br>
          <select id="scoreCol"></select>
        </div>
        <div>
          <label for="labelCol">Label Column</label><br>
          <select id="labelCol"></select>
        </div>
        <div>
          <label for="uidCol">UID Column</label><br>
          <select id="uidCol"></select>
        </div>
      </div>
    </div>
    <div class="live-card">
      <span class="eyebrow" style="color:rgba(255,255,255,.72)">Selected Cutoff</span>
      <div class="controls">
        <select id="topKSelect"></select>
      </div>
      <strong id="liveConversions">53</strong>
      <p id="liveNarrative">actual conversions in selected group</p>
      <div class="mini-grid">
        <div class="mini"><b id="livePrecision">53.0%</b><span>precision</span></div>
        <div class="mini"><b id="liveRecall">4.2%</b><span>recall</span></div>
        <div class="mini"><b id="liveLift">63.9x</b><span>lift</span></div>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Metrics</h2>
    <div id="precisionBars"></div>
  </section>

  <section class="panel">
    <h2>Graphs</h2>
    <div class="graph-grid">
      <div class="graph-card">
        <h3>Precision & Recall by Top-K</h3>
        <p>Shows how model quality changes as the business expands the contact list.</p>
        <canvas id="topKChart"></canvas>
      </div>
      <div class="graph-card">
        <h3>3D Decile Columns</h3>
        <p>Higher columns on the left mean the model is concentrating conversions in the top-ranked users.</p>
        <canvas id="decileChart"></canvas>
      </div>
      <div class="graph-card">
        <h3>3D Score Distribution</h3>
        <p>Interactive score histogram for the loaded file.</p>
        <canvas id="scoreHistogram"></canvas>
      </div>
      <div class="graph-card">
        <h3>Cumulative Gains</h3>
        <p>Share of all conversions captured as we move down the ranked list.</p>
        <canvas id="gainsChart"></canvas>
      </div>
      <div class="graph-card">
        <h3>3D Conversion Source Pie</h3>
        <p>DIY vs SFDC among labeled converted rows, when those columns are available.</p>
        <canvas id="sourceDonut"></canvas>
      </div>
      <div class="graph-card">
        <h3>3D Campaign / State Columns</h3>
        <p>Conversions by campaign or state in the current loaded dataset.</p>
        <div class="controls" style="margin-top:0">
          <select id="groupChartMode">
            <option value="campaign_id">Campaign</option>
            <option value="state">State</option>
          </select>
        </div>
        <canvas id="groupChart"></canvas>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Decile Performance</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Decile</th><th>Users</th><th>Positives</th><th>Positive Rate</th><th>Lift</th><th>Score Range</th></tr>
        </thead>
        <tbody id="decileBody"></tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>User Explorer</h2>
    <div class="controls">
      <div>
        <label>Search</label><br>
        <input id="userSearch" type="search" placeholder="UID, user_id, campaign, state">
      </div>
      <div>
        <label>Outcome Filter</label><br>
        <select id="outcomeFilter">
          <option value="all">All</option>
          <option value="converted">Converted</option>
          <option value="not_converted">Not converted</option>
          <option value="diy">DIY converted</option>
          <option value="sfdc">SFDC converted</option>
          <option value="unlabeled">Unlabeled/unknown</option>
        </select>
      </div>
      <div>
        <label>Rows</label><br>
        <select id="rowLimit"><option>10</option><option selected>25</option><option>50</option><option>100</option><option>500</option></select>
      </div>
      <button id="downloadTop">Download visible CSV</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Rank</th><th>User</th><th>Score</th><th>Outcome</th><th>Source</th><th>Calls</th><th>Answered</th><th>Avg Duration</th><th>Campaign</th><th>State</th></tr>
        </thead>
        <tbody id="userBody"></tbody>
      </table>
    </div>
    <div id="userCount" class="footer"></div>
  </section>

  <section class="panel grid three">
    <div class="note"><b>Current Dataset</b><span id="datasetNote"></span></div>
    <div class="note"><b>Label Source Split</b><span id="sourceSplit"></span></div>
    <div class="note"><b>Mode</b><span id="modeNote"></span></div>
  </section>

  <p class="footer">
    This is a browser-side dashboard. It recalculates metrics from uploaded scored CSVs.
    It does not run the Python model on raw call/user files yet.
  </p>
</main>

<script>
const DEFAULT_PAYLOAD = __DATA_JSON__;
const DEFAULT_ROWS = DEFAULT_PAYLOAD.defaultRows;
const DEFAULT_SUMMARY = DEFAULT_PAYLOAD.defaultSummary;
let state = {
  rows: DEFAULT_ROWS.map(r => ({...r})),
  predictionRows: DEFAULT_ROWS.map(r => ({...r})),
  labelRows: null,
  columns: Object.keys(DEFAULT_ROWS[0] || {}),
  scoreCol: "t1_loan_conversion_score",
  labelCol: "converted_full",
  uidCol: "uid",
  sourceName: "April default top-100 preview"
};
let visibleRows = [];

const scoreCandidates = ["t1_loan_conversion_score","t1_pred_prob","pred_prob","prediction_score","score","probability","prob","model_score"];
const labelCandidates = ["converted_full","converted","actual","label","y","outcome"];
const uidCandidates = ["uid","csd_id","Lead Id","lead_id","CSD ID"];
const kOptions = [20,50,100,500,1000,5000,10000];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[c]));
}
function n(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const cleaned = String(value).replace(/,/g, "").replace(/%/g, "").trim();
  const out = Number(cleaned);
  return Number.isFinite(out) ? out : NaN;
}
function fmtNum(value) { return Number(value || 0).toLocaleString("en-IN"); }
function fmtPct(value, digits = 1) { return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "-"; }
function fmtScore(value) { return Number.isFinite(value) ? Number(value).toFixed(4) : "-"; }
function normalizeUid(value) { return String(value ?? "").trim().toUpperCase(); }

const CHART_HITS = {};
let chartTooltip = null;

function ensureChartTooltip() {
  if (!chartTooltip) {
    chartTooltip = document.createElement("div");
    chartTooltip.className = "chart-tip";
    document.body.appendChild(chartTooltip);
  }
  return chartTooltip;
}

function registerChartHits(id, hits) {
  CHART_HITS[id] = hits || [];
  const canvas = document.getElementById(id);
  if (!canvas || canvas.dataset.hoverReady === "1") return;
  canvas.dataset.hoverReady = "1";
  canvas.addEventListener("mousemove", event => {
    const rect = canvas.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const found = findChartHit(id, point);
    if (!found) return hideChartTooltip();
    const tip = ensureChartTooltip();
    tip.innerHTML = found.tip;
    tip.style.left = `${event.clientX}px`;
    tip.style.top = `${event.clientY}px`;
    tip.style.opacity = "1";
  });
  canvas.addEventListener("mouseleave", hideChartTooltip);
}

function hideChartTooltip() {
  if (chartTooltip) chartTooltip.style.opacity = "0";
}

function angleWithin(angle, start, end) {
  const twoPi = Math.PI * 2;
  while (end < start) end += twoPi;
  while (angle < start) angle += twoPi;
  return angle >= start && angle <= end;
}

function findChartHit(id, point) {
  const hits = CHART_HITS[id] || [];
  for (let i = hits.length - 1; i >= 0; i--) {
    const hit = hits[i];
    if (hit.type === "rect") {
      if (point.x >= hit.x && point.x <= hit.x + hit.w && point.y >= hit.y && point.y <= hit.y + hit.h) return hit;
    }
    if (hit.type === "circle") {
      const dx = point.x - hit.x, dy = point.y - hit.y;
      if (Math.sqrt(dx * dx + dy * dy) <= hit.r) return hit;
    }
    if (hit.type === "ellipseArc") {
      const dx = point.x - hit.cx;
      const dy = (point.y - hit.cy) / hit.yScale;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);
      if (dist <= hit.r && angleWithin(angle, hit.start, hit.end)) return hit;
    }
  }
  return null;
}

function hexToRgb(hex) {
  const cleaned = String(hex || "").replace("#", "");
  if (cleaned.length !== 6) return { r: 15, g: 118, b: 110 };
  return {
    r: parseInt(cleaned.slice(0, 2), 16),
    g: parseInt(cleaned.slice(2, 4), 16),
    b: parseInt(cleaned.slice(4, 6), 16)
  };
}

function shadeHex(hex, amount) {
  const { r, g, b } = hexToRgb(hex);
  const clamp = value => Math.max(0, Math.min(255, Math.round(value)));
  return `rgb(${clamp(r + amount)}, ${clamp(g + amount)}, ${clamp(b + amount)})`;
}

function chartValue(value, percent = false) {
  return percent ? fmtPct(value, 1) : fmtNum(value);
}

function setupCanvas(id) {
  const canvas = document.getElementById(id);
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(220, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { canvas, ctx, width: canvas.width / ratio, height: canvas.height / ratio };
}

function clearChart(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.fillRect(0, 0, width, height);
}

function drawNoData(id, message) {
  const { ctx, width, height } = setupCanvas(id);
  clearChart(ctx, width, height);
  registerChartHits(id, []);
  ctx.fillStyle = "#65737b";
  ctx.font = "700 15px Bahnschrift, Candara, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, width / 2, height / 2);
}

function drawAxes(ctx, x, y, w, h) {
  ctx.strokeStyle = "rgba(20,32,39,.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.stroke();
  ctx.strokeStyle = "rgba(20,32,39,.07)";
  for (let i = 1; i <= 4; i++) {
    const gy = y + (h * i / 5);
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + w, gy);
    ctx.stroke();
  }
}

function drawLineChart(id, seriesList, labels, yMax, formatter = v => v.toFixed(2)) {
  const { ctx, width, height } = setupCanvas(id);
  clearChart(ctx, width, height);
  const hits = [];
  const pad = { l: 48, r: 20, t: 18, b: 42 };
  const x = pad.l, y = pad.t, w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  drawAxes(ctx, x, y, w, h);
  ctx.fillStyle = "#65737b";
  ctx.font = "11px Bahnschrift, Candara, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const value = yMax * (1 - i / 5);
    ctx.fillText(formatter(value), x - 8, y + h * i / 5 + 4);
  }
  seriesList.forEach(series => {
    ctx.strokeStyle = series.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    series.values.forEach((value, idx) => {
      const px = x + (labels.length === 1 ? w / 2 : idx * w / (labels.length - 1));
      const py = y + h - ((value || 0) / yMax) * h;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    series.values.forEach((value, idx) => {
      const px = x + (labels.length === 1 ? w / 2 : idx * w / (labels.length - 1));
      const py = y + h - ((value || 0) / yMax) * h;
      ctx.fillStyle = series.color;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
      hits.push({
        type: "circle",
        x: px,
        y: py,
        r: 11,
        tip: `${escapeHtml(series.name)}<br><b>${escapeHtml(labels[idx])}</b>: ${escapeHtml(formatter(value || 0))}`
      });
    });
  });
  ctx.textAlign = "center";
  ctx.fillStyle = "#65737b";
  labels.forEach((label, idx) => {
    if (idx % Math.ceil(labels.length / 5) !== 0 && idx !== labels.length - 1) return;
    const px = x + (labels.length === 1 ? w / 2 : idx * w / (labels.length - 1));
    ctx.fillText(String(label), px, y + h + 22);
  });
  let lx = x + 8;
  seriesList.forEach(series => {
    ctx.fillStyle = series.color;
    ctx.fillRect(lx, y + 6, 12, 4);
    ctx.fillStyle = "#142027";
    ctx.textAlign = "left";
    ctx.fillText(series.name, lx + 18, y + 10);
    lx += 92;
  });
  registerChartHits(id, hits);
}

function drawBarChart(id, bars, options = {}) {
  const { ctx, width, height } = setupCanvas(id);
  clearChart(ctx, width, height);
  if (!bars.length || bars.every(bar => !Number(bar.value))) return drawNoData(id, "No data for this chart");
  const hits = [];
  const pad = { l: 54, r: 30, t: 26, b: 64 };
  const x = pad.l, y = pad.t, w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const maxVal = Math.max(...bars.map(b => b.value), 0.01);
  drawAxes(ctx, x, y, w, h);
  const gap = Math.max(8, Math.min(14, w / (bars.length * 5)));
  const slot = (w - gap * (bars.length - 1)) / bars.length;
  const depth = Math.min(18, Math.max(9, slot * 0.20));
  const bw = Math.max(8, slot - depth);
  const baseColor = options.color || "#0f766e";
  const endColor = options.color2 || "#d9911b";
  bars.forEach((bar, idx) => {
    const value = Math.max(0, Number(bar.value) || 0);
    const bh = (value / maxVal) * h;
    const bx = x + idx * (slot + gap);
    const by = y + h - bh;
    const topLift = depth * 0.55;

    ctx.save();
    ctx.shadowColor = "rgba(20,32,39,.18)";
    ctx.shadowBlur = 14;
    ctx.shadowOffsetY = 8;
    ctx.fillStyle = "rgba(20,32,39,.08)";
    ctx.beginPath();
    ctx.ellipse(bx + bw / 2 + depth / 2, y + h + 7, Math.max(10, bw * 0.55), 6, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.beginPath();
    ctx.moveTo(bx + bw, by);
    ctx.lineTo(bx + bw + depth, by - topLift);
    ctx.lineTo(bx + bw + depth, y + h - topLift);
    ctx.lineTo(bx + bw, y + h);
    ctx.closePath();
    ctx.fillStyle = shadeHex(baseColor, -46);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + depth, by - topLift);
    ctx.lineTo(bx + bw + depth, by - topLift);
    ctx.lineTo(bx + bw, by);
    ctx.closePath();
    ctx.fillStyle = shadeHex(baseColor, 34);
    ctx.fill();

    const gradient = ctx.createLinearGradient(0, by, 0, y + h);
    gradient.addColorStop(0, baseColor);
    gradient.addColorStop(1, endColor);
    ctx.fillStyle = gradient;
    ctx.fillRect(bx, by, bw, bh);

    ctx.strokeStyle = "rgba(255,255,255,.42)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx + 3, by + 2);
    ctx.lineTo(bx + 3, y + h - 2);
    ctx.stroke();

    ctx.fillStyle = "#142027";
    ctx.font = "900 10px Bahnschrift, Candara, sans-serif";
    ctx.textAlign = "center";
    if (bw > 22 && bh > 20) {
      ctx.fillText(chartValue(value, options.percent), bx + bw / 2, Math.max(y + 12, by - topLift - 5));
    }

    ctx.fillStyle = "#65737b";
    ctx.font = "10px Bahnschrift, Candara, sans-serif";
    ctx.textAlign = "center";
    ctx.save();
    ctx.translate(bx + bw / 2, y + h + 12);
    ctx.rotate(-Math.PI / 5);
    ctx.fillText(String(bar.label).slice(0, 14), 0, 0);
    ctx.restore();

    hits.push({
      type: "rect",
      x: bx,
      y: Math.min(by - topLift, y + h - 10),
      w: bw + depth,
      h: Math.max(12, bh + topLift + 10),
      tip: `<b>${escapeHtml(bar.label)}</b><br>${escapeHtml(options.percent ? "Rate" : "Value")}: ${escapeHtml(chartValue(value, options.percent))}`
    });
  });
  ctx.fillStyle = "#65737b";
  ctx.font = "11px Bahnschrift, Candara, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const value = maxVal * (1 - i / 5);
    ctx.fillText(options.percent ? fmtPct(value, 1) : fmtNum(value), x - 8, y + h * i / 5 + 4);
  }
  registerChartHits(id, hits);
}

function drawDonut(id, segments) {
  const { ctx, width, height } = setupCanvas(id);
  clearChart(ctx, width, height);
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  if (!total) return drawNoData(id, "No source split available");
  const hits = [];
  const cx = width * 0.37, cy = height * 0.42, r = Math.min(width, height) * 0.30;
  const yScale = 0.62, depth = 24;

  function pieSlice(centerY, start, end, color) {
    ctx.save();
    ctx.translate(cx, centerY);
    ctx.scale(1, yScale);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, r, start, end);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
  }

  let start = -Math.PI / 2;
  const arcs = segments.map(seg => {
    const end = start + (seg.value / total) * Math.PI * 2;
    const arc = { ...seg, start, end };
    start = end;
    return arc;
  });

  for (let layer = depth; layer >= 2; layer -= 2) {
    arcs.forEach(seg => pieSlice(cy + layer, seg.start, seg.end, shadeHex(seg.color, -58)));
  }

  arcs.forEach(seg => {
    pieSlice(cy, seg.start, seg.end, seg.color);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(1, yScale);
    ctx.strokeStyle = "rgba(255,255,255,.55)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, r, seg.start, seg.end);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
    hits.push({
      type: "ellipseArc",
      cx,
      cy,
      r,
      yScale,
      start: seg.start,
      end: seg.end,
      tip: `<b>${escapeHtml(seg.name)}</b><br>${fmtNum(seg.value)} users<br>${fmtPct(seg.value / total, 1)} of converted`
    });
  });

  ctx.fillStyle = "#142027";
  ctx.font = "900 24px Bahnschrift, Candara, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(fmtNum(total), cx, cy + depth + 47);
  ctx.font = "700 11px Bahnschrift, Candara, sans-serif";
  ctx.fillStyle = "#65737b";
  ctx.fillText("converted users", cx, cy + depth + 63);

  let ly = height * 0.34;
  segments.forEach(seg => {
    const pct = total ? seg.value / total : 0;
    ctx.fillStyle = seg.color;
    ctx.beginPath();
    ctx.moveTo(width * 0.68, ly - 9);
    ctx.lineTo(width * 0.68 + 13, ly - 14);
    ctx.lineTo(width * 0.68 + 27, ly - 9);
    ctx.lineTo(width * 0.68 + 14, ly - 3);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#142027";
    ctx.font = "700 13px Bahnschrift, Candara, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${seg.name}: ${fmtNum(seg.value)} (${fmtPct(pct, 1)})`, width * 0.68 + 34, ly + 2);
    ly += 28;
  });
  registerChartHits(id, hits);
}

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i], next = text[i + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') { field += '"'; i++; }
      else if (char === '"') inQuotes = false;
      else field += char;
    } else {
      if (char === '"') inQuotes = true;
      else if (char === ",") { row.push(field); field = ""; }
      else if (char === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else if (char !== "\r") field += char;
    }
  }
  row.push(field);
  if (row.some(v => String(v).trim() !== "")) rows.push(row);
  if (!rows.length) return [];
  const headers = rows.shift().map(h => String(h).trim());
  return rows
    .filter(values => values.some(v => String(v).trim() !== ""))
    .map(values => {
      const obj = {};
      headers.forEach((header, idx) => obj[header] = values[idx] ?? "");
      return obj;
    });
}

function detectColumn(columns, candidates, fallback = "") {
  const lowerMap = new Map(columns.map(c => [String(c).toLowerCase(), c]));
  for (const candidate of candidates) {
    const found = lowerMap.get(String(candidate).toLowerCase());
    if (found) return found;
  }
  return fallback || columns[0] || "";
}

function setSelectOptions(selectId, columns, selected, includeNone = false) {
  const select = document.querySelector(selectId);
  const opts = includeNone ? [`<option value="">No label / unlabeled</option>`] : [];
  opts.push(...columns.map(col => `<option value="${escapeHtml(col)}" ${col === selected ? "selected" : ""}>${escapeHtml(col)}</option>`));
  select.innerHTML = opts.join("");
}

function configureColumns() {
  state.columns = Object.keys(state.rows[0] || {});
  state.uidCol = state.columns.includes(state.uidCol) ? state.uidCol : detectColumn(state.columns, uidCandidates);
  state.scoreCol = state.columns.includes(state.scoreCol) ? state.scoreCol : detectColumn(state.columns, scoreCandidates);
  state.labelCol = state.columns.includes(state.labelCol) ? state.labelCol : detectColumn(state.columns, labelCandidates, "");
  setSelectOptions("#uidCol", state.columns, state.uidCol);
  setSelectOptions("#scoreCol", state.columns, state.scoreCol);
  setSelectOptions("#labelCol", state.columns, state.labelCol, true);
}

function mergeLabelsIfNeeded() {
  let rows = state.predictionRows.map(row => ({...row}));
  if (state.labelRows && state.labelRows.length) {
    const labelColumns = Object.keys(state.labelRows[0] || {});
    const labelUid = detectColumn(labelColumns, uidCandidates);
    const labelMap = new Map();
    for (const row of state.labelRows) labelMap.set(normalizeUid(row[labelUid]), row);
    rows = rows.map(row => {
      const uid = normalizeUid(row[state.uidCol] ?? row.uid);
      const label = labelMap.get(uid);
      return label ? {...row, ...label, [state.uidCol]: row[state.uidCol] ?? uid} : row;
    });
  }
  state.rows = rows;
  configureColumns();
}

function preparedRows() {
  return state.rows
    .map((row, idx) => ({
      raw: row,
      rank: idx + 1,
      uid: normalizeUid(row[state.uidCol] ?? row.uid),
      score: n(row[state.scoreCol]),
      label: state.labelCol ? n(row[state.labelCol]) : NaN,
      diy: n(row.converted_from_diy),
      sfdc: n(row.converted_from_sfdc),
      user_id: row.user_id ?? row.userId ?? "",
      total_calls: n(row.total_calls),
      answered_calls: n(row.answered_calls),
      answered_rate: n(row.answered_rate),
      avg_call_duration: n(row.avg_call_duration),
      campaign_id: row.campaign_id ?? row.campaign ?? "",
      state: row.state ?? row.State ?? ""
    }))
    .filter(row => Number.isFinite(row.score))
    .sort((a, b) => b.score - a.score)
    .map((row, idx) => ({...row, rank: idx + 1}));
}

function hasLabels(rows) {
  return Boolean(state.labelCol) && rows.some(row => Number.isFinite(row.label));
}

function auc(rows) {
  const labeled = rows.filter(r => Number.isFinite(r.label));
  const positives = labeled.filter(r => r.label === 1).length;
  const negatives = labeled.filter(r => r.label !== 1).length;
  if (!positives || !negatives) return NaN;
  const sorted = labeled.slice().sort((a,b) => a.score - b.score);
  let rank = 1, sumPosRanks = 0, i = 0;
  while (i < sorted.length) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].score === sorted[i].score) j++;
    const avgRank = (rank + rank + (j - i) - 1) / 2;
    for (let k = i; k < j; k++) if (sorted[k].label === 1) sumPosRanks += avgRank;
    rank += (j - i);
    i = j;
  }
  return (sumPosRanks - positives * (positives + 1) / 2) / (positives * negatives);
}

function metrics(rows) {
  const labels = hasLabels(rows);
  const total = rows.length;
  const positives = labels ? rows.filter(r => r.label === 1).length : 0;
  const baseline = labels && total ? positives / total : NaN;
  const ks = kOptions.filter(k => k <= total || k <= 100).map(k => Math.min(k, total)).filter((v, i, arr) => v > 0 && arr.indexOf(v) === i);
  const topK = ks.map(k => {
    const top = rows.slice(0, k);
    const conv = labels ? top.filter(r => r.label === 1).length : NaN;
    return {
      k, conversions: conv, precision: labels ? conv / k : NaN,
      recall: labels && positives ? conv / positives : NaN,
      scoreMin: top.length ? top[top.length - 1].score : NaN
    };
  });
  const deciles = buildDeciles(rows, labels, baseline);
  const topDecile = deciles.find(d => d.decile === 9);
  return { labels, total, positives, baseline, auc: labels ? auc(rows) : NaN, topK, deciles, topDecileLift: topDecile && labels ? topDecile.lift : NaN };
}

function buildDeciles(rows, labels, baseline) {
  const asc = rows.slice().sort((a,b) => a.score - b.score);
  const buckets = Array.from({length: 10}, (_, decile) => ({decile, users:0, positives:0, scoreMin:Infinity, scoreMax:-Infinity, scoreSum:0}));
  asc.forEach((row, idx) => {
    const decile = Math.min(9, Math.floor(idx * 10 / asc.length));
    const b = buckets[decile];
    b.users += 1;
    b.positives += labels && row.label === 1 ? 1 : 0;
    b.scoreMin = Math.min(b.scoreMin, row.score);
    b.scoreMax = Math.max(b.scoreMax, row.score);
    b.scoreSum += row.score;
  });
  return buckets.reverse().map(b => {
    const rate = labels && b.users ? b.positives / b.users : NaN;
    return {...b, rate, lift: labels && baseline ? rate / baseline : NaN, meanScore: b.users ? b.scoreSum / b.users : NaN};
  });
}

function updateCards(m) {
  const p100 = m.topK.find(x => x.k === Math.min(100, m.total));
  const cards = [
    ["Rows", fmtNum(m.total), "Current uploaded/scored rows"],
    ["Labels", m.labels ? fmtNum(m.positives) : "Unlabeled", m.labels ? `Baseline ${fmtPct(m.baseline, 3)}` : "Upload label CSV to validate"],
    ["AUC", m.labels ? fmtScore(m.auc) : "-", "Needs labels"],
    ["Precision@100", m.labels && p100 ? fmtPct(p100.precision) : "-", m.labels && p100 ? `${fmtNum(p100.conversions)} / ${fmtNum(p100.k)} converted` : "Needs labels"]
  ];
  document.querySelector("#cards").innerHTML = cards.map(([label, value, note]) => `
    <article class="metric-card"><p>${label}</p><strong>${value}</strong><span>${note}</span></article>
  `).join("");
}

function renderTopKOptions(m) {
  const select = document.querySelector("#topKSelect");
  const previous = Number(select.value) || 100;
  select.innerHTML = m.topK.map(item => `<option value="${item.k}" ${item.k === previous || (!m.topK.some(x => x.k === previous) && item.k === Math.min(100, m.total)) ? "selected" : ""}>Top ${fmtNum(item.k)}</option>`).join("");
}

function renderPrecisionBars(m) {
  if (!m.topK.length) { document.querySelector("#precisionBars").innerHTML = "<p class='footer'>No scored rows loaded.</p>"; return; }
  const maxPrecision = Math.max(...m.topK.map(item => Number.isFinite(item.precision) ? item.precision : 0), 0.01);
  document.querySelector("#precisionBars").innerHTML = m.topK.map(item => {
    const width = Number.isFinite(item.precision) ? (item.precision / maxPrecision) * 100 : 8;
    const value = m.labels ? `${fmtPct(item.precision)} <em>${fmtNum(item.conversions)} conv, ${fmtPct(item.recall)} recall</em>` : `<em>score min ${fmtScore(item.scoreMin)}</em>`;
    return `<div class="bar-row"><div class="bar-label">Top ${fmtNum(item.k)}</div><div class="bar-track"><span style="width:${width}%"></span></div><div class="bar-value">${value}</div></div>`;
  }).join("");
}

function updateLiveCard(m) {
  const k = Number(document.querySelector("#topKSelect").value) || Math.min(100, m.total);
  const item = m.topK.find(x => x.k === k) || m.topK[0];
  if (!item) return;
  document.querySelector("#liveConversions").textContent = m.labels ? fmtNum(item.conversions) : "-";
  document.querySelector("#liveNarrative").textContent = m.labels ? `actual conversions in top ${fmtNum(item.k)} users` : `labels not loaded for top ${fmtNum(item.k)}`;
  document.querySelector("#livePrecision").textContent = m.labels ? fmtPct(item.precision) : "-";
  document.querySelector("#liveRecall").textContent = m.labels ? fmtPct(item.recall) : "-";
  document.querySelector("#liveLift").textContent = m.labels && m.baseline ? `${(item.precision / m.baseline).toFixed(1)}x` : "-";
}

function renderDeciles(m) {
  const maxRate = Math.max(...m.deciles.map(d => Number.isFinite(d.rate) ? d.rate : 0), 0.01);
  document.querySelector("#decileBody").innerHTML = m.deciles.map(d => {
    const width = Number.isFinite(d.rate) ? (d.rate / maxRate) * 100 : 6;
    return `<tr>
      <td>D${d.decile}</td><td>${fmtNum(d.users)}</td><td>${m.labels ? fmtNum(d.positives) : "-"}</td>
      <td><div class="inline-bar"><span style="width:${width}%"></span></div><b>${m.labels ? fmtPct(d.rate,2) : "-"}</b></td>
      <td>${m.labels ? `${d.lift.toFixed(2)}x` : "-"}</td><td>${fmtScore(d.scoreMin)} - ${fmtScore(d.scoreMax)}</td>
    </tr>`;
  }).join("");
}

function source(row) {
  if (row.diy === 1) return "DIY";
  if (row.sfdc === 1) return "SFDC";
  return "-";
}

function renderUsers(rows) {
  const query = document.querySelector("#userSearch").value.trim().toLowerCase();
  const filter = document.querySelector("#outcomeFilter").value;
  const limit = Number(document.querySelector("#rowLimit").value);
  const filtered = rows.filter(row => {
    const haystack = `${row.uid} ${row.user_id} ${row.campaign_id} ${row.state}`.toLowerCase();
    const searchOk = haystack.includes(query);
    let filterOk = true;
    if (filter === "converted") filterOk = row.label === 1;
    if (filter === "not_converted") filterOk = Number.isFinite(row.label) && row.label !== 1;
    if (filter === "diy") filterOk = row.diy === 1;
    if (filter === "sfdc") filterOk = row.sfdc === 1;
    if (filter === "unlabeled") filterOk = !Number.isFinite(row.label);
    return searchOk && filterOk;
  });
  visibleRows = filtered.slice(0, limit);
  document.querySelector("#userBody").innerHTML = visibleRows.length ? visibleRows.map(row => {
    const labeled = Number.isFinite(row.label);
    const converted = labeled && row.label === 1;
    return `<tr>
      <td>${row.rank}</td><td><b>${escapeHtml(row.uid)}</b><small>${escapeHtml(row.user_id)}</small></td>
      <td>${fmtScore(row.score)}</td>
      <td><span class="pill ${converted ? "good" : (labeled ? "quiet" : "")}">${labeled ? (converted ? "Converted" : "Not converted") : "Unknown"}</span></td>
      <td>${source(row)}</td><td>${Number.isFinite(row.total_calls) ? fmtNum(row.total_calls) : "-"}</td>
      <td>${Number.isFinite(row.answered_calls) ? fmtNum(row.answered_calls) : "-"}</td>
      <td>${Number.isFinite(row.avg_call_duration) ? `${Number(row.avg_call_duration).toFixed(1)}s` : "-"}</td>
      <td>${escapeHtml(row.campaign_id)}</td><td>${escapeHtml(row.state)}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="10">No users match this filter.</td></tr>`;
  document.querySelector("#userCount").textContent = `Showing ${fmtNum(visibleRows.length)} of ${fmtNum(filtered.length)} matching users.`;
}

function renderNotes(m, rows) {
  document.querySelector("#datasetNote").innerHTML = `${escapeHtml(state.sourceName)}<br>${fmtNum(rows.length)} scored rows.`;
  const diy = rows.filter(r => r.diy === 1).length;
  const sfdc = rows.filter(r => r.sfdc === 1).length;
  document.querySelector("#sourceSplit").innerHTML = m.labels ? `${fmtNum(diy)} DIY rows and ${fmtNum(sfdc)} SFDC rows in current view/data.` : "Upload labels with converted_from_diy / converted_from_sfdc to see source split.";
  document.querySelector("#modeNote").innerHTML = m.labels ? "Validation mode: metrics are calculated against outcomes." : "Scoring-only mode: labels are absent, so AUC/precision/lift are hidden.";
}

function scoreHistogram(rows) {
  if (!rows.length) return [];
  const minScore = Math.min(...rows.map(r => r.score));
  const maxScore = Math.max(...rows.map(r => r.score));
  const bins = 10;
  const width = (maxScore - minScore) || 1;
  const counts = Array.from({ length: bins }, (_, idx) => ({
    label: `${(minScore + idx * width / bins).toFixed(2)}`,
    value: 0
  }));
  rows.forEach(row => {
    const idx = Math.min(bins - 1, Math.floor((row.score - minScore) / width * bins));
    counts[idx].value += 1;
  });
  return counts;
}

function cumulativeGains(rows, m) {
  if (!m.labels || !m.positives) return [];
  const points = [];
  const cuts = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00];
  cuts.forEach(cut => {
    const k = Math.max(1, Math.floor(rows.length * cut));
    const captured = rows.slice(0, k).filter(r => r.label === 1).length / m.positives;
    points.push({ label: `${Math.round(cut * 100)}%`, value: captured, random: cut });
  });
  return points;
}

function groupedBars(rows, m) {
  if (!m.labels) return [];
  const key = document.querySelector("#groupChartMode").value;
  const map = new Map();
  rows.forEach(row => {
    const label = String(row[key] || "Unknown").trim() || "Unknown";
    const rec = map.get(label) || { label, value: 0 };
    if (row.label === 1) rec.value += 1;
    map.set(label, rec);
  });
  return Array.from(map.values())
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);
}

function renderGraphs(rows, m) {
  if (!rows.length) {
    ["topKChart", "decileChart", "scoreHistogram", "gainsChart", "sourceDonut", "groupChart"].forEach(id => drawNoData(id, "No scored rows loaded"));
    return;
  }

  if (m.labels) {
    drawLineChart(
      "topKChart",
      [
        { name: "Precision", color: "#0f766e", values: m.topK.map(item => item.precision) },
        { name: "Recall", color: "#d9911b", values: m.topK.map(item => item.recall) }
      ],
      m.topK.map(item => `Top ${item.k}`),
      Math.max(...m.topK.flatMap(item => [item.precision, item.recall]), 0.01),
      value => fmtPct(value, 0)
    );
  } else {
    drawNoData("topKChart", "Upload labels to plot precision and recall");
  }

  drawBarChart(
    "decileChart",
    m.deciles.map(row => ({ label: `D${row.decile}`, value: m.labels ? row.rate : row.meanScore })),
    { percent: m.labels, color: "#0f766e", color2: "#99f6e4" }
  );

  drawBarChart(
    "scoreHistogram",
    scoreHistogram(rows),
    { color: "#17324c", color2: "#0f766e" }
  );

  const gains = cumulativeGains(rows, m);
  if (gains.length) {
    drawLineChart(
      "gainsChart",
      [
        { name: "Model", color: "#0f766e", values: gains.map(point => point.value) },
        { name: "Random", color: "#c2573a", values: gains.map(point => point.random) }
      ],
      gains.map(point => point.label),
      1,
      value => fmtPct(value, 0)
    );
  } else {
    drawNoData("gainsChart", "Upload labels to plot cumulative gains");
  }

  const diy = rows.filter(r => r.label === 1 && r.diy === 1).length;
  const sfdc = rows.filter(r => r.label === 1 && r.sfdc === 1).length;
  const other = rows.filter(r => r.label === 1 && r.diy !== 1 && r.sfdc !== 1).length;
  drawDonut("sourceDonut", [
    { name: "DIY", value: diy, color: "#0f766e" },
    { name: "SFDC", value: sfdc, color: "#d9911b" },
    { name: "Other", value: other, color: "#c2573a" }
  ]);

  drawBarChart(
    "groupChart",
    groupedBars(rows, m),
    { color: "#d9911b", color2: "#0f766e" }
  );
}

function renderAll() {
  const rows = preparedRows();
  const m = metrics(rows);
  updateCards(m);
  renderTopKOptions(m);
  renderPrecisionBars(m);
  updateLiveCard(m);
  renderGraphs(rows, m);
  renderDeciles(m);
  renderUsers(rows);
  renderNotes(m, rows);
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function loadPredictionFile(file) {
  const text = await readFile(file);
  const rows = parseCSV(text);
  if (!rows.length) throw new Error("Prediction CSV had no rows.");
  state.predictionRows = rows;
  state.labelRows = null;
  state.sourceName = file.name;
  state.rows = rows.map(r => ({...r}));
  configureColumns();
  mergeLabelsIfNeeded();
  document.querySelector("#dataStatus").textContent = `Loaded ${fmtNum(rows.length)} prediction rows from ${file.name}.`;
  renderAll();
}

async function loadLabelFile(file) {
  const text = await readFile(file);
  const rows = parseCSV(text);
  if (!rows.length) throw new Error("Label CSV had no rows.");
  state.labelRows = rows;
  mergeLabelsIfNeeded();
  document.querySelector("#dataStatus").textContent = `Joined ${fmtNum(rows.length)} label rows from ${file.name} by UID.`;
  renderAll();
}

function resetDefault(event) {
  event.preventDefault();
  state = {
    rows: DEFAULT_ROWS.map(r => ({...r})),
    predictionRows: DEFAULT_ROWS.map(r => ({...r})),
    labelRows: null,
    columns: Object.keys(DEFAULT_ROWS[0] || {}),
    scoreCol: "t1_loan_conversion_score",
    labelCol: "converted_full",
    uidCol: "uid",
    sourceName: "April default top-100 preview"
  };
  document.querySelector("#predictionFile").value = "";
  document.querySelector("#labelFile").value = "";
  document.querySelector("#csvPaste").value = "";
  document.querySelector("#dataStatus").textContent = "Reset to April default top-100 preview.";
  configureColumns();
  renderAll();
}

function downloadVisible() {
  if (!visibleRows.length) return;
  const headers = ["rank","uid","user_id","score","label","source","total_calls","answered_calls","avg_call_duration","campaign_id","state"];
  const lines = [headers.join(",")];
  for (const row of visibleRows) {
    const vals = [row.rank,row.uid,row.user_id,row.score,Number.isFinite(row.label) ? row.label : "",source(row),row.total_calls,row.answered_calls,row.avg_call_duration,row.campaign_id,row.state]
      .map(value => `"${String(value ?? "").replace(/"/g,'""')}"`);
    lines.push(vals.join(","));
  }
  const blob = new Blob([lines.join("\n")], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "dashboard_visible_users.csv"; a.click();
  URL.revokeObjectURL(url);
}

function wireEvents() {
  document.querySelector("#predictionFile").addEventListener("change", async e => {
    try { if (e.target.files[0]) await loadPredictionFile(e.target.files[0]); }
    catch (err) { document.querySelector("#dataStatus").textContent = `Error: ${err.message}`; }
  });
  document.querySelector("#labelFile").addEventListener("change", async e => {
    try { if (e.target.files[0]) await loadLabelFile(e.target.files[0]); }
    catch (err) { document.querySelector("#dataStatus").textContent = `Error: ${err.message}`; }
  });
  document.querySelector("#loadPasted").addEventListener("click", event => {
    event.preventDefault();
    const rows = parseCSV(document.querySelector("#csvPaste").value);
    if (!rows.length) { document.querySelector("#dataStatus").textContent = "Pasted CSV had no rows."; return; }
    state.predictionRows = rows; state.labelRows = null; state.rows = rows.map(r => ({...r})); state.sourceName = "Pasted CSV";
    configureColumns(); renderAll();
    document.querySelector("#dataStatus").textContent = `Loaded ${fmtNum(rows.length)} pasted rows.`;
  });
  document.querySelector("#scoreCol").addEventListener("change", e => { state.scoreCol = e.target.value; renderAll(); });
  document.querySelector("#labelCol").addEventListener("change", e => { state.labelCol = e.target.value; renderAll(); });
  document.querySelector("#uidCol").addEventListener("change", e => { state.uidCol = e.target.value; mergeLabelsIfNeeded(); renderAll(); });
  document.querySelector("#topKSelect").addEventListener("change", renderAll);
  document.querySelector("#userSearch").addEventListener("input", renderAll);
  document.querySelector("#outcomeFilter").addEventListener("change", renderAll);
  document.querySelector("#rowLimit").addEventListener("change", renderAll);
  document.querySelector("#groupChartMode").addEventListener("change", renderAll);
  document.querySelector("#resetDefault").addEventListener("click", resetDefault);
  document.querySelector("#downloadTop").addEventListener("click", downloadVisible);
  window.addEventListener("resize", () => {
    clearTimeout(window.__chartResizeTimer);
    window.__chartResizeTimer = setTimeout(renderAll, 150);
  });
}

configureColumns();
wireEvents();
renderAll();
</script>
</body>
</html>
"""
    html = html.replace("__CARDS__", cards).replace("__DATA_JSON__", data_json)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(DASHBOARD_PATH)


if __name__ == "__main__":
    main()
