import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "multi_month_training" / "test_outputs"
REPORT_PATH = OUT_DIR / "april_labeled_evaluation_report.json"
DECILE_PATH = OUT_DIR / "april_labeled_decile_tables.csv"
TOP100_PATH = OUT_DIR / "april_t1_top100_with_conversions.csv"
TOP_COMPARE_PATH = OUT_DIR / "april_unlabeled_vs_labeled_top100_comparison_summary.csv"
FULL_COMPARE_PATH = OUT_DIR / "april_unlabeled_vs_labeled_full_rank_comparison_summary.csv"
DASHBOARD_PATH = OUT_DIR / "april_model_validation_dashboard_interactive.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def metric_cards(report: dict) -> str:
    t1 = report["metrics"]["t1_vs_conversion"]
    baseline = t1["baseline_rate"]
    top100_lift = t1["precision_at_100"] / baseline
    cards = [
        ("April Users", f"{t1['rows']:,.0f}", "Validated against DIY + SFDC"),
        ("Conversions", f"{t1['positives']:,.0f}", f"Baseline {pct(baseline, 3)}"),
        ("T1 AUC", f"{t1['auc']:.4f}", "Strong ranking quality"),
        ("Precision@100", pct(t1["precision_at_100"]), f"{t1['top_100_conversions']}/100 converted"),
        ("Top 100 Lift", f"{top100_lift:.1f}x", "Vs random targeting"),
        ("Top-Decile Lift", f"{t1['top_decile_lift']:.2f}x", f"Top decile {pct(t1['top_decile_rate'], 2)}"),
    ]
    return "\n".join(
        f"""
        <article class="metric-card">
          <p>{label}</p>
          <strong>{value}</strong>
          <span>{note}</span>
        </article>
        """
        for label, value, note in cards
    )


def prepare_payload() -> dict:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    deciles = pd.read_csv(DECILE_PATH)
    top100 = pd.read_csv(TOP100_PATH)
    compare = pd.read_csv(TOP_COMPARE_PATH)
    full_compare = pd.read_csv(FULL_COMPARE_PATH)

    top100 = top100.copy()
    top100["rank"] = range(1, len(top100) + 1)
    user_cols = [
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
    users = top100[user_cols].to_dict(orient="records")

    decile_data = {}
    for key, score_col, label_col in [
        ("t1_conversion", "t1_loan_conversion_score", "converted_full"),
        ("t0_conversion", "t0_call_targeting_score", "converted_full"),
        ("t0_call_engagement", "t0_call_targeting_score", "call_engaged_10s"),
    ]:
        frame = deciles[(deciles["score_col"] == score_col) & (deciles["label_col"] == label_col)].copy()
        decile_data[key] = frame.to_dict(orient="records")

    top_k = []
    t1 = report["metrics"]["t1_vs_conversion"]
    for k in [20, 50, 100, 500, 1000, 5000, 10000]:
        top_k.append(
            {
                "k": k,
                "conversions": t1[f"top_{k}_conversions"],
                "precision": t1[f"precision_at_{k}"],
                "recall": t1[f"recall_at_{k}"],
                "score_min": t1.get(f"top_{k}_score_min"),
            }
        )

    comparison = compare.merge(
        full_compare[["stage", "same_order"]].rename(columns={"same_order": "full_same_order"}),
        on="stage",
        how="left",
    ).to_dict(orient="records")

    return {
        "report": report,
        "metric_cards": metric_cards(report),
        "top_users": users,
        "deciles": decile_data,
        "top_k": top_k,
        "comparison": comparison,
    }


def main() -> None:
    payload = prepare_payload()
    report = payload["report"]
    t1 = report["metrics"]["t1_vs_conversion"]
    t0_conv = report["metrics"]["t0_vs_conversion_secondary"]
    t0_call = report["metrics"]["t0_vs_call_engagement_primary"]
    label_info = report["label_info"]

    data_json = json.dumps(
        {
            "baseline": t1["baseline_rate"],
            "totalUsers": t1["rows"],
            "totalConversions": t1["positives"],
            "topK": payload["top_k"],
            "topUsers": payload["top_users"],
            "deciles": payload["deciles"],
            "comparison": payload["comparison"],
            "labelInfo": label_info,
            "t0": {
                "conversionAuc": t0_conv["auc"],
                "conversionP100": t0_conv["precision_at_100"],
                "callAuc": t0_call["auc"],
                "callP100": t0_call["precision_at_100"],
                "callLift": t0_call["top_decile_lift"],
            },
        },
        ensure_ascii=False,
    )

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive April Model Dashboard</title>
  <style>
    :root {
      --ink: #142027;
      --muted: #64737c;
      --paper: #fff8ea;
      --card: rgba(255, 255, 255, 0.80);
      --line: rgba(20, 32, 39, 0.13);
      --teal: #0f766e;
      --mint: #ccfbf1;
      --gold: #d9911b;
      --rust: #c2573a;
      --navy: #17324c;
      --green: #15803d;
      --red: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Bahnschrift", "Candara", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at 10% 10%, rgba(15, 118, 110, 0.23), transparent 28%),
        radial-gradient(circle at 92% 7%, rgba(217, 145, 27, 0.22), transparent 30%),
        linear-gradient(145deg, #fff8ea 0%, #f0eee5 48%, #e6f5f0 100%);
      min-height: 100vh;
    }
    body:before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.25;
      background-image:
        linear-gradient(rgba(20,32,39,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20,32,39,0.05) 1px, transparent 1px);
      background-size: 38px 38px;
    }
    .wrap { width: min(1460px, calc(100% - 34px)); margin: 0 auto; padding: 30px 0 58px; }
    .hero, .panel, .metric-card {
      border: 1px solid var(--line);
      background: var(--card);
      box-shadow: 0 24px 70px rgba(23, 50, 76, 0.13);
      backdrop-filter: blur(12px);
    }
    .hero {
      position: relative;
      overflow: hidden;
      border-radius: 34px;
      padding: clamp(28px, 5vw, 58px);
      background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(235, 249, 244, 0.78));
      animation: rise 650ms ease both;
    }
    .hero:after {
      content: "53";
      position: absolute;
      right: -10px;
      bottom: -70px;
      font-family: Georgia, "Palatino Linotype", serif;
      font-size: clamp(150px, 24vw, 320px);
      font-weight: 900;
      color: rgba(15, 118, 110, 0.08);
      letter-spacing: -0.08em;
    }
    h1, h2, h3 { font-family: Georgia, "Palatino Linotype", serif; letter-spacing: -0.035em; }
    h1 { font-size: clamp(42px, 7vw, 86px); line-height: 0.95; margin: 0 0 18px; max-width: 980px; }
    h2 { font-size: clamp(26px, 3vw, 42px); margin: 0 0 16px; }
    h3 { font-size: 24px; margin: 0 0 12px; }
    .eyebrow { color: var(--teal); font-weight: 900; text-transform: uppercase; letter-spacing: 0.16em; font-size: 13px; }
    .hero p { max-width: 850px; color: #43535b; font-size: 19px; line-height: 1.55; }
    .chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-weight: 900;
      font-size: 12px;
      background: rgba(23,50,76,0.08);
      color: var(--navy);
      border: 1px solid rgba(23,50,76,0.10);
    }
    .pill.good { color: #0f5132; background: #dff7e8; border-color: #b9eccd; }
    .pill.bad { color: var(--red); background: #fee4e2; border-color: #fecdca; }
    .pill.quiet { color: #56636b; background: #eef1f1; }
    .grid { display: grid; gap: 18px; }
    .metrics { grid-template-columns: repeat(6, minmax(0, 1fr)); margin: 20px 0; }
    .metric-card {
      min-height: 145px;
      padding: 20px;
      border-radius: 26px;
      animation: rise 760ms ease both;
    }
    .metric-card p { margin: 0; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 900; }
    .metric-card strong { display: block; margin: 13px 0 8px; font-size: clamp(30px, 3vw, 46px); line-height: 1; letter-spacing: -0.04em; }
    .metric-card span { color: var(--muted); line-height: 1.35; }
    .panel { margin-top: 22px; padding: 24px; border-radius: 30px; animation: rise 820ms ease both; }
    .two { grid-template-columns: 1.1fr 0.9fr; }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin: 14px 0 20px;
    }
    label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 900; }
    select, input {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.86);
      color: var(--ink);
      font: inherit;
      outline-color: var(--teal);
    }
    input[type="search"] { min-width: min(420px, 100%); }
    .live-card {
      border-radius: 28px;
      padding: 22px;
      background: linear-gradient(135deg, #17324c, #0f766e);
      color: white;
      min-height: 100%;
      position: relative;
      overflow: hidden;
    }
    .live-card:after {
      content: "";
      position: absolute;
      width: 220px;
      height: 220px;
      right: -90px;
      top: -90px;
      border-radius: 50%;
      background: rgba(255,255,255,0.16);
    }
    .live-card strong { display: block; font-size: clamp(46px, 7vw, 88px); line-height: .9; letter-spacing: -0.06em; }
    .live-card p { color: rgba(255,255,255,.82); line-height: 1.5; max-width: 520px; }
    .mini-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
    .mini { padding: 14px; border-radius: 18px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.14); }
    .mini b { display: block; font-size: 24px; }
    .bar-row {
      display: grid;
      grid-template-columns: 92px 1fr 220px;
      align-items: center;
      gap: 12px;
      margin: 13px 0;
    }
    .bar-label { font-weight: 900; }
    .bar-track, .inline-bar {
      height: 15px;
      border-radius: 999px;
      background: rgba(20,32,39,0.08);
      overflow: hidden;
      border: 1px solid rgba(20,32,39,0.05);
    }
    .bar-track span, .inline-bar span {
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--teal), var(--gold));
    }
    .bar-value { font-weight: 900; }
    .bar-value em { display: block; color: var(--muted); font-size: 12px; font-style: normal; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 12px 10px; border-bottom: 1px solid rgba(20,32,39,0.10); text-align: left; vertical-align: middle; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    td small { display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }
    .table-wrap { overflow-x: auto; border-radius: 22px; border: 1px solid rgba(20,32,39,0.10); background: rgba(255,255,255,0.58); }
    .inline-bar { width: min(170px, 42vw); display: inline-block; margin-right: 10px; vertical-align: middle; }
    .note { padding: 18px; border-radius: 22px; background: rgba(15,118,110,0.08); border: 1px solid rgba(15,118,110,0.14); }
    .note b { display: block; margin-bottom: 8px; }
    .footer { color: var(--muted); margin-top: 18px; line-height: 1.5; font-size: 13px; }
    .empty { color: var(--muted); padding: 18px; }
    @media (max-width: 1100px) {
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .two, .three { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 80px 1fr; }
      .bar-value { grid-column: 2; }
    }
    @media (max-width: 700px) {
      .wrap { width: min(100% - 20px, 1460px); padding-top: 10px; }
      .hero, .panel { border-radius: 22px; padding: 18px; }
      .metrics, .mini-grid { grid-template-columns: 1fr; }
      .controls { align-items: stretch; }
      .controls > div, select, input { width: 100%; }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="eyebrow">Interactive April validation</div>
      <h1>Move the knobs. The T1 story still holds.</h1>
      <p>
        This version lets you filter the top users, switch decile views, and change call-capacity assumptions.
        The April predictions were still generated blind, before DIY/SFDC labels were added.
      </p>
      <div class="chips">
        <span class="pill good">Blind T1 top 100 stayed identical</span>
        <span class="pill good">53 / 100 converted</span>
        <span class="pill">AUC 0.8491</span>
        <span class="pill">Baseline 0.829%</span>
      </div>
    </section>

    <section class="grid metrics">
      __METRIC_CARDS__
    </section>

    <section class="panel grid two">
      <div>
        <h2>Call Capacity Simulator</h2>
        <p class="footer">Choose how many top-ranked T1 users the team can act on. The result uses actual April outcomes at that cutoff.</p>
        <div class="controls">
          <div>
            <label for="topKSelect">Targeting Size</label><br>
            <select id="topKSelect"></select>
          </div>
        </div>
        <div id="precisionBars"></div>
      </div>
      <div class="live-card">
        <span class="eyebrow" style="color:rgba(255,255,255,.72)">Expected April Outcome</span>
        <strong id="liveConversions">53</strong>
        <p id="liveNarrative">actual conversions in selected group</p>
        <div class="mini-grid">
          <div class="mini"><b id="livePrecision">53.0%</b><span>precision</span></div>
          <div class="mini"><b id="liveRecall">4.2%</b><span>recall</span></div>
          <div class="mini"><b id="liveLift">63.9x</b><span>lift vs baseline</span></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Interactive Decile View</h2>
      <div class="controls">
        <div>
          <label for="decileMode">Metric</label><br>
          <select id="decileMode">
            <option value="t1_conversion">T1 vs loan conversion</option>
            <option value="t0_conversion">T0 vs loan conversion</option>
            <option value="t0_call_engagement">T0 vs call engagement</option>
          </select>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Decile</th>
              <th>Users</th>
              <th>Positives</th>
              <th>Positive Rate</th>
              <th>Lift</th>
              <th>Score Range</th>
            </tr>
          </thead>
          <tbody id="decileBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Top T1 Users Explorer</h2>
      <p class="footer">Search by UID, user_id, campaign, or state. Filter by actual outcome/source after labels were added.</p>
      <div class="controls">
        <div>
          <label for="userSearch">Search</label><br>
          <input id="userSearch" type="search" placeholder="Try CSD-55126913, BIHAR, LPL Topup">
        </div>
        <div>
          <label for="outcomeFilter">Outcome</label><br>
          <select id="outcomeFilter">
            <option value="all">All top 100</option>
            <option value="converted">Converted only</option>
            <option value="not_converted">Not converted</option>
            <option value="diy">DIY converted</option>
            <option value="sfdc">SFDC converted</option>
          </select>
        </div>
        <div>
          <label for="rowLimit">Rows</label><br>
          <select id="rowLimit">
            <option value="10">10</option>
            <option value="25" selected>25</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>User</th>
              <th>Score</th>
              <th>Outcome</th>
              <th>Source</th>
              <th>Calls</th>
              <th>Answered</th>
              <th>Avg Duration</th>
              <th>Campaign</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody id="userBody"></tbody>
        </table>
      </div>
      <div id="userCount" class="footer"></div>
    </section>

    <section class="panel">
      <h2>Blind Prediction Integrity Check</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Status</th>
              <th>Rows</th>
              <th>Same Users</th>
              <th>Same Order</th>
              <th>Same Scores</th>
              <th>Top 100 Conversions</th>
            </tr>
          </thead>
          <tbody id="integrityBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel grid three">
      <div class="note"><b>Label Source Split</b><span id="sourceSplit"></span></div>
      <div class="note"><b>T0 Warning</b><span id="t0Note"></span></div>
      <div class="note"><b>Business Verdict</b>T1 is good enough for a business pilot. T0 still needs improvement before we rely on it as the main call-targeting engine.</div>
    </section>

    <p class="footer">
      Static file with local JavaScript only. No server, no retraining, no label processing when opened.
    </p>
  </main>

  <script>
    const DATA = __DATA_JSON__;

    const fmtPct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
    const fmtNum = (value) => Number(value).toLocaleString("en-IN");
    const fmtScore = (value) => Number(value).toFixed(4);
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, char => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[char]));

    function renderTopKOptions() {
      const select = document.querySelector("#topKSelect");
      select.innerHTML = DATA.topK.map(item => `<option value="${item.k}" ${item.k === 100 ? "selected" : ""}>Top ${fmtNum(item.k)} users</option>`).join("");
    }

    function renderPrecisionBars() {
      const maxPrecision = Math.max(...DATA.topK.map(item => item.precision));
      document.querySelector("#precisionBars").innerHTML = DATA.topK.map(item => {
        const width = maxPrecision ? (item.precision / maxPrecision) * 100 : 0;
        return `
          <div class="bar-row">
            <div class="bar-label">Top ${fmtNum(item.k)}</div>
            <div class="bar-track"><span style="width:${width}%"></span></div>
            <div class="bar-value">${fmtPct(item.precision)} <em>${fmtNum(item.conversions)} conv, ${fmtPct(item.recall)} recall</em></div>
          </div>
        `;
      }).join("");
    }

    function updateTopK() {
      const k = Number(document.querySelector("#topKSelect").value);
      const item = DATA.topK.find(row => row.k === k);
      const lift = item.precision / DATA.baseline;
      document.querySelector("#liveConversions").textContent = fmtNum(item.conversions);
      document.querySelector("#liveNarrative").textContent = `actual conversions in top ${fmtNum(k)} users`;
      document.querySelector("#livePrecision").textContent = fmtPct(item.precision);
      document.querySelector("#liveRecall").textContent = fmtPct(item.recall);
      document.querySelector("#liveLift").textContent = `${lift.toFixed(1)}x`;
    }

    function renderDeciles() {
      const mode = document.querySelector("#decileMode").value;
      const rows = DATA.deciles[mode] || [];
      const baseline = mode === "t0_call_engagement" ? rows.reduce((a, b) => a + b.positives, 0) / rows.reduce((a, b) => a + b.users, 0) : DATA.baseline;
      const maxRate = Math.max(...rows.map(row => row.positive_rate));
      document.querySelector("#decileBody").innerHTML = rows.map(row => {
        const width = maxRate ? (row.positive_rate / maxRate) * 100 : 0;
        const lift = baseline ? row.positive_rate / baseline : 0;
        return `
          <tr>
            <td>D${row.decile}</td>
            <td>${fmtNum(row.users)}</td>
            <td>${fmtNum(row.positives)}</td>
            <td><div class="inline-bar"><span style="width:${width}%"></span></div><b>${fmtPct(row.positive_rate, 2)}</b></td>
            <td>${lift.toFixed(2)}x</td>
            <td>${fmtScore(row.min_score)} - ${fmtScore(row.max_score)}</td>
          </tr>
        `;
      }).join("");
    }

    function userSource(row) {
      if (Number(row.converted_from_diy) === 1) return "DIY";
      if (Number(row.converted_from_sfdc) === 1) return "SFDC";
      return "-";
    }

    function rowMatchesFilter(row, filter) {
      if (filter === "converted") return Number(row.converted_full) === 1;
      if (filter === "not_converted") return Number(row.converted_full) === 0;
      if (filter === "diy") return Number(row.converted_from_diy) === 1;
      if (filter === "sfdc") return Number(row.converted_from_sfdc) === 1;
      return true;
    }

    function renderUsers() {
      const query = document.querySelector("#userSearch").value.trim().toLowerCase();
      const filter = document.querySelector("#outcomeFilter").value;
      const limit = Number(document.querySelector("#rowLimit").value);
      const filtered = DATA.topUsers.filter(row => {
        const haystack = `${row.uid} ${row.user_id} ${row.campaign_id} ${row.state}`.toLowerCase();
        return haystack.includes(query) && rowMatchesFilter(row, filter);
      });
      const visible = filtered.slice(0, limit);
      document.querySelector("#userBody").innerHTML = visible.length ? visible.map(row => {
        const converted = Number(row.converted_full) === 1;
        return `
          <tr>
            <td>${row.rank}</td>
            <td><b>${escapeHtml(row.uid)}</b><small>${escapeHtml(row.user_id)}</small></td>
            <td>${fmtScore(row.t1_loan_conversion_score)}</td>
            <td><span class="pill ${converted ? "good" : "quiet"}">${converted ? "Converted" : "Not converted"}</span></td>
            <td>${userSource(row)}</td>
            <td>${fmtNum(row.total_calls)}</td>
            <td>${fmtNum(row.answered_calls)}</td>
            <td>${Number(row.avg_call_duration).toFixed(1)}s</td>
            <td>${escapeHtml(row.campaign_id)}</td>
            <td>${escapeHtml(row.state)}</td>
          </tr>
        `;
      }).join("") : `<tr><td colspan="10" class="empty">No users match this filter.</td></tr>`;
      document.querySelector("#userCount").textContent = `Showing ${fmtNum(visible.length)} of ${fmtNum(filtered.length)} matching top-100 users.`;
    }

    function renderIntegrity() {
      document.querySelector("#integrityBody").innerHTML = DATA.comparison.map(row => {
        const pass = row.same_uid_set && row.same_scores;
        return `
          <tr>
            <td>${escapeHtml(row.stage)}</td>
            <td><span class="pill ${pass ? "good" : "bad"}">${pass ? "PASS" : "CHECK"}</span></td>
            <td>${row.unlabeled_rows} / ${row.labeled_rows}</td>
            <td>${row.same_uid_set}</td>
            <td>${row.same_order}</td>
            <td>${row.same_scores}</td>
            <td>${row.converted_in_top100}</td>
          </tr>
        `;
      }).join("");
    }

    function renderNotes() {
      const label = DATA.labelInfo;
      document.querySelector("#sourceSplit").innerHTML = `
        ${fmtNum(label.converted_from_diy)} DIY conversions and ${fmtNum(label.converted_from_sfdc)} SFDC conversions.
        Combined April conversions: ${fmtNum(label.converted_full)}.
      `;
      document.querySelector("#t0Note").innerHTML = `
        T0 conversion AUC ${fmtScore(DATA.t0.conversionAuc)}, conversion Precision@100 ${fmtPct(DATA.t0.conversionP100)}.
        Call engagement AUC ${fmtScore(DATA.t0.callAuc)}, call Precision@100 ${fmtPct(DATA.t0.callP100)}.
      `;
    }

    function wireEvents() {
      document.querySelector("#topKSelect").addEventListener("change", updateTopK);
      document.querySelector("#decileMode").addEventListener("change", renderDeciles);
      document.querySelector("#userSearch").addEventListener("input", renderUsers);
      document.querySelector("#outcomeFilter").addEventListener("change", renderUsers);
      document.querySelector("#rowLimit").addEventListener("change", renderUsers);
    }

    renderTopKOptions();
    renderPrecisionBars();
    updateTopK();
    renderDeciles();
    renderUsers();
    renderIntegrity();
    renderNotes();
    wireEvents();
  </script>
</body>
</html>
"""

    html = template.replace("__METRIC_CARDS__", payload["metric_cards"]).replace("__DATA_JSON__", data_json)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(DASHBOARD_PATH)


if __name__ == "__main__":
    main()
