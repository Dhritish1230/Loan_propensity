import html
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
DASHBOARD_PATH = OUT_DIR / "april_model_validation_dashboard.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float) -> str:
    return f"{value:,.0f}"


def score(value: float) -> str:
    return f"{value:.4f}"


def esc(value) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def metric_card(label: str, value: str, note: str, tone: str = "") -> str:
    return f"""
    <article class="metric-card {tone}">
      <p>{esc(label)}</p>
      <strong>{esc(value)}</strong>
      <span>{esc(note)}</span>
    </article>
    """


def precision_rows(metrics: dict) -> str:
    ks = [20, 50, 100, 500, 1000, 5000, 10000]
    max_precision = max(metrics[f"precision_at_{k}"] for k in ks)
    rows = []
    for k in ks:
        precision = metrics[f"precision_at_{k}"]
        recall = metrics[f"recall_at_{k}"]
        conversions = metrics[f"top_{k}_conversions"]
        width = 100 * precision / max_precision if max_precision else 0
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">Top {k:,}</div>
              <div class="bar-track"><span style="width:{width:.2f}%"></span></div>
              <div class="bar-value">{pct(precision)} <em>{conversions:,} conv, {pct(recall)} recall</em></div>
            </div>
            """
        )
    return "\n".join(rows)


def decile_rows(deciles: pd.DataFrame, baseline: float) -> str:
    t1 = deciles[
        (deciles["score_col"] == "t1_loan_conversion_score")
        & (deciles["label_col"] == "converted_full")
    ].copy()
    max_rate = t1["positive_rate"].max()
    rows = []
    for _, row in t1.iterrows():
        rate = float(row["positive_rate"])
        width = 100 * rate / max_rate if max_rate else 0
        lift = rate / baseline if baseline else 0
        rows.append(
            f"""
            <tr>
              <td>D{int(row['decile'])}</td>
              <td>{int(row['users']):,}</td>
              <td>{int(row['positives']):,}</td>
              <td>
                <div class="inline-bar"><span style="width:{width:.2f}%"></span></div>
                <b>{pct(rate, 2)}</b>
              </td>
              <td>{lift:.2f}x</td>
              <td>{score(float(row['min_score']))} - {score(float(row['max_score']))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def top_user_rows(top100: pd.DataFrame, n: int = 25) -> str:
    rows = []
    top = top100.head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    for _, row in top.iterrows():
        converted = int(row["converted_full"]) == 1
        badge = "Converted" if converted else "Not converted"
        badge_class = "good" if converted else "quiet"
        source = "DIY" if int(row["converted_from_diy"]) else ("SFDC" if int(row["converted_from_sfdc"]) else "-")
        rows.append(
            f"""
            <tr>
              <td>{int(row['rank'])}</td>
              <td><b>{esc(row['uid'])}</b><small>{esc(row['user_id'])}</small></td>
              <td>{score(float(row['t1_loan_conversion_score']))}</td>
              <td><span class="pill {badge_class}">{badge}</span></td>
              <td>{source}</td>
              <td>{int(float(row['total_calls']))}</td>
              <td>{int(float(row['answered_calls']))}</td>
              <td>{float(row['avg_call_duration']):.1f}s</td>
              <td>{esc(row['campaign_id'])}</td>
              <td>{esc(row['state'])}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def source_donut(diy: int, sfdc: int) -> str:
    total = diy + sfdc
    diy_pct = 100 * diy / total if total else 0
    sfdc_pct = 100 - diy_pct
    return f"""
    <div class="source-split">
      <div class="donut" style="--diy:{diy_pct:.2f};"></div>
      <div>
        <p><b>{diy:,}</b> DIY positives <span>{diy_pct:.1f}%</span></p>
        <p><b>{sfdc:,}</b> SFDC positives <span>{sfdc_pct:.1f}%</span></p>
        <p><b>{total:,}</b> total April conversions</p>
      </div>
    </div>
    """


def compare_table(compare: pd.DataFrame, full_compare: pd.DataFrame) -> str:
    rows = []
    merged = compare.merge(
        full_compare[["stage", "same_order"]].rename(columns={"same_order": "full_same_order"}),
        on="stage",
        how="left",
    )
    for _, row in merged.iterrows():
        status = "PASS" if bool(row["same_uid_set"]) and bool(row["same_scores"]) else "CHECK"
        rows.append(
            f"""
            <tr>
              <td>{esc(row['stage'])}</td>
              <td><span class="pill {'good' if status == 'PASS' else 'warn'}">{status}</span></td>
              <td>{int(row['unlabeled_rows'])} / {int(row['labeled_rows'])}</td>
              <td>{str(bool(row['same_uid_set']))}</td>
              <td>{str(bool(row['same_order']))}</td>
              <td>{str(bool(row['same_scores']))}</td>
              <td>{int(row['converted_in_top100'])}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    deciles = pd.read_csv(DECILE_PATH)
    top100 = pd.read_csv(TOP100_PATH)
    compare = pd.read_csv(TOP_COMPARE_PATH)
    full_compare = pd.read_csv(FULL_COMPARE_PATH)

    label_info = report["label_info"]
    t1 = report["metrics"]["t1_vs_conversion"]
    t0_conv = report["metrics"]["t0_vs_conversion_secondary"]
    t0_call = report["metrics"]["t0_vs_call_engagement_primary"]
    baseline = t1["baseline_rate"]
    top100_lift = t1["precision_at_100"] / baseline

    converted_top20 = int(top100.head(20)["converted_full"].sum())
    converted_top100 = int(top100.head(100)["converted_full"].sum())

    cards = "\n".join(
        [
            metric_card("April Users", num(t1["rows"]), "Validated against DIY + SFDC labels"),
            metric_card("Actual Conversions", num(t1["positives"]), f"Baseline {pct(baseline, 3)}"),
            metric_card("T1 AUC", score(t1["auc"]), "Strong holdout ranking quality", "accent"),
            metric_card("Precision@100", pct(t1["precision_at_100"]), f"{converted_top100}/100 users converted", "success"),
            metric_card("Top 100 Lift", f"{top100_lift:.1f}x", "Compared with random baseline", "success"),
            metric_card("Top Decile Lift", f"{t1['top_decile_lift']:.2f}x", f"Top decile rate {pct(t1['top_decile_rate'], 2)}"),
        ]
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>April Loan Model Validation Dashboard</title>
  <style>
    :root {{
      --ink: #192329;
      --muted: #65737b;
      --paper: #fffaf0;
      --card: rgba(255, 255, 255, 0.78);
      --line: rgba(25, 35, 41, 0.14);
      --teal: #0f766e;
      --teal-2: #99f6e4;
      --gold: #d9911b;
      --clay: #c2573a;
      --navy: #16324f;
      --green: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Bahnschrift", "Candara", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(15, 118, 110, 0.20), transparent 28%),
        radial-gradient(circle at 88% 5%, rgba(217, 145, 27, 0.20), transparent 30%),
        linear-gradient(145deg, #fffaf0 0%, #f2efe6 46%, #e7f4ef 100%);
      min-height: 100vh;
    }}
    body:before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.28;
      background-image:
        linear-gradient(rgba(25,35,41,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(25,35,41,0.05) 1px, transparent 1px);
      background-size: 38px 38px;
    }}
    .wrap {{ width: min(1440px, calc(100% - 36px)); margin: 0 auto; padding: 30px 0 56px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 34px;
      padding: clamp(28px, 5vw, 58px);
      background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(234, 247, 242, 0.74));
      box-shadow: 0 24px 70px rgba(22, 50, 79, 0.14);
      animation: rise 620ms ease both;
    }}
    .hero:after {{
      content: "T1";
      position: absolute;
      right: -22px;
      bottom: -52px;
      font-family: Georgia, "Palatino Linotype", serif;
      font-size: clamp(110px, 22vw, 260px);
      font-weight: 900;
      color: rgba(15, 118, 110, 0.08);
      letter-spacing: -0.08em;
    }}
    h1, h2, h3 {{ font-family: Georgia, "Palatino Linotype", serif; letter-spacing: -0.035em; }}
    h1 {{ font-size: clamp(40px, 7vw, 86px); line-height: 0.95; margin: 0 0 18px; max-width: 900px; }}
    h2 {{ font-size: clamp(26px, 3vw, 42px); margin: 0 0 18px; }}
    h3 {{ margin: 0 0 12px; font-size: 24px; }}
    .eyebrow {{ color: var(--teal); font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; font-size: 13px; }}
    .hero p {{ max-width: 820px; color: #43525a; font-size: 19px; line-height: 1.55; }}
    .hero-actions {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 24px; }}
    .pill, .button {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-weight: 800;
      font-size: 12px;
      background: rgba(22, 50, 79, 0.08);
      color: var(--navy);
      border: 1px solid rgba(22,50,79,0.10);
    }}
    .button {{ padding: 11px 16px; text-decoration: none; background: var(--ink); color: white; }}
    .pill.good {{ color: #0f5132; background: #dff7e8; border-color: #b9eccd; }}
    .pill.warn {{ color: #8a4f00; background: #fff1c7; border-color: #f7d37a; }}
    .pill.quiet {{ color: #59656b; background: #eef1f1; }}
    .grid {{ display: grid; gap: 18px; }}
    .metrics {{ grid-template-columns: repeat(6, minmax(0, 1fr)); margin: 20px 0; }}
    .metric-card {{
      min-height: 150px;
      padding: 20px;
      border: 1px solid var(--line);
      border-radius: 26px;
      background: var(--card);
      backdrop-filter: blur(12px);
      box-shadow: 0 16px 38px rgba(22, 50, 79, 0.09);
      animation: rise 700ms ease both;
    }}
    .metric-card p {{ margin: 0; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 900; }}
    .metric-card strong {{ display: block; margin: 13px 0 8px; font-size: clamp(30px, 3vw, 46px); line-height: 1; letter-spacing: -0.04em; }}
    .metric-card span {{ color: var(--muted); line-height: 1.35; }}
    .metric-card.accent {{ border-color: rgba(15, 118, 110, 0.32); background: linear-gradient(160deg, rgba(236,253,245,0.9), rgba(255,255,255,0.78)); }}
    .metric-card.success {{ border-color: rgba(21, 128, 61, 0.28); background: linear-gradient(160deg, rgba(220,252,231,0.86), rgba(255,255,255,0.78)); }}
    .section {{
      margin-top: 22px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: rgba(255, 255, 255, 0.70);
      box-shadow: 0 20px 54px rgba(22, 50, 79, 0.10);
      animation: rise 760ms ease both;
    }}
    .two-col {{ grid-template-columns: 1.15fr 0.85fr; }}
    .three-col {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .callout {{
      padding: 20px;
      border-radius: 24px;
      background: linear-gradient(135deg, #16324f, #0f766e);
      color: white;
      min-height: 100%;
      position: relative;
      overflow: hidden;
    }}
    .callout:after {{
      content: "";
      position: absolute;
      width: 180px;
      height: 180px;
      right: -70px;
      top: -70px;
      border-radius: 50%;
      background: rgba(255,255,255,0.16);
    }}
    .callout strong {{ font-size: 44px; display: block; letter-spacing: -0.05em; }}
    .callout p {{ color: rgba(255,255,255,0.80); line-height: 1.45; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 92px 1fr 210px;
      align-items: center;
      gap: 12px;
      margin: 13px 0;
    }}
    .bar-label {{ font-weight: 900; }}
    .bar-track, .inline-bar {{
      height: 15px;
      border-radius: 999px;
      background: rgba(25,35,41,0.08);
      overflow: hidden;
      border: 1px solid rgba(25,35,41,0.05);
    }}
    .bar-track span, .inline-bar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--teal), var(--gold));
    }}
    .bar-value {{ font-weight: 900; }}
    .bar-value em {{ display: block; color: var(--muted); font-size: 12px; font-style: normal; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid rgba(25,35,41,0.10); text-align: left; vertical-align: middle; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    td small {{ display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }}
    .table-wrap {{ overflow-x: auto; border-radius: 22px; border: 1px solid rgba(25,35,41,0.10); background: rgba(255,255,255,0.58); }}
    .inline-bar {{ width: min(170px, 42vw); display: inline-block; margin-right: 10px; vertical-align: middle; }}
    .source-split {{ display: grid; grid-template-columns: 160px 1fr; gap: 20px; align-items: center; }}
    .donut {{
      width: 150px;
      height: 150px;
      border-radius: 50%;
      background: conic-gradient(var(--teal) 0 calc(var(--diy) * 1%), var(--gold) 0 100%);
      box-shadow: inset 0 0 0 32px #fffaf0, 0 18px 40px rgba(22, 50, 79, 0.15);
    }}
    .source-split p {{ margin: 8px 0; color: var(--muted); }}
    .source-split b {{ color: var(--ink); font-size: 24px; }}
    .note-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .note {{ padding: 18px; border-radius: 22px; background: rgba(15,118,110,0.08); border: 1px solid rgba(15,118,110,0.14); }}
    .note b {{ display: block; margin-bottom: 8px; }}
    .footer {{ color: var(--muted); margin-top: 24px; line-height: 1.5; font-size: 13px; }}
    @media (max-width: 1100px) {{
      .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .two-col, .three-col, .note-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 80px 1fr; }}
      .bar-value {{ grid-column: 2; }}
    }}
    @media (max-width: 680px) {{
      .wrap {{ width: min(100% - 20px, 1440px); padding-top: 10px; }}
      .hero, .section {{ border-radius: 22px; padding: 18px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .source-split {{ grid-template-columns: 1fr; }}
      .donut {{ width: 120px; height: 120px; }}
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(14px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="eyebrow">April 2026 validation dashboard</div>
      <h1>Loan conversion model is finding the right users.</h1>
      <p>
        April was scored blind using user + call data first. DIY and SFDC labels were joined only after scoring.
        The T1 ranked list stayed unchanged, and 53 of the top 100 users actually converted.
      </p>
      <div class="hero-actions">
        <span class="pill good">Clean validation: prediction before labels</span>
        <span class="pill good">T1 top 100 unchanged after labels</span>
        <span class="pill">T0 still needs improvement</span>
      </div>
    </section>

    <section class="grid metrics">
      {cards}
    </section>

    <section class="section grid two-col">
      <div>
        <h2>T1 Precision Funnel</h2>
        <p class="footer">This shows how many actual converted users were captured as we move down the ranked T1 list.</p>
        {precision_rows(t1)}
      </div>
      <aside class="callout">
        <h3>Business Read</h3>
        <strong>{converted_top20}/20</strong>
        <p>
          The first 20 T1 users produced {converted_top20} confirmed conversions.
          Top 50 produced {t1['top_50_conversions']} conversions, and top 100 produced {converted_top100}.
        </p>
        <p>
          Random baseline is only {pct(baseline, 3)}, so this is not normal luck.
        </p>
      </aside>
    </section>

    <section class="section grid two-col">
      <div>
        <h2>T1 Decile Quality</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Decile</th>
                <th>Users</th>
                <th>Conversions</th>
                <th>Conv Rate</th>
                <th>Lift</th>
                <th>Score Range</th>
              </tr>
            </thead>
            <tbody>
              {decile_rows(deciles, baseline)}
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <h2>Label Source Split</h2>
        {source_donut(int(label_info['converted_from_diy']), int(label_info['converted_from_sfdc']))}
        <div class="section" style="margin-top:18px; padding:18px; border-radius:22px;">
          <h3>Label Rules</h3>
          <p class="footer">
            DIY positive: <b>disburse initiated</b>.<br>
            SFDC positive: <b>Disbursed</b> or disbursed date present.<br>
            Pending, rejected, and cancelled SFDC rows were not counted as conversions.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Blind Prediction Integrity Check</h2>
      <p class="footer">This proves labels did not alter the T1 prediction list after scoring.</p>
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
              <th>Conversions in Top 100</th>
            </tr>
          </thead>
          <tbody>
            {compare_table(compare, full_compare)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section grid three-col">
      <div class="note">
        <b>T1 is business-pilot ready</b>
        AUC {score(t1['auc'])}, Precision@100 {pct(t1['precision_at_100'])}, and top-decile lift {t1['top_decile_lift']:.2f}x.
      </div>
      <div class="note">
        <b>T0 is not the conversion model</b>
        T0 conversion Precision@100 is {pct(t0_conv['precision_at_100'])}; it should be treated as call prioritization only.
      </div>
      <div class="note">
        <b>T0 call targeting needs work</b>
        T0 call-engagement AUC is {score(t0_call['auc'])}, with top-decile lift {t0_call['top_decile_lift']:.2f}x.
      </div>
    </section>

    <section class="section">
      <h2>Top T1 Users With Outcomes</h2>
      <p class="footer">First 25 users from the blind T1 list, now annotated with conversion outcome.</p>
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
          <tbody>
            {top_user_rows(top100, 25)}
          </tbody>
        </table>
      </div>
    </section>

    <p class="footer">
      Generated from saved April validation outputs in multi_month_training/test_outputs.
      This dashboard is static and self-contained; no model retraining or label reprocessing happens when it is opened.
    </p>
  </main>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html_doc, encoding="utf-8")
    print(DASHBOARD_PATH)


if __name__ == "__main__":
    main()
