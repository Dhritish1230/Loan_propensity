import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
APR_USER_PATH = ROOT / "april" / "hero_fincorp_loan_upselling_user_data_2026_april_updated.csv"
APR_DIY_PATH = ROOT / "april" / "DIY_APRIL.csv"
APR_SFDC_PATH = ROOT / "april" / "SFDC_APRIL.csv"
OUT_DIR = ROOT / "multi_month_training" / "test_outputs"

PRED_PATH = OUT_DIR / "april_unlabeled_t1_post_call_predictions.csv"
LABEL_PATH = OUT_DIR / "april_conversion_labels_from_diy_sfdc.csv"
EVAL_PATH = OUT_DIR / "april_t1_evaluated_predictions.csv"
T0_EVAL_PATH = OUT_DIR / "april_t0_evaluated_predictions.csv"
T1_TOP100_PATH = OUT_DIR / "april_t1_top100_with_conversions.csv"
T0_TOP100_PATH = OUT_DIR / "april_t0_top100_with_conversions.csv"
T1_CONVERTED_TOP_BUCKETS_PATH = OUT_DIR / "april_t1_converted_users_in_top_buckets.csv"
DECILE_PATH = OUT_DIR / "april_labeled_decile_tables.csv"
REPORT_PATH = OUT_DIR / "april_labeled_evaluation_report.json"


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def safe_read_sfdc(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", engine="python")


def build_labels() -> tuple[pd.DataFrame, dict]:
    users = pd.read_csv(APR_USER_PATH, usecols=["uid"], low_memory=False)
    users.columns = users.columns.str.strip()
    users["uid"] = normalize_uid(users["uid"])
    users = users.loc[users["uid"].str.match(r"^CSD-\d+$", na=False)].drop_duplicates("uid")
    user_ids = set(users["uid"])

    diy = pd.read_csv(APR_DIY_PATH, encoding="latin1", engine="python")
    diy.columns = diy.columns.str.strip()
    diy["uid"] = normalize_uid(diy["csd_id"])
    diy["stage_clean"] = diy["stage"].astype(str).str.strip().str.lower()
    diy_valid = diy.loc[diy["uid"].isin(user_ids)].copy()
    diy_positive = diy_valid.loc[diy_valid["stage_clean"].eq("disburse initiated"), "uid"]
    diy_broad_positive = diy_valid.loc[
        diy_valid["stage_clean"].isin(["disburse initiated", "congratulation"]),
        "uid",
    ]

    sfdc = safe_read_sfdc(APR_SFDC_PATH)
    sfdc.columns = sfdc.columns.str.strip()
    sfdc["uid"] = normalize_uid(sfdc["Lead Id"])
    sfdc["stage_clean"] = sfdc["LPL Loan: Stage"].astype(str).str.strip().str.lower()
    sfdc["sub_stage_clean"] = sfdc["LPL Loan: Sub Stage"].astype(str).str.strip().str.lower()
    disbursed_date = pd.to_datetime(sfdc["LPL Loan: Loan Disbursed Date"], errors="coerce")
    sfdc_valid = sfdc.loc[sfdc["uid"].isin(user_ids)].copy()
    sfdc_disbursed_date = pd.to_datetime(sfdc_valid["LPL Loan: Loan Disbursed Date"], errors="coerce")
    sfdc_positive = sfdc_valid.loc[
        sfdc_valid["stage_clean"].eq("disbursed") | sfdc_disbursed_date.notna(),
        "uid",
    ]

    diy_positive_ids = set(diy_positive.dropna().unique())
    diy_broad_positive_ids = set(diy_broad_positive.dropna().unique())
    sfdc_positive_ids = set(sfdc_positive.dropna().unique())

    labels = users.copy()
    labels["converted_from_diy"] = labels["uid"].isin(diy_positive_ids).astype(int)
    labels["converted_from_diy_broad"] = labels["uid"].isin(diy_broad_positive_ids).astype(int)
    labels["converted_from_sfdc"] = labels["uid"].isin(sfdc_positive_ids).astype(int)
    labels["converted_full"] = (
        labels["converted_from_diy"].eq(1) | labels["converted_from_sfdc"].eq(1)
    ).astype(int)
    labels["converted_full_broad_diy"] = (
        labels["converted_from_diy_broad"].eq(1) | labels["converted_from_sfdc"].eq(1)
    ).astype(int)

    info = {
        "user_rows": int(len(users)),
        "diy_rows": int(len(diy)),
        "diy_unique_csd": int(diy["uid"].nunique()),
        "diy_linked_rows": int(len(diy_valid)),
        "diy_linked_unique": int(diy_valid["uid"].nunique()),
        "diy_stage_distribution": diy["stage_clean"].value_counts(dropna=False).to_dict(),
        "diy_positive_linked_unique_strict": int(len(diy_positive_ids)),
        "diy_positive_linked_unique_broad": int(len(diy_broad_positive_ids)),
        "sfdc_rows": int(len(sfdc)),
        "sfdc_unique_lead_id": int(sfdc["uid"].nunique()),
        "sfdc_linked_rows": int(len(sfdc_valid)),
        "sfdc_linked_unique": int(sfdc_valid["uid"].nunique()),
        "sfdc_stage_distribution": sfdc["stage_clean"].value_counts(dropna=False).to_dict(),
        "sfdc_sub_stage_distribution": sfdc["sub_stage_clean"].value_counts(dropna=False).to_dict(),
        "sfdc_disbursed_date_non_null_rows": int(disbursed_date.notna().sum()),
        "sfdc_positive_linked_unique": int(len(sfdc_positive_ids)),
        "diy_sfdc_positive_overlap": int(len(diy_positive_ids & sfdc_positive_ids)),
        "converted_from_diy": int(labels["converted_from_diy"].sum()),
        "converted_from_sfdc": int(labels["converted_from_sfdc"].sum()),
        "converted_full": int(labels["converted_full"].sum()),
        "converted_full_broad_diy": int(labels["converted_full_broad_diy"].sum()),
        "conversion_rate": float(labels["converted_full"].mean()),
    }
    return labels, info


def precision_recall_at_k(y: pd.Series, score: pd.Series, k: int) -> dict:
    ranked = pd.DataFrame({"y": y.astype(int).to_numpy(), "score": score.astype(float).to_numpy()})
    top = ranked.sort_values("score", ascending=False).head(k)
    positives = int(y.sum())
    hits = int(top["y"].sum())
    return {
        f"top_{k}_conversions": hits,
        f"precision_at_{k}": float(hits / k),
        f"recall_at_{k}": float(hits / positives) if positives else 0.0,
    }


def metric_block(df: pd.DataFrame, label_col: str, score_col: str) -> dict:
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0).astype(float)
    ranked = df.assign(_y=y, _score=score).sort_values("_score", ascending=False)
    baseline = float(y.mean())
    decile = decile_table(df, label_col, score_col)
    top_decile_rate = float(decile.iloc[0]["positive_rate"]) if len(decile) else 0.0

    metrics = {
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "baseline_rate": baseline,
        "auc": float(roc_auc_score(y, score)) if y.nunique() == 2 else None,
        "top_decile_rate": top_decile_rate,
        "top_decile_lift": float(top_decile_rate / baseline) if baseline else None,
        "top_20_score_min": float(ranked.head(20)["_score"].min()),
        "top_100_score_min": float(ranked.head(100)["_score"].min()),
        "top_500_score_min": float(ranked.head(500)["_score"].min()),
        "top_1000_score_min": float(ranked.head(1000)["_score"].min()),
    }
    for k in [20, 50, 100, 500, 1000, 5000, 10000]:
        metrics.update(precision_recall_at_k(y, score, k))
    return metrics


def decile_table(df: pd.DataFrame, label_col: str, score_col: str) -> pd.DataFrame:
    out = df[[label_col, score_col]].copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0)
    out["decile"] = pd.qcut(out[score_col].rank(method="first"), 10, labels=False)
    table = (
        out.groupby("decile")
        .agg(
            users=(label_col, "size"),
            positives=(label_col, "sum"),
            positive_rate=(label_col, "mean"),
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            mean_score=(score_col, "mean"),
        )
        .sort_index(ascending=False)
        .reset_index()
    )
    table.insert(0, "score_col", score_col)
    table.insert(1, "label_col", label_col)
    return table


def top_converted_rows(df: pd.DataFrame, score_col: str, max_rank: int) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    cols = [
        "rank",
        "uid",
        "user_id",
        score_col,
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
    cols = [col for col in cols if col in ranked.columns]
    return ranked.loc[ranked["rank"].le(max_rank) & ranked["converted_full"].eq(1), cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels, label_info = build_labels()
    labels.to_csv(LABEL_PATH, index=False)

    pred = pd.read_csv(PRED_PATH, low_memory=False)
    pred.columns = pred.columns.str.strip()
    pred["uid"] = normalize_uid(pred["uid"])
    labels["uid"] = normalize_uid(labels["uid"])
    df = pred.merge(labels, on="uid", how="left")
    for col in [
        "converted_from_diy",
        "converted_from_sfdc",
        "converted_full",
        "converted_from_diy_broad",
        "converted_full_broad_diy",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["call_engaged_10s"] = (
        (pd.to_numeric(df["answered_calls"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(df["avg_call_duration"], errors="coerce").fillna(0) >= 10)
    ).astype(int)

    df.sort_values("t1_loan_conversion_score", ascending=False).to_csv(EVAL_PATH, index=False)
    df.sort_values("t0_call_targeting_score", ascending=False).to_csv(T0_EVAL_PATH, index=False)
    df.sort_values("t1_loan_conversion_score", ascending=False).head(100).to_csv(T1_TOP100_PATH, index=False)
    df.sort_values("t0_call_targeting_score", ascending=False).head(100).to_csv(T0_TOP100_PATH, index=False)

    converted_buckets = []
    for k in [100, 500, 1000]:
        bucket = top_converted_rows(df, "t1_loan_conversion_score", k)
        bucket.insert(0, "bucket", f"top_{k}")
        converted_buckets.append(bucket)
    pd.concat(converted_buckets, ignore_index=True).to_csv(T1_CONVERTED_TOP_BUCKETS_PATH, index=False)

    deciles = pd.concat(
        [
            decile_table(df, "converted_full", "t1_loan_conversion_score"),
            decile_table(df, "converted_full", "t0_call_targeting_score"),
            decile_table(df, "call_engaged_10s", "t0_call_targeting_score"),
        ],
        ignore_index=True,
    )
    deciles.to_csv(DECILE_PATH, index=False)

    report = {
        "label_policy": {
            "diy_positive": "stage == 'disburse initiated'",
            "sfdc_positive": "LPL Loan: Stage == 'Disbursed' OR LPL Loan: Loan Disbursed Date present",
            "combined_label": "converted_full = converted_from_diy OR converted_from_sfdc",
            "broad_diy_check": "Also counted DIY 'congratulation' as positive separately; it changes only the broad label.",
        },
        "label_info": label_info,
        "metrics": {
            "t1_vs_conversion": metric_block(df, "converted_full", "t1_loan_conversion_score"),
            "t1_vs_conversion_broad_diy": metric_block(df, "converted_full_broad_diy", "t1_loan_conversion_score"),
            "t0_vs_conversion_secondary": metric_block(df, "converted_full", "t0_call_targeting_score"),
            "t0_vs_call_engagement_primary": metric_block(df, "call_engaged_10s", "t0_call_targeting_score"),
        },
        "outputs": {
            "labels": str(LABEL_PATH),
            "t1_evaluated_predictions": str(EVAL_PATH),
            "t0_evaluated_predictions": str(T0_EVAL_PATH),
            "t1_top100": str(T1_TOP100_PATH),
            "t0_top100": str(T0_TOP100_PATH),
            "t1_converted_top_buckets": str(T1_CONVERTED_TOP_BUCKETS_PATH),
            "deciles": str(DECILE_PATH),
            "report": str(REPORT_PATH),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
