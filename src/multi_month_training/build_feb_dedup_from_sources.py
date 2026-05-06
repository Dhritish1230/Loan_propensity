from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
USER_PATH = ROOT / "feb" / "hero_fincorp_loan_upselling_user_data_feb_2026_updated.csv"
CALL_PATH = ROOT / "feb" / "hero_fincorp_loan_upselling_call_data_feb_2026_updated.csv"
LABEL_PATH = ROOT / "feb" / "FEB_CONVERSION_FROM_DIY_SFDC.csv"
OUTPUT_PATH = ROOT / "multi_month_training" / "dedup_cache" / "FEB_dedup.csv"


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def build_call_agg(chunksize: int = 500_000) -> pd.DataFrame:
    totals = defaultdict(float)
    outbound = defaultdict(float)
    answered = defaultdict(float)
    duration_sum = defaultdict(float)

    usecols = ["call_id", "user_id", "call_duration", "call_type", "call_status"]
    for chunk in pd.read_csv(
        CALL_PATH,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        index_col=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        user_id = chunk["user_id"].fillna("").astype(str).str.strip()
        valid = user_id.ne("") & user_id.ne("nan") & user_id.ne("0")
        chunk = chunk.loc[valid].copy()
        user_id = user_id.loc[valid]

        duration = pd.to_numeric(chunk["call_duration"], errors="coerce").fillna(0).clip(lower=0)
        call_type = chunk["call_type"].fillna("").astype(str).str.lower().str.strip()
        call_status = chunk["call_status"].fillna("").astype(str).str.lower().str.strip()

        for key, value in user_id.value_counts().items():
            totals[key] += float(value)
        for key, value in call_type.eq("outbound").groupby(user_id).sum().items():
            outbound[key] += float(value)
        for key, value in call_status.eq("answered").groupby(user_id).sum().items():
            answered[key] += float(value)
        for key, value in duration.groupby(user_id).sum().items():
            duration_sum[key] += float(value)

    rows = []
    for user_id in sorted(totals):
        total = totals[user_id]
        rows.append(
            {
                "user_id": user_id,
                "total_calls": total,
                "outbound_calls": outbound[user_id],
                "answered_calls": answered[user_id],
                "avg_call_duration": duration_sum[user_id] / total if total else 0.0,
                "answered_rate": answered[user_id] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    user_df = pd.read_csv(USER_PATH, low_memory=False)
    label_df = pd.read_csv(LABEL_PATH, low_memory=False)
    call_agg = build_call_agg()

    user_df.columns = user_df.columns.str.strip()
    label_df.columns = label_df.columns.str.strip()

    user_df["uid"] = normalize_uid(user_df["uid"])
    label_df["uid"] = normalize_uid(label_df["uid"])
    user_df = user_df.loc[user_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()

    label_df = (
        label_df.loc[label_df["uid"].str.match(r"^CSD-\d+$", na=False)]
        .groupby("uid", as_index=False)
        .agg(converted_full=("converted_full", "max"))
    )

    merged = user_df.merge(label_df, on="uid", how="left")
    merged["converted"] = pd.to_numeric(merged["converted_full"], errors="coerce").fillna(0).astype(int)
    merged["user_id"] = merged["user_id"].astype(str).str.strip()
    call_agg["user_id"] = call_agg["user_id"].astype(str).str.strip()
    merged = merged.merge(call_agg, on="user_id", how="left")

    for col in ["age", "decile", "total_calls", "answered_calls", "answered_rate", "avg_call_duration"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["language"] = merged["language"].fillna("unknown").astype(str).str.strip().str.lower()
    merged["state"] = merged["state"].fillna("unknown").astype(str).str.strip().str.upper()
    merged["flow_phase"] = merged["flow_phase"].fillna("unknown").astype(str).str.strip()

    out = (
        merged.groupby("uid", as_index=False)
        .agg(
            language=("language", mode_or_unknown),
            user_id=("user_id", mode_or_unknown),
            state=("state", mode_or_unknown),
            flow_phase=("flow_phase", mode_or_unknown),
            age=("age", "median"),
            decile=("decile", "median"),
            total_calls=("total_calls", "max"),
            answered_calls=("answered_calls", "max"),
            avg_call_duration=("avg_call_duration", "mean"),
            converted=("converted", "max"),
        )
    )
    out["age"] = out["age"].fillna(out["age"].median())
    out["decile"] = out["decile"].fillna(out["decile"].median())
    out["total_calls"] = out["total_calls"].fillna(0).clip(lower=0)
    out["answered_calls"] = out["answered_calls"].fillna(0).clip(lower=0)
    out["avg_call_duration"] = out["avg_call_duration"].fillna(0).clip(lower=0)
    out["answered_calls"] = out[["answered_calls", "total_calls"]].min(axis=1)
    out["answered_rate"] = np.where(out["total_calls"] > 0, out["answered_calls"] / out["total_calls"], 0)
    out["month"] = "FEB"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {OUTPUT_PATH} rows={len(out):,} conversions={int(out['converted'].sum()):,} "
        f"users_with_calls={int((out['total_calls'] > 0).sum()):,}"
    )


if __name__ == "__main__":
    main()
