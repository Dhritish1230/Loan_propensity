import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_month_dedup_from_sources import MONTH_CONFIG, mode_or_unknown, normalize_uid


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
CALL_AGG_DIR = ROOT / "multi_month_training" / "call_agg_cache"
OUTPUT_DIR = ROOT / "multi_month_training" / "dedup_cache"


def load_labels(conf: dict) -> pd.DataFrame:
    label_df = pd.read_csv(conf["label_path"], low_memory=False)
    label_df.columns = label_df.columns.str.strip()
    label_df["uid"] = normalize_uid(label_df["uid"])
    label_df = label_df.loc[label_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    label_col = conf["label_col"]
    label_df[label_col] = pd.to_numeric(label_df[label_col], errors="coerce").fillna(0).astype(int)
    return label_df.groupby("uid", as_index=False).agg(converted=(label_col, "max"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=sorted(MONTH_CONFIG), required=True)
    args = parser.parse_args()

    conf = MONTH_CONFIG[args.month]
    user_df = pd.read_csv(conf["user_path"], low_memory=False)
    label_df = load_labels(conf)
    call_agg = pd.read_csv(CALL_AGG_DIR / f"{args.month}_call_agg.csv", low_memory=False)

    user_df.columns = user_df.columns.str.strip()
    user_df["uid"] = normalize_uid(user_df["uid"])
    user_df = user_df.loc[user_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    user_df["user_id"] = user_df["user_id"].astype(str).str.strip()
    user_df = user_df.drop_duplicates(subset=["uid"], keep="first").copy()
    call_agg["user_id"] = call_agg["user_id"].astype(str).str.strip()

    merged = user_df.merge(label_df, on="uid", how="left")
    merged["converted"] = pd.to_numeric(merged["converted"], errors="coerce").fillna(0).astype(int)
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
    out["month"] = args.month

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.month}_dedup.csv"
    out.to_csv(output_path, index=False)
    print(
        f"{args.month}: rows={len(out):,}, conversions={int(out['converted'].sum()):,}, "
        f"users_with_calls={int((out['total_calls'] > 0).sum()):,}, output={output_path}"
    )


if __name__ == "__main__":
    main()
