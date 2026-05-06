import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\m\Desktop\PROJECT INTERNSHIP")
CALL_PATHS = {
    "OCT": ROOT / "hero_fincorp_loan_upselling_call_data_2025oct.csv",
    "DEC": ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2025_dec_updated.csv",
    "JAN": ROOT / "jan" / "hero_fincorp_loan_upselling_call_data_january_2026_updated.csv",
    "FEB": ROOT / "feb" / "hero_fincorp_loan_upselling_call_data_feb_2026_updated.csv",
    "MAR": ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2026_march_updated.csv",
    "NOV": ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2025_nov_updated.csv",
}
OUT_DIR = ROOT / "multi_month_training" / "sequence_call_cache"

FULL_FIRST_COLUMNS = [
    "call_id",
    "user_id",
    "call_duration",
    "call_type",
    "call_time",
    "call_endtime",
    "created_at",
    "modified_at",
    "call_status",
    "hangup_cause_code",
    "flow",
    "did",
    "language",
]
OCT_COLUMNS = [
    "call_id",
    "user_id",
    "call_duration",
    "call_type",
    "call_status",
    "call_time",
    "flow",
    "language",
]

SEQUENCE_FEATURES = [
    "seq_first_call_hour",
    "seq_last_call_hour",
    "seq_first_call_dayofweek",
    "seq_last_call_dayofweek",
    "seq_call_span_hours",
    "seq_avg_gap_hours",
    "seq_first_call_duration",
    "seq_last_call_duration",
    "seq_first_answer_duration",
    "seq_last_answer_duration",
    "seq_hours_to_first_answer",
    "seq_answered_span_hours",
    "seq_business_hour_share",
    "seq_weekend_call_share",
    "seq_mean_call_hour",
    "seq_std_call_hour",
    "seq_first_call_answered",
    "seq_last_call_answered",
]


def read_chunks(path: Path, month: str, chunksize: int):
    if month == "OCT":
        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            index_col=False,
            usecols=lambda col: col in OCT_COLUMNS,
        ):
            yield chunk
    else:
        # Later call exports have an unquoted JSON context field with commas.
        # Reading fixed leading columns by position prevents context from
        # shifting user_id/call_status/timestamp columns.
        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            index_col=False,
            usecols=list(range(len(FULL_FIRST_COLUMNS))),
        ):
            chunk.columns = FULL_FIRST_COLUMNS
            yield chunk[["call_id", "user_id", "call_duration", "call_type", "call_status", "call_time"]]


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def update_record(records: dict, user_id: str, row: dict) -> None:
    rec = records[user_id]
    total_calls = row["total_calls"]
    rec["total_calls"] += total_calls
    rec["hour_sum"] += row["hour_sum"]
    rec["hour_sq_sum"] += row["hour_sq_sum"]
    rec["business_hour_calls"] += row["business_hour_calls"]
    rec["weekend_calls"] += row["weekend_calls"]

    first_ts = row["first_ts"]
    if pd.notna(first_ts) and (pd.isna(rec["first_ts"]) or first_ts < rec["first_ts"]):
        rec["first_ts"] = first_ts
        rec["first_duration"] = row["first_duration"]
        rec["first_answered"] = row["first_answered"]

    last_ts = row["last_ts"]
    if pd.notna(last_ts) and (pd.isna(rec["last_ts"]) or last_ts > rec["last_ts"]):
        rec["last_ts"] = last_ts
        rec["last_duration"] = row["last_duration"]
        rec["last_answered"] = row["last_answered"]

    first_answer_ts = row["first_answer_ts"]
    if pd.notna(first_answer_ts) and (
        pd.isna(rec["first_answer_ts"]) or first_answer_ts < rec["first_answer_ts"]
    ):
        rec["first_answer_ts"] = first_answer_ts
        rec["first_answer_duration"] = row["first_answer_duration"]

    last_answer_ts = row["last_answer_ts"]
    if pd.notna(last_answer_ts) and (
        pd.isna(rec["last_answer_ts"]) or last_answer_ts > rec["last_answer_ts"]
    ):
        rec["last_answer_ts"] = last_answer_ts
        rec["last_answer_duration"] = row["last_answer_duration"]


def aggregate(month: str, input_path: Path, output_path: Path, chunksize: int) -> None:
    def blank_record():
        return {
            "total_calls": 0.0,
            "hour_sum": 0.0,
            "hour_sq_sum": 0.0,
            "business_hour_calls": 0.0,
            "weekend_calls": 0.0,
            "first_ts": pd.NaT,
            "last_ts": pd.NaT,
            "first_duration": 0.0,
            "last_duration": 0.0,
            "first_answered": 0.0,
            "last_answered": 0.0,
            "first_answer_ts": pd.NaT,
            "last_answer_ts": pd.NaT,
            "first_answer_duration": 0.0,
            "last_answer_duration": 0.0,
        }

    records = defaultdict(blank_record)

    for chunk in read_chunks(input_path, month, chunksize):
        chunk.columns = chunk.columns.str.strip()
        if "user_id" not in chunk.columns:
            raise ValueError(f"user_id column missing in {input_path}")

        chunk["user_id"] = chunk["user_id"].fillna("").astype(str).str.strip()
        valid = chunk["user_id"].ne("") & chunk["user_id"].ne("nan") & chunk["user_id"].ne("0")
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue

        chunk["call_ts"] = to_datetime(chunk["call_time"])
        chunk = chunk.loc[chunk["call_ts"].notna()].copy()
        if chunk.empty:
            continue

        chunk["duration"] = pd.to_numeric(chunk["call_duration"], errors="coerce").fillna(0).clip(lower=0)
        chunk["answered"] = (
            chunk["call_status"].fillna("").astype(str).str.strip().str.upper().eq("ANSWERED").astype(int)
        )
        chunk["hour"] = chunk["call_ts"].dt.hour.astype(float)
        chunk["dayofweek"] = chunk["call_ts"].dt.dayofweek.astype(float)
        chunk["business_hour"] = chunk["hour"].between(9, 18, inclusive="left").astype(int)
        chunk["weekend"] = chunk["dayofweek"].isin([5, 6]).astype(int)

        grouped = (
            chunk.groupby("user_id", as_index=False)
            .agg(
                total_calls=("call_id", "count"),
                hour_sum=("hour", "sum"),
                hour_sq_sum=("hour", lambda s: float(np.square(s).sum())),
                business_hour_calls=("business_hour", "sum"),
                weekend_calls=("weekend", "sum"),
            )
            .set_index("user_id")
        )

        ordered = chunk.sort_values(["user_id", "call_ts"])
        first = ordered.drop_duplicates("user_id", keep="first").set_index("user_id")
        last = ordered.drop_duplicates("user_id", keep="last").set_index("user_id")

        answered = ordered.loc[ordered["answered"].eq(1)]
        if not answered.empty:
            first_answer = answered.drop_duplicates("user_id", keep="first").set_index("user_id")
            last_answer = answered.drop_duplicates("user_id", keep="last").set_index("user_id")
        else:
            first_answer = pd.DataFrame()
            last_answer = pd.DataFrame()

        for user_id, agg_row in grouped.iterrows():
            row = {
                "total_calls": float(agg_row["total_calls"]),
                "hour_sum": float(agg_row["hour_sum"]),
                "hour_sq_sum": float(agg_row["hour_sq_sum"]),
                "business_hour_calls": float(agg_row["business_hour_calls"]),
                "weekend_calls": float(agg_row["weekend_calls"]),
                "first_ts": first.at[user_id, "call_ts"],
                "first_duration": float(first.at[user_id, "duration"]),
                "first_answered": float(first.at[user_id, "answered"]),
                "last_ts": last.at[user_id, "call_ts"],
                "last_duration": float(last.at[user_id, "duration"]),
                "last_answered": float(last.at[user_id, "answered"]),
                "first_answer_ts": pd.NaT,
                "first_answer_duration": 0.0,
                "last_answer_ts": pd.NaT,
                "last_answer_duration": 0.0,
            }
            if not first_answer.empty and user_id in first_answer.index:
                row["first_answer_ts"] = first_answer.at[user_id, "call_ts"]
                row["first_answer_duration"] = float(first_answer.at[user_id, "duration"])
            if not last_answer.empty and user_id in last_answer.index:
                row["last_answer_ts"] = last_answer.at[user_id, "call_ts"]
                row["last_answer_duration"] = float(last_answer.at[user_id, "duration"])
            update_record(records, user_id, row)

    rows = []
    for user_id, rec in records.items():
        total = rec["total_calls"]
        first_ts = rec["first_ts"]
        last_ts = rec["last_ts"]
        first_ans = rec["first_answer_ts"]
        last_ans = rec["last_answer_ts"]

        span_hours = (
            (last_ts - first_ts).total_seconds() / 3600.0
            if pd.notna(first_ts) and pd.notna(last_ts) and last_ts >= first_ts
            else 0.0
        )
        answered_span_hours = (
            (last_ans - first_ans).total_seconds() / 3600.0
            if pd.notna(first_ans) and pd.notna(last_ans) and last_ans >= first_ans
            else 0.0
        )
        hours_to_first_answer = (
            (first_ans - first_ts).total_seconds() / 3600.0
            if pd.notna(first_ts) and pd.notna(first_ans) and first_ans >= first_ts
            else -1.0
        )

        mean_hour = rec["hour_sum"] / total if total else 0.0
        hour_variance = (rec["hour_sq_sum"] / total) - mean_hour**2 if total else 0.0
        rows.append(
            {
                "user_id": user_id,
                "seq_first_call_hour": float(first_ts.hour) if pd.notna(first_ts) else 0.0,
                "seq_last_call_hour": float(last_ts.hour) if pd.notna(last_ts) else 0.0,
                "seq_first_call_dayofweek": float(first_ts.dayofweek) if pd.notna(first_ts) else 0.0,
                "seq_last_call_dayofweek": float(last_ts.dayofweek) if pd.notna(last_ts) else 0.0,
                "seq_call_span_hours": span_hours,
                "seq_avg_gap_hours": span_hours / (total - 1.0) if total > 1 else 0.0,
                "seq_first_call_duration": rec["first_duration"],
                "seq_last_call_duration": rec["last_duration"],
                "seq_first_answer_duration": rec["first_answer_duration"],
                "seq_last_answer_duration": rec["last_answer_duration"],
                "seq_hours_to_first_answer": hours_to_first_answer,
                "seq_answered_span_hours": answered_span_hours,
                "seq_business_hour_share": rec["business_hour_calls"] / total if total else 0.0,
                "seq_weekend_call_share": rec["weekend_calls"] / total if total else 0.0,
                "seq_mean_call_hour": mean_hour,
                "seq_std_call_hour": float(np.sqrt(max(hour_variance, 0.0))),
                "seq_first_call_answered": rec["first_answered"],
                "seq_last_call_answered": rec["last_answered"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"{month}: saved sequence call features {output_path} rows={len(rows):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=sorted(CALL_PATHS), required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    aggregate(
        args.month,
        CALL_PATHS[args.month],
        OUT_DIR / f"{args.month}_sequence_call_features.csv",
        args.chunksize,
    )


if __name__ == "__main__":
    main()
