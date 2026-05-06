import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


THIS_FILE = Path(__file__).resolve()
MT_DIR = THIS_FILE.parent
SRC_DIR = MT_DIR.parent
REPO_ROOT = SRC_DIR.parent
MODEL_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", REPO_ROOT / "models")).resolve()
OUTPUT_DIR = Path(os.getenv("LOAN_PROPENSITY_OUTPUT_DIR", REPO_ROOT / "dashboard_outputs")).resolve()

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from aggregate_call_sequence_features import SEQUENCE_FEATURES  # noqa: E402
from evaluate_model_slice import RAW_CALL_FEATURES  # noqa: E402


USER_CATEGORICAL_FEATURES = [
    "language",
    "state",
    "campaign_id",
    "flow_phase",
    "scheme",
    "zone",
    "base_type",
    "flow_type",
]
USER_NUMERIC_FEATURES = [
    "age",
    "decile",
    "minimum_loan_amount",
    "maximum_loan_amount",
    "loan_amount_span",
    "loan_amount_ratio",
    "created_day",
    "created_dayofweek",
    "created_month",
    "days_to_expiry",
    "has_expiry_date",
]
USER_FEATURES = USER_CATEGORICAL_FEATURES + USER_NUMERIC_FEATURES
T1_BASE_NUMERIC_FEATURES = [
    "age",
    "decile",
    "total_calls",
    "answered_calls",
    "answered_rate",
    "avg_call_duration",
]

CALL_FIRST_COLUMNS = [
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
CALL_MIN_COLUMNS = ["call_id", "user_id", "call_duration", "call_type", "call_status", "call_time"]
STATUS_VALUES = ["ANSWERED", "NOT_ANSWERED"]
TYPE_VALUES = ["INBOUND", "OUTBOUND"]
FLOW_VALUES = ["personal", "business", "unknown"]
LANGUAGE_VALUES = ["hin", "ka", "te", "ta", "bn", "unknown"]
CONTEXT_TOKENS = ["DND", "PERSONAL", "BUSINESS"]

PREDICTION_COLUMNS = [
    "uid",
    "user_id",
    "campaign_id",
    "state",
    "language",
    "flow_phase",
    "age",
    "decile",
    "minimum_loan_amount",
    "maximum_loan_amount",
    "total_calls",
    "answered_calls",
    "answered_rate",
    "avg_call_duration",
    "raw_total_calls",
    "raw_avg_duration",
    "seq_first_call_hour",
    "seq_last_call_hour",
    "t0_call_targeting_score",
    "t1_loan_conversion_score",
    "t0_call_priority_rank",
    "t1_conversion_rank",
    "recommended_stage",
    "priority_band",
]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_uid(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.upper()


def normalize_upper(series: pd.Series) -> pd.Series:
    return normalize_text(series).replace("", "unknown").str.upper()


def normalize_lower(series: pd.Series) -> pd.Series:
    return normalize_text(series).replace("", "unknown").str.lower()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    parsed_text = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed_text.fillna(parsed_numeric)


def make_unique_column_names(columns) -> list[str]:
    seen = defaultdict(int)
    out = []
    for col in columns:
        name = str(col).strip()
        seen[name] += 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def prepare_user_features_from_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = make_unique_column_names(df.columns)

    uid_col = find_column(df, ["uid", "csd_id", "CSD ID", "lead_id", "Lead Id"])
    if uid_col is None:
        raise ValueError("User file must contain a uid/CSD ID column.")
    if uid_col != "uid":
        df = df.rename(columns={uid_col: "uid"})

    user_id_col = find_column(df, ["user_id", "User Id", "userid", "user uuid"])
    if user_id_col and user_id_col != "user_id":
        df = df.rename(columns={user_id_col: "user_id"})
    if "user_id" not in df.columns:
        df["user_id"] = ""

    df["uid"] = normalize_uid(df["uid"])
    valid_uid = df["uid"].str.match(r"^CSD-\d+$", na=False)
    df = df.loc[valid_uid].copy()

    for col in USER_CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip()
    for col in ["language", "flow_type", "base_type"]:
        if col in df.columns:
            df[col] = df[col].str.lower()
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()

    for col in ["age", "decile", "minimum_loan_amount", "maximum_loan_amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    created_at = parse_dates(df["created_at"]) if "created_at" in df.columns else pd.Series(pd.NaT, index=df.index)
    expiry_date = parse_dates(df["expiry_date"]) if "expiry_date" in df.columns else pd.Series(pd.NaT, index=df.index)

    df["loan_amount_span"] = df["maximum_loan_amount"] - df["minimum_loan_amount"]
    df["loan_amount_ratio"] = df["maximum_loan_amount"] / df["minimum_loan_amount"].where(
        df["minimum_loan_amount"] > 0
    )
    df["created_day"] = created_at.dt.day
    df["created_dayofweek"] = created_at.dt.dayofweek
    df["created_month"] = created_at.dt.month
    df["days_to_expiry"] = (expiry_date - created_at).dt.days
    df["has_expiry_date"] = expiry_date.notna().astype(int)

    meta_cols = [
        col
        for col in [
            "uid",
            "user_id",
            "status",
            "sub_status",
            "campaign_id",
            "state",
            "language",
            "flow_phase",
            "age",
            "decile",
            "minimum_loan_amount",
            "maximum_loan_amount",
        ]
        if col in df.columns
    ]
    meta = df[meta_cols].drop_duplicates("uid", keep="first")
    features = df[["uid"] + USER_FEATURES].drop_duplicates("uid", keep="first").copy()
    features = features.merge(
        meta.drop(columns=[col for col in meta.columns if col in features.columns and col != "uid"]),
        on="uid",
        how="left",
    )

    for col in USER_NUMERIC_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    info = {
        "raw_user_rows": int(len(raw)),
        "valid_csd_rows": int(len(df)),
        "unique_uid": int(df["uid"].nunique()),
        "duplicate_uid_rows": int(df["uid"].duplicated().sum()),
        "feature_rows_after_uid_dedupe": int(len(features)),
        "user_id_available": bool(features["user_id"].astype(str).str.strip().ne("").any()),
    }
    return features, info


def csv_column_count(path: Path) -> int:
    try:
        return len(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return 0


def normalize_call_chunk(chunk: pd.DataFrame, force_first_columns: bool = False) -> pd.DataFrame:
    out = chunk.copy()
    if force_first_columns:
        out.columns = CALL_FIRST_COLUMNS[: len(out.columns)]
    out.columns = make_unique_column_names(out.columns)
    rename_map = {}
    for wanted in CALL_FIRST_COLUMNS:
        found = find_column(out, [wanted])
        if found and found != wanted:
            rename_map[found] = wanted
    if rename_map:
        out = out.rename(columns=rename_map)
    for col in CALL_FIRST_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[CALL_FIRST_COLUMNS]


def iter_call_chunks(path: Path, chunksize: int):
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = normalize_call_chunk(pd.read_excel(path), force_first_columns=False)
        yield df
        return

    column_count = csv_column_count(path)
    if column_count >= len(CALL_FIRST_COLUMNS):
        for chunk in pd.read_csv(
            path,
            low_memory=False,
            chunksize=chunksize,
            usecols=list(range(len(CALL_FIRST_COLUMNS))),
            index_col=False,
        ):
            yield normalize_call_chunk(chunk, force_first_columns=True)
    else:
        for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize, index_col=False):
            yield normalize_call_chunk(chunk, force_first_columns=False)


def add_sum(store: dict, key: str, values: pd.Series) -> None:
    grouped = values.groupby(values.index).sum()
    for user_id, value in grouped.items():
        store[key][user_id] += float(value)


def add_max(store: dict, key: str, values: pd.Series) -> None:
    grouped = values.groupby(values.index).max()
    for user_id, value in grouped.items():
        store[key][user_id] = max(store[key][user_id], float(value))


def blank_sequence_record() -> dict:
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


def update_sequence_record(records: dict, user_id: str, row: dict) -> None:
    rec = records[user_id]
    rec["total_calls"] += row["total_calls"]
    rec["hour_sum"] += row["hour_sum"]
    rec["hour_sq_sum"] += row["hour_sq_sum"]
    rec["business_hour_calls"] += row["business_hour_calls"]
    rec["weekend_calls"] += row["weekend_calls"]

    for prefix in ["first", "last"]:
        ts_key = f"{prefix}_ts"
        current = rec[ts_key]
        incoming = row[ts_key]
        is_better = incoming < current if prefix == "first" and pd.notna(current) else incoming > current
        if pd.notna(incoming) and (pd.isna(current) or is_better):
            rec[ts_key] = incoming
            rec[f"{prefix}_duration"] = row[f"{prefix}_duration"]
            rec[f"{prefix}_answered"] = row[f"{prefix}_answered"]

    for prefix in ["first_answer", "last_answer"]:
        ts_key = f"{prefix}_ts"
        current = rec[ts_key]
        incoming = row[ts_key]
        is_better = incoming < current if prefix == "first_answer" and pd.notna(current) else incoming > current
        if pd.notna(incoming) and (pd.isna(current) or is_better):
            rec[ts_key] = incoming
            rec[f"{prefix}_duration"] = row[f"{prefix}_duration"]


def update_sequence_features(records: dict, chunk: pd.DataFrame, duration: pd.Series, status: pd.Series) -> None:
    seq = chunk[["call_id", "user_id", "call_time"]].copy()
    seq["duration"] = duration.to_numpy()
    seq["answered"] = status.eq("ANSWERED").astype(int).to_numpy()
    seq["call_ts"] = pd.to_datetime(seq["call_time"], errors="coerce")
    seq = seq.loc[seq["call_ts"].notna()].copy()
    if seq.empty:
        return

    seq["hour"] = seq["call_ts"].dt.hour.astype(float)
    seq["dayofweek"] = seq["call_ts"].dt.dayofweek.astype(float)
    seq["business_hour"] = seq["hour"].between(9, 18, inclusive="left").astype(int)
    seq["weekend"] = seq["dayofweek"].isin([5, 6]).astype(int)

    grouped = (
        seq.groupby("user_id", as_index=False)
        .agg(
            total_calls=("call_id", "count"),
            hour_sum=("hour", "sum"),
            hour_sq_sum=("hour", lambda s: float(np.square(s).sum())),
            business_hour_calls=("business_hour", "sum"),
            weekend_calls=("weekend", "sum"),
        )
        .set_index("user_id")
    )
    ordered = seq.sort_values(["user_id", "call_ts"])
    first = ordered.drop_duplicates("user_id", keep="first").set_index("user_id")
    last = ordered.drop_duplicates("user_id", keep="last").set_index("user_id")
    answered = ordered.loc[ordered["answered"].eq(1)]
    first_answer = answered.drop_duplicates("user_id", keep="first").set_index("user_id") if not answered.empty else pd.DataFrame()
    last_answer = answered.drop_duplicates("user_id", keep="last").set_index("user_id") if not answered.empty else pd.DataFrame()

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
        update_sequence_record(records, user_id, row)


def aggregate_call_features(call_path: Path, chunksize: int = 500_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    basic = defaultdict(lambda: defaultdict(float))
    raw_sums = defaultdict(lambda: defaultdict(float))
    raw_maxes = defaultdict(lambda: defaultdict(float))
    seq_records = defaultdict(blank_sequence_record)
    all_users = set()
    total_rows = 0
    invalid_user_rows = 0

    for chunk in iter_call_chunks(call_path, chunksize=chunksize):
        chunk["user_id"] = normalize_text(chunk["user_id"])
        valid = chunk["user_id"].ne("") & chunk["user_id"].str.lower().ne("nan") & chunk["user_id"].ne("0")
        invalid_user_rows += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue

        total_rows += len(chunk)
        all_users.update(chunk["user_id"].unique())
        duration = pd.to_numeric(chunk["call_duration"], errors="coerce").fillna(0).clip(lower=0)
        status = normalize_upper(chunk["call_status"])
        call_type = normalize_upper(chunk["call_type"])
        flow = normalize_lower(chunk["flow"])
        language = normalize_lower(chunk["language"])
        hangup = pd.to_numeric(chunk["hangup_cause_code"], errors="coerce").fillna(-1)
        did = normalize_text(chunk["did"]).replace("", "unknown")
        context = pd.Series("unknown", index=chunk.index)
        user_index = chunk["user_id"]

        add_sum(basic, "total_calls", pd.Series(1, index=user_index))
        add_sum(basic, "outbound_calls", call_type.eq("OUTBOUND").astype(int).set_axis(user_index))
        add_sum(basic, "answered_calls", status.eq("ANSWERED").astype(int).set_axis(user_index))
        add_sum(basic, "total_call_duration", duration.set_axis(user_index))

        add_sum(raw_sums, "raw_total_calls", pd.Series(1, index=user_index))
        add_sum(raw_sums, "raw_total_duration", duration.set_axis(user_index))
        add_sum(raw_sums, "raw_duration_sq_sum", (duration**2).set_axis(user_index))
        add_max(raw_maxes, "raw_max_duration", duration.set_axis(user_index))
        add_sum(raw_sums, "raw_zero_duration_calls", duration.eq(0).astype(int).set_axis(user_index))
        add_sum(raw_sums, "raw_positive_duration_calls", duration.gt(0).astype(int).set_axis(user_index))

        for value in STATUS_VALUES:
            add_sum(raw_sums, f"raw_status_{value.lower()}_calls", status.eq(value).astype(int).set_axis(user_index))
        for value in TYPE_VALUES:
            add_sum(raw_sums, f"raw_type_{value.lower()}_calls", call_type.eq(value).astype(int).set_axis(user_index))
        for value in FLOW_VALUES:
            add_sum(raw_sums, f"raw_flow_{value}_calls", flow.eq(value).astype(int).set_axis(user_index))
        for value in LANGUAGE_VALUES:
            add_sum(raw_sums, f"raw_language_{value}_calls", language.eq(value).astype(int).set_axis(user_index))

        add_sum(raw_sums, "raw_hangup_known_calls", hangup.ge(0).astype(int).set_axis(user_index))
        add_sum(raw_sums, "raw_hangup_31_calls", hangup.eq(31).astype(int).set_axis(user_index))
        add_sum(raw_sums, "raw_hangup_111_calls", hangup.eq(111).astype(int).set_axis(user_index))
        add_sum(raw_sums, "raw_did_known_calls", did.ne("unknown").astype(int).set_axis(user_index))
        for token in CONTEXT_TOKENS:
            add_sum(
                raw_sums,
                f"raw_context_{token.lower()}_calls",
                context.str.contains(token, regex=False).astype(int).set_axis(user_index),
            )

        update_sequence_features(seq_records, chunk, duration, status)

    call_rows = []
    raw_rows = []
    seq_rows = []
    for user_id in sorted(all_users):
        total_calls = basic["total_calls"][user_id]
        total_duration = basic["total_call_duration"][user_id]
        answered_calls = basic["answered_calls"][user_id]
        call_rows.append(
            {
                "user_id": user_id,
                "total_calls": total_calls,
                "outbound_calls": basic["outbound_calls"][user_id],
                "answered_calls": answered_calls,
                "total_call_duration": total_duration,
                "avg_call_duration": total_duration / total_calls if total_calls else 0.0,
                "answered_rate": answered_calls / total_calls if total_calls else 0.0,
            }
        )

        raw_row = {"user_id": user_id}
        for key, values in raw_sums.items():
            raw_row[key] = values[user_id]
        for key, values in raw_maxes.items():
            raw_row[key] = values[user_id]
        total = raw_row.get("raw_total_calls", 0.0)
        duration_sum = raw_row.get("raw_total_duration", 0.0)
        duration_sq_sum = raw_row.get("raw_duration_sq_sum", 0.0)
        raw_row["raw_avg_duration"] = duration_sum / total if total else 0.0
        variance = (duration_sq_sum / total) - (raw_row["raw_avg_duration"] ** 2) if total else 0.0
        raw_row["raw_std_duration"] = float(np.sqrt(max(variance, 0.0)))
        for key in list(raw_row.keys()):
            if key.endswith("_calls") and key != "raw_total_calls":
                raw_row[key.replace("_calls", "_share")] = raw_row[key] / total if total else 0.0
        raw_rows.append(raw_row)

        rec = seq_records[user_id]
        first_ts, last_ts = rec["first_ts"], rec["last_ts"]
        first_ans, last_ans = rec["first_answer_ts"], rec["last_answer_ts"]
        seq_total = rec["total_calls"]
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
        mean_hour = rec["hour_sum"] / seq_total if seq_total else 0.0
        hour_variance = (rec["hour_sq_sum"] / seq_total) - mean_hour**2 if seq_total else 0.0
        seq_rows.append(
            {
                "user_id": user_id,
                "seq_first_call_hour": float(first_ts.hour) if pd.notna(first_ts) else 0.0,
                "seq_last_call_hour": float(last_ts.hour) if pd.notna(last_ts) else 0.0,
                "seq_first_call_dayofweek": float(first_ts.dayofweek) if pd.notna(first_ts) else 0.0,
                "seq_last_call_dayofweek": float(last_ts.dayofweek) if pd.notna(last_ts) else 0.0,
                "seq_call_span_hours": span_hours,
                "seq_avg_gap_hours": span_hours / (seq_total - 1.0) if seq_total > 1 else 0.0,
                "seq_first_call_duration": rec["first_duration"],
                "seq_last_call_duration": rec["last_duration"],
                "seq_first_answer_duration": rec["first_answer_duration"],
                "seq_last_answer_duration": rec["last_answer_duration"],
                "seq_hours_to_first_answer": hours_to_first_answer,
                "seq_answered_span_hours": answered_span_hours,
                "seq_business_hour_share": rec["business_hour_calls"] / seq_total if seq_total else 0.0,
                "seq_weekend_call_share": rec["weekend_calls"] / seq_total if seq_total else 0.0,
                "seq_mean_call_hour": mean_hour,
                "seq_std_call_hour": float(np.sqrt(max(hour_variance, 0.0))),
                "seq_first_call_answered": rec["first_answered"],
                "seq_last_call_answered": rec["last_answered"],
            }
        )

    info = {
        "raw_call_rows_processed": int(total_rows),
        "invalid_call_user_rows_skipped": int(invalid_user_rows),
        "call_cache_users": int(len(all_users)),
    }
    return pd.DataFrame(call_rows), pd.DataFrame(raw_rows), pd.DataFrame(seq_rows), info


def merge_call_features(users: pd.DataFrame, call_path: Path | None, chunksize: int = 500_000) -> tuple[pd.DataFrame, dict]:
    out = users.copy()
    out["user_id"] = normalize_text(out["user_id"])
    call_info = {
        "raw_call_rows_processed": 0,
        "invalid_call_user_rows_skipped": 0,
        "call_cache_users": 0,
        "users_with_any_joined_call": 0,
        "users_with_answered_call": 0,
        "users_with_avg_duration_10s": 0,
        "total_joined_calls": 0,
        "total_joined_answered_calls": 0,
        "join_rate": 0.0,
    }

    if call_path is not None:
        call_agg, raw, seq, info = aggregate_call_features(call_path, chunksize=chunksize)
        call_info.update(info)
        for frame in [call_agg, raw, seq]:
            if not frame.empty:
                frame["user_id"] = normalize_text(frame["user_id"])
        if not call_agg.empty:
            out = out.merge(call_agg, on="user_id", how="left")
        if not raw.empty:
            out = out.merge(raw, on="user_id", how="left")
        if not seq.empty:
            out = out.merge(seq, on="user_id", how="left")

    for col in ["total_calls", "outbound_calls", "answered_calls", "total_call_duration", "avg_call_duration", "answered_rate"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in RAW_CALL_FEATURES + SEQUENCE_FEATURES:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    call_info.update(
        {
            "users_with_any_joined_call": int((out["total_calls"] > 0).sum()),
            "users_with_answered_call": int((out["answered_calls"] > 0).sum()),
            "users_with_avg_duration_10s": int((out["avg_call_duration"] >= 10).sum()),
            "total_joined_calls": int(out["total_calls"].sum()),
            "total_joined_answered_calls": int(out["answered_calls"].sum()),
            "join_rate": float((out["total_calls"] > 0).mean()) if len(out) else 0.0,
        }
    )
    return out, call_info


def add_t1_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in set(T1_BASE_NUMERIC_FEATURES + RAW_CALL_FEATURES + SEQUENCE_FEATURES):
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["total_calls_log1p"] = np.log1p(out["total_calls"].clip(lower=0))
    out["answered_calls_log1p"] = np.log1p(out["answered_calls"].clip(lower=0))
    out["avg_call_duration_log1p"] = np.log1p(out["avg_call_duration"].clip(lower=0))
    out["raw_total_calls_log1p"] = np.log1p(out["raw_total_calls"].clip(lower=0))
    out["raw_total_duration_log1p"] = np.log1p(out["raw_total_duration"].clip(lower=0))
    out["unanswered_calls"] = (out["total_calls"] - out["answered_calls"]).clip(lower=0)
    out["unanswered_rate"] = np.where(out["total_calls"] > 0, out["unanswered_calls"] / out["total_calls"], 0)
    out["duration_per_answered_call"] = np.where(
        out["answered_calls"] > 0,
        out["avg_call_duration"] * out["total_calls"] / out["answered_calls"],
        0,
    )
    out["raw_duration_per_answered_call"] = np.where(
        out["raw_status_answered_calls"] > 0,
        out["raw_total_duration"] / out["raw_status_answered_calls"],
        0,
    )
    out["has_any_call"] = (out["total_calls"] > 0).astype(int)
    out["has_answered_call"] = (out["answered_calls"] > 0).astype(int)
    out["has_10s_avg_call"] = (out["avg_call_duration"] >= 10).astype(int)
    out["has_30s_avg_call"] = (out["avg_call_duration"] >= 30).astype(int)
    out["has_60s_avg_call"] = (out["avg_call_duration"] >= 60).astype(int)
    out["raw_answered_x_duration"] = out["raw_status_answered_share"] * out["raw_avg_duration"]
    out["raw_outbound_x_answered_share"] = out["raw_type_outbound_share"] * out["raw_status_answered_share"]
    return out


def priority_band(score: float) -> str:
    if score >= 0.90:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def score_summary(scores: pd.Series) -> dict:
    s = pd.Series(scores).astype(float)
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {
        "min": float(s.min()) if len(s) else 0.0,
        "max": float(s.max()) if len(s) else 0.0,
        "mean": float(s.mean()) if len(s) else 0.0,
        "std": float(s.std(ddof=0)) if len(s) else 0.0,
        "percentiles": {f"p{int(p * 100)}": float(s.quantile(p)) for p in percentiles} if len(s) else {},
        "count_ge_0_50": int((s >= 0.50).sum()),
        "count_ge_0_75": int((s >= 0.75).sum()),
        "count_ge_0_90": int((s >= 0.90).sum()),
    }


def decile_table(df: pd.DataFrame, score_col: str, label_col: str | None = None) -> list[dict]:
    cols = ["uid", score_col, "total_calls", "answered_calls", "avg_call_duration"]
    if label_col:
        cols.append(label_col)
    scored = df[cols].copy()
    scored["model_decile"] = pd.qcut(scored[score_col].rank(method="first"), 10, labels=False)
    agg = {
        "users": ("uid", "size"),
        "min_score": (score_col, "min"),
        "max_score": (score_col, "max"),
        "mean_score": (score_col, "mean"),
        "users_with_calls": ("total_calls", lambda s: int((s > 0).sum())),
        "users_with_answered": ("answered_calls", lambda s: int((s > 0).sum())),
        "avg_total_calls": ("total_calls", "mean"),
        "avg_answered_calls": ("answered_calls", "mean"),
        "avg_call_duration": ("avg_call_duration", "mean"),
    }
    if label_col:
        agg["positives"] = (label_col, "sum")
        agg["positive_rate"] = (label_col, "mean")
    table = scored.groupby("model_decile").agg(**agg).sort_index(ascending=False).reset_index()
    return json.loads(table.to_json(orient="records"))


def build_top_rows(df: pd.DataFrame, score_col: str, n: int) -> list[dict]:
    keep = [col for col in PREDICTION_COLUMNS if col in df.columns]
    top = df.sort_values(score_col, ascending=False).head(n)[keep].copy()
    return json.loads(top.to_json(orient="records"))


def decision_curve(df: pd.DataFrame, score_col: str, label_col: str | None = None) -> list[dict]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    cutoffs = [20, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    cutoffs = [k for k in cutoffs if k <= len(ranked)]
    if len(ranked) and (not cutoffs or cutoffs[-1] != len(ranked)):
        cutoffs.append(len(ranked))

    rows = []
    for k in cutoffs:
        top = ranked.head(k)
        row = {
            "k": int(k),
            "coverage": float(k / len(ranked)) if len(ranked) else 0.0,
            "expected_conversions": float(top[score_col].sum()),
            "avg_score": float(top[score_col].mean()) if len(top) else 0.0,
            "min_score": float(top[score_col].min()) if len(top) else 0.0,
            "users_with_calls": int((top["total_calls"] > 0).sum()) if "total_calls" in top.columns else 0,
            "users_with_answered": int((top["answered_calls"] > 0).sum()) if "answered_calls" in top.columns else 0,
        }
        if label_col and label_col in top.columns:
            labels = pd.to_numeric(top[label_col], errors="coerce").fillna(0).astype(int)
            row["actual_conversions"] = int(labels.sum())
            row["actual_precision"] = float(labels.mean()) if len(labels) else 0.0
        rows.append(row)
    return rows


def score_histogram(scores: pd.Series, bins: int = 10) -> list[dict]:
    s = pd.Series(scores).astype(float)
    if s.empty:
        return []
    counts, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    rows = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "label": f"{edges[idx]:.1f}-{edges[idx + 1]:.1f}",
                "value": int(count),
                "min_score": float(edges[idx]),
                "max_score": float(edges[idx + 1]),
            }
        )
    return rows


def top_groups(df: pd.DataFrame, group_col: str, score_col: str, limit: int = 10) -> list[dict]:
    if group_col not in df.columns:
        return []
    grouped = (
        df.assign(_group=df[group_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown"))
        .groupby("_group", dropna=False)
        .agg(
            users=("uid", "size"),
            mean_t1_score=(score_col, "mean"),
            users_with_calls=("total_calls", lambda s: int((s > 0).sum())),
            avg_calls=("total_calls", "mean"),
        )
        .sort_values(["mean_t1_score", "users"], ascending=[False, False])
        .head(limit)
        .reset_index()
        .rename(columns={"_group": "label"})
    )
    return json.loads(grouped.to_json(orient="records"))


def optional_label_join(df: pd.DataFrame, label_path: Path | None) -> tuple[pd.DataFrame, dict]:
    if label_path is None:
        return df, {"labels_available": False}
    labels = read_table(label_path)
    labels.columns = make_unique_column_names(labels.columns)
    uid_col = find_column(labels, ["uid", "csd_id", "CSD ID", "lead_id", "Lead Id"])
    if uid_col is None:
        return df, {"labels_available": False, "label_warning": "Label file had no UID column."}
    label_col = find_column(labels, ["converted_full", "converted", "actual", "label", "y", "outcome"])
    if label_col is None:
        return df, {"labels_available": False, "label_warning": "Label file had no converted/label column."}

    labels = labels.rename(columns={uid_col: "uid", label_col: "actual_converted"})
    labels["uid"] = normalize_uid(labels["uid"])
    labels["actual_converted"] = pd.to_numeric(labels["actual_converted"], errors="coerce").fillna(0).astype(int)
    labels = labels[["uid", "actual_converted"]].drop_duplicates("uid", keep="first")
    out = df.merge(labels, on="uid", how="left")
    out["actual_converted"] = pd.to_numeric(out["actual_converted"], errors="coerce")
    matched = out["actual_converted"].notna()
    out.loc[matched, "actual_converted"] = out.loc[matched, "actual_converted"].astype(int)
    return out, {
        "labels_available": bool(matched.any()),
        "label_rows": int(len(labels)),
        "matched_label_users": int(matched.sum()),
        "actual_conversions": int(out.loc[matched, "actual_converted"].sum()) if matched.any() else 0,
    }


def metrics_if_labeled(df: pd.DataFrame, score_col: str) -> dict | None:
    if "actual_converted" not in df.columns:
        return None
    valid = df["actual_converted"].notna()
    if not valid.any():
        return None
    y = df.loc[valid, "actual_converted"].astype(int)
    p = df.loc[valid, score_col].astype(float)
    ranked = pd.DataFrame({"y": y.to_numpy(), "p": p.to_numpy()}).sort_values("p", ascending=False)
    baseline = float(y.mean()) if len(y) else 0.0
    out = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "baseline_rate": baseline,
    }
    if y.nunique() == 2:
        out["auc"] = float(roc_auc_score(y, p))
    for k in [20, 50, 100, 500, 1000]:
        take = min(k, len(ranked))
        out[f"precision_at_{k}"] = float(ranked.head(take)["y"].mean()) if take else 0.0
        out[f"positives_at_{k}"] = int(ranked.head(take)["y"].sum()) if take else 0
    if len(ranked) >= 10 and baseline:
        scored = pd.DataFrame({"y": y.to_numpy(), "p": p.to_numpy()})
        scored["decile"] = pd.qcut(scored["p"].rank(method="first"), 10, labels=False)
        top_rate = float(scored.groupby("decile")["y"].mean().sort_index(ascending=False).iloc[0])
        out["top_decile_rate"] = top_rate
        out["top_decile_lift"] = float(top_rate / baseline)
    return out


def load_models() -> tuple[object, object, list[str], list[str]]:
    t0_model = joblib.load(MODEL_DIR / "loan_model_t0_call_targeting_mixed_hybrid.pkl")
    t1_model = joblib.load(MODEL_DIR / "loan_model_t1_sequence_mixed_hybrid.pkl")
    t0_features = joblib.load(MODEL_DIR / "feature_columns_t0_call_targeting_mixed_hybrid.pkl")
    t1_features = joblib.load(MODEL_DIR / "feature_columns_t1_sequence_mixed_hybrid.pkl")
    return t0_model, t1_model, t0_features, t1_features


def score_raw_files(
    user_path: Path,
    call_path: Path | None,
    label_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    chunksize: int = 500_000,
    top_n: int = 500,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]

    users, user_info = prepare_user_features_from_frame(read_table(user_path))
    scored, call_info = merge_call_features(users, call_path, chunksize=chunksize)
    scored = add_t1_enhancements(scored)

    t0_model, t1_model, t0_features, t1_features = load_models()
    for col in set(t0_features + t1_features):
        if col not in scored.columns:
            scored[col] = 0
    scored = scored.copy()

    scored["t0_call_targeting_score"] = t0_model.predict_proba(scored[t0_features])[:, 1]
    scored["t1_loan_conversion_score"] = t1_model.predict_proba(scored[t1_features])[:, 1]
    scored["t0_call_priority_rank"] = scored["t0_call_targeting_score"].rank(method="first", ascending=False).astype(int)
    scored["t1_conversion_rank"] = scored["t1_loan_conversion_score"].rank(method="first", ascending=False).astype(int)
    scored["priority_band"] = scored["t1_loan_conversion_score"].map(priority_band)
    scored["recommended_stage"] = np.where(
        scored["total_calls"].fillna(0).gt(0),
        "T1: post-call conversion follow-up",
        "T0: call priority queue",
    )

    scored, label_info = optional_label_join(scored, label_path)

    export_cols = [col for col in PREDICTION_COLUMNS + ["actual_converted"] if col in scored.columns]
    export = scored[export_cols].sort_values("t1_loan_conversion_score", ascending=False)
    full_path = output_dir / f"{job_id}_raw_scored_predictions.csv"
    t0_path = output_dir / f"{job_id}_top_t0_call_targets.csv"
    t1_path = output_dir / f"{job_id}_top_t1_conversion_predictions.csv"
    export.to_csv(full_path, index=False)
    scored.sort_values("t0_call_targeting_score", ascending=False)[export_cols].head(top_n).to_csv(t0_path, index=False)
    scored.sort_values("t1_loan_conversion_score", ascending=False)[export_cols].head(top_n).to_csv(t1_path, index=False)

    t0_similarity = t0_model.explain_month_similarity(scored[t0_features]) if hasattr(t0_model, "explain_month_similarity") else {}
    t1_similarity = t1_model.explain_month_similarity(scored[t1_features]) if hasattr(t1_model, "explain_month_similarity") else {}

    summary = {
        "job_id": job_id,
        "users_scored": int(len(scored)),
        "user_info": user_info,
        "call_join_info": call_info,
        "label_info": label_info,
        "score_summary": {
            "t0": score_summary(scored["t0_call_targeting_score"]),
            "t1": score_summary(scored["t1_loan_conversion_score"]),
        },
        "month_similarity": {"t0": t0_similarity, "t1": t1_similarity},
        "deciles": {
            "t0": decile_table(scored, "t0_call_targeting_score", "actual_converted" if label_info.get("labels_available") else None),
            "t1": decile_table(scored, "t1_loan_conversion_score", "actual_converted" if label_info.get("labels_available") else None),
        },
        "metrics": {
            "t0": metrics_if_labeled(scored, "t0_call_targeting_score"),
            "t1": metrics_if_labeled(scored, "t1_loan_conversion_score"),
        },
        "decision_curve": {
            "t0": decision_curve(
                scored,
                "t0_call_targeting_score",
                "actual_converted" if label_info.get("labels_available") else None,
            ),
            "t1": decision_curve(
                scored,
                "t1_loan_conversion_score",
                "actual_converted" if label_info.get("labels_available") else None,
            ),
        },
        "score_histogram": {
            "t0": score_histogram(scored["t0_call_targeting_score"]),
            "t1": score_histogram(scored["t1_loan_conversion_score"]),
        },
        "priority_bands": scored["priority_band"].value_counts().reindex(["Very High", "High", "Medium", "Low"], fill_value=0).to_dict(),
        "top_groups": {
            "campaign_id": top_groups(scored, "campaign_id", "t1_loan_conversion_score"),
            "state": top_groups(scored, "state", "t1_loan_conversion_score"),
        },
        "top_t0": build_top_rows(scored, "t0_call_targeting_score", top_n),
        "top_t1": build_top_rows(scored, "t1_loan_conversion_score", top_n),
        "outputs": {
            "full": str(full_path),
            "top_t0": str(t0_path),
            "top_t1": str(t1_path),
        },
        "model_files": {
            "t0": str(MODEL_DIR / "loan_model_t0_call_targeting_mixed_hybrid.pkl"),
            "t1": str(MODEL_DIR / "loan_model_t1_sequence_mixed_hybrid.pkl"),
        },
    }
    (output_dir / f"{job_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name)
    return cleaned or "upload.csv"
