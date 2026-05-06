import argparse
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
MT_DIR = ROOT / "multi_month_training"
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()
OUT_DIR = MT_DIR / "test_outputs"

APR_USER_PATH = ROOT / "april" / "hero_fincorp_loan_upselling_user_data_2026_april_updated.csv"
APR_CALL_PATH = ROOT / "april" / "hero_fincorp_loan_upselling_call_data_2026_april_updated.csv"

CALL_AGG_PATH = MT_DIR / "call_agg_cache" / "APR_call_agg.csv"
RAW_CALL_PATH = MT_DIR / "raw_call_cache" / "APR_raw_call_features.csv"
SEQ_CALL_PATH = MT_DIR / "sequence_call_cache" / "APR_sequence_call_features.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for path_entry in [SRC_DIR, CODE_DIR]:
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from evaluate_model_slice import RAW_CALL_FEATURES  # noqa: E402
from aggregate_call_sequence_features import SEQUENCE_FEATURES  # noqa: E402
from run_stage_model_experiments import add_t1_enhancements  # noqa: E402
from train_t0_call_targeting_model import prepare_user_features  # noqa: E402


FIRST_CALL_COLUMNS = [
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

STATUS_VALUES = ["ANSWERED", "NOT_ANSWERED"]
TYPE_VALUES = ["INBOUND", "OUTBOUND"]
FLOW_VALUES = ["personal", "business", "unknown"]
LANGUAGE_VALUES = ["hin", "ka", "te", "ta", "bn", "unknown"]
CONTEXT_TOKENS = ["DND", "PERSONAL", "BUSINESS"]

OUTPUT_COLUMNS = [
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
]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_uid(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.upper()


def normalize_upper(series: pd.Series) -> pd.Series:
    return normalize_text(series).replace("", "unknown").str.upper()


def normalize_lower(series: pd.Series) -> pd.Series:
    return normalize_text(series).replace("", "unknown").str.lower()


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
    if pd.notna(first_answer_ts) and (pd.isna(rec["first_answer_ts"]) or first_answer_ts < rec["first_answer_ts"]):
        rec["first_answer_ts"] = first_answer_ts
        rec["first_answer_duration"] = row["first_answer_duration"]

    last_answer_ts = row["last_answer_ts"]
    if pd.notna(last_answer_ts) and (pd.isna(rec["last_answer_ts"]) or last_answer_ts > rec["last_answer_ts"]):
        rec["last_answer_ts"] = last_answer_ts
        rec["last_answer_duration"] = row["last_answer_duration"]


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


def build_call_feature_caches(chunksize: int = 500_000, force: bool = False) -> None:
    if not force and CALL_AGG_PATH.exists() and RAW_CALL_PATH.exists() and SEQ_CALL_PATH.exists():
        print("April call caches already exist; reusing them.")
        return

    for path in [CALL_AGG_PATH, RAW_CALL_PATH, SEQ_CALL_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    basic = defaultdict(lambda: defaultdict(float))
    raw_sums = defaultdict(lambda: defaultdict(float))
    raw_maxes = defaultdict(lambda: defaultdict(float))
    seq_records = defaultdict(blank_sequence_record)
    all_users = set()
    total_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            APR_CALL_PATH,
            low_memory=False,
            chunksize=chunksize,
            usecols=list(range(len(FIRST_CALL_COLUMNS))),
            index_col=False,
        ),
        start=1,
    ):
        chunk.columns = FIRST_CALL_COLUMNS
        chunk["user_id"] = normalize_text(chunk["user_id"])
        valid = chunk["user_id"].ne("") & chunk["user_id"].str.lower().ne("nan") & chunk["user_id"].ne("0")
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
        indexed = pd.DataFrame(index=user_index)

        add_sum(basic, "total_calls", pd.Series(1, index=indexed.index))
        add_sum(basic, "outbound_calls", call_type.eq("OUTBOUND").astype(int).set_axis(indexed.index))
        add_sum(basic, "answered_calls", status.eq("ANSWERED").astype(int).set_axis(indexed.index))
        add_sum(basic, "total_call_duration", duration.set_axis(indexed.index))

        add_sum(raw_sums, "raw_total_calls", pd.Series(1, index=indexed.index))
        add_sum(raw_sums, "raw_total_duration", duration.set_axis(indexed.index))
        add_sum(raw_sums, "raw_duration_sq_sum", (duration**2).set_axis(indexed.index))
        add_max(raw_maxes, "raw_max_duration", duration.set_axis(indexed.index))
        add_sum(raw_sums, "raw_zero_duration_calls", duration.eq(0).astype(int).set_axis(indexed.index))
        add_sum(raw_sums, "raw_positive_duration_calls", duration.gt(0).astype(int).set_axis(indexed.index))

        for value in STATUS_VALUES:
            add_sum(raw_sums, f"raw_status_{value.lower()}_calls", status.eq(value).astype(int).set_axis(indexed.index))
        for value in TYPE_VALUES:
            add_sum(raw_sums, f"raw_type_{value.lower()}_calls", call_type.eq(value).astype(int).set_axis(indexed.index))
        for value in FLOW_VALUES:
            add_sum(raw_sums, f"raw_flow_{value}_calls", flow.eq(value).astype(int).set_axis(indexed.index))
        for value in LANGUAGE_VALUES:
            add_sum(raw_sums, f"raw_language_{value}_calls", language.eq(value).astype(int).set_axis(indexed.index))

        add_sum(raw_sums, "raw_hangup_known_calls", hangup.ge(0).astype(int).set_axis(indexed.index))
        add_sum(raw_sums, "raw_hangup_31_calls", hangup.eq(31).astype(int).set_axis(indexed.index))
        add_sum(raw_sums, "raw_hangup_111_calls", hangup.eq(111).astype(int).set_axis(indexed.index))
        add_sum(raw_sums, "raw_did_known_calls", did.ne("unknown").astype(int).set_axis(indexed.index))

        for token in CONTEXT_TOKENS:
            add_sum(raw_sums, f"raw_context_{token.lower()}_calls", context.str.contains(token, regex=False).astype(int).set_axis(indexed.index))

        update_sequence_features(seq_records, chunk, duration, status)

        if chunk_idx % 10 == 0:
            print(f"Processed {total_rows:,} April call rows across {len(all_users):,} users...", flush=True)

    call_rows = []
    raw_rows = []
    seq_rows = []
    for user_id in sorted(all_users):
        total_calls = basic["total_calls"][user_id]
        total_call_duration = basic["total_call_duration"][user_id]
        call_rows.append(
            {
                "user_id": user_id,
                "total_calls": total_calls,
                "outbound_calls": basic["outbound_calls"][user_id],
                "answered_calls": basic["answered_calls"][user_id],
                "total_call_duration": total_call_duration,
                "avg_call_duration": total_call_duration / total_calls if total_calls else 0.0,
                "answered_rate": basic["answered_calls"][user_id] / total_calls if total_calls else 0.0,
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
        first_ts = rec["first_ts"]
        last_ts = rec["last_ts"]
        first_ans = rec["first_answer_ts"]
        last_ans = rec["last_answer_ts"]
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

    pd.DataFrame(call_rows).to_csv(CALL_AGG_PATH, index=False)
    pd.DataFrame(raw_rows).to_csv(RAW_CALL_PATH, index=False)
    pd.DataFrame(seq_rows).to_csv(SEQ_CALL_PATH, index=False)
    print(
        f"Saved April call caches: users={len(all_users):,}, call_rows={total_rows:,}, "
        f"call_agg={CALL_AGG_PATH}, raw={RAW_CALL_PATH}, sequence={SEQ_CALL_PATH}",
        flush=True,
    )


def prepare_april_user_frame() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(APR_USER_PATH, low_memory=False)
    raw.columns = raw.columns.str.strip()
    raw["uid"] = normalize_uid(raw["uid"])
    raw = raw.loc[raw["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    raw["user_id"] = normalize_text(raw["user_id"])
    first_meta_cols = [
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
        if col in raw.columns
    ]
    meta = raw[first_meta_cols].drop_duplicates("uid", keep="first")
    features = prepare_user_features(APR_USER_PATH)
    df = features.merge(meta.drop(columns=[c for c in features.columns if c in meta.columns and c != "uid"]), on="uid", how="left")

    info = {
        "raw_user_rows": int(len(raw)),
        "unique_uid": int(raw["uid"].nunique()),
        "duplicate_uid_rows": int(raw["uid"].duplicated().sum()),
        "feature_rows_after_uid_dedupe": int(len(df)),
    }
    return df, info


def merge_call_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["user_id"] = normalize_text(out["user_id"])

    call_agg = pd.read_csv(CALL_AGG_PATH, low_memory=False)
    raw = pd.read_csv(RAW_CALL_PATH, low_memory=False)
    seq = pd.read_csv(SEQ_CALL_PATH, low_memory=False)
    for frame in [call_agg, raw, seq]:
        frame.columns = frame.columns.str.strip()
        frame["user_id"] = normalize_text(frame["user_id"])

    out = out.merge(call_agg, on="user_id", how="left")
    out = out.merge(raw, on="user_id", how="left")
    out = out.merge(seq, on="user_id", how="left")

    for col in ["total_calls", "outbound_calls", "answered_calls", "total_call_duration", "avg_call_duration", "answered_rate"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in RAW_CALL_FEATURES + SEQUENCE_FEATURES:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    info = {
        "call_cache_users": int(call_agg["user_id"].nunique()),
        "raw_call_cache_users": int(raw["user_id"].nunique()),
        "sequence_call_cache_users": int(seq["user_id"].nunique()),
        "users_with_any_joined_call": int((out["total_calls"] > 0).sum()),
        "users_with_answered_call": int((out["answered_calls"] > 0).sum()),
        "users_with_avg_duration_10s": int((out["avg_call_duration"] >= 10).sum()),
        "total_joined_calls": int(out["total_calls"].sum()),
        "total_joined_answered_calls": int(out["answered_calls"].sum()),
    }
    return out, info


def score_summary(scores: pd.Series) -> dict:
    s = pd.Series(scores).astype(float)
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "percentiles": {f"p{int(p * 100)}": float(s.quantile(p)) for p in percentiles},
        "count_ge_0_50": int((s >= 0.50).sum()),
        "count_ge_0_60": int((s >= 0.60).sum()),
        "count_ge_0_70": int((s >= 0.70).sum()),
        "count_ge_0_80": int((s >= 0.80).sum()),
        "count_ge_0_90": int((s >= 0.90).sum()),
    }


def unlabeled_decile_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    scored = df[["uid", score_col, "total_calls", "answered_calls", "avg_call_duration"]].copy()
    scored["decile"] = pd.qcut(scored[score_col].rank(method="first"), 10, labels=False)
    table = (
        scored.groupby("decile")
        .agg(
            users=("uid", "size"),
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            mean_score=(score_col, "mean"),
            users_with_calls=("total_calls", lambda s: int((s > 0).sum())),
            users_with_answered=("answered_calls", lambda s: int((s > 0).sum())),
            avg_total_calls=("total_calls", "mean"),
            avg_answered_calls=("answered_calls", "mean"),
            avg_call_duration=("avg_call_duration", "mean"),
        )
        .sort_index(ascending=False)
        .reset_index()
    )
    table.insert(0, "model", score_col)
    return table


def top_rows(df: pd.DataFrame, score_col: str, n: int = 20) -> list[dict]:
    cols = [col for col in OUTPUT_COLUMNS + [score_col] if col in df.columns]
    return df.sort_values(score_col, ascending=False).head(n)[cols].to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--force-rebuild-call-cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_call_feature_caches(chunksize=args.chunksize, force=args.force_rebuild_call_cache)

    users, user_info = prepare_april_user_frame()
    scored, call_info = merge_call_features(users)
    scored = add_t1_enhancements(scored)

    t0_model = joblib.load(LOAN_API_DIR / "loan_model_t0_call_targeting_mixed_hybrid.pkl")
    t1_model = joblib.load(LOAN_API_DIR / "loan_model_t1_sequence_mixed_hybrid.pkl")
    t0_features = joblib.load(LOAN_API_DIR / "feature_columns_t0_call_targeting_mixed_hybrid.pkl")
    t1_features = joblib.load(LOAN_API_DIR / "feature_columns_t1_sequence_mixed_hybrid.pkl")

    for col in t0_features + t1_features:
        if col not in scored.columns:
            scored[col] = 0

    scored["t0_call_targeting_score"] = t0_model.predict_proba(scored[t0_features])[:, 1]
    scored["t1_loan_conversion_score"] = t1_model.predict_proba(scored[t1_features])[:, 1]

    pred_cols = [col for col in OUTPUT_COLUMNS if col in scored.columns]
    pred_cols += ["t0_call_targeting_score", "t1_loan_conversion_score"]

    t0_path = OUT_DIR / "april_unlabeled_t0_call_targets.csv"
    t1_path = OUT_DIR / "april_unlabeled_t1_post_call_predictions.csv"
    top100_t0_path = OUT_DIR / "april_unlabeled_t0_top100.csv"
    top100_t1_path = OUT_DIR / "april_unlabeled_t1_top100.csv"
    decile_path = OUT_DIR / "april_unlabeled_score_deciles.csv"
    report_path = OUT_DIR / "april_unlabeled_score_report.json"

    scored.sort_values("t0_call_targeting_score", ascending=False)[pred_cols].to_csv(t0_path, index=False)
    scored.sort_values("t1_loan_conversion_score", ascending=False)[pred_cols].to_csv(t1_path, index=False)
    scored.sort_values("t0_call_targeting_score", ascending=False)[pred_cols].head(100).to_csv(top100_t0_path, index=False)
    scored.sort_values("t1_loan_conversion_score", ascending=False)[pred_cols].head(100).to_csv(top100_t1_path, index=False)

    deciles = pd.concat(
        [
            unlabeled_decile_table(scored, "t0_call_targeting_score"),
            unlabeled_decile_table(scored, "t1_loan_conversion_score"),
        ],
        ignore_index=True,
    )
    deciles.to_csv(decile_path, index=False)

    report = {
        "mode": "unlabeled_april_scoring",
        "important_note": "DIY_APRIL.csv and SFDC_APRIL.csv were not read. Therefore AUC, precision, recall, and lift cannot be calculated yet.",
        "user_file": str(APR_USER_PATH),
        "call_file": str(APR_CALL_PATH),
        "models": {
            "t0": str(LOAN_API_DIR / "loan_model_t0_call_targeting_mixed_hybrid.pkl"),
            "t1": str(LOAN_API_DIR / "loan_model_t1_sequence_mixed_hybrid.pkl"),
        },
        "user_info": user_info,
        "call_join_info": call_info,
        "score_summary": {
            "t0_call_targeting_score": score_summary(scored["t0_call_targeting_score"]),
            "t1_loan_conversion_score": score_summary(scored["t1_loan_conversion_score"]),
        },
        "month_similarity": {
            "t0": t0_model.explain_month_similarity(scored[t0_features]),
            "t1": t1_model.explain_month_similarity(scored[t1_features]),
        },
        "outputs": {
            "t0_all_ranked": str(t0_path),
            "t1_all_ranked": str(t1_path),
            "t0_top100": str(top100_t0_path),
            "t1_top100": str(top100_t1_path),
            "deciles": str(decile_path),
            "report": str(report_path),
        },
        "top_20_t0": top_rows(scored, "t0_call_targeting_score", 20),
        "top_20_t1": top_rows(scored, "t1_loan_conversion_score", 20),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
