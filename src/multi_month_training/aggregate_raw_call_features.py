import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_VALUES = ["ANSWERED", "NOT_ANSWERED"]
TYPE_VALUES = ["INBOUND", "OUTBOUND"]
FLOW_VALUES = ["personal", "business", "unknown"]
LANGUAGE_VALUES = ["hin", "ka", "te", "ta", "bn", "unknown"]
CONTEXT_TOKENS = ["DND", "PERSONAL", "BUSINESS"]
RAW_CALL_USECOLS = [
    "call_id",
    "user_id",
    "call_duration",
    "call_type",
    "call_status",
    "hangup_cause_code",
    "flow",
    "did",
    "language",
]
RAW_CALL_FIRST_COLUMNS = [
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


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).str.strip()


def normalize_upper(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.upper()


def normalize_lower(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.lower()


def get_column(chunk: pd.DataFrame, column: str, default):
    if column in chunk.columns:
        return chunk[column]
    return pd.Series(default, index=chunk.index)


def add_sum(store: dict, key: str, values: pd.Series):
    grouped = values.groupby(values.index).sum()
    for user_id, value in grouped.items():
        store[key][user_id] += float(value)


def add_max(store: dict, key: str, values: pd.Series):
    grouped = values.groupby(values.index).max()
    for user_id, value in grouped.items():
        store[key][user_id] = max(store[key][user_id], float(value))


def aggregate(input_path: Path, output_path: Path, chunksize: int):
    sums = defaultdict(lambda: defaultdict(float))
    maxes = defaultdict(lambda: defaultdict(float))
    all_users = set()

    # Some raw call exports have an unquoted JSON context field containing commas.
    # Read the fixed leading columns by position so extra commas inside context
    # cannot shift named columns or silently corrupt the user_id join.
    for chunk in pd.read_csv(
        input_path,
        low_memory=False,
        chunksize=chunksize,
        usecols=list(range(len(RAW_CALL_FIRST_COLUMNS))),
        index_col=False,
    ):
        chunk.columns = RAW_CALL_FIRST_COLUMNS
        chunk = chunk[[col for col in RAW_CALL_USECOLS if col in chunk.columns]]
        if "user_id" not in chunk.columns:
            raise ValueError(f"user_id column missing in {input_path}")

        user_id = normalize_text(chunk["user_id"])
        valid = user_id.ne("") & user_id.ne("nan") & user_id.ne("0")
        chunk = chunk.loc[valid].copy()
        user_id = user_id.loc[valid]
        chunk.index = user_id
        all_users.update(user_id.unique())

        duration = pd.to_numeric(get_column(chunk, "call_duration", 0), errors="coerce").fillna(0).clip(lower=0)
        status = normalize_upper(get_column(chunk, "call_status", "unknown"))
        call_type = normalize_upper(get_column(chunk, "call_type", "unknown"))
        flow = normalize_lower(get_column(chunk, "flow", "unknown"))
        language = normalize_lower(get_column(chunk, "language", "unknown"))
        hangup = pd.to_numeric(get_column(chunk, "hangup_cause_code", -1), errors="coerce").fillna(-1)
        did = normalize_text(get_column(chunk, "did", "unknown"))
        context = normalize_upper(get_column(chunk, "context", "unknown"))

        add_sum(sums, "raw_total_calls", pd.Series(1, index=chunk.index))
        add_sum(sums, "raw_total_duration", duration)
        add_sum(sums, "raw_duration_sq_sum", duration ** 2)
        add_max(maxes, "raw_max_duration", duration)
        add_sum(sums, "raw_zero_duration_calls", (duration == 0).astype(int))
        add_sum(sums, "raw_positive_duration_calls", (duration > 0).astype(int))

        for value in STATUS_VALUES:
            add_sum(sums, f"raw_status_{value.lower()}_calls", (status == value).astype(int))

        for value in TYPE_VALUES:
            add_sum(sums, f"raw_type_{value.lower()}_calls", (call_type == value).astype(int))

        for value in FLOW_VALUES:
            add_sum(sums, f"raw_flow_{value}_calls", (flow == value).astype(int))

        for value in LANGUAGE_VALUES:
            add_sum(sums, f"raw_language_{value}_calls", (language == value).astype(int))

        add_sum(sums, "raw_hangup_known_calls", (hangup >= 0).astype(int))
        add_sum(sums, "raw_hangup_31_calls", (hangup == 31).astype(int))
        add_sum(sums, "raw_hangup_111_calls", (hangup == 111).astype(int))
        add_sum(sums, "raw_did_known_calls", did.ne("unknown").astype(int))

        for token in CONTEXT_TOKENS:
            add_sum(sums, f"raw_context_{token.lower()}_calls", context.str.contains(token, regex=False).astype(int))

    rows = []
    for user_id in sorted(all_users):
        row = {"user_id": user_id}
        for key, values in sums.items():
            row[key] = values[user_id]
        for key, values in maxes.items():
            row[key] = values[user_id]

        total = row.get("raw_total_calls", 0.0)
        duration_sum = row.get("raw_total_duration", 0.0)
        duration_sq_sum = row.get("raw_duration_sq_sum", 0.0)
        row["raw_avg_duration"] = duration_sum / total if total else 0.0
        variance = (duration_sq_sum / total) - (row["raw_avg_duration"] ** 2) if total else 0.0
        row["raw_std_duration"] = float(np.sqrt(max(variance, 0.0)))

        for key in list(row.keys()):
            if key.endswith("_calls") and key != "raw_total_calls":
                row[key.replace("_calls", "_share")] = row[key] / total if total else 0.0
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved raw call features: {output_path} rows={len(rows):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    aggregate(Path(args.input), Path(args.output), args.chunksize)


if __name__ == "__main__":
    main()
