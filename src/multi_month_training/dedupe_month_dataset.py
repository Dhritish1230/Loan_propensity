import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "uid",
    "user_id",
    "converted",
    "language",
    "state",
    "flow_phase",
    "age",
    "decile",
    "total_calls",
    "answered_calls",
    "answered_rate",
    "avg_call_duration",
]


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        input_path,
        low_memory=False,
        usecols=lambda col: col in REQUIRED_COLUMNS,
    )
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["uid"] = df["uid"].astype(str).str.strip().str.upper()
    df["user_id"] = df["user_id"].fillna("unknown").astype(str).str.strip()
    df = df.loc[df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    df["language"] = df["language"].fillna("unknown").astype(str).str.strip().str.lower()
    df["state"] = df["state"].fillna("unknown").astype(str).str.strip().str.upper()
    df["flow_phase"] = df["flow_phase"].fillna("unknown").astype(str).str.strip()

    for col in ["age", "decile", "total_calls", "answered_calls", "answered_rate", "avg_call_duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["converted"] = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)

    out = (
        df.groupby("uid", as_index=False)
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
    out["answered_rate"] = np.where(
        out["total_calls"] > 0,
        out["answered_calls"] / out["total_calls"],
        0,
    )
    out["month"] = args.month

    out.to_csv(output_path, index=False)
    print(
        f"{args.month}: rows={len(out):,}, conversions={int(out['converted'].sum()):,}, "
        f"users_with_call_signal={int((out['total_calls'] > 0).sum()):,}, output={output_path}"
    )


if __name__ == "__main__":
    main()
