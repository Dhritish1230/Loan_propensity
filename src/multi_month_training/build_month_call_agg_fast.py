import argparse
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
CALL_PATHS = {
    "JAN": ROOT / "jan" / "hero_fincorp_loan_upselling_call_data_january_2026_updated.csv",
    "FEB": ROOT / "feb" / "hero_fincorp_loan_upselling_call_data_feb_2026_updated.csv",
    "MAR": ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2026_march_updated.csv",
}
OUT_DIR = ROOT / "multi_month_training" / "call_agg_cache"


def build_call_agg(month: str, chunksize: int) -> pd.DataFrame:
    chunks = []
    usecols = ["call_id", "user_id", "call_duration", "call_type", "call_status"]
    for chunk in pd.read_csv(
        CALL_PATHS[month],
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        index_col=False,
    ):
        user_id = chunk["user_id"].fillna("").astype(str).str.strip()
        valid = user_id.ne("") & user_id.ne("nan") & user_id.ne("0")
        chunk = chunk.loc[valid].copy()
        chunk["user_id"] = user_id.loc[valid]
        chunk["call_duration"] = pd.to_numeric(chunk["call_duration"], errors="coerce").fillna(0).clip(lower=0)
        chunk["is_answered"] = chunk["call_status"].fillna("").astype(str).str.lower().str.strip().eq("answered").astype(int)
        chunk["is_outbound"] = chunk["call_type"].fillna("").astype(str).str.lower().str.strip().eq("outbound").astype(int)

        part = (
            chunk.groupby("user_id", as_index=False)
            .agg(
                total_calls=("call_id", "count"),
                outbound_calls=("is_outbound", "sum"),
                answered_calls=("is_answered", "sum"),
                total_call_duration=("call_duration", "sum"),
            )
        )
        chunks.append(part)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "user_id",
                "total_calls",
                "outbound_calls",
                "answered_calls",
                "total_call_duration",
                "avg_call_duration",
                "answered_rate",
            ]
        )

    out = (
        pd.concat(chunks, ignore_index=True)
        .groupby("user_id", as_index=False)
        .agg(
            total_calls=("total_calls", "sum"),
            outbound_calls=("outbound_calls", "sum"),
            answered_calls=("answered_calls", "sum"),
            total_call_duration=("total_call_duration", "sum"),
        )
    )
    out["avg_call_duration"] = out["total_call_duration"] / out["total_calls"].where(out["total_calls"] > 0, 1)
    out["answered_rate"] = out["answered_calls"] / out["total_calls"].where(out["total_calls"] > 0, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=sorted(CALL_PATHS), required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    out = build_call_agg(args.month, args.chunksize)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.month}_call_agg.csv"
    out.to_csv(out_path, index=False)
    print(
        f"{args.month}: call_agg_rows={len(out):,}, total_calls={int(out['total_calls'].sum()):,}, output={out_path}"
    )


if __name__ == "__main__":
    main()
